//! AES-256-GCM encrypted token storage at `~/.isartor/tokens/`.
//!
//! ## Encryption model
//!
//! A 32-byte master key is derived the first time `TokenStore::open()` is
//! called, generated via `rand::random`, and written to
//! `~/.isartor/tokens/.key` with `0600` permissions.  All subsequent token
//! files are encrypted with AES-256-GCM using this key plus a fresh random
//! 96-bit nonce prepended to every ciphertext.
//!
//! Plaintext tokens are serialised as compact JSON.

use std::fs;
use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use anyhow::{Context, bail};
use rand::RngCore;

use super::StoredToken;

const NONCE_LEN: usize = 12;

/// Encrypted on-disk token store.
pub struct TokenStore {
    base_dir: PathBuf,
    cipher: Aes256Gcm,
}

impl TokenStore {
    /// Open (or create) the token store.
    ///
    /// On first call, a random master key is generated and saved to
    /// `~/.isartor/tokens/.key` with restricted permissions.
    pub fn open() -> anyhow::Result<Self> {
        Self::open_in(token_dir()?)
    }

    /// Open (or create) the token store at a specific directory.
    pub fn open_in(base_dir: PathBuf) -> anyhow::Result<Self> {
        fs::create_dir_all(&base_dir)
            .with_context(|| format!("cannot create token dir {}", base_dir.display()))?;
        set_dir_permissions(&base_dir)?;

        let key_path = base_dir.join(".key");
        let key_bytes = if key_path.exists() {
            let raw = fs::read(&key_path)
                .with_context(|| format!("cannot read token key {}", key_path.display()))?;
            if raw.len() != 32 {
                bail!(
                    "token key file is corrupt (expected 32 bytes, got {})",
                    raw.len()
                );
            }
            let mut k = [0u8; 32];
            k.copy_from_slice(&raw);
            k
        } else {
            let mut k = [0u8; 32];
            OsRng.fill_bytes(&mut k);
            fs::write(&key_path, k)
                .with_context(|| format!("cannot write token key {}", key_path.display()))?;
            set_file_permissions(&key_path)?;
            k
        };

        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);
        Ok(Self { base_dir, cipher })
    }

    /// Persist a token for `provider` (overwrites any existing entry).
    pub fn save(&self, token: &StoredToken) -> anyhow::Result<()> {
        let plaintext = serde_json::to_vec(token).context("failed to serialise token")?;

        let mut nonce_bytes = [0u8; NONCE_LEN];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext.as_ref())
            .map_err(|e| anyhow::anyhow!("token encryption failed: {e}"))?;

        // Format: [12-byte nonce][ciphertext]
        let mut blob = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ciphertext);

        let path = self.token_path(&token.provider);
        fs::write(&path, &blob)
            .with_context(|| format!("cannot write token file {}", path.display()))?;
        set_file_permissions(&path)?;
        Ok(())
    }

    /// Load a stored token for `provider`, or `None` if none exists.
    pub fn load(&self, provider: &str) -> anyhow::Result<Option<StoredToken>> {
        let path = self.token_path(provider);
        if !path.exists() {
            return Ok(None);
        }
        let blob = fs::read(&path)
            .with_context(|| format!("cannot read token file {}", path.display()))?;

        if blob.len() < NONCE_LEN {
            bail!("token file for '{}' is corrupt", provider);
        }
        let (nonce_bytes, ciphertext) = blob.split_at(NONCE_LEN);
        let nonce = Nonce::from_slice(nonce_bytes);

        let plaintext = self.cipher.decrypt(nonce, ciphertext).map_err(|_| {
            anyhow::anyhow!(
                "failed to decrypt token for '{}' — file may be corrupt or key changed",
                provider
            )
        })?;

        let token: StoredToken =
            serde_json::from_slice(&plaintext).context("stored token is invalid JSON")?;
        Ok(Some(token))
    }

    /// Delete the stored token for `provider`. No-op if none exists.
    pub fn delete(&self, provider: &str) -> anyhow::Result<()> {
        let path = self.token_path(provider);
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("cannot delete token file {}", path.display()))?;
        }
        Ok(())
    }

    /// Return the list of provider names that have a stored token.
    pub fn list_authenticated(&self) -> anyhow::Result<Vec<String>> {
        let mut providers = Vec::new();
        for entry in fs::read_dir(&self.base_dir)
            .with_context(|| format!("cannot read token dir {}", self.base_dir.display()))?
        {
            let entry = entry?;
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if s.starts_with('.') || !s.ends_with(".enc") {
                continue;
            }
            providers.push(s.trim_end_matches(".enc").to_string());
        }
        providers.sort();
        Ok(providers)
    }

    /// Load a token and auto-refresh it if it is expired and has a refresh token.
    ///
    /// Returns `None` if no token is stored. On successful refresh, the new
    /// token is persisted before being returned.
    pub async fn load_refreshed(
        &self,
        provider: &str,
        http: &reqwest::Client,
    ) -> anyhow::Result<Option<StoredToken>> {
        let Some(token) = self.load(provider)? else {
            return Ok(None);
        };
        if !token.is_expired() {
            return Ok(Some(token));
        }
        let Some(refresh_token) = &token.refresh_token else {
            return Ok(Some(token)); // expired, no refresh token — caller decides
        };

        // Look up the provider implementation for refresh.
        let Some(provider_impl) = super::find_provider(provider) else {
            return Ok(Some(token));
        };
        match provider_impl.refresh_token(http, refresh_token).await {
            Ok(new_token) => {
                self.save(&new_token)?;
                Ok(Some(new_token))
            }
            Err(e) => {
                tracing::warn!(provider = provider, error = %e, "OAuth token refresh failed");
                Ok(Some(token))
            }
        }
    }

    /// Return the access token for use as an API key, refreshing if needed.
    ///
    /// Returns `None` if no token is stored for the provider.
    pub async fn get_access_token(
        &self,
        provider: &str,
        http: &reqwest::Client,
    ) -> anyhow::Result<Option<String>> {
        Ok(self
            .load_refreshed(provider, http)
            .await?
            .map(|t| t.access_token.clone()))
    }

    fn token_path(&self, provider: &str) -> PathBuf {
        self.base_dir.join(format!("{provider}.enc"))
    }
}

// ── Path helpers ──────────────────────────────────────────────────────────────

fn token_dir() -> anyhow::Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".isartor").join("tokens"))
}

// ── Permission helpers (Unix-only) ────────────────────────────────────────────

#[cfg(unix)]
fn set_file_permissions(path: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("cannot set permissions on {}", path.display()))
}

#[cfg(not(unix))]
fn set_file_permissions(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_dir_permissions(path: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("cannot set permissions on {}", path.display()))
}

#[cfg(not(unix))]
fn set_dir_permissions(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}
