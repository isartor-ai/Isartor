//! Generalized OAuth framework for multi-provider authentication.
//!
//! Provides a shared [`OAuthProvider`] trait, encrypted token storage via
//! [`TokenStore`], and a device-flow polling helper. Implementations live in
//! the [`providers`] submodule.
//!
//! ## Supported providers
//!
//! | Provider | Flow | Notes |
//! |----------|------|-------|
//! | `copilot` | GitHub device flow | wraps existing implementation |
//! | `gemini` | Google OAuth2 device flow | `generative-language` scope |
//! | `kiro` | AWS Builder ID SSO OIDC | dynamic client registration |
//! | `anthropic` | API-key prompt | no public device flow |
//! | `openai` | API-key prompt | no public device flow |
//!
//! ## Usage
//! ```ignore
//! // Authenticate a provider interactively:
//! let store = TokenStore::open()?;
//! let provider = find_provider("gemini").unwrap();
//! crate::cli::auth::run_auth_flow(&store, provider.as_ref()).await?;
//!
//! // Read a stored token in the L3 dispatch path:
//! if let Some(token) = store.load("gemini")? {
//!     // use token.access_token
//! }
//! ```

pub mod device_flow;
pub mod providers;
pub mod token_store;

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::bail;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

pub use token_store::TokenStore;

// ── Token types ───────────────────────────────────────────────────────────────

/// A stored OAuth or API-key credential, kept on disk in encrypted form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredToken {
    pub provider: String,
    /// The bearer token / API key used to authenticate requests.
    pub access_token: String,
    /// Used to obtain a new `access_token` when the current one expires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Unix timestamp (seconds) when this token expires, or `None` for no expiry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    #[serde(default = "default_token_type")]
    pub token_type: String,
    #[serde(default)]
    pub scopes: Vec<String>,
}

fn default_token_type() -> String {
    "Bearer".to_string()
}

impl Drop for StoredToken {
    fn drop(&mut self) {
        self.access_token.zeroize();
        if let Some(rt) = &mut self.refresh_token {
            rt.zeroize();
        }
    }
}

impl StoredToken {
    /// Returns `true` if the token has expired or will expire in the next 60 s.
    pub fn is_expired(&self) -> bool {
        let Some(expires_at) = self.expires_at else {
            return false;
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        now >= expires_at - 60
    }

    /// Seconds until expiry, negative if already expired.
    pub fn expires_in_secs(&self) -> Option<i64> {
        let expires_at = self.expires_at?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        Some(expires_at - now)
    }
}

// ── Device flow intermediary ──────────────────────────────────────────────────

/// State returned at the start of a device/browser OAuth flow.
#[derive(Debug, Clone)]
pub struct DeviceFlowState {
    pub device_code: String,
    pub user_code: Option<String>,
    pub verification_uri: Option<String>,
    /// Pre-filled URL that can be opened directly in a browser.
    pub verification_uri_complete: Option<String>,
    pub expires_in: u64,
    pub interval: u64,
}

/// Outcome of a single device-flow poll.
pub enum PollResult {
    Ready(StoredToken),
    /// Still waiting — caller should sleep `interval` seconds and retry.
    Pending,
    Error(String),
}

// ── OAuthProvider trait ───────────────────────────────────────────────────────

/// Authentication method used by a given provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMethod {
    /// RFC 8628 device authorization grant.
    DeviceFlow,
    /// OAuth2 with PKCE (browser redirect required).
    BrowserOAuth,
    /// No public OAuth — prompts for a manually-obtained API key.
    ApiKey,
}

/// Implemented by each OAuth-capable provider.
#[async_trait]
pub trait OAuthProvider: Send + Sync {
    fn provider_name(&self) -> &str;
    fn display_name(&self) -> &str;
    fn auth_method(&self) -> AuthMethod;

    /// Start the device authorization grant. Returns instructions to display.
    async fn start_device_flow(&self, _http: &Client) -> anyhow::Result<DeviceFlowState> {
        bail!("{} does not support device flow", self.provider_name());
    }

    /// Poll the token endpoint once during a device flow.
    async fn poll_device_token(
        &self,
        _http: &Client,
        _state: &DeviceFlowState,
    ) -> anyhow::Result<PollResult> {
        bail!(
            "{} does not support device flow polling",
            self.provider_name()
        );
    }

    /// Obtain a fresh `StoredToken` using a stored refresh token.
    async fn refresh_token(
        &self,
        _http: &Client,
        _refresh_token: &str,
    ) -> anyhow::Result<StoredToken> {
        bail!("{} does not support token refresh", self.provider_name());
    }

    /// Interactively prompt the user for an API key and return it as a token.
    async fn prompt_api_key(&self) -> anyhow::Result<StoredToken> {
        bail!("{} does not support API key prompt", self.provider_name());
    }
}

// ── Provider registry ─────────────────────────────────────────────────────────

/// Return all known OAuth provider implementations.
pub fn all_providers() -> Vec<Box<dyn OAuthProvider>> {
    vec![
        Box::new(providers::copilot::CopilotOAuth),
        Box::new(providers::gemini::GeminiOAuth),
        Box::new(providers::kiro::KiroOAuth),
        Box::new(providers::anthropic::AnthropicApiKey),
        Box::new(providers::openai::OpenAiApiKey),
    ]
}

/// Look up a provider by `provider_name` or `display_name` (case-insensitive).
pub fn find_provider(name: &str) -> Option<Box<dyn OAuthProvider>> {
    let name_lc = name.to_lowercase();
    all_providers()
        .into_iter()
        .find(|p| p.provider_name() == name_lc || p.display_name().to_lowercase() == name_lc)
}
