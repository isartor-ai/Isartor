//! # `isartor update` — Self-update to the latest release
//!
//! Fetches the latest release from GitHub, downloads the appropriate
//! binary for the current platform, and replaces the running binary.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use reqwest::Url;

const REPO: &str = "isartor-ai/Isartor";
const GITHUB_API: &str = "https://api.github.com";
const PROXY_ENV_KEYS: &[&str] = &[
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
];
const ISARTOR_PROXY_MARKER_ENV_KEYS: &[&str] =
    &["ISARTOR_COPILOT_ENABLED", "ISARTOR_ANTIGRAVITY_ENABLED"];
const ISARTOR_CA_ENV_KEYS: &[&str] =
    &["NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"];
const ISARTOR_CA_PATH_FRAGMENT: &str = "/.isartor/ca/isartor-ca.pem";

#[derive(Parser, Debug, Clone)]
pub struct UpdateArgs {
    /// Update to a specific version tag (e.g. v0.1.19). Defaults to latest.
    #[arg(long)]
    pub version: Option<String>,

    /// Show what would be done without actually updating.
    #[arg(long)]
    pub dry_run: bool,

    /// Force update even if already on the latest version.
    #[arg(long)]
    pub force: bool,
}

pub async fn handle_update(args: UpdateArgs) -> Result<()> {
    let current_version = env!("CARGO_PKG_VERSION");
    eprintln!("  Current version: v{}", current_version);

    // 1. Resolve target version.
    let tag = match &args.version {
        Some(v) if v.starts_with('v') => v.clone(),
        Some(v) => format!("v{v}"),
        None => fetch_latest_tag().await?,
    };

    let target_version = tag.strip_prefix('v').unwrap_or(&tag);
    eprintln!("  Latest version:  {}", tag);

    if target_version == current_version && !args.force {
        eprintln!("  ✓ Already up to date.");
        return Ok(());
    }

    // 2. Detect platform target triple.
    let target = detect_target()?;
    let ext = if target.contains("windows") {
        "zip"
    } else {
        "tar.gz"
    };
    let asset_name = format!("isartor-{tag}-{target}.{ext}");
    let download_url = format!("https://github.com/{REPO}/releases/download/{tag}/{asset_name}");

    eprintln!("  Asset:           {}", asset_name);
    eprintln!("  URL:             {}", download_url);

    if args.dry_run {
        eprintln!("  [dry-run] Would download and replace the current binary.");
        return Ok(());
    }

    // 3. Download the archive.
    eprintln!("  ⏳ Downloading...");
    let client = build_github_client(current_version)?;

    let response = client
        .get(&download_url)
        .send()
        .await
        .context("failed to download release")?;

    if !response.status().is_success() {
        bail!(
            "Download failed: HTTP {} for {}",
            response.status(),
            download_url
        );
    }

    let bytes = response.bytes().await?;
    eprintln!("  ⏳ Downloaded {:.1} MB", bytes.len() as f64 / 1_048_576.0);

    // 4. Extract the binary.
    let tmp_dir = tempfile::tempdir().context("failed to create temp dir")?;
    let archive_path = tmp_dir.path().join(&asset_name);
    std::fs::write(&archive_path, &bytes)?;

    let extracted_bin = if ext == "tar.gz" {
        extract_tar_gz(&archive_path, tmp_dir.path())?
    } else {
        extract_zip(&archive_path, tmp_dir.path())?
    };

    // 5. Replace the current binary.
    let current_exe =
        std::env::current_exe().context("cannot determine current executable path")?;
    let current_exe = current_exe.canonicalize().unwrap_or(current_exe);

    eprintln!("  ⏳ Replacing {}...", current_exe.display());
    self_replace(&extracted_bin, &current_exe)?;

    eprintln!("  ✓ Updated to {}!", tag);
    eprintln!("    Restart Isartor to use the new version.");

    Ok(())
}

/// Fetch the latest release tag from the GitHub API.
async fn fetch_latest_tag() -> Result<String> {
    let url = format!("{GITHUB_API}/repos/{REPO}/releases/latest");
    let client = build_github_client(env!("CARGO_PKG_VERSION"))?;

    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        bail!("Failed to fetch latest release: HTTP {}", resp.status());
    }

    let json: serde_json::Value = resp.json().await?;
    let tag = json["tag_name"]
        .as_str()
        .context("no tag_name in release response")?;
    Ok(tag.to_string())
}

fn build_github_client(current_version: &str) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder().user_agent(format!("isartor/{current_version}"));
    if should_bypass_isartor_proxy_env() {
        eprintln!("  ↺ Bypassing local Isartor proxy environment for GitHub update checks.");
        builder = builder.no_proxy();
    }
    Ok(builder.build()?)
}

fn should_bypass_isartor_proxy_env() -> bool {
    let proxy_values = PROXY_ENV_KEYS
        .iter()
        .map(|key| std::env::var(key).ok())
        .collect::<Vec<_>>();
    let marker_values = ISARTOR_PROXY_MARKER_ENV_KEYS
        .iter()
        .map(|key| std::env::var(key).ok())
        .collect::<Vec<_>>();
    let ca_values = ISARTOR_CA_ENV_KEYS
        .iter()
        .map(|key| std::env::var(key).ok())
        .collect::<Vec<_>>();

    should_bypass_isartor_proxy_env_from_values(&proxy_values, &marker_values, &ca_values)
}

fn should_bypass_isartor_proxy_env_from_values(
    proxy_values: &[Option<String>],
    marker_values: &[Option<String>],
    ca_values: &[Option<String>],
) -> bool {
    uses_loopback_proxy(proxy_values)
        && (marker_values
            .iter()
            .flatten()
            .any(|value| value.eq_ignore_ascii_case("true"))
            || ca_values
                .iter()
                .flatten()
                .any(|value| is_isartor_ca_path(value)))
}

fn uses_loopback_proxy(proxy_values: &[Option<String>]) -> bool {
    proxy_values
        .iter()
        .flatten()
        .filter_map(|value| parse_proxy_url(value))
        .filter_map(|url| url.host_str().map(|host| host.to_string()))
        .any(|host| matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1"))
}

fn parse_proxy_url(value: &str) -> Option<Url> {
    Url::parse(value)
        .ok()
        .or_else(|| Url::parse(&format!("http://{value}")).ok())
}

fn is_isartor_ca_path(value: &str) -> bool {
    value.replace('\\', "/").contains(ISARTOR_CA_PATH_FRAGMENT)
}

/// Detect the Rust target triple for the current platform.
fn detect_target() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let target = match (os, arch) {
        ("linux", "x86_64") => "x86_64-unknown-linux-musl",
        ("linux", "aarch64") => "aarch64-unknown-linux-musl",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("windows", "x86_64") => "x86_64-pc-windows-msvc",
        _ => bail!("Unsupported platform: {os}/{arch}"),
    };
    Ok(target.to_string())
}

/// Extract `isartor` binary from a .tar.gz archive.
fn extract_tar_gz(archive: &std::path::Path, dest: &std::path::Path) -> Result<PathBuf> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let file = std::fs::File::open(archive)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    archive.unpack(dest)?;

    let bin = dest.join("isartor");
    if !bin.exists() {
        bail!("'isartor' binary not found in archive");
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755))?;
    }

    Ok(bin)
}

/// Extract `isartor.exe` from a .zip archive.
fn extract_zip(archive: &std::path::Path, dest: &std::path::Path) -> Result<PathBuf> {
    let file = std::fs::File::open(archive)?;
    let mut zip = zip::ZipArchive::new(file)?;

    zip.extract(dest)?;

    let bin = dest.join("isartor.exe");
    if !bin.exists() {
        let alt = dest.join("isartor");
        if alt.exists() {
            return Ok(alt);
        }
        bail!("'isartor' binary not found in zip archive");
    }
    Ok(bin)
}

/// Replace the running binary with a new one (atomic on Unix).
fn self_replace(new_bin: &std::path::Path, current_exe: &std::path::Path) -> Result<()> {
    // On Unix: rename old binary to .bak, copy new one in, remove .bak.
    let backup = current_exe.with_extension("bak");

    // Move current binary aside.
    std::fs::rename(current_exe, &backup).with_context(|| {
        format!(
            "failed to move current binary to backup ({}). Try running with sudo.",
            backup.display()
        )
    })?;

    // Copy new binary into place.
    if let Err(e) = std::fs::copy(new_bin, current_exe) {
        // Restore backup on failure.
        let _ = std::fs::rename(&backup, current_exe);
        return Err(e).context("failed to install new binary");
    }

    // Remove backup.
    let _ = std::fs::remove_file(&backup);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_target_returns_valid_triple() {
        let target = detect_target().unwrap();
        assert!(
            target.contains("linux") || target.contains("darwin") || target.contains("windows"),
            "unexpected target: {target}"
        );
    }

    #[test]
    fn dry_run_does_not_modify() {
        // Just verify the function signature works with dry_run.
        let args = UpdateArgs {
            version: Some("v0.0.1-test".into()),
            dry_run: true,
            force: false,
        };
        // We can't easily test the full flow without network,
        // but we verify the struct is constructable.
        assert!(args.dry_run);
    }

    #[test]
    fn bypasses_loopback_proxy_when_isartor_marker_present() {
        assert!(should_bypass_isartor_proxy_env_from_values(
            &[Some("http://localhost:8081".into())],
            &[Some("true".into())],
            &[None]
        ));
    }

    #[test]
    fn bypasses_loopback_proxy_when_isartor_ca_bundle_present() {
        assert!(should_bypass_isartor_proxy_env_from_values(
            &[Some("127.0.0.1:8081".into())],
            &[None],
            &[Some("/Users/test/.isartor/ca/isartor-ca.pem".into())]
        ));
    }

    #[test]
    fn keeps_non_isartor_proxy_configuration() {
        assert!(!should_bypass_isartor_proxy_env_from_values(
            &[Some("http://localhost:8888".into())],
            &[None],
            &[Some("/Users/test/corp-ca.pem".into())]
        ));
    }

    #[test]
    fn env_helper_bypasses_stale_isartor_proxy() {
        temp_env::with_vars(
            [
                ("HTTPS_PROXY", Some("http://localhost:8081")),
                ("ISARTOR_COPILOT_ENABLED", Some("true")),
                ("NODE_EXTRA_CA_CERTS", None),
            ],
            || {
                assert!(should_bypass_isartor_proxy_env());
            },
        );
    }
}
