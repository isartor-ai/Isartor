use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, bail};
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::get};
use pbkdf2::pbkdf2_hmac_array;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use toml_edit::DocumentMut;

const DEFAULT_SYNC_SERVER: &str = "http://127.0.0.1:8787";
const DEFAULT_CONFIG_PATH: &str = "./isartor.toml";
const PROFILE_FILE: &str = "sync-profile.json";
const SERVER_DIR: &str = "sync-server";
const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;
const KDF_ITERATIONS: u32 = 600_000;

const SYNCABLE_KEYS: &[&str] = &[
    "llm_provider",
    "external_llm_url",
    "external_llm_model",
    "external_llm_api_key",
    "provider_keys",
    "key_rotation_strategy",
    "key_cooldown_secs",
    "l3_timeout_secs",
    "azure_deployment_id",
    "azure_api_version",
    "enable_slm_router",
    "local_slm_url",
    "local_slm_model",
    "enable_context_optimizer",
    "context_optimizer_dedup",
    "context_optimizer_minify",
    "usage_window_hours",
    "fallback_providers",
    "model_aliases",
    "usage_pricing",
    "quota",
    "layer2",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncProfile {
    pub server: String,
    pub user_hash: String,
    pub salt_hex: String,
    pub config_path: String,
    pub last_remote_updated_at: Option<i64>,
    pub last_local_sync_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncBlob {
    pub version: u8,
    pub updated_at: i64,
    pub config_toml: String,
    pub config_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRecord {
    pub user_hash: String,
    pub salt_hex: String,
    pub encrypted_blob_hex: String,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutSyncRequest {
    pub salt_hex: String,
    pub encrypted_blob_hex: String,
    pub updated_at: i64,
    #[serde(default)]
    pub previous_updated_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResponse {
    pub error: String,
    pub current_updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct SyncServerState {
    pub data_dir: PathBuf,
}

pub fn default_sync_server() -> &'static str {
    DEFAULT_SYNC_SERVER
}

pub fn default_config_path() -> &'static str {
    DEFAULT_CONFIG_PATH
}

pub fn profile_path() -> anyhow::Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".isartor").join(PROFILE_FILE))
}

pub fn default_server_data_dir() -> anyhow::Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".isartor").join(SERVER_DIR))
}

pub fn derive_user_hash(identifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(identifier.trim().as_bytes());
    hex::encode(hasher.finalize())
}

pub fn generate_salt_hex() -> String {
    let mut salt = [0u8; KEY_LEN];
    OsRng.fill_bytes(&mut salt);
    hex::encode(salt)
}

pub fn save_profile(profile: &SyncProfile) -> anyhow::Result<()> {
    let path = profile_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    fs::write(&path, serde_json::to_vec_pretty(profile)?)
        .with_context(|| format!("cannot write {}", path.display()))
}

pub fn load_profile() -> anyhow::Result<SyncProfile> {
    let path = profile_path()?;
    let raw =
        fs::read(&path).with_context(|| format!("cannot read sync profile {}", path.display()))?;
    serde_json::from_slice(&raw).context("sync profile is invalid JSON")
}

pub fn config_sha256(raw: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

pub fn extract_syncable_config(raw: &str) -> anyhow::Result<String> {
    let source = raw
        .parse::<DocumentMut>()
        .context("failed to parse local isartor.toml")?;
    let mut filtered = DocumentMut::new();

    for key in SYNCABLE_KEYS {
        if let Some(item) = source.get(key)
            && !item.is_none()
        {
            filtered[*key] = item.clone();
        }
    }

    Ok(filtered.to_string())
}

pub fn merge_syncable_config(local_raw: &str, remote_sync_toml: &str) -> anyhow::Result<String> {
    let mut local = local_raw
        .parse::<DocumentMut>()
        .context("failed to parse local isartor.toml for merge")?;
    let remote = remote_sync_toml
        .parse::<DocumentMut>()
        .context("failed to parse remote synced config")?;

    for key in SYNCABLE_KEYS {
        local.remove(key);
        if let Some(item) = remote.get(key)
            && !item.is_none()
        {
            local[*key] = item.clone();
        }
    }

    Ok(local.to_string())
}

fn derive_key(passphrase: &str, salt_hex: &str) -> anyhow::Result<[u8; KEY_LEN]> {
    let salt = hex::decode(salt_hex).context("salt hex is invalid")?;
    if salt.len() != KEY_LEN {
        bail!("salt must be {KEY_LEN} bytes");
    }
    Ok(pbkdf2_hmac_array::<Sha256, KEY_LEN>(
        passphrase.as_bytes(),
        &salt,
        KDF_ITERATIONS,
    ))
}

pub fn encrypt_blob(blob: &SyncBlob, passphrase: &str, salt_hex: &str) -> anyhow::Result<String> {
    let key = derive_key(passphrase, salt_hex)?;
    let cipher = Aes256Gcm::new_from_slice(&key).context("failed to initialize cipher")?;
    let plaintext = serde_json::to_vec(blob).context("failed to serialize sync blob")?;

    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_ref())
        .map_err(|e| anyhow::anyhow!("sync encryption failed: {e}"))?;

    let mut out = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(hex::encode(out))
}

pub fn decrypt_blob(
    encrypted_blob_hex: &str,
    passphrase: &str,
    salt_hex: &str,
) -> anyhow::Result<SyncBlob> {
    let key = derive_key(passphrase, salt_hex)?;
    let cipher = Aes256Gcm::new_from_slice(&key).context("failed to initialize cipher")?;
    let blob = hex::decode(encrypted_blob_hex).context("encrypted blob hex is invalid")?;
    if blob.len() <= NONCE_LEN {
        bail!("encrypted blob is corrupt");
    }
    let (nonce_bytes, ciphertext) = blob.split_at(NONCE_LEN);
    let nonce = Nonce::from_slice(nonce_bytes);
    let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|_| {
        anyhow::anyhow!("failed to decrypt sync blob — passphrase or identifier may be incorrect")
    })?;
    serde_json::from_slice(&plaintext).context("decrypted sync blob is invalid JSON")
}

pub fn build_sync_blob(config_toml: String) -> SyncBlob {
    SyncBlob {
        version: 1,
        updated_at: now_unix_secs(),
        config_sha256: config_sha256(&config_toml),
        config_toml,
    }
}

pub fn detect_push_conflict(profile: &SyncProfile, remote: Option<&SyncRecord>) -> bool {
    match (profile.last_remote_updated_at, remote.map(|r| r.updated_at)) {
        (_, None) => false,
        (None, Some(_)) => true,
        (Some(local), Some(remote)) => local != remote,
    }
}

pub fn detect_pull_conflict(
    profile: &SyncProfile,
    remote: &SyncRecord,
    local_config_path: &Path,
) -> anyhow::Result<bool> {
    let Some(last_local_sync_at) = profile.last_local_sync_at else {
        return Ok(false);
    };

    let modified = file_modified_unix_secs(local_config_path)?;
    let locally_changed = modified > last_local_sync_at;
    let remote_changed = profile
        .last_remote_updated_at
        .map(|seen| seen != remote.updated_at)
        .unwrap_or(true);

    Ok(locally_changed && remote_changed)
}

pub fn file_modified_unix_secs(path: &Path) -> anyhow::Result<i64> {
    let metadata = fs::metadata(path).with_context(|| format!("cannot stat {}", path.display()))?;
    let modified = metadata
        .modified()
        .with_context(|| format!("cannot read mtime for {}", path.display()))?;
    Ok(modified
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64)
}

pub fn ensure_config_exists(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        return Ok(());
    }
    bail!("config file {} does not exist", path.display())
}

pub fn backup_path(path: &Path) -> PathBuf {
    let mut backup = path.as_os_str().to_os_string();
    backup.push(".sync-backup");
    PathBuf::from(backup)
}

pub fn save_backup(path: &Path, contents: &str) -> anyhow::Result<PathBuf> {
    let backup = backup_path(path);
    fs::write(&backup, contents)
        .with_context(|| format!("cannot write backup {}", backup.display()))?;
    Ok(backup)
}

pub async fn fetch_remote_record(
    http: &reqwest::Client,
    server: &str,
    user_hash: &str,
) -> anyhow::Result<Option<SyncRecord>> {
    let url = format!("{}/sync/{user_hash}", server.trim_end_matches('/'));
    let resp = http
        .get(&url)
        .send()
        .await
        .context("sync GET request failed")?;
    match resp.status() {
        StatusCode::OK => Ok(Some(
            resp.json()
                .await
                .context("invalid sync GET response JSON")?,
        )),
        StatusCode::NOT_FOUND => Ok(None),
        other => {
            let body = resp.text().await.unwrap_or_default();
            bail!("sync GET failed with HTTP {other}: {body}")
        }
    }
}

pub async fn push_remote_record(
    http: &reqwest::Client,
    server: &str,
    user_hash: &str,
    req: &PutSyncRequest,
) -> anyhow::Result<SyncRecord> {
    let url = format!("{}/sync/{user_hash}", server.trim_end_matches('/'));
    let resp = http
        .put(&url)
        .json(req)
        .send()
        .await
        .context("sync PUT request failed")?;

    match resp.status() {
        StatusCode::OK | StatusCode::CREATED => Ok(resp
            .json()
            .await
            .context("invalid sync PUT response JSON")?),
        StatusCode::CONFLICT => {
            let conflict: ConflictResponse = resp
                .json()
                .await
                .context("invalid conflict response JSON")?;
            bail!(
                "remote sync conflict detected (server updated_at={}); re-run with --force or pull first",
                conflict.current_updated_at
            )
        }
        other => {
            let body = resp.text().await.unwrap_or_default();
            bail!("sync PUT failed with HTTP {other}: {body}")
        }
    }
}

pub fn sync_server_router(data_dir: PathBuf) -> Router {
    Router::new()
        .route("/sync/{user_hash}", get(get_sync_blob).put(put_sync_blob))
        .with_state(SyncServerState { data_dir })
}

pub async fn run_sync_server(listen: &str, data_dir: PathBuf) -> anyhow::Result<()> {
    fs::create_dir_all(&data_dir)
        .with_context(|| format!("cannot create sync data dir {}", data_dir.display()))?;
    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("failed to bind sync server on {listen}"))?;
    axum::serve(listener, sync_server_router(data_dir))
        .await
        .context("sync server failed")
}

async fn get_sync_blob(
    AxumPath(user_hash): AxumPath<String>,
    State(state): State<SyncServerState>,
) -> impl IntoResponse {
    if !is_valid_user_hash(&user_hash) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "user_hash must be 64 hex characters"})),
        )
            .into_response();
    }

    match load_sync_record(&state.data_dir, &user_hash) {
        Ok(Some(record)) => (StatusCode::OK, Json(record)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "sync record not found"})),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

async fn put_sync_blob(
    AxumPath(user_hash): AxumPath<String>,
    State(state): State<SyncServerState>,
    Json(request): Json<PutSyncRequest>,
) -> impl IntoResponse {
    if !is_valid_user_hash(&user_hash) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "user_hash must be 64 hex characters"})),
        )
            .into_response();
    }
    if request.updated_at <= 0 {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": "updated_at must be greater than zero"})),
        )
            .into_response();
    }

    let existing = match load_sync_record(&state.data_dir, &user_hash) {
        Ok(record) => record,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };

    if let Some(current) = existing.as_ref()
        && request.previous_updated_at != Some(current.updated_at)
    {
        return (
            StatusCode::CONFLICT,
            Json(ConflictResponse {
                error: "remote record changed".to_string(),
                current_updated_at: current.updated_at,
            }),
        )
            .into_response();
    }

    let record = SyncRecord {
        user_hash,
        salt_hex: request.salt_hex,
        encrypted_blob_hex: request.encrypted_blob_hex,
        updated_at: request.updated_at,
    };

    match save_sync_record(&state.data_dir, &record) {
        Ok(created) => (
            if created {
                StatusCode::CREATED
            } else {
                StatusCode::OK
            },
            Json(record),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

fn is_valid_user_hash(user_hash: &str) -> bool {
    user_hash.len() == 64 && user_hash.chars().all(|c| c.is_ascii_hexdigit())
}

fn record_path(data_dir: &Path, user_hash: &str) -> PathBuf {
    data_dir.join(format!("{user_hash}.json"))
}

fn load_sync_record(data_dir: &Path, user_hash: &str) -> anyhow::Result<Option<SyncRecord>> {
    let path = record_path(data_dir, user_hash);
    if !path.exists() {
        return Ok(None);
    }
    let raw =
        fs::read(&path).with_context(|| format!("cannot read sync record {}", path.display()))?;
    let record = serde_json::from_slice(&raw).context("sync record on disk is invalid JSON")?;
    Ok(Some(record))
}

fn save_sync_record(data_dir: &Path, record: &SyncRecord) -> anyhow::Result<bool> {
    fs::create_dir_all(data_dir)
        .with_context(|| format!("cannot create sync data dir {}", data_dir.display()))?;
    let path = record_path(data_dir, &record.user_hash);
    let created = !path.exists();
    fs::write(&path, serde_json::to_vec_pretty(record)?)
        .with_context(|| format!("cannot write sync record {}", path.display()))?;
    Ok(created)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[test]
    fn extract_syncable_config_excludes_local_only_keys() {
        let raw = r#"
host_port = "0.0.0.0:8080"
proxy_port = "0.0.0.0:8081"
llm_provider = "openai"
external_llm_model = "gpt-4o"
external_llm_api_key = "secret"
usage_window_hours = 24
request_log_path = "~/.isartor/request_logs"
[model_aliases]
fast = "gpt-4o-mini"
"#;
        let filtered = extract_syncable_config(raw).unwrap();
        assert!(filtered.contains("llm_provider = \"openai\""));
        assert!(filtered.contains("external_llm_api_key = \"secret\""));
        assert!(filtered.contains("usage_window_hours = 24"));
        assert!(filtered.contains("[model_aliases]"));
        assert!(!filtered.contains("host_port"));
        assert!(!filtered.contains("proxy_port"));
        assert!(!filtered.contains("request_log_path"));
    }

    #[test]
    fn merge_syncable_config_replaces_only_allowed_keys() {
        let local = r#"
host_port = "0.0.0.0:8080"
llm_provider = "openai"
external_llm_model = "gpt-4o-mini"
"#;
        let remote = r#"
llm_provider = "anthropic"
external_llm_model = "claude-3-7-sonnet"
"#;
        let merged = merge_syncable_config(local, remote).unwrap();
        assert!(merged.contains("host_port = \"0.0.0.0:8080\""));
        assert!(merged.contains("llm_provider = \"anthropic\""));
        assert!(merged.contains("external_llm_model = \"claude-3-7-sonnet\""));
    }

    #[test]
    fn encrypt_decrypt_round_trip() {
        let blob = build_sync_blob("llm_provider = \"openai\"\n".to_string());
        let salt = generate_salt_hex();
        let encrypted = encrypt_blob(&blob, "passphrase", &salt).unwrap();
        let decrypted = decrypt_blob(&encrypted, "passphrase", &salt).unwrap();
        assert_eq!(decrypted.config_toml, blob.config_toml);
        assert_eq!(decrypted.config_sha256, blob.config_sha256);
    }

    #[test]
    fn detect_push_conflict_when_remote_has_newer_version() {
        let profile = SyncProfile {
            server: default_sync_server().to_string(),
            user_hash: "u".repeat(64),
            salt_hex: generate_salt_hex(),
            config_path: default_config_path().to_string(),
            last_remote_updated_at: Some(10),
            last_local_sync_at: Some(10),
        };
        let remote = SyncRecord {
            user_hash: "u".repeat(64),
            salt_hex: generate_salt_hex(),
            encrypted_blob_hex: "aa".to_string(),
            updated_at: 11,
        };
        assert!(detect_push_conflict(&profile, Some(&remote)));
    }

    #[tokio::test]
    async fn sync_server_put_get_and_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let app = sync_server_router(dir.path().to_path_buf());
        let user_hash = "a".repeat(64);

        let request = Request::builder()
            .method("PUT")
            .uri(format!("/sync/{user_hash}"))
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&PutSyncRequest {
                    salt_hex: generate_salt_hex(),
                    encrypted_blob_hex: "beef".to_string(),
                    updated_at: 1,
                    previous_updated_at: None,
                })
                .unwrap(),
            ))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);

        let get = Request::builder()
            .uri(format!("/sync/{user_hash}"))
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(get).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let conflict = Request::builder()
            .method("PUT")
            .uri(format!("/sync/{user_hash}"))
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&PutSyncRequest {
                    salt_hex: generate_salt_hex(),
                    encrypted_blob_hex: "cafe".to_string(),
                    updated_at: 2,
                    previous_updated_at: Some(0),
                })
                .unwrap(),
            ))
            .unwrap();
        let response = app.oneshot(conflict).await.unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }
}
