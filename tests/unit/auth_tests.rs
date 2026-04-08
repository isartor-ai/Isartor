// OAuth framework unit tests.
//
// Covers encrypted token persistence, expiry handling, and provider registry
// lookup for the shared auth module.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use isartor::auth::{StoredToken, TokenStore, find_provider};
use uuid::Uuid;

fn temp_store_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("isartor-auth-test-{}", Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[test]
fn token_store_round_trip_encrypts_payload() {
    let dir = temp_store_dir();
    let store = TokenStore::open_in(dir.clone()).unwrap();
    let token = StoredToken {
        provider: "openai".to_string(),
        access_token: "sk-test-secret-value".to_string(),
        refresh_token: Some("refresh-secret".to_string()),
        expires_at: Some(now_unix() + 3600),
        token_type: "Bearer".to_string(),
        scopes: vec!["scope:a".to_string()],
    };

    store.save(&token).unwrap();

    let raw = fs::read(dir.join("openai.enc")).unwrap();
    assert!(
        !String::from_utf8_lossy(&raw).contains("sk-test-secret-value"),
        "encrypted token file should not contain plaintext access token"
    );

    let loaded = store.load("openai").unwrap().unwrap();
    assert_eq!(loaded.provider, "openai");
    assert_eq!(loaded.access_token, "sk-test-secret-value");
    assert_eq!(loaded.refresh_token.as_deref(), Some("refresh-secret"));
    assert_eq!(loaded.scopes, vec!["scope:a"]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn token_store_lists_and_deletes_credentials() {
    let dir = temp_store_dir();
    let store = TokenStore::open_in(dir.clone()).unwrap();

    for provider in ["anthropic", "gemini"] {
        store
            .save(&StoredToken {
                provider: provider.to_string(),
                access_token: format!("{provider}-secret"),
                refresh_token: None,
                expires_at: None,
                token_type: "ApiKey".to_string(),
                scopes: vec![],
            })
            .unwrap();
    }

    let listed = store.list_authenticated().unwrap();
    assert_eq!(listed, vec!["anthropic".to_string(), "gemini".to_string()]);

    store.delete("anthropic").unwrap();
    assert!(store.load("anthropic").unwrap().is_none());
    assert_eq!(
        store.list_authenticated().unwrap(),
        vec!["gemini".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn stored_token_expiry_uses_grace_period() {
    let future = StoredToken {
        provider: "copilot".to_string(),
        access_token: "token".to_string(),
        refresh_token: None,
        expires_at: Some(now_unix() + 30),
        token_type: "Bearer".to_string(),
        scopes: vec![],
    };
    let mut farther_future = future.clone();
    farther_future.expires_at = Some(now_unix() + 3600);
    let mut no_expiry = future.clone();
    no_expiry.expires_at = None;

    assert!(
        future.is_expired(),
        "60s grace period should mark near-expiry tokens expired"
    );
    assert!(!farther_future.is_expired());
    assert!(!no_expiry.is_expired());
}

#[test]
fn provider_registry_finds_canonical_and_display_names() {
    let canonical = find_provider("openai").unwrap();
    assert_eq!(canonical.provider_name(), "openai");

    let display = find_provider("Google Gemini").unwrap();
    assert_eq!(display.provider_name(), "gemini");

    assert!(find_provider("missing-provider").is_none());
}
