use axum::http::HeaderMap;
use serde_json::Value;
use sha2::{Digest, Sha256};

const SESSION_HEADER_NAMES: [&str; 4] = [
    "x-isartor-session-id",
    "x-thread-id",
    "x-session-id",
    "x-conversation-id",
];
const SESSION_BODY_FIELD_NAMES: [&str; 3] = ["session_id", "thread_id", "conversation_id"];
const MAX_SESSION_IDENTIFIER_LEN: usize = 512;

pub fn extract_session_cache_scope(headers: &HeaderMap, body: &[u8]) -> Option<String> {
    extract_session_identifier_from_headers(headers)
        .or_else(|| extract_session_identifier_from_body(body))
        .and_then(|identifier| derive_session_cache_scope(&identifier))
}

pub fn derive_session_cache_scope(identifier: &str) -> Option<String> {
    normalize_session_identifier(identifier)
        .map(|normalized| hex::encode(Sha256::digest(normalized.as_bytes())))
}

pub fn build_exact_cache_key(
    cache_namespace: &str,
    cache_key_material: &str,
    session_cache_scope: Option<&str>,
) -> String {
    hex::encode(Sha256::digest(
        build_exact_cache_input(cache_namespace, cache_key_material, session_cache_scope)
            .as_bytes(),
    ))
}

pub fn namespaced_semantic_cache_input(cache_namespace: &str, semantic_prompt: &str) -> String {
    format!("{cache_namespace}|{semantic_prompt}")
}

fn build_exact_cache_input(
    cache_namespace: &str,
    cache_key_material: &str,
    session_cache_scope: Option<&str>,
) -> String {
    match session_cache_scope {
        Some(scope) => format!("{cache_namespace}|session:{scope}|{cache_key_material}"),
        None => format!("{cache_namespace}|{cache_key_material}"),
    }
}

fn extract_session_identifier_from_headers(headers: &HeaderMap) -> Option<String> {
    SESSION_HEADER_NAMES.iter().find_map(|name| {
        headers
            .get(*name)
            .and_then(|value| value.to_str().ok())
            .and_then(normalize_session_identifier)
    })
}

pub fn extract_session_identifier_from_body(body: &[u8]) -> Option<String> {
    let value = serde_json::from_slice::<Value>(body).ok()?;

    SESSION_BODY_FIELD_NAMES
        .iter()
        .find_map(|field| value.get(*field).and_then(Value::as_str))
        .and_then(normalize_session_identifier)
        .or_else(|| {
            value
                .get("metadata")
                .and_then(Value::as_object)
                .and_then(|metadata| {
                    SESSION_BODY_FIELD_NAMES
                        .iter()
                        .find_map(|field| metadata.get(*field).and_then(Value::as_str))
                })
                .and_then(normalize_session_identifier)
        })
}

fn normalize_session_identifier(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > MAX_SESSION_IDENTIFIER_LEN {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_session_identifier_takes_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert("x-thread-id", "thread-from-header".parse().unwrap());
        let body = br#"{"session_id":"session-from-body"}"#;

        let scope = extract_session_cache_scope(&headers, body);

        assert_eq!(scope, derive_session_cache_scope("thread-from-header"),);
    }

    #[test]
    fn extracts_session_identifier_from_top_level_body_fields() {
        let body = br#"{"thread_id":"thread-123"}"#;

        assert_eq!(
            extract_session_identifier_from_body(body),
            Some("thread-123".to_string())
        );
    }

    #[test]
    fn extracts_session_identifier_from_metadata_body_fields() {
        let body = br#"{"metadata":{"conversation_id":"conv-456"}}"#;

        assert_eq!(
            extract_session_identifier_from_body(body),
            Some("conv-456".to_string())
        );
    }

    #[test]
    fn rejects_blank_or_oversized_session_identifiers() {
        let oversized = "a".repeat(MAX_SESSION_IDENTIFIER_LEN + 1);

        assert_eq!(derive_session_cache_scope("   "), None);
        assert_eq!(derive_session_cache_scope(&oversized), None);
    }

    #[test]
    fn exact_cache_key_changes_when_session_scope_changes() {
        let session_a = derive_session_cache_scope("session-a");
        let session_b = derive_session_cache_scope("session-b");

        assert_ne!(
            build_exact_cache_key("native", "prompt", session_a.as_deref()),
            build_exact_cache_key("native", "prompt", session_b.as_deref())
        );
        assert_ne!(
            build_exact_cache_key("native", "prompt", None),
            build_exact_cache_key("native", "prompt", session_a.as_deref())
        );
    }
}
