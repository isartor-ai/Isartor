//! HTTP CONNECT proxy with TLS MITM for intercepting Copilot CLI traffic.
//!
//! Runs a raw TCP listener (separate from the Axum gateway) that speaks
//! the HTTP CONNECT protocol. Allowed domains get TLS-terminated with a
//! leaf certificate signed by the local Isartor CA, then their request
//! bodies are routed through the Deflection Stack (L1a exact cache →
//! L1b semantic cache → L3 cloud LLM).
//!
//! Non-allowed domains and non-chat-completion paths are tunnelled
//! transparently without interception.

use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::BytesMut;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;

use crate::config::CacheMode;
use crate::core::prompt::extract_prompt;
use crate::proxy::tls::IsartorCa;
use crate::state::AppState;

/// Domains that may be intercepted via TLS MITM.
const ALLOWED_DOMAINS: &[&str] = &[
    "copilot-proxy.githubusercontent.com",
    "api.github.com",
    "api.individual.githubcopilot.com",
    "api.business.githubcopilot.com",
    "api.enterprise.githubcopilot.com",
];

/// HTTP paths that trigger Deflection Stack interception.
const INTERCEPTED_PATHS: &[&str] = &["/v1/chat/completions"];

/// Start the CONNECT proxy on the given address.
///
/// This function blocks (via `loop { accept }`) and should be spawned
/// as a separate task. Errors on individual connections are logged and
/// do not crash the proxy.
pub async fn run_connect_proxy(
    bind_addr: &str,
    ca: Arc<IsartorCa>,
    state: Arc<AppState>,
) -> Result<()> {
    let listener = TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("CONNECT proxy: failed to bind {bind_addr}"))?;

    tracing::info!(addr = %bind_addr, "CONNECT proxy listening");

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                tracing::warn!(error = %e, "CONNECT proxy: accept error");
                continue;
            }
        };

        let ca = ca.clone();
        let state = state.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connect(stream, ca, state).await {
                tracing::debug!(peer = %peer, error = %e, "CONNECT proxy: connection error");
            }
        });
    }
}

/// Handle a single CONNECT request.
async fn handle_connect(
    mut client: TcpStream,
    ca: Arc<IsartorCa>,
    state: Arc<AppState>,
) -> Result<()> {
    let mut reader = BufReader::new(&mut client);

    // Read the CONNECT request line: "CONNECT host:port HTTP/1.1\r\n"
    let mut request_line = String::new();
    reader.read_line(&mut request_line).await?;

    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 3 || parts[0] != "CONNECT" {
        send_error(&mut client, 400, "Bad Request").await?;
        return Ok(());
    }

    let host_port = parts[1];
    let (hostname, port) = parse_host_port(host_port)?;

    // Consume remaining headers until the blank line.
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        if line.trim().is_empty() {
            break;
        }
    }

    // Decide: intercept or tunnel.
    let should_intercept = ALLOWED_DOMAINS.iter().any(|d| hostname == *d);

    // Send 200 Connection Established — tunnel is open.
    client
        .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        .await?;
    client.flush().await?;

    if should_intercept {
        handle_mitm(client, &hostname, port, ca, state).await
    } else {
        handle_tunnel(client, &hostname, port).await
    }
}

/// TLS MITM path: terminate TLS, inspect HTTP, route through Deflection Stack.
async fn handle_mitm(
    client: TcpStream,
    hostname: &str,
    port: u16,
    ca: Arc<IsartorCa>,
    state: Arc<AppState>,
) -> Result<()> {
    // Generate a leaf cert for this hostname and accept TLS from client.
    let server_config = ca.server_config_for_host(hostname)?;
    let acceptor = TlsAcceptor::from(server_config);

    let mut tls_stream = acceptor
        .accept(client)
        .await
        .context("MITM: TLS handshake with client failed")?;

    // Read the inner HTTP request from the TLS stream.
    let (method, path, headers, body) = read_http_request(&mut tls_stream).await?;

    tracing::debug!(
        hostname = %hostname,
        method = %method,
        path = %path,
        body_len = body.len(),
        "MITM: intercepted request"
    );

    // Only intercept POST to chat completions paths.
    if method == "POST" && INTERCEPTED_PATHS.iter().any(|p| path == *p) {
        // Route through Deflection Stack.
        match deflection_stack(&body, &path, &state).await {
            Some((status_code, response_body)) => {
                let resp = format_http_response(status_code, &response_body);
                tls_stream.write_all(resp.as_bytes()).await?;
                tls_stream.flush().await?;
                return Ok(());
            }
            None => {
                // Cache miss — forward to upstream.
            }
        }
    }

    // Forward to the real upstream (transparent for non-intercepted paths or cache misses).
    let upstream_response =
        forward_to_upstream(hostname, port, &method, &path, &headers, &body).await?;

    // If this was an intercepted path, cache the upstream response.
    if method == "POST" && INTERCEPTED_PATHS.iter().any(|p| path == *p) {
        cache_response(&body, &path, &upstream_response, &state).await;
    }

    tls_stream.write_all(&upstream_response).await?;
    tls_stream.flush().await?;
    Ok(())
}

/// Transparent tunnel: splice client ↔ upstream TCP without inspection.
async fn handle_tunnel(mut client: TcpStream, hostname: &str, port: u16) -> Result<()> {
    let mut upstream = TcpStream::connect(format!("{hostname}:{port}"))
        .await
        .with_context(|| format!("Tunnel: failed to connect to {hostname}:{port}"))?;

    tokio::io::copy_bidirectional(&mut client, &mut upstream).await?;
    Ok(())
}

// ── Deflection Stack ─────────────────────────────────────────────────

/// Run the prompt through the Isartor Deflection Stack (L1a → L1b → L3).
///
/// Returns `Some((status_code, body))` on a cache hit, or `None` on a miss
/// (caller should forward to the real upstream).
async fn deflection_stack(body: &[u8], path: &str, state: &Arc<AppState>) -> Option<(u16, String)> {
    let prompt = extract_prompt(body);

    if prompt.is_empty() {
        return None;
    }

    // Namespace-keyed to avoid cross-format cache collisions.
    let cache_ns = match path {
        "/v1/chat/completions" => "openai",
        "/v1/messages" => "anthropic",
        _ => "proxy",
    };
    let cache_prompt = format!("{cache_ns}|{prompt}");
    let mode = &state.config.cache_mode;

    // ── L1a: Exact-match cache ──────────────────────────────────────
    let exact_key = if *mode == CacheMode::Exact || *mode == CacheMode::Both {
        let key = hex::encode(Sha256::digest(cache_prompt.as_bytes()));
        if let Some(cached) = state.exact_cache.get(&key) {
            tracing::info!(cache.key = %key, "Proxy L1a: exact cache HIT");
            return Some((200, cached));
        }
        Some(key)
    } else {
        None
    };

    // ── L1b: Semantic cache ─────────────────────────────────────────
    let embedding: Option<Vec<f32>> = if *mode == CacheMode::Semantic || *mode == CacheMode::Both {
        match state.text_embedder.generate_embedding(&prompt) {
            Ok(emb) => {
                if let Some(cached) = state.vector_cache.search(&emb).await {
                    tracing::info!("Proxy L1b: semantic cache HIT");
                    return Some((200, cached));
                }
                Some(emb)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Proxy L1b: embedding generation failed, skipping");
                None
            }
        }
    } else {
        None
    };

    // ── L3: Cloud LLM ───────────────────────────────────────────────
    if state.config.offline_mode {
        tracing::debug!("Proxy: offline mode, skipping L3");
        return None; // Let the caller forward to upstream (which will also fail, but keeps the flow consistent)
    }

    match state.llm_agent.chat(&prompt).await {
        Ok(llm_response) => {
            // Build an OpenAI-compatible response (since Copilot expects this format).
            let response_json = serde_json::json!({
                "id": format!("isartor-{}", uuid::Uuid::new_v4()),
                "object": "chat.completion",
                "created": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                "model": "isartor-proxy",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": llm_response,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            });
            let resp_string = response_json.to_string();

            // Cache the LLM response.
            if let Some(key) = exact_key {
                state.exact_cache.put(key, resp_string.clone());
            }
            if let Some(emb) = embedding {
                state.vector_cache.insert(emb, resp_string.clone()).await;
            }

            tracing::info!(provider = %state.llm_agent.provider_name(), "Proxy L3: LLM response cached");
            Some((200, resp_string))
        }
        Err(e) => {
            tracing::warn!(error = %e, "Proxy L3: LLM call failed, forwarding to upstream");
            None
        }
    }
}

/// Cache a response received from the real upstream.
async fn cache_response(body: &[u8], path: &str, response: &[u8], state: &Arc<AppState>) {
    let prompt = extract_prompt(body);
    if prompt.is_empty() {
        return;
    }

    let cache_ns = match path {
        "/v1/chat/completions" => "openai",
        "/v1/messages" => "anthropic",
        _ => "proxy",
    };
    let cache_prompt = format!("{cache_ns}|{prompt}");
    let mode = &state.config.cache_mode;

    let resp_string = String::from_utf8_lossy(response).to_string();

    // Try to find just the JSON body (skip HTTP headers).
    let cache_value = if let Some(body_start) = resp_string.find("\r\n\r\n") {
        resp_string[body_start + 4..].to_string()
    } else {
        resp_string
    };

    if *mode == CacheMode::Exact || *mode == CacheMode::Both {
        let key = hex::encode(Sha256::digest(cache_prompt.as_bytes()));
        state.exact_cache.put(key, cache_value.clone());
    }

    if *mode == CacheMode::Semantic || *mode == CacheMode::Both {
        if let Ok(emb) = state.text_embedder.generate_embedding(&prompt) {
            state.vector_cache.insert(emb, cache_value).await;
        }
    }
}

// ── HTTP helpers ─────────────────────────────────────────────────────

/// Read a full HTTP/1.1 request from a stream.
/// Returns (method, path, raw_headers_string, body_bytes).
async fn read_http_request<S: AsyncReadExt + AsyncBufReadExt + Unpin>(
    stream: &mut S,
) -> Result<(String, String, String, Vec<u8>)> {
    let mut headers_buf = String::new();
    let mut reader = BufReader::new(stream);

    // Read request line.
    let mut request_line = String::new();
    reader.read_line(&mut request_line).await?;
    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    let method = parts.first().unwrap_or(&"GET").to_string();
    let path = parts.get(1).unwrap_or(&"/").to_string();

    // Read headers.
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        if line.trim().is_empty() {
            break;
        }
        if let Some(val) = line
            .strip_prefix("Content-Length:")
            .or_else(|| line.strip_prefix("content-length:"))
        {
            content_length = val.trim().parse().unwrap_or(0);
        }
        headers_buf.push_str(&line);
    }

    // Read body.
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).await?;
    }

    Ok((method, path, headers_buf, body))
}

/// Forward a request to the real upstream server over TLS.
async fn forward_to_upstream(
    hostname: &str,
    port: u16,
    method: &str,
    path: &str,
    headers: &str,
    body: &[u8],
) -> Result<Vec<u8>> {
    // Use reqwest for simplicity — it handles TLS to the real upstream.
    let url = format!("https://{hostname}:{port}{path}");

    let client = reqwest::Client::new();
    let mut req = match method {
        "POST" => client.post(&url),
        "PUT" => client.put(&url),
        "DELETE" => client.delete(&url),
        "PATCH" => client.patch(&url),
        _ => client.get(&url),
    };

    // Forward relevant headers.
    for line in headers.lines() {
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim();
            let val = val.trim();
            // Skip hop-by-hop headers.
            let skip = [
                "host",
                "connection",
                "proxy-connection",
                "keep-alive",
                "transfer-encoding",
                "content-length",
            ];
            if !skip.iter().any(|s| key.eq_ignore_ascii_case(s)) {
                req = req.header(key, val);
            }
        }
    }

    req = req.header("Host", hostname);

    if !body.is_empty() {
        req = req.body(body.to_vec());
    }

    let resp = req.send().await.context("Forward to upstream failed")?;
    let status = resp.status();
    let resp_headers = resp.headers().clone();
    let resp_body = resp.bytes().await.context("Failed to read upstream body")?;

    // Reconstruct raw HTTP response for the client.
    let mut buf = BytesMut::new();
    buf.extend_from_slice(
        format!(
            "HTTP/1.1 {} {}\r\n",
            status.as_u16(),
            status.canonical_reason().unwrap_or("OK")
        )
        .as_bytes(),
    );

    for (key, val) in resp_headers.iter() {
        let skip = ["transfer-encoding", "connection"];
        if !skip.iter().any(|s| key.as_str().eq_ignore_ascii_case(s)) {
            buf.extend_from_slice(
                format!("{}: {}\r\n", key, val.to_str().unwrap_or("")).as_bytes(),
            );
        }
    }

    buf.extend_from_slice(format!("Content-Length: {}\r\n", resp_body.len()).as_bytes());
    buf.extend_from_slice(b"\r\n");
    buf.extend_from_slice(&resp_body);

    Ok(buf.to_vec())
}

fn format_http_response(status_code: u16, body: &str) -> String {
    let reason = match status_code {
        200 => "OK",
        503 => "Service Unavailable",
        _ => "Error",
    };
    format!(
        "HTTP/1.1 {status_code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn parse_host_port(s: &str) -> Result<(String, u16)> {
    if let Some((host, port_str)) = s.rsplit_once(':') {
        let port: u16 = port_str.parse().unwrap_or(443);
        Ok((host.to_string(), port))
    } else {
        Ok((s.to_string(), 443))
    }
}

async fn send_error(stream: &mut TcpStream, code: u16, msg: &str) -> Result<()> {
    let body = format!(r#"{{"error":"{msg}"}}"#);
    let response = format!(
        "HTTP/1.1 {code} {msg}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_host_port() {
        let (host, port) = parse_host_port("example.com:443").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(port, 443);

        let (host, port) = parse_host_port("example.com:8443").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(port, 8443);

        let (host, port) = parse_host_port("example.com").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(port, 443);
    }

    #[test]
    fn test_format_http_response() {
        let resp = format_http_response(200, r#"{"ok":true}"#);
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(resp.contains("Content-Length: 11"));
        assert!(resp.ends_with(r#"{"ok":true}"#));
    }

    #[test]
    fn test_allowed_domains() {
        assert!(ALLOWED_DOMAINS
            .iter()
            .any(|d| *d == "copilot-proxy.githubusercontent.com"));
        assert!(ALLOWED_DOMAINS.iter().any(|d| *d == "api.github.com"));
        assert!(!ALLOWED_DOMAINS.iter().any(|d| *d == "evil.example.com"));
    }

    #[test]
    fn test_intercepted_paths() {
        assert!(INTERCEPTED_PATHS
            .iter()
            .any(|p| *p == "/v1/chat/completions"));
        assert!(!INTERCEPTED_PATHS.iter().any(|p| *p == "/v1/models"));
    }
}
