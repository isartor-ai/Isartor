//! HTTP CONNECT proxy with TLS MITM for intercepting supported AI client traffic.
//!
//! Runs a raw TCP listener (separate from the Axum gateway) that speaks
//! the HTTP CONNECT protocol. Allowed domains get TLS-terminated with a
//! leaf certificate signed by the local Isartor CA, then their request
//! bodies are routed through the Deflection Stack (L1a exact cache →
//! L1b semantic cache → L2 local model → L3 native upstream passthrough).
//!
//! Non-allowed domains and non-supported paths are tunnelled
//! transparently without interception.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::{Json, extract::Query, response::IntoResponse};
use bytes::BytesMut;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;

use crate::config::CacheMode;
use crate::core::prompt::{extract_cache_key, extract_prompt, has_tooling};
use crate::metrics;
use crate::middleware::slm_triage::answer_quality_ok;
use crate::models::{
    FinalLayer, OpenAiChatChoice, OpenAiChatResponse, OpenAiMessage, OpenAiMessageContent,
    PromptVisibilityEntry, ProxyRecentResponse, ProxyRouteDecision,
};
use crate::proxy::tls::IsartorCa;
use crate::state::AppState;
use crate::visibility;

/// Domains that may be intercepted via TLS MITM.
const COPILOT_DOMAINS: &[&str] = &[
    "copilot-proxy.githubusercontent.com",
    "api.github.com",
    "api.individual.githubcopilot.com",
    "api.business.githubcopilot.com",
    "api.enterprise.githubcopilot.com",
];

const CLAUDE_DOMAINS: &[&str] = &["api.anthropic.com"];

const ANTIGRAVITY_DOMAINS: &[&str] = &[
    "cloudcode-pa.googleapis.com",
    "daily-cloudcode-pa.googleapis.com",
    "daily-cloudcode-pa.sandbox.googleapis.com",
];

const ALLOWED_DOMAINS: &[&str] = &[
    "copilot-proxy.githubusercontent.com",
    "api.github.com",
    "api.individual.githubcopilot.com",
    "api.business.githubcopilot.com",
    "api.enterprise.githubcopilot.com",
    "api.anthropic.com",
    "cloudcode-pa.googleapis.com",
    "daily-cloudcode-pa.googleapis.com",
    "daily-cloudcode-pa.sandbox.googleapis.com",
];

/// HTTP paths that trigger Deflection Stack interception.
const INTERCEPTED_PATHS: &[&str] = &["/v1/chat/completions", "/v1/messages"];
const RECENT_PROXY_DECISIONS_CAPACITY: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProxyClient {
    Copilot,
    Claude,
    Antigravity,
}

impl ProxyClient {
    fn id(self) -> &'static str {
        match self {
            Self::Copilot => "copilot",
            Self::Claude => "claude",
            Self::Antigravity => "antigravity",
        }
    }

    fn layer3_label(self) -> &'static str {
        match self {
            Self::Copilot => "copilot_upstream",
            Self::Claude => "claude_upstream",
            Self::Antigravity => "antigravity_upstream",
        }
    }

    fn endpoint_family(self, path: &str) -> &'static str {
        match path {
            "/v1/messages" => "anthropic",
            _ => "openai",
        }
    }
}

fn identify_proxy_client(hostname: &str) -> Option<ProxyClient> {
    if !ALLOWED_DOMAINS.contains(&hostname) {
        None
    } else if COPILOT_DOMAINS.contains(&hostname) {
        Some(ProxyClient::Copilot)
    } else if CLAUDE_DOMAINS.contains(&hostname) {
        Some(ProxyClient::Claude)
    } else if ANTIGRAVITY_DOMAINS.contains(&hostname) {
        Some(ProxyClient::Antigravity)
    } else {
        None
    }
}

static RECENT_PROXY_DECISIONS: std::sync::OnceLock<Mutex<VecDeque<ProxyRouteDecision>>> =
    std::sync::OnceLock::new();

fn recent_proxy_decisions_store() -> &'static Mutex<VecDeque<ProxyRouteDecision>> {
    RECENT_PROXY_DECISIONS
        .get_or_init(|| Mutex::new(VecDeque::with_capacity(RECENT_PROXY_DECISIONS_CAPACITY)))
}

fn record_proxy_decision(decision: ProxyRouteDecision) {
    let mut decisions = recent_proxy_decisions_store().lock();
    decisions.push_front(decision);
    while decisions.len() > RECENT_PROXY_DECISIONS_CAPACITY {
        decisions.pop_back();
    }
}

pub fn recent_proxy_decisions(limit: usize) -> Vec<ProxyRouteDecision> {
    recent_proxy_decisions_store()
        .lock()
        .iter()
        .take(limit.min(RECENT_PROXY_DECISIONS_CAPACITY))
        .cloned()
        .collect()
}

pub fn recent_proxy_decisions_count() -> usize {
    recent_proxy_decisions_store().lock().len()
}

#[cfg(test)]
fn clear_recent_proxy_decisions() {
    recent_proxy_decisions_store().lock().clear();
}

#[derive(Debug, serde::Deserialize)]
pub struct RecentProxyQuery {
    pub limit: Option<usize>,
}

pub async fn recent_proxy_requests_handler(
    Query(query): Query<RecentProxyQuery>,
) -> impl IntoResponse {
    Json(ProxyRecentResponse {
        entries: recent_proxy_decisions(query.limit.unwrap_or(20)),
    })
}

enum ProxyInterceptResolution {
    Local {
        final_layer: FinalLayer,
        resolved_by: &'static str,
        response_body: String,
    },
    ForwardToUpstream {
        final_layer: FinalLayer,
        resolved_by: &'static str,
    },
    Blocked {
        final_layer: FinalLayer,
        resolved_by: &'static str,
        status_code: u16,
        response_body: String,
    },
}

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

    let parts: Vec<&str> = request_line.split_whitespace().collect();
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
    let should_intercept = identify_proxy_client(&hostname).is_some();

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
        let start = Instant::now();
        let prompt = extract_prompt(&body);
        let estimated_tokens = if prompt.is_empty() {
            None
        } else {
            Some(metrics::estimate_tokens(&prompt))
        };
        let prompt_hash = visibility::prompt_hash_from_body(&body);

        let proxy_client = identify_proxy_client(hostname);
        match resolve_intercepted_request(&body, &path, proxy_client, &state).await {
            ProxyInterceptResolution::Local {
                final_layer,
                resolved_by,
                response_body,
            } => {
                let resp = format_http_response(200, &response_body);
                tls_stream.write_all(resp.as_bytes()).await?;
                tls_stream.flush().await?;
                emit_proxy_decision(ProxyDecisionContext {
                    proxy_client,
                    hostname,
                    path: &path,
                    prompt_hash,
                    final_layer,
                    resolved_by,
                    status_code: 200,
                    estimated_tokens,
                    latency_ms: start.elapsed().as_millis() as u64,
                });
                return Ok(());
            }
            ProxyInterceptResolution::Blocked {
                final_layer,
                resolved_by,
                status_code,
                response_body,
            } => {
                let resp = format_http_response(status_code, &response_body);
                tls_stream.write_all(resp.as_bytes()).await?;
                tls_stream.flush().await?;
                emit_proxy_decision(ProxyDecisionContext {
                    proxy_client,
                    hostname,
                    path: &path,
                    prompt_hash,
                    final_layer,
                    resolved_by,
                    status_code,
                    estimated_tokens,
                    latency_ms: start.elapsed().as_millis() as u64,
                });
                return Ok(());
            }
            ProxyInterceptResolution::ForwardToUpstream {
                final_layer,
                resolved_by,
            } => {
                let upstream_response =
                    forward_to_upstream(hostname, port, &method, &path, &headers, &body).await?;
                cache_response(&body, &path, &upstream_response, &state).await;
                emit_proxy_decision(ProxyDecisionContext {
                    proxy_client,
                    hostname,
                    path: &path,
                    prompt_hash,
                    final_layer,
                    resolved_by,
                    status_code: parse_status_code(&upstream_response),
                    estimated_tokens,
                    latency_ms: start.elapsed().as_millis() as u64,
                });
                tls_stream.write_all(&upstream_response).await?;
                tls_stream.flush().await?;
                return Ok(());
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

/// Run the prompt through the proxy deflection stack (L1a → L1b → L2 → Copilot upstream).
async fn resolve_intercepted_request(
    body: &[u8],
    path: &str,
    proxy_client: Option<ProxyClient>,
    state: &Arc<AppState>,
) -> ProxyInterceptResolution {
    let tool = proxy_client.map(ProxyClient::id).unwrap_or("unknown");
    let prompt = extract_prompt(body);
    let cache_key_material = extract_cache_key(body);

    if prompt.is_empty() {
        return ProxyInterceptResolution::ForwardToUpstream {
            final_layer: FinalLayer::Cloud,
            resolved_by: proxy_client
                .map(ProxyClient::layer3_label)
                .unwrap_or("native_upstream"),
        };
    }

    let cache_ns = match path {
        "/v1/chat/completions" => "openai",
        "/v1/messages" => "anthropic",
        _ => "proxy",
    };
    let cache_prompt = format!("{cache_ns}|{cache_key_material}");
    let semantic_cache_enabled = path != "/v1/messages" && !has_tooling(body);
    let mode = &state.config.cache_mode;

    let exact_key = if *mode == CacheMode::Exact || *mode == CacheMode::Both {
        let key = hex::encode(Sha256::digest(cache_prompt.as_bytes()));
        if let Some(cached) = state.exact_cache.get(&key) {
            tracing::info!(cache.key = %key, "Proxy L1a: exact cache HIT");
            metrics::record_cache_event_with_tool("l1a", "hit", tool);
            metrics::record_cache_event_with_tool("l1", "hit", tool);
            visibility::record_agent_cache_event(tool, "l1a", "hit");
            visibility::record_agent_cache_event(tool, "l1", "hit");
            return ProxyInterceptResolution::Local {
                final_layer: FinalLayer::ExactCache,
                resolved_by: "exact_cache",
                response_body: cached,
            };
        }
        metrics::record_cache_event_with_tool("l1a", "miss", tool);
        visibility::record_agent_cache_event(tool, "l1a", "miss");
        Some(key)
    } else {
        None
    };

    let embedding: Option<Vec<f32>> =
        if semantic_cache_enabled && (*mode == CacheMode::Semantic || *mode == CacheMode::Both) {
            match state.text_embedder.generate_embedding(&prompt) {
                Ok(emb) => {
                    if let Some(cached) = state.vector_cache.search(&emb, None).await {
                        tracing::info!("Proxy L1b: semantic cache HIT");
                        metrics::record_cache_event_with_tool("l1b", "hit", tool);
                        metrics::record_cache_event_with_tool("l1", "hit", tool);
                        visibility::record_agent_cache_event(tool, "l1b", "hit");
                        visibility::record_agent_cache_event(tool, "l1", "hit");
                        return ProxyInterceptResolution::Local {
                            final_layer: FinalLayer::SemanticCache,
                            resolved_by: "semantic_cache",
                            response_body: cached,
                        };
                    }
                    metrics::record_cache_event_with_tool("l1b", "miss", tool);
                    visibility::record_agent_cache_event(tool, "l1b", "miss");
                    Some(emb)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Proxy L1b: embedding generation failed, skipping");
                    metrics::record_error_with_tool("L1b_Embedding", "retryable", tool);
                    visibility::record_agent_error(tool);
                    None
                }
            }
        } else {
            None
        };

    if let Some(response_body) = try_proxy_slm_resolution(&prompt, path, state).await {
        if let Some(key) = exact_key {
            state.exact_cache.put(key, response_body.clone());
        }
        if let Some(emb) = embedding {
            state
                .vector_cache
                .insert(emb, response_body.clone(), None)
                .await;
        }

        return ProxyInterceptResolution::Local {
            final_layer: FinalLayer::Slm,
            resolved_by: "slm",
            response_body,
        };
    }

    metrics::record_cache_event_with_tool("l1", "miss", tool);
    visibility::record_agent_cache_event(tool, "l1", "miss");

    if state.config.offline_mode {
        tracing::debug!("Proxy: offline mode, blocking native upstream");
        return ProxyInterceptResolution::Blocked {
            final_layer: FinalLayer::Cloud,
            resolved_by: "offline_blocked",
            status_code: 503,
            response_body: match path {
                "/v1/messages" => serde_json::json!({
                    "error": { "message": "offline mode active" }
                })
                .to_string(),
                _ => serde_json::json!({
                    "error": { "message": "offline mode active" }
                })
                .to_string(),
            },
        };
    }

    ProxyInterceptResolution::ForwardToUpstream {
        final_layer: FinalLayer::Cloud,
        resolved_by: proxy_client
            .map(ProxyClient::layer3_label)
            .unwrap_or("native_upstream"),
    }
}

fn build_openai_proxy_response(content: String, model: String) -> String {
    serde_json::to_string(&OpenAiChatResponse {
        choices: vec![OpenAiChatChoice {
            message: OpenAiMessage {
                role: "assistant".to_string(),
                content: Some(OpenAiMessageContent::text(content)),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                function_call: None,
            },
            index: 0,
            finish_reason: Some("stop".to_string()),
        }],
        model: Some(model),
    })
    .unwrap_or_else(|_| {
        serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "index": 0,
                "finish_reason": "stop"
            }]
        })
        .to_string()
    })
}

fn build_anthropic_proxy_response(content: String, model: String) -> String {
    serde_json::json!({
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn"
    })
    .to_string()
}

async fn try_proxy_slm_resolution(
    prompt: &str,
    path: &str,
    state: &Arc<AppState>,
) -> Option<String> {
    if !state.config.enable_slm_router {
        return None;
    }

    if state.config.inference_engine == crate::config::InferenceEngineMode::Embedded {
        #[cfg(feature = "embedded-inference")]
        {
            if let Some(classifier) = &state.embedded_classifier {
                let use_tiered =
                    state.config.layer2.classifier_mode == crate::config::ClassifierMode::Tiered;
                match classifier.classify(prompt).await {
                    Ok((label, _))
                        if label == "SIMPLE"
                            || (use_tiered && (label == "TEMPLATE" || label == "SNIPPET")) =>
                    {
                        match classifier.execute(prompt).await {
                            Ok(answer) if answer_quality_ok(&answer) => {
                                let model = "embedded(gemma-2)".to_string();
                                return Some(match path {
                                    "/v1/messages" => build_anthropic_proxy_response(answer, model),
                                    _ => build_openai_proxy_response(answer, model),
                                });
                            }
                            Ok(_) => {
                                tracing::info!("Proxy L2: answer quality guard rejected output");
                                return None;
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Proxy L2: embedded answer generation failed");
                                return None;
                            }
                        }
                    }
                    Ok(_) => return None,
                    Err(e) => {
                        tracing::warn!(error = %e, "Proxy L2: embedded classification failed");
                        return None;
                    }
                }
            }
            return None;
        }
        #[cfg(not(feature = "embedded-inference"))]
        {
            tracing::warn!("Proxy L2: embedded inference requested but feature disabled");
            return None;
        }
    }

    match state
        .slm_client
        .classify_deflectable(prompt, &state.config.layer2.classifier_mode)
        .await
    {
        Ok(true) => match state.slm_client.answer_prompt(prompt).await {
            Ok(answer) if answer_quality_ok(&answer) => {
                let model = state.config.layer2.model_name.clone();
                Some(match path {
                    "/v1/messages" => build_anthropic_proxy_response(answer, model),
                    _ => build_openai_proxy_response(answer, model),
                })
            }
            Ok(_) => {
                tracing::info!("Proxy L2: sidecar answer quality guard rejected output");
                None
            }
            Err(e) => {
                tracing::warn!(error = %e, "Proxy L2: sidecar answer generation failed");
                None
            }
        },
        Ok(false) => None,
        Err(e) => {
            tracing::warn!(error = %e, "Proxy L2: sidecar classification failed");
            None
        }
    }
}

struct ProxyDecisionContext<'a> {
    proxy_client: Option<ProxyClient>,
    hostname: &'a str,
    path: &'a str,
    prompt_hash: Option<String>,
    final_layer: FinalLayer,
    resolved_by: &'a str,
    status_code: u16,
    estimated_tokens: Option<u64>,
    latency_ms: u64,
}

fn emit_proxy_decision(context: ProxyDecisionContext<'_>) {
    let ProxyDecisionContext {
        proxy_client,
        hostname,
        path,
        prompt_hash,
        final_layer,
        resolved_by,
        status_code,
        estimated_tokens,
        latency_ms,
    } = context;
    let client = proxy_client
        .map(ProxyClient::id)
        .unwrap_or("unknown")
        .to_string();
    let endpoint_family = proxy_client
        .map(|proxy_client| proxy_client.endpoint_family(path))
        .unwrap_or("proxy")
        .to_string();
    let decision = ProxyRouteDecision {
        request_id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        client: client.clone(),
        hostname: hostname.to_string(),
        path: path.to_string(),
        prompt_hash,
        final_layer: final_layer.as_header_value().to_string(),
        resolved_by: resolved_by.to_string(),
        deflected: final_layer.is_deflected(),
        latency_ms,
    };

    metrics::record_request_with_tool(
        final_layer.as_str(),
        status_code,
        latency_ms as f64 / 1000.0,
        "proxy",
        &client,
        &endpoint_family,
        &client,
    );
    if final_layer.is_deflected() {
        metrics::record_tokens_saved_with_tool(
            final_layer.as_str(),
            estimated_tokens.unwrap_or(256),
            "proxy",
            &client,
            &endpoint_family,
            &client,
        );
    }

    tracing::info!(
        client = %decision.client,
        hostname = %decision.hostname,
        path = %decision.path,
        final_layer = %decision.final_layer,
        resolved_by = %decision.resolved_by,
        deflected = decision.deflected,
        latency_ms = decision.latency_ms,
        "Proxy request resolved"
    );

    visibility::record_prompt(PromptVisibilityEntry {
        timestamp: decision.timestamp.clone(),
        traffic_surface: "proxy".to_string(),
        client: decision.client.clone(),
        endpoint_family,
        route: format!("{hostname} {path}"),
        prompt_hash: decision.prompt_hash.clone(),
        final_layer: decision.final_layer.clone(),
        resolved_by: Some(decision.resolved_by.clone()),
        deflected: decision.deflected,
        latency_ms: decision.latency_ms,
        status_code,
        tool: decision.client.clone(),
    });
    record_proxy_decision(decision);
}

fn parse_status_code(response: &[u8]) -> u16 {
    let prefix = String::from_utf8_lossy(response);
    prefix
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .unwrap_or(200)
}

/// Cache a response received from the real upstream.
async fn cache_response(body: &[u8], path: &str, response: &[u8], state: &Arc<AppState>) {
    let prompt = extract_prompt(body);
    let cache_key_material = extract_cache_key(body);
    if prompt.is_empty() {
        return;
    }

    let cache_ns = match path {
        "/v1/chat/completions" => "openai",
        "/v1/messages" => "anthropic",
        _ => "proxy",
    };
    let cache_prompt = format!("{cache_ns}|{cache_key_material}");
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

    if (*mode == CacheMode::Semantic || *mode == CacheMode::Both)
        && let Ok(emb) = state.text_embedder.generate_embedding(&prompt)
    {
        state.vector_cache.insert(emb, cache_value, None).await;
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
    let parts: Vec<&str> = request_line.split_whitespace().collect();
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
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::clients::slm::SlmClient;
    use crate::config::{
        AppConfig, CacheBackend, CacheMode, EmbeddingSidecarSettings, InferenceEngineMode,
        Layer2Settings, RouterBackend,
    };
    use crate::core::context_compress::InstructionCache;
    use crate::layer1::embeddings::shared_test_embedder;
    use crate::layer1::layer1a_cache::ExactMatchCache;
    use crate::state::{AppLlmAgent, AppState};
    use crate::vector_cache::VectorCache;

    struct PanicAgent;

    #[async_trait::async_trait]
    impl AppLlmAgent for PanicAgent {
        async fn chat(&self, _prompt: &str) -> anyhow::Result<String> {
            panic!("proxy resolution should not call generic llm_agent");
        }
        fn provider_name(&self) -> &'static str {
            "panic"
        }
    }

    fn test_config(
        sidecar_url: &str,
        enable_slm_router: bool,
        offline_mode: bool,
    ) -> Arc<AppConfig> {
        Arc::new(AppConfig {
            host_port: "127.0.0.1:0".into(),
            inference_engine: InferenceEngineMode::Sidecar,
            gateway_api_key: "test-key".into(),
            cache_mode: CacheMode::Both,
            cache_backend: CacheBackend::Memory,
            redis_url: "redis://127.0.0.1:6379".into(),
            router_backend: RouterBackend::Embedded,
            vllm_url: "http://127.0.0.1:8000".into(),
            vllm_model: "gemma-2-2b-it".into(),
            embedding_model: "all-minilm".into(),
            similarity_threshold: 0.85,
            cache_ttl_secs: 300,
            cache_max_capacity: 100,
            layer2: Layer2Settings {
                sidecar_url: sidecar_url.into(),
                model_name: "phi-3-mini".into(),
                timeout_seconds: 5,
                classifier_mode: crate::config::ClassifierMode::Tiered,
                max_answer_tokens: 2048,
            },
            local_slm_url: "http://localhost:11434/api/generate".into(),
            local_slm_model: "llama3".into(),
            embedding_sidecar: EmbeddingSidecarSettings {
                sidecar_url: "http://127.0.0.1:8082".into(),
                model_name: "test".into(),
                timeout_seconds: 5,
            },
            llm_provider: "openai".into(),
            external_llm_url: "https://api.openai.com/v1/chat/completions".into(),
            external_llm_model: "gpt-4o-mini".into(),
            model_aliases: std::collections::HashMap::new(),
            external_llm_api_key: "".into(),
            l3_timeout_secs: 120,
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring: false,
            enable_slm_router,
            otel_exporter_endpoint: "http://localhost:4317".into(),
            enable_request_logs: false,
            request_log_path: "~/.isartor/request_logs".into(),
            offline_mode,
            proxy_port: "0.0.0.0:8081".into(),
            enable_context_optimizer: true,
            context_optimizer_dedup: true,
            context_optimizer_minify: true,
        })
    }

    fn test_state(sidecar_url: &str, enable_slm_router: bool, offline_mode: bool) -> Arc<AppState> {
        let config = test_config(sidecar_url, enable_slm_router, offline_mode);
        Arc::new(AppState {
            config: config.clone(),
            http_client: reqwest::Client::new(),
            exact_cache: Arc::new(ExactMatchCache::new(NonZeroUsize::new(100).unwrap())),
            vector_cache: Arc::new(VectorCache::new(0.85, 300, 100)),
            llm_agent: Arc::new(PanicAgent),
            slm_client: Arc::new(SlmClient::new(&config.layer2)),
            text_embedder: shared_test_embedder(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        })
    }

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
        assert!(ALLOWED_DOMAINS.contains(&"copilot-proxy.githubusercontent.com"));
        assert!(ALLOWED_DOMAINS.contains(&"api.github.com"));
        assert!(ALLOWED_DOMAINS.contains(&"api.anthropic.com"));
        assert!(ALLOWED_DOMAINS.contains(&"cloudcode-pa.googleapis.com"));
        assert!(!ALLOWED_DOMAINS.contains(&"evil.example.com"));
    }

    #[test]
    fn test_intercepted_paths() {
        assert!(INTERCEPTED_PATHS.contains(&"/v1/chat/completions"));
        assert!(INTERCEPTED_PATHS.contains(&"/v1/messages"));
        assert!(!INTERCEPTED_PATHS.contains(&"/v1/models"));
    }

    #[tokio::test]
    async fn proxy_resolution_uses_exact_cache_before_anything_else() {
        let state = test_state("http://127.0.0.1:1", false, false);
        let key = hex::encode(Sha256::digest(b"openai|hello"));
        state.exact_cache.put(key, r#"{"cached":true}"#.to_string());

        let result = resolve_intercepted_request(
            br#"{"prompt":"hello"}"#,
            "/v1/chat/completions",
            Some(ProxyClient::Copilot),
            &state,
        )
        .await;

        match result {
            ProxyInterceptResolution::Local {
                final_layer,
                resolved_by,
                response_body,
            } => {
                assert_eq!(final_layer, FinalLayer::ExactCache);
                assert_eq!(resolved_by, "exact_cache");
                assert_eq!(response_body, r#"{"cached":true}"#);
            }
            _ => panic!("expected exact-cache local resolution"),
        }
    }

    #[tokio::test]
    async fn proxy_resolution_can_resolve_at_l2_via_sidecar() {
        let server = MockServer::start().await;

        // Tiered classifier prompt matcher
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_string_contains("coding task classifier"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"choices":[{"message":{"role":"assistant","content":"TEMPLATE"},"index":0,"finish_reason":"stop"}]})),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_string_contains("what is 2+2"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"choices":[{"message":{"role":"assistant","content":"local answer"},"index":0,"finish_reason":"stop"}], "model":"phi-3-mini"})),
            )
            .expect(1)
            .mount(&server)
            .await;

        let state = test_state(&server.uri(), true, false);
        let result = resolve_intercepted_request(
            br#"{"prompt":"what is 2+2"}"#,
            "/v1/chat/completions",
            Some(ProxyClient::Copilot),
            &state,
        )
        .await;

        match result {
            ProxyInterceptResolution::Local {
                final_layer,
                resolved_by,
                response_body,
            } => {
                assert_eq!(final_layer, FinalLayer::Slm);
                assert_eq!(resolved_by, "slm");
                assert!(response_body.contains("local answer"));
            }
            _ => panic!("expected l2 local resolution"),
        }
    }

    #[tokio::test]
    async fn proxy_resolution_falls_back_to_copilot_upstream_without_llm_key() {
        let state = test_state("http://127.0.0.1:1", false, false);
        let result = resolve_intercepted_request(
            br#"{"prompt":"cache miss"}"#,
            "/v1/chat/completions",
            Some(ProxyClient::Copilot),
            &state,
        )
        .await;

        match result {
            ProxyInterceptResolution::ForwardToUpstream {
                final_layer,
                resolved_by,
            } => {
                assert_eq!(final_layer, FinalLayer::Cloud);
                assert_eq!(resolved_by, "copilot_upstream");
            }
            _ => panic!("expected copilot upstream passthrough"),
        }
    }

    #[tokio::test]
    async fn proxy_resolution_returns_anthropic_l2_shape_for_claude() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_string_contains("coding task classifier"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"choices":[{"message":{"role":"assistant","content":"SNIPPET"},"index":0,"finish_reason":"stop"}]})),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_string_contains("explain the patch"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"choices":[{"message":{"role":"assistant","content":"anthropic local answer"},"index":0,"finish_reason":"stop"}], "model":"phi-3-mini"})),
            )
            .expect(1)
            .mount(&server)
            .await;

        let state = test_state(&server.uri(), true, false);
        let result = resolve_intercepted_request(
            br#"{"messages":[{"role":"user","content":[{"type":"text","text":"explain the patch"}]}]}"#,
            "/v1/messages",
            Some(ProxyClient::Claude),
            &state,
        )
        .await;

        match result {
            ProxyInterceptResolution::Local {
                final_layer,
                resolved_by,
                response_body,
            } => {
                assert_eq!(final_layer, FinalLayer::Slm);
                assert_eq!(resolved_by, "slm");
                assert!(response_body.contains(r#""type":"message""#));
                assert!(response_body.contains("anthropic local answer"));
            }
            _ => panic!("expected anthropic l2 local resolution"),
        }
    }

    #[tokio::test]
    async fn proxy_resolution_falls_back_to_claude_upstream_without_llm_key() {
        let state = test_state("http://127.0.0.1:1", false, false);
        let result = resolve_intercepted_request(
            br#"{"messages":[{"role":"user","content":[{"type":"text","text":"cache miss"}]}]}"#,
            "/v1/messages",
            Some(ProxyClient::Claude),
            &state,
        )
        .await;

        match result {
            ProxyInterceptResolution::ForwardToUpstream {
                final_layer,
                resolved_by,
            } => {
                assert_eq!(final_layer, FinalLayer::Cloud);
                assert_eq!(resolved_by, "claude_upstream");
            }
            _ => panic!("expected claude upstream passthrough"),
        }
    }

    #[tokio::test]
    async fn proxy_resolution_falls_back_to_antigravity_upstream_without_llm_key() {
        let state = test_state("http://127.0.0.1:1", false, false);
        let result = resolve_intercepted_request(
            br#"{"messages":[{"role":"user","content":"cache miss"}]}"#,
            "/v1/chat/completions",
            Some(ProxyClient::Antigravity),
            &state,
        )
        .await;

        match result {
            ProxyInterceptResolution::ForwardToUpstream {
                final_layer,
                resolved_by,
            } => {
                assert_eq!(final_layer, FinalLayer::Cloud);
                assert_eq!(resolved_by, "antigravity_upstream");
            }
            _ => panic!("expected antigravity upstream passthrough"),
        }
    }

    #[test]
    fn recent_proxy_decisions_are_recorded() {
        clear_recent_proxy_decisions();
        emit_proxy_decision(ProxyDecisionContext {
            proxy_client: Some(ProxyClient::Copilot),
            hostname: "copilot-proxy.githubusercontent.com",
            path: "/v1/chat/completions",
            prompt_hash: Some("abc123".into()),
            final_layer: FinalLayer::SemanticCache,
            resolved_by: "semantic_cache",
            status_code: 200,
            estimated_tokens: Some(256),
            latency_ms: 12,
        });

        let entries = recent_proxy_decisions(1);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].client, "copilot");
        assert_eq!(entries[0].final_layer, "l1b");
        assert_eq!(entries[0].resolved_by, "semantic_cache");
        assert!(entries[0].deflected);
    }
}
