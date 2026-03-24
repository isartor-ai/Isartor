use std::collections::{BTreeMap, VecDeque};

use axum::{Json, extract::Query, response::IntoResponse};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use crate::core::prompt::extract_prompt;
use crate::models::{
    AgentStatsEntry, AgentStatsResponse, PromptStatsResponse, PromptVisibilityEntry,
};

const RECENT_PROMPT_ENTRIES_CAPACITY: usize = 200;

static PROMPT_STATS: std::sync::OnceLock<Mutex<PromptVisibilityState>> = std::sync::OnceLock::new();

#[derive(Debug, Default)]
struct PromptVisibilityState {
    total_prompts: u64,
    total_deflected_prompts: u64,
    by_layer: BTreeMap<String, u64>,
    by_surface: BTreeMap<String, u64>,
    by_client: BTreeMap<String, u64>,
    by_tool: BTreeMap<String, u64>,
    by_agent: BTreeMap<String, AgentVisibilityState>,
    recent: VecDeque<PromptVisibilityEntry>,
}

#[derive(Debug, Default)]
struct AgentVisibilityState {
    requests: u64,
    cache_hits: u64,
    cache_misses: u64,
    l1a_hits: u64,
    l1a_misses: u64,
    l1b_hits: u64,
    l1b_misses: u64,
    total_latency_ms: u64,
    retry_count: u64,
    error_count: u64,
}

impl PromptVisibilityState {
    fn record(&mut self, entry: PromptVisibilityEntry) {
        self.total_prompts += 1;
        if entry.deflected {
            self.total_deflected_prompts += 1;
        }
        *self.by_layer.entry(entry.final_layer.clone()).or_insert(0) += 1;
        *self
            .by_surface
            .entry(entry.traffic_surface.clone())
            .or_insert(0) += 1;
        *self.by_client.entry(entry.client.clone()).or_insert(0) += 1;
        if !entry.tool.is_empty() {
            *self.by_tool.entry(entry.tool.clone()).or_insert(0) += 1;
            let agent = self.by_agent.entry(entry.tool.clone()).or_default();
            agent.requests += 1;
            agent.total_latency_ms += entry.latency_ms;
        }

        self.recent.push_front(entry);
        while self.recent.len() > RECENT_PROMPT_ENTRIES_CAPACITY {
            self.recent.pop_back();
        }
    }

    fn snapshot(&self, limit: usize) -> PromptStatsResponse {
        PromptStatsResponse {
            total_prompts: self.total_prompts,
            total_deflected_prompts: self.total_deflected_prompts,
            by_layer: self.by_layer.clone(),
            by_surface: self.by_surface.clone(),
            by_client: self.by_client.clone(),
            by_tool: self.by_tool.clone(),
            recent: self
                .recent
                .iter()
                .take(limit.min(RECENT_PROMPT_ENTRIES_CAPACITY))
                .cloned()
                .collect(),
        }
    }

    fn agent_snapshot(&self) -> AgentStatsResponse {
        AgentStatsResponse {
            agents: self
                .by_agent
                .iter()
                .map(|(tool, stats)| {
                    (
                        tool.clone(),
                        AgentStatsEntry {
                            requests: stats.requests,
                            cache_hits: stats.cache_hits,
                            cache_misses: stats.cache_misses,
                            l1a_hits: stats.l1a_hits,
                            l1a_misses: stats.l1a_misses,
                            l1b_hits: stats.l1b_hits,
                            l1b_misses: stats.l1b_misses,
                            average_latency_ms: if stats.requests == 0 {
                                0.0
                            } else {
                                stats.total_latency_ms as f64 / stats.requests as f64
                            },
                            retry_count: stats.retry_count,
                            error_count: stats.error_count,
                        },
                    )
                })
                .collect(),
        }
    }

    fn record_agent_cache_event(&mut self, tool: &str, cache_layer: &str, outcome: &str) {
        if tool.is_empty() {
            return;
        }
        let agent = self.by_agent.entry(tool.to_string()).or_default();
        match (cache_layer, outcome) {
            ("l1", "hit") => agent.cache_hits += 1,
            ("l1", "miss") => agent.cache_misses += 1,
            ("l1a", "hit") => agent.l1a_hits += 1,
            ("l1a", "miss") => agent.l1a_misses += 1,
            ("l1b", "hit") => agent.l1b_hits += 1,
            ("l1b", "miss") => agent.l1b_misses += 1,
            _ => {}
        }
    }

    fn record_agent_retry(&mut self, tool: &str, attempts: u32) {
        if tool.is_empty() {
            return;
        }
        let agent = self.by_agent.entry(tool.to_string()).or_default();
        agent.retry_count += u64::from(attempts.saturating_sub(1));
    }

    fn record_agent_error(&mut self, tool: &str) {
        if tool.is_empty() {
            return;
        }
        let agent = self.by_agent.entry(tool.to_string()).or_default();
        agent.error_count += 1;
    }
}

fn prompt_stats_store() -> &'static Mutex<PromptVisibilityState> {
    PROMPT_STATS.get_or_init(|| Mutex::new(PromptVisibilityState::default()))
}

pub fn record_prompt(entry: PromptVisibilityEntry) {
    prompt_stats_store().lock().record(entry);
}

pub fn prompt_stats_snapshot(limit: usize) -> PromptStatsResponse {
    prompt_stats_store().lock().snapshot(limit)
}

pub fn agent_stats_snapshot() -> AgentStatsResponse {
    prompt_stats_store().lock().agent_snapshot()
}

pub fn record_agent_cache_event(tool: &str, cache_layer: &str, outcome: &str) {
    prompt_stats_store()
        .lock()
        .record_agent_cache_event(tool, cache_layer, outcome);
}

pub fn record_agent_retry(tool: &str, attempts: u32) {
    prompt_stats_store()
        .lock()
        .record_agent_retry(tool, attempts);
}

pub fn record_agent_error(tool: &str) {
    prompt_stats_store().lock().record_agent_error(tool);
}

#[cfg(test)]
pub fn clear_prompt_stats() {
    *prompt_stats_store().lock() = PromptVisibilityState::default();
}

pub fn prompt_total_requests() -> u64 {
    prompt_stats_store().lock().total_prompts
}

pub fn prompt_total_deflected_requests() -> u64 {
    prompt_stats_store().lock().total_deflected_prompts
}

pub fn prompt_hash_from_body(body: &[u8]) -> Option<String> {
    let prompt = extract_prompt(body);
    if prompt.is_empty() {
        return None;
    }
    Some(hex::encode(Sha256::digest(prompt.as_bytes())))
}

#[derive(Debug, serde::Deserialize)]
pub struct PromptStatsQuery {
    pub limit: Option<usize>,
}

pub async fn prompt_stats_handler(Query(query): Query<PromptStatsQuery>) -> impl IntoResponse {
    Json(prompt_stats_snapshot(query.limit.unwrap_or(20)))
}

pub async fn agent_stats_handler() -> impl IntoResponse {
    Json(agent_stats_snapshot())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, routing::get};
    use tower::ServiceExt;

    #[test]
    fn local_state_tracks_counts_and_recent_entries() {
        let mut state = PromptVisibilityState::default();
        state.record(PromptVisibilityEntry {
            timestamp: "2026-01-01T00:00:00Z".into(),
            traffic_surface: "gateway".into(),
            client: "direct".into(),
            endpoint_family: "native".into(),
            route: "/api/chat".into(),
            prompt_hash: Some("abc".into()),
            final_layer: "l2".into(),
            resolved_by: None,
            deflected: true,
            latency_ms: 12,
            status_code: 200,
            tool: "curl".into(),
        });
        state.record(PromptVisibilityEntry {
            timestamp: "2026-01-01T00:00:01Z".into(),
            traffic_surface: "proxy".into(),
            client: "copilot".into(),
            endpoint_family: "openai".into(),
            route: "copilot-proxy.githubusercontent.com /v1/chat/completions".into(),
            prompt_hash: Some("def".into()),
            final_layer: "l3".into(),
            resolved_by: Some("copilot_upstream".into()),
            deflected: false,
            latency_ms: 20,
            status_code: 200,
            tool: "copilot".into(),
        });

        let snapshot = state.snapshot(10);
        assert_eq!(snapshot.total_prompts, 2);
        assert_eq!(snapshot.total_deflected_prompts, 1);
        assert_eq!(snapshot.by_layer.get("l2"), Some(&1));
        assert_eq!(snapshot.by_layer.get("l3"), Some(&1));
        assert_eq!(snapshot.by_surface.get("gateway"), Some(&1));
        assert_eq!(snapshot.by_surface.get("proxy"), Some(&1));
        assert_eq!(snapshot.by_client.get("direct"), Some(&1));
        assert_eq!(snapshot.by_client.get("copilot"), Some(&1));
        assert_eq!(snapshot.by_tool.get("curl"), Some(&1));
        assert_eq!(snapshot.by_tool.get("copilot"), Some(&1));
        assert_eq!(snapshot.recent.len(), 2);
        assert_eq!(snapshot.recent[0].client, "copilot");

        let agent_snapshot = state.agent_snapshot();
        assert_eq!(
            agent_snapshot
                .agents
                .get("curl")
                .unwrap()
                .average_latency_ms,
            12.0
        );
        assert_eq!(
            agent_snapshot
                .agents
                .get("copilot")
                .unwrap()
                .average_latency_ms,
            20.0
        );
    }

    #[test]
    fn agent_state_tracks_cache_retries_and_errors() {
        let mut state = PromptVisibilityState::default();
        state.record(PromptVisibilityEntry {
            timestamp: "2026-01-01T00:00:00Z".into(),
            traffic_surface: "gateway".into(),
            client: "direct".into(),
            endpoint_family: "native".into(),
            route: "/api/chat".into(),
            prompt_hash: None,
            final_layer: "l3".into(),
            resolved_by: None,
            deflected: false,
            latency_ms: 25,
            status_code: 200,
            tool: "cursor".into(),
        });
        state.record_agent_cache_event("cursor", "l1a", "miss");
        state.record_agent_cache_event("cursor", "l1b", "hit");
        state.record_agent_cache_event("cursor", "l1", "hit");
        state.record_agent_retry("cursor", 3);
        state.record_agent_error("cursor");

        let snapshot = state.agent_snapshot();
        assert_eq!(
            snapshot.agents.get("cursor"),
            Some(&AgentStatsEntry {
                requests: 1,
                cache_hits: 1,
                cache_misses: 0,
                l1a_hits: 0,
                l1a_misses: 1,
                l1b_hits: 1,
                l1b_misses: 0,
                average_latency_ms: 25.0,
                retry_count: 2,
                error_count: 1,
            })
        );
    }

    #[tokio::test]
    async fn agent_stats_handler_returns_programmatic_json() {
        clear_prompt_stats();
        record_prompt(PromptVisibilityEntry {
            timestamp: "2026-01-01T00:00:00Z".into(),
            traffic_surface: "gateway".into(),
            client: "direct".into(),
            endpoint_family: "native".into(),
            route: "/api/chat".into(),
            prompt_hash: None,
            final_layer: "l1a".into(),
            resolved_by: None,
            deflected: true,
            latency_ms: 10,
            status_code: 200,
            tool: "copilot".into(),
        });
        record_agent_cache_event("copilot", "l1a", "hit");
        record_agent_cache_event("copilot", "l1", "hit");

        let app = Router::new().route("/debug/stats/agents", get(agent_stats_handler));
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/debug/stats/agents")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let payload: AgentStatsResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload.agents.get("copilot").unwrap().requests, 1);
        assert_eq!(payload.agents.get("copilot").unwrap().l1a_hits, 1);
    }

    #[test]
    fn prompt_hash_is_stable_for_supported_payloads() {
        let hash = prompt_hash_from_body(br#"{"prompt":"hello"}"#).unwrap();
        assert_eq!(hash.len(), 64);
    }
}
