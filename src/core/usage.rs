use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Json;
use axum::extract::{Query, Request};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use chrono::{DateTime, Duration, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::config::{AppConfig, ProviderPricingConfig};
use crate::core::prompt::{extract_request_model, extract_route_model};
use crate::metrics;
use crate::models::{UsageStatsEntry, UsageStatsResponse};
use crate::state::AppState;

const USAGE_LOG_FILE_NAME: &str = "usage.jsonl";
const DEFAULT_USAGE_RETENTION_DAYS: u64 = 30;
const DEFAULT_USAGE_WINDOW_HOURS: u64 = 24;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageEvent {
    pub timestamp: String,
    pub provider: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub estimated_cost_usd: f64,
    pub final_layer: String,
    pub deflected: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct UsageQuery {
    pub hours: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProviderUsageTotals {
    pub requests_total: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub estimated_cost_usd: f64,
}

/// Aggregated usage broken down by provider + model — used by the dashboard breakdown table.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ProviderModelBreakdown {
    pub provider: String,
    pub model: String,
    pub requests_total: u64,
    pub deflected_total: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub estimated_cost_usd: f64,
}

#[derive(Debug)]
pub struct UsageTracker {
    config: Arc<AppConfig>,
    path: PathBuf,
    retention_days: u64,
    events: Mutex<Vec<UsageEvent>>,
}

impl UsageTracker {
    pub fn new(config: impl Into<Arc<AppConfig>>) -> Result<Self> {
        let config = config.into();
        let path = usage_log_file_path(&config)?;
        let retention_days = config.usage_retention_days.max(1);
        let mut events = load_events(&path)?;
        let original_len = events.len();
        prune_events_in_place(&mut events, retention_days);
        if events.len() != original_len {
            rewrite_events(&path, &events)?;
        }
        Ok(Self {
            config,
            path,
            retention_days,
            events: Mutex::new(events),
        })
    }

    pub fn record_cloud_usage(
        &self,
        provider: &str,
        model: &str,
        prompt: &str,
        response: &str,
    ) -> Result<()> {
        let prompt_tokens = estimate_prompt_tokens(prompt);
        let completion_tokens = estimate_completion_tokens(response);
        self.record_event(UsageEvent {
            timestamp: Utc::now().to_rfc3339(),
            provider: provider.to_string(),
            model: model.to_string(),
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            estimated_cost_usd: estimate_event_cost_usd(
                &self.config,
                provider,
                prompt_tokens,
                completion_tokens,
            ),
            final_layer: "l3".to_string(),
            deflected: false,
        })
    }

    pub fn record_deflection(
        &self,
        provider: &str,
        model: &str,
        prompt_tokens: u64,
        final_layer: &str,
    ) -> Result<()> {
        self.record_event(UsageEvent {
            timestamp: Utc::now().to_rfc3339(),
            provider: provider.to_string(),
            model: model.to_string(),
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
            estimated_cost_usd: estimate_event_cost_usd(&self.config, provider, prompt_tokens, 0),
            final_layer: final_layer.to_string(),
            deflected: true,
        })
    }

    pub fn snapshot(&self, window_hours: Option<u64>) -> UsageStatsResponse {
        let window_hours = window_hours
            .unwrap_or(self.config.usage_window_hours)
            .max(1);
        let cutoff = Utc::now() - Duration::hours(window_hours as i64);
        let cutoff_ts = cutoff.to_rfc3339();

        let events = self.events.lock();
        let mut by_key: BTreeMap<(String, String, String), UsageStatsEntry> = BTreeMap::new();
        let mut response = UsageStatsResponse {
            window_hours,
            usage_log_path: self.path.display().to_string(),
            retention_days: self.retention_days,
            ..UsageStatsResponse::default()
        };

        for event in events.iter().filter(|event| event.timestamp >= cutoff_ts) {
            response.total_prompt_tokens = response
                .total_prompt_tokens
                .saturating_add(event.prompt_tokens);
            response.total_completion_tokens = response
                .total_completion_tokens
                .saturating_add(event.completion_tokens);
            response.total_tokens = response.total_tokens.saturating_add(event.total_tokens);

            if event.deflected {
                response.total_deflected_requests =
                    response.total_deflected_requests.saturating_add(1);
                response.estimated_saved_cost_usd += event.estimated_cost_usd;
            } else {
                response.total_requests = response.total_requests.saturating_add(1);
                response.estimated_cost_usd += event.estimated_cost_usd;
            }

            let day = event.timestamp.get(..10).unwrap_or("unknown").to_string();
            let entry = by_key
                .entry((day.clone(), event.provider.clone(), event.model.clone()))
                .or_insert_with(|| UsageStatsEntry {
                    day,
                    provider: event.provider.clone(),
                    model: event.model.clone(),
                    ..UsageStatsEntry::default()
                });
            entry.prompt_tokens = entry.prompt_tokens.saturating_add(event.prompt_tokens);
            entry.completion_tokens = entry
                .completion_tokens
                .saturating_add(event.completion_tokens);
            entry.total_tokens = entry.total_tokens.saturating_add(event.total_tokens);
            if event.deflected {
                entry.deflected_total = entry.deflected_total.saturating_add(1);
                entry.estimated_saved_cost_usd += event.estimated_cost_usd;
            } else {
                entry.requests_total = entry.requests_total.saturating_add(1);
                entry.estimated_cost_usd += event.estimated_cost_usd;
            }
        }

        response.entries = by_key.into_values().collect();
        let denominator = response
            .total_requests
            .saturating_add(response.total_deflected_requests);
        if denominator > 0 {
            response.deflection_rate =
                response.total_deflected_requests as f64 / denominator as f64;
        }
        response
    }

    pub fn provider_usage_since(
        &self,
        provider: &str,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> ProviderUsageTotals {
        let events = self.events.lock();
        let mut totals = ProviderUsageTotals::default();

        for event in events.iter() {
            let Some(timestamp) = parse_timestamp(&event.timestamp) else {
                continue;
            };
            if event.deflected
                || event.provider != provider
                || timestamp < since
                || timestamp > until
            {
                continue;
            }

            totals.requests_total = totals.requests_total.saturating_add(1);
            totals.prompt_tokens = totals.prompt_tokens.saturating_add(event.prompt_tokens);
            totals.completion_tokens = totals
                .completion_tokens
                .saturating_add(event.completion_tokens);
            totals.total_tokens = totals.total_tokens.saturating_add(event.total_tokens);
            totals.estimated_cost_usd += event.estimated_cost_usd;
        }

        totals
    }

    /// Per-provider-model breakdown of usage within the given window.
    pub fn snapshot_by_provider(&self, window_hours: Option<u64>) -> Vec<ProviderModelBreakdown> {
        let window_hours = window_hours
            .unwrap_or(self.config.usage_window_hours)
            .max(1);
        let cutoff = Utc::now() - Duration::hours(window_hours as i64);
        let cutoff_ts = cutoff.to_rfc3339();

        let events = self.events.lock();
        let mut by_key: BTreeMap<(String, String), ProviderModelBreakdown> = BTreeMap::new();

        for event in events.iter().filter(|e| e.timestamp >= cutoff_ts) {
            let entry = by_key
                .entry((event.provider.clone(), event.model.clone()))
                .or_insert_with(|| ProviderModelBreakdown {
                    provider: event.provider.clone(),
                    model: event.model.clone(),
                    ..ProviderModelBreakdown::default()
                });
            if event.deflected {
                entry.deflected_total = entry.deflected_total.saturating_add(1);
            } else {
                entry.requests_total = entry.requests_total.saturating_add(1);
                entry.estimated_cost_usd += event.estimated_cost_usd;
            }
            entry.prompt_tokens = entry.prompt_tokens.saturating_add(event.prompt_tokens);
            entry.completion_tokens = entry
                .completion_tokens
                .saturating_add(event.completion_tokens);
            entry.total_tokens = entry.total_tokens.saturating_add(event.total_tokens);
        }

        by_key.into_values().collect()
    }
}

fn load_events(path: &Path) -> Result<Vec<UsageEvent>> {
    let file = match OpenOptions::new().read(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to open {}", path.display()));
        }
    };

    let reader = BufReader::new(file);
    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<UsageEvent>(trimmed) {
            Ok(event) => events.push(event),
            Err(error) => {
                tracing::warn!(error = %error, path = %path.display(), "Skipping malformed usage record")
            }
        }
    }
    Ok(events)
}

fn prune_events_in_place(events: &mut Vec<UsageEvent>, retention_days: u64) {
    let cutoff = Utc::now() - Duration::days(retention_days as i64);
    events.retain(|event| parse_timestamp(&event.timestamp).is_some_and(|ts| ts >= cutoff));
}

fn rewrite_events(path: &Path, events: &[UsageEvent]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create usage directory {}", parent.display()))?;
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("failed to rewrite {}", path.display()))?;
    for event in events {
        serde_json::to_writer(&mut file, event)
            .with_context(|| format!("failed to serialize {}", path.display()))?;
        writeln!(file)
            .with_context(|| format!("failed to append newline to {}", path.display()))?;
    }
    Ok(())
}

fn append_event(path: &Path, event: &UsageEvent) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create usage directory {}", parent.display()))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    serde_json::to_writer(&mut file, event)
        .with_context(|| format!("failed to serialize {}", path.display()))?;
    writeln!(file).with_context(|| format!("failed to append newline to {}", path.display()))?;
    Ok(())
}

fn parse_timestamp(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|ts| ts.with_timezone(&Utc))
}

impl UsageTracker {
    fn record_event(&self, event: UsageEvent) -> Result<()> {
        let mut events = self.events.lock();
        events.push(event.clone());
        prune_events_in_place(&mut events, self.retention_days);
        append_event(&self.path, &event)
    }
}

pub fn usage_log_file_path(config: &AppConfig) -> Result<PathBuf> {
    Ok(expand_usage_log_path(&config.usage_log_path)?.join(USAGE_LOG_FILE_NAME))
}

pub fn default_usage_log_dir_string() -> String {
    "~/.isartor".to_string()
}

fn expand_usage_log_path(path: &str) -> Result<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        anyhow::bail!("usage_log_path must not be empty");
    }

    if trimmed == "~" {
        return dirs::home_dir().context("cannot determine home directory");
    }

    if let Some(rest) = trimmed.strip_prefix("~/") {
        let home = dirs::home_dir().context("cannot determine home directory")?;
        return Ok(home.join(rest));
    }

    Ok(PathBuf::from(trimmed))
}

fn pricing_for_provider(config: &AppConfig, provider: &str) -> ProviderPricingConfig {
    config
        .usage_pricing
        .get(provider)
        .cloned()
        .unwrap_or_default()
}

pub fn estimate_event_cost_usd(
    config: &AppConfig,
    provider: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> f64 {
    let pricing = pricing_for_provider(config, provider);
    (prompt_tokens as f64 / 1_000_000.0) * pricing.input_cost_per_million_usd
        + (completion_tokens as f64 / 1_000_000.0) * pricing.output_cost_per_million_usd
}

pub fn estimate_prompt_tokens(prompt: &str) -> u64 {
    metrics::estimate_prompt_tokens(prompt)
}

pub fn estimate_completion_tokens(response: &str) -> u64 {
    metrics::estimate_completion_tokens(response)
}

pub fn resolved_usage_model(config: &AppConfig, path: &str, body: &[u8]) -> String {
    extract_request_model(body)
        .or_else(|| extract_route_model(path))
        .map(|requested| config.resolve_model_alias(&requested))
        .unwrap_or_else(|| config.configured_model_id())
}

pub async fn usage_stats_handler(
    Query(query): Query<UsageQuery>,
    request: Request,
) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {"message": "missing application state"}
                })),
            )
                .into_response();
        }
    };

    (
        StatusCode::OK,
        Json(state.usage_tracker.snapshot(query.hours)),
    )
        .into_response()
}

impl Default for UsageEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now().to_rfc3339(),
            provider: String::new(),
            model: String::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            estimated_cost_usd: 0.0,
            final_layer: String::new(),
            deflected: false,
        }
    }
}

pub fn default_usage_retention_days() -> u64 {
    DEFAULT_USAGE_RETENTION_DAYS
}

pub fn default_usage_window_hours() -> u64 {
    DEFAULT_USAGE_WINDOW_HOURS
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::AppConfig;
    use tempfile::tempdir;

    fn test_config(log_dir: &str) -> Arc<AppConfig> {
        let mut usage_pricing = std::collections::HashMap::new();
        usage_pricing.insert(
            "openai".to_string(),
            ProviderPricingConfig {
                input_cost_per_million_usd: 1.0,
                output_cost_per_million_usd: 2.0,
            },
        );
        let mut cfg = AppConfig::test_default();
        cfg.usage_log_path = log_dir.into();
        cfg.external_llm_url = "https://api.openai.com/v1/chat/completions".into();
        cfg.external_llm_model = "gpt-4o-mini".into();
        cfg.external_llm_api_key = "sk-test".into();
        cfg.cache_mode = crate::config::CacheMode::Both;
        cfg.azure_api_version = "2024-08-01-preview".into();
        cfg.usage_pricing = usage_pricing;
        Arc::new(cfg)
    }

    #[test]
    fn tracker_records_and_aggregates_spend_and_savings() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_str().unwrap());
        let tracker = UsageTracker::new(config.clone()).unwrap();
        tracker
            .record_cloud_usage("openai", "gpt-4o-mini", "hello world", "done")
            .unwrap();
        tracker
            .record_deflection("openai", "gpt-4o-mini", 1000, "l1a")
            .unwrap();

        let snapshot = tracker.snapshot(Some(24));
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.total_deflected_requests, 1);
        assert!(snapshot.estimated_saved_cost_usd > 0.0);
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].requests_total, 1);
        assert_eq!(snapshot.entries[0].deflected_total, 1);
    }

    #[test]
    fn tracker_reloads_existing_records() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_str().unwrap());
        let tracker = UsageTracker::new(config.clone()).unwrap();
        tracker
            .record_deflection("openai", "gpt-4o-mini", 500, "l2")
            .unwrap();
        drop(tracker);

        let reloaded = UsageTracker::new(config).unwrap();
        let snapshot = reloaded.snapshot(Some(24));
        assert_eq!(snapshot.total_deflected_requests, 1);
        assert_eq!(snapshot.entries.len(), 1);
    }
}
