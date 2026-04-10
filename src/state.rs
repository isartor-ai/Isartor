#![allow(dead_code)]
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use parking_lot::Mutex;
use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::client::Nothing;
use rig::completion::Prompt;
use rig::providers::{
    anthropic, azure, cohere, deepseek, galadriel, gemini, groq, huggingface, hyperbolic, mira,
    mistral, moonshot, ollama, openai, openrouter, perplexity, together, xai,
};

use crate::classifier::MiniLmMultiHeadClassifier;
use crate::clients::slm::SlmClient;
use crate::config::{
    AppConfig, DEFAULT_OPENAI_CHAT_COMPLETIONS_URL, KeyRotationStrategy, LlmProvider,
    ProviderKeyConfig, default_chat_completions_url, effective_provider_keys,
};
use crate::core::context_compress::InstructionCache;
use crate::core::usage::UsageTracker;
use crate::layer1::embeddings::TextEmbedder;
use crate::layer1::layer1a_cache::ExactMatchCache;
use crate::models::{
    ProviderHealthStatus, ProviderKeyHealthStatus, ProviderKeyStatusEntry, ProviderStatusEntry,
    ProviderStatusResponse,
};
use crate::providers::copilot::CopilotAgent;
use crate::vector_cache::VectorCache;

// ── Multi-provider Agent Wrapper ─────────────────────────────────────

#[async_trait::async_trait]
pub trait AppLlmAgent: Send + Sync {
    async fn chat(&self, prompt: &str) -> anyhow::Result<String>;
    fn provider_name(&self) -> &'static str;
}

pub struct RigAgent<M: rig::completion::CompletionModel> {
    pub name: &'static str,
    pub agent: Agent<M>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedProviderConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub api_key: String,
    pub provider_keys: Vec<ProviderKeyConfig>,
    pub key_rotation_strategy: KeyRotationStrategy,
    pub key_cooldown_secs: u64,
    pub endpoint: String,
    pub azure_deployment_id: String,
    pub azure_api_version: String,
    pub active: bool,
}

impl ResolvedProviderConfig {
    pub fn provider_name(&self) -> &'static str {
        self.provider.as_str()
    }

    pub fn configured_model_id(&self) -> &str {
        if self.provider == LlmProvider::Azure && !self.azure_deployment_id.trim().is_empty() {
            &self.azure_deployment_id
        } else {
            &self.model
        }
    }

    fn tracking_key(&self) -> String {
        format!("{}::{}", self.provider_name(), self.configured_model_id())
    }

    fn endpoint_configured(&self) -> bool {
        !self.endpoint.trim().is_empty()
    }

    fn api_key_configured(&self) -> bool {
        matches!(self.provider, LlmProvider::Ollama)
            || !self.provider_keys.is_empty()
            || !self.api_key.trim().is_empty()
    }

    fn effective_api_key(&self) -> String {
        self.provider_keys
            .first()
            .map(|entry| entry.key.clone())
            .filter(|key| !key.trim().is_empty())
            .unwrap_or_else(|| self.api_key.clone())
    }

    pub fn with_api_key(&self, api_key: String) -> Self {
        let mut cloned = self.clone();
        cloned.api_key = api_key;
        cloned
    }

    fn selection_tracking_key(&self) -> String {
        format!("{}::selection", self.tracking_key())
    }
}

#[derive(Debug, Clone)]
pub struct SelectedProviderKey {
    pub api_key: String,
}

#[derive(Debug, Clone)]
struct ProviderKeyDescriptor {
    masked_key: String,
    label: String,
    priority: u32,
}

#[derive(Debug, Clone)]
struct ProviderKeyRuntimeState {
    requests_total: u64,
    rate_limit_total: u64,
    last_used: Option<String>,
    cooldown_until: Option<String>,
    cooldown_until_instant: Option<Instant>,
}

#[derive(Debug, Clone)]
struct ProviderKeyPoolEntry {
    key: String,
    descriptor: ProviderKeyDescriptor,
    runtime: ProviderKeyRuntimeState,
}

#[derive(Debug)]
struct ProviderKeyPool {
    strategy: KeyRotationStrategy,
    cooldown_secs: u64,
    next_index: usize,
    entries: Vec<ProviderKeyPoolEntry>,
}

#[derive(Debug, Default)]
pub struct ProviderKeyPoolManager {
    pools: Mutex<HashMap<String, ProviderKeyPool>>,
}

fn mask_provider_key(key: &str) -> String {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return "(not configured)".to_string();
    }
    if trimmed.len() <= 8 {
        return "********".to_string();
    }
    format!("{}…{}", &trimmed[..4], &trimmed[trimmed.len() - 4..])
}

fn is_rate_limit_error(error: &str) -> bool {
    let lower = error.to_lowercase();
    lower.contains("429")
        || lower.contains("quota")
        || lower.contains("insufficient_quota")
        || lower.contains("rate limit")
        || lower.contains("rate_limit")
}

impl ProviderKeyPoolManager {
    pub fn from_provider_chain(provider_chain: &[ResolvedProviderConfig]) -> Self {
        let pools = provider_chain
            .iter()
            .filter(|provider| !provider.provider_keys.is_empty())
            .map(|provider| {
                (
                    provider.selection_tracking_key(),
                    ProviderKeyPool {
                        strategy: provider.key_rotation_strategy.clone(),
                        cooldown_secs: provider.key_cooldown_secs,
                        next_index: 0,
                        entries: provider
                            .provider_keys
                            .iter()
                            .map(|entry| ProviderKeyPoolEntry {
                                key: entry.key.clone(),
                                descriptor: ProviderKeyDescriptor {
                                    masked_key: mask_provider_key(&entry.key),
                                    label: entry.label.clone(),
                                    priority: entry.priority,
                                },
                                runtime: ProviderKeyRuntimeState {
                                    requests_total: 0,
                                    rate_limit_total: 0,
                                    last_used: None,
                                    cooldown_until: None,
                                    cooldown_until_instant: None,
                                },
                            })
                            .collect(),
                    },
                )
            })
            .collect();

        Self {
            pools: Mutex::new(pools),
        }
    }

    pub fn acquire(
        &self,
        provider: &ResolvedProviderConfig,
    ) -> anyhow::Result<SelectedProviderKey> {
        if provider.provider_keys.is_empty() {
            return Ok(SelectedProviderKey {
                api_key: provider.api_key.clone(),
            });
        }

        let mut pools = self.pools.lock();
        let Some(pool) = pools.get_mut(&provider.selection_tracking_key()) else {
            return Ok(SelectedProviderKey {
                api_key: provider.effective_api_key(),
            });
        };

        let now = Instant::now();
        for entry in &mut pool.entries {
            if entry
                .runtime
                .cooldown_until_instant
                .is_some_and(|deadline| deadline <= now)
            {
                entry.runtime.cooldown_until = None;
                entry.runtime.cooldown_until_instant = None;
            }
        }
        let available_indexes = pool
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let cooling = entry
                    .runtime
                    .cooldown_until_instant
                    .is_some_and(|deadline| deadline > now);
                if cooling { None } else { Some(index) }
            })
            .collect::<Vec<_>>();

        let Some(index) = select_provider_key_index(pool, &available_indexes) else {
            anyhow::bail!("all provider keys are on cooldown due to recent rate limit responses");
        };

        let entry = &mut pool.entries[index];
        entry.runtime.requests_total += 1;
        entry.runtime.last_used = Some(Utc::now().to_rfc3339());
        entry.runtime.cooldown_until = None;
        entry.runtime.cooldown_until_instant = None;

        Ok(SelectedProviderKey {
            api_key: entry.key.clone(),
        })
    }

    pub fn record_result(
        &self,
        provider: &ResolvedProviderConfig,
        api_key: &str,
        error: Option<&str>,
    ) {
        if provider.provider_keys.is_empty() || api_key.trim().is_empty() {
            return;
        }

        let mut pools = self.pools.lock();
        let Some(pool) = pools.get_mut(&provider.selection_tracking_key()) else {
            return;
        };
        let Some(entry) = pool.entries.iter_mut().find(|entry| entry.key == api_key) else {
            return;
        };

        if let Some(error) = error
            && is_rate_limit_error(error)
        {
            entry.runtime.rate_limit_total += 1;
            let deadline = Instant::now() + Duration::from_secs(pool.cooldown_secs);
            entry.runtime.cooldown_until = Some(
                (Utc::now() + chrono::TimeDelta::seconds(pool.cooldown_secs as i64)).to_rfc3339(),
            );
            entry.runtime.cooldown_until_instant = Some(deadline);
        }
    }

    pub fn snapshot_for_provider(
        &self,
        provider: &ResolvedProviderConfig,
    ) -> Vec<ProviderKeyStatusEntry> {
        let pools = self.pools.lock();
        let Some(pool) = pools.get(&provider.selection_tracking_key()) else {
            return provider
                .provider_keys
                .iter()
                .map(|entry| ProviderKeyStatusEntry {
                    masked_key: mask_provider_key(&entry.key),
                    label: entry.label.clone(),
                    priority: entry.priority,
                    status: ProviderKeyHealthStatus::Available,
                    requests_total: 0,
                    rate_limit_total: 0,
                    last_used: None,
                    cooldown_until: None,
                })
                .collect();
        };

        let now = Instant::now();
        pool.entries
            .iter()
            .map(|entry| ProviderKeyStatusEntry {
                masked_key: entry.descriptor.masked_key.clone(),
                label: entry.descriptor.label.clone(),
                priority: entry.descriptor.priority,
                status: if entry
                    .runtime
                    .cooldown_until_instant
                    .is_some_and(|deadline| deadline > now)
                {
                    ProviderKeyHealthStatus::CoolingDown
                } else {
                    ProviderKeyHealthStatus::Available
                },
                requests_total: entry.runtime.requests_total,
                rate_limit_total: entry.runtime.rate_limit_total,
                last_used: entry.runtime.last_used.clone(),
                cooldown_until: if entry
                    .runtime
                    .cooldown_until_instant
                    .is_some_and(|deadline| deadline > now)
                {
                    entry.runtime.cooldown_until.clone()
                } else {
                    None
                },
            })
            .collect()
    }
}

fn select_provider_key_index(
    pool: &mut ProviderKeyPool,
    available_indexes: &[usize],
) -> Option<usize> {
    if available_indexes.is_empty() {
        return None;
    }

    match pool.strategy {
        KeyRotationStrategy::RoundRobin => {
            for offset in 0..pool.entries.len() {
                let candidate = (pool.next_index + offset) % pool.entries.len();
                if available_indexes.contains(&candidate) {
                    pool.next_index = (candidate + 1) % pool.entries.len();
                    return Some(candidate);
                }
            }
            None
        }
        KeyRotationStrategy::Priority => {
            let selected = available_indexes.iter().copied().min_by_key(|index| {
                let entry = &pool.entries[*index];
                (entry.descriptor.priority, *index)
            })?;
            pool.next_index = (selected + 1) % pool.entries.len();
            Some(selected)
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
enum LastProviderOutcome {
    Healthy,
    Failing,
    #[default]
    Unknown,
}

#[derive(Debug, Clone)]
struct ProviderHealthDescriptor {
    tracking_key: String,
    provider_name: String,
    configured_model: String,
    endpoint: String,
    api_key_configured: bool,
    endpoint_configured: bool,
    key_rotation_strategy: String,
    key_cooldown_secs: u64,
    keys: Vec<ProviderKeyStatusEntry>,
    active: bool,
}

#[derive(Debug, Default)]
struct ProviderHealthState {
    requests_total: u64,
    errors_total: u64,
    last_success: Option<String>,
    last_error: Option<String>,
    last_error_message: Option<String>,
    last_outcome: LastProviderOutcome,
}

#[derive(Debug, Clone, Default)]
pub struct ProviderHealthStateSnapshot {
    pub requests_total: u64,
    pub errors_total: u64,
    pub last_success: Option<String>,
    pub last_error: Option<String>,
    pub last_error_message: Option<String>,
    pub status: ProviderHealthStatus,
}

#[derive(Debug)]
pub struct ProviderHealthTracker {
    active_provider: String,
    providers: Vec<ProviderHealthDescriptor>,
    state: Mutex<HashMap<String, ProviderHealthState>>,
}

impl ProviderHealthTracker {
    pub fn from_config(config: &AppConfig) -> Self {
        Self::from_provider_chain(&resolved_provider_chain(config))
    }

    pub fn from_provider_chain(provider_chain: &[ResolvedProviderConfig]) -> Self {
        Self {
            active_provider: provider_chain
                .first()
                .map(|provider| provider.provider_name().to_string())
                .unwrap_or_default(),
            providers: provider_chain
                .iter()
                .map(|provider| ProviderHealthDescriptor {
                    tracking_key: provider.tracking_key(),
                    provider_name: provider.provider_name().to_string(),
                    configured_model: provider.configured_model_id().to_string(),
                    endpoint: provider_status_endpoint(provider),
                    api_key_configured: provider.api_key_configured(),
                    endpoint_configured: provider.endpoint_configured(),
                    key_rotation_strategy: key_rotation_strategy_label(
                        &provider.key_rotation_strategy,
                    ),
                    key_cooldown_secs: provider.key_cooldown_secs,
                    keys: provider
                        .provider_keys
                        .iter()
                        .map(|entry| ProviderKeyStatusEntry {
                            masked_key: mask_provider_key(&entry.key),
                            label: entry.label.clone(),
                            priority: entry.priority,
                            status: ProviderKeyHealthStatus::Available,
                            requests_total: 0,
                            rate_limit_total: 0,
                            last_used: None,
                            cooldown_until: None,
                        })
                        .collect(),
                    active: provider.active,
                })
                .collect(),
            state: Mutex::new(HashMap::new()),
        }
    }

    pub fn record_success(&self, provider: &ResolvedProviderConfig) {
        let mut state = self.state.lock();
        let provider_state = state.entry(provider.tracking_key()).or_default();
        provider_state.requests_total += 1;
        provider_state.last_success = Some(Utc::now().to_rfc3339());
        provider_state.last_error_message = None;
        provider_state.last_outcome = LastProviderOutcome::Healthy;
    }

    pub fn record_failure(&self, provider: &ResolvedProviderConfig, error: &str) {
        let mut state = self.state.lock();
        let provider_state = state.entry(provider.tracking_key()).or_default();
        provider_state.requests_total += 1;
        provider_state.errors_total += 1;
        provider_state.last_error = Some(Utc::now().to_rfc3339());
        provider_state.last_error_message = Some(compact_provider_error(error));
        provider_state.last_outcome = LastProviderOutcome::Failing;
    }

    pub fn record_probe_success(&self, provider: &ResolvedProviderConfig) {
        let mut state = self.state.lock();
        let provider_state = state.entry(provider.tracking_key()).or_default();
        provider_state.last_success = Some(Utc::now().to_rfc3339());
        provider_state.last_error_message = None;
        provider_state.last_outcome = LastProviderOutcome::Healthy;
    }

    pub fn record_probe_failure(&self, provider: &ResolvedProviderConfig, error: Option<String>) {
        let mut state = self.state.lock();
        let provider_state = state.entry(provider.tracking_key()).or_default();
        provider_state.last_error = Some(Utc::now().to_rfc3339());
        provider_state.last_error_message = error.map(|message| compact_provider_error(&message));
        provider_state.last_outcome = LastProviderOutcome::Failing;
    }

    pub fn health_state_snapshot(&self) -> HashMap<String, ProviderHealthStateSnapshot> {
        let state = self.state.lock();
        state
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    ProviderHealthStateSnapshot {
                        requests_total: value.requests_total,
                        errors_total: value.errors_total,
                        last_success: value.last_success.clone(),
                        last_error: value.last_error.clone(),
                        last_error_message: value.last_error_message.clone(),
                        status: match value.last_outcome {
                            LastProviderOutcome::Healthy => ProviderHealthStatus::Healthy,
                            LastProviderOutcome::Failing => ProviderHealthStatus::Failing,
                            LastProviderOutcome::Unknown => ProviderHealthStatus::Unknown,
                        },
                    },
                )
            })
            .collect()
    }

    pub fn snapshot(&self) -> ProviderStatusResponse {
        ProviderStatusResponse {
            active_provider: self.active_provider.clone(),
            providers: self.snapshot_entries(),
        }
    }

    fn snapshot_entries(&self) -> Vec<ProviderStatusEntry> {
        let state = self.state.lock();
        self.providers
            .iter()
            .map(|provider| {
                let provider_state = state.get(&provider.tracking_key);
                ProviderStatusEntry {
                    name: provider.provider_name.clone(),
                    active: provider.active,
                    status: match provider_state
                        .map(|entry| entry.last_outcome)
                        .unwrap_or_default()
                    {
                        LastProviderOutcome::Healthy => ProviderHealthStatus::Healthy,
                        LastProviderOutcome::Failing => ProviderHealthStatus::Failing,
                        LastProviderOutcome::Unknown => ProviderHealthStatus::Unknown,
                    },
                    model: provider.configured_model.clone(),
                    raw_model: None,
                    endpoint: provider.endpoint.clone(),
                    config_url: None,
                    api_key_configured: provider.api_key_configured,
                    endpoint_configured: provider.endpoint_configured,
                    config_index: None,
                    azure_deployment_id: None,
                    azure_api_version: None,
                    requests_total: provider_state
                        .map(|entry| entry.requests_total)
                        .unwrap_or(0),
                    errors_total: provider_state.map(|entry| entry.errors_total).unwrap_or(0),
                    key_rotation_strategy: provider.key_rotation_strategy.clone(),
                    key_cooldown_secs: provider.key_cooldown_secs,
                    keys: provider.keys.clone(),
                    last_success: provider_state.and_then(|entry| entry.last_success.clone()),
                    last_error: provider_state.and_then(|entry| entry.last_error.clone()),
                    last_error_message: provider_state
                        .and_then(|entry| entry.last_error_message.clone()),
                }
            })
            .collect()
    }
}

macro_rules! build_rig_agent {
    ($name:literal, $client:path, $api_key:expr, $model:expr, $http_client:expr) => {{
        let client = <$client>::builder()
            .api_key($api_key.clone())
            .http_client($http_client.clone())
            .build()
            .expect(concat!("Failed to initialize ", $name, " client"));
        Arc::new(RigAgent {
            name: $name,
            agent: client.agent($model).build(),
        })
    }};
}

fn is_openai_compatible_runtime_provider(provider: &str) -> bool {
    matches!(
        provider,
        "openai" | "cerebras" | "nebius" | "siliconflow" | "fireworks" | "nvidia" | "chutes"
    )
}

fn openai_compatible_base_url(endpoint: &str) -> String {
    endpoint
        .trim()
        .trim_end_matches('/')
        .trim_end_matches("/chat/completions")
        .to_string()
}

pub fn resolved_provider_chain(config: &AppConfig) -> Vec<ResolvedProviderConfig> {
    let primary_provider_keys =
        effective_provider_keys(&config.external_llm_api_key, &config.provider_keys);

    // Attempt to inject an OAuth token for the primary provider if no API key is configured.
    let primary_api_key = primary_provider_keys
        .first()
        .map(|entry| entry.key.clone())
        .unwrap_or_else(|| config.external_llm_api_key.clone());
    let primary_api_key =
        inject_oauth_token_if_empty(primary_api_key, &config.llm_provider.to_string());

    let mut providers = vec![ResolvedProviderConfig {
        provider: config.llm_provider.clone(),
        model: config.external_llm_model.clone(),
        api_key: primary_api_key,
        provider_keys: primary_provider_keys,
        key_rotation_strategy: config.key_rotation_strategy.clone(),
        key_cooldown_secs: config.key_cooldown_secs,
        endpoint: config.external_llm_url.clone(),
        azure_deployment_id: config.azure_deployment_id.clone(),
        azure_api_version: config.azure_api_version.clone(),
        active: true,
    }];

    providers.extend(config.fallback_providers.iter().map(|provider| {
        let endpoint = if provider.url.trim().is_empty() {
            default_chat_completions_url(&provider.provider)
                .unwrap_or_default()
                .to_string()
        } else {
            provider.url.clone()
        };
        let provider_keys = effective_provider_keys(&provider.api_key, &provider.provider_keys);
        let api_key = provider_keys
            .first()
            .map(|entry| entry.key.clone())
            .unwrap_or_else(|| provider.api_key.clone());
        let api_key = inject_oauth_token_if_empty(api_key, &provider.provider.to_string());

        ResolvedProviderConfig {
            provider: provider.provider.clone(),
            model: provider.model.clone(),
            api_key,
            provider_keys,
            key_rotation_strategy: provider.key_rotation_strategy.clone(),
            key_cooldown_secs: provider.key_cooldown_secs,
            endpoint,
            azure_deployment_id: provider.azure_deployment_id.clone(),
            azure_api_version: provider.azure_api_version.clone(),
            active: false,
        }
    }));

    providers
}

/// If `api_key` is empty, attempt to load a stored OAuth token for `provider_name`.
///
/// This is a synchronous best-effort read — if the token store is unavailable or
/// the token is missing/expired, the original (empty) key is returned unchanged.
fn inject_oauth_token_if_empty(api_key: String, provider_name: &str) -> String {
    if !api_key.trim().is_empty() {
        return api_key;
    }
    match load_stored_oauth_token(provider_name) {
        Some(token) => token,
        None => api_key,
    }
}

fn load_stored_oauth_token(provider_name: &str) -> Option<String> {
    let store = crate::auth::TokenStore::open().ok()?;
    let token = store.load(provider_name).ok()??;

    if token.access_token.trim().is_empty() {
        return None;
    }

    if !token.is_expired() {
        tracing::debug!(
            provider = provider_name,
            "using stored OAuth token as provider API key"
        );
        return Some(token.access_token.clone());
    }

    token.refresh_token.as_deref()?;
    let http = reqwest::Client::builder().build().ok()?;
    let refresh_future = store.load_refreshed(provider_name, &http);
    let refreshed = match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(refresh_future))
                .ok()?
                .filter(|token| !token.is_expired())
        }
        Ok(_) => {
            tracing::debug!(
                provider = provider_name,
                "stored OAuth token is expired and refresh is unavailable on the current-thread runtime"
            );
            return None;
        }
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?
            .block_on(refresh_future)
            .ok()?
            .filter(|token| !token.is_expired()),
    }?;

    if refreshed.access_token.trim().is_empty() {
        return None;
    }

    tracing::debug!(
        provider = provider_name,
        "using refreshed OAuth token as provider API key"
    );
    Some(refreshed.access_token.clone())
}

fn provider_status_endpoint(config: &ResolvedProviderConfig) -> String {
    match &config.provider {
        LlmProvider::Azure => {
            if config.endpoint.trim().is_empty()
                || config.azure_deployment_id.trim().is_empty()
                || config.azure_api_version.trim().is_empty()
            {
                config.endpoint.trim().to_string()
            } else {
                format!(
                    "{}/openai/deployments/{}/chat/completions?api-version={}",
                    config.endpoint.trim_end_matches('/'),
                    config.azure_deployment_id,
                    config.azure_api_version
                )
            }
        }
        LlmProvider::Anthropic => "https://api.anthropic.com/v1/messages".to_string(),
        LlmProvider::Copilot => {
            let configured = config.endpoint.trim();
            if configured.is_empty() {
                "https://api.githubcopilot.com/chat/completions".to_string()
            } else {
                configured.to_string()
            }
        }
        LlmProvider::Gemini => format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}",
            config.model
        ),
        LlmProvider::Ollama => {
            let configured = config.endpoint.trim();
            if configured.is_empty() {
                "http://localhost:11434".to_string()
            } else {
                configured.to_string()
            }
        }
        LlmProvider::Cohere => "https://api.cohere.ai/v1/chat".to_string(),
        LlmProvider::Huggingface => format!(
            "https://api-inference.huggingface.co/models/{}",
            config.model
        ),
        provider => {
            let configured = config.endpoint.trim();
            if let Some(default_url) = default_chat_completions_url(provider) {
                if configured.is_empty()
                    || (*provider != LlmProvider::Openai
                        && configured == DEFAULT_OPENAI_CHAT_COMPLETIONS_URL)
                {
                    default_url.to_string()
                } else {
                    configured.to_string()
                }
            } else {
                configured.to_string()
            }
        }
    }
}

fn uses_openai_compatible_runtime(provider: &ResolvedProviderConfig) -> bool {
    is_openai_compatible_runtime_provider(provider.provider_name())
        || (provider.provider == LlmProvider::Openai
            && !provider.endpoint.trim().is_empty()
            && provider.endpoint.trim() != DEFAULT_OPENAI_CHAT_COMPLETIONS_URL)
}

fn compact_provider_error(error: &str) -> String {
    let single_line = error.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX_LEN: usize = 240;
    if single_line.len() <= MAX_LEN {
        single_line
    } else {
        format!("{}...", &single_line[..MAX_LEN - 3])
    }
}

fn key_rotation_strategy_label(strategy: &KeyRotationStrategy) -> String {
    match strategy {
        KeyRotationStrategy::RoundRobin => "round_robin".to_string(),
        KeyRotationStrategy::Priority => "priority".to_string(),
    }
}

fn build_openai_compatible_rig_agent(
    name: &'static str,
    api_key: &str,
    model: &str,
    endpoint: &str,
    http_client: rig::http_client::ReqwestClient,
) -> Arc<dyn AppLlmAgent> {
    let client = openai::Client::builder()
        .api_key(api_key)
        .base_url(openai_compatible_base_url(endpoint))
        .http_client(http_client)
        .build()
        .expect("Failed to initialize OpenAI-compatible client");
    Arc::new(RigAgent {
        name,
        agent: client.agent(model).build(),
    })
}

fn build_agent_for_provider(
    provider: &ResolvedProviderConfig,
    http_client: reqwest::Client,
    rig_http_client: rig::http_client::ReqwestClient,
    l3_timeout: Duration,
) -> Arc<dyn AppLlmAgent> {
    match provider.provider_name() {
        "azure" => {
            let client: azure::Client = azure::Client::builder()
                .api_key(provider.api_key.as_str())
                .http_client(rig_http_client)
                .azure_endpoint(provider.endpoint.clone())
                .api_version(&provider.azure_api_version)
                .build()
                .expect("Failed to initialize Azure OpenAI client");
            Arc::new(RigAgent {
                name: "azure",
                agent: client.agent(&provider.azure_deployment_id).build(),
            })
        }
        "anthropic" => {
            build_rig_agent!(
                "anthropic",
                anthropic::Client,
                provider.api_key,
                &provider.model,
                rig_http_client
            )
        }
        "copilot" => Arc::new(CopilotAgent::new(
            http_client,
            provider.api_key.clone(),
            provider.model.clone(),
            l3_timeout,
        )),
        "xai" => build_rig_agent!(
            "xai",
            xai::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "gemini" => build_rig_agent!(
            "gemini",
            gemini::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "mistral" => build_rig_agent!(
            "mistral",
            mistral::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "groq" => build_rig_agent!(
            "groq",
            groq::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "deepseek" => build_rig_agent!(
            "deepseek",
            deepseek::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "cohere" => build_rig_agent!(
            "cohere",
            cohere::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "galadriel" => build_rig_agent!(
            "galadriel",
            galadriel::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "hyperbolic" => build_rig_agent!(
            "hyperbolic",
            hyperbolic::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "huggingface" => build_rig_agent!(
            "huggingface",
            huggingface::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "mira" => build_rig_agent!(
            "mira",
            mira::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "moonshot" => build_rig_agent!(
            "moonshot",
            moonshot::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "ollama" => {
            unsafe {
                env::set_var("OLLAMA_HOST", &provider.endpoint);
            }
            let client = ollama::Client::builder()
                .api_key(Nothing)
                .http_client(rig_http_client)
                .build()
                .expect("Failed to initialize Ollama client");
            Arc::new(RigAgent {
                name: "ollama",
                agent: client.agent(&provider.model).build(),
            })
        }
        "openrouter" => build_rig_agent!(
            "openrouter",
            openrouter::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "perplexity" => build_rig_agent!(
            "perplexity",
            perplexity::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        "together" => build_rig_agent!(
            "together",
            together::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
        _ if uses_openai_compatible_runtime(provider) => build_openai_compatible_rig_agent(
            provider.provider_name(),
            &provider.api_key,
            &provider.model,
            &provider.endpoint,
            rig_http_client,
        ),
        _ => build_rig_agent!(
            "openai",
            openai::Client,
            provider.api_key,
            &provider.model,
            rig_http_client
        ),
    }
}

#[async_trait::async_trait]
impl<M> AppLlmAgent for RigAgent<M>
where
    M: rig::completion::CompletionModel + Send + Sync,
{
    async fn chat(&self, prompt: &str) -> anyhow::Result<String> {
        self.agent
            .prompt(prompt)
            .await
            .map_err(|e| anyhow::anyhow!(e))
    }

    fn provider_name(&self) -> &'static str {
        self.name
    }
}

// ── App State ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub http_client: reqwest::Client,
    pub exact_cache: Arc<ExactMatchCache>,
    pub vector_cache: Arc<VectorCache>,
    pub provider_health: Arc<ProviderHealthTracker>,
    pub provider_key_pools: Arc<ProviderKeyPoolManager>,
    pub provider_chain: Arc<Vec<ResolvedProviderConfig>>,
    pub usage_tracker: Arc<UsageTracker>,

    /// Rig AI Agent encapsulating the configured Layer 3 provider.
    pub llm_agent: Arc<dyn AppLlmAgent>,

    /// Dedicated HTTP client for the llama.cpp generation sidecar.
    pub slm_client: Arc<SlmClient>,

    /// In-process sentence embedding model for Layer 1 semantic cache.
    /// Pure-Rust candle BertModel with sentence-transformers/all-MiniLM-L6-v2.
    pub text_embedder: Arc<TextEmbedder>,

    /// L2.5 instruction dedup cache for cross-turn session deduplication.
    pub instruction_cache: Arc<InstructionCache>,

    /// Gateway start time — used by the dashboard to compute uptime.
    pub started_at: Instant,

    /// Optional MiniLM multi-head classifier used for request routing.
    pub minilm_classifier: Option<Arc<MiniLmMultiHeadClassifier>>,

    #[cfg(feature = "embedded-inference")]
    pub embedded_classifier: Option<Arc<crate::services::local_inference::EmbeddedClassifier>>,
}

impl AppState {
    pub fn new(config: Arc<AppConfig>, text_embedder: Arc<TextEmbedder>) -> Self {
        let l3_timeout = Duration::from_secs(config.l3_timeout_secs);
        let http_client = reqwest::Client::builder()
            .timeout(l3_timeout)
            .build()
            .expect("failed to build reqwest client");
        let rig_http_client = rig::http_client::ReqwestClient::builder()
            .timeout(l3_timeout)
            .build()
            .expect("failed to build rig reqwest client");
        let provider_chain = Arc::new(resolved_provider_chain(&config));
        let provider_key_pools = Arc::new(ProviderKeyPoolManager::from_provider_chain(
            provider_chain.as_slice(),
        ));
        let primary_provider = provider_chain
            .first()
            .expect("provider chain must contain a primary provider")
            .clone();

        let exact_cache = Arc::new(ExactMatchCache::new(
            std::num::NonZeroUsize::new(config.cache_max_capacity as usize)
                .unwrap_or_else(|| std::num::NonZeroUsize::new(128).unwrap()),
        ));
        let vector_cache = Arc::new(VectorCache::new(
            config.similarity_threshold,
            config.cache_ttl_secs,
            config.cache_max_capacity,
        ));

        let agent = build_agent_for_provider(
            &primary_provider,
            http_client.clone(),
            rig_http_client,
            l3_timeout,
        );

        let slm_client = Arc::new(SlmClient::new(&config.layer2));
        let usage_tracker = Arc::new(
            UsageTracker::new(config.clone()).expect("failed to initialize usage tracker"),
        );
        let minilm_classifier = if config.classifier_routing.enabled {
            if config.classifier_routing.artifacts_path.trim().is_empty() {
                tracing::warn!(
                    "Classifier routing enabled but classifier_routing.artifacts_path is empty; falling back to existing routing"
                );
                None
            } else {
                match MiniLmMultiHeadClassifier::from_path(
                    &config.classifier_routing.artifacts_path,
                ) {
                    Ok(classifier) => Some(Arc::new(classifier)),
                    Err(error) => {
                        tracing::warn!(
                            error = %error,
                            artifacts_path = %config.classifier_routing.artifacts_path,
                            "Failed to load MiniLM classifier artifact; falling back to existing routing"
                        );
                        None
                    }
                }
            }
        } else {
            None
        };

        #[cfg(feature = "embedded-inference")]
        let embedded_classifier =
            if config.inference_engine == crate::config::InferenceEngineMode::Embedded {
                // NOTE: In a real app we would want to bubble up this error instead of
                // doing blocking initialization or panic, but for the sake of the architecture
                // state encapsulation we can block_on it or pass it in. Assuming blocking for now
                // or we change AppState::new to be async.
                let mut cfg = crate::services::local_inference::EmbeddedClassifierConfig::default();
                // Allow overriding the model path via env var (e.g. Docker image with baked-in model).
                if let Ok(path) = std::env::var("ISARTOR__EMBEDDED__MODEL_PATH")
                    && !path.is_empty()
                {
                    cfg.model_path = Some(path);
                }
                // Since `AppState::new` is not async, we use a blocking fallback or expect initialization elsewhere.
                // For simplicity in this sync constructor we will leave it as None and assume an async `init` method later,
                // or just block_on. We use block_on here for convenience.
                let engine = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        crate::services::local_inference::EmbeddedClassifier::new(cfg).await
                    })
                })
                .expect("Failed to initialize Embedded Classifier");
                Some(Arc::new(engine))
            } else {
                None
            };

        Self {
            config,
            http_client,
            exact_cache,
            vector_cache,
            provider_health: Arc::new(ProviderHealthTracker::from_provider_chain(
                provider_chain.as_slice(),
            )),
            provider_key_pools,
            provider_chain,
            usage_tracker,
            llm_agent: agent,
            slm_client,
            text_embedder,
            instruction_cache: Arc::new(InstructionCache::new()),
            started_at: Instant::now(),
            minilm_classifier,
            #[cfg(feature = "embedded-inference")]
            embedded_classifier,
        }
    }

    pub async fn chat_with_model(&self, prompt: &str, model: &str) -> anyhow::Result<String> {
        self.chat_with_provider(self.primary_provider(), prompt, Some(model))
            .await
    }

    pub fn primary_provider(&self) -> &ResolvedProviderConfig {
        self.provider_chain
            .first()
            .expect("provider chain must contain a primary provider")
    }

    pub async fn chat_with_provider(
        &self,
        provider: &ResolvedProviderConfig,
        prompt: &str,
        model_override: Option<&str>,
    ) -> anyhow::Result<String> {
        let model = model_override.unwrap_or(provider.configured_model_id());
        let selected_key = self.provider_key_pools.acquire(provider)?;
        let execution_provider = provider.with_api_key(selected_key.api_key.clone());

        let result = if execution_provider.active
            && model == self.config.configured_model_id()
            && execution_provider.api_key == provider.api_key
        {
            self.llm_agent.chat(prompt).await
        } else {
            let l3_timeout = Duration::from_secs(self.config.l3_timeout_secs);
            let rig_http_client = rig::http_client::ReqwestClient::builder()
                .timeout(l3_timeout)
                .build()
                .expect("failed to build rig reqwest client");

            match execution_provider.provider_name() {
                "azure" => {
                    let client: azure::Client = azure::Client::builder()
                        .api_key(execution_provider.api_key.as_str())
                        .http_client(rig_http_client)
                        .azure_endpoint(execution_provider.endpoint.clone())
                        .api_version(&execution_provider.azure_api_version)
                        .build()
                        .expect("Failed to initialize Azure OpenAI client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "anthropic" => {
                    let client = anthropic::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize anthropic client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "copilot" => {
                    CopilotAgent::chat_with_model(
                        &self.http_client,
                        &execution_provider.api_key,
                        model,
                        l3_timeout,
                        prompt,
                    )
                    .await
                }
                "xai" => {
                    let client = xai::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize xai client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "gemini" => {
                    let client = gemini::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize gemini client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "mistral" => {
                    let client = mistral::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize mistral client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "groq" => {
                    let client = groq::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize groq client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "deepseek" => {
                    let client = deepseek::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize deepseek client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "cohere" => {
                    let client = cohere::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize cohere client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "galadriel" => {
                    let client = galadriel::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize galadriel client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "hyperbolic" => {
                    let client = hyperbolic::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize hyperbolic client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "huggingface" => {
                    let client = huggingface::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize huggingface client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "mira" => {
                    let client = mira::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize mira client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "moonshot" => {
                    let client = moonshot::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize moonshot client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "ollama" => {
                    unsafe {
                        env::set_var("OLLAMA_HOST", &execution_provider.endpoint);
                    }
                    let client = ollama::Client::builder()
                        .api_key(Nothing)
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize Ollama client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "openrouter" => {
                    let client = openrouter::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize openrouter client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "perplexity" => {
                    let client = perplexity::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize perplexity client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                "together" => {
                    let client = together::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize together client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                _ if uses_openai_compatible_runtime(provider) => {
                    let client = openai::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .base_url(openai_compatible_base_url(&execution_provider.endpoint))
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize OpenAI-compatible client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
                _ => {
                    let client = openai::Client::builder()
                        .api_key(execution_provider.api_key.clone())
                        .http_client(rig_http_client)
                        .build()
                        .expect("Failed to initialize openai client");
                    client
                        .agent(model)
                        .build()
                        .prompt(prompt)
                        .await
                        .map_err(|e| anyhow::anyhow!(e))
                }
            }
        };

        let error_message = result.as_ref().err().map(|error| error.to_string());
        self.provider_key_pools.record_result(
            provider,
            &execution_provider.api_key,
            error_message.as_deref(),
        );

        result
    }

    pub fn record_provider_success(&self, provider: &ResolvedProviderConfig) {
        self.provider_health.record_success(provider);
    }

    pub fn record_provider_failure(&self, provider: &ResolvedProviderConfig, error: &str) {
        self.provider_health.record_failure(provider, error);
    }

    pub fn provider_status(&self) -> ProviderStatusResponse {
        let mut snapshot = self.provider_health.snapshot();
        for (index, (provider, entry)) in self
            .provider_chain
            .iter()
            .zip(snapshot.providers.iter_mut())
            .enumerate()
        {
            entry.keys = self.provider_key_pools.snapshot_for_provider(provider);
            entry.raw_model = Some(provider.model.clone());
            entry.config_url = Some(provider.endpoint.clone());
            entry.config_index = if provider.active {
                None
            } else {
                Some(index - 1)
            };
            if provider.provider == LlmProvider::Azure {
                entry.azure_deployment_id = Some(provider.azure_deployment_id.clone());
                entry.azure_api_version = Some(provider.azure_api_version.clone());
            }
        }
        snapshot
    }
}

#[cfg(any(test, feature = "test-helpers"))]
impl AppState {
    /// Build a test `AppState` from the given agent and config.
    pub fn test_with_agent(agent: Arc<dyn AppLlmAgent>, config: Arc<AppConfig>) -> Arc<Self> {
        use crate::clients::slm::SlmClient;
        use crate::core::context_compress::InstructionCache;
        use crate::core::usage::UsageTracker;
        use crate::layer1::layer1a_cache::ExactMatchCache;
        use crate::vector_cache::VectorCache;
        use std::num::NonZeroUsize;

        Arc::new(Self {
            http_client: reqwest::Client::new(),
            exact_cache: Arc::new(ExactMatchCache::new(NonZeroUsize::new(100).unwrap())),
            vector_cache: Arc::new(VectorCache::new(
                config.similarity_threshold,
                config.cache_ttl_secs,
                config.cache_max_capacity,
            )),
            provider_chain: Arc::new(resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: agent,
            slm_client: Arc::new(SlmClient::new(&config.layer2)),
            text_embedder: crate::layer1::embeddings::shared_test_embedder(),
            instruction_cache: Arc::new(InstructionCache::new()),
            started_at: Instant::now(),
            provider_health: Arc::new(ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(ProviderKeyPoolManager::from_provider_chain(
                resolved_provider_chain(&config).as_slice(),
            )),
            config,
            minilm_classifier: None,
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer1::embeddings::shared_test_embedder;
    // ── ExactCache tests ─────────────────────────────────────────

    #[tokio::test]
    async fn exact_cache_insert_and_get() {
        // Tested in layer1::layer1a_cache::tests — retained as placeholder.
    }

    // ── AppLlmAgent mock for testing ─────────────────────────────

    struct MockAgent;

    #[async_trait::async_trait]
    impl AppLlmAgent for MockAgent {
        async fn chat(&self, prompt: &str) -> anyhow::Result<String> {
            Ok(format!("Mock response to: {prompt}"))
        }
        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    #[tokio::test]
    async fn mock_agent_chat() {
        let agent = MockAgent;
        let result = agent.chat("hello").await.unwrap();
        assert_eq!(result, "Mock response to: hello");
        assert_eq!(agent.provider_name(), "mock");
    }

    // ── AppState::new tests ──────────────────────────────────────

    fn make_test_config(provider: &str) -> Arc<AppConfig> {
        let mut cfg = AppConfig::test_default();
        cfg.llm_provider = provider.into();
        cfg.external_llm_url = "https://api.openai.com".into();
        cfg.external_llm_api_key = "sk-test".into();
        cfg.azure_deployment_id = "my-deployment".into();
        cfg.azure_api_version = "2024-02-15-preview".into();
        cfg.similarity_threshold = 0.92;
        cfg.cache_mode = crate::config::CacheMode::Both;
        cfg.cache_max_capacity = 1000;
        cfg.embedding_model = "test".into();
        cfg.otel_exporter_endpoint = String::new();
        cfg.layer2.sidecar_url = "http://localhost:8081".into();
        cfg.layer2.model_name = "test-model".into();
        cfg.layer2.timeout_seconds = 30;
        cfg.embedding_sidecar.sidecar_url = "http://localhost:8082".into();
        cfg.embedding_sidecar.model_name = "test-embed".into();
        cfg.embedding_sidecar.timeout_seconds = 30;
        Arc::new(cfg)
    }

    #[tokio::test]
    async fn app_state_new_default_openai_provider() {
        let state = AppState::new(make_test_config("openai"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "openai");
        assert_eq!(state.config.llm_provider, "openai".into());
    }

    #[tokio::test]
    async fn app_state_new_unknown_provider_defaults_to_openai() {
        // "unknown-provider" falls to the default branch → openai.
        let state = AppState::new(make_test_config("unknown-provider"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "openai");
    }

    #[tokio::test]
    async fn app_state_new_anthropic_provider() {
        let state = AppState::new(make_test_config("anthropic"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "anthropic");
    }

    #[tokio::test]
    async fn app_state_new_xai_provider() {
        let state = AppState::new(make_test_config("xai"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "xai");
    }

    #[tokio::test]
    async fn app_state_new_azure_provider() {
        let state = AppState::new(make_test_config("azure"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "azure");
    }

    #[tokio::test]
    async fn app_state_new_gemini_provider() {
        let state = AppState::new(make_test_config("gemini"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "gemini");
    }

    #[tokio::test]
    async fn app_state_new_mistral_provider() {
        let state = AppState::new(make_test_config("mistral"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "mistral");
    }

    #[tokio::test]
    async fn app_state_new_groq_provider() {
        let state = AppState::new(make_test_config("groq"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "groq");
    }

    #[tokio::test]
    async fn app_state_new_cerebras_provider() {
        let mut config = (*make_test_config("cerebras")).clone();
        config.external_llm_url = "https://api.cerebras.ai/v1/chat/completions".into();
        let state = AppState::new(Arc::new(config), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "cerebras");
    }

    #[tokio::test]
    async fn app_state_new_deepseek_provider() {
        let state = AppState::new(make_test_config("deepseek"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "deepseek");
    }

    #[tokio::test]
    async fn app_state_new_cohere_provider() {
        let state = AppState::new(make_test_config("cohere"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "cohere");
    }

    #[test]
    fn provider_key_pool_round_robin_rotates_keys() {
        let provider = ResolvedProviderConfig {
            provider: LlmProvider::Openai,
            model: "gpt-4o-mini".into(),
            api_key: "sk-primary".into(),
            provider_keys: vec![
                ProviderKeyConfig {
                    key: "sk-primary".into(),
                    priority: 1,
                    label: "primary".into(),
                },
                ProviderKeyConfig {
                    key: "sk-shared".into(),
                    priority: 2,
                    label: "shared".into(),
                },
            ],
            key_rotation_strategy: KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            endpoint: "https://api.openai.com/v1/chat/completions".into(),
            azure_deployment_id: String::new(),
            azure_api_version: "2024-08-01-preview".into(),
            active: true,
        };
        let manager = ProviderKeyPoolManager::from_provider_chain(std::slice::from_ref(&provider));

        let first = manager.acquire(&provider).unwrap();
        let second = manager.acquire(&provider).unwrap();

        assert_eq!(first.api_key, "sk-primary");
        assert_eq!(second.api_key, "sk-shared");
    }

    #[test]
    fn provider_key_pool_priority_skips_key_on_cooldown() {
        let provider = ResolvedProviderConfig {
            provider: LlmProvider::Openai,
            model: "gpt-4o-mini".into(),
            api_key: "sk-primary".into(),
            provider_keys: vec![
                ProviderKeyConfig {
                    key: "sk-primary".into(),
                    priority: 1,
                    label: "primary".into(),
                },
                ProviderKeyConfig {
                    key: "sk-shared".into(),
                    priority: 2,
                    label: "shared".into(),
                },
            ],
            key_rotation_strategy: KeyRotationStrategy::Priority,
            key_cooldown_secs: 60,
            endpoint: "https://api.openai.com/v1/chat/completions".into(),
            azure_deployment_id: String::new(),
            azure_api_version: "2024-08-01-preview".into(),
            active: true,
        };
        let manager = ProviderKeyPoolManager::from_provider_chain(std::slice::from_ref(&provider));

        let first = manager.acquire(&provider).unwrap();
        manager.record_result(&provider, &first.api_key, Some("HTTP 429 rate limit"));
        let second = manager.acquire(&provider).unwrap();
        let snapshot = manager.snapshot_for_provider(&provider);

        assert_eq!(first.api_key, "sk-primary");
        assert_eq!(second.api_key, "sk-shared");
        assert_eq!(snapshot[0].status, ProviderKeyHealthStatus::CoolingDown);
        assert_eq!(snapshot[1].status, ProviderKeyHealthStatus::Available);
    }

    #[tokio::test]
    async fn provider_key_pool_is_safe_under_concurrent_acquire() {
        let provider = Arc::new(ResolvedProviderConfig {
            provider: LlmProvider::Openai,
            model: "gpt-4o-mini".into(),
            api_key: "sk-primary".into(),
            provider_keys: vec![
                ProviderKeyConfig {
                    key: "sk-primary".into(),
                    priority: 1,
                    label: "primary".into(),
                },
                ProviderKeyConfig {
                    key: "sk-shared".into(),
                    priority: 2,
                    label: "shared".into(),
                },
            ],
            key_rotation_strategy: KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            endpoint: "https://api.openai.com/v1/chat/completions".into(),
            azure_deployment_id: String::new(),
            azure_api_version: "2024-08-01-preview".into(),
            active: true,
        });
        let manager = Arc::new(ProviderKeyPoolManager::from_provider_chain(&[
            (*provider).clone()
        ]));

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let manager = manager.clone();
            let provider = provider.clone();
            tasks.push(tokio::spawn(async move {
                manager.acquire(&provider).unwrap().api_key
            }));
        }

        let mut counts = HashMap::new();
        for task in tasks {
            let key = task.await.unwrap();
            *counts.entry(key).or_insert(0usize) += 1;
        }

        assert_eq!(counts.get("sk-primary"), Some(&4));
        assert_eq!(counts.get("sk-shared"), Some(&4));
    }

    #[tokio::test]
    async fn app_state_new_galadriel_provider() {
        let state = AppState::new(make_test_config("galadriel"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "galadriel");
    }

    #[tokio::test]
    async fn app_state_new_hyperbolic_provider() {
        let state = AppState::new(make_test_config("hyperbolic"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "hyperbolic");
    }

    #[tokio::test]
    async fn app_state_new_huggingface_provider() {
        let state = AppState::new(make_test_config("huggingface"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "huggingface");
    }

    #[tokio::test]
    async fn app_state_new_mira_provider() {
        let state = AppState::new(make_test_config("mira"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "mira");
    }

    #[tokio::test]
    async fn app_state_new_moonshot_provider() {
        let state = AppState::new(make_test_config("moonshot"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "moonshot");
    }

    #[tokio::test]
    async fn app_state_new_ollama_provider() {
        let state = AppState::new(make_test_config("ollama"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "ollama");
    }

    #[tokio::test]
    async fn app_state_new_openrouter_provider() {
        let state = AppState::new(make_test_config("openrouter"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "openrouter");
    }

    #[tokio::test]
    async fn app_state_new_perplexity_provider() {
        let state = AppState::new(make_test_config("perplexity"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "perplexity");
    }

    #[tokio::test]
    async fn app_state_new_together_provider() {
        let state = AppState::new(make_test_config("together"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "together");
    }

    #[tokio::test]
    async fn app_state_new_copilot_provider() {
        let state = AppState::new(make_test_config("copilot"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "copilot");
    }

    #[tokio::test]
    async fn app_state_caches_are_initialised() {
        let state = AppState::new(make_test_config("openai"), shared_test_embedder());
        // Verify caches and clients are properly initialised.
        assert!(Arc::strong_count(&state.exact_cache) >= 1);
        assert!(Arc::strong_count(&state.vector_cache) >= 1);
        assert!(Arc::strong_count(&state.slm_client) >= 1);
    }
}
