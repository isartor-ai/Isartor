use chrono::{DateTime, Datelike, Duration, NaiveTime, Utc, Weekday};

use crate::config::{AppConfig, ProviderQuotaConfig, QuotaLimitAction};
use crate::core::usage::{ProviderUsageTotals, UsageTracker};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuotaPeriod {
    Daily,
    Weekly,
    Monthly,
}

impl QuotaPeriod {
    fn label(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Weekly => "weekly",
            Self::Monthly => "monthly",
        }
    }

    fn window_start(self, now: DateTime<Utc>) -> DateTime<Utc> {
        let date = now.date_naive();
        let midnight = NaiveTime::MIN;
        match self {
            Self::Daily => date.and_time(midnight).and_utc(),
            Self::Weekly => {
                let days_from_monday = match date.weekday() {
                    Weekday::Mon => 0,
                    Weekday::Tue => 1,
                    Weekday::Wed => 2,
                    Weekday::Thu => 3,
                    Weekday::Fri => 4,
                    Weekday::Sat => 5,
                    Weekday::Sun => 6,
                };
                (date - Duration::days(days_from_monday))
                    .and_time(midnight)
                    .and_utc()
            }
            Self::Monthly => date
                .with_day(1)
                .expect("valid first day of month")
                .and_time(midnight)
                .and_utc(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderQuotaDecision {
    pub action: QuotaLimitAction,
    pub warning_messages: Vec<String>,
    pub status_lines: Vec<String>,
    pub limit_message: Option<String>,
}

pub fn evaluate_provider_quota(
    config: &AppConfig,
    usage_tracker: &UsageTracker,
    provider: &str,
    projected_total_tokens: u64,
    projected_cost_usd: f64,
    now: DateTime<Utc>,
) -> Option<ProviderQuotaDecision> {
    let quota = config.quota_for_provider(provider)?;
    Some(evaluate_with_quota(
        usage_tracker,
        provider,
        quota,
        projected_total_tokens,
        projected_cost_usd,
        now,
    ))
}

pub fn current_quota_status_lines(
    config: &AppConfig,
    usage_tracker: &UsageTracker,
    provider: &str,
    now: DateTime<Utc>,
) -> Option<(QuotaLimitAction, Vec<String>)> {
    let decision = evaluate_provider_quota(config, usage_tracker, provider, 0, 0.0, now)?;
    Some((decision.action, decision.status_lines))
}

fn evaluate_with_quota(
    usage_tracker: &UsageTracker,
    provider: &str,
    quota: &ProviderQuotaConfig,
    projected_total_tokens: u64,
    projected_cost_usd: f64,
    now: DateTime<Utc>,
) -> ProviderQuotaDecision {
    let mut state = QuotaEvaluationState {
        warning_messages: Vec::new(),
        status_lines: Vec::new(),
        breached: Vec::new(),
    };

    for (period, token_limit) in [
        (QuotaPeriod::Daily, quota.daily_token_limit),
        (QuotaPeriod::Weekly, quota.weekly_token_limit),
        (QuotaPeriod::Monthly, quota.monthly_token_limit),
    ] {
        let Some(limit) = token_limit else {
            continue;
        };
        let usage = usage_tracker.provider_usage_since(provider, period.window_start(now), now);
        evaluate_tokens_limit(
            &mut state,
            quota,
            period,
            usage.total_tokens,
            projected_total_tokens,
            limit,
        );
    }

    for (period, cost_limit) in [
        (QuotaPeriod::Daily, quota.daily_cost_limit_usd),
        (QuotaPeriod::Weekly, quota.weekly_cost_limit_usd),
        (QuotaPeriod::Monthly, quota.monthly_cost_limit_usd),
    ] {
        let Some(limit) = cost_limit else {
            continue;
        };
        let usage = usage_tracker.provider_usage_since(provider, period.window_start(now), now);
        evaluate_cost_limit(&mut state, quota, period, &usage, projected_cost_usd, limit);
    }

    let limit_message = if state.breached.is_empty() {
        None
    } else {
        Some(format!(
            "provider quota exceeded for {provider}: {}",
            state.breached.join("; ")
        ))
    };

    ProviderQuotaDecision {
        action: quota.action_on_limit.clone(),
        warning_messages: state.warning_messages,
        status_lines: state.status_lines,
        limit_message,
    }
}

struct QuotaEvaluationState {
    warning_messages: Vec<String>,
    status_lines: Vec<String>,
    breached: Vec<String>,
}

fn evaluate_tokens_limit(
    state: &mut QuotaEvaluationState,
    quota: &ProviderQuotaConfig,
    period: QuotaPeriod,
    used_tokens: u64,
    projected_total_tokens: u64,
    limit: u64,
) {
    let projected = used_tokens.saturating_add(projected_total_tokens);
    let ratio = projected as f64 / limit as f64;
    state.status_lines.push(format!(
        "{} tokens: {} / {} ({:.1}%)",
        period.label(),
        used_tokens,
        limit,
        (used_tokens as f64 / limit as f64) * 100.0
    ));
    if projected >= limit {
        state.breached.push(format!(
            "{} token limit {} / {} ({:.1}% projected)",
            period.label(),
            projected,
            limit,
            ratio * 100.0
        ));
    } else if ratio >= quota.warning_threshold_ratio {
        state.warning_messages.push(format!(
            "provider is approaching {} token quota: projected {} / {} ({:.1}%)",
            period.label(),
            projected,
            limit,
            ratio * 100.0
        ));
    }
}

fn evaluate_cost_limit(
    state: &mut QuotaEvaluationState,
    quota: &ProviderQuotaConfig,
    period: QuotaPeriod,
    usage: &ProviderUsageTotals,
    projected_cost_usd: f64,
    limit: f64,
) {
    let projected = usage.estimated_cost_usd + projected_cost_usd;
    let ratio = projected / limit;
    state.status_lines.push(format!(
        "{} cost: ${:.4} / ${:.4} ({:.1}%)",
        period.label(),
        usage.estimated_cost_usd,
        limit,
        (usage.estimated_cost_usd / limit) * 100.0
    ));
    if projected >= limit {
        state.breached.push(format!(
            "{} cost limit ${:.4} / ${:.4} ({:.1}% projected)",
            period.label(),
            projected,
            limit,
            ratio * 100.0
        ));
    } else if ratio >= quota.warning_threshold_ratio {
        state.warning_messages.push(format!(
            "provider is approaching {} cost quota: projected ${:.4} / ${:.4} ({:.1}%)",
            period.label(),
            projected,
            limit,
            ratio * 100.0
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use chrono::TimeZone;

    use super::*;
    use crate::config::{
        AppConfig, CacheBackend, CacheMode, ClassifierMode, EmbeddingSidecarSettings,
        FallbackProviderConfig, InferenceEngineMode, KeyRotationStrategy, Layer2Settings,
        ProviderPricingConfig, RouterBackend,
    };
    use crate::core::usage::UsageTracker;

    fn test_config() -> Arc<AppConfig> {
        let mut usage_pricing = HashMap::new();
        usage_pricing.insert(
            "openai".to_string(),
            ProviderPricingConfig {
                input_cost_per_million_usd: 1.0,
                output_cost_per_million_usd: 2.0,
            },
        );
        let mut quota = HashMap::new();
        quota.insert(
            "openai".to_string(),
            ProviderQuotaConfig {
                daily_token_limit: Some(1_000),
                monthly_cost_limit_usd: Some(1.0),
                action_on_limit: QuotaLimitAction::Fallback,
                warning_threshold_ratio: 0.8,
                ..ProviderQuotaConfig::default()
            },
        );

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
                sidecar_url: "http://127.0.0.1:8081".into(),
                model_name: "phi-3-mini".into(),
                timeout_seconds: 5,
                classifier_mode: ClassifierMode::Tiered,
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
            model_aliases: HashMap::new(),
            external_llm_api_key: "sk-test".into(),
            provider_keys: Vec::new(),
            key_rotation_strategy: KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            fallback_providers: Vec::<FallbackProviderConfig>::new(),
            l3_timeout_secs: 120,
            azure_deployment_id: "".into(),
            azure_api_version: "2024-08-01-preview".into(),
            enable_slm_router: false,
            enable_context_optimizer: true,
            context_optimizer_dedup: true,
            context_optimizer_minify: true,
            enable_monitoring: false,
            otel_exporter_endpoint: "http://localhost:4317".into(),
            enable_request_logs: false,
            request_log_path: "~/.isartor/request_logs".into(),
            usage_log_path: "~/.isartor".into(),
            usage_retention_days: 30,
            usage_window_hours: 24,
            usage_pricing,
            quota,
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
        })
    }

    #[test]
    fn warns_when_projected_usage_crosses_threshold() {
        let config = test_config();
        let tracker = UsageTracker::new(config.clone()).unwrap();
        let decision = evaluate_provider_quota(
            &config,
            &tracker,
            "openai",
            850,
            0.0,
            Utc.with_ymd_and_hms(2026, 3, 29, 12, 0, 0).unwrap(),
        )
        .unwrap();
        assert!(
            decision
                .warning_messages
                .iter()
                .any(|msg| msg.contains("approaching"))
        );
        assert!(decision.limit_message.is_none());
    }

    #[test]
    fn breaches_when_projected_usage_exceeds_limit() {
        let config = test_config();
        let tracker = UsageTracker::new(config.clone()).unwrap();
        let decision = evaluate_provider_quota(
            &config,
            &tracker,
            "openai",
            1_200,
            0.0,
            Utc.with_ymd_and_hms(2026, 3, 29, 12, 0, 0).unwrap(),
        )
        .unwrap();
        assert_eq!(decision.action, QuotaLimitAction::Fallback);
        assert!(
            decision
                .limit_message
                .as_deref()
                .is_some_and(|msg| msg.contains("token limit"))
        );
    }
}
