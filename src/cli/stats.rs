use clap::Parser;

use crate::config::AppConfig;
use crate::models::{AgentStatsResponse, PromptStatsResponse, UsageStatsResponse};

#[derive(Parser, Debug, Clone)]
pub struct StatsArgs {
    /// Isartor gateway URL (default: http://localhost:8080)
    #[arg(long, default_value = "http://localhost:8080")]
    pub gateway_url: String,

    /// Gateway API key (optional). If omitted, stats will try the locally loaded config.
    #[arg(long, env = "ISARTOR__GATEWAY_API_KEY")]
    pub gateway_api_key: Option<String>,

    /// Number of recent prompts to show.
    #[arg(long, default_value_t = 10)]
    pub recent_limit: usize,

    /// Show per-tool breakdown.
    #[arg(long, default_value_t = false)]
    pub by_tool: bool,

    /// Show provider/model usage analytics instead of prompt stats.
    #[arg(long, default_value_t = false)]
    pub usage: bool,

    /// Usage analytics window in hours.
    #[arg(long, default_value_t = 24)]
    pub usage_hours: u64,

    /// Output as JSON instead of human-readable text.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

pub async fn handle_stats(args: StatsArgs) -> anyhow::Result<()> {
    let gateway = args.gateway_url.trim_end_matches('/').to_string();
    let Some(health) = fetch_health(&gateway).await else {
        anyhow::bail!("Isartor is not reachable at {}", gateway);
    };
    let gateway_api_key = effective_gateway_api_key(args.gateway_api_key.as_deref());

    if args.usage {
        let Some(usage) = fetch_usage_stats(&gateway, gateway_api_key, args.usage_hours).await
        else {
            anyhow::bail!("Unable to read usage stats. Provide --gateway-api-key if needed.");
        };
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&usage).unwrap_or_default()
            );
            return Ok(());
        }
        print!("{}", render_usage_report(&gateway, &health.version, &usage));
        return Ok(());
    }

    let Some(stats) =
        fetch_prompt_stats(&gateway, gateway_api_key.clone(), args.recent_limit).await
    else {
        anyhow::bail!("Unable to read prompt stats. Provide --gateway-api-key if needed.");
    };
    let agent_stats = if args.by_tool {
        fetch_agent_stats(&gateway, gateway_api_key.clone()).await
    } else {
        None
    };

    if args.json {
        if args.by_tool {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "prompts": stats,
                    "agents": agent_stats.unwrap_or_default(),
                }))
                .unwrap_or_default()
            );
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(&stats).unwrap_or_default()
            );
        }
        return Ok(());
    }

    print!(
        "{}",
        render_stats_report(
            &gateway,
            &health.version,
            &stats,
            agent_stats.as_ref(),
            args.by_tool,
        )
    );

    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct Health {
    version: String,
}

async fn fetch_health(gateway: &str) -> Option<Health> {
    let client = reqwest::Client::new();
    client
        .get(format!("{}/health", gateway))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .ok()?
        .json::<Health>()
        .await
        .ok()
}

async fn fetch_prompt_stats(
    gateway: &str,
    gateway_api_key: Option<String>,
    limit: usize,
) -> Option<PromptStatsResponse> {
    let client = reqwest::Client::new();
    let mut req = client
        .get(format!("{}/debug/stats/prompts?limit={}", gateway, limit))
        .timeout(std::time::Duration::from_secs(2));
    if let Some(key) = gateway_api_key {
        req = req.header("X-API-Key", key);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<PromptStatsResponse>().await.ok()
}

async fn fetch_agent_stats(
    gateway: &str,
    gateway_api_key: Option<String>,
) -> Option<AgentStatsResponse> {
    let client = reqwest::Client::new();
    let mut req = client
        .get(format!("{}/debug/stats/agents", gateway))
        .timeout(std::time::Duration::from_secs(2));
    if let Some(key) = gateway_api_key {
        req = req.header("X-API-Key", key);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<AgentStatsResponse>().await.ok()
}

async fn fetch_usage_stats(
    gateway: &str,
    gateway_api_key: Option<String>,
    hours: u64,
) -> Option<UsageStatsResponse> {
    let client = reqwest::Client::new();
    let mut req = client
        .get(format!("{}/debug/usage?hours={}", gateway, hours))
        .timeout(std::time::Duration::from_secs(2));
    if let Some(key) = gateway_api_key {
        req = req.header("X-API-Key", key);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<UsageStatsResponse>().await.ok()
}

fn effective_gateway_api_key(cli_value: Option<&str>) -> Option<String> {
    if let Some(value) = cli_value {
        return Some(value.to_string());
    }
    AppConfig::load().ok().map(|cfg| cfg.gateway_api_key)
}

fn render_stats_report(
    gateway: &str,
    version: &str,
    stats: &PromptStatsResponse,
    agent_stats: Option<&AgentStatsResponse>,
    by_tool_requested: bool,
) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    writeln!(&mut output, "\nIsartor Prompt Stats").ok();
    writeln!(&mut output, "  URL:        {}", gateway).ok();
    writeln!(&mut output, "  Version:    {}", version).ok();
    writeln!(&mut output, "  Total:      {}", stats.total_prompts).ok();
    writeln!(
        &mut output,
        "  Deflected:  {}",
        stats.total_deflected_prompts
    )
    .ok();
    writeln!(
        &mut output,
        "  Cloud:      {}",
        stats.by_layer.get("l3").copied().unwrap_or(0)
    )
    .ok();

    writeln!(&mut output, "\nBy Layer").ok();
    let known_layers = ["l0", "l1a", "l1b", "l2", "l3"];
    for layer in known_layers {
        writeln!(
            &mut output,
            "  {:<3} {}",
            layer.to_uppercase(),
            stats.by_layer.get(layer).copied().unwrap_or(0)
        )
        .ok();
    }
    for (layer, count) in &stats.by_layer {
        if known_layers.contains(&layer.as_str()) {
            continue;
        }
        writeln!(&mut output, "  {:<3} {}", layer.to_uppercase(), count).ok();
    }

    writeln!(&mut output, "\nBy Surface").ok();
    for (surface, count) in &stats.by_surface {
        writeln!(&mut output, "  {:<10} {}", surface, count).ok();
    }

    writeln!(&mut output, "\nBy Client").ok();
    for (client, count) in &stats.by_client {
        writeln!(&mut output, "  {:<10} {}", client, count).ok();
    }

    if by_tool_requested || !stats.by_tool.is_empty() {
        writeln!(&mut output, "\nBy Tool").ok();
        if let Some(agent_stats) = agent_stats {
            if agent_stats.agents.is_empty() {
                writeln!(&mut output, "  No tool-level traffic recorded yet.").ok();
            } else {
                writeln!(
                    &mut output,
                    "  {:<14} {:>8} {:>6} {:>7} {:>8} {:>8} {:>7} {:>11} {:>11}",
                    "Tool",
                    "Reqs",
                    "Hits",
                    "Misses",
                    "Avg ms",
                    "Retries",
                    "Errors",
                    "L1a H/M",
                    "L1b H/M",
                )
                .ok();
                for (tool, entry) in &agent_stats.agents {
                    writeln!(
                        &mut output,
                        "  {:<14} {:>8} {:>6} {:>7} {:>8.1} {:>8} {:>7} {:>5}/{:<5} {:>5}/{:<5}",
                        tool,
                        entry.requests,
                        entry.cache_hits,
                        entry.cache_misses,
                        entry.average_latency_ms,
                        entry.retry_count,
                        entry.error_count,
                        entry.l1a_hits,
                        entry.l1a_misses,
                        entry.l1b_hits,
                        entry.l1b_misses,
                    )
                    .ok();
                }
            }
        } else if stats.by_tool.is_empty() {
            writeln!(&mut output, "  No tool-level traffic recorded yet.").ok();
        } else {
            for (tool, count) in &stats.by_tool {
                writeln!(&mut output, "  {:<14} {}", tool, count).ok();
            }
        }
    }

    writeln!(&mut output, "\nRecent Prompts").ok();
    if stats.recent.is_empty() {
        writeln!(&mut output, "  No prompt traffic recorded yet.").ok();
    } else {
        for entry in &stats.recent {
            let tool_tag = if entry.tool.is_empty() {
                String::new()
            } else {
                format!(" [{}]", entry.tool)
            };
            writeln!(
                &mut output,
                "  {} {} {}{} {} via {} ({} ms, HTTP {})",
                entry.timestamp,
                entry.traffic_surface,
                entry.client,
                tool_tag,
                entry.final_layer.to_uppercase(),
                entry.route,
                entry.latency_ms,
                entry.status_code
            )
            .ok();
        }
    }
    writeln!(&mut output).ok();
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::PromptVisibilityEntry;
    use axum::{Json, Router, routing::get};
    use tokio::net::TcpListener;

    #[test]
    fn render_stats_report_includes_rich_agent_table() {
        let mut by_layer = std::collections::BTreeMap::new();
        by_layer.insert("l1a".to_string(), 1);
        let mut by_surface = std::collections::BTreeMap::new();
        by_surface.insert("gateway".to_string(), 1);
        let mut by_client = std::collections::BTreeMap::new();
        by_client.insert("direct".to_string(), 1);
        let mut by_tool = std::collections::BTreeMap::new();
        by_tool.insert("cursor".to_string(), 1);

        let report = render_stats_report(
            "http://localhost:8080",
            "test",
            &PromptStatsResponse {
                total_prompts: 1,
                total_deflected_prompts: 1,
                by_layer,
                by_surface,
                by_client,
                by_tool,
                recent: vec![PromptVisibilityEntry {
                    timestamp: "2026-01-01T00:00:00Z".into(),
                    traffic_surface: "gateway".into(),
                    client: "direct".into(),
                    endpoint_family: "native".into(),
                    route: "/api/chat".into(),
                    prompt_hash: None,
                    final_layer: "l1a".into(),
                    resolved_by: None,
                    deflected: true,
                    latency_ms: 12,
                    status_code: 200,
                    tool: "cursor".into(),
                }],
            },
            Some(&AgentStatsResponse {
                agents: std::collections::BTreeMap::from([(
                    "cursor".to_string(),
                    crate::models::AgentStatsEntry {
                        requests: 1,
                        cache_hits: 1,
                        cache_misses: 0,
                        l1a_hits: 1,
                        l1a_misses: 0,
                        l1b_hits: 0,
                        l1b_misses: 0,
                        average_latency_ms: 12.0,
                        retry_count: 2,
                        error_count: 1,
                    },
                )]),
            }),
            true,
        );

        assert!(report.contains("By Tool"));
        assert!(report.contains("Retries"));
        assert!(report.contains("L1a H/M"));
        assert!(report.contains("cursor"));
        assert!(report.contains("[cursor]"));
    }

    #[tokio::test]
    async fn fetch_agent_stats_reads_debug_endpoint() {
        let app = Router::new()
            .route(
                "/debug/stats/agents",
                get(|| async {
                    Json(AgentStatsResponse {
                        agents: std::collections::BTreeMap::from([(
                            "copilot".to_string(),
                            crate::models::AgentStatsEntry {
                                requests: 2,
                                cache_hits: 1,
                                cache_misses: 1,
                                l1a_hits: 1,
                                l1a_misses: 1,
                                l1b_hits: 0,
                                l1b_misses: 1,
                                average_latency_ms: 22.5,
                                retry_count: 1,
                                error_count: 0,
                            },
                        )]),
                    })
                }),
            )
            .route(
                "/health",
                get(|| async { Json(serde_json::json!({ "version": "test" })) }),
            )
            .route(
                "/debug/stats/prompts",
                get(|| async { Json(PromptStatsResponse::default()) }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let stats = fetch_agent_stats(&format!("http://{}", addr), None)
            .await
            .unwrap();
        assert_eq!(stats.agents.get("copilot").unwrap().requests, 2);
    }
}

type UsageAggregate = (u64, u64, u64, f64, f64);

fn render_usage_report(gateway: &str, version: &str, usage: &UsageStatsResponse) -> String {
    use std::collections::BTreeMap;
    use std::fmt::Write;

    let mut output = String::new();
    writeln!(&mut output, "\nIsartor Usage Stats").ok();
    writeln!(&mut output, "  URL:             {}", gateway).ok();
    writeln!(&mut output, "  Version:         {}", version).ok();
    writeln!(
        &mut output,
        "  Window:          last {}h",
        usage.window_hours
    )
    .ok();
    writeln!(&mut output, "  Requests:        {}", usage.total_requests).ok();
    writeln!(
        &mut output,
        "  Deflected:       {}",
        usage.total_deflected_requests
    )
    .ok();
    writeln!(
        &mut output,
        "  Deflection rate: {:.1}%",
        usage.deflection_rate * 100.0
    )
    .ok();
    writeln!(
        &mut output,
        "  Prompt tokens:   {}",
        usage.total_prompt_tokens
    )
    .ok();
    writeln!(
        &mut output,
        "  Completion:      {}",
        usage.total_completion_tokens
    )
    .ok();
    writeln!(&mut output, "  Total tokens:    {}", usage.total_tokens).ok();
    writeln!(
        &mut output,
        "  Est. spend:      ${:.4}",
        usage.estimated_cost_usd
    )
    .ok();
    writeln!(
        &mut output,
        "  Est. saved:      ${:.4}",
        usage.estimated_saved_cost_usd
    )
    .ok();
    writeln!(&mut output, "  Log:             {}", usage.usage_log_path).ok();

    let mut grouped: BTreeMap<(String, String), UsageAggregate> = BTreeMap::new();
    for entry in &usage.entries {
        let aggregate = grouped
            .entry((entry.provider.clone(), entry.model.clone()))
            .or_insert((0, 0, 0, 0.0, 0.0));
        aggregate.0 = aggregate.0.saturating_add(entry.requests_total);
        aggregate.1 = aggregate.1.saturating_add(entry.deflected_total);
        aggregate.2 = aggregate.2.saturating_add(entry.total_tokens);
        aggregate.3 += entry.estimated_cost_usd;
        aggregate.4 += entry.estimated_saved_cost_usd;
    }

    writeln!(&mut output, "By Provider / Model").ok();
    if grouped.is_empty() {
        writeln!(&mut output, "  No usage recorded yet.").ok();
    } else {
        writeln!(
            &mut output,
            "  {:<12} {:<28} {:>7} {:>9} {:>12} {:>12} {:>12}",
            "Provider", "Model", "Reqs", "Saved", "Tokens", "Spend $", "Saved $"
        )
        .ok();
        for ((provider, model), (requests, deflected, tokens, spend, saved)) in grouped {
            writeln!(
                &mut output,
                "  {:<12} {:<28} {:>7} {:>9} {:>12} {:>12.4} {:>12.4}",
                provider, model, requests, deflected, tokens, spend, saved
            )
            .ok();
        }
    }

    writeln!(&mut output, "\nDaily Breakdown").ok();
    if usage.entries.is_empty() {
        writeln!(&mut output, "  No daily entries recorded yet.").ok();
    } else {
        for entry in &usage.entries {
            writeln!(
                &mut output,
                "  {}  {} / {}  req={} saved={} tokens={} spend=${:.4} saved=${:.4}",
                entry.day,
                entry.provider,
                entry.model,
                entry.requests_total,
                entry.deflected_total,
                entry.total_tokens,
                entry.estimated_cost_usd,
                entry.estimated_saved_cost_usd
            )
            .ok();
        }
    }

    writeln!(&mut output).ok();
    output
}

#[cfg(test)]
mod usage_tests_extra {
    use super::*;
    use crate::models::UsageStatsEntry;
    use axum::{Json, Router, routing::get};
    use tokio::net::TcpListener;

    #[test]
    fn render_usage_report_includes_costs() {
        let report = render_usage_report(
            "http://localhost:8080",
            "test",
            &UsageStatsResponse {
                window_hours: 24,
                usage_log_path: "~/.isartor/usage.jsonl".into(),
                retention_days: 30,
                total_requests: 2,
                total_deflected_requests: 1,
                total_prompt_tokens: 100,
                total_completion_tokens: 50,
                total_tokens: 150,
                estimated_cost_usd: 0.12,
                estimated_saved_cost_usd: 0.03,
                deflection_rate: 1.0 / 3.0,
                entries: vec![UsageStatsEntry {
                    day: "2026-01-01".into(),
                    provider: "openai".into(),
                    model: "gpt-4o-mini".into(),
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    total_tokens: 150,
                    requests_total: 2,
                    deflected_total: 1,
                    estimated_cost_usd: 0.12,
                    estimated_saved_cost_usd: 0.03,
                }],
            },
        );
        assert!(report.contains("Est. spend"));
        assert!(report.contains("gpt-4o-mini"));
        assert!(report.contains("Saved $"));
    }

    #[tokio::test]
    async fn fetch_usage_stats_reads_debug_endpoint() {
        let app = Router::new()
            .route(
                "/debug/usage",
                get(|| async {
                    Json(UsageStatsResponse {
                        window_hours: 24,
                        usage_log_path: "test".into(),
                        retention_days: 30,
                        total_requests: 1,
                        total_deflected_requests: 0,
                        total_prompt_tokens: 10,
                        total_completion_tokens: 5,
                        total_tokens: 15,
                        estimated_cost_usd: 0.01,
                        estimated_saved_cost_usd: 0.0,
                        deflection_rate: 0.0,
                        entries: vec![],
                    })
                }),
            )
            .route(
                "/health",
                get(|| async { Json(serde_json::json!({ "version": "test" })) }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let stats = fetch_usage_stats(&format!("http://{}", addr), None, 24)
            .await
            .unwrap();
        assert_eq!(stats.total_requests, 1);
    }
}
