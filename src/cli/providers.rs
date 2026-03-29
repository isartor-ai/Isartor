use clap::Parser;

use crate::config::AppConfig;
use crate::models::{ProviderHealthStatus, ProviderStatusResponse};
use crate::state::ProviderHealthTracker;

#[derive(Parser, Debug, Clone)]
pub struct ProvidersArgs {
    /// Isartor gateway URL (default: http://localhost:8080)
    #[arg(long, default_value = "http://localhost:8080")]
    pub gateway_url: String,

    /// Gateway API key (optional). If omitted, the local config is used when available.
    #[arg(long, env = "ISARTOR__GATEWAY_API_KEY")]
    pub gateway_api_key: Option<String>,

    /// Output as JSON instead of human-readable text.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

pub async fn handle_providers(args: ProvidersArgs) -> anyhow::Result<()> {
    let gateway = args.gateway_url.trim_end_matches('/').to_string();
    let gateway_api_key = effective_gateway_api_key(args.gateway_api_key.as_deref());

    let (status, source) = if let Some(status) =
        fetch_provider_status(&gateway, gateway_api_key.clone()).await
    {
        (status, ProviderStatusSource::Gateway)
    } else if let Ok(config) = AppConfig::load() {
        (
            ProviderHealthTracker::from_config(&config).snapshot(),
            ProviderStatusSource::LocalConfig,
        )
    } else {
        anyhow::bail!(
            "Unable to read provider status from the gateway or local config. Provide --gateway-api-key if needed."
        );
    };

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "source": source.as_str(),
                "status": status,
            }))
            .unwrap_or_default()
        );
        return Ok(());
    }

    print!("{}", render_provider_report(&gateway, &status, source));
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderStatusSource {
    Gateway,
    LocalConfig,
}

impl ProviderStatusSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Gateway => "gateway",
            Self::LocalConfig => "local_config",
        }
    }
}

async fn fetch_provider_status(
    gateway: &str,
    gateway_api_key: Option<String>,
) -> Option<ProviderStatusResponse> {
    let client = reqwest::Client::new();
    let mut req = client
        .get(format!("{}/debug/providers", gateway))
        .timeout(std::time::Duration::from_secs(2));
    if let Some(key) = gateway_api_key {
        req = req.header("X-API-Key", key);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<ProviderStatusResponse>().await.ok()
}

fn effective_gateway_api_key(cli_value: Option<&str>) -> Option<String> {
    if let Some(value) = cli_value {
        return Some(value.to_string());
    }
    AppConfig::load().ok().map(|cfg| cfg.gateway_api_key)
}

fn render_provider_report(
    gateway: &str,
    status: &ProviderStatusResponse,
    source: ProviderStatusSource,
) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    writeln!(&mut output, "\nProvider Status").ok();
    writeln!(&mut output, "  Gateway: {}", gateway).ok();
    writeln!(&mut output, "  Source:  {}", source.as_str()).ok();
    writeln!(&mut output, "  Active:  {}", status.active_provider).ok();

    if source == ProviderStatusSource::LocalConfig {
        writeln!(
            &mut output,
            "  Note:    Gateway status endpoint unavailable; showing local config only."
        )
        .ok();
    }

    writeln!(&mut output).ok();
    writeln!(
        &mut output,
        "  {:<14} {:<9} {:<22} {:>8} {:>8}",
        "Provider", "Status", "Model", "Reqs", "Errors"
    )
    .ok();

    for entry in &status.providers {
        writeln!(
            &mut output,
            "  {:<14} {:<9} {:<22} {:>8} {:>8}",
            entry.name,
            status_label(entry.status),
            truncate_cell(&entry.model, 22),
            entry.requests_total,
            entry.errors_total
        )
        .ok();
        writeln!(&mut output, "    Endpoint:   {}", entry.endpoint).ok();
        writeln!(
            &mut output,
            "    Configured: api key={}, endpoint={}",
            yes_no(entry.api_key_configured),
            yes_no(entry.endpoint_configured)
        )
        .ok();
        writeln!(
            &mut output,
            "    Last ok:    {}",
            entry.last_success.as_deref().unwrap_or("never")
        )
        .ok();
        writeln!(
            &mut output,
            "    Last error: {}",
            entry.last_error.as_deref().unwrap_or("never")
        )
        .ok();
        if let Some(message) = &entry.last_error_message {
            writeln!(&mut output, "    Error msg:  {}", message).ok();
        }
    }

    writeln!(&mut output).ok();
    output
}

fn status_label(status: ProviderHealthStatus) -> &'static str {
    match status {
        ProviderHealthStatus::Healthy => "healthy",
        ProviderHealthStatus::Failing => "failing",
        ProviderHealthStatus::Unknown => "unknown",
    }
}

fn truncate_cell(value: &str, max_len: usize) -> String {
    if value.chars().count() <= max_len {
        return value.to_string();
    }
    value
        .chars()
        .take(max_len.saturating_sub(3))
        .collect::<String>()
        + "..."
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::get};
    use tokio::net::TcpListener;

    #[test]
    fn render_provider_report_includes_status_fields() {
        let report = render_provider_report(
            "http://localhost:8080",
            &ProviderStatusResponse {
                active_provider: "azure".into(),
                providers: vec![crate::models::ProviderStatusEntry {
                    name: "azure".into(),
                    active: true,
                    status: ProviderHealthStatus::Healthy,
                    model: "gpt-4o-mini".into(),
                    endpoint: "https://example.invalid".into(),
                    api_key_configured: true,
                    endpoint_configured: true,
                    requests_total: 142,
                    errors_total: 2,
                    last_success: Some("2026-03-29T12:00:00Z".into()),
                    last_error: None,
                    last_error_message: None,
                }],
            },
            ProviderStatusSource::Gateway,
        );

        assert!(report.contains("Provider Status"));
        assert!(report.contains("healthy"));
        assert!(report.contains("gpt-4o-mini"));
        assert!(report.contains("142"));
        assert!(report.contains("Endpoint"));
    }

    #[tokio::test]
    async fn fetch_provider_status_reads_debug_endpoint() {
        let app = Router::new().route(
            "/debug/providers",
            get(|| async {
                Json(ProviderStatusResponse {
                    active_provider: "copilot".into(),
                    providers: vec![crate::models::ProviderStatusEntry {
                        name: "copilot".into(),
                        active: true,
                        status: ProviderHealthStatus::Failing,
                        model: "gpt-4o-mini".into(),
                        endpoint: "https://api.githubcopilot.com/chat/completions".into(),
                        api_key_configured: true,
                        endpoint_configured: true,
                        requests_total: 3,
                        errors_total: 1,
                        last_success: Some("2026-03-29T12:00:00Z".into()),
                        last_error: Some("2026-03-29T12:05:00Z".into()),
                        last_error_message: Some("provider timeout".into()),
                    }],
                })
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let snapshot = fetch_provider_status(&format!("http://{}", addr), None)
            .await
            .unwrap();
        assert_eq!(snapshot.active_provider, "copilot");
        assert_eq!(snapshot.providers[0].requests_total, 3);
    }
}
