use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::state::AppLlmAgent;

pub const GITHUB_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
pub const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
pub const GITHUB_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
pub const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
pub const COPILOT_COMPLETIONS_URL: &str = "https://api.githubcopilot.com/chat/completions";
const USER_AGENT: &str = "GitHubCopilotChat/0.29.1";
const EDITOR_VERSION: &str = "vscode/1.99.0";
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 16_000;
const COPILOT_MAX_OUTPUT_TOKENS: u32 = 16_384;

#[derive(Debug, Clone)]
pub struct DeviceFlowResult {
    pub github_token: String,
    pub token_type: String,
    pub scope: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: u64,
    #[serde(default = "default_device_interval")]
    pub interval: u64,
}

fn default_device_interval() -> u64 {
    5
}

#[derive(Debug, Clone)]
pub struct CopilotAgent {
    http: Client,
    github_token: String,
    model: String,
    request_timeout: Duration,
}

impl CopilotAgent {
    pub fn new(
        http: Client,
        github_token: String,
        model: String,
        request_timeout: Duration,
    ) -> Self {
        Self {
            http,
            github_token,
            model,
            request_timeout,
        }
    }

    pub async fn device_flow_auth(http: &Client) -> anyhow::Result<DeviceFlowResult> {
        let resp = http
            .post(GITHUB_DEVICE_CODE_URL)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&json!({
                "client_id": GITHUB_CLIENT_ID,
                "scope": "read:user"
            }))
            .send()
            .await
            .context("device code request failed")?;

        if !resp.status().is_success() {
            bail!("device code request returned HTTP {}", resp.status());
        }

        let device_code: DeviceCodeResponse = resp
            .json()
            .await
            .context("failed to parse device code response")?;

        println!();
        println!("GitHub Copilot authentication");
        println!("────────────────────────────");
        println!("1. Open: {}", device_code.verification_uri);
        println!("2. Enter code: {}", device_code.user_code);
        println!();
        println!(
            "Waiting for authentication (expires in {}s)...",
            device_code.expires_in
        );

        let poll_interval = Duration::from_secs(device_code.interval.max(5));
        let deadline = Instant::now() + Duration::from_secs(device_code.expires_in);

        loop {
            if Instant::now() > deadline {
                bail!("authentication timed out; please try again");
            }

            sleep(poll_interval).await;

            let response = http
                .post(GITHUB_TOKEN_URL)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .json(&json!({
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": device_code.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
                }))
                .send()
                .await;

            let Ok(resp) = response else {
                continue;
            };

            if !resp.status().is_success() {
                continue;
            }

            #[derive(Debug, Deserialize)]
            struct TokenResp {
                access_token: Option<String>,
                error: Option<String>,
                token_type: Option<String>,
                scope: Option<String>,
            }

            let body: TokenResp = resp
                .json()
                .await
                .context("failed to parse device flow token response")?;

            if let Some(err) = body.error.as_deref() {
                if err == "authorization_pending" || err == "slow_down" {
                    print!(".");
                    let _ = std::io::stdout().flush();
                    continue;
                }
                bail!("authentication failed: {err}");
            }

            if let Some(token) = body.access_token {
                println!();
                return Ok(DeviceFlowResult {
                    github_token: token,
                    token_type: body.token_type.unwrap_or_else(|| "bearer".to_string()),
                    scope: body.scope.unwrap_or_default(),
                });
            }
        }
    }

    pub async fn validate_github_token(http: &Client, github_token: &str) -> anyhow::Result<()> {
        let _ = exchange_copilot_session_token(http, github_token).await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl AppLlmAgent for CopilotAgent {
    async fn chat(&self, prompt: &str) -> anyhow::Result<String> {
        let copilot_token = exchange_copilot_session_token(&self.http, &self.github_token).await?;
        let body = build_completion_body(&self.model, prompt, DEFAULT_MAX_OUTPUT_TOKENS);

        let response = self
            .http
            .post(COPILOT_COMPLETIONS_URL)
            .header("Authorization", format!("Bearer {copilot_token}"))
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .header("User-Agent", USER_AGENT)
            .header("Editor-Version", EDITOR_VERSION)
            .header("Editor-Plugin-Version", "copilot-chat/0.29.1")
            .header("Copilot-Integration-Id", "vscode-chat")
            .header("X-GitHub-Api-Version", "2025-04-01")
            .json(&body)
            .send()
            .await
            .with_context(|| {
                format!(
                    "Copilot completions request failed after {:?}",
                    self.request_timeout
                )
            })?;

        let status = response.status();
        let payload: Value = response
            .json()
            .await
            .context("failed to parse Copilot completions response")?;

        if !status.is_success() {
            bail!("Copilot completions returned HTTP {status}: {}", payload);
        }

        extract_completion_text(&payload).ok_or_else(|| {
            anyhow!(
                "Copilot completions response did not contain assistant text: {}",
                payload
            )
        })
    }

    fn provider_name(&self) -> &'static str {
        "copilot"
    }
}

pub async fn exchange_copilot_session_token(
    http: &Client,
    github_token: &str,
) -> anyhow::Result<String> {
    let response = http
        .get(COPILOT_TOKEN_URL)
        .header("Authorization", format!("token {github_token}"))
        .header("Accept", "application/json")
        .header("User-Agent", USER_AGENT)
        .send()
        .await
        .context("Copilot session token request failed")?;

    let status = response.status();
    let payload: Value = response
        .json()
        .await
        .context("failed to parse Copilot session token response")?;

    if !status.is_success() {
        if status.as_u16() == 404 {
            bail!("no active GitHub Copilot subscription found");
        }
        if status.as_u16() == 401 {
            bail!("invalid GitHub token");
        }
        bail!(
            "Copilot session token request returned HTTP {status}: {}",
            payload
        );
    }

    payload
        .get("token")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            anyhow!(
                "Copilot session token response missing or empty `token`: {}",
                payload
            )
        })
}

pub fn build_completion_body(model: &str, prompt: &str, max_tokens: u32) -> Value {
    let model = model.trim_start_matches("github_copilot/");
    json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": false,
        "temperature": 0.2,
        "max_tokens": max_tokens.min(COPILOT_MAX_OUTPUT_TOKENS),
    })
}

fn extract_completion_text(payload: &Value) -> Option<String> {
    if let Some(content) = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
    {
        return extract_content_field(content);
    }

    payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("text"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn extract_content_field(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() { None } else { Some(text) }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_completion_body_caps_max_tokens() {
        let body = build_completion_body("github_copilot/gpt-4o", "hello", 100_000);
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["max_tokens"], 16_384);
    }

    #[test]
    fn extract_completion_text_from_string_content() {
        let payload = json!({
            "choices": [{
                "message": {
                    "content": "hello"
                }
            }]
        });
        assert_eq!(extract_completion_text(&payload).as_deref(), Some("hello"));
    }

    #[test]
    fn extract_completion_text_from_array_content() {
        let payload = json!({
            "choices": [{
                "message": {
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"}
                    ]
                }
            }]
        });
        assert_eq!(
            extract_completion_text(&payload).as_deref(),
            Some("hello\nworld")
        );
    }
}
