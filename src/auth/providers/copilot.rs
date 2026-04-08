//! GitHub Copilot OAuth provider (GitHub device flow).
//!
//! Wraps the existing device-flow implementation in `src/providers/copilot.rs`
//! in the shared [`OAuthProvider`] trait.

use anyhow::Context;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::auth::{AuthMethod, DeviceFlowState, OAuthProvider, PollResult, StoredToken};
use crate::providers::copilot::{GITHUB_CLIENT_ID, GITHUB_DEVICE_CODE_URL, GITHUB_TOKEN_URL};

pub struct CopilotOAuth;

#[async_trait]
impl OAuthProvider for CopilotOAuth {
    fn provider_name(&self) -> &str {
        "copilot"
    }

    fn display_name(&self) -> &str {
        "GitHub Copilot"
    }

    fn auth_method(&self) -> AuthMethod {
        AuthMethod::DeviceFlow
    }

    async fn start_device_flow(&self, http: &Client) -> anyhow::Result<DeviceFlowState> {
        #[derive(Deserialize)]
        struct DeviceCodeResp {
            device_code: String,
            user_code: String,
            verification_uri: String,
            expires_in: u64,
            #[serde(default = "default_interval")]
            interval: u64,
        }

        fn default_interval() -> u64 {
            5
        }

        let resp = http
            .post(GITHUB_DEVICE_CODE_URL)
            .header("Accept", "application/json")
            .json(&json!({
                "client_id": GITHUB_CLIENT_ID,
                "scope": "read:user"
            }))
            .send()
            .await
            .context("device code request failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("device code request returned HTTP {}", resp.status());
        }

        let dc: DeviceCodeResp = resp
            .json()
            .await
            .context("failed to parse device code response")?;

        Ok(DeviceFlowState {
            device_code: dc.device_code,
            user_code: Some(dc.user_code),
            verification_uri: Some(dc.verification_uri.clone()),
            verification_uri_complete: None,
            expires_in: dc.expires_in,
            interval: dc.interval,
        })
    }

    async fn poll_device_token(
        &self,
        http: &Client,
        state: &DeviceFlowState,
    ) -> anyhow::Result<PollResult> {
        #[derive(Deserialize)]
        struct TokenResp {
            access_token: Option<String>,
            error: Option<String>,
            token_type: Option<String>,
            scope: Option<String>,
        }

        let resp = http
            .post(GITHUB_TOKEN_URL)
            .header("Accept", "application/json")
            .json(&json!({
                "client_id": GITHUB_CLIENT_ID,
                "device_code": state.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
            }))
            .send()
            .await;

        let Ok(resp) = resp else {
            return Ok(PollResult::Pending);
        };
        if !resp.status().is_success() {
            return Ok(PollResult::Pending);
        }

        let body: TokenResp = match resp.json().await {
            Ok(b) => b,
            Err(_) => return Ok(PollResult::Pending),
        };

        if let Some(err) = body.error.as_deref() {
            if err == "authorization_pending" || err == "slow_down" {
                return Ok(PollResult::Pending);
            }
            return Ok(PollResult::Error(err.to_string()));
        }

        if let Some(token) = body.access_token {
            return Ok(PollResult::Ready(StoredToken {
                provider: "copilot".to_string(),
                access_token: token,
                refresh_token: None,
                // GitHub tokens don't carry expiry in the device flow response;
                // the Copilot session token exchange handles expiry internally.
                expires_at: None,
                token_type: body.token_type.unwrap_or_else(|| "Bearer".to_string()),
                scopes: body
                    .scope
                    .unwrap_or_default()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            }));
        }

        Ok(PollResult::Pending)
    }
}

/// Convenience: run the full Copilot device flow and block until authenticated.
pub async fn run_copilot_device_flow(http: &Client) -> anyhow::Result<StoredToken> {
    let provider = CopilotOAuth;
    let state = provider.start_device_flow(http).await?;
    crate::auth::device_flow::print_device_flow_instructions("GitHub Copilot", &state);
    crate::auth::device_flow::poll_until_ready(&provider, http, &state).await
}
