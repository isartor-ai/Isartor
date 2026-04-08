//! Google Gemini OAuth2 device-flow provider.
//!
//! Uses the Google Cloud SDK OAuth2 client to obtain credentials for the
//! `https://www.googleapis.com/auth/generative-language` scope.
//!
//! The device-code endpoint and token endpoint require no secret — the client
//! ID is the public Google Cloud SDK client which supports installed-app flows.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::auth::{AuthMethod, DeviceFlowState, OAuthProvider, PollResult, StoredToken};

/// Google Cloud SDK public installed-app OAuth2 client ID.
const GOOGLE_CLIENT_ID: &str =
    "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com";

/// Client secret for the Google Cloud SDK installed-app client (public, not
/// sensitive — same value distributed with the public `gcloud` CLI).
const GOOGLE_CLIENT_SECRET: &str = "d-FL95Q19q7MQmFpd7hHD0Ty";

const GOOGLE_DEVICE_CODE_URL: &str = "https://oauth2.googleapis.com/device/code";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GEMINI_SCOPE: &str = "https://www.googleapis.com/auth/generative-language";

pub struct GeminiOAuth;

#[async_trait]
impl OAuthProvider for GeminiOAuth {
    fn provider_name(&self) -> &str {
        "gemini"
    }

    fn display_name(&self) -> &str {
        "Google Gemini"
    }

    fn auth_method(&self) -> AuthMethod {
        AuthMethod::DeviceFlow
    }

    async fn start_device_flow(&self, http: &Client) -> anyhow::Result<DeviceFlowState> {
        #[derive(Deserialize)]
        struct DeviceCodeResp {
            device_code: String,
            user_code: String,
            verification_url: String,
            expires_in: u64,
            #[serde(default = "default_interval")]
            interval: u64,
        }

        fn default_interval() -> u64 {
            5
        }

        let resp = http
            .post(GOOGLE_DEVICE_CODE_URL)
            .form(&[("client_id", GOOGLE_CLIENT_ID), ("scope", GEMINI_SCOPE)])
            .send()
            .await
            .context("Google device code request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Google device code request returned HTTP {status}: {body}");
        }

        let dc: DeviceCodeResp = resp
            .json()
            .await
            .context("failed to parse Google device code response")?;

        Ok(DeviceFlowState {
            device_code: dc.device_code,
            user_code: Some(dc.user_code),
            verification_uri: Some(dc.verification_url),
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
            refresh_token: Option<String>,
            expires_in: Option<u64>,
            token_type: Option<String>,
            error: Option<String>,
            error_description: Option<String>,
        }

        let resp = http
            .post(GOOGLE_TOKEN_URL)
            .form(&[
                ("client_id", GOOGLE_CLIENT_ID),
                ("client_secret", GOOGLE_CLIENT_SECRET),
                ("device_code", &state.device_code),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ])
            .send()
            .await;

        let Ok(resp) = resp else {
            return Ok(PollResult::Pending);
        };

        let body: TokenResp = match resp.json().await {
            Ok(b) => b,
            Err(_) => return Ok(PollResult::Pending),
        };

        if let Some(err) = body.error.as_deref() {
            if err == "authorization_pending" || err == "slow_down" {
                return Ok(PollResult::Pending);
            }
            let description = body.error_description.unwrap_or_default();
            return Ok(PollResult::Error(format!("{err}: {description}")));
        }

        if let Some(access_token) = body.access_token {
            let expires_at = body.expires_in.map(|secs| {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                now + secs as i64
            });

            return Ok(PollResult::Ready(StoredToken {
                provider: "gemini".to_string(),
                access_token,
                refresh_token: body.refresh_token,
                expires_at,
                token_type: body.token_type.unwrap_or_else(|| "Bearer".to_string()),
                scopes: vec![GEMINI_SCOPE.to_string()],
            }));
        }

        Ok(PollResult::Pending)
    }

    async fn refresh_token(
        &self,
        http: &Client,
        refresh_token: &str,
    ) -> anyhow::Result<StoredToken> {
        #[derive(Deserialize)]
        struct TokenResp {
            access_token: String,
            expires_in: Option<u64>,
            token_type: Option<String>,
        }

        let resp = http
            .post(GOOGLE_TOKEN_URL)
            .form(&[
                ("client_id", GOOGLE_CLIENT_ID),
                ("client_secret", GOOGLE_CLIENT_SECRET),
                ("refresh_token", refresh_token),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await
            .context("Google token refresh failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            anyhow::bail!("Google token refresh returned HTTP {status}");
        }

        let body: TokenResp = resp
            .json()
            .await
            .context("failed to parse token refresh response")?;

        let expires_at = body.expires_in.map(|secs| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            now + secs as i64
        });

        Ok(StoredToken {
            provider: "gemini".to_string(),
            access_token: body.access_token,
            refresh_token: Some(refresh_token.to_string()),
            expires_at,
            token_type: body.token_type.unwrap_or_else(|| "Bearer".to_string()),
            scopes: vec![GEMINI_SCOPE.to_string()],
        })
    }
}
