//! AWS Builder ID / Kiro OAuth provider via AWS SSO OIDC device flow.
//!
//! Uses the `https://oidc.us-east-1.amazonaws.com` OIDC endpoint, performs
//! dynamic client registration, then runs RFC 8628 device authorization grant.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::auth::{AuthMethod, DeviceFlowState, OAuthProvider, PollResult, StoredToken};

const AWS_OIDC_BASE: &str = "https://oidc.us-east-1.amazonaws.com";
const REGISTRATION_URL: &str = "https://oidc.us-east-1.amazonaws.com/client/register";

/// Client registration cache — we hold the registered client_id/secret in this struct
/// because the dynamic registration must happen before the device flow.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct OidcClient {
    client_id: String,
    client_secret: String,
    client_id_issued_at: i64,
    client_secret_expires_at: i64,
}

pub struct KiroOAuth;

#[async_trait]
impl OAuthProvider for KiroOAuth {
    fn provider_name(&self) -> &str {
        "kiro"
    }

    fn display_name(&self) -> &str {
        "AWS Kiro / Builder ID"
    }

    fn auth_method(&self) -> AuthMethod {
        AuthMethod::DeviceFlow
    }

    async fn start_device_flow(&self, http: &Client) -> anyhow::Result<DeviceFlowState> {
        // 1. Register an ephemeral OIDC client.
        let oidc = register_client(http).await?;

        // 2. Start device authorization.
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct DeviceAuthResp {
            device_code: String,
            user_code: String,
            verification_uri: String,
            verification_uri_complete: Option<String>,
            expires_in: u64,
            #[serde(default = "default_interval")]
            interval: u64,
        }

        fn default_interval() -> u64 {
            5
        }

        let url = format!("{AWS_OIDC_BASE}/device_authorization");
        let resp = http
            .post(&url)
            .json(&json!({
                "clientId": oidc.client_id,
                "clientSecret": oidc.client_secret,
                "scopes": ["openid", "profile", "sso:account:access", "aws.cognito.signin.user.admin"]
            }))
            .send()
            .await
            .context("Kiro device authorization request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Kiro device authorization returned HTTP {status}: {body}");
        }

        let da: DeviceAuthResp = resp
            .json()
            .await
            .context("failed to parse Kiro device authorization response")?;

        // Encode oidc credentials into the device_code field so poll_device_token
        // can access them without extra storage. Format: "<device_code>|<client_id>|<client_secret>"
        let encoded_device_code = format!(
            "{}|{}|{}",
            da.device_code, oidc.client_id, oidc.client_secret
        );

        Ok(DeviceFlowState {
            device_code: encoded_device_code,
            user_code: Some(da.user_code),
            verification_uri: Some(da.verification_uri),
            verification_uri_complete: da.verification_uri_complete,
            expires_in: da.expires_in,
            interval: da.interval,
        })
    }

    async fn poll_device_token(
        &self,
        http: &Client,
        state: &DeviceFlowState,
    ) -> anyhow::Result<PollResult> {
        // Decode credentials from device_code field (see start_device_flow above).
        let parts: Vec<&str> = state.device_code.splitn(3, '|').collect();
        if parts.len() != 3 {
            return Ok(PollResult::Error(
                "invalid Kiro device code format".to_string(),
            ));
        }
        let (device_code, client_id, client_secret) = (parts[0], parts[1], parts[2]);

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct TokenResp {
            access_token: Option<String>,
            refresh_token: Option<String>,
            expires_in: Option<u64>,
            token_type: Option<String>,
            // Errors use "error" key
            error: Option<String>,
            error_description: Option<String>,
        }

        let url = format!("{AWS_OIDC_BASE}/token");
        let resp = http
            .post(&url)
            .json(&json!({
                "clientId": client_id,
                "clientSecret": client_secret,
                "deviceCode": device_code,
                "grantType": "urn:ietf:params:oauth:grant-type:device_code"
            }))
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
            let desc = body.error_description.unwrap_or_default();
            return Ok(PollResult::Error(format!("{err}: {desc}")));
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
                provider: "kiro".to_string(),
                access_token,
                refresh_token: body.refresh_token,
                expires_at,
                token_type: body.token_type.unwrap_or_else(|| "Bearer".to_string()),
                scopes: vec![],
            }));
        }

        Ok(PollResult::Pending)
    }
}

async fn register_client(http: &Client) -> anyhow::Result<OidcClient> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct RegistrationResp {
        client_id: String,
        client_secret: String,
        #[serde(default)]
        client_id_issued_at: i64,
        #[serde(default)]
        client_secret_expires_at: i64,
    }

    let resp = http
        .post(REGISTRATION_URL)
        .json(&json!({
            "clientName": "isartor",
            "clientType": "public",
            "scopes": ["openid", "profile", "sso:account:access", "aws.cognito.signin.user.admin"]
        }))
        .send()
        .await
        .context("Kiro OIDC client registration failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Kiro OIDC client registration returned HTTP {status}: {body}");
    }

    let r: RegistrationResp = resp
        .json()
        .await
        .context("failed to parse OIDC client registration response")?;

    Ok(OidcClient {
        client_id: r.client_id,
        client_secret: r.client_secret,
        client_id_issued_at: r.client_id_issued_at,
        client_secret_expires_at: r.client_secret_expires_at,
    })
}
