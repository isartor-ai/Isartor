//! Anthropic API-key credential provider.
//!
//! Anthropic does not expose a public device flow. This provider prompts the
//! user to paste their API key securely using [`rpassword`].

use async_trait::async_trait;
use std::io::{self, Write};

use crate::auth::{AuthMethod, OAuthProvider, StoredToken};

pub struct AnthropicApiKey;

#[async_trait]
impl OAuthProvider for AnthropicApiKey {
    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn display_name(&self) -> &str {
        "Anthropic (Claude)"
    }

    fn auth_method(&self) -> AuthMethod {
        AuthMethod::ApiKey
    }

    async fn prompt_api_key(&self) -> anyhow::Result<StoredToken> {
        println!();
        println!("Anthropic (Claude) — API key setup");
        println!("───────────────────────────────────");
        println!("Get your API key from: https://console.anthropic.com/account/keys");
        println!();

        print!("Paste your Anthropic API key: ");
        io::stdout()
            .flush()
            .map_err(|e| anyhow::anyhow!("failed to flush stdout: {e}"))?;
        let key = rpassword::read_password()
            .map_err(|e| anyhow::anyhow!("failed to read API key: {e}"))?;

        let key = key.trim().to_string();
        if key.is_empty() {
            anyhow::bail!("API key cannot be empty");
        }
        if !key.starts_with("sk-ant-") {
            eprintln!("Warning: key does not start with 'sk-ant-' — double-check it before use.");
        }

        Ok(StoredToken {
            provider: "anthropic".to_string(),
            access_token: key,
            refresh_token: None,
            expires_at: None,
            token_type: "ApiKey".to_string(),
            scopes: vec![],
        })
    }
}
