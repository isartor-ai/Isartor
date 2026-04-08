//! Shared device-flow polling loop (RFC 8628).
//!
//! Providers that implement RFC 8628 device authorization grant call
//! [`poll_until_ready`] after starting their flow and displaying the user code.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::bail;
use reqwest::Client;
use tokio::time::sleep;

use super::{DeviceFlowState, OAuthProvider, PollResult};

/// Run the polling loop for a device authorization grant.
///
/// Prints progress dots to stdout. Returns the `StoredToken` once the user
/// authenticates, or an error if the flow expires or encounters a fatal error.
pub async fn poll_until_ready(
    provider: &dyn OAuthProvider,
    http: &Client,
    state: &DeviceFlowState,
) -> anyhow::Result<super::StoredToken> {
    let interval = Duration::from_secs(state.interval.max(5));
    let deadline = Instant::now() + Duration::from_secs(state.expires_in);

    loop {
        if Instant::now() > deadline {
            bail!(
                "authentication timed out; please try `isartor auth {}` again",
                provider.provider_name()
            );
        }

        sleep(interval).await;

        match provider.poll_device_token(http, state).await? {
            PollResult::Ready(token) => {
                println!();
                return Ok(token);
            }
            PollResult::Pending => {
                print!(".");
                let _ = std::io::stdout().flush();
            }
            PollResult::Error(msg) => {
                bail!("authentication failed: {msg}");
            }
        }
    }
}

/// Display the standard device-flow prompt.
pub fn print_device_flow_instructions(provider_name: &str, state: &DeviceFlowState) {
    println!();
    println!("{provider_name} authentication");
    println!("{}", "─".repeat(provider_name.len() + 15));

    if let Some(uri_complete) = &state.verification_uri_complete {
        println!("Open this URL in your browser:");
        println!("  {uri_complete}");
        if let Some(user_code) = &state.user_code {
            println!(
                "(or go to {} and enter code: {})",
                state.verification_uri.as_deref().unwrap_or(""),
                user_code
            );
        }
    } else if let Some(uri) = &state.verification_uri {
        println!("1. Open: {uri}");
        if let Some(user_code) = &state.user_code {
            println!("2. Enter code: {user_code}");
        }
    }

    println!();
    println!(
        "Waiting for authentication (expires in {}s)...",
        state.expires_in
    );
}
