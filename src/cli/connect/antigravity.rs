use clap::Parser;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};
use crate::proxy::tls::IsartorCa;

#[derive(Parser, Debug, Clone)]
pub struct AntigravityArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// CONNECT proxy listen address (default: 0.0.0.0:8081).
    /// Antigravity traffic uses this proxy so Isartor can preserve Antigravity upstream as Layer 3.
    #[arg(long, default_value = "0.0.0.0:8081")]
    pub proxy_port: String,
}

pub async fn handle_antigravity_connect(args: AntigravityArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_antigravity(&args, &mut changes);
    }

    let ca = match IsartorCa::load_or_generate() {
        Ok(ca) => ca,
        Err(e) => {
            return ConnectResult {
                client_name: "Antigravity".to_string(),
                success: false,
                message: format!(
                    "Failed to generate Isartor CA certificate: {e}\n\
                     The CONNECT proxy requires a local CA for TLS interception."
                ),
                changes_made: changes,
                test_result: None,
            };
        }
    };
    let ca_cert_path = ca.ca_cert_path().to_path_buf();
    changes.push(ConfigChange {
        change_type: ConfigChangeType::FileCreated,
        target: ca_cert_path.to_string_lossy().to_string(),
        description: "Isartor CA certificate (for TLS MITM)".to_string(),
    });

    let proxy_port_num = args
        .proxy_port
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8081);
    let proxy_url = format!("http://localhost:{proxy_port_num}");

    // Best-effort env-file integration (actual Antigravity config may differ).
    let env_path = home_path(".isartor/env/antigravity.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/antigravity.sh"));

    let content = format!(
        "# Isartor — Antigravity integration (CONNECT proxy)\n\
         export HTTPS_PROXY=\"{proxy_url}\"\n\
         export HTTP_PROXY=\"{proxy_url}\"\n\
         export NODE_EXTRA_CA_CERTS=\"{}\"\n\
         export SSL_CERT_FILE=\"{}\"\n\
         export REQUESTS_CA_BUNDLE=\"{}\"\n\
         export ISARTOR_ANTIGRAVITY_ENABLED=true\n",
        ca_cert_path.display(),
        ca_cert_path.display(),
        ca_cert_path.display(),
    );

    if args.base.show_config || args.base.dry_run {
        println!("{}", content);
    }

    if write_file(&env_path, &content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: env_path.to_string_lossy().to_string(),
            description: "Shell env file with HTTPS_PROXY and CA trust".to_string(),
        });
    }

    let test = test_isartor_connection(
        &gateway,
        gateway_key.as_deref(),
        "Hello from Antigravity test",
    )
    .await;

    ConnectResult {
        client_name: "Antigravity".to_string(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "Run: source {}\nThen restart Antigravity.\nProxy: {}\nCA: {}\nLayer 3 for proxied Antigravity requests: Antigravity upstream passthrough (no separate Isartor Layer 3 key required for this path).",
            env_path.display(),
            proxy_url,
            ca_cert_path.display()
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect_antigravity(
    args: &AntigravityArgs,
    changes: &mut Vec<ConfigChange>,
) -> ConnectResult {
    let env_path = home_path(".isartor/env/antigravity.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/antigravity.sh"));
    if env_path.exists() {
        remove_file(&env_path, args.base.dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: env_path.to_string_lossy().to_string(),
            description: "Removed".to_string(),
        });
    }

    ConnectResult {
        client_name: "Antigravity".to_string(),
        success: true,
        message: "Antigravity disconnected. Restart your shell to unset variables.".to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}
