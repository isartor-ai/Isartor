use clap::Parser;

use super::{
    home_path, remove_file, test_isartor_connection, write_file, BaseClientArgs, ConfigChange,
    ConfigChangeType, ConnectResult,
};
use crate::proxy::tls::IsartorCa;

#[derive(Parser, Debug, Clone)]
pub struct CopilotArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// GitHub personal access token (ghp_... or gho_...)
    #[arg(long, env = "GITHUB_TOKEN")]
    pub github_token: Option<String>,

    /// Shell to write exports to: bash | zsh | fish | powershell
    #[arg(long, default_value = "bash")]
    pub shell: String,

    /// CONNECT proxy listen address (default: 0.0.0.0:8081).
    /// This is the proxy that intercepts Copilot CLI HTTPS traffic.
    #[arg(long, default_value = "0.0.0.0:8081")]
    pub proxy_port: String,
}

pub async fn handle_copilot_connect(args: CopilotArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect(&args, &mut changes);
    }

    // Step 1: Ensure the Isartor CA exists (generates if first time).
    let ca = match IsartorCa::load_or_generate() {
        Ok(ca) => ca,
        Err(e) => {
            return ConnectResult {
                client_name: "GitHub Copilot CLI".to_string(),
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

    // Step 2: Derive the CONNECT proxy URL from --proxy-port.
    let proxy_port_num = args
        .proxy_port
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8081);
    let proxy_url = format!("http://localhost:{proxy_port_num}");

    // Step 3: Write the shell env file.
    let node_ca = ca_cert_path.to_string_lossy();

    let env_content = match args.shell.as_str() {
        "fish" => format!(
            "# Isartor — GitHub Copilot CLI integration (CONNECT proxy)\n\
             # Source this file: source ~/.isartor/env/copilot.fish\n\
             set -x HTTPS_PROXY \"{proxy_url}\"\n\
             set -x NODE_EXTRA_CA_CERTS \"{node_ca}\"\n\
             set -x ISARTOR_COPILOT_ENABLED true\n"
        ),
        "powershell" => format!(
            "# Isartor — GitHub Copilot CLI integration (CONNECT proxy)\n\
             # Dot-source: . ~/.isartor/env/copilot.ps1\n\
             $env:HTTPS_PROXY = \"{proxy_url}\"\n\
             $env:NODE_EXTRA_CA_CERTS = \"{node_ca}\"\n\
             $env:ISARTOR_COPILOT_ENABLED = \"true\"\n"
        ),
        _ => format!(
            "# Isartor — GitHub Copilot CLI integration (CONNECT proxy)\n\
             # Source this file: source ~/.isartor/env/copilot.sh\n\
             export HTTPS_PROXY=\"{proxy_url}\"\n\
             export NODE_EXTRA_CA_CERTS=\"{node_ca}\"\n\
             export ISARTOR_COPILOT_ENABLED=true\n"
        ),
    };

    let ext = match args.shell.as_str() {
        "fish" => "fish",
        "powershell" => "ps1",
        _ => "sh",
    };

    let env_path = home_path(&format!(".isartor/env/copilot.{ext}"))
        .unwrap_or_else(|_| std::path::PathBuf::from(format!(".isartor/env/copilot.{ext}")));

    if args.base.show_config || args.base.dry_run {
        println!("{}", env_content);
    }

    if write_file(&env_path, &env_content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: env_path.to_string_lossy().to_string(),
            description: "Shell env file with HTTPS_PROXY + NODE_EXTRA_CA_CERTS".to_string(),
        });
    }

    // Step 4: Store GitHub token if provided.
    if let Some(token) = &args.github_token {
        let token_path = home_path(".isartor/providers/copilot.json")
            .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/providers/copilot.json"));
        let cfg = serde_json::json!({
            "provider": "copilot",
            "github_token": token,
        });
        let content = serde_json::to_string_pretty(&cfg).unwrap_or_default();

        if args.base.show_config || args.base.dry_run {
            println!("\n{}", content);
        }

        if write_file(&token_path, &content, args.base.dry_run).is_ok() {
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileCreated,
                target: token_path.to_string_lossy().to_string(),
                description: "Copilot credentials (local)".to_string(),
            });
        }
    }

    // Step 5: Test the gateway API connection (not the proxy — the proxy requires
    // the server to be running, which is a separate step).
    let test = test_isartor_connection(
        &gateway,
        gateway_key.as_deref(),
        "What is the capital of France?",
    )
    .await;

    let source_cmd = match args.shell.as_str() {
        "fish" => format!("source {}", env_path.display()),
        "powershell" => format!(". {}", env_path.display()),
        _ => format!("source {}", env_path.display()),
    };

    ConnectResult {
        client_name: "GitHub Copilot CLI".to_string(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "1. Start Isartor:  isartor\n\
             2. Activate proxy: {source_cmd}\n\
             3. Use Copilot CLI normally — traffic routes through Isartor\n\
             \n\
             CONNECT proxy: {proxy_url}  (intercepting Copilot HTTPS traffic)\n\
             NODE_EXTRA_CA_CERTS: {node_ca}\n\
             \n\
             Note: The CA at {node_ca} is trusted by Node.js only (via NODE_EXTRA_CA_CERTS).\n\
             No system-level trust changes are made."
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect(args: &CopilotArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    for ext in ["sh", "fish", "ps1"] {
        let path = home_path(&format!(".isartor/env/copilot.{ext}"))
            .unwrap_or_else(|_| std::path::PathBuf::from(format!(".isartor/env/copilot.{ext}")));
        if path.exists() {
            remove_file(&path, args.base.dry_run);
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileModified,
                target: path.to_string_lossy().to_string(),
                description: "Removed".to_string(),
            });
        }
    }

    ConnectResult {
        client_name: "GitHub Copilot CLI".to_string(),
        success: true,
        message:
            "Copilot disconnected. Restart your shell to unset HTTPS_PROXY and NODE_EXTRA_CA_CERTS."
                .to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}
