use clap::Parser;

use super::{
    home_path, remove_file, test_isartor_connection, write_file, BaseClientArgs, ConfigChange,
    ConfigChangeType, ConnectResult,
};

#[derive(Parser, Debug, Clone)]
pub struct AntigravityArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,
}

pub async fn handle_antigravity_connect(args: AntigravityArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_antigravity(&args, &mut changes);
    }

    // Best-effort env-file integration (actual Antigravity config may differ).
    let env_path = home_path(".isartor/env/antigravity.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/antigravity.sh"));

    let content = format!(
        "# Isartor — Antigravity integration\n\
         export ANTIGRAVITY_BASE_URL=\"{}/v1\"\n\
         export ANTIGRAVITY_API_KEY=\"{}\"\n",
        gateway.trim_end_matches('/'),
        gateway_key
            .clone()
            .unwrap_or_else(|| "isartor-local".to_string())
    );

    if args.base.show_config || args.base.dry_run {
        println!("{}", content);
    }

    if write_file(&env_path, &content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: env_path.to_string_lossy().to_string(),
            description: "Shell env file for Antigravity".to_string(),
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
            "Run: source {}\nThen restart Antigravity.",
            env_path.display()
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
