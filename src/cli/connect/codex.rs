use clap::Parser;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};

#[derive(Parser, Debug, Clone)]
pub struct CodexArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// Default model to use with Codex (e.g. o3-mini, gpt-4o)
    #[arg(long, default_value = "o3-mini")]
    pub model: String,
}

pub async fn handle_codex_connect(args: CodexArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_codex(&args, &mut changes);
    }

    let base_url = format!("{}/v1", gateway.trim_end_matches('/'));
    let api_key = gateway_key
        .clone()
        .unwrap_or_else(|| "isartor-local".to_string());

    let env_path = home_path(".isartor/env/codex.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/codex.sh"));

    let content = format!(
        "# Isartor — OpenAI Codex CLI integration (base URL override)\n\
         # Source this file: source ~/.isartor/env/codex.sh\n\
         # Then run: codex --model {model}\n\
         export OPENAI_BASE_URL=\"{base_url}\"\n\
         export OPENAI_API_KEY=\"{api_key}\"\n\
         export ISARTOR_CODEX_ENABLED=true\n",
        model = args.model,
    );

    if args.base.show_config || args.base.dry_run {
        println!("{}", content);
    }

    if write_file(&env_path, &content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: env_path.to_string_lossy().to_string(),
            description: "Shell env file with base URL override for Codex".to_string(),
        });
    }

    let test =
        test_isartor_connection(&gateway, gateway_key.as_deref(), "Hello from Codex test").await;

    ConnectResult {
        client_name: "Codex".to_string(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "1. Start Isartor:  isartor up\n\
             2. Activate env:   source {}\n\
             3. Run Codex:      codex --model {}\n\
             \n\
             Method: base URL override (OPENAI_BASE_URL)\n\
             Base URL: {base_url}",
            env_path.display(),
            args.model,
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect_codex(args: &CodexArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    let env_path = home_path(".isartor/env/codex.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/codex.sh"));
    if env_path.exists() {
        remove_file(&env_path, args.base.dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: env_path.to_string_lossy().to_string(),
            description: "Removed".to_string(),
        });
    }

    ConnectResult {
        client_name: "Codex".to_string(),
        success: true,
        message: "Codex disconnected. Restart your shell to unset variables.".to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}
