use clap::Parser;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};

#[derive(Parser, Debug, Clone)]
pub struct GeminiArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// Default model to use with Gemini CLI (e.g. gemini-2.0-flash)
    #[arg(long, default_value = "gemini-2.0-flash")]
    pub model: String,
}

pub async fn handle_gemini_connect(args: GeminiArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_gemini(&args, &mut changes);
    }

    let base_url = gateway.trim_end_matches('/').to_string();
    let api_key = gateway_key
        .clone()
        .unwrap_or_else(|| "isartor-local".to_string());

    let env_path = home_path(".isartor/env/gemini.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/gemini.sh"));

    let content = format!(
        "# Isartor — Gemini CLI integration (base URL override)\n\
         # Source this file: source ~/.isartor/env/gemini.sh\n\
         # Then run: gemini\n\
         export GEMINI_API_BASE_URL=\"{base_url}\"\n\
         export GEMINI_API_KEY=\"{api_key}\"\n\
         export ISARTOR_GEMINI_ENABLED=true\n"
    );

    if args.base.show_config || args.base.dry_run {
        println!("{}", content);
    }

    if write_file(&env_path, &content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: env_path.to_string_lossy().to_string(),
            description: "Shell env file with base URL override for Gemini CLI".to_string(),
        });
    }

    let test =
        test_isartor_connection(&gateway, gateway_key.as_deref(), "Hello from Gemini test").await;

    ConnectResult {
        client_name: "Gemini CLI".to_string(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "1. Start Isartor:  isartor up\n\
             2. Activate env:   source {}\n\
             3. Run Gemini CLI: gemini\n\
             \n\
             Method: base URL override (GEMINI_API_BASE_URL)\n\
             Base URL: {base_url}",
            env_path.display(),
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect_gemini(args: &GeminiArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    let env_path = home_path(".isartor/env/gemini.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/gemini.sh"));
    if env_path.exists() {
        remove_file(&env_path, args.base.dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: env_path.to_string_lossy().to_string(),
            description: "Removed".to_string(),
        });
    }

    ConnectResult {
        client_name: "Gemini CLI".to_string(),
        success: true,
        message: "Gemini CLI disconnected. Restart your shell to unset variables.".to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}
