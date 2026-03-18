use clap::Parser;

use super::{
    home_path, remove_file, test_isartor_connection, write_file, BaseClientArgs, ConfigChange,
    ConfigChangeType, ConnectResult,
};

#[derive(Parser, Debug, Clone)]
pub struct OpenclawArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// Primary model for OpenClaw agent (routed through Isartor)
    #[arg(long, default_value = "gpt-4o")]
    pub model: String,

    /// Fallback models (comma-separated)
    #[arg(long, default_value = "claude-sonnet-4-6,groq/llama-3.1-8b-instant")]
    pub fallbacks: String,

    /// Path to openclaw.json (auto-detected if not specified)
    #[arg(long)]
    pub config_path: Option<String>,
}

pub async fn handle_openclaw_connect(args: OpenclawArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_openclaw(&args, &mut changes);
    }

    // Find openclaw.json (best-effort).
    let config_path = args
        .config_path
        .as_deref()
        .map(std::path::PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".openclaw/openclaw.json")))
        .or_else(|| {
            let cwd = std::env::current_dir().ok()?;
            let path = cwd.join("openclaw.json");
            if path.exists() {
                Some(path)
            } else {
                None
            }
        });

    // Build provider block (JSON5-ish).
    let base_url = format!("{}/v1", gateway.trim_end_matches('/'));
    let api_key = gateway_key
        .clone()
        .unwrap_or_else(|| "isartor-local".to_string());

    let isartor_provider_block = format!(
        r#"// Added by: isartor connect openclaw
// Remove this block to disconnect Isartor
"isartor": {{
  baseUrl: "{base_url}",
  apiKey: "{api_key}",
  api: "openai-chat",
}},"#
    );

    let fallback_list: Vec<String> = args
        .fallbacks
        .split(',')
        .map(|s| format!("\"isartor/{}\"", s.trim()))
        .collect();

    println!(
        "\nAdd this to models.providers in your openclaw.json:\n\n{}",
        isartor_provider_block
    );
    println!(
        "\nThen set your agent model to:\n  agent: {{ model: {{ primary: \"isartor/{}\", fallbacks: [{}] }} }}",
        args.model,
        fallback_list.join(", ")
    );

    // Back up existing config if found.
    if let Some(cfg_path) = &config_path {
        if cfg_path.exists() && !args.base.dry_run {
            let backup = cfg_path.with_extension("json.isartor-backup");
            let _ = std::fs::copy(cfg_path, &backup);
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileBackedUp,
                target: backup.to_string_lossy().to_string(),
                description: "Original openclaw.json backed up".to_string(),
            });
        }
    }

    // Write a patch file for manual application.
    let patch_path = home_path(".isartor/patches/openclaw.patch.json5")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/patches/openclaw.patch.json5"));
    let patch_content = format!(
        "// Paste into the models.providers block of your openclaw.json:\n{}\n",
        isartor_provider_block
    );

    if args.base.show_config || args.base.dry_run {
        println!("\n{}", patch_content);
    }

    if write_file(&patch_path, &patch_content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: patch_path.to_string_lossy().to_string(),
            description: "Patch file — paste into openclaw.json models.providers".to_string(),
        });
    }

    let test =
        test_isartor_connection(&gateway, gateway_key.as_deref(), "Hello from OpenClaw test").await;

    ConnectResult {
        client_name: "OpenClaw".to_string(),
        success: test.response_received || args.base.dry_run,
        message: if let Some(p) = config_path {
            format!(
                "Patch file written. Apply it to {} then restart OpenClaw.",
                p.display()
            )
        } else {
            "Patch file written. Apply it to your openclaw.json then restart OpenClaw.".to_string()
        },
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect_openclaw(args: &OpenclawArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    let patch_path = home_path(".isartor/patches/openclaw.patch.json5")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/patches/openclaw.patch.json5"));
    if patch_path.exists() {
        remove_file(&patch_path, args.base.dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: patch_path.to_string_lossy().to_string(),
            description: "Removed".to_string(),
        });
    }

    ConnectResult {
        client_name: "OpenClaw".to_string(),
        success: true,
        message: "OpenClaw disconnected (patch file removed). Restore your openclaw.json manually if needed.".to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}
