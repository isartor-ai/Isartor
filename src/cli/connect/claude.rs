use clap::Parser;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, test_isartor_connection,
};

#[derive(Parser, Debug, Clone)]
pub struct ClaudeArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// Anthropic API key (sk-ant-...) — optional; not required for routing to Isartor.
    /// When provided, it is stored locally for future use.
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    pub key: Option<String>,

    /// Primary model for Claude Code
    #[arg(long, default_value = "claude-sonnet-4-6")]
    pub model: String,

    /// Fast/background model for Claude Code
    #[arg(long, default_value = "claude-haiku-4-5")]
    pub fast_model: String,
}

pub async fn handle_claude_connect(args: ClaudeArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_claude(&args, &mut changes);
    }

    let settings_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".claude/settings.json");

    let mut existing: serde_json::Value = if settings_path.exists() {
        let content = std::fs::read_to_string(&settings_path).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    // Back up existing settings.
    if settings_path.exists() && !args.base.dry_run {
        let backup = settings_path.with_extension("json.isartor-backup");
        let _ = std::fs::copy(&settings_path, &backup);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileBackedUp,
            target: backup.to_string_lossy().to_string(),
            description: "Original settings.json backed up".to_string(),
        });
    }

    // Ensure env section exists.
    if existing.get("env").and_then(|v| v.as_object()).is_none() {
        existing["env"] = serde_json::json!({});
    }
    let env = existing["env"].as_object_mut().unwrap();

    env.insert(
        "ANTHROPIC_BASE_URL".to_string(),
        serde_json::Value::String(gateway.clone()),
    );

    let auth_token = gateway_key
        .clone()
        .unwrap_or_else(|| "isartor-local".to_string());
    env.insert(
        "ANTHROPIC_AUTH_TOKEN".to_string(),
        serde_json::Value::String(auth_token),
    );

    env.insert(
        "ANTHROPIC_DEFAULT_SONNET_MODEL".to_string(),
        serde_json::Value::String(args.model.clone()),
    );
    env.insert(
        "ANTHROPIC_DEFAULT_HAIKU_MODEL".to_string(),
        serde_json::Value::String(args.fast_model.clone()),
    );

    if args.base.show_config || args.base.dry_run {
        println!(
            "Would write to {}:\n{}",
            settings_path.display(),
            serde_json::to_string_pretty(&existing).unwrap_or_default()
        );
    }

    if !args.base.dry_run {
        if let Some(parent) = settings_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(
            &settings_path,
            serde_json::to_string_pretty(&existing).unwrap_or_default(),
        );
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: settings_path.to_string_lossy().to_string(),
            description: "Configured Claude Code to use Isartor base URL override".to_string(),
        });
    }

    // Store optional Anthropic key for future use.
    if let Some(key) = &args.key {
        let key_path = dirs::home_dir()
            .unwrap_or_default()
            .join(".isartor/providers/anthropic.json");
        if !args.base.dry_run {
            if let Some(parent) = key_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let cfg = serde_json::json!({"provider":"anthropic","api_key":key});
            let _ = std::fs::write(
                &key_path,
                serde_json::to_string_pretty(&cfg).unwrap_or_default(),
            );
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileCreated,
                target: key_path.to_string_lossy().to_string(),
                description: "Stored Anthropic API key (local)".to_string(),
            });
        }
    }

    let test = test_isartor_connection(&gateway, gateway_key.as_deref(), "What is 2+2?").await;

    ConnectResult {
        client_name: "Claude Code".to_string(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "Claude Code now routes through Isartor via base URL override. \
             Start a new claude session to apply.\n\
             Gateway: {gateway}\n\
             Method: ANTHROPIC_BASE_URL override in ~/.claude/settings.json"
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect_claude(args: &ClaudeArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    let settings_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".claude/settings.json");

    let backup = settings_path.with_extension("json.isartor-backup");
    if backup.exists() && !args.base.dry_run {
        let _ = std::fs::copy(&backup, &settings_path);
        let _ = std::fs::remove_file(&backup);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: settings_path.to_string_lossy().to_string(),
            description: "Restored from backup".to_string(),
        });
    }

    ConnectResult {
        client_name: "Claude Code".to_string(),
        success: true,
        message: "Claude Code disconnected from Isartor. Start a new claude session to apply."
            .to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}
