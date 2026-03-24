use clap::Parser;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};

#[derive(Parser, Debug, Clone)]
pub struct CursorArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,
}

pub async fn handle_cursor_connect(args: CursorArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_cursor(&args, &mut changes);
    }

    // Cursor uses an OpenAI-compatible base URL with a /cursor suffix.
    let cursor_base_url = format!("{}/v1", gateway.trim_end_matches('/'));
    let api_key = gateway_key
        .clone()
        .unwrap_or_else(|| "isartor-local".to_string());

    // 1. Write env script for shell-based activation.
    let env_path = home_path(".isartor/env/cursor.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/cursor.sh"));

    let env_content = format!(
        "# Isartor — Cursor IDE integration (base URL override)\n\
         # Source this file: source ~/.isartor/env/cursor.sh\n\
         # Then configure Cursor IDE:\n\
         #   Settings → Cursor Settings → Models\n\
         #   Enable \"Override OpenAI Base URL\" and enter:\n\
         #     {cursor_base_url}\n\
         #   Paste this API key:\n\
         #     {api_key}\n\
         export ISARTOR_CURSOR_BASE_URL=\"{cursor_base_url}\"\n\
         export ISARTOR_CURSOR_API_KEY=\"{api_key}\"\n\
         export ISARTOR_CURSOR_ENABLED=true\n"
    );

    if args.base.show_config || args.base.dry_run {
        println!("{}", env_content);
    }

    if write_file(&env_path, &env_content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: env_path.to_string_lossy().to_string(),
            description: "Shell env file with Cursor configuration values".to_string(),
        });
    }

    // 2. Write Cursor MCP config for tool-based integration.
    install_cursor_mcp_config(&gateway, gateway_key.as_deref(), &mut changes, &args);

    // 3. Test connection.
    let test =
        test_isartor_connection(&gateway, gateway_key.as_deref(), "Hello from Cursor test").await;

    ConnectResult {
        client_name: "Cursor".to_string(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "Configure Cursor IDE:\n\
             \n\
             1. Start Isartor:  isartor up\n\
             2. Open Cursor → Settings → Cursor Settings → Models\n\
             3. Enable \"Override OpenAI Base URL\" and enter:\n\
             \n\
                Base URL: {cursor_base_url}\n\
                API Key:  {api_key}\n\
             \n\
             4. Add your desired model name (e.g. gpt-4o)\n\
             5. Use Ask or Plan mode (Agent mode doesn't support custom keys yet)\n\
             \n\
             Method: base URL override (OpenAI-compatible)\n\
             MCP:    ~/.cursor/mcp.json (for tool integration)"
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn install_cursor_mcp_config(
    gateway_url: &str,
    gateway_api_key: Option<&str>,
    changes: &mut Vec<ConfigChange>,
    args: &CursorArgs,
) {
    let mcp_path = match home_path(".cursor/mcp.json") {
        Ok(p) => p,
        Err(_) => return,
    };

    let base = gateway_url.trim_end_matches('/');
    let api_key = gateway_api_key.unwrap_or("isartor-local");

    let isartor_entry = serde_json::json!({
        "url": format!("{base}/mcp/"),
        "type": "http",
        "headers": {
            "Authorization": format!("Bearer {api_key}")
        }
    });

    let mut root: serde_json::Value = if mcp_path.exists() {
        std::fs::read_to_string(&mcp_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| serde_json::json!({"mcpServers": {}}))
    } else {
        serde_json::json!({"mcpServers": {}})
    };

    if let Some(servers) = root.get_mut("mcpServers").and_then(|v| v.as_object_mut()) {
        servers.insert("isartor".to_string(), isartor_entry);
    } else {
        root["mcpServers"] = serde_json::json!({"isartor": isartor_entry});
    }

    let content = serde_json::to_string_pretty(&root).unwrap_or_default();

    if args.base.show_config {
        println!("--- ~/.cursor/mcp.json ---\n{content}\n");
    }

    let change_type = if mcp_path.exists() {
        ConfigChangeType::FileModified
    } else {
        ConfigChangeType::FileCreated
    };

    if write_file(&mcp_path, &content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type,
            target: mcp_path.to_string_lossy().to_string(),
            description: "Cursor MCP server registration".to_string(),
        });
    }
}

fn remove_isartor_from_cursor_mcp(changes: &mut Vec<ConfigChange>, dry_run: bool) {
    let mcp = match home_path(".cursor/mcp.json") {
        Ok(p) => p,
        Err(_) => return,
    };
    let raw = match std::fs::read_to_string(&mcp) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut root: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return,
    };
    let removed = root
        .get_mut("mcpServers")
        .and_then(|v| v.as_object_mut())
        .and_then(|servers| servers.remove("isartor"))
        .is_some();
    if removed {
        let content = serde_json::to_string_pretty(&root).unwrap_or_default();
        let _ = write_file(&mcp, &content, dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: mcp.to_string_lossy().to_string(),
            description: "Removed isartor from MCP servers".to_string(),
        });
    }
}

fn disconnect_cursor(args: &CursorArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    // Remove env script.
    let env_path = home_path(".isartor/env/cursor.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/env/cursor.sh"));
    if env_path.exists() {
        remove_file(&env_path, args.base.dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: env_path.to_string_lossy().to_string(),
            description: "Removed".to_string(),
        });
    }

    // Remove isartor from Cursor MCP config.
    remove_isartor_from_cursor_mcp(changes, args.base.dry_run);

    ConnectResult {
        client_name: "Cursor".to_string(),
        success: true,
        message: "Cursor disconnected. Remove the base URL override in Cursor settings."
            .to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}
