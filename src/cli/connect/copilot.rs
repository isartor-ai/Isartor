use clap::Parser;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};

const COPILOT_INSTRUCTIONS_PATH: &str = ".copilot/copilot-instructions.md";
const ISARTOR_INSTRUCTION_START: &str = "<!-- isartor:copilot-instructions:start -->";
const ISARTOR_INSTRUCTION_END: &str = "<!-- isartor:copilot-instructions:end -->";

#[derive(Parser, Debug, Clone)]
pub struct CopilotArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// GitHub personal access token (ghp_... or gho_...)
    #[arg(long, env = "GITHUB_TOKEN")]
    pub github_token: Option<String>,
}

pub async fn handle_copilot_connect(args: CopilotArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect(&args, &mut changes);
    }

    // Clean up legacy files from previous integration approaches.
    if !args.base.dry_run {
        cleanup_legacy_files(&mut changes);
    }

    // Step 1: Find the isartor binary path for MCP config.
    let isartor_bin =
        std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("isartor"));

    // Step 2: Build MCP server entry for ~/.copilot/mcp-config.json
    let mut mcp_args = vec!["mcp".to_string()];
    if gateway != super::DEFAULT_GATEWAY_URL {
        mcp_args.push("--gateway-url".to_string());
        mcp_args.push(gateway.clone());
    }
    if let Some(ref key) = gateway_key
        && !key.is_empty()
    {
        mcp_args.push("--gateway-api-key".to_string());
        mcp_args.push(key.clone());
    }

    let isartor_entry = serde_json::json!({
        "type": "stdio",
        "command": isartor_bin.to_string_lossy(),
        "args": mcp_args,
    });

    // Step 3: Read existing mcp-config.json, merge, write back.
    let mcp_config_path = home_path(".copilot/mcp-config.json")
        .unwrap_or_else(|_| std::path::PathBuf::from(".copilot/mcp-config.json"));

    let mut mcp_config: serde_json::Value = if mcp_config_path.exists() {
        let content = std::fs::read_to_string(&mcp_config_path).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    // Support both { "mcpServers": { ... } } and flat { "server": { ... } } formats.
    // Copilot CLI accepts both; prefer mcpServers wrapper.
    if mcp_config.get("mcpServers").is_none() {
        mcp_config["mcpServers"] = serde_json::json!({});
    }
    mcp_config["mcpServers"]["isartor"] = isartor_entry;

    let mcp_json = serde_json::to_string_pretty(&mcp_config).unwrap_or_default();

    if args.base.show_config || args.base.dry_run {
        println!("{}", mcp_json);
    }

    let mcp_written = if !args.base.dry_run {
        write_file(&mcp_config_path, &mcp_json, false).is_ok()
    } else {
        true
    };

    if mcp_written {
        changes.push(ConfigChange {
            change_type: if mcp_config_path.exists() {
                ConfigChangeType::FileModified
            } else {
                ConfigChangeType::FileCreated
            },
            target: mcp_config_path.to_string_lossy().to_string(),
            description: "Registered isartor MCP server".to_string(),
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

    // Step 5: Ensure Isartor gateway URL is in Copilot's allowed URLs.
    add_allowed_url(&gateway, &mut changes, args.base.dry_run);

    // Step 6: Install persistent Copilot instructions so plain prompts prefer
    // the cache lookup tool before falling back to Copilot's own model.
    install_copilot_instructions(&mut changes, args.base.dry_run, args.base.show_config);

    // Step 7: Test the gateway connection.
    let test = test_isartor_connection(
        &gateway,
        gateway_key.as_deref(),
        "What is the capital of France?",
    )
    .await;

    let success = test.response_received || test.layer_resolved == "timeout" || args.base.dry_run;

    ConnectResult {
        client_name: "GitHub Copilot CLI".to_string(),
        success,
        message: format!(
            "MCP server registered in:\n  {}\n\n\
             Copilot CLI will now have `isartor_chat` and `isartor_cache_store`\n\
             tools available, plus a managed instruction block in:\n  {}\n\n\
             Plain conversational prompts will prefer `isartor_chat` first.\n\
             On a cache miss, Copilot answers with its own model and then stores\n\
             the result back via `isartor_cache_store`.\n\n\
             Start Copilot CLI normally — no env vars or hooks needed:\n  copilot",
            mcp_config_path.display(),
            home_path(COPILOT_INSTRUCTIONS_PATH)
                .unwrap_or_else(|_| std::path::PathBuf::from(COPILOT_INSTRUCTIONS_PATH))
                .display(),
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

/// Add the gateway URL to Copilot's allowed_urls in ~/.copilot/config.json.
fn add_allowed_url(gateway_url: &str, changes: &mut Vec<ConfigChange>, dry_run: bool) {
    let config_path = match home_path(".copilot/config.json") {
        Ok(p) => p,
        Err(_) => return,
    };
    if !config_path.exists() {
        return;
    }

    let content = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut config: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return,
    };

    let urls = config
        .get_mut("allowed_urls")
        .and_then(|v| v.as_array_mut());

    if let Some(urls) = urls {
        let gw = serde_json::Value::String(gateway_url.to_string());
        if !urls.contains(&gw) {
            urls.push(gw);
            if !dry_run && let Ok(json) = serde_json::to_string_pretty(&config) {
                let _ = std::fs::write(&config_path, json);
                changes.push(ConfigChange {
                    change_type: ConfigChangeType::FileModified,
                    target: config_path.to_string_lossy().to_string(),
                    description: format!("Added {gateway_url} to allowed_urls"),
                });
            }
        }
    }
}

fn install_copilot_instructions(changes: &mut Vec<ConfigChange>, dry_run: bool, show_config: bool) {
    let path = match home_path(COPILOT_INSTRUCTIONS_PATH) {
        Ok(p) => p,
        Err(_) => return,
    };
    let existed = path.exists();
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let updated = upsert_managed_block(&existing, &managed_instruction_block());

    if show_config {
        println!("\n{}", updated);
    }

    if updated == existing {
        return;
    }

    if write_file(&path, &updated, dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: if existed {
                ConfigChangeType::FileModified
            } else {
                ConfigChangeType::FileCreated
            },
            target: path.to_string_lossy().to_string(),
            description: "Installed Isartor Copilot instructions".to_string(),
        });
    }
}

fn remove_copilot_instructions(changes: &mut Vec<ConfigChange>, dry_run: bool) {
    let path = match home_path(COPILOT_INSTRUCTIONS_PATH) {
        Ok(p) => p,
        Err(_) => return,
    };
    if !path.exists() {
        return;
    }

    let existing = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(_) => return,
    };
    let updated = remove_managed_block(&existing);
    if updated == existing {
        return;
    }

    if updated.trim().is_empty() {
        remove_file(&path, dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: path.to_string_lossy().to_string(),
            description: "Removed Isartor Copilot instructions".to_string(),
        });
        return;
    }

    if write_file(&path, &updated, dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: path.to_string_lossy().to_string(),
            description: "Removed Isartor Copilot instructions".to_string(),
        });
    }
}

fn managed_instruction_block() -> String {
    format!(
        "{ISARTOR_INSTRUCTION_START}\n\
         # Isartor cache-first workflow\n\n\
         For plain conversational user prompts that do not require reading files,\n\
         editing code, or running shell commands, call `isartor_chat` first with\n\
         the user's prompt verbatim.\n\n\
         - If `isartor_chat` returns non-empty text, use that cached answer as the\n\
           final response.\n\
         - If `isartor_chat` returns an empty string, answer with your own model,\n\
           then immediately call `isartor_cache_store` with the exact prompt and\n\
           your exact final answer.\n\
         - This keeps Copilot's own LLM as the source of truth on cache misses\n\
           while allowing Isartor L1a/L1b cache hits on repeats.\n\
         - Do not skip this flow for simple questions like factual Q&A, summaries,\n\
           or general chat. Only bypass it when the task clearly requires repo\n\
           tools, file edits, commands, or GitHub-specific MCP tools.\n\
         {ISARTOR_INSTRUCTION_END}\n"
    )
}

fn upsert_managed_block(existing: &str, block: &str) -> String {
    if let Some((start, end)) = managed_block_range(existing) {
        let mut result = String::new();
        result.push_str(&existing[..start]);
        result.push_str(block);
        result.push_str(&existing[end..]);
        return normalize_instruction_spacing(&result);
    }

    if existing.trim().is_empty() {
        return block.to_string();
    }

    let mut result = existing.trim_end().to_string();
    result.push_str("\n\n");
    result.push_str(block);
    normalize_instruction_spacing(&result)
}

fn remove_managed_block(existing: &str) -> String {
    let Some((start, end)) = managed_block_range(existing) else {
        return existing.to_string();
    };

    let mut result = String::new();
    result.push_str(&existing[..start]);
    result.push_str(&existing[end..]);
    normalize_instruction_spacing(&result)
}

fn managed_block_range(content: &str) -> Option<(usize, usize)> {
    let start = content.find(ISARTOR_INSTRUCTION_START)?;
    let end_marker = content.find(ISARTOR_INSTRUCTION_END)?;
    let end = end_marker + ISARTOR_INSTRUCTION_END.len();
    Some((start, end))
}

fn normalize_instruction_spacing(content: &str) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    format!("{trimmed}\n")
}

fn cleanup_legacy_files(changes: &mut Vec<ConfigChange>) {
    // Legacy proxy-era env files (v0.1.33 and earlier).
    for ext in ["sh", "fish", "ps1"] {
        let path = home_path(&format!(".isartor/env/copilot.{ext}")).unwrap_or_default();
        if path.exists() {
            remove_file(&path, false);
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileModified,
                target: path.to_string_lossy().to_string(),
                description: "Removed legacy proxy env file".to_string(),
            });
        }
    }

    // Legacy hook files (v0.1.34 hook approach).
    for filename in [
        ".isartor/hooks/copilot_pretooluse.sh",
        ".isartor/copilot-hook-setup.txt",
    ] {
        let path = home_path(filename).unwrap_or_default();
        if path.exists() {
            remove_file(&path, false);
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileModified,
                target: path.to_string_lossy().to_string(),
                description: "Removed legacy hook file".to_string(),
            });
        }
    }
}

fn disconnect(args: &CopilotArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    // Remove isartor entry from mcp-config.json.
    let mcp_config_path = home_path(".copilot/mcp-config.json")
        .unwrap_or_else(|_| std::path::PathBuf::from(".copilot/mcp-config.json"));

    if mcp_config_path.exists()
        && let Ok(content) = std::fs::read_to_string(&mcp_config_path)
        && let Ok(mut config) = serde_json::from_str::<serde_json::Value>(&content)
    {
        let removed = config
            .get_mut("mcpServers")
            .and_then(|s| s.as_object_mut())
            .map(|s| s.remove("isartor").is_some())
            .unwrap_or(false);

        if removed && !args.base.dry_run {
            if let Ok(json) = serde_json::to_string_pretty(&config) {
                let _ = std::fs::write(&mcp_config_path, json);
            }
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileModified,
                target: mcp_config_path.to_string_lossy().to_string(),
                description: "Removed isartor MCP server".to_string(),
            });
        }
    }

    // Also clean up any legacy files.
    if !args.base.dry_run {
        cleanup_legacy_files(changes);
    }
    remove_copilot_instructions(changes, args.base.dry_run);

    ConnectResult {
        client_name: "GitHub Copilot CLI".to_string(),
        success: true,
        message: "Copilot CLI disconnected from Isartor.\n\
                  The isartor MCP server has been removed from ~/.copilot/mcp-config.json,\n\
                  and the managed Isartor block was removed from ~/.copilot/copilot-instructions.md."
            .to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ISARTOR_INSTRUCTION_END, ISARTOR_INSTRUCTION_START, managed_instruction_block,
        remove_managed_block, upsert_managed_block,
    };

    #[test]
    fn upsert_appends_block_when_missing() {
        let existing = "# User instructions\n\nKeep answers concise.\n";
        let updated = upsert_managed_block(existing, &managed_instruction_block());
        assert!(updated.contains(ISARTOR_INSTRUCTION_START));
        assert!(updated.contains(ISARTOR_INSTRUCTION_END));
        assert!(updated.starts_with("# User instructions"));
    }

    #[test]
    fn upsert_replaces_existing_managed_block() {
        let existing =
            format!("# Header\n\n{ISARTOR_INSTRUCTION_START}\nold\n{ISARTOR_INSTRUCTION_END}\n");
        let updated = upsert_managed_block(&existing, &managed_instruction_block());
        assert_eq!(updated.matches(ISARTOR_INSTRUCTION_START).count(), 1);
        assert!(!updated.contains("\nold\n"));
    }

    #[test]
    fn remove_preserves_user_content() {
        let existing = format!(
            "# Header\n\n{ISARTOR_INSTRUCTION_START}\nmanaged\n{ISARTOR_INSTRUCTION_END}\n\n# Footer\n"
        );
        let updated = remove_managed_block(&existing);
        assert!(!updated.contains(ISARTOR_INSTRUCTION_START));
        assert!(updated.contains("# Header"));
        assert!(updated.contains("# Footer"));
    }
}
