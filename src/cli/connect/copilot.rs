use clap::Parser;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};

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

    // Clean up legacy proxy-era env files (from v0.1.33 and earlier).
    if !args.base.dry_run {
        for ext in ["sh", "fish", "ps1"] {
            let legacy = home_path(&format!(".isartor/env/copilot.{ext}")).unwrap_or_default();
            if legacy.exists() {
                remove_file(&legacy, false);
                changes.push(ConfigChange {
                    change_type: ConfigChangeType::FileModified,
                    target: legacy.to_string_lossy().to_string(),
                    description: "Removed legacy proxy env file".to_string(),
                });
            }
        }
    }

    // Step 1: Write the preToolUse hook script.
    let hook_script = generate_hook_script(&gateway);
    let hook_path = home_path(".isartor/hooks/copilot_pretooluse.sh")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/hooks/copilot_pretooluse.sh"));

    if args.base.show_config || args.base.dry_run {
        println!("{}", hook_script);
    }

    if !args.base.dry_run {
        if let Some(parent) = hook_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if std::fs::write(&hook_path, &hook_script).is_ok() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ =
                    std::fs::set_permissions(&hook_path, std::fs::Permissions::from_mode(0o755));
            }
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileCreated,
                target: hook_path.to_string_lossy().to_string(),
                description: "preToolUse hook script".to_string(),
            });
        }
    }

    // Step 2: Write hook registration instructions.
    let instructions = format!(
        "# Isartor preToolUse hook for GitHub Copilot CLI\n\
         #\n\
         # Register this hook in your Copilot CLI config:\n\
         #   gh copilot config set preToolUse \"{}\"\n\
         #\n\
         # Or add to your Copilot CLI hooks config:\n\
         # {{\n\
         #   \"preToolUse\": \"{}\"\n\
         # }}\n",
        hook_path.display(),
        hook_path.display()
    );

    let instructions_path = home_path(".isartor/copilot-hook-setup.txt")
        .unwrap_or_else(|_| std::path::PathBuf::from(".isartor/copilot-hook-setup.txt"));

    if !args.base.dry_run && write_file(&instructions_path, &instructions, false).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: instructions_path.to_string_lossy().to_string(),
            description: "Hook registration instructions".to_string(),
        });
    }

    // Step 3: Store GitHub token if provided.
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

    // Step 4: Test the gateway API connection.
    let test = test_isartor_connection(
        &gateway,
        gateway_key.as_deref(),
        "What is the capital of France?",
    )
    .await;

    ConnectResult {
        client_name: "GitHub Copilot CLI".to_string(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "Hook script written to:\n  {}\n\n\
             IMPORTANT: Register the hook in your Copilot CLI config.\n\
             See instructions at:\n  {}\n\n\
             Copilot CLI integration uses preToolUse hooks (not HTTPS proxy).\n\
             This provides tool-call caching and audit logging.",
            hook_path.display(),
            instructions_path.display(),
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn generate_hook_script(gateway_url: &str) -> String {
    format!(
        r#"#!/usr/bin/env bash
# Isartor preToolUse hook for GitHub Copilot CLI
# Generated by: isartor connect copilot
# Do not edit manually.
#
# This hook is called by Copilot CLI before each tool use.
# It sends tool call metadata to Isartor for:
#   - Tool call result caching
#   - Audit logging
#   - Budget enforcement
#   - Policy enforcement

ISARTOR_URL="{}"
TOOL_NAME="${{1:-unknown}}"
TOOL_ARGS="${{2:-}}"

# POST tool call info to Isartor hook endpoint
RESPONSE=$(curl -s -X POST \
  "${{ISARTOR_URL}}/api/v1/hook/pretooluse" \
  -H "Content-Type: application/json" \
  -d "{{
    \"tool\": \"${{TOOL_NAME}}\",
    \"args\": \"${{TOOL_ARGS}}\",
    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
  }}" \
  --connect-timeout 1 \
  --max-time 2 \
  2>/dev/null)

# If Isartor says to block (policy violation):
if echo "${{RESPONSE}}" | grep -q '"action":"block"'; then
  REASON=$(echo "${{RESPONSE}}" | grep -o '"reason":"[^"]*"' \
    | sed 's/"reason":"//;s/"//')
  echo "Isartor policy: ${{REASON}}" >&2
  exit 1
fi

# If Isartor has a cached result, print it
if echo "${{RESPONSE}}" | grep -q '"cached":true'; then
  echo "${{RESPONSE}}" | grep -o '"result":"[^"]*"' \
    | sed 's/"result":"//;s/"$//'
  exit 0
fi

# Otherwise: allow the tool call to proceed normally
exit 0
"#,
        gateway_url
    )
}

fn disconnect(args: &CopilotArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    // Remove hook script and instructions.
    for filename in [
        ".isartor/hooks/copilot_pretooluse.sh",
        ".isartor/copilot-hook-setup.txt",
    ] {
        let path = home_path(filename).unwrap_or_else(|_| std::path::PathBuf::from(filename));
        if path.exists() {
            remove_file(&path, args.base.dry_run);
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileModified,
                target: path.to_string_lossy().to_string(),
                description: "Removed".to_string(),
            });
        }
    }

    // Also remove legacy shell env files from the proxy era.
    for ext in ["sh", "fish", "ps1"] {
        let path = home_path(&format!(".isartor/env/copilot.{ext}"))
            .unwrap_or_else(|_| std::path::PathBuf::from(format!(".isartor/env/copilot.{ext}")));
        if path.exists() {
            remove_file(&path, args.base.dry_run);
            changes.push(ConfigChange {
                change_type: ConfigChangeType::FileModified,
                target: path.to_string_lossy().to_string(),
                description: "Removed (legacy proxy env)".to_string(),
            });
        }
    }

    ConnectResult {
        client_name: "GitHub Copilot CLI".to_string(),
        success: true,
        message: "Copilot CLI hooks disconnected.\n\
                  Remember to remove the hook registration from your Copilot CLI config."
            .to_string(),
        changes_made: changes.clone(),
        test_result: None,
    }
}
