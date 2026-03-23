use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::Parser;
use serde_json::Value;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, TestResult, home_path,
    remove_file, test_isartor_connection, write_file,
};
use crate::providers::copilot::{CopilotAgent, DeviceFlowResult};

const TOKEN_PATH: &str = ".isartor/providers/copilot.json";
const CLAUDE_SETTINGS_PATH: &str = ".claude/settings.json";
const CLAUDE_SETTINGS_BACKUP_PATH: &str = ".claude/settings.json.claude-copilot-backup";
const CONFIG_PATH: &str = "isartor.toml";
const CONFIG_BACKUP_PATH: &str = "isartor.toml.claude-copilot-backup";
const MAX_OUTPUT_TOKENS: &str = "16000";

#[derive(Parser, Debug, Clone)]
pub struct ClaudeCopilotArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// GitHub personal access token (ghp_... or gho_...).
    /// If omitted, starts interactive browser authentication.
    #[arg(long, env = "GITHUB_TOKEN")]
    pub github_token: Option<String>,

    /// Primary model to use via GitHub Copilot.
    #[arg(long, default_value = "claude-sonnet-4.5")]
    pub model: String,

    /// Fast/background model to use via GitHub Copilot.
    #[arg(long, default_value = "gpt-4o-mini")]
    pub fast_model: String,

    /// Skip validating the GitHub token against the Copilot subscription endpoint.
    #[arg(long, default_value_t = false)]
    pub skip_validation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedGithubCredential {
    github_token: String,
    auth_type: String,
    token_type: Option<String>,
    scope: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SavedGithubCredential {
    github_token: String,
    #[serde(default)]
    auth_type: Option<String>,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

pub async fn handle_claude_copilot_connect(args: ClaudeCopilotArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();
    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_claude_copilot(&args, &mut changes);
    }

    let http = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            return ConnectResult {
                client_name: "Claude Code + GitHub Copilot".to_string(),
                success: false,
                message: format!("Failed to create HTTP client: {err}"),
                changes_made: vec![],
                test_result: None,
            };
        }
    };

    let credential = match resolve_github_credential(&args, &http).await {
        Ok(credential) => credential,
        Err(err) => {
            return ConnectResult {
                client_name: "Claude Code + GitHub Copilot".to_string(),
                success: false,
                message: format!(
                    "GitHub authentication failed: {err}\n\n\
                     Browser sign-in (recommended):\n  isartor connect claude-copilot\n\n\
                     Or provide a PAT explicitly:\n  isartor connect claude-copilot --github-token ghp_..."
                ),
                changes_made: vec![],
                test_result: None,
            };
        }
    };

    if !args.skip_validation
        && let Err(err) = CopilotAgent::validate_github_token(&http, &credential.github_token).await
    {
        return ConnectResult {
            client_name: "Claude Code + GitHub Copilot".to_string(),
            success: false,
            message: format!(
                "GitHub Copilot validation failed: {err}\n\n\
                 If you are on Copilot Business / Enterprise, browser login usually works better than a PAT:\n\
                   isartor connect claude-copilot\n\n\
                 Make sure the account has an active Copilot plan or seat assignment:\n\
                   https://github.com/features/copilot"
            ),
            changes_made: vec![],
            test_result: None,
        };
    }

    if let Err(err) = write_copilot_token_file(&credential, &args, &mut changes) {
        return ConnectResult {
            client_name: "Claude Code + GitHub Copilot".to_string(),
            success: false,
            message: format!("Failed to store Copilot token: {err}"),
            changes_made: changes,
            test_result: None,
        };
    }

    if let Err(err) = write_runtime_config(&credential.github_token, &args, &mut changes) {
        return ConnectResult {
            client_name: "Claude Code + GitHub Copilot".to_string(),
            success: false,
            message: format!("Failed to configure Isartor L3 runtime: {err}"),
            changes_made: changes,
            test_result: None,
        };
    }

    if let Err(err) = write_claude_settings(&gateway, gateway_key.as_deref(), &args, &mut changes) {
        return ConnectResult {
            client_name: "Claude Code + GitHub Copilot".to_string(),
            success: false,
            message: format!("Failed to configure Claude Code: {err}"),
            changes_made: changes,
            test_result: None,
        };
    }

    // Auto-restart Isartor so it picks up the new Copilot provider config.
    // Without this, the running instance still has the old provider and the
    // test request would hit the stale agent (often OpenAI with empty key → 401).
    let restarted = restart_isartor_if_running();

    let gateway_test = test_isartor_connection(
        &gateway,
        gateway_key.as_deref(),
        "reply with the single word: hello",
    )
    .await;

    let got_response = gateway_test.response_received;

    let test_result = if got_response || gateway_test.layer_resolved == "timeout"
    {
        Some(gateway_test)
    } else {
        Some(TestResult {
            request_sent: "Claude Code → /v1/messages".to_string(),
            response_received: false,
            layer_resolved: if restarted {
                "restart-failed".to_string()
            } else {
                "not-running".to_string()
            },
            latency_ms: 0,
            deflected: false,
        })
    };

    let next_steps = if got_response {
        format!(
            "Claude Code is configured to route through Isartor, and Isartor is configured to use GitHub Copilot for Layer 3.\n\n\
             Start a fresh Claude Code session:\n\
               claude\n\n\
             Runtime routing:\n\
               • Claude Code → {gateway}/v1/messages\n\
               • Cache hits (L1a/L1b) stop locally and consume 0 Copilot quota\n\
               • Cache misses route to api.githubcopilot.com using model {}",
            args.model
        )
    } else {
        format!(
            "Claude Code is configured to route through Isartor, and Isartor is configured to use GitHub Copilot for Layer 3.\n\n\
             Next steps:\n\
               1. Start Isartor\n\
                  isartor up --detach\n\
               2. Start a fresh Claude Code session\n\
                  claude\n\n\
             Runtime routing:\n\
               • Claude Code → {gateway}/v1/messages\n\
               • Cache hits (L1a/L1b) stop locally and consume 0 Copilot quota\n\
               • Cache misses route to api.githubcopilot.com using model {}",
            args.model
        )
    };

    ConnectResult {
        client_name: "Claude Code + GitHub Copilot".to_string(),
        success: true,
        message: format!(
            "{next_steps}\n\n\
             Files updated:\n\
               • ~/.claude/settings.json\n\
               • ./isartor.toml\n\
               • ~/.isartor/providers/copilot.json"
        ),
        changes_made: changes,
        test_result,
    }
}

async fn resolve_github_credential(
    args: &ClaudeCopilotArgs,
    http: &reqwest::Client,
) -> anyhow::Result<ResolvedGithubCredential> {
    if let Some(token) = args.github_token.clone() {
        return Ok(ResolvedGithubCredential {
            github_token: token,
            auth_type: "pat".to_string(),
            token_type: None,
            scope: None,
        });
    }

    if let Some(saved) = read_saved_github_credential()?
        && should_reuse_saved_credential(&saved)
    {
        if args.skip_validation
            || CopilotAgent::validate_github_token(http, &saved.github_token)
                .await
                .is_ok()
        {
            return Ok(ResolvedGithubCredential {
                github_token: saved.github_token,
                auth_type: saved.auth_type.unwrap_or_else(|| "oauth".to_string()),
                token_type: saved.token_type,
                scope: saved.scope,
            });
        }

        eprintln!(
            "Saved GitHub OAuth token is no longer valid for Copilot. Starting browser login..."
        );
    }

    let device = CopilotAgent::device_flow_auth(http).await?;
    Ok(ResolvedGithubCredential::from_device_flow(device))
}

impl ResolvedGithubCredential {
    fn from_device_flow(device: DeviceFlowResult) -> Self {
        Self {
            github_token: device.github_token,
            auth_type: "oauth".to_string(),
            token_type: Some(device.token_type),
            scope: Some(device.scope),
        }
    }
}

fn read_saved_github_credential() -> anyhow::Result<Option<SavedGithubCredential>> {
    let path = home_path(TOKEN_PATH)?;
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed: SavedGithubCredential = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(parsed))
}

fn should_reuse_saved_credential(saved: &SavedGithubCredential) -> bool {
    matches!(saved.auth_type.as_deref(), Some("oauth")) && !saved.github_token.is_empty()
}

/// Restart a running Isartor so it picks up the new config.
///
/// Returns `true` if an instance was found and restart was attempted.
fn restart_isartor_if_running() -> bool {
    use crate::cli::stop::pid_file_path;

    let pid_path = match pid_file_path() {
        Ok(p) => p,
        Err(_) => return false,
    };
    if !pid_path.exists() {
        return false;
    }

    eprintln!("  ↻ Restarting Isartor to load new Copilot provider config...");

    // Stop the old instance.
    let stop_result = std::process::Command::new(std::env::current_exe().unwrap_or("isartor".into()))
        .args(["stop"])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();

    if let Ok(status) = stop_result
        && !status.success()
    {
        eprintln!("  ⚠ Failed to stop Isartor (exit {}). You may need to restart manually.", status);
        return true;
    }

    // Brief pause for port release.
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Start fresh with --detach.
    let up_result = std::process::Command::new(std::env::current_exe().unwrap_or("isartor".into()))
        .args(["up", "--detach"])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();

    match up_result {
        Ok(status) if status.success() => {
            // Wait for the gateway to become ready.
            std::thread::sleep(std::time::Duration::from_secs(2));
            eprintln!("  ✓ Isartor restarted with Copilot provider.");
        }
        _ => {
            eprintln!("  ⚠ Failed to restart Isartor. Start it manually: isartor up --detach");
        }
    }

    true
}

fn write_copilot_token_file(
    credential: &ResolvedGithubCredential,
    args: &ClaudeCopilotArgs,
    changes: &mut Vec<ConfigChange>,
) -> anyhow::Result<()> {
    let path = home_path(TOKEN_PATH)?;
    let existed = path.exists();
    let mut payload = serde_json::json!({
        "provider": "copilot",
        "github_token": credential.github_token,
        "auth_type": credential.auth_type,
    });
    if let Some(token_type) = &credential.token_type {
        payload["token_type"] = Value::String(token_type.clone());
    }
    if let Some(scope) = &credential.scope {
        payload["scope"] = Value::String(scope.clone());
    }
    let content = serde_json::to_string_pretty(&payload)?;

    if args.base.show_config || args.base.dry_run {
        println!("--- {} ---\n{}\n", path.display(), content);
    }

    write_file(&path, &content, args.base.dry_run)?;
    changes.push(ConfigChange {
        change_type: if existed {
            ConfigChangeType::FileModified
        } else {
            ConfigChangeType::FileCreated
        },
        target: path.to_string_lossy().to_string(),
        description: format!(
            "Stored GitHub Copilot {} credential for local Isartor use",
            credential.auth_type
        ),
    });
    Ok(())
}

fn write_runtime_config(
    github_token: &str,
    args: &ClaudeCopilotArgs,
    changes: &mut Vec<ConfigChange>,
) -> anyhow::Result<()> {
    let config_path = PathBuf::from(CONFIG_PATH);
    let backup_path = PathBuf::from(CONFIG_BACKUP_PATH);
    let existed = config_path.exists();
    backup_file(&config_path, &backup_path, args.base.dry_run, changes)?;

    let existing = if config_path.exists() {
        std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?
    } else {
        String::new()
    };

    let mut doc = if existing.trim().is_empty() {
        toml_edit::DocumentMut::new()
    } else {
        existing
            .parse::<toml_edit::DocumentMut>()
            .with_context(|| format!("failed to parse {}", config_path.display()))?
    };

    doc["llm_provider"] = toml_edit::value("copilot");
    doc["external_llm_model"] = toml_edit::value(args.model.as_str());
    doc["external_llm_api_key"] = toml_edit::value(github_token);
    doc["external_llm_url"] = toml_edit::value("https://api.githubcopilot.com/chat/completions");

    let output = doc.to_string();

    if args.base.show_config || args.base.dry_run {
        println!("--- {} ---\n{}\n", config_path.display(), output);
    }

    write_file(&config_path, &output, args.base.dry_run)?;
    changes.push(ConfigChange {
        change_type: if existed {
            ConfigChangeType::FileModified
        } else {
            ConfigChangeType::FileCreated
        },
        target: config_path.to_string_lossy().to_string(),
        description: "Configured Isartor Layer 3 to use GitHub Copilot".to_string(),
    });
    Ok(())
}

fn write_claude_settings(
    gateway_url: &str,
    gateway_api_key: Option<&str>,
    args: &ClaudeCopilotArgs,
    changes: &mut Vec<ConfigChange>,
) -> anyhow::Result<()> {
    let settings_path = home_path(CLAUDE_SETTINGS_PATH)?;
    let backup_path = home_path(CLAUDE_SETTINGS_BACKUP_PATH)?;
    backup_file(&settings_path, &backup_path, args.base.dry_run, changes)?;

    let mut existing = read_json_file_or_default(&settings_path)?;
    let existed = settings_path.exists();

    if existing.get("env").and_then(Value::as_object).is_none() {
        existing["env"] = serde_json::json!({});
    }
    let env = existing["env"].as_object_mut().expect("env just created");

    env.insert(
        "ANTHROPIC_BASE_URL".to_string(),
        Value::String(gateway_url.to_string()),
    );
    env.insert(
        "ANTHROPIC_AUTH_TOKEN".to_string(),
        Value::String(
            gateway_api_key
                .filter(|key| !key.is_empty())
                .unwrap_or("dummy")
                .to_string(),
        ),
    );
    env.insert(
        "ANTHROPIC_MODEL".to_string(),
        Value::String(args.model.clone()),
    );
    env.insert(
        "ANTHROPIC_DEFAULT_SONNET_MODEL".to_string(),
        Value::String(args.model.clone()),
    );
    env.insert(
        "ANTHROPIC_DEFAULT_HAIKU_MODEL".to_string(),
        Value::String(args.fast_model.clone()),
    );
    env.insert(
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS".to_string(),
        Value::String("1".to_string()),
    );
    env.insert(
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".to_string(),
        Value::String("1".to_string()),
    );
    env.insert(
        "ENABLE_TOOL_SEARCH".to_string(),
        Value::String("true".to_string()),
    );
    env.insert(
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS".to_string(),
        Value::String(MAX_OUTPUT_TOKENS.to_string()),
    );

    let content = serde_json::to_string_pretty(&existing)?;
    if args.base.show_config || args.base.dry_run {
        println!("--- {} ---\n{}\n", settings_path.display(), content);
    }

    write_file(&settings_path, &content, args.base.dry_run)?;
    changes.push(ConfigChange {
        change_type: if existed {
            ConfigChangeType::FileModified
        } else {
            ConfigChangeType::FileCreated
        },
        target: settings_path.to_string_lossy().to_string(),
        description: "Configured Claude Code to use Isartor + GitHub Copilot".to_string(),
    });

    Ok(())
}

fn disconnect_claude_copilot(
    args: &ClaudeCopilotArgs,
    changes: &mut Vec<ConfigChange>,
) -> ConnectResult {
    let settings_path =
        home_path(CLAUDE_SETTINGS_PATH).unwrap_or_else(|_| PathBuf::from(CLAUDE_SETTINGS_PATH));
    let settings_backup_path = home_path(CLAUDE_SETTINGS_BACKUP_PATH)
        .unwrap_or_else(|_| PathBuf::from(CLAUDE_SETTINGS_BACKUP_PATH));
    let config_path = PathBuf::from(CONFIG_PATH);
    let config_backup_path = PathBuf::from(CONFIG_BACKUP_PATH);

    let mut restore_errors = Vec::new();

    if let Err(err) = restore_file(
        &settings_path,
        &settings_backup_path,
        args.base.dry_run,
        changes,
        "Restored Claude Code settings",
    ) {
        restore_errors.push(err.to_string());
    }
    if let Err(err) = restore_file(
        &config_path,
        &config_backup_path,
        args.base.dry_run,
        changes,
        "Restored Isartor runtime config",
    ) {
        restore_errors.push(err.to_string());
    }

    let message = if restore_errors.is_empty() {
        "Disconnected Claude Code + GitHub Copilot integration.\n\n\
         Restart Isartor and start a fresh `claude` session to restore the previous routing."
            .to_string()
    } else {
        format!(
            "Disconnected with warnings:\n{}\n\n\
             If needed, restore backups manually from:\n  {}\n  {}",
            restore_errors.join("\n"),
            settings_backup_path.display(),
            config_backup_path.display()
        )
    };

    ConnectResult {
        client_name: "Claude Code + GitHub Copilot".to_string(),
        success: restore_errors.is_empty(),
        message,
        changes_made: changes.clone(),
        test_result: None,
    }
}

fn read_json_file_or_default(path: &Path) -> anyhow::Result<Value> {
    if !path.exists() {
        return Ok(serde_json::json!({}));
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if content.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(&content).with_context(|| format!("failed to parse {}", path.display()))
}

fn backup_file(
    original_path: &Path,
    backup_path: &Path,
    dry_run: bool,
    changes: &mut Vec<ConfigChange>,
) -> anyhow::Result<()> {
    if backup_path.exists() {
        return Ok(());
    }

    let content = if original_path.exists() {
        std::fs::read_to_string(original_path)
            .with_context(|| format!("failed to read {}", original_path.display()))?
    } else {
        String::new()
    };

    write_file(backup_path, &content, dry_run)?;
    changes.push(ConfigChange {
        change_type: ConfigChangeType::FileBackedUp,
        target: backup_path.to_string_lossy().to_string(),
        description: format!("Backup for {}", original_path.display()),
    });
    Ok(())
}

fn restore_file(
    original_path: &Path,
    backup_path: &Path,
    dry_run: bool,
    changes: &mut Vec<ConfigChange>,
    description: &str,
) -> anyhow::Result<()> {
    if !backup_path.exists() {
        return Ok(());
    }

    let content = std::fs::read_to_string(backup_path)
        .with_context(|| format!("failed to read {}", backup_path.display()))?;
    if content.is_empty() {
        remove_file(original_path, dry_run);
    } else {
        write_file(original_path, &content, dry_run)?;
    }
    remove_file(backup_path, dry_run);

    changes.push(ConfigChange {
        change_type: ConfigChangeType::FileModified,
        target: original_path.to_string_lossy().to_string(),
        description: description.to_string(),
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn writes_expected_claude_settings_keys() {
        let mut existing = serde_json::json!({"env": {"OTHER": "keep"}});
        let env = existing["env"].as_object_mut().unwrap();
        env.insert(
            "ANTHROPIC_BASE_URL".to_string(),
            Value::String("http://localhost:8080".to_string()),
        );
        env.insert(
            "ANTHROPIC_AUTH_TOKEN".to_string(),
            Value::String("dummy".to_string()),
        );
        env.insert(
            "ENABLE_TOOL_SEARCH".to_string(),
            Value::String("true".to_string()),
        );
        env.insert(
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS".to_string(),
            Value::String(MAX_OUTPUT_TOKENS.to_string()),
        );

        assert_eq!(existing["env"]["OTHER"], "keep");
        assert_eq!(
            existing["env"]["ANTHROPIC_BASE_URL"],
            "http://localhost:8080"
        );
        assert_eq!(existing["env"]["ANTHROPIC_AUTH_TOKEN"], "dummy");
        assert_eq!(existing["env"]["ENABLE_TOOL_SEARCH"], "true");
        assert_eq!(
            existing["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"],
            MAX_OUTPUT_TOKENS
        );
    }

    #[test]
    fn backup_and_restore_round_trip() {
        let dir = tempdir().unwrap();
        let original = dir.path().join("settings.json");
        let backup = dir.path().join("settings.json.backup");
        let mut changes = Vec::new();

        std::fs::write(&original, "{\"env\":{\"EXISTING\":\"1\"}}").unwrap();
        backup_file(&original, &backup, false, &mut changes).unwrap();
        std::fs::write(&original, "{\"env\":{\"MODIFIED\":\"1\"}}").unwrap();

        restore_file(&original, &backup, false, &mut changes, "restore").unwrap();

        let restored = std::fs::read_to_string(&original).unwrap();
        assert!(restored.contains("EXISTING"));
        assert!(!backup.exists());
    }

    #[test]
    fn backup_of_missing_file_restores_to_absent() {
        let dir = tempdir().unwrap();
        let original = dir.path().join("missing.json");
        let backup = dir.path().join("missing.json.backup");
        let mut changes = Vec::new();

        backup_file(&original, &backup, false, &mut changes).unwrap();
        std::fs::write(&original, "{}").unwrap();

        restore_file(&original, &backup, false, &mut changes, "restore").unwrap();

        assert!(!original.exists());
        assert!(!backup.exists());
    }

    #[test]
    fn reuses_saved_oauth_credentials_only() {
        let oauth = SavedGithubCredential {
            github_token: "token".to_string(),
            auth_type: Some("oauth".to_string()),
            token_type: Some("bearer".to_string()),
            scope: Some("read:user".to_string()),
        };
        let legacy_pat = SavedGithubCredential {
            github_token: "token".to_string(),
            auth_type: None,
            token_type: None,
            scope: None,
        };
        let pat = SavedGithubCredential {
            github_token: "token".to_string(),
            auth_type: Some("pat".to_string()),
            token_type: None,
            scope: None,
        };

        assert!(should_reuse_saved_credential(&oauth));
        assert!(!should_reuse_saved_credential(&legacy_pat));
        assert!(!should_reuse_saved_credential(&pat));
    }

    #[test]
    fn device_flow_credentials_are_stored_as_oauth() {
        let credential = ResolvedGithubCredential::from_device_flow(DeviceFlowResult {
            github_token: "abc".to_string(),
            token_type: "bearer".to_string(),
            scope: "read:user".to_string(),
        });

        assert_eq!(credential.auth_type, "oauth");
        assert_eq!(credential.token_type.as_deref(), Some("bearer"));
        assert_eq!(credential.scope.as_deref(), Some("read:user"));
    }
}
