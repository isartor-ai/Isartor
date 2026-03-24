use std::path::Path;

use anyhow::Context;
use clap::Parser;
use serde_json::{Map, Value};

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};
use crate::config::{AppConfig, LlmProvider};

const OPENCODE_CONFIG_PATH: &str = ".config/opencode/opencode.json";
const OPENCODE_CONFIG_BACKUP_PATH: &str = ".config/opencode/opencode.json.isartor-backup";
const OPENCODE_AUTH_PATH: &str = ".local/share/opencode/auth.json";
const OPENCODE_AUTH_BACKUP_PATH: &str = ".local/share/opencode/auth.json.isartor-backup";
const OPENCODE_SCHEMA_URL: &str = "https://opencode.ai/config.json";
const OPENCODE_PROVIDER_ID: &str = "isartor";
const DEFAULT_OPENCODE_MODEL: &str = "gpt-4o-mini";
const DUMMY_AUTH_KEY: &str = "isartor-local";

#[derive(Parser, Debug, Clone)]
pub struct OpencodeArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// Primary model ID to register in OpenCode (defaults to Isartor's configured model)
    #[arg(long)]
    pub model: Option<String>,

    /// Lightweight model ID for OpenCode background tasks (defaults to --model)
    #[arg(long)]
    pub small_model: Option<String>,
}

pub async fn handle_opencode_connect(args: OpencodeArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();
    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_opencode(&args, &mut changes);
    }

    let config_path = home_path(OPENCODE_CONFIG_PATH)
        .unwrap_or_else(|_| std::path::PathBuf::from(OPENCODE_CONFIG_PATH));
    let auth_path = home_path(OPENCODE_AUTH_PATH)
        .unwrap_or_else(|_| std::path::PathBuf::from(OPENCODE_AUTH_PATH));
    let config_backup_path = home_path(OPENCODE_CONFIG_BACKUP_PATH)
        .unwrap_or_else(|_| std::path::PathBuf::from(OPENCODE_CONFIG_BACKUP_PATH));
    let auth_backup_path = home_path(OPENCODE_AUTH_BACKUP_PATH)
        .unwrap_or_else(|_| std::path::PathBuf::from(OPENCODE_AUTH_BACKUP_PATH));

    if let Err(err) = backup_file(
        &config_path,
        &config_backup_path,
        args.base.dry_run,
        &mut changes,
    ) {
        return failure_result(
            "OpenCode",
            format!("Failed to back up OpenCode config: {err}"),
            changes,
        );
    }
    if let Err(err) = backup_file(
        &auth_path,
        &auth_backup_path,
        args.base.dry_run,
        &mut changes,
    ) {
        return failure_result(
            "OpenCode",
            format!("Failed to back up OpenCode auth: {err}"),
            changes,
        );
    }

    let base_url = format!("{}/v1", gateway.trim_end_matches('/'));
    let model = args.model.clone().unwrap_or_else(resolve_default_model);
    let small_model = args.small_model.clone().unwrap_or_else(|| model.clone());
    let auth_key = gateway_key
        .clone()
        .filter(|key| !key.trim().is_empty())
        .unwrap_or_else(|| DUMMY_AUTH_KEY.to_string());

    let config_exists = config_path.exists();
    let auth_exists = auth_path.exists();

    let config_json = serde_json::to_string_pretty(&build_opencode_config(
        read_json_or_empty(&config_path),
        &base_url,
        &model,
        &small_model,
    ))
    .unwrap_or_default();
    let auth_json = serde_json::to_string_pretty(&build_opencode_auth(
        read_json_or_empty(&auth_path),
        &auth_key,
    ))
    .unwrap_or_default();

    if args.base.show_config || args.base.dry_run {
        println!("--- {} ---\n{}\n", config_path.display(), config_json);
        println!("--- {} ---\n{}\n", auth_path.display(), auth_json);
    }

    if let Err(err) = write_file(&config_path, &config_json, args.base.dry_run) {
        return failure_result(
            "OpenCode",
            format!("Failed to write OpenCode config: {err}"),
            changes,
        );
    }
    changes.push(ConfigChange {
        change_type: if config_exists {
            ConfigChangeType::FileModified
        } else {
            ConfigChangeType::FileCreated
        },
        target: config_path.to_string_lossy().to_string(),
        description: "Configured OpenCode to use Isartor via @ai-sdk/openai-compatible".to_string(),
    });

    if let Err(err) = write_file(&auth_path, &auth_json, args.base.dry_run) {
        return failure_result(
            "OpenCode",
            format!("Failed to write OpenCode auth config: {err}"),
            changes,
        );
    }
    changes.push(ConfigChange {
        change_type: if auth_exists {
            ConfigChangeType::FileModified
        } else {
            ConfigChangeType::FileCreated
        },
        target: auth_path.to_string_lossy().to_string(),
        description: "Stored OpenCode auth entry for Isartor".to_string(),
    });

    let test =
        test_isartor_connection(&gateway, gateway_key.as_deref(), "Hello from OpenCode test").await;
    let provider_model = format!("{OPENCODE_PROVIDER_ID}/{model}");
    let provider_small_model = format!("{OPENCODE_PROVIDER_ID}/{small_model}");

    ConnectResult {
        client_name: "OpenCode".to_string(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "OpenCode is configured to use Isartor.\n\
             \n\
             1. Start Isartor:  isartor up\n\
             2. Start OpenCode: opencode\n\
             \n\
             Provider:  {OPENCODE_PROVIDER_ID}\n\
             Base URL:  {base_url}\n\
             Model:     {provider_model}\n\
             Small:     {provider_small_model}\n\
             Config:    {}\n\
             Auth:      {}\n\
             \n\
             Disconnect with: isartor connect opencode --disconnect",
            config_path.display(),
            auth_path.display(),
            provider_model = provider_model,
            provider_small_model = provider_small_model,
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect_opencode(args: &OpencodeArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    let config_path = home_path(OPENCODE_CONFIG_PATH)
        .unwrap_or_else(|_| std::path::PathBuf::from(OPENCODE_CONFIG_PATH));
    let auth_path = home_path(OPENCODE_AUTH_PATH)
        .unwrap_or_else(|_| std::path::PathBuf::from(OPENCODE_AUTH_PATH));
    let config_backup_path = home_path(OPENCODE_CONFIG_BACKUP_PATH)
        .unwrap_or_else(|_| std::path::PathBuf::from(OPENCODE_CONFIG_BACKUP_PATH));
    let auth_backup_path = home_path(OPENCODE_AUTH_BACKUP_PATH)
        .unwrap_or_else(|_| std::path::PathBuf::from(OPENCODE_AUTH_BACKUP_PATH));

    let mut restore_errors = Vec::new();
    let mut changed = false;

    if config_backup_path.exists() {
        changed = true;
        if let Err(err) = restore_file(
            &config_path,
            &config_backup_path,
            args.base.dry_run,
            changes,
            "Restored OpenCode config from backup",
        ) {
            restore_errors.push(err.to_string());
        }
    } else {
        changed |= remove_isartor_provider(&config_path, args.base.dry_run, changes);
    }

    if auth_backup_path.exists() {
        changed = true;
        if let Err(err) = restore_file(
            &auth_path,
            &auth_backup_path,
            args.base.dry_run,
            changes,
            "Restored OpenCode auth from backup",
        ) {
            restore_errors.push(err.to_string());
        }
    } else {
        changed |= remove_isartor_auth(&auth_path, args.base.dry_run, changes);
    }

    let message = if restore_errors.is_empty() {
        if changed {
            "OpenCode disconnected from Isartor. Existing config/auth files were restored or cleaned up."
                .to_string()
        } else {
            "OpenCode was not connected to Isartor. No managed files needed changes.".to_string()
        }
    } else {
        format!(
            "Failed to fully disconnect OpenCode:\n{}\n\nIf needed, restore backups manually from:\n  {}\n  {}",
            restore_errors.join("\n"),
            config_backup_path.display(),
            auth_backup_path.display(),
        )
    };

    ConnectResult {
        client_name: "OpenCode".to_string(),
        success: restore_errors.is_empty(),
        message,
        changes_made: changes.clone(),
        test_result: None,
    }
}

fn failure_result(client_name: &str, message: String, changes: Vec<ConfigChange>) -> ConnectResult {
    ConnectResult {
        client_name: client_name.to_string(),
        success: false,
        message,
        changes_made: changes,
        test_result: None,
    }
}

fn resolve_default_model() -> String {
    AppConfig::load_with_validation(false)
        .ok()
        .map(|cfg| match cfg.llm_provider {
            LlmProvider::Azure if !cfg.azure_deployment_id.trim().is_empty() => {
                cfg.azure_deployment_id
            }
            _ if !cfg.external_llm_model.trim().is_empty() => cfg.external_llm_model,
            _ => DEFAULT_OPENCODE_MODEL.to_string(),
        })
        .unwrap_or_else(|| DEFAULT_OPENCODE_MODEL.to_string())
}

fn build_opencode_config(existing: Value, base_url: &str, model: &str, small_model: &str) -> Value {
    let mut root = as_object(existing);
    root.insert(
        "$schema".to_string(),
        Value::String(OPENCODE_SCHEMA_URL.to_string()),
    );

    let mut provider_map = root
        .remove("provider")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    provider_map.insert(
        OPENCODE_PROVIDER_ID.to_string(),
        Value::Object(build_provider_definition(base_url, model, small_model)),
    );
    root.insert("provider".to_string(), Value::Object(provider_map));
    root.insert(
        "model".to_string(),
        Value::String(format!("{OPENCODE_PROVIDER_ID}/{model}")),
    );
    root.insert(
        "small_model".to_string(),
        Value::String(format!("{OPENCODE_PROVIDER_ID}/{small_model}")),
    );

    Value::Object(root)
}

fn build_provider_definition(base_url: &str, model: &str, small_model: &str) -> Map<String, Value> {
    let mut models = Map::new();
    models.insert(
        model.to_string(),
        serde_json::json!({ "name": format!("Isartor ({model})") }),
    );
    if small_model != model {
        models.insert(
            small_model.to_string(),
            serde_json::json!({ "name": format!("Isartor ({small_model})") }),
        );
    }

    let mut provider = Map::new();
    provider.insert(
        "npm".to_string(),
        Value::String("@ai-sdk/openai-compatible".to_string()),
    );
    provider.insert("name".to_string(), Value::String("Isartor".to_string()));
    provider.insert(
        "options".to_string(),
        serde_json::json!({
            "baseURL": base_url,
        }),
    );
    provider.insert("models".to_string(), Value::Object(models));
    provider
}

fn build_opencode_auth(existing: Value, api_key: &str) -> Value {
    let mut root = as_object(existing);
    root.insert(
        OPENCODE_PROVIDER_ID.to_string(),
        serde_json::json!({
            "type": "api",
            "key": api_key,
        }),
    );
    Value::Object(root)
}

fn remove_isartor_provider(path: &Path, dry_run: bool, changes: &mut Vec<ConfigChange>) -> bool {
    let mut root = match read_json_file(path) {
        Some(value) => value,
        None => return false,
    };

    let mut modified = false;
    if let Some(provider) = root
        .get_mut("provider")
        .and_then(|value| value.as_object_mut())
    {
        modified |= provider.remove(OPENCODE_PROVIDER_ID).is_some();
    }
    if root
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|value| value.starts_with(&format!("{OPENCODE_PROVIDER_ID}/")))
    {
        modified = true;
        if let Some(map) = root.as_object_mut() {
            map.remove("model");
        }
    }
    if root
        .get("small_model")
        .and_then(Value::as_str)
        .is_some_and(|value| value.starts_with(&format!("{OPENCODE_PROVIDER_ID}/")))
    {
        modified = true;
        if let Some(map) = root.as_object_mut() {
            map.remove("small_model");
        }
    }

    if modified {
        let content = serde_json::to_string_pretty(&root).unwrap_or_default();
        let _ = write_file(path, &content, dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: path.to_string_lossy().to_string(),
            description: "Removed Isartor-managed OpenCode provider".to_string(),
        });
    }

    modified
}

fn remove_isartor_auth(path: &Path, dry_run: bool, changes: &mut Vec<ConfigChange>) -> bool {
    let mut root = match read_json_file(path) {
        Some(value) => value,
        None => return false,
    };
    let removed = root
        .as_object_mut()
        .map(|map| map.remove(OPENCODE_PROVIDER_ID).is_some())
        .unwrap_or(false);
    if removed {
        let content = serde_json::to_string_pretty(&root).unwrap_or_default();
        let _ = write_file(path, &content, dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: path.to_string_lossy().to_string(),
            description: "Removed Isartor-managed OpenCode auth entry".to_string(),
        });
    }
    removed
}

fn read_json_or_empty(path: &Path) -> Value {
    read_json_file(path).unwrap_or_else(|| Value::Object(Map::new()))
}

fn read_json_file(path: &Path) -> Option<Value> {
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn as_object(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
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
    fn build_config_preserves_existing_fields_and_sets_provider() {
        let config = build_opencode_config(
            serde_json::json!({
                "provider": {
                    "other": {
                        "name": "Other"
                    }
                },
                "share": "manual"
            }),
            "http://localhost:8080/v1",
            "gpt-4o-mini",
            "gpt-4o-mini",
        );

        assert_eq!(config["$schema"], OPENCODE_SCHEMA_URL);
        assert_eq!(config["share"], "manual");
        assert_eq!(
            config["provider"]["isartor"]["npm"],
            "@ai-sdk/openai-compatible"
        );
        assert_eq!(
            config["provider"]["isartor"]["options"]["baseURL"],
            "http://localhost:8080/v1"
        );
        assert_eq!(config["provider"]["other"]["name"], "Other");
        assert_eq!(config["model"], "isartor/gpt-4o-mini");
        assert_eq!(config["small_model"], "isartor/gpt-4o-mini");
    }

    #[test]
    fn remove_helpers_only_strip_isartor_entries() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("opencode.json");
        let auth_path = dir.path().join("auth.json");
        let mut changes = Vec::new();

        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "provider": {
                    "isartor": { "name": "Isartor" },
                    "other": { "name": "Other" }
                },
                "model": "isartor/gpt-4o-mini",
                "small_model": "other/cheap-model"
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            &auth_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "isartor": { "type": "api", "key": "secret" },
                "other": { "type": "api", "key": "keep" }
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(remove_isartor_provider(&config_path, false, &mut changes));
        assert!(remove_isartor_auth(&auth_path, false, &mut changes));

        let config: Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        let auth: Value =
            serde_json::from_str(&std::fs::read_to_string(&auth_path).unwrap()).unwrap();

        assert!(config["provider"].get(OPENCODE_PROVIDER_ID).is_none());
        assert_eq!(config["provider"]["other"]["name"], "Other");
        assert!(config.get("model").is_none());
        assert_eq!(config["small_model"], "other/cheap-model");
        assert!(auth.get(OPENCODE_PROVIDER_ID).is_none());
        assert_eq!(auth["other"]["key"], "keep");
    }

    #[test]
    fn backup_and_restore_round_trip() {
        let dir = tempdir().unwrap();
        let original = dir.path().join("opencode.json");
        let backup = dir.path().join("opencode.json.isartor-backup");
        let mut changes = Vec::new();

        std::fs::write(&original, "{\"provider\":{\"other\":true}}").unwrap();
        backup_file(&original, &backup, false, &mut changes).unwrap();
        std::fs::write(&original, "{\"provider\":{\"isartor\":true}}").unwrap();

        restore_file(&original, &backup, false, &mut changes, "restore").unwrap();

        assert_eq!(
            std::fs::read_to_string(&original).unwrap(),
            "{\"provider\":{\"other\":true}}"
        );
        assert!(!backup.exists());
    }

    #[test]
    fn backup_of_missing_file_restores_to_absent() {
        let dir = tempdir().unwrap();
        let original = dir.path().join("missing.json");
        let backup = dir.path().join("missing.json.isartor-backup");
        let mut changes = Vec::new();

        backup_file(&original, &backup, false, &mut changes).unwrap();
        std::fs::write(&original, "{}").unwrap();

        restore_file(&original, &backup, false, &mut changes, "restore").unwrap();

        assert!(!original.exists());
        assert!(!backup.exists());
    }

    #[test]
    fn connect_and_disconnect_round_trip_files() {
        let dir = tempdir().unwrap();
        let home = dir.path().to_string_lossy().to_string();

        temp_env::with_vars([("HOME", Some(home.as_str()))], || {
            tokio_test::block_on(async {
                let config_path = dir.path().join(OPENCODE_CONFIG_PATH);
                let auth_path = dir.path().join(OPENCODE_AUTH_PATH);
                std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
                std::fs::create_dir_all(auth_path.parent().unwrap()).unwrap();
                std::fs::write(
                    &config_path,
                    serde_json::to_string_pretty(&serde_json::json!({
                        "provider": { "other": { "name": "Other" } },
                        "model": "other/model"
                    }))
                    .unwrap(),
                )
                .unwrap();
                std::fs::write(
                    &auth_path,
                    serde_json::to_string_pretty(&serde_json::json!({
                        "other": { "type": "api", "key": "keep" }
                    }))
                    .unwrap(),
                )
                .unwrap();

                let result = handle_opencode_connect(OpencodeArgs {
                    base: BaseClientArgs {
                        gateway_url: "http://127.0.0.1:9".to_string(),
                        gateway_api_key: Some("gateway-secret".to_string()),
                        disconnect: false,
                        dry_run: false,
                        show_config: false,
                    },
                    model: Some("gpt-4o-mini".to_string()),
                    small_model: Some("gpt-4o-mini".to_string()),
                })
                .await;

                assert!(!result.changes_made.is_empty());

                let config: Value =
                    serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
                let auth: Value =
                    serde_json::from_str(&std::fs::read_to_string(&auth_path).unwrap()).unwrap();
                assert_eq!(
                    config["provider"]["isartor"]["options"]["baseURL"],
                    "http://127.0.0.1:9/v1"
                );
                assert_eq!(config["model"], "isartor/gpt-4o-mini");
                assert_eq!(auth["isartor"]["key"], "gateway-secret");
                assert!(dir.path().join(OPENCODE_CONFIG_BACKUP_PATH).exists());
                assert!(dir.path().join(OPENCODE_AUTH_BACKUP_PATH).exists());

                let disconnect = handle_opencode_connect(OpencodeArgs {
                    base: BaseClientArgs {
                        gateway_url: "http://127.0.0.1:9".to_string(),
                        gateway_api_key: Some("gateway-secret".to_string()),
                        disconnect: true,
                        dry_run: false,
                        show_config: false,
                    },
                    model: None,
                    small_model: None,
                })
                .await;

                assert!(disconnect.success);
                let restored_config: Value =
                    serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
                let restored_auth: Value =
                    serde_json::from_str(&std::fs::read_to_string(&auth_path).unwrap()).unwrap();
                assert!(restored_config["provider"].get("isartor").is_none());
                assert_eq!(restored_config["provider"]["other"]["name"], "Other");
                assert_eq!(restored_config["model"], "other/model");
                assert!(restored_auth.get("isartor").is_none());
                assert_eq!(restored_auth["other"]["key"], "keep");
                assert!(!dir.path().join(OPENCODE_CONFIG_BACKUP_PATH).exists());
                assert!(!dir.path().join(OPENCODE_AUTH_BACKUP_PATH).exists());
            });
        });
    }
}
