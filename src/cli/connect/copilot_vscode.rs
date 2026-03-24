use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use clap::Parser;
use serde_json::{Map, Value};

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, remove_file,
    test_isartor_connection, write_file,
};

const ISARTOR_WARNING: &str = "Warning: github.copilot.advanced.debug.overrideProxyUrl, debug.overrideCAPIUrl, and debug.chatOverrideProxyUrl are unsupported internal GitHub Copilot settings. They may stop working or change on any Copilot extension update.";
const ADVANCED_SETTINGS_KEY: &str = "github.copilot.advanced";
const OVERRIDE_PROXY_URL_KEY: &str = "debug.overrideProxyUrl";
const OVERRIDE_CAPI_URL_KEY: &str = "debug.overrideCAPIUrl";
const CHAT_OVERRIDE_PROXY_URL_KEY: &str = "debug.chatOverrideProxyUrl";

#[derive(Parser, Debug, Clone)]
pub struct CopilotVscodeArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,
}

pub async fn handle_copilot_vscode_connect(args: CopilotVscodeArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();
    let mut changes = Vec::new();

    let settings_path = match detect_vscode_settings_path() {
        Ok(path) => path,
        Err(err) => {
            return ConnectResult {
                client_name: "GitHub Copilot in VS Code".to_string(),
                success: false,
                message: format!(
                    "Could not locate the VS Code user settings path: {err}\n\n{ISARTOR_WARNING}"
                ),
                changes_made: changes,
                test_result: None,
            };
        }
    };

    if args.base.disconnect {
        return disconnect_copilot_vscode(&args, &settings_path, &mut changes);
    }

    let preflight = test_isartor_connection(
        &gateway,
        gateway_key.as_deref(),
        "VS Code Copilot preflight check",
    )
    .await;
    if !(preflight.response_received || preflight.layer_resolved == "timeout") {
        return ConnectResult {
            client_name: "GitHub Copilot in VS Code".to_string(),
            success: false,
            message: format!(
                "Isartor is not reachable at {gateway}. Start it first (for example `isartor up --detach`) before writing VS Code settings.\n\nTarget settings file:\n  {}\n\n{ISARTOR_WARNING}",
                settings_path.display()
            ),
            changes_made: changes,
            test_result: Some(preflight),
        };
    }

    let mut settings = match read_settings_json(&settings_path) {
        Ok(settings) => settings,
        Err(err) => {
            return ConnectResult {
                client_name: "GitHub Copilot in VS Code".to_string(),
                success: false,
                message: format!(
                    "Failed to read VS Code settings from {}: {err}\n\n{ISARTOR_WARNING}",
                    settings_path.display()
                ),
                changes_made: changes,
                test_result: Some(preflight),
            };
        }
    };

    let backup_path = backup_path_for(&settings_path);
    if let Err(err) = backup_file(
        &settings_path,
        &backup_path,
        args.base.dry_run,
        &mut changes,
    ) {
        return ConnectResult {
            client_name: "GitHub Copilot in VS Code".to_string(),
            success: false,
            message: format!(
                "Failed to back up {}: {err}\n\n{ISARTOR_WARNING}",
                settings_path.display()
            ),
            changes_made: changes,
            test_result: Some(preflight),
        };
    }

    if let Err(err) = upsert_copilot_vscode_settings(&mut settings, &gateway) {
        return ConnectResult {
            client_name: "GitHub Copilot in VS Code".to_string(),
            success: false,
            message: format!(
                "Failed to prepare VS Code settings for {}: {err}\n\n{ISARTOR_WARNING}",
                settings_path.display()
            ),
            changes_made: changes,
            test_result: Some(preflight),
        };
    }

    let rendered = match serde_json::to_string_pretty(&settings) {
        Ok(json) => format!("{json}\n"),
        Err(err) => {
            return ConnectResult {
                client_name: "GitHub Copilot in VS Code".to_string(),
                success: false,
                message: format!(
                    "Failed to render VS Code settings for {}: {err}\n\n{ISARTOR_WARNING}",
                    settings_path.display()
                ),
                changes_made: changes,
                test_result: Some(preflight),
            };
        }
    };

    if args.base.show_config || args.base.dry_run {
        println!("--- {} ---\n{rendered}", settings_path.display());
    }

    let existed = settings_path.exists();
    if let Err(err) = write_file(&settings_path, &rendered, args.base.dry_run) {
        return ConnectResult {
            client_name: "GitHub Copilot in VS Code".to_string(),
            success: false,
            message: format!(
                "Failed to write VS Code settings to {}: {err}\n\n{ISARTOR_WARNING}",
                settings_path.display()
            ),
            changes_made: changes,
            test_result: Some(preflight),
        };
    }

    changes.push(ConfigChange {
        change_type: if existed {
            ConfigChangeType::FileModified
        } else {
            ConfigChangeType::FileCreated
        },
        target: settings_path.to_string_lossy().to_string(),
        description: "Configured GitHub Copilot VS Code debug overrides".to_string(),
    });

    ConnectResult {
        client_name: "GitHub Copilot in VS Code".to_string(),
        success: true,
        message: format!(
            "Configured GitHub Copilot in VS Code to route through Isartor.\n\nVS Code settings:\n  {}\nBackup:\n  {}\n\nReload VS Code with “Developer: Reload Window” or restart the editor.\n\n{}",
            settings_path.display(),
            backup_path.display(),
            ISARTOR_WARNING,
        ),
        changes_made: changes,
        test_result: Some(preflight),
    }
}

fn disconnect_copilot_vscode(
    args: &CopilotVscodeArgs,
    settings_path: &Path,
    changes: &mut Vec<ConfigChange>,
) -> ConnectResult {
    let backup_path = backup_path_for(settings_path);
    let outcome = if backup_path.exists() {
        restore_file(
            settings_path,
            &backup_path,
            args.base.dry_run,
            changes,
            "Restored VS Code settings from backup",
        )
        .map(|_| {
            "Restored the original VS Code settings from backup. Reload VS Code to apply."
                .to_string()
        })
    } else {
        remove_isartor_settings(settings_path, args.base.dry_run, changes).map(|removed| {
            if removed {
                "Removed Isartor GitHub Copilot VS Code debug overrides. Reload VS Code to apply."
                    .to_string()
            } else {
                "No Isartor GitHub Copilot VS Code overrides were present. Nothing to disconnect."
                    .to_string()
            }
        })
    };

    match outcome {
        Ok(message) => ConnectResult {
            client_name: "GitHub Copilot in VS Code".to_string(),
            success: true,
            message,
            changes_made: changes.clone(),
            test_result: None,
        },
        Err(err) => ConnectResult {
            client_name: "GitHub Copilot in VS Code".to_string(),
            success: false,
            message: format!(
                "Failed to disconnect GitHub Copilot in VS Code: {err}\n\n{ISARTOR_WARNING}"
            ),
            changes_made: changes.clone(),
            test_result: None,
        },
    }
}

fn detect_vscode_settings_path() -> anyhow::Result<PathBuf> {
    let home_dir = dirs::home_dir();
    let xdg_config_home = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
    let appdata = std::env::var_os("APPDATA").map(PathBuf::from);
    detect_vscode_settings_path_for(
        std::env::consts::OS,
        home_dir.as_deref(),
        xdg_config_home.as_deref(),
        appdata.as_deref(),
    )
}

fn detect_vscode_settings_path_for(
    os: &str,
    home_dir: Option<&Path>,
    xdg_config_home: Option<&Path>,
    appdata: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    match os {
        "macos" => home_dir
            .map(|home| home.join("Library/Application Support/Code/User/settings.json"))
            .ok_or_else(|| anyhow!("HOME is not set")),
        "linux" => Ok(match xdg_config_home {
            Some(xdg) => xdg.join("Code/User/settings.json"),
            None => home_dir
                .map(|home| home.join(".config/Code/User/settings.json"))
                .ok_or_else(|| anyhow!("HOME is not set and XDG_CONFIG_HOME is not set"))?,
        }),
        "windows" => {
            if let Some(appdata) = appdata {
                Ok(appdata.join("Code/User/settings.json"))
            } else if let Some(home) = home_dir {
                Ok(home.join("AppData/Roaming/Code/User/settings.json"))
            } else {
                Err(anyhow!("APPDATA is not set and HOME is not set"))
            }
        }
        other => Err(anyhow!("unsupported operating system: {other}")),
    }
}

fn read_settings_json(path: &Path) -> anyhow::Result<Value> {
    if !path.exists() {
        return Ok(serde_json::json!({}));
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if content.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }

    let value: Value =
        json5::from_str(&content).with_context(|| format!("failed to parse {}", path.display()))?;
    if !value.is_object() {
        return Err(anyhow!("expected a JSON object at {}", path.display()));
    }

    Ok(value)
}

fn upsert_copilot_vscode_settings(settings: &mut Value, gateway_url: &str) -> anyhow::Result<()> {
    let settings_obj = settings
        .as_object_mut()
        .context("VS Code settings must be a JSON object")?;
    let advanced = settings_obj
        .entry(ADVANCED_SETTINGS_KEY.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let advanced_obj = advanced
        .as_object_mut()
        .context("\"github.copilot.advanced\" must be an object")?;

    let gateway = gateway_url.trim_end_matches('/').to_string();
    advanced_obj.insert(
        OVERRIDE_PROXY_URL_KEY.to_string(),
        Value::String(gateway.clone()),
    );
    advanced_obj.insert(
        OVERRIDE_CAPI_URL_KEY.to_string(),
        Value::String(format!("{gateway}/v1")),
    );
    advanced_obj.insert(
        CHAT_OVERRIDE_PROXY_URL_KEY.to_string(),
        Value::String(format!("{gateway}/v1/chat/completions")),
    );

    Ok(())
}

fn remove_isartor_settings(
    settings_path: &Path,
    dry_run: bool,
    changes: &mut Vec<ConfigChange>,
) -> anyhow::Result<bool> {
    if !settings_path.exists() {
        return Ok(false);
    }

    let mut settings = read_settings_json(settings_path)?;
    let removed = remove_isartor_overrides(&mut settings)?;
    if !removed {
        return Ok(false);
    }

    let is_empty = settings
        .as_object()
        .map(|obj| obj.is_empty())
        .unwrap_or(false);
    if is_empty {
        remove_file(settings_path, dry_run);
    } else {
        let rendered = format!("{}\n", serde_json::to_string_pretty(&settings)?);
        write_file(settings_path, &rendered, dry_run)?;
    }

    changes.push(ConfigChange {
        change_type: ConfigChangeType::FileModified,
        target: settings_path.to_string_lossy().to_string(),
        description: "Removed GitHub Copilot VS Code debug overrides".to_string(),
    });

    Ok(true)
}

fn remove_isartor_overrides(settings: &mut Value) -> anyhow::Result<bool> {
    let settings_obj = settings
        .as_object_mut()
        .context("VS Code settings must be a JSON object")?;
    let Some(advanced) = settings_obj.get_mut(ADVANCED_SETTINGS_KEY) else {
        return Ok(false);
    };

    let advanced_obj = advanced
        .as_object_mut()
        .context("\"github.copilot.advanced\" must be an object")?;

    let removed_proxy = advanced_obj.remove(OVERRIDE_PROXY_URL_KEY).is_some();
    let removed_capi = advanced_obj.remove(OVERRIDE_CAPI_URL_KEY).is_some();
    let removed_chat = advanced_obj.remove(CHAT_OVERRIDE_PROXY_URL_KEY).is_some();
    let removed = removed_proxy || removed_capi || removed_chat;

    if advanced_obj.is_empty() {
        settings_obj.remove(ADVANCED_SETTINGS_KEY);
    }

    Ok(removed)
}

fn backup_path_for(settings_path: &Path) -> PathBuf {
    settings_path.with_extension("json.isartor-backup")
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
    fn detects_supported_vscode_settings_paths() {
        let home = Path::new("/tmp/home");
        let xdg = Path::new("/tmp/xdg");
        let appdata = Path::new("/tmp/appdata");

        assert_eq!(
            detect_vscode_settings_path_for("macos", Some(home), None, None).unwrap(),
            home.join("Library/Application Support/Code/User/settings.json")
        );
        assert_eq!(
            detect_vscode_settings_path_for("linux", Some(home), None, None).unwrap(),
            home.join(".config/Code/User/settings.json")
        );
        assert_eq!(
            detect_vscode_settings_path_for("linux", Some(home), Some(xdg), None).unwrap(),
            xdg.join("Code/User/settings.json")
        );
        assert_eq!(
            detect_vscode_settings_path_for("windows", Some(home), None, Some(appdata)).unwrap(),
            appdata.join("Code/User/settings.json")
        );
    }

    #[test]
    fn reads_jsonc_settings_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{
                // keep this user setting
                "editor.fontSize": 14,
                "github.copilot.advanced": {
                    "debug.overrideProxyUrl": "http://localhost:8080",
                },
            }"#,
        )
        .unwrap();

        let settings = read_settings_json(&path).unwrap();
        assert_eq!(settings["editor.fontSize"], 14);
        assert_eq!(
            settings[ADVANCED_SETTINGS_KEY][OVERRIDE_PROXY_URL_KEY],
            "http://localhost:8080"
        );
    }

    #[test]
    fn upsert_preserves_existing_settings() {
        let mut settings = serde_json::json!({
            "editor.fontSize": 14,
            ADVANCED_SETTINGS_KEY: {
                "debug.otherFlag": true
            }
        });

        upsert_copilot_vscode_settings(&mut settings, "http://localhost:18080").unwrap();

        assert_eq!(settings["editor.fontSize"], 14);
        assert_eq!(
            settings[ADVANCED_SETTINGS_KEY]["debug.otherFlag"],
            Value::Bool(true)
        );
        assert_eq!(
            settings[ADVANCED_SETTINGS_KEY][OVERRIDE_PROXY_URL_KEY],
            "http://localhost:18080"
        );
        assert_eq!(
            settings[ADVANCED_SETTINGS_KEY][OVERRIDE_CAPI_URL_KEY],
            "http://localhost:18080/v1"
        );
        assert_eq!(
            settings[ADVANCED_SETTINGS_KEY][CHAT_OVERRIDE_PROXY_URL_KEY],
            "http://localhost:18080/v1/chat/completions"
        );
    }

    #[test]
    fn remove_preserves_non_isartor_settings() {
        let mut settings = serde_json::json!({
            "editor.fontSize": 14,
            ADVANCED_SETTINGS_KEY: {
                "debug.otherFlag": true,
                OVERRIDE_PROXY_URL_KEY: "http://localhost:8080",
                OVERRIDE_CAPI_URL_KEY: "http://localhost:8080/v1",
                CHAT_OVERRIDE_PROXY_URL_KEY: "http://localhost:8080/v1/chat/completions"
            }
        });

        let removed = remove_isartor_overrides(&mut settings).unwrap();

        assert!(removed);
        assert_eq!(settings["editor.fontSize"], 14);
        assert_eq!(
            settings[ADVANCED_SETTINGS_KEY]["debug.otherFlag"],
            Value::Bool(true)
        );
        assert!(settings[ADVANCED_SETTINGS_KEY][OVERRIDE_PROXY_URL_KEY].is_null());
        assert!(settings[ADVANCED_SETTINGS_KEY][OVERRIDE_CAPI_URL_KEY].is_null());
        assert!(settings[ADVANCED_SETTINGS_KEY][CHAT_OVERRIDE_PROXY_URL_KEY].is_null());
    }

    #[test]
    fn backup_and_restore_round_trip() {
        let dir = tempdir().unwrap();
        let original = dir.path().join("settings.json");
        let backup = dir.path().join("settings.json.isartor-backup");
        let mut changes = Vec::new();

        std::fs::write(&original, "{\"editor.fontSize\":14}").unwrap();
        backup_file(&original, &backup, false, &mut changes).unwrap();
        std::fs::write(&original, "{\"editor.fontSize\":16}").unwrap();

        restore_file(&original, &backup, false, &mut changes, "restore").unwrap();

        assert_eq!(
            std::fs::read_to_string(&original).unwrap(),
            "{\"editor.fontSize\":14}"
        );
        assert!(!backup.exists());
    }
}
