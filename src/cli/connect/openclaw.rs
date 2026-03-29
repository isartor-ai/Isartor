use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use clap::Parser;
use serde_json::{Map, Value};

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};
use crate::config::{AppConfig, LlmProvider};

const OPENCLAW_PROVIDER_ID: &str = "isartor";
const OPENCLAW_CONFIG_REL_PATH: &str = ".openclaw/openclaw.json";

#[derive(Parser, Debug, Clone)]
pub struct OpenclawArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// Model ID to expose to OpenClaw. Defaults to Isartor's configured upstream model.
    #[arg(long)]
    pub model: Option<String>,

    /// Path to openclaw.json (auto-detected if not specified)
    #[arg(long)]
    pub config_path: Option<String>,
}

pub async fn handle_openclaw_connect(args: OpenclawArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();
    let mut changes = Vec::new();

    let config_path = resolve_openclaw_config_path(args.config_path.as_deref());

    if args.base.disconnect {
        return disconnect_openclaw(&args, &config_path, &mut changes);
    }

    let backup_path = backup_path_for(&config_path);
    if let Err(err) = backup_file(&config_path, &backup_path, args.base.dry_run, &mut changes) {
        return failure_result(format!("Failed to back up OpenClaw config: {err}"), changes);
    }

    let existing = match read_openclaw_config(&config_path) {
        Ok(existing) => existing,
        Err(err) => {
            return failure_result(
                format!(
                    "Failed to read OpenClaw config from {}: {err}",
                    config_path.display()
                ),
                changes,
            );
        }
    };

    let model = args.model.clone().unwrap_or_else(resolve_default_model);
    let provider_ref = format!("{OPENCLAW_PROVIDER_ID}/{model}");
    let base_url = format!("{}/v1", gateway.trim_end_matches('/'));
    let api_key = gateway_key
        .clone()
        .filter(|key| !key.trim().is_empty())
        .unwrap_or_else(|| "isartor-local".to_string());

    let updated = build_openclaw_config(existing, &base_url, &api_key, &model);
    let rendered = match serde_json::to_string_pretty(&updated) {
        Ok(rendered) => format!("{rendered}\n"),
        Err(err) => {
            return failure_result(format!("Failed to render OpenClaw config: {err}"), changes);
        }
    };

    if args.base.show_config || args.base.dry_run {
        println!("--- {} ---\n{rendered}", config_path.display());
    }

    let existed = config_path.exists();
    if let Err(err) = write_file(&config_path, &rendered, args.base.dry_run) {
        return failure_result(
            format!(
                "Failed to write OpenClaw config to {}: {err}",
                config_path.display()
            ),
            changes,
        );
    }

    changes.push(ConfigChange {
        change_type: if existed {
            ConfigChangeType::FileModified
        } else {
            ConfigChangeType::FileCreated
        },
        target: config_path.to_string_lossy().to_string(),
        description: "Configured OpenClaw to use Isartor as an OpenAI-compatible provider"
            .to_string(),
    });

    if let Err(err) =
        remove_stale_agent_model_registries(&updated, &config_path, args.base.dry_run, &mut changes)
    {
        return failure_result(
            format!("Failed to refresh OpenClaw agent model registries: {err}"),
            changes,
        );
    }

    let test =
        test_isartor_connection(&gateway, gateway_key.as_deref(), "Hello from OpenClaw test").await;

    ConnectResult {
        client_name: "OpenClaw".to_string(),
        success: test.response_received || test.layer_resolved == "timeout" || args.base.dry_run,
        message: format!(
            "OpenClaw is configured to route through Isartor.\n\n\
             OpenClaw config:\n  {}\n\
             Backup:\n  {}\n\n\
             Provider:  {OPENCLAW_PROVIDER_ID}\n\
             Model:     {provider_ref}\n\
             Base URL:  {base_url}\n\n\
             Pragmatic note: OpenClaw now sees one managed Isartor model, which mirrors Isartor's current upstream model setting. If you change Isartor's provider, model, or gateway API key later, rerun `isartor connect openclaw` so both `openclaw.json` and the per-agent model registry refresh together.\n\n\
             Recommended next steps:\n\
             1. Start Isartor: `isartor up --detach`\n\
             2. Check OpenClaw's active model/auth: `openclaw models status --agent main --probe`\n\
             3. Smoke test a prompt: `openclaw agent --agent main -m \"hello\"`",
            config_path.display(),
            backup_path.display(),
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect_openclaw(
    args: &OpenclawArgs,
    config_path: &Path,
    changes: &mut Vec<ConfigChange>,
) -> ConnectResult {
    let backup_path = backup_path_for(config_path);
    let outcome = if backup_path.exists() {
        restore_file(
            config_path,
            &backup_path,
            args.base.dry_run,
            changes,
            "Restored OpenClaw config from backup",
        )
        .map(|_| {
            "Restored the original OpenClaw config from backup. Restart or re-open OpenClaw if needed."
                .to_string()
        })
    } else {
        remove_isartor_provider(config_path, args.base.dry_run, changes).map(|removed| {
            if removed {
                "Removed Isartor from OpenClaw's managed provider config.".to_string()
            } else {
                "No managed Isartor OpenClaw config was present. Nothing to disconnect.".to_string()
            }
        })
    };

    match outcome {
        Ok(message) => ConnectResult {
            client_name: "OpenClaw".to_string(),
            success: true,
            message,
            changes_made: changes.clone(),
            test_result: None,
        },
        Err(err) => failure_result(
            format!("Failed to disconnect OpenClaw from Isartor: {err}"),
            changes.clone(),
        ),
    }
}

fn failure_result(message: String, changes: Vec<ConfigChange>) -> ConnectResult {
    ConnectResult {
        client_name: "OpenClaw".to_string(),
        success: false,
        message,
        changes_made: changes,
        test_result: None,
    }
}

fn resolve_openclaw_config_path(cli_override: Option<&str>) -> PathBuf {
    if let Some(path) = cli_override {
        return PathBuf::from(path);
    }

    let home_default = home_path(OPENCLAW_CONFIG_REL_PATH)
        .unwrap_or_else(|_| PathBuf::from(OPENCLAW_CONFIG_REL_PATH));
    if home_default.exists() {
        return home_default;
    }

    let cwd_default = std::env::current_dir()
        .ok()
        .map(|cwd| cwd.join("openclaw.json"));
    if let Some(cwd_default) = cwd_default
        && cwd_default.exists()
    {
        return cwd_default;
    }

    home_default
}

fn resolve_default_model() -> String {
    AppConfig::load_with_validation(false)
        .ok()
        .map(|cfg| match cfg.llm_provider {
            LlmProvider::Azure if !cfg.azure_deployment_id.trim().is_empty() => {
                cfg.azure_deployment_id
            }
            _ if !cfg.external_llm_model.trim().is_empty() => cfg.external_llm_model,
            _ => "gpt-4o-mini".to_string(),
        })
        .unwrap_or_else(|| "gpt-4o-mini".to_string())
}

fn read_openclaw_config(path: &Path) -> anyhow::Result<Value> {
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

fn build_openclaw_config(existing: Value, base_url: &str, api_key: &str, model: &str) -> Value {
    let mut root = as_object(existing);

    let mut models_root = root
        .remove("models")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    let mut providers = models_root
        .remove("providers")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    providers.insert(
        OPENCLAW_PROVIDER_ID.to_string(),
        Value::Object(build_provider_definition(base_url, api_key, model)),
    );
    models_root.insert("providers".to_string(), Value::Object(providers));
    root.insert("models".to_string(), Value::Object(models_root));

    let mut agents_root = root
        .remove("agents")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    let mut defaults = agents_root
        .remove("defaults")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    let existing_model = defaults.remove("model");
    let existing_allowlist = defaults.remove("models");
    defaults.insert(
        "model".to_string(),
        update_default_model(existing_model, model),
    );
    if let Some(updated_allowlist) = update_model_allowlist(existing_allowlist, model) {
        defaults.insert("models".to_string(), updated_allowlist);
    }
    agents_root.insert("defaults".to_string(), Value::Object(defaults));
    if let Some(updated_list) = update_agent_list_models(agents_root.remove("list"), model) {
        agents_root.insert("list".to_string(), updated_list);
    }
    root.insert("agents".to_string(), Value::Object(agents_root));

    Value::Object(root)
}

fn build_provider_definition(base_url: &str, api_key: &str, model: &str) -> Map<String, Value> {
    let mut provider = Map::new();
    provider.insert("baseUrl".to_string(), Value::String(base_url.to_string()));
    provider.insert("apiKey".to_string(), Value::String(api_key.to_string()));
    provider.insert(
        "api".to_string(),
        Value::String("openai-completions".to_string()),
    );
    provider.insert(
        "models".to_string(),
        Value::Array(vec![serde_json::json!({
            "id": model,
            "name": format!("Isartor ({model})"),
        })]),
    );
    provider
}

fn update_default_model(existing: Option<Value>, model: &str) -> Value {
    let provider_ref = format!("{OPENCLAW_PROVIDER_ID}/{model}");
    match existing {
        Some(Value::Object(mut model_obj)) => {
            model_obj.insert("primary".to_string(), Value::String(provider_ref));
            Value::Object(model_obj)
        }
        _ => serde_json::json!({ "primary": provider_ref }),
    }
}

fn update_model_allowlist(existing: Option<Value>, model: &str) -> Option<Value> {
    let existing = existing?;
    let mut allowlist = existing.as_object().cloned().unwrap_or_default();
    allowlist.insert(
        format!("{OPENCLAW_PROVIDER_ID}/{model}"),
        serde_json::json!({
            "alias": format!("Isartor ({model})")
        }),
    );
    Some(Value::Object(allowlist))
}

fn update_agent_list_models(existing: Option<Value>, model: &str) -> Option<Value> {
    match existing {
        Some(Value::Array(mut agents)) => {
            for agent in &mut agents {
                let Some(agent_obj) = agent.as_object_mut() else {
                    continue;
                };

                let is_main = agent_obj
                    .get("id")
                    .and_then(|value| value.as_str())
                    .map(|value| value == "main")
                    .unwrap_or(false);
                let is_default = agent_obj
                    .get("default")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);

                if is_main || is_default {
                    let existing_model = agent_obj.remove("model");
                    agent_obj.insert(
                        "model".to_string(),
                        update_default_model(existing_model, model),
                    );
                }
            }
            Some(Value::Array(agents))
        }
        other => other,
    }
}

fn remove_stale_agent_model_registries(
    config: &Value,
    config_path: &Path,
    dry_run: bool,
    changes: &mut Vec<ConfigChange>,
) -> anyhow::Result<()> {
    for path in agent_model_registry_paths(config, config_path)? {
        if !path.exists() {
            continue;
        }
        remove_file(&path, dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: path.to_string_lossy().to_string(),
            description:
                "Removed stale OpenClaw per-agent model registry so it can regenerate with the refreshed Isartor provider settings"
                    .to_string(),
        });
    }
    Ok(())
}

fn agent_model_registry_paths(config: &Value, config_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let state_dir = resolve_openclaw_state_dir(config_path)?;
    let mut agent_dirs = vec![state_dir.join("agents/main/agent")];

    if let Some(agents) = config
        .get("agents")
        .and_then(|value| value.as_object())
        .and_then(|agents| agents.get("list"))
        .and_then(|value| value.as_array())
    {
        for agent in agents {
            let Some(agent_obj) = agent.as_object() else {
                continue;
            };

            if let Some(agent_dir) = agent_obj.get("agentDir").and_then(|value| value.as_str()) {
                agent_dirs.push(expand_openclaw_path(agent_dir, &state_dir)?);
                continue;
            }

            if let Some(agent_id) = agent_obj.get("id").and_then(|value| value.as_str()) {
                agent_dirs.push(state_dir.join("agents").join(agent_id).join("agent"));
            }
        }
    }

    agent_dirs.sort();
    agent_dirs.dedup();
    Ok(agent_dirs
        .into_iter()
        .map(|agent_dir| agent_dir.join("models.json"))
        .collect())
}

fn resolve_openclaw_state_dir(config_path: &Path) -> anyhow::Result<PathBuf> {
    if let Ok(path) = std::env::var("OPENCLAW_STATE_DIR")
        && !path.trim().is_empty()
    {
        return expand_openclaw_path(&path, &std::env::current_dir()?);
    }

    if let Ok(path) = std::env::var("OPENCLAW_HOME")
        && !path.trim().is_empty()
    {
        let openclaw_home = expand_openclaw_path(&path, &std::env::current_dir()?)?;
        return Ok(openclaw_home.join(".openclaw"));
    }

    Ok(config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(".")))
}

fn expand_openclaw_path(path: &str, fallback_home: &Path) -> anyhow::Result<PathBuf> {
    if path == "~" {
        return resolve_openclaw_home(fallback_home);
    }

    if let Some(stripped) = path.strip_prefix("~/") {
        return Ok(resolve_openclaw_home(fallback_home)?.join(stripped));
    }

    Ok(PathBuf::from(path))
}

fn resolve_openclaw_home(fallback_home: &Path) -> anyhow::Result<PathBuf> {
    if let Ok(path) = std::env::var("OPENCLAW_HOME")
        && !path.trim().is_empty()
    {
        return Ok(PathBuf::from(path));
    }

    if let Some(home) = dirs::home_dir() {
        return Ok(home);
    }

    if !fallback_home.as_os_str().is_empty() {
        return Ok(fallback_home.to_path_buf());
    }

    Err(anyhow!("could not resolve OpenClaw home directory"))
}

fn remove_isartor_provider(
    path: &Path,
    dry_run: bool,
    changes: &mut Vec<ConfigChange>,
) -> anyhow::Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    let mut root = read_openclaw_config(path)?;
    let mut changed = false;

    if let Some(providers) = root
        .get_mut("models")
        .and_then(|value| value.as_object_mut())
        .and_then(|models| models.get_mut("providers"))
        .and_then(|value| value.as_object_mut())
        && providers.remove(OPENCLAW_PROVIDER_ID).is_some()
    {
        changed = true;
    }

    if let Some(defaults) = root
        .get_mut("agents")
        .and_then(|value| value.as_object_mut())
        .and_then(|agents| agents.get_mut("defaults"))
        .and_then(|value| value.as_object_mut())
    {
        changed |= cleanup_default_model(defaults);
        changed |= cleanup_model_allowlist(defaults);
    }
    if let Some(agents_list) = root
        .get_mut("agents")
        .and_then(|value| value.as_object_mut())
        .and_then(|agents| agents.get_mut("list"))
        .and_then(|value| value.as_array_mut())
    {
        for agent in agents_list {
            let Some(agent_obj) = agent.as_object_mut() else {
                continue;
            };
            if should_sync_agent_override(agent_obj) {
                changed |= cleanup_agent_override(agent_obj);
            }
        }
    }

    if !changed {
        let _ = remove_stale_agent_model_registries(&root, path, dry_run, changes);
        return Ok(false);
    }

    let rendered = format!("{}\n", serde_json::to_string_pretty(&root)?);
    write_file(path, &rendered, dry_run)?;
    changes.push(ConfigChange {
        change_type: ConfigChangeType::FileModified,
        target: path.to_string_lossy().to_string(),
        description: "Removed Isartor provider config from OpenClaw".to_string(),
    });
    remove_stale_agent_model_registries(&root, path, dry_run, changes)?;
    Ok(true)
}

fn should_sync_agent_override(agent_obj: &Map<String, Value>) -> bool {
    agent_obj
        .get("id")
        .and_then(|value| value.as_str())
        .map(|value| value == "main")
        .unwrap_or(false)
        || agent_obj
            .get("default")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
}

fn cleanup_agent_override(agent_obj: &mut Map<String, Value>) -> bool {
    cleanup_default_model(agent_obj)
}

fn cleanup_default_model(defaults: &mut Map<String, Value>) -> bool {
    let Some(existing) = defaults.get_mut("model") else {
        return false;
    };

    match existing {
        Value::Object(model_obj) => {
            let mut changed = false;
            let mut fallbacks = model_obj
                .get("fallbacks")
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default();
            fallbacks.retain(|value| {
                !value
                    .as_str()
                    .map(|value| value.starts_with("isartor/"))
                    .unwrap_or(false)
            });

            let primary_is_isartor = model_obj
                .get("primary")
                .and_then(|value| value.as_str())
                .map(|value| value.starts_with("isartor/"))
                .unwrap_or(false);

            if primary_is_isartor {
                changed = true;
                if let Some(next_primary) = fallbacks.first().and_then(|value| value.as_str()) {
                    model_obj.insert(
                        "primary".to_string(),
                        Value::String(next_primary.to_string()),
                    );
                    fallbacks.remove(0);
                } else {
                    model_obj.remove("primary");
                }
            }

            if model_obj.contains_key("fallbacks") || !fallbacks.is_empty() {
                changed = true;
                if fallbacks.is_empty() {
                    model_obj.remove("fallbacks");
                } else {
                    model_obj.insert("fallbacks".to_string(), Value::Array(fallbacks));
                }
            }

            if model_obj.is_empty() {
                defaults.remove("model");
                changed = true;
            }

            changed
        }
        Value::String(primary) if primary.starts_with("isartor/") => {
            defaults.remove("model");
            true
        }
        _ => false,
    }
}

fn cleanup_model_allowlist(defaults: &mut Map<String, Value>) -> bool {
    let Some(Value::Object(models)) = defaults.get_mut("models") else {
        return false;
    };

    let before = models.len();
    models.retain(|key, _| !key.starts_with("isartor/"));
    let changed = before != models.len();
    if models.is_empty() {
        defaults.remove("models");
        return true;
    }
    changed
}

fn backup_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|name| format!("{}.isartor-backup", name.to_string_lossy()))
        .unwrap_or_else(|| "openclaw.json.isartor-backup".to_string());
    path.with_file_name(file_name)
}

fn backup_file(
    original_path: &Path,
    backup_path: &Path,
    dry_run: bool,
    changes: &mut Vec<ConfigChange>,
) -> anyhow::Result<()> {
    if !original_path.exists() || backup_path.exists() {
        return Ok(());
    }

    let content = std::fs::read_to_string(original_path)
        .with_context(|| format!("failed to read {}", original_path.display()))?;
    write_file(backup_path, &content, dry_run)?;
    changes.push(ConfigChange {
        change_type: ConfigChangeType::FileBackedUp,
        target: backup_path.to_string_lossy().to_string(),
        description: "Original OpenClaw config backed up".to_string(),
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

fn as_object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn build_openclaw_config_sets_provider_and_primary_model() {
        let config = build_openclaw_config(
            serde_json::json!({
                "models": {
                    "providers": {
                        "litellm": { "baseUrl": "http://localhost:4000" }
                    }
                },
                "agents": {
                    "defaults": {
                        "model": {
                            "primary": "litellm/gpt-4o",
                            "fallbacks": ["openai/gpt-5.4"]
                        }
                    }
                }
            }),
            "http://localhost:8080/v1",
            "isartor-local",
            "openai/gpt-oss-120b",
        );

        assert_eq!(
            config["models"]["providers"]["isartor"]["baseUrl"],
            "http://localhost:8080/v1"
        );
        assert_eq!(
            config["models"]["providers"]["isartor"]["api"],
            "openai-completions"
        );
        assert_eq!(
            config["agents"]["defaults"]["model"]["primary"],
            "isartor/openai/gpt-oss-120b"
        );
        assert_eq!(
            config["agents"]["defaults"]["model"]["fallbacks"][0],
            "openai/gpt-5.4"
        );
        assert!(config["models"]["providers"]["litellm"].is_object());
    }

    #[test]
    fn build_openclaw_config_augments_allowlist_when_present() {
        let config = build_openclaw_config(
            serde_json::json!({
                "agents": {
                    "defaults": {
                        "models": {
                            "openai/gpt-5.4": { "alias": "GPT" }
                        }
                    }
                }
            }),
            "http://localhost:8080/v1",
            "isartor-local",
            "gpt-4o-mini",
        );

        assert_eq!(
            config["agents"]["defaults"]["models"]["isartor/gpt-4o-mini"]["alias"],
            "Isartor (gpt-4o-mini)"
        );
        assert!(config["agents"]["defaults"]["models"]["openai/gpt-5.4"].is_object());
    }

    #[test]
    fn build_openclaw_config_updates_main_agent_override() {
        let config = build_openclaw_config(
            serde_json::json!({
                "agents": {
                    "list": [
                        {
                            "id": "main",
                            "model": {
                                "primary": "openai/gpt-5.4",
                                "fallbacks": ["anthropic/claude-opus-4-6"]
                            }
                        },
                        {
                            "id": "ops",
                            "model": {
                                "primary": "anthropic/claude-opus-4-6"
                            }
                        }
                    ]
                }
            }),
            "http://localhost:8080/v1",
            "fresh-key",
            "gpt-4o-mini",
        );

        assert_eq!(
            config["agents"]["list"][0]["model"]["primary"],
            "isartor/gpt-4o-mini"
        );
        assert_eq!(
            config["agents"]["list"][0]["model"]["fallbacks"][0],
            "anthropic/claude-opus-4-6"
        );
        assert_eq!(
            config["agents"]["list"][1]["model"]["primary"],
            "anthropic/claude-opus-4-6"
        );
    }

    #[test]
    fn remove_stale_agent_model_registries_clears_main_and_custom_agent_dirs() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("openclaw.json");
        let main_models = dir.path().join("agents/main/agent/models.json");
        let custom_models = dir.path().join("custom-agent/models.json");
        std::fs::create_dir_all(main_models.parent().unwrap()).unwrap();
        std::fs::create_dir_all(custom_models.parent().unwrap()).unwrap();
        std::fs::write(&main_models, "{}\n").unwrap();
        std::fs::write(&custom_models, "{}\n").unwrap();

        let mut changes = Vec::new();
        remove_stale_agent_model_registries(
            &serde_json::json!({
                "agents": {
                    "list": [
                        {
                            "id": "main"
                        },
                        {
                            "id": "work",
                            "agentDir": custom_models.parent().unwrap().to_string_lossy().to_string()
                        }
                    ]
                }
            }),
            &config_path,
            false,
            &mut changes,
        )
        .unwrap();

        assert!(!main_models.exists());
        assert!(!custom_models.exists());
        assert_eq!(changes.len(), 2);
    }

    #[test]
    fn remove_isartor_provider_promotes_first_non_isartor_fallback() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("openclaw.json");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&serde_json::json!({
                "models": {
                    "providers": {
                        "isartor": { "baseUrl": "http://localhost:8080/v1" },
                        "openai": { "baseUrl": "https://api.openai.com/v1" }
                    }
                },
                "agents": {
                    "defaults": {
                        "model": {
                            "primary": "isartor/gpt-4o-mini",
                            "fallbacks": ["openai/gpt-5.4", "isartor/old-model"]
                        },
                        "models": {
                            "isartor/gpt-4o-mini": { "alias": "Isartor" },
                            "openai/gpt-5.4": { "alias": "GPT" }
                        }
                    },
                    "list": [
                        {
                            "id": "main",
                            "model": {
                                "primary": "isartor/gpt-4o-mini"
                            }
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut changes = Vec::new();
        let removed = remove_isartor_provider(&path, false, &mut changes).unwrap();
        let updated: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

        assert!(removed);
        assert!(updated["models"]["providers"].get("isartor").is_none());
        assert_eq!(
            updated["agents"]["defaults"]["model"]["primary"],
            "openai/gpt-5.4"
        );
        assert!(
            updated["agents"]["defaults"]["model"]
                .get("fallbacks")
                .is_none()
        );
        assert!(
            updated["agents"]["defaults"]["models"]
                .get("isartor/gpt-4o-mini")
                .is_none()
        );
        assert!(updated["agents"]["list"][0].get("model").is_none());
    }

    #[test]
    fn backup_and_restore_round_trip() {
        let dir = tempdir().unwrap();
        let original = dir.path().join("openclaw.json");
        let backup = backup_path_for(&original);
        std::fs::write(&original, "{\"models\":{}}\n").unwrap();

        let mut changes = Vec::new();
        backup_file(&original, &backup, false, &mut changes).unwrap();
        std::fs::write(&original, "{\"models\":{\"providers\":{\"isartor\":{}}}}\n").unwrap();
        restore_file(&original, &backup, false, &mut changes, "restore").unwrap();

        assert_eq!(
            std::fs::read_to_string(&original).unwrap(),
            "{\"models\":{}}\n"
        );
        assert!(!backup.exists());
    }
}
