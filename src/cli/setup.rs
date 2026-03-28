use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde_json::json;
use toml_edit::DocumentMut;

use crate::cli::connect::status::StatusArgs;
use crate::cli::connect::{
    self, BaseClientArgs, ConnectArgs, ConnectClient, claude::ClaudeArgs, codex::CodexArgs,
    copilot::CopilotArgs, cursor::CursorArgs, gemini::GeminiArgs, generic::GenericArgs,
    openclaw::OpenclawArgs,
};
use crate::cli::set_key::{apply_provider_config, default_model};
use crate::config::{
    AppConfig, DEFAULT_OPENAI_CHAT_COMPLETIONS_URL, InferenceEngineMode, LlmProvider,
    default_chat_completions_url,
};
use crate::first_run::write_config_scaffold;

const CONFIG_PATH: &str = "isartor.toml";

#[derive(Parser, Debug, Clone)]
pub struct SetupArgs {
    /// Isartor gateway URL used for connector verification.
    #[arg(long, default_value = connect::DEFAULT_GATEWAY_URL)]
    pub gateway_url: String,

    /// Print the final config and connector payloads without writing files.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    /// Show generated connector configs while applying them.
    #[arg(long, default_value_t = false)]
    pub show_config: bool,
}

#[derive(Debug, Clone)]
struct SetupPlan {
    provider: LlmProvider,
    provider_api_key: String,
    model: String,
    provider_endpoint: Option<String>,
    azure_deployment_id: Option<String>,
    azure_api_version: Option<String>,
    gateway_api_key: Option<String>,
    l2_mode: L2Mode,
    connectors: Vec<ConnectorChoice>,
}

#[derive(Debug, Clone)]
enum L2Mode {
    Disabled,
    Embedded,
    Sidecar {
        sidecar_url: String,
        model_name: String,
    },
}

#[derive(Debug, Clone)]
enum ConnectorChoice {
    Claude,
    Openclaw,
    Copilot,
    Cursor,
    Codex,
    Gemini,
    Generic {
        tool_name: String,
        base_url_var: String,
        api_key_var: String,
        append_v1: bool,
    },
}

pub async fn handle_setup(args: SetupArgs) -> Result<()> {
    ensure_interactive()?;

    println!();
    println!("Isartor setup wizard");
    println!("This guided flow configures provider credentials, optional L2 settings,");
    println!("tool connectors, and a final verification pass.");
    println!();

    let current = AppConfig::load_with_validation(false)
        .context("failed to load current Isartor config for setup defaults")?;
    let plan = collect_setup_plan(&current)?;

    let rendered = write_setup_config(&plan, &args)?;
    if args.show_config || args.dry_run {
        println!("--- {} ---", CONFIG_PATH);
        println!("{rendered}");
    }

    let gateway_api_key = plan.gateway_api_key.clone().filter(|key| !key.is_empty());
    for connector in &plan.connectors {
        connect::handle_connect(build_connect_args(
            connector,
            &args,
            gateway_api_key.clone(),
        ))
        .await?;
    }

    print_setup_summary(&plan, &args);

    if !args.dry_run {
        let config = AppConfig::load_with_validation(false)
            .context("failed to reload Isartor config after setup")?;
        println!();
        println!("Provider ping:");
        println!("  {}", provider_ping_summary(&config).await);
        println!();
        println!("Gateway + connector status:");
        connect::status::handle_status(StatusArgs {
            gateway_url: args.gateway_url.clone(),
            gateway_api_key,
            proxy_recent_limit: 5,
        })
        .await;
    } else {
        println!();
        println!("Dry run complete. No files were written.");
    }

    Ok(())
}

fn ensure_interactive() -> Result<()> {
    if std::env::var_os("CI").is_some() {
        bail!(
            "`isartor setup` is interactive. Run it in a terminal, or unset CI if you intend to answer prompts manually."
        );
    }

    Ok(())
}

fn collect_setup_plan(current: &AppConfig) -> Result<SetupPlan> {
    let provider = prompt_provider(current)?;
    let model_default =
        if provider == current.llm_provider && !current.external_llm_model.is_empty() {
            current.external_llm_model.clone()
        } else {
            default_model(&provider).to_string()
        };
    let model = prompt_text("L3 model", Some(&model_default))?;
    let provider_api_key = prompt_provider_api_key(current, &provider)?;
    let (provider_endpoint, azure_deployment_id, azure_api_version) =
        prompt_provider_details(current, &provider)?;
    let gateway_api_key = prompt_gateway_api_key(current)?;
    let l2_mode = prompt_l2_mode(current)?;
    let connectors = prompt_connectors()?;

    Ok(SetupPlan {
        provider,
        provider_api_key,
        model,
        provider_endpoint,
        azure_deployment_id,
        azure_api_version,
        gateway_api_key,
        l2_mode,
        connectors,
    })
}

fn prompt_provider(current: &AppConfig) -> Result<LlmProvider> {
    let options = provider_options();
    let default_index = options
        .iter()
        .position(|(provider, _)| *provider == current.llm_provider)
        .unwrap_or(0);

    let choice = prompt_select(
        "Choose your Layer 3 provider",
        &options
            .iter()
            .map(|(provider, label)| format!("{label} ({})", provider.as_str()))
            .collect::<Vec<_>>(),
        default_index,
    )?;

    Ok(options[choice].0.clone())
}

fn prompt_provider_api_key(current: &AppConfig, provider: &LlmProvider) -> Result<String> {
    if *provider == LlmProvider::Ollama {
        return Ok(String::new());
    }

    let keep_current =
        *provider == current.llm_provider && !current.external_llm_api_key.trim().is_empty();
    loop {
        eprint!(
            "Provider API key{}: ",
            if keep_current {
                " (leave blank to keep current)"
            } else {
                ""
            }
        );
        io::stderr().flush()?;
        let key = rpassword::read_password().context("Failed to read API key from stdin")?;
        let trimmed = key.trim().to_string();

        if trimmed.is_empty() && keep_current {
            return Ok(current.external_llm_api_key.clone());
        }
        if trimmed.is_empty() {
            eprintln!("API key is required for {}.", provider.as_str());
            continue;
        }
        return Ok(trimmed);
    }
}

fn prompt_gateway_api_key(current: &AppConfig) -> Result<Option<String>> {
    eprint!("Gateway API key (optional; blank keeps current, '-' clears): ");
    io::stderr().flush()?;
    let key = rpassword::read_password().context("Failed to read gateway API key from stdin")?;
    let trimmed = key.trim();

    if trimmed.is_empty() {
        return Ok(Some(current.gateway_api_key.clone()));
    }
    if trimmed == "-" {
        return Ok(Some(String::new()));
    }

    Ok(Some(trimmed.to_string()))
}

fn prompt_provider_details(
    current: &AppConfig,
    provider: &LlmProvider,
) -> Result<(Option<String>, Option<String>, Option<String>)> {
    match provider {
        LlmProvider::Azure => {
            let endpoint_default = if current.llm_provider == LlmProvider::Azure
                && !current.external_llm_url.trim().is_empty()
            {
                current.external_llm_url.as_str()
            } else {
                "https://<resource>.openai.azure.com"
            };
            let deployment_default = if current.llm_provider == LlmProvider::Azure
                && !current.azure_deployment_id.trim().is_empty()
            {
                current.azure_deployment_id.as_str()
            } else {
                "gpt-4o-mini"
            };
            let api_version_default = if !current.azure_api_version.trim().is_empty() {
                current.azure_api_version.as_str()
            } else {
                "2024-08-01-preview"
            };

            Ok((
                Some(prompt_text("Azure endpoint", Some(endpoint_default))?),
                Some(prompt_text(
                    "Azure deployment ID",
                    Some(deployment_default),
                )?),
                Some(prompt_text("Azure API version", Some(api_version_default))?),
            ))
        }
        LlmProvider::Ollama => {
            let default_url = if current.llm_provider == LlmProvider::Ollama
                && !current.external_llm_url.trim().is_empty()
            {
                current.external_llm_url.as_str()
            } else {
                "http://localhost:11434"
            };
            Ok((
                Some(prompt_text("Ollama base URL", Some(default_url))?),
                None,
                None,
            ))
        }
        _ => Ok((None, None, None)),
    }
}

fn prompt_l2_mode(current: &AppConfig) -> Result<L2Mode> {
    let current_mode = if !current.enable_slm_router {
        0
    } else if current.inference_engine == InferenceEngineMode::Embedded {
        1
    } else {
        2
    };
    let choice = prompt_select(
        "Configure Layer 2",
        &[
            "Disabled (recommended if you only want L1 cache + L3 provider)".to_string(),
            "Embedded inference".to_string(),
            "Sidecar inference".to_string(),
        ],
        current_mode,
    )?;

    Ok(match choice {
        0 => L2Mode::Disabled,
        1 => L2Mode::Embedded,
        2 => {
            let sidecar_default = current.layer2.sidecar_url.clone();
            let model_default = current.layer2.model_name.clone();
            L2Mode::Sidecar {
                sidecar_url: prompt_text("Layer 2 sidecar URL", Some(&sidecar_default))?,
                model_name: prompt_text("Layer 2 model name", Some(&model_default))?,
            }
        }
        _ => unreachable!(),
    })
}

fn prompt_connectors() -> Result<Vec<ConnectorChoice>> {
    let options = [
        "Claude Code",
        "OpenClaw",
        "GitHub Copilot CLI",
        "Cursor",
        "OpenAI Codex CLI",
        "Gemini CLI",
        "Generic OpenAI-compatible tool",
    ];
    let selections = prompt_multi_select(
        "Select connectors to configure now (comma-separated, 0 to skip)",
        &options.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
    )?;

    let mut connectors = Vec::new();
    for selection in selections {
        connectors.push(match selection {
            0 => ConnectorChoice::Claude,
            1 => ConnectorChoice::Openclaw,
            2 => ConnectorChoice::Copilot,
            3 => ConnectorChoice::Cursor,
            4 => ConnectorChoice::Codex,
            5 => ConnectorChoice::Gemini,
            6 => ConnectorChoice::Generic {
                tool_name: prompt_text("Generic tool name", Some("My Tool"))?,
                base_url_var: prompt_text("Base URL env var", Some("OPENAI_BASE_URL"))?,
                api_key_var: prompt_text(
                    "API key env var (blank if not needed)",
                    Some("OPENAI_API_KEY"),
                )?,
                append_v1: prompt_yes_no("Append /v1 to the gateway URL?", true)?,
            },
            _ => unreachable!(),
        });
    }

    Ok(connectors)
}

fn write_setup_config(plan: &SetupPlan, args: &SetupArgs) -> Result<String> {
    let config_path = Path::new(CONFIG_PATH);
    if !config_path.exists() && !args.dry_run {
        let _ = write_config_scaffold()?;
    }

    let existing = if config_path.exists() {
        std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else {
        String::new()
    };
    let mut doc = existing
        .parse::<DocumentMut>()
        .with_context(|| format!("Failed to parse {}", config_path.display()))?;

    apply_setup_config(&mut doc, plan);

    let rendered = doc.to_string();
    if !args.dry_run {
        std::fs::write(config_path, &rendered)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
    }

    Ok(rendered)
}

fn apply_setup_config(doc: &mut DocumentMut, plan: &SetupPlan) {
    apply_provider_config(doc, &plan.provider, &plan.model, &plan.provider_api_key);

    if let Some(gateway_api_key) = &plan.gateway_api_key {
        doc["gateway_api_key"] = toml_edit::value(gateway_api_key.as_str());
    }

    match &plan.l2_mode {
        L2Mode::Disabled => {
            doc["enable_slm_router"] = toml_edit::value(false);
            doc["inference_engine"] = toml_edit::value("sidecar");
        }
        L2Mode::Embedded => {
            doc["enable_slm_router"] = toml_edit::value(true);
            doc["inference_engine"] = toml_edit::value("embedded");
        }
        L2Mode::Sidecar {
            sidecar_url,
            model_name,
        } => {
            doc["enable_slm_router"] = toml_edit::value(true);
            doc["inference_engine"] = toml_edit::value("sidecar");
            doc["layer2"]["sidecar_url"] = toml_edit::value(sidecar_url.as_str());
            doc["layer2"]["model_name"] = toml_edit::value(model_name.as_str());
        }
    }

    match &plan.provider {
        LlmProvider::Azure => {
            if let Some(endpoint) = &plan.provider_endpoint {
                doc["external_llm_url"] = toml_edit::value(endpoint.as_str());
            }
            if let Some(deployment) = &plan.azure_deployment_id {
                doc["azure_deployment_id"] = toml_edit::value(deployment.as_str());
            }
            if let Some(api_version) = &plan.azure_api_version {
                doc["azure_api_version"] = toml_edit::value(api_version.as_str());
            }
        }
        LlmProvider::Ollama => {
            if let Some(endpoint) = &plan.provider_endpoint {
                doc["external_llm_url"] = toml_edit::value(endpoint.as_str());
            }
        }
        provider => {
            if *provider != LlmProvider::Openai
                && doc["external_llm_url"].as_str() == Some(DEFAULT_OPENAI_CHAT_COMPLETIONS_URL)
                && let Some(url) = default_chat_completions_url(provider)
            {
                doc["external_llm_url"] = toml_edit::value(url);
            }
        }
    }
}

fn build_connect_args(
    connector: &ConnectorChoice,
    args: &SetupArgs,
    gateway_api_key: Option<String>,
) -> ConnectArgs {
    let base = BaseClientArgs {
        gateway_url: args.gateway_url.clone(),
        gateway_api_key,
        disconnect: false,
        dry_run: args.dry_run,
        show_config: args.show_config,
    };

    let client = match connector {
        ConnectorChoice::Claude => ConnectClient::Claude(ClaudeArgs {
            base,
            key: None,
            model: "claude-sonnet-4-6".to_string(),
            fast_model: "claude-haiku-4-5".to_string(),
        }),
        ConnectorChoice::Openclaw => ConnectClient::Openclaw(OpenclawArgs {
            base,
            model: None,
            config_path: None,
        }),
        ConnectorChoice::Copilot => ConnectClient::Copilot(CopilotArgs {
            base,
            github_token: None,
        }),
        ConnectorChoice::Cursor => ConnectClient::Cursor(CursorArgs { base }),
        ConnectorChoice::Codex => ConnectClient::Codex(CodexArgs {
            base,
            model: "o3-mini".to_string(),
        }),
        ConnectorChoice::Gemini => ConnectClient::Gemini(GeminiArgs {
            base,
            model: "gemini-2.0-flash".to_string(),
        }),
        ConnectorChoice::Generic {
            tool_name,
            base_url_var,
            api_key_var,
            append_v1,
        } => ConnectClient::Generic(GenericArgs {
            base,
            tool_name: tool_name.clone(),
            base_url_var: base_url_var.clone(),
            api_key_var: api_key_var.clone(),
            append_v1: *append_v1,
        }),
    };

    ConnectArgs { client }
}

fn print_setup_summary(plan: &SetupPlan, args: &SetupArgs) {
    println!();
    println!("Setup summary");
    println!("  Config file:  {}", CONFIG_PATH);
    println!("  Gateway URL:  {}", args.gateway_url);
    println!("  Provider:     {}", plan.provider.as_str());
    println!("  Model:        {}", plan.model);
    println!(
        "  Gateway auth: {}",
        if plan
            .gateway_api_key
            .as_deref()
            .unwrap_or_default()
            .trim()
            .is_empty()
        {
            "disabled"
        } else {
            "enabled"
        }
    );
    println!(
        "  Layer 2:      {}",
        match &plan.l2_mode {
            L2Mode::Disabled => "disabled".to_string(),
            L2Mode::Embedded => "embedded".to_string(),
            L2Mode::Sidecar {
                sidecar_url,
                model_name,
            } => format!("sidecar ({sidecar_url}, {model_name})"),
        }
    );
    println!(
        "  Connectors:   {}",
        if plan.connectors.is_empty() {
            "none".to_string()
        } else {
            plan.connectors
                .iter()
                .map(connector_label)
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
}

fn connector_label(connector: &ConnectorChoice) -> String {
    match connector {
        ConnectorChoice::Claude => "Claude Code".to_string(),
        ConnectorChoice::Openclaw => "OpenClaw".to_string(),
        ConnectorChoice::Copilot => "GitHub Copilot CLI".to_string(),
        ConnectorChoice::Cursor => "Cursor".to_string(),
        ConnectorChoice::Codex => "OpenAI Codex CLI".to_string(),
        ConnectorChoice::Gemini => "Gemini CLI".to_string(),
        ConnectorChoice::Generic { tool_name, .. } => tool_name.clone(),
    }
}

fn provider_options() -> Vec<(LlmProvider, &'static str)> {
    vec![
        (LlmProvider::Openai, "OpenAI"),
        (LlmProvider::Anthropic, "Anthropic"),
        (LlmProvider::Groq, "Groq"),
        (LlmProvider::Gemini, "Google Gemini"),
        (LlmProvider::Xai, "xAI"),
        (LlmProvider::Mistral, "Mistral"),
        (LlmProvider::Deepseek, "DeepSeek"),
        (LlmProvider::Openrouter, "OpenRouter"),
        (LlmProvider::Together, "Together"),
        (LlmProvider::Perplexity, "Perplexity"),
        (LlmProvider::Copilot, "GitHub Copilot"),
        (LlmProvider::Ollama, "Ollama"),
        (LlmProvider::Azure, "Azure OpenAI"),
    ]
}

fn prompt_text(label: &str, default: Option<&str>) -> Result<String> {
    loop {
        print!("{label}");
        if let Some(default) = default {
            print!(" [{default}]");
        }
        print!(": ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            if let Some(default) = default {
                return Ok(default.to_string());
            }
            eprintln!("{label} is required.");
            continue;
        }

        return Ok(trimmed.to_string());
    }
}

fn prompt_yes_no(label: &str, default: bool) -> Result<bool> {
    let suffix = if default { "[Y/n]" } else { "[y/N]" };
    loop {
        print!("{label} {suffix}: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim().to_ascii_lowercase();

        if trimmed.is_empty() {
            return Ok(default);
        }
        match trimmed.as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => eprintln!("Please answer y or n."),
        }
    }
}

fn prompt_select(label: &str, options: &[String], default_index: usize) -> Result<usize> {
    println!("{label}:");
    for (index, option) in options.iter().enumerate() {
        println!("  {}. {}", index + 1, option);
    }

    loop {
        print!("Choice [{}]: ", default_index + 1);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return Ok(default_index);
        }

        if let Ok(choice) = trimmed.parse::<usize>()
            && (1..=options.len()).contains(&choice)
        {
            return Ok(choice - 1);
        }

        eprintln!("Please enter a number between 1 and {}.", options.len());
    }
}

fn prompt_multi_select(label: &str, options: &[String]) -> Result<Vec<usize>> {
    println!("{label}:");
    println!("  0. Skip connector setup");
    for (index, option) in options.iter().enumerate() {
        println!("  {}. {}", index + 1, option);
    }

    loop {
        print!("Choices [0]: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed == "0" {
            return Ok(Vec::new());
        }

        match parse_multi_select(trimmed, options.len()) {
            Ok(indices) => return Ok(indices),
            Err(err) => eprintln!("{err}"),
        }
    }
}

fn parse_multi_select(input: &str, max: usize) -> Result<Vec<usize>> {
    let mut parsed = BTreeSet::new();
    for token in input.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        let choice: usize = trimmed
            .parse()
            .with_context(|| format!("Invalid connector choice '{trimmed}'"))?;
        if !(1..=max).contains(&choice) {
            bail!("Connector choices must be between 1 and {max}.");
        }
        parsed.insert(choice - 1);
    }

    Ok(parsed.into_iter().collect())
}

struct L3ConnectivityTarget {
    endpoint: String,
    ping_kind: L3PingKind,
    requires_api_key: bool,
}

#[derive(Clone, Copy)]
enum L3PingKind {
    OpenAiModels,
    AzureChatCompletions,
    AnthropicMessages,
    GeminiModelInfo,
    CopilotSessionToken,
    OllamaTags,
    CohereModels,
    HuggingFaceModelInfo,
}

async fn provider_ping_summary(config: &AppConfig) -> String {
    let target = l3_connectivity_target(config);
    if target.requires_api_key && config.external_llm_api_key.trim().is_empty() {
        return "SKIPPED — API key not configured".to_string();
    }

    let timeout_secs = config.l3_timeout_secs.clamp(1, 15);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
    {
        Ok(client) => client,
        Err(err) => return format!("FAILED — could not build HTTP client: {err}"),
    };

    let result = match target.ping_kind {
        L3PingKind::OpenAiModels => client
            .get(&target.endpoint)
            .header(
                AUTHORIZATION,
                format!("Bearer {}", config.external_llm_api_key),
            )
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::AzureChatCompletions => client
            .post(&target.endpoint)
            .header("api-key", &config.external_llm_api_key)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json")
            .json(&json!({
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1
            }))
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::AnthropicMessages => client
            .post(&target.endpoint)
            .header("x-api-key", &config.external_llm_api_key)
            .header("anthropic-version", "2023-06-01")
            .header(CONTENT_TYPE, "application/json")
            .json(&json!({
                "model": config.external_llm_model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}]
            }))
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::GeminiModelInfo => client
            .get(format!(
                "{}?key={}",
                target.endpoint, config.external_llm_api_key
            ))
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::CopilotSessionToken => {
            match crate::providers::copilot::exchange_copilot_session_token(
                &client,
                &config.external_llm_api_key,
            )
            .await
            {
                Ok(_) => Ok("OK — session token exchange succeeded".to_string()),
                Err(err) => Err(err),
            }
        }
        L3PingKind::OllamaTags => client
            .get(&target.endpoint)
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::CohereModels => client
            .get(&target.endpoint)
            .header(
                AUTHORIZATION,
                format!("Bearer {}", config.external_llm_api_key),
            )
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::HuggingFaceModelInfo => client
            .get(&target.endpoint)
            .header(
                AUTHORIZATION,
                format!("Bearer {}", config.external_llm_api_key),
            )
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
    };

    match result {
        Ok(summary) => summary,
        Err(err) => format!("FAILED — {err}"),
    }
}

fn l3_connectivity_target(config: &AppConfig) -> L3ConnectivityTarget {
    match config.llm_provider {
        LlmProvider::Azure => L3ConnectivityTarget {
            endpoint: format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                config.external_llm_url.trim_end_matches('/'),
                config.azure_deployment_id,
                config.azure_api_version
            ),
            ping_kind: L3PingKind::AzureChatCompletions,
            requires_api_key: true,
        },
        LlmProvider::Anthropic => L3ConnectivityTarget {
            endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            ping_kind: L3PingKind::AnthropicMessages,
            requires_api_key: true,
        },
        LlmProvider::Copilot => L3ConnectivityTarget {
            endpoint: crate::providers::copilot::COPILOT_TOKEN_URL.to_string(),
            ping_kind: L3PingKind::CopilotSessionToken,
            requires_api_key: true,
        },
        LlmProvider::Gemini => L3ConnectivityTarget {
            endpoint: format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}",
                config.external_llm_model
            ),
            ping_kind: L3PingKind::GeminiModelInfo,
            requires_api_key: true,
        },
        LlmProvider::Ollama => L3ConnectivityTarget {
            endpoint: format!("{}/api/tags", config.external_llm_url.trim_end_matches('/')),
            ping_kind: L3PingKind::OllamaTags,
            requires_api_key: false,
        },
        LlmProvider::Cohere => L3ConnectivityTarget {
            endpoint: "https://api.cohere.ai/v1/models".to_string(),
            ping_kind: L3PingKind::CohereModels,
            requires_api_key: true,
        },
        LlmProvider::Huggingface => L3ConnectivityTarget {
            endpoint: format!(
                "https://api-inference.huggingface.co/models/{}",
                config.external_llm_model
            ),
            ping_kind: L3PingKind::HuggingFaceModelInfo,
            requires_api_key: true,
        },
        LlmProvider::Openai => openai_models_target("openai", "https://api.openai.com/v1/models"),
        LlmProvider::Xai => openai_models_target("xai", "https://api.x.ai/v1/models"),
        LlmProvider::Mistral => openai_models_target("mistral", "https://api.mistral.ai/v1/models"),
        LlmProvider::Groq => openai_models_target("groq", "https://api.groq.com/openai/v1/models"),
        LlmProvider::Deepseek => {
            openai_models_target("deepseek", "https://api.deepseek.com/models")
        }
        LlmProvider::Galadriel => {
            openai_models_target("galadriel", "https://api.galadriel.com/v1/models")
        }
        LlmProvider::Hyperbolic => {
            openai_models_target("hyperbolic", "https://api.hyperbolic.xyz/v1/models")
        }
        LlmProvider::Mira => openai_models_target("mira", "https://api.mira.network/v1/models"),
        LlmProvider::Moonshot => {
            openai_models_target("moonshot", "https://api.moonshot.cn/v1/models")
        }
        LlmProvider::Openrouter => {
            openai_models_target("openrouter", "https://openrouter.ai/api/v1/models")
        }
        LlmProvider::Perplexity => {
            openai_models_target("perplexity", "https://api.perplexity.ai/models")
        }
        LlmProvider::Together => {
            openai_models_target("together", "https://api.together.xyz/v1/models")
        }
    }
}

fn openai_models_target(_provider: &'static str, endpoint: &str) -> L3ConnectivityTarget {
    L3ConnectivityTarget {
        endpoint: endpoint.to_string(),
        ping_kind: L3PingKind::OpenAiModels,
        requires_api_key: true,
    }
}

fn summarize_ping_response(resp: reqwest::Response) -> Result<String> {
    let status = resp.status();
    if status.is_success() {
        return Ok(format!("OK — HTTP {status}"));
    }
    Err(anyhow::anyhow!("HTTP {status}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_multi_select_deduplicates_choices() {
        let parsed = parse_multi_select("1, 3, 3", 4).unwrap();
        assert_eq!(parsed, vec![0, 2]);
    }

    #[test]
    fn apply_setup_config_embedded_enables_l2() {
        let mut doc = String::new().parse::<DocumentMut>().unwrap();
        let plan = SetupPlan {
            provider: LlmProvider::Groq,
            provider_api_key: "gsk_test".into(),
            model: "llama-3.1-8b-instant".into(),
            provider_endpoint: None,
            azure_deployment_id: None,
            azure_api_version: None,
            gateway_api_key: Some("gateway-secret".into()),
            l2_mode: L2Mode::Embedded,
            connectors: Vec::new(),
        };

        apply_setup_config(&mut doc, &plan);

        assert_eq!(doc["llm_provider"].as_str(), Some("groq"));
        assert_eq!(doc["gateway_api_key"].as_str(), Some("gateway-secret"));
        assert_eq!(doc["enable_slm_router"].as_bool(), Some(true));
        assert_eq!(doc["inference_engine"].as_str(), Some("embedded"));
    }

    #[test]
    fn apply_setup_config_sidecar_sets_layer2_fields() {
        let mut doc = String::new().parse::<DocumentMut>().unwrap();
        let plan = SetupPlan {
            provider: LlmProvider::Openai,
            provider_api_key: "sk-test".into(),
            model: "gpt-4o-mini".into(),
            provider_endpoint: None,
            azure_deployment_id: None,
            azure_api_version: None,
            gateway_api_key: Some(String::new()),
            l2_mode: L2Mode::Sidecar {
                sidecar_url: "http://127.0.0.1:9000".into(),
                model_name: "phi-3-mini-custom".into(),
            },
            connectors: Vec::new(),
        };

        apply_setup_config(&mut doc, &plan);

        assert_eq!(doc["enable_slm_router"].as_bool(), Some(true));
        assert_eq!(doc["inference_engine"].as_str(), Some("sidecar"));
        assert_eq!(
            doc["layer2"]["sidecar_url"].as_str(),
            Some("http://127.0.0.1:9000")
        );
        assert_eq!(
            doc["layer2"]["model_name"].as_str(),
            Some("phi-3-mini-custom")
        );
    }
}
