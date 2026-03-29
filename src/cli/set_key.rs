use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use toml_edit::DocumentMut;

use crate::config::{KeyRotationStrategy, LlmProvider, default_chat_completions_url};

/// Set the API key for an LLM provider (writes to isartor.toml or env file).
#[derive(Parser, Debug, Clone)]
pub struct SetKeyArgs {
    /// LLM provider name (e.g. openai, anthropic, groq, ollama).
    #[arg(short, long)]
    pub provider: String,

    /// API key string. If omitted and provider requires one, prompts interactively.
    #[arg(short, long)]
    pub key: Option<String>,

    /// Model name. If omitted, uses the sensible default for the provider.
    #[arg(short, long)]
    pub model: Option<String>,

    /// Path to isartor.toml config file.
    #[arg(long, default_value = "./isartor.toml")]
    pub config_path: PathBuf,

    /// Print what would be written without modifying any files.
    #[arg(long)]
    pub dry_run: bool,

    /// Write shell export statements to ~/.isartor/env instead of isartor.toml.
    #[arg(long)]
    pub env_file: bool,

    /// Append this key to the multi-key pool instead of replacing the legacy single key field.
    #[arg(long, default_value_t = false)]
    pub add: bool,

    /// Optional human-readable label when adding to a key pool.
    #[arg(long)]
    pub label: Option<String>,

    /// Priority used by priority-based key rotation (lower wins).
    #[arg(long, default_value_t = 1)]
    pub priority: u32,

    /// Rotation strategy used when multiple keys are configured.
    #[arg(long)]
    pub strategy: Option<String>,

    /// Cooldown applied after a rate-limit / quota response.
    #[arg(long)]
    pub cooldown_secs: Option<u64>,
}

/// Set or update a user-facing model alias in `isartor.toml`.
#[derive(Parser, Debug, Clone)]
pub struct SetAliasArgs {
    /// Alias name clients can send as the request model (for example: fast).
    #[arg(long)]
    pub alias: String,

    /// Real provider model identifier that the alias resolves to.
    #[arg(short, long)]
    pub model: String,

    /// Path to isartor.toml config file.
    #[arg(long, default_value = "./isartor.toml")]
    pub config_path: PathBuf,

    /// Print what would be written without modifying any files.
    #[arg(long)]
    pub dry_run: bool,
}

/// All known LLM provider names (must match LlmProvider enum variants).
const KNOWN_PROVIDERS: &[&str] = &[
    "openai",
    "azure",
    "anthropic",
    "copilot",
    "xai",
    "gemini",
    "mistral",
    "groq",
    "cerebras",
    "nebius",
    "siliconflow",
    "fireworks",
    "nvidia",
    "chutes",
    "deepseek",
    "cohere",
    "galadriel",
    "hyperbolic",
    "huggingface",
    "mira",
    "moonshot",
    "ollama",
    "openrouter",
    "perplexity",
    "together",
];

/// Return the default model for a given provider.
pub fn default_model(provider: &LlmProvider) -> &'static str {
    match provider {
        LlmProvider::Openai => "gpt-4o-mini",
        LlmProvider::Azure => "gpt-4o-mini",
        LlmProvider::Anthropic => "claude-3-5-sonnet-20241022",
        LlmProvider::Copilot => "gpt-4.1",
        LlmProvider::Xai => "grok-2",
        LlmProvider::Gemini => "gemini-2.0-flash",
        LlmProvider::Mistral => "mistral-small-latest",
        LlmProvider::Groq => "llama-3.1-8b-instant",
        LlmProvider::Cerebras => "llama-3.3-70b",
        LlmProvider::Nebius => "meta-llama/Meta-Llama-3.1-70B-Instruct",
        LlmProvider::Siliconflow => "Qwen/Qwen2.5-72B-Instruct",
        LlmProvider::Fireworks => "accounts/fireworks/models/llama-v3p1-8b-instruct",
        LlmProvider::Nvidia => "meta/llama-3.1-8b-instruct",
        LlmProvider::Chutes => "deepseek-ai/DeepSeek-V3-0324",
        LlmProvider::Deepseek => "deepseek-chat",
        LlmProvider::Cohere => "command-r",
        LlmProvider::Ollama => "llama3.2",
        LlmProvider::Openrouter => "openai/gpt-4o-mini",
        LlmProvider::Perplexity => "sonar",
        LlmProvider::Together => "meta-llama/Meta-Llama-3.1-8B-Instruct",
        _ => "gpt-4o-mini",
    }
}

/// Mask an API key for display: show first 4 + last 4 chars.
fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        return "*".repeat(key.len());
    }
    format!("{}...{}", &key[..4], &key[key.len() - 4..])
}

/// Validate that a provider string matches a known LlmProvider variant.
fn validate_provider(s: &str) -> Result<LlmProvider> {
    let lower = s.to_lowercase();
    if !KNOWN_PROVIDERS.contains(&lower.as_str()) {
        bail!(
            "Unknown provider: '{}'. Supported providers:\n  {}",
            s,
            KNOWN_PROVIDERS.join(", ")
        );
    }
    Ok(LlmProvider::from(lower.as_str()))
}

pub fn apply_provider_config(
    doc: &mut DocumentMut,
    provider: &LlmProvider,
    model: &str,
    api_key: &str,
) {
    doc["llm_provider"] = toml_edit::value(provider.as_str());
    if let Some(url) = default_chat_completions_url(provider) {
        doc["external_llm_url"] = toml_edit::value(url);
    }
    doc["external_llm_model"] = toml_edit::value(model);
    doc["external_llm_api_key"] = toml_edit::value(api_key);
}

fn parse_rotation_strategy(value: &str) -> Result<KeyRotationStrategy> {
    match value.trim().to_lowercase().as_str() {
        "round_robin" | "round-robin" | "roundrobin" => Ok(KeyRotationStrategy::RoundRobin),
        "priority" => Ok(KeyRotationStrategy::Priority),
        other => bail!("Unknown key rotation strategy: '{other}'. Use round_robin or priority."),
    }
}

fn rotation_strategy_value(strategy: &KeyRotationStrategy) -> &'static str {
    match strategy {
        KeyRotationStrategy::RoundRobin => "round_robin",
        KeyRotationStrategy::Priority => "priority",
    }
}

fn append_provider_key(
    doc: &mut DocumentMut,
    api_key: &str,
    label: Option<&str>,
    priority: u32,
    strategy: Option<&KeyRotationStrategy>,
    cooldown_secs: Option<u64>,
) {
    if doc.get("provider_keys").is_none() {
        doc["provider_keys"] = toml_edit::Item::ArrayOfTables(toml_edit::ArrayOfTables::new());
    }

    let mut table = toml_edit::Table::new();
    table["key"] = toml_edit::value(api_key);
    table["priority"] = toml_edit::value(i64::from(priority));
    if let Some(label) = label.filter(|label| !label.trim().is_empty()) {
        table["label"] = toml_edit::value(label.trim());
    }

    doc["provider_keys"]
        .as_array_of_tables_mut()
        .expect("provider_keys must be an array of tables")
        .push(table);

    if let Some(strategy) = strategy {
        doc["key_rotation_strategy"] = toml_edit::value(rotation_strategy_value(strategy));
    }
    if let Some(cooldown_secs) = cooldown_secs {
        doc["key_cooldown_secs"] = toml_edit::value(i64::try_from(cooldown_secs).unwrap_or(60));
    }
}

pub fn write_provider_config(
    config_path: &Path,
    provider: &LlmProvider,
    model: &str,
    api_key: &str,
    dry_run: bool,
) -> Result<String> {
    let existing = if config_path.exists() {
        std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else {
        String::new()
    };

    let mut doc = existing
        .parse::<DocumentMut>()
        .with_context(|| format!("Failed to parse {}", config_path.display()))?;

    apply_provider_config(&mut doc, provider, model, api_key);

    let output = doc.to_string();
    if dry_run {
        return Ok(output);
    }

    std::fs::write(config_path, &output)
        .with_context(|| format!("Failed to write {}", config_path.display()))?;

    Ok(output)
}

pub fn apply_model_alias(doc: &mut DocumentMut, alias: &str, model: &str) {
    if doc.get("model_aliases").is_none() {
        doc["model_aliases"] = toml_edit::table();
    }
    doc["model_aliases"][alias] = toml_edit::value(model);
}

pub fn write_model_alias_config(
    config_path: &Path,
    alias: &str,
    model: &str,
    dry_run: bool,
) -> Result<String> {
    let existing = if config_path.exists() {
        std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else {
        String::new()
    };

    let mut doc = existing
        .parse::<DocumentMut>()
        .with_context(|| format!("Failed to parse {}", config_path.display()))?;

    apply_model_alias(&mut doc, alias, model);

    let output = doc.to_string();
    if dry_run {
        return Ok(output);
    }

    std::fs::write(config_path, &output)
        .with_context(|| format!("Failed to write {}", config_path.display()))?;

    Ok(output)
}

pub async fn handle_set_key(args: SetKeyArgs) -> Result<()> {
    // 1. Validate provider
    let provider = validate_provider(&args.provider)?;
    let provider_str = provider.as_str();

    // 2. Resolve API key
    let api_key = if let Some(k) = args.key {
        k
    } else if provider == LlmProvider::Ollama {
        String::new()
    } else {
        eprint!("Enter API key for {}: ", provider_str);
        std::io::stderr().flush()?;
        rpassword::read_password().context("Failed to read API key from stdin")?
    };

    let api_key = api_key.trim().to_string();

    if api_key.is_empty() && provider != LlmProvider::Ollama {
        bail!("API key is required for provider '{}'", provider_str);
    }
    if args.priority == 0 {
        bail!("--priority must be greater than zero");
    }
    if args.add && args.env_file {
        bail!("--add is only supported for isartor.toml mode");
    }
    let strategy = args
        .strategy
        .as_deref()
        .map(parse_rotation_strategy)
        .transpose()?;
    if let Some(cooldown_secs) = args.cooldown_secs
        && cooldown_secs == 0
    {
        bail!("--cooldown-secs must be greater than zero");
    }

    // 3. Resolve model
    let model = args
        .model
        .unwrap_or_else(|| default_model(&provider).to_string());

    // 4. Handle --env-file mode
    if args.env_file {
        let env_content = format!(
            "export ISARTOR__LLM_PROVIDER=\"{}\"\n{}export ISARTOR__EXTERNAL_LLM_MODEL=\"{}\"\nexport ISARTOR__EXTERNAL_LLM_API_KEY=\"{}\"\n",
            provider_str,
            default_chat_completions_url(&provider)
                .map(|url| format!("export ISARTOR__EXTERNAL_LLM_URL=\"{}\"\n", url))
                .unwrap_or_default(),
            model,
            api_key
        );

        if args.dry_run {
            eprintln!("[dry-run] Would write to ~/.isartor/env:");
            eprintln!("{}", env_content);
            return Ok(());
        }

        let isartor_dir = dirs::home_dir()
            .context("Could not determine home directory")?
            .join(".isartor");
        std::fs::create_dir_all(&isartor_dir).context("Failed to create ~/.isartor directory")?;

        let env_path = isartor_dir.join("env");
        std::fs::write(&env_path, &env_content)
            .with_context(|| format!("Failed to write {}", env_path.display()))?;

        eprintln!();
        eprintln!("  ✓ Provider:  {}", provider_str);
        eprintln!("  ✓ Model:     {}", model);
        if !api_key.is_empty() {
            eprintln!("  ✓ API key:   {}", mask_key(&api_key));
        }
        eprintln!("  ✓ Written:   {}", env_path.display());
        eprintln!();
        eprintln!("  Run: source {}", env_path.display());
        eprintln!();
        return Ok(());
    }

    // 5. Handle isartor.toml mode (default)
    let config_path = &args.config_path;

    let output = if args.add {
        let existing = if config_path.exists() {
            std::fs::read_to_string(config_path)
                .with_context(|| format!("Failed to read {}", config_path.display()))?
        } else {
            String::new()
        };

        let mut doc = existing
            .parse::<DocumentMut>()
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        apply_provider_config(&mut doc, &provider, &model, "");
        append_provider_key(
            &mut doc,
            &api_key,
            args.label.as_deref(),
            args.priority,
            strategy.as_ref(),
            args.cooldown_secs,
        );
        let output = doc.to_string();
        if !args.dry_run {
            std::fs::write(config_path, &output)
                .with_context(|| format!("Failed to write {}", config_path.display()))?;
        }
        output
    } else {
        write_provider_config(config_path, &provider, &model, &api_key, args.dry_run)?
    };

    if args.dry_run {
        eprintln!("[dry-run] Would write to {}:", config_path.display());
        eprintln!("{}", output);
        return Ok(());
    }

    eprintln!();
    eprintln!("  ✓ Provider:  {}", provider_str);
    eprintln!("  ✓ Model:     {}", model);
    if !api_key.is_empty() {
        eprintln!("  ✓ API key:   {}", mask_key(&api_key));
    }
    if args.add {
        eprintln!("  ✓ Added to:  provider_keys");
        eprintln!("  ✓ Priority:  {}", args.priority);
        if let Some(label) = args.label.as_deref() {
            eprintln!("  ✓ Label:     {}", label);
        }
        if let Some(strategy) = strategy {
            eprintln!("  ✓ Strategy:  {}", rotation_strategy_value(&strategy));
        }
        if let Some(cooldown_secs) = args.cooldown_secs {
            eprintln!("  ✓ Cooldown:  {}s", cooldown_secs);
        }
    }
    eprintln!("  ✓ Written:   {}", config_path.display());
    eprintln!();

    Ok(())
}

pub async fn handle_set_alias(args: SetAliasArgs) -> Result<()> {
    let alias = args.alias.trim();
    let model = args.model.trim();

    if alias.is_empty() {
        bail!("Alias name must not be empty");
    }
    if model.is_empty() {
        bail!("Alias target model must not be empty");
    }

    let output = write_model_alias_config(&args.config_path, alias, model, args.dry_run)?;

    if args.dry_run {
        eprintln!("[dry-run] Would write to {}:", args.config_path.display());
        eprintln!("{output}");
        return Ok(());
    }

    eprintln!();
    eprintln!("  ✓ Alias:     {}", alias);
    eprintln!("  ✓ Resolves:  {}", model);
    eprintln!("  ✓ Written:   {}", args.config_path.display());
    eprintln!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_key_long() {
        assert_eq!(mask_key("sk-ant-1234567890abcdef"), "sk-a...cdef");
    }

    #[test]
    fn test_mask_key_short() {
        assert_eq!(mask_key("abc"), "***");
    }

    #[test]
    fn test_mask_key_exactly_8() {
        assert_eq!(mask_key("12345678"), "********");
    }

    #[test]
    fn test_validate_provider_valid() {
        assert!(validate_provider("openai").is_ok());
        assert!(validate_provider("OpenAI").is_ok());
        assert!(validate_provider("ANTHROPIC").is_ok());
        assert!(validate_provider("ollama").is_ok());
        assert!(validate_provider("cerebras").is_ok());
    }

    #[test]
    fn test_validate_provider_invalid() {
        assert!(validate_provider("foobar").is_err());
        assert!(validate_provider("").is_err());
    }

    #[test]
    fn test_default_models() {
        assert_eq!(default_model(&LlmProvider::Openai), "gpt-4o-mini");
        assert_eq!(
            default_model(&LlmProvider::Anthropic),
            "claude-3-5-sonnet-20241022"
        );
        assert_eq!(default_model(&LlmProvider::Ollama), "llama3.2");
        assert_eq!(
            default_model(&LlmProvider::Together),
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        );
        assert_eq!(default_model(&LlmProvider::Cerebras), "llama-3.3-70b");
    }

    #[tokio::test]
    async fn test_set_key_dry_run_toml() {
        let tmp = std::env::temp_dir().join("isartor_test_set_key.toml");
        // Ensure clean state
        let _ = std::fs::remove_file(&tmp);

        let args = SetKeyArgs {
            provider: "openai".to_string(),
            key: Some("sk-test1234567890abcdef".to_string()),
            model: Some("gpt-4o".to_string()),
            config_path: tmp.clone(),
            dry_run: true,
            env_file: false,
            add: false,
            label: None,
            priority: 1,
            strategy: None,
            cooldown_secs: None,
        };

        handle_set_key(args).await.unwrap();

        // dry_run should NOT create the file
        assert!(!tmp.exists());
    }

    #[tokio::test]
    async fn test_set_key_writes_toml() {
        let tmp = std::env::temp_dir().join("isartor_test_set_key_write.toml");
        let _ = std::fs::remove_file(&tmp);

        let args = SetKeyArgs {
            provider: "groq".to_string(),
            key: Some("gsk_testkey12345678".to_string()),
            model: None,
            config_path: tmp.clone(),
            dry_run: false,
            env_file: false,
            add: false,
            label: None,
            priority: 1,
            strategy: None,
            cooldown_secs: None,
        };

        handle_set_key(args).await.unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("llm_provider = \"groq\""));
        assert!(
            content
                .contains("external_llm_url = \"https://api.groq.com/openai/v1/chat/completions\"")
        );
        assert!(content.contains("external_llm_model = \"llama-3.1-8b-instant\""));
        assert!(content.contains("external_llm_api_key = \"gsk_testkey12345678\""));

        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_set_key_preserves_existing_toml() {
        let tmp = std::env::temp_dir().join("isartor_test_set_key_preserve.toml");
        std::fs::write(
            &tmp,
            "host_port = \"0.0.0.0:9090\"\ngateway_api_key = \"mykey\"\n",
        )
        .unwrap();

        let args = SetKeyArgs {
            provider: "anthropic".to_string(),
            key: Some("sk-ant-test".to_string()),
            model: None,
            config_path: tmp.clone(),
            dry_run: false,
            env_file: false,
            add: false,
            label: None,
            priority: 1,
            strategy: None,
            cooldown_secs: None,
        };

        handle_set_key(args).await.unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        // Existing fields preserved
        assert!(content.contains("host_port = \"0.0.0.0:9090\""));
        assert!(content.contains("gateway_api_key = \"mykey\""));
        // New fields added
        assert!(content.contains("llm_provider = \"anthropic\""));
        assert!(content.contains("external_llm_model = \"claude-3-5-sonnet-20241022\""));

        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_set_key_ollama_no_key() {
        let tmp = std::env::temp_dir().join("isartor_test_ollama.toml");
        let _ = std::fs::remove_file(&tmp);

        let args = SetKeyArgs {
            provider: "ollama".to_string(),
            key: None,
            model: None,
            config_path: tmp.clone(),
            dry_run: false,
            env_file: false,
            add: false,
            label: None,
            priority: 1,
            strategy: None,
            cooldown_secs: None,
        };

        handle_set_key(args).await.unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("llm_provider = \"ollama\""));
        assert!(content.contains("external_llm_model = \"llama3.2\""));
        assert!(content.contains("external_llm_api_key = \"\""));

        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_set_key_add_appends_provider_key_pool() {
        let tmp = std::env::temp_dir().join("isartor_test_set_key_add.toml");
        std::fs::write(
            &tmp,
            "llm_provider = \"openai\"\nexternal_llm_model = \"gpt-4o-mini\"\nexternal_llm_api_key = \"sk-legacy\"\n",
        )
        .unwrap();

        let args = SetKeyArgs {
            provider: "openai".to_string(),
            key: Some("sk-shared".to_string()),
            model: Some("gpt-4o-mini".to_string()),
            config_path: tmp.clone(),
            dry_run: false,
            env_file: false,
            add: true,
            label: Some("team".to_string()),
            priority: 2,
            strategy: Some("priority".to_string()),
            cooldown_secs: Some(120),
        };

        handle_set_key(args).await.unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("[[provider_keys]]"));
        assert!(content.contains("key = \"sk-shared\""));
        assert!(content.contains("label = \"team\""));
        assert!(content.contains("priority = 2"));
        assert!(content.contains("key_rotation_strategy = \"priority\""));
        assert!(content.contains("key_cooldown_secs = 120"));

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_write_model_alias_config() {
        let tmp = std::env::temp_dir().join("isartor_test_set_alias.toml");
        let _ = std::fs::remove_file(&tmp);

        let output = write_model_alias_config(&tmp, "fast", "gpt-4o-mini", false).unwrap();
        assert!(output.contains("[model_aliases]"));
        assert!(output.contains("fast = \"gpt-4o-mini\""));

        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("fast = \"gpt-4o-mini\""));

        let _ = std::fs::remove_file(&tmp);
    }
}
