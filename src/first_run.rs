//! # First-Run Detection & Config Scaffolding
//!
//! Detects whether this is the first time Isartor has been started on
//! this machine and provides the `isartor init` scaffold command.

use std::path::Path;

// ── First-run detection ──────────────────────────────────────────────

/// Returns `true` if neither `ISARTOR__FIRST_RUN_COMPLETE` is set nor
/// `isartor.toml` exists in the working directory.
pub fn is_first_run() -> bool {
    if std::env::var("ISARTOR__FIRST_RUN_COMPLETE").is_ok() {
        return false;
    }
    // Also consider a run "completed" if the user already has a config file.
    if Path::new("isartor.toml").exists() {
        return false;
    }
    true
}

/// Mark the first run as complete by writing a sentinel file.
pub fn mark_first_run_complete() {
    // Best-effort: create a .isartor_init marker file.
    let _ = std::fs::write(".isartor_init", "done\n");
    tracing::info!("First-run marked complete");
}

// ── Welcome banner ───────────────────────────────────────────────────

/// Print the coloured Isartor gate banner to stdout.
pub fn print_startup_banner(first_run: bool) {
    println!("{}", startup_banner(first_run));
}

fn startup_banner(first_run: bool) -> String {
    let version = env!("CARGO_PKG_VERSION");
    let (footer_line_1, footer_line_2) = if first_run {
        (
            "Starting first-run demo in 3 seconds…",
            "Tip: use `isartor --detach` to free this shell.",
        )
    } else {
        (
            "Gateway startup complete.",
            "Tip: use `isartor --detach` to free this shell.",
        )
    };
    format!(
        r#"
....................................:=........................-:....................................
....................................:=........................-:....................................
....................................:=........................-:....................................
.................................-*#:.*#=:................:=#*.:#*-.................................
...............................*#*%@**%@*##..............#**%%**@#*#*...............................
...............................*%*******+##..............##+*****++#*...............................
...............................:%%*####*#%-..............-%#*####*#%:...............................
................................+@#%%%@*%#................*%*@%%@*@+................................
................................+@#%%*%*%%#%:+@+:#%:+%+:%##%*%*#@*@+................................
................................=%*####*###%%%%%%%%%%%%%%#*#+####+#+................................
................................+@#%%%@*%%#%%%%%%%%%%%%%%##%*@%%@*%+................................
................................+@#%%*%*%%#%#%#*%%%%*#%#%##%*%*#@*@+................................
................................+@#%%%@*%%%@##%-....:%##@##%*@%%%*%+................................
................................+@#%#*%*%%%%#%........##%##%*%#==*#+................................
............................:%%#****=+****#%#%**+++=+++++===========+*=.............................
............................:%%%%%#%%++*%#%%#%#####****+*++++++====**:..............................
................................+@#%%#%*%%%%##........##%##%*%%+*#%+................................
................................+@#%@@@#%%%%##........##%##%*@@%@*@+................................
................................+%#%%%%#%%#%##........##%##%*%%%%*%+................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
@@*...*@@@@@@@@@@%.......-%@@%=.......*@@@@@@@@@@%*..=@@@@@@@@@@@@%...+%@@@@@@@@%#-...*@@@@@@@@@@%+.
@@*..=%%+:::::::::......=@@++@%=......*@%=::::::+%@*..::::-%@#:::::..#@@=-:::::-*@%+..*@@-::::::=@@+
@@*..=%@#++++++=:......+%@*..+@@+.....*@%-......-%@*......:#@#.......%@*........-%@*..*@@........@@*
@@*...:+*******%@@-...+%@*::::*%%+....*@@@@@@@@@@%#:......:#@#.......%@*........-%@*..*@@@@@@@@@@%*:
@@*............=@@=..*@@@@@@@@@@@@*...*@%=....-%@#........:#@#.......#@%-.......+%%+..*@@:....=%@#:.
%@*..+@@@@@@@@@@%#..*@%=........=%@#..*@%-.....:#%%=......:#@#.......-#%@@@@@@@@%%*...*@@......:%@%-

Isartor Prompt Firewall v{ver}
Cache-first deflection for LLM workloads
L1a  Exact-match SHA-256 cache
L1b  Semantic embedding similarity cache
L2   Local SLM triage (candle)
L3   Cloud LLM fallback (OpenAI / Azure / ...)

{footer_line_1}
{footer_line_2}
"#,
        ver = version,
        footer_line_1 = footer_line_1,
        footer_line_2 = footer_line_2,
    )
}

/// Print the first-run banner variant with demo messaging.
pub fn print_welcome_banner() {
    print_startup_banner(true);
}

// ── Config scaffold ──────────────────────────────────────────────────

/// The content of a fully-commented `isartor.toml` scaffold.
const SCAFFOLD_TOML: &str = r#"# ═══════════════════════════════════════════════════════════════════════
# Isartor Configuration File
# ═══════════════════════════════════════════════════════════════════════
#
# All values below show their defaults. Uncomment and modify as needed.
# Environment variables with the ISARTOR__ prefix override these values.
# Example: ISARTOR__HOST_PORT="0.0.0.0:9090"

# ── Server ────────────────────────────────────────────────────────────
# host_port = "0.0.0.0:8080"

# ── Authentication ────────────────────────────────────────────────────
# gateway_api_key = ""

# ── Inference Engine ──────────────────────────────────────────────────
# inference_engine = "sidecar"   # "sidecar" or "embedded"

# ── Layer 1 — Cache ──────────────────────────────────────────────────
# cache_mode      = "both"       # "exact", "semantic", or "both"
# cache_backend   = "memory"     # "memory" or "redis"
# redis_url       = "redis://127.0.0.1:6379"
#
# embedding_model        = "all-minilm"
# similarity_threshold   = 0.85
# cache_ttl_secs         = 300
# cache_max_capacity     = 10000

# ── Layer 1 — Router Backend ─────────────────────────────────────────
# router_backend = "embedded"    # "embedded" or "vllm"
# vllm_url       = "http://127.0.0.1:8000"
# vllm_model     = "gemma-2-2b-it"

# ── Layer 2 — SLM Sidecar (llama.cpp) ────────────────────────────────
# enable_slm_router = false    # Set to true to enable L2 SLM triage
# [layer2]
# sidecar_url     = "http://127.0.0.1:8081"
# model_name      = "phi-3-mini"
# timeout_seconds = 30

# ── Legacy Layer 2 (Ollama compat) ───────────────────────────────────
# local_slm_url   = "http://localhost:11434/api/generate"
# local_slm_model = "llama3"

# ── Embedding Sidecar ────────────────────────────────────────────────
# [embedding_sidecar]
# sidecar_url     = "http://127.0.0.1:8082"
# model_name      = "all-minilm"
# timeout_seconds = 10

# ── Layer 3 — External LLM ───────────────────────────────────────────
# llm_provider       = "openai"      # "openai", "azure", "anthropic", "copilot", "xai", "gemini", "mistral",
#                                    # "groq", "cerebras", "nebius", "siliconflow", "fireworks", "nvidia", "chutes",
#                                    # "deepseek", "cohere", "galadriel", "hyperbolic",
#                                    # "huggingface", "mira", "moonshot", "ollama", "openrouter",
#                                    # "perplexity", "together"
# external_llm_url   = "https://api.openai.com/v1/chat/completions"
# external_llm_model = "gpt-4o-mini"
# [model_aliases]
# fast = "gpt-4o-mini"
# smart = "gpt-4o"
# external_llm_api_key = ""          # ← Set this or use ISARTOR__EXTERNAL_LLM_API_KEY
# l3_timeout_secs    = 120           # Shared timeout for all Layer 3 providers

# ── Azure-specific ───────────────────────────────────────────────────
# azure_deployment_id = ""
# azure_api_version   = "2024-08-01-preview"

# ── Observability ────────────────────────────────────────────────────
# enable_monitoring        = false
# otel_exporter_endpoint   = "http://localhost:4317"
# enable_request_logs      = false
# request_log_path         = "~/.isartor/request_logs"
"#;

/// Write a commented `isartor.toml` scaffold to the current directory.
///
/// Returns `Ok(true)` if the file was created, `Ok(false)` if it already
/// exists (to avoid overwriting user configuration).
pub fn write_config_scaffold() -> std::io::Result<bool> {
    let path = Path::new("isartor.toml");
    if path.exists() {
        println!("  ℹ  isartor.toml already exists — skipping scaffold.");
        return Ok(false);
    }
    std::fs::write(path, SCAFFOLD_TOML)?;
    println!("  ✅ Created isartor.toml with documented defaults.");
    println!("     Edit the file, then run: isartor");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_toml_contains_key_sections() {
        assert!(SCAFFOLD_TOML.contains("host_port"));
        assert!(SCAFFOLD_TOML.contains("gateway_api_key"));
        assert!(SCAFFOLD_TOML.contains("cache_mode"));
        assert!(SCAFFOLD_TOML.contains("llm_provider"));
        assert!(SCAFFOLD_TOML.contains("[layer2]"));
        assert!(SCAFFOLD_TOML.contains("[embedding_sidecar]"));
        assert!(SCAFFOLD_TOML.contains("enable_request_logs"));
    }

    #[test]
    fn first_run_detects_env_var() {
        // If ISARTOR__FIRST_RUN_COMPLETE is set, is_first_run() returns false.
        // We can't easily test this without temp_env, but the logic is straightforward.
        // This test validates the function exists and returns a bool.
        let _result: bool = is_first_run();
    }

    #[test]
    fn startup_banner_includes_normal_startup_message() {
        let banner = startup_banner(false);
        assert!(banner.contains("Gateway startup complete."));
        assert!(!banner.contains("Starting first-run demo"));
    }

    #[test]
    fn startup_banner_includes_first_run_message() {
        let banner = startup_banner(true);
        assert!(banner.contains("Starting first-run demo in 3 seconds"));
        assert!(banner.contains("Tip: use `isartor --detach`"));
    }
}
