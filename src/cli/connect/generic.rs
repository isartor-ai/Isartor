use clap::Parser;

use super::{
    BaseClientArgs, ConfigChange, ConfigChangeType, ConnectResult, home_path, remove_file,
    test_isartor_connection, write_file,
};

#[derive(Parser, Debug, Clone)]
pub struct GenericArgs {
    #[command(flatten)]
    pub base: BaseClientArgs,

    /// Name of the tool (used for the env script filename and display)
    #[arg(long)]
    pub tool_name: String,

    /// Environment variable the tool reads for its base URL
    /// (e.g. OPENAI_BASE_URL, ANTHROPIC_BASE_URL)
    #[arg(long)]
    pub base_url_var: String,

    /// Environment variable the tool reads for its API key (optional)
    #[arg(long, default_value = "")]
    pub api_key_var: String,

    /// Whether to append /v1 to the gateway URL
    #[arg(long, default_value_t = true)]
    pub append_v1: bool,
}

pub async fn handle_generic_connect(args: GenericArgs) -> ConnectResult {
    let gateway = args.base.effective_gateway_url();
    let gateway_key = args.base.effective_gateway_api_key();

    let mut changes = Vec::new();

    if args.base.disconnect {
        return disconnect_generic(&args, &mut changes);
    }

    let base_url = if args.append_v1 {
        format!("{}/v1", gateway.trim_end_matches('/'))
    } else {
        gateway.trim_end_matches('/').to_string()
    };
    let api_key = gateway_key
        .clone()
        .unwrap_or_else(|| "isartor-local".to_string());

    let slug = sanitize_tool_name(&args.tool_name);
    let env_path = home_path(&format!(".isartor/env/{slug}.sh"))
        .unwrap_or_else(|_| std::path::PathBuf::from(format!(".isartor/env/{slug}.sh")));

    let mut lines = vec![
        format!(
            "# Isartor — {} integration (base URL override)",
            args.tool_name
        ),
        format!("# Source this file: source ~/.isartor/env/{slug}.sh"),
        format!("export {}=\"{}\"", args.base_url_var, base_url),
    ];
    if !args.api_key_var.is_empty() {
        lines.push(format!("export {}=\"{}\"", args.api_key_var, api_key));
    }
    lines.push(format!(
        "export ISARTOR_{}_ENABLED=true",
        slug.to_uppercase()
    ));
    lines.push(String::new()); // trailing newline

    let content = lines.join("\n");

    if args.base.show_config || args.base.dry_run {
        println!("{}", content);
    }

    if write_file(&env_path, &content, args.base.dry_run).is_ok() {
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileCreated,
            target: env_path.to_string_lossy().to_string(),
            description: format!(
                "Shell env file with base URL override for {}",
                args.tool_name
            ),
        });
    }

    let test = test_isartor_connection(
        &gateway,
        gateway_key.as_deref(),
        &format!("Hello from {} test", args.tool_name),
    )
    .await;

    ConnectResult {
        client_name: args.tool_name.clone(),
        success: test.response_received || args.base.dry_run,
        message: format!(
            "1. Start Isartor:  isartor up\n\
             2. Activate env:   source {}\n\
             3. Start {}\n\
             \n\
             Method: base URL override ({})\n\
             Base URL: {base_url}",
            env_path.display(),
            args.tool_name,
            args.base_url_var,
        ),
        changes_made: changes,
        test_result: Some(test),
    }
}

fn disconnect_generic(args: &GenericArgs, changes: &mut Vec<ConfigChange>) -> ConnectResult {
    let slug = sanitize_tool_name(&args.tool_name);
    let env_path = home_path(&format!(".isartor/env/{slug}.sh"))
        .unwrap_or_else(|_| std::path::PathBuf::from(format!(".isartor/env/{slug}.sh")));
    if env_path.exists() {
        remove_file(&env_path, args.base.dry_run);
        changes.push(ConfigChange {
            change_type: ConfigChangeType::FileModified,
            target: env_path.to_string_lossy().to_string(),
            description: "Removed".to_string(),
        });
    }

    ConnectResult {
        client_name: args.tool_name.clone(),
        success: true,
        message: format!(
            "{} disconnected. Restart your shell to unset variables.",
            args.tool_name
        ),
        changes_made: changes.clone(),
        test_result: None,
    }
}

/// Produce a filesystem-safe slug from a tool name.
fn sanitize_tool_name(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_handles_spaces_and_special_chars() {
        assert_eq!(sanitize_tool_name("Windsurf IDE"), "windsurf_ide");
        assert_eq!(sanitize_tool_name("my-tool"), "my-tool");
        assert_eq!(sanitize_tool_name("Roo/Code"), "roo_code");
    }
}
