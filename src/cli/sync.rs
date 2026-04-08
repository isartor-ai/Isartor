use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, bail};
use clap::{Args, Subcommand};

use crate::sync::{
    PutSyncRequest, SyncProfile, backup_path, build_sync_blob, decrypt_blob, default_config_path,
    default_server_data_dir, default_sync_server, derive_user_hash, detect_pull_conflict,
    detect_push_conflict, encrypt_blob, ensure_config_exists, extract_syncable_config,
    fetch_remote_record, file_modified_unix_secs, load_profile, merge_syncable_config,
    now_unix_secs, push_remote_record, run_sync_server, save_backup, save_profile,
};

#[derive(Debug, Args)]
pub struct SyncArgs {
    #[command(subcommand)]
    pub command: SyncCommand,
}

#[derive(Debug, Subcommand)]
pub enum SyncCommand {
    /// Initialize an opt-in local sync profile.
    Init(SyncInitArgs),
    /// Encrypt and upload the current syncable config.
    Push(SyncTransferArgs),
    /// Download, decrypt, and merge synced config into the local file.
    Pull(SyncTransferArgs),
    /// Show the local sync profile.
    Status(SyncStatusArgs),
    /// Run the self-hostable zero-knowledge sync server.
    Serve(SyncServeArgs),
}

#[derive(Debug, Clone, Args)]
pub struct SyncInitArgs {
    /// Server base URL used for PUT/GET sync operations.
    #[arg(long, default_value = default_sync_server())]
    pub server: String,

    /// Path to the local isartor.toml file.
    #[arg(long, default_value = default_config_path())]
    pub config_path: PathBuf,

    /// User-chosen sync identifier (never transmitted directly; only SHA-256 hash is used).
    #[arg(long)]
    pub identifier: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct SyncTransferArgs {
    /// Override the sync server URL saved in the local profile.
    #[arg(long)]
    pub server: Option<String>,

    /// Path to the local isartor.toml file.
    #[arg(long)]
    pub config_path: Option<PathBuf>,

    /// Proceed even if sync conflict detection reports divergence.
    #[arg(long, default_value_t = false)]
    pub force: bool,
}

#[derive(Debug, Clone, Args)]
pub struct SyncStatusArgs {
    /// Override the displayed server URL with a specific value.
    #[arg(long)]
    pub server: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct SyncServeArgs {
    /// Listen address for the self-hosted sync server.
    #[arg(long, default_value = "0.0.0.0:8787")]
    pub listen: String,

    /// Directory where encrypted sync blobs are stored on disk.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
}

pub async fn handle_sync(args: SyncArgs) -> anyhow::Result<()> {
    match args.command {
        SyncCommand::Init(args) => handle_sync_init(args),
        SyncCommand::Push(args) => handle_sync_push(args).await,
        SyncCommand::Pull(args) => handle_sync_pull(args).await,
        SyncCommand::Status(args) => handle_sync_status(args),
        SyncCommand::Serve(args) => handle_sync_serve(args).await,
    }
}

fn handle_sync_init(args: SyncInitArgs) -> anyhow::Result<()> {
    let identifier = match args.identifier {
        Some(identifier) => identifier,
        None => prompt_line("Choose a sync identifier (not your email): ")?,
    };
    if identifier.trim().is_empty() {
        bail!("sync identifier cannot be empty");
    }

    let passphrase = prompt_passphrase("Choose a sync passphrase: ")?;
    let confirm = prompt_passphrase("Confirm the sync passphrase: ")?;
    if passphrase != confirm {
        bail!("passphrases do not match");
    }
    if passphrase.trim().is_empty() {
        bail!("sync passphrase cannot be empty");
    }

    let profile = SyncProfile {
        server: args.server,
        user_hash: derive_user_hash(&identifier),
        salt_hex: crate::sync::generate_salt_hex(),
        config_path: args.config_path.display().to_string(),
        last_remote_updated_at: None,
        last_local_sync_at: None,
    };
    save_profile(&profile)?;

    println!("✓ Sync profile initialized.");
    println!("  Server: {}", profile.server);
    println!("  User hash: {}", profile.user_hash);
    println!("  Config path: {}", profile.config_path);
    Ok(())
}

async fn handle_sync_push(args: SyncTransferArgs) -> anyhow::Result<()> {
    let mut profile = load_profile().context("run `isartor sync init` first")?;
    if let Some(server) = args.server {
        profile.server = server;
    }
    let config_path = args
        .config_path
        .unwrap_or_else(|| PathBuf::from(profile.config_path.clone()));
    ensure_config_exists(&config_path)?;

    let http = reqwest::Client::builder()
        .user_agent(format!("isartor/{}", env!("CARGO_PKG_VERSION")))
        .build()?;

    let remote = fetch_remote_record(&http, &profile.server, &profile.user_hash).await?;
    if detect_push_conflict(&profile, remote.as_ref()) && !args.force {
        bail!(
            "remote sync blob changed since the last local sync; pull first or re-run with --force"
        );
    }

    let raw = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let syncable = extract_syncable_config(&raw)?;
    let blob = build_sync_blob(syncable);
    let passphrase = prompt_passphrase("Sync passphrase: ")?;
    let encrypted_blob_hex = encrypt_blob(&blob, &passphrase, &profile.salt_hex)?;

    let record = push_remote_record(
        &http,
        &profile.server,
        &profile.user_hash,
        &PutSyncRequest {
            salt_hex: profile.salt_hex.clone(),
            encrypted_blob_hex,
            updated_at: blob.updated_at,
            previous_updated_at: if args.force {
                remote.as_ref().map(|record| record.updated_at)
            } else {
                profile.last_remote_updated_at
            },
        },
    )
    .await?;

    profile.config_path = config_path.display().to_string();
    profile.last_remote_updated_at = Some(record.updated_at);
    profile.last_local_sync_at = Some(file_modified_unix_secs(&config_path)?);
    save_profile(&profile)?;

    println!("✓ Synced encrypted config to {}.", profile.server);
    Ok(())
}

async fn handle_sync_pull(args: SyncTransferArgs) -> anyhow::Result<()> {
    let mut profile = load_profile().context("run `isartor sync init` first")?;
    if let Some(server) = args.server {
        profile.server = server;
    }
    let config_path = args
        .config_path
        .unwrap_or_else(|| PathBuf::from(profile.config_path.clone()));

    let http = reqwest::Client::builder()
        .user_agent(format!("isartor/{}", env!("CARGO_PKG_VERSION")))
        .build()?;
    let remote = fetch_remote_record(&http, &profile.server, &profile.user_hash)
        .await?
        .context("no synced config exists on the server yet")?;

    let pull_conflict = if config_path.exists() {
        detect_pull_conflict(&profile, &remote, &config_path)?
    } else {
        false
    };
    if pull_conflict && !args.force {
        bail!(
            "local config changed since the last sync and remote config also changed; re-run with --force after reviewing the backup"
        );
    }

    if profile.salt_hex != remote.salt_hex {
        profile.salt_hex = remote.salt_hex.clone();
    }

    let passphrase = prompt_passphrase("Sync passphrase: ")?;
    let blob = decrypt_blob(&remote.encrypted_blob_hex, &passphrase, &profile.salt_hex)?;

    let local_raw = if config_path.exists() {
        fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?
    } else {
        String::new()
    };

    if config_path.exists() {
        let backup = save_backup(&config_path, &local_raw)?;
        println!("  Backup written to {}", backup.display());
    }

    let merged = merge_syncable_config(&local_raw, &blob.config_toml)?;
    fs::write(&config_path, &merged)
        .with_context(|| format!("failed to write {}", config_path.display()))?;

    profile.config_path = config_path.display().to_string();
    profile.last_remote_updated_at = Some(remote.updated_at);
    profile.last_local_sync_at = Some(now_unix_secs());
    save_profile(&profile)?;

    println!(
        "✓ Pulled and merged encrypted config from {}.",
        profile.server
    );
    println!("  Local config: {}", config_path.display());
    println!("  Backup path: {}", backup_path(&config_path).display());
    Ok(())
}

fn handle_sync_status(args: SyncStatusArgs) -> anyhow::Result<()> {
    let mut profile = load_profile().context("run `isartor sync init` first")?;
    if let Some(server) = args.server {
        profile.server = server;
    }

    println!("Server: {}", profile.server);
    println!("User hash: {}", profile.user_hash);
    println!("Config path: {}", profile.config_path);
    println!(
        "Last remote update: {}",
        profile
            .last_remote_updated_at
            .map(|ts| ts.to_string())
            .unwrap_or_else(|| "never".to_string())
    );
    println!(
        "Last local sync: {}",
        profile
            .last_local_sync_at
            .map(|ts| ts.to_string())
            .unwrap_or_else(|| "never".to_string())
    );
    Ok(())
}

async fn handle_sync_serve(args: SyncServeArgs) -> anyhow::Result<()> {
    let data_dir = match args.data_dir {
        Some(dir) => dir,
        None => default_server_data_dir()?,
    };
    println!("Starting sync server on {}", args.listen);
    println!("Data dir: {}", data_dir.display());
    run_sync_server(&args.listen, data_dir).await
}

fn prompt_line(prompt: &str) -> anyhow::Result<String> {
    print!("{prompt}");
    io::stdout().flush().context("failed to flush stdout")?;
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .context("failed to read line from stdin")?;
    Ok(input.trim().to_string())
}

fn prompt_passphrase(prompt: &str) -> anyhow::Result<String> {
    print!("{prompt}");
    io::stdout().flush().context("failed to flush stdout")?;
    let passphrase = rpassword::read_password()
        .map_err(|e| anyhow::anyhow!("failed to read passphrase: {e}"))?;
    Ok(passphrase.trim().to_string())
}
