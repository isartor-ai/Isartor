//! # `isartor stop` — Graceful server shutdown
//!
//! Reads the PID from `~/.isartor/isartor.pid` and sends SIGTERM to
//! stop a running Isartor server.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;

/// Default PID file location: `~/.isartor/isartor.pid`
pub fn pid_file_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".isartor").join("isartor.pid"))
}

/// Write the current process PID to `~/.isartor/isartor.pid`.
pub fn write_pid_file() -> Result<()> {
    let path = pid_file_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, std::process::id().to_string())?;
    tracing::debug!(path = %path.display(), pid = std::process::id(), "PID file written");
    Ok(())
}

/// Remove the PID file (best-effort, called on shutdown).
pub fn remove_pid_file() {
    if let Ok(path) = pid_file_path() {
        let _ = std::fs::remove_file(path);
    }
}

// ── CLI ──────────────────────────────────────────────────────────────

#[derive(Parser, Debug, Clone)]
pub struct StopArgs {
    /// Path to PID file (default: ~/.isartor/isartor.pid).
    #[arg(long)]
    pub pid_file: Option<PathBuf>,

    /// Send SIGKILL instead of SIGTERM (force kill).
    #[arg(long)]
    pub force: bool,
}

pub fn handle_stop(args: StopArgs) -> Result<()> {
    let path = match args.pid_file {
        Some(p) => p,
        None => pid_file_path()?,
    };

    if !path.exists() {
        bail!(
            "No PID file found at {}. Is Isartor running?",
            path.display()
        );
    }

    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read PID file: {}", path.display()))?;
    let pid: i32 = contents
        .trim()
        .parse()
        .with_context(|| format!("invalid PID in {}: {:?}", path.display(), contents.trim()))?;

    // Verify the process is actually running.
    let alive = unsafe { libc::kill(pid, 0) } == 0;
    if !alive {
        // Stale PID file — clean up.
        let _ = std::fs::remove_file(&path);
        bail!(
            "Process {} is not running (stale PID file removed). Isartor may have already stopped.",
            pid
        );
    }

    let signal = if args.force {
        libc::SIGKILL
    } else {
        libc::SIGTERM
    };
    let signal_name = if args.force { "SIGKILL" } else { "SIGTERM" };

    let result = unsafe { libc::kill(pid, signal) };
    if result != 0 {
        let err = std::io::Error::last_os_error();
        bail!("Failed to send {} to PID {}: {}", signal_name, pid, err);
    }

    eprintln!("  ✓ Sent {} to Isartor (PID {})", signal_name, pid);

    // Wait briefly for the process to exit, then clean up the PID file.
    for _ in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let still_alive = unsafe { libc::kill(pid, 0) } == 0;
        if !still_alive {
            let _ = std::fs::remove_file(&path);
            eprintln!("  ✓ Isartor stopped.");
            return Ok(());
        }
    }

    if args.force {
        eprintln!("  ⚠ Process {} did not exit within 2 seconds.", pid);
    } else {
        eprintln!(
            "  ⚠ Process {} still running after 2 seconds. Try: isartor stop --force",
            pid
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_file_path_returns_expected_location() {
        let path = pid_file_path().unwrap();
        assert!(path.ends_with(".isartor/isartor.pid"));
    }

    #[test]
    fn write_and_remove_pid_file() {
        // Write PID file.
        write_pid_file().unwrap();
        let path = pid_file_path().unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        let pid: u32 = content.trim().parse().unwrap();
        assert_eq!(pid, std::process::id());

        // Remove PID file.
        remove_pid_file();
        assert!(!path.exists());
    }

    #[test]
    fn stop_with_no_pid_file_returns_error() {
        let args = StopArgs {
            pid_file: Some(PathBuf::from("/tmp/isartor-nonexistent-test.pid")),
            force: false,
        };
        let result = handle_stop(args);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No PID file found")
        );
    }

    #[test]
    fn stop_with_stale_pid_file() {
        // Write a PID file with a PID that is definitely not running.
        let path = PathBuf::from("/tmp/isartor-stale-test.pid");
        std::fs::write(&path, "999999999").unwrap();

        let args = StopArgs {
            pid_file: Some(path.clone()),
            force: false,
        };
        let result = handle_stop(args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not running"));
        // PID file should be cleaned up.
        assert!(!path.exists());
    }
}
