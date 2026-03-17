use anyhow::{anyhow, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

fn dir_is_writable(dir: &Path) -> bool {
    if fs::create_dir_all(dir).is_err() {
        return false;
    }

    let test_path = dir.join(format!(".isartor-write-test-{}", std::process::id()));

    let mut file = match fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&test_path)
    {
        Ok(f) => f,
        Err(_) => return false,
    };

    let ok = file.write_all(b"ok").is_ok();
    let _ = fs::remove_file(&test_path);
    ok
}

/// Resolve a writable cache directory for hf-hub.
///
/// hf-hub expects a *hub cache dir* (usually `~/.cache/huggingface/hub`).
/// In distroless/non-root containers, `HOME` can point to an unwritable
/// location (often `/`), which causes downloads to fail with EACCES.
///
/// Priority:
/// 1) `ISARTOR_HF_CACHE_DIR` / `ISARTOR__HF_CACHE_DIR` (explicit override)
/// 2) `HF_HOME` / `ISARTOR_HF_HOME` (uses `$HF_HOME/hub`)
/// 3) `$HOME/.cache/huggingface/hub` (when HOME looks sane)
/// 4) `$TMPDIR/huggingface/hub` (safe fallback)
pub fn writable_hf_hub_cache_dir() -> Result<PathBuf> {
    let mut tried: Vec<PathBuf> = Vec::new();

    if let Ok(v) =
        std::env::var("ISARTOR_HF_CACHE_DIR").or_else(|_| std::env::var("ISARTOR__HF_CACHE_DIR"))
    {
        tried.push(PathBuf::from(v));
    }

    if let Ok(hf_home) = std::env::var("HF_HOME").or_else(|_| std::env::var("ISARTOR_HF_HOME")) {
        tried.push(PathBuf::from(hf_home).join("hub"));
    }

    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        if home != PathBuf::from("/") {
            tried.push(home.join(".cache").join("huggingface").join("hub"));
        }
    }

    tried.push(std::env::temp_dir().join("huggingface").join("hub"));

    for dir in &tried {
        if dir_is_writable(dir) {
            return Ok(dir.clone());
        }
    }

    Err(anyhow!(
        "No writable hf-hub cache directory found. Tried: {}. Hint: set HF_HOME=/tmp/huggingface (or ISARTOR_HF_CACHE_DIR) to a writable path.",
        tried
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}
