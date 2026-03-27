use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;

const DEFAULT_TAIL_LINES: usize = 200;
const FOLLOW_POLL_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Parser, Debug, Clone)]
pub struct LogsArgs {
    /// Follow appended log lines in real time.
    #[arg(long, default_value_t = false)]
    pub follow: bool,
}

pub fn handle_logs(args: LogsArgs) -> Result<()> {
    let path = crate::cli::up::startup_log_path()?;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    handle_logs_at_path(&path, &args, &mut lock)
}

fn handle_logs_at_path(path: &Path, args: &LogsArgs, out: &mut dyn Write) -> Result<()> {
    if !path.exists() {
        bail!(
            "No log file found at {}. Start Isartor with `isartor up --detach` first.",
            path.display()
        );
    }

    writeln!(out, "Isartor Logs")?;
    writeln!(out, "  Path: {}", path.display())?;
    writeln!(
        out,
        "  Mode: {}",
        if args.follow {
            "follow (Ctrl+C to stop)"
        } else {
            "snapshot"
        }
    )?;
    writeln!(out)?;

    let initial = read_last_lines(path, DEFAULT_TAIL_LINES)?;
    if initial.is_empty() {
        writeln!(out, "  (log file is empty)")?;
    } else {
        write!(out, "{}", initial)?;
        if !initial.ends_with('\n') {
            writeln!(out)?;
        }
    }
    out.flush()?;

    if args.follow {
        follow_log(path, out)?;
    }

    Ok(())
}

fn read_last_lines(path: &Path, max_lines: usize) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = VecDeque::with_capacity(max_lines);

    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
        if lines.len() == max_lines {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    Ok(lines.into_iter().collect::<Vec<_>>().join("\n"))
}

fn follow_log(path: &Path, out: &mut dyn Write) -> Result<()> {
    let mut offset = std::fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();

    loop {
        std::thread::sleep(FOLLOW_POLL_INTERVAL);
        offset = emit_new_log_bytes(path, offset, out)?;
        out.flush()?;
    }
}

fn emit_new_log_bytes(path: &Path, mut offset: u64, out: &mut dyn Write) -> Result<u64> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    if metadata.len() < offset {
        writeln!(
            out,
            "\n--- log file truncated; restarting from beginning ---"
        )?;
        offset = 0;
    }

    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .with_context(|| format!("failed to seek {}", path.display()))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .with_context(|| format!("failed to read {}", path.display()))?;

    if !buffer.is_empty() {
        write!(out, "{}", String::from_utf8_lossy(&buffer))?;
    }

    Ok(offset + buffer.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn read_last_lines_returns_tail_only() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("isartor.log");
        std::fs::write(&path, "one\ntwo\nthree\nfour\n").unwrap();

        let tail = read_last_lines(&path, 2).unwrap();
        assert_eq!(tail, "three\nfour");
    }

    #[test]
    fn handle_logs_at_path_errors_when_missing() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("missing.log");
        let mut output = Vec::new();
        let err = handle_logs_at_path(&path, &LogsArgs { follow: false }, &mut output).unwrap_err();
        assert!(
            err.to_string()
                .contains("Start Isartor with `isartor up --detach` first")
        );
    }

    #[test]
    fn handle_logs_at_path_prints_header_and_snapshot() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("isartor.log");
        std::fs::write(&path, "alpha\nbeta\n").unwrap();

        let mut output = Vec::new();
        handle_logs_at_path(&path, &LogsArgs { follow: false }, &mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Isartor Logs"));
        assert!(rendered.contains("snapshot"));
        assert!(rendered.contains("alpha\nbeta"));
    }

    #[test]
    fn emit_new_log_bytes_reads_appended_content() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("isartor.log");
        std::fs::write(&path, "seed\n").unwrap();
        let offset = std::fs::metadata(&path).unwrap().len();

        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(file, "next").unwrap();

        let mut output = Vec::new();
        let next_offset = emit_new_log_bytes(&path, offset, &mut output).unwrap();
        let rendered = String::from_utf8(output).unwrap();

        assert_eq!(rendered, "next\n");
        assert!(next_offset > offset);
    }

    #[test]
    fn emit_new_log_bytes_recovers_after_truncation() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("isartor.log");
        std::fs::write(&path, "before\n").unwrap();
        let offset = std::fs::metadata(&path).unwrap().len();

        std::fs::write(&path, "after\n").unwrap();

        let mut output = Vec::new();
        let next_offset = emit_new_log_bytes(&path, offset, &mut output).unwrap();
        let rendered = String::from_utf8(output).unwrap();

        assert!(rendered.contains("log file truncated"));
        assert!(rendered.contains("after\n"));
        assert_eq!(next_offset, std::fs::metadata(&path).unwrap().len());
    }
}
