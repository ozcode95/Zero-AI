//! llama-server child-process management.
//!
//! Mirrors [`crate::ovms::process`] but emits its log lines on
//! `llama://log` so the Server page can show two independent log
//! streams without the frontend having to demux a single event channel.

use crate::events;
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde::Serialize;
use std::collections::VecDeque;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Serialize)]
pub struct LlamaLogLine {
    pub ts: String,
    pub level: String,
    pub line: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ExitReason {
    Expected,
    Crashed { code: Option<i32> },
}

pub struct LlamaProcess {
    pid: u32,
    kill_tx: Option<oneshot::Sender<()>>,
    monitor: Option<JoinHandle<()>>,
    log_task: Option<JoinHandle<()>>,
    error_buf: Arc<Mutex<VecDeque<String>>>,
}

const ERROR_BUF_CAP: usize = 32;

impl LlamaProcess {
    pub async fn spawn<F>(
        app: &AppHandle,
        executable: &Path,
        args: &[String],
        env: &[(String, String)],
        working_dir: &Path,
        on_exit: F,
    ) -> Result<Self>
    where
        F: FnOnce(ExitReason) + Send + 'static,
    {
        let mut cmd = Command::new(executable);
        cmd.args(args)
            .envs(env.iter().cloned())
            .current_dir(working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .kill_on_drop(true);

        #[cfg(windows)]
        {
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {}", executable.display()))?;
        let pid = child
            .id()
            .ok_or_else(|| anyhow!("spawned llama-server reported no pid"))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("llama-server stdout pipe missing"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("llama-server stderr pipe missing"))?;

        let error_buf: Arc<Mutex<VecDeque<String>>> =
            Arc::new(Mutex::new(VecDeque::with_capacity(ERROR_BUF_CAP)));

        let stdout_app = app.clone();
        let stderr_app = app.clone();
        let stdout_buf = Arc::clone(&error_buf);
        let stderr_buf = Arc::clone(&error_buf);
        let stdout_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                handle_line(&stdout_app, &stdout_buf, &line).await;
            }
        });
        let stderr_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                handle_line(&stderr_app, &stderr_buf, &line).await;
            }
        });
        let log_task = tokio::spawn(async move {
            let _ = stdout_task.await;
            let _ = stderr_task.await;
        });

        let (kill_tx, kill_rx) = oneshot::channel::<()>();
        let monitor = tokio::spawn(async move {
            let reason = tokio::select! {
                _ = kill_rx => {
                    if let Err(e) = child.start_kill() {
                        tracing::warn!("llama-server start_kill failed: {e}");
                    }
                    let _ = child.wait().await;
                    ExitReason::Expected
                }
                status = child.wait() => {
                    let code = status.ok().and_then(|s| s.code());
                    ExitReason::Crashed { code }
                }
            };
            on_exit(reason);
        });

        Ok(Self {
            pid,
            kill_tx: Some(kill_tx),
            monitor: Some(monitor),
            log_task: Some(log_task),
            error_buf,
        })
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Last few `ERROR`-classified lines joined into a one-paragraph
    /// excerpt. Used by the controller to surface the actual upstream
    /// failure when readiness times out instead of a generic "did not
    /// become ready" message.
    pub async fn error_excerpt(&self) -> Option<String> {
        let g = self.error_buf.lock().await;
        if g.is_empty() {
            return None;
        }
        let joined = g.iter().cloned().collect::<Vec<_>>().join(" | ");
        Some(truncate(&joined, 480))
    }

    pub async fn shutdown(mut self) -> Result<()> {
        if let Some(tx) = self.kill_tx.take() {
            let _ = tx.send(());
        }
        if let Some(m) = self.monitor.take() {
            let _ = m.await;
        }
        if let Some(l) = self.log_task.take() {
            let _ = l.await;
        }
        Ok(())
    }
}

impl Drop for LlamaProcess {
    fn drop(&mut self) {
        if let Some(tx) = self.kill_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Process a single child log line: classify it, forward it to the frontend
/// log stream and the backend tracing log, buffer errors for the readiness
/// excerpt, and surface targeted guidance for well-known failure modes.
async fn handle_line(app: &AppHandle, buf: &Arc<Mutex<VecDeque<String>>>, line: &str) {
    let level = classify(line);
    emit_log(app, level, line);
    // Forward to tracing so startup failures are visible in the backend log
    // file, not just the frontend popover.
    match level {
        "ERROR" => {
            tracing::error!(target: "llama::server", "{line}");
            push_error(buf, line).await;
        }
        "WARN" => tracing::warn!(target: "llama::server", "{line}"),
        _ => tracing::info!(target: "llama::server", "{line}"),
    }

    // The OpenVINO backend silently drops to CPU when it can't enumerate the
    // requested device. That single upstream line is easy to miss, so when we
    // see it we emit an explicit, actionable hint on the same log stream.
    if is_openvino_device_fallback(line) {
        let hint = "Intel GPU not detected by OpenVINO — running on CPU. The Intel GPU \
            compute runtime (OpenCL/Level-Zero) is not registered on this system. Reinstall \
            the latest Intel Graphics driver from intel.com (use the 'clean installation' \
            option) to enable GPU inference, or set GGML_OPENVINO_DEVICE=CPU to silence this.";
        emit_log(app, "WARN", hint);
        tracing::warn!(target: "llama::server", "{hint}");
    }
}

/// Detect the OpenVINO backend's "requested device unavailable, using CPU"
/// line (e.g. `W GGML OpenVINO Backend: device GPU is not available,
/// fallback to CPU`) regardless of the exact device named.
fn is_openvino_device_fallback(line: &str) -> bool {
    let l = line.to_lowercase();
    l.contains("openvino") && l.contains("not available") && l.contains("fallback to cpu")
}

fn emit_log(app: &AppHandle, level: &'static str, line: &str) {
    let _ = app.emit(
        events::LLAMA_LOG,
        LlamaLogLine {
            ts: Utc::now().to_rfc3339(),
            level: level.to_string(),
            line: line.to_string(),
        },
    );
}

async fn push_error(buf: &Arc<Mutex<VecDeque<String>>>, line: &str) {
    let mut g = buf.lock().await;
    if g.len() == ERROR_BUF_CAP {
        g.pop_front();
    }
    g.push_back(line.to_string());
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

/// llama.cpp logs lines like `slot ... | err: ...` and `error: ...`
/// to stderr — keep the classifier coarse-grained, like OVMS's.
fn classify(line: &str) -> &'static str {
    let l = line.to_lowercase();
    if l.contains("error") || l.contains(" err:") || l.contains("fatal") || l.contains("panic") {
        "ERROR"
    } else if l.contains("warn") || is_openvino_device_fallback(line) {
        "WARN"
    } else {
        "INFO"
    }
}
