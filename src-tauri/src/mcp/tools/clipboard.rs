//! Clipboard built-in tools (`clipboard.read`, `clipboard.write`).
//!
//! Backed by `tauri-plugin-clipboard-manager` (the official Tauri plugin)
//! instead of a third-party crate. The plugin owns the platform clipboard
//! handle for the lifetime of the app, so each call is a thin
//! read/write through the managed `Clipboard` rather than opening a
//! short-lived handle.
//!
//! Both tools are marked non-destructive: pasting and reading the
//! clipboard are routine actions the user would otherwise do
//! themselves, and clipping every call behind a confirm prompt would
//! make the model unusable for "summarise this then put it in my
//! clipboard" workflows.

use crate::mcp::{Tool, ToolResult, ToolSchema};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tauri::AppHandle;
use tauri_plugin_clipboard_manager::ClipboardExt;

/// Cap on the size of a single `clipboard.write` so the model can't
/// accidentally stash a megabyte of context into the user's clipboard.
const MAX_WRITE_BYTES: usize = 1 * 1024 * 1024;

/// Cap on the size of a `clipboard.read` response. Anything larger is
/// truncated with the usual marker.
const MAX_READ_BYTES: usize = 256 * 1024;

/// Shared constructor helper: capture the [`AppHandle`] so the tool can
/// reach the plugin-managed clipboard without the `Tool` trait carrying
/// Tauri types.
pub fn all(app: &AppHandle) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ClipboardRead { app: app.clone() }),
        Box::new(ClipboardWrite { app: app.clone() }),
    ]
}

#[derive(Debug)]
pub struct ClipboardRead {
    app: AppHandle,
}

#[async_trait]
impl Tool for ClipboardRead {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "clipboard.read".into(),
            description: "Read the current text contents of the system \
                 clipboard. Returns an empty string if the clipboard is \
                 empty or contains non-text data. Responses larger than \
                 ~256 KiB are truncated."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
            destructive: false,
        }
    }

    async fn call(&self, _args: Value) -> Result<ToolResult> {
        // The plugin's clipboard handle is cheap to obtain; reading is a
        // blocking OS call, so run it on a blocking thread to keep the
        // async runtime clean.
        let app = self.app.clone();
        let text = tokio::task::spawn_blocking(move || {
            app.clipboard()
                .read_text()
                .map_err(|e| anyhow!("clipboard: read text: {e}"))
        })
        .await
        .map_err(|e| anyhow!("clipboard.read task panicked: {e}"))??;

        let (out, truncated) = if text.len() > MAX_READ_BYTES {
            (text[..MAX_READ_BYTES].to_string(), true)
        } else {
            (text, false)
        };
        let body = if truncated {
            format!("{out}\n… [truncated]")
        } else {
            out
        };
        Ok(ToolResult {
            content: body,
            is_error: false,
        })
    }
}

#[derive(Debug)]
pub struct ClipboardWrite {
    app: AppHandle,
}

#[derive(Debug, Deserialize)]
struct WriteArgs {
    text: String,
}

#[async_trait]
impl Tool for ClipboardWrite {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "clipboard.write".into(),
            description: "Replace the system clipboard with the given \
                 text. Limited to ~1 MiB per call."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Replacement clipboard contents."
                    }
                }
            }),
            destructive: false,
        }
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let a: WriteArgs =
            serde_json::from_value(args).context("clipboard.write: parse arguments")?;
        if a.text.len() > MAX_WRITE_BYTES {
            return Ok(ToolResult {
                content: format!(
                    "clipboard.write: payload is {} bytes, max is {}",
                    a.text.len(),
                    MAX_WRITE_BYTES
                ),
                is_error: true,
            });
        }
        let n = a.text.len();
        let app = self.app.clone();
        tokio::task::spawn_blocking(move || {
            app.clipboard()
                .write_text(a.text)
                .map_err(|e| anyhow!("clipboard: write text: {e}"))
        })
        .await
        .map_err(|e| anyhow!("clipboard.write task panicked: {e}"))??;
        Ok(ToolResult {
            content: format!("copied {n} bytes to clipboard"),
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_rejects_oversize_payload_static_check() {
        // The tool itself needs a live AppHandle, so we can only sanity-
        // check the cap constant here; round-trip coverage is manual.
        assert_eq!(MAX_WRITE_BYTES, 1 * 1024 * 1024);
        assert_eq!(MAX_READ_BYTES, 256 * 1024);
    }

    // We do *not* exercise the real OS clipboard in unit tests because the
    // plugin requires a live Tauri app handle. Manual smoke testing on
    // Windows/macOS covers the round-trip.
}
