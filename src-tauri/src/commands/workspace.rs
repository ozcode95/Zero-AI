//! Workspace (project root) IPC commands.
//!
//! The user "opens" a folder to work on a codebase. Its path persists in
//! `Settings.workspace_root` and is mirrored into the process-global cache
//! in [`crate::workspace`] so the built-in `fs.*` tools resolve relative
//! paths against it. Opening a workspace also lifts the core file tools out
//! of the globally-disabled set so coding works immediately — destructive
//! writes stay gated behind the per-call confirm prompt.

use crate::error::IpcResult;
use crate::settings::Settings;
use crate::workspace::CODING_TOOLS;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceInfo {
    /// Absolute path to the workspace root.
    pub path: String,
    /// Final path component, shown as the workspace's name in the UI.
    pub name: String,
    /// Whether the path still resolves to a directory on disk.
    pub exists: bool,
}

fn info_for(path: &Path) -> WorkspaceInfo {
    WorkspaceInfo {
        path: path.display().to_string(),
        name: crate::workspace::folder_name(path),
        exists: path.is_dir(),
    }
}

/// The currently-open workspace, or `None` if the user hasn't opened one.
#[tauri::command]
pub async fn workspace_get() -> IpcResult<Option<WorkspaceInfo>> {
    Ok(crate::workspace::get().as_deref().map(info_for))
}

/// Open `path` as the active workspace. Validates that it's a directory,
/// persists it, refreshes the in-process cache, and enables the core
/// coding tools.
#[tauri::command]
pub async fn workspace_set(path: String) -> IpcResult<WorkspaceInfo> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("workspace path is empty".into());
    }
    let pb = PathBuf::from(trimmed);
    if !pb.is_dir() {
        return Err(format!("not a directory: {}", pb.display()).into());
    }

    let mut s = Settings::load().await.map_err(|e| e.to_string())?;
    s.workspace_root = Some(pb.display().to_string());
    enable_coding_tools(&mut s);
    s.save().await.map_err(|e| e.to_string())?;

    crate::workspace::set(Some(pb.clone()));
    Ok(info_for(&pb))
}

/// Close the active workspace. Relative paths fall back to the process
/// working directory again. Idempotent.
#[tauri::command]
pub async fn workspace_clear() -> IpcResult<()> {
    let mut s = Settings::load().await.map_err(|e| e.to_string())?;
    s.workspace_root = None;
    s.save().await.map_err(|e| e.to_string())?;
    crate::workspace::set(None);
    Ok(())
}

/// Drop the [`CODING_TOOLS`] from the user's globally-disabled set. Only
/// ever removes entries — never re-disables — so a user who deliberately
/// turned a tool back off keeps their choice on the next workspace open
/// only insofar as we don't touch tools outside this list.
fn enable_coding_tools(s: &mut Settings) {
    s.builtin_tools_disabled
        .retain(|t| !CODING_TOOLS.contains(&t.as_str()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enable_coding_tools_removes_only_file_tools() {
        let mut s = Settings::default();
        s.builtin_tools_disabled = vec![
            "fs.edit".into(),
            "fs.write".into(),
            "http.fetch".into(),
            "clipboard.read".into(),
        ];
        enable_coding_tools(&mut s);
        // File tools lifted, unrelated tools untouched.
        assert!(!s.builtin_tools_disabled.contains(&"fs.edit".to_string()));
        assert!(!s.builtin_tools_disabled.contains(&"fs.write".to_string()));
        assert!(s.builtin_tools_disabled.contains(&"http.fetch".to_string()));
        assert!(s
            .builtin_tools_disabled
            .contains(&"clipboard.read".to_string()));
    }
}
