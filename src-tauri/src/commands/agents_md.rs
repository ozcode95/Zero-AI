//! Tauri commands for the AGENTS.md context-injection feature.
//!
//! Surfaces a small editor + toggle over the two AGENTS.md scopes
//! (global `~/.zero/AGENTS.md` and the active project's `AGENTS.md` /
//! `CLAUDE.md` fallback chain) so users can manage their standing
//! instructions from the UI without hand-editing files. The runner
//! re-reads every file on each turn, so writes take effect on the next
//! message — no restart needed.

use crate::error::IpcResult;
use crate::{paths, workspace};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentsMdFile {
    /// Absolute path of the file the editor would write to. For the
    /// project scope with no open workspace this is `None` and the UI
    /// should gate the editor on that.
    pub path: Option<String>,
    /// Whether `exists` was true on disk at read time. Lets the UI show
    /// \"(not yet created)\" on a fresh install.
    pub exists: bool,
    /// File contents, or the empty string when `path` is `None` or the
    /// file doesn't exist yet — so the editor starts from a blank slate.
    pub content: String,
}

/// Read the AGENTS.md file for a scope.
///
/// `scope = "global"` returns `~/.zero/AGENTS.md`; `scope = "project"`
/// returns the active workspace's preferred instruction file — the
/// same fallback chain the runner uses (`AGENTS.md` → `CLAUDE.md` →
/// `.zero/AGENTS.md`), with the first existing one surfaced for editing.
/// When none exist we fall back to the `<workspace>/AGENTS.md` path so
/// the editor defaults to the conventional name on first save.
#[tauri::command]
pub async fn agents_md_get(scope: String) -> IpcResult<AgentsMdFile> {
    let path = resolve_scope_path(&scope)?;
    let Some(path) = path else {
        return Ok(AgentsMdFile {
            path: None,
            exists: false,
            content: String::new(),
        });
    };
    let exists = path.is_file();
    let content = if exists {
        tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| e.to_string())?
    } else {
        String::new()
    };
    Ok(AgentsMdFile {
        path: Some(path.to_string_lossy().into_owned()),
        exists,
        content,
    })
}

/// Write the AGENTS.md file for a scope, creating it (and parent
/// directories) as needed. An empty `content` clears the file. Project
/// scope requires an open workspace; otherwise returns an error so the
/// UI can prompt the user to pick one.
#[tauri::command]
pub async fn agents_md_set(scope: String, content: String) -> IpcResult<()> {
    let Some(path) = resolve_scope_path(&scope)? else {
        return Err("no workspace is open — cannot edit the project AGENTS.md".into());
    };
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| e.to_string())?;
    }
    tokio::fs::write(&path, content)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Resolve a scope label (`"global"` / `"project"`) to the path the
/// editor should read/write.
///
/// Returns `Ok(None)` for the project scope when no workspace is open.
/// For the project scope, follows the same fallback chain the runner
/// uses on the read side, defaulting to `<workspace>/AGENTS.md` when
/// none exist (so a first-time save picks the conventional name).
fn resolve_scope_path(scope: &str) -> IpcResult<Option<PathBuf>> {
    match scope {
        "global" => Ok(Some(paths::agents_md_global().map_err(|e| e.to_string())?)),
        "project" => {
            let Some(root) = workspace::get() else {
                return Ok(None);
            };
            let candidates: [PathBuf; 3] = [
                root.join("AGENTS.md"),
                root.join("CLAUDE.md"),
                root.join(".zero").join("AGENTS.md"),
            ];
            Ok(Some(
                candidates
                    .into_iter()
                    .find(|p| p.is_file())
                    .unwrap_or_else(|| root.join("AGENTS.md")),
            ))
        }
        other => Err(
            format!("unknown AGENTS.md scope `{other}` — expected `global` or `project`").into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_global_scope_points_into_zero_root() {
        let _g = crate::workspace::WORKSPACE_TEST_GUARD.lock().unwrap();
        crate::workspace::set(None);
        let p = resolve_scope_path("global")
            .unwrap()
            .expect("global path resolves");
        assert!(p.ends_with("AGENTS.md"));
    }

    #[test]
    fn resolve_project_scope_none_without_workspace() {
        let _g = crate::workspace::WORKSPACE_TEST_GUARD.lock().unwrap();
        crate::workspace::set(None);
        assert!(resolve_scope_path("project").unwrap().is_none());
    }

    #[test]
    fn resolve_project_scope_picks_existing_claude_md_fallback() {
        let _g = crate::workspace::WORKSPACE_TEST_GUARD.lock().unwrap();
        let dir = std::env::temp_dir().join("zero-agents-md-cli-resolve-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("CLAUDE.md"), "legacy").unwrap();

        crate::workspace::set(Some(dir.clone()));
        let p = resolve_scope_path("project")
            .unwrap()
            .expect("workspace open");
        crate::workspace::set(None);
        assert_eq!(p, dir.join("CLAUDE.md"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
