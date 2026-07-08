//! Active coding workspace (project root).
//!
//! A single optional folder the user "opens" to work on a codebase. When
//! set:
//!
//!   * the built-in `fs.*` tools resolve **relative** paths against it, so
//!     the agent can use project-relative paths like `src/main.rs`;
//!   * the chat runner renders file edits relative to it (`src/main.rs`
//!     instead of a long absolute path); and
//!   * the system prompt tells the model where the project lives.
//!
//! The canonical value persists in `Settings.workspace_root`. This module
//! keeps a process-global cache so the app-less, synchronous `resolve_path`
//! helper in `mcp::tools::fs` can consult it without a disk read on every
//! call. `AppState::init` hydrates it at startup and the `workspace_*`
//! commands keep it in sync.

use once_cell::sync::Lazy;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

static ROOT: Lazy<RwLock<Option<PathBuf>>> = Lazy::new(|| RwLock::new(None));

/// Serialises every `#[test]` / `#[tokio::test]` that mutates or reads the
/// process-global [`ROOT`] cache so tests across the crate that depend on
/// the workspace state (agents_md loader, hooks resolver,
/// `commands::agents_md`, this module's round-trip test) can't race
/// each other. Async tests that hold this guard across `.await` must use
/// `#[tokio::test(flavor = "current_thread")]` so the future need not be
/// `Send` (a `std::sync::MutexGuard` is `!Send`).
#[cfg(test)]
pub(crate) static WORKSPACE_TEST_GUARD: Lazy<std::sync::Mutex<()>> =
    Lazy::new(|| std::sync::Mutex::new(()));

/// The built-in file tools an agent needs to read and edit a project.
/// This is the single source of truth for "which tools make coding work":
///
///   * `commands::workspace` lifts them out of the globally-disabled set
///     when the user opens a folder, and
///   * `chat::runner` force-advertises them on every round (bypassing lazy
///     tool discovery) while a workspace is open, so the model can read and
///     edit immediately instead of first discovering them via `tools.list`.
pub const CODING_TOOLS: &[&str] = &[
    "fs.list", "fs.read", "fs.stat", "fs.glob", "fs.grep", "fs.edit", "fs.write",
];

/// Whether `name` is one of the [`CODING_TOOLS`].
pub fn is_coding_tool(name: &str) -> bool {
    CODING_TOOLS.contains(&name)
}

/// Replace the active workspace root. `None` clears it. A poisoned lock is
/// treated as "no workspace" rather than panicking — the feature degrades
/// to plain absolute/cwd path resolution instead of taking the app down.
pub fn set(root: Option<PathBuf>) {
    if let Ok(mut guard) = ROOT.write() {
        *guard = root;
    }
}

/// The current workspace root, if one is set.
pub fn get() -> Option<PathBuf> {
    ROOT.read().ok().and_then(|g| g.clone())
}

/// The workspace folder's display name (its final path component).
pub fn name() -> Option<String> {
    get().map(|p| folder_name(&p))
}

/// Final path component of `p`, falling back to the whole path for roots
/// like `C:\` that have no file name.
pub fn folder_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

/// Render `path` relative to the workspace root when it lives inside it,
/// otherwise return its absolute display form. Used for compact, readable
/// file-edit headers (`src/main.rs` instead of `C:\Users\…\src\main.rs`).
/// Path separators are normalised to `/` so the same chat reads the same
/// on every platform.
pub fn relativize(path: &Path) -> String {
    if let Some(root) = get() {
        if let Ok(rel) = path.strip_prefix(&root) {
            let s = rel.to_string_lossy();
            if s.is_empty() {
                return folder_name(path);
            }
            return s.replace('\\', "/");
        }
    }
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // These mutate process-global state, so they run serially behind a
    // single test fn to avoid cross-test interference.
    #[test]
    fn set_get_relativize_roundtrip() {
        let _g = super::WORKSPACE_TEST_GUARD.lock().unwrap();
        set(None);
        assert!(get().is_none());
        assert!(name().is_none());

        let root = PathBuf::from(if cfg!(windows) {
            r"C:\proj\app"
        } else {
            "/proj/app"
        });
        set(Some(root.clone()));
        assert_eq!(get().as_deref(), Some(root.as_path()));
        assert_eq!(name().as_deref(), Some("app"));

        // A path inside the workspace relativises to a forward-slash path.
        let inside = root.join("src").join("main.rs");
        assert_eq!(relativize(&inside), "src/main.rs");

        // A path outside the workspace keeps its absolute form.
        let outside = PathBuf::from(if cfg!(windows) {
            r"D:\other\file.txt"
        } else {
            "/other/file.txt"
        });
        assert_eq!(relativize(&outside), outside.display().to_string());

        // The root itself relativises to the folder name.
        assert_eq!(relativize(&root), "app");

        // Clean up so we don't leak state into other tests.
        set(None);
    }

    #[test]
    fn coding_tools_cover_read_and_write_paths() {
        // The read/search tools and both mutation tools must be classed as
        // coding tools so the runner force-advertises them in a workspace.
        for name in [
            "fs.read", "fs.list", "fs.glob", "fs.grep", "fs.edit", "fs.write",
        ] {
            assert!(is_coding_tool(name), "{name} should be a coding tool");
        }
        // Unrelated built-ins must not be swept in.
        assert!(!is_coding_tool("http.fetch"));
        assert!(!is_coding_tool("clipboard.read"));
        assert!(!is_coding_tool("memory"));
    }
}
