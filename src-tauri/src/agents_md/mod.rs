//! AGENTS.md — auto-loaded project / user instruction files.
//!
//! Mirrors Claude Code's `CLAUDE.md` / `AGENTS.md` convention: a markdown
//! file the user authors with standing instructions for the assistant,
//! injected into the system prompt of every chat turn. Resolution picks
//! up to two files (global then project) so a user's personal conventions
//! compose with a repo-scoped policy file.
//!
//! The loader is **best-effort**: a missing file is the common case and
//! contributes nothing; a read error or an oversized file is logged and
//! skipped so a corrupt `AGENTS.md` never bricks a chat turn. Each body
//! is capped at [`MAX_AGENTS_MD_BYTES`] (mirroring the skills cap) so a
//! stray 5 MB `AGENTS.md` can't silently nuke the context window.
//!
//! Resolution hierarchy (later entries are appended **after** earlier):
//!
//! 1. Global / user: `~/.zero/AGENTS.md`
//! 2. Project, in fallback order (first that exists wins):
//!    - `<workspace_root>/AGENTS.md`
//!    - `<workspace_root>/CLAUDE.md`
//!    - `<workspace_root>/.zero/AGENTS.md`

use crate::paths;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Soft cap on a single AGENTS.md body. Mirrors the skills cap — a
/// prompt-injected file bigger than 32 KiB almost certainly means the
/// user is dumping a manual they should be linking instead.
pub const MAX_AGENTS_MD_BYTES: usize = 32 * 1024;

/// One loaded instruction file. `source` is a short display label so the
/// injected header can name the section without leaking absolute paths
/// the user wouldn't recognise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentsDoc {
    /// Human label for the prompt header: `"user instructions"` for the
    /// global file, `"the project's AGENTS.md"` for the project file.
    pub source: String,
    /// Truncated prompt body. Frontmatter is not stripped — AGENTS.md is
    /// plain markdown and everything in it is treated as instructions.
    pub body: String,
}

/// Best-effort load of every AGENTS.md in the resolution hierarchy.
///
/// Missing files are skipped silently (the common case). Read / decode
/// errors and oversize files log a warning and are skipped so a corrupt
/// file never breaks the chat turn. Returns the resolved docs **in
/// hierarchy order** (global before project) so the caller can append
/// them to the system prompt in that order.
pub async fn load() -> Vec<AgentsDoc> {
    let mut out = Vec::new();

    if let Ok(global) = paths::agents_md_global() {
        if let Some(body) = read_capped(&global).await {
            out.push(AgentsDoc {
                source: "user instructions".to_string(),
                body,
            });
        }
    }

    if let Some(root) = crate::workspace::get() {
        if let Some((path, label)) = resolve_project_file(&root) {
            if let Some(body) = read_capped(&path).await {
                out.push(AgentsDoc {
                    source: label,
                    body,
                });
            }
        }
    }

    out
}

/// Pick the first existing project-level instruction file from the
/// fallback list. Returns the path plus a short header label that names
/// which file was found (so the prompt section reads accurately even
/// when the user opted for the legacy `CLAUDE.md` name).
fn resolve_project_file(root: &Path) -> Option<(PathBuf, String)> {
    let candidates: [(PathBuf, &str); 3] = [
        (root.join("AGENTS.md"), "AGENTS.md"),
        (root.join("CLAUDE.md"), "CLAUDE.md"),
        (root.join(".zero").join("AGENTS.md"), ".zero/AGENTS.md"),
    ];
    for (path, name) in candidates {
        if path.is_file() {
            return Some((path, format!("the project's {name}")));
        }
    }
    None
}

/// Read `path` to a string, capping it at [`MAX_AGENTS_MD_BYTES`] and
/// appending an `[truncated]` marker on overflow. Returns `None` when
/// the file is missing, unreadable, or empty after trimming so a
/// missing AGENTS.md contributes zero prompt weight.
async fn read_capped(path: &Path) -> Option<String> {
    let bytes = match tokio::fs::read_to_string(path).await {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!("AGENTS.md: failed to read {}: {e}", path.display());
            return None;
        }
    };
    if bytes.is_empty() {
        return None;
    }
    Some(truncate_body(&bytes))
}

/// Cap a body to [`MAX_AGENTS_MD_BYTES`], snapping back to a UTF-8 char
/// boundary and appending a `[truncated]` marker so the model knows the
/// file was clipped. Mirrors [`crate::skills::truncate_body`] semantics
/// without depending on its private visibility.
fn truncate_body(body: &str) -> String {
    if body.len() <= MAX_AGENTS_MD_BYTES {
        return body.to_string();
    }
    let mut cut = MAX_AGENTS_MD_BYTES;
    while !body.is_char_boundary(cut) && cut > 0 {
        cut -= 1;
    }
    let mut out = body[..cut].to_string();
    out.push_str("\n\n[truncated]");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_body_under_cap_is_unchanged() {
        assert_eq!(truncate_body("hello"), "hello");
    }

    #[test]
    fn truncate_body_appends_marker_on_overflow_and_snaps_to_char_boundary() {
        let mut big = "x".repeat(MAX_AGENTS_MD_BYTES + 10);
        // Insert a multibyte char near the cut point so the boundary-snap
        // logic actually has to step back. 'é' is two bytes in UTF-8.
        big.insert(MAX_AGENTS_MD_BYTES - 1, 'é');
        let out = truncate_body(&big);
        assert!(out.ends_with("[truncated]"));
        // Should still be valid UTF-8 (the snap-back is the whole point).
        let _ = out.chars().count();
    }

    #[test]
    fn truncate_body_marker_keeps_total_size_bounded() {
        let big = "x".repeat(MAX_AGENTS_MD_BYTES + 1024);
        let out = truncate_body(&big);
        assert!(out.len() <= MAX_AGENTS_MD_BYTES + 32);
    }

    #[tokio::test]
    async fn read_capped_returns_none_for_missing_file() {
        let p = if cfg!(windows) {
            Path::new(r"C:\zero-ai-nonexistent-agents-md-test-1234.md")
        } else {
            Path::new("/zero-ai-nonexistent-agents-md-test-1234.md")
        };
        assert!(read_capped(p).await.is_none());
    }

    #[tokio::test]
    async fn read_capped_returns_none_for_empty_file() {
        let dir = std::env::temp_dir().join("zero-agents-md-test-empty");
        let _ = tokio::fs::create_dir_all(&dir).await;
        let p = dir.join("empty.md");
        tokio::fs::write(&p, "").await.unwrap();
        assert!(read_capped(&p).await.is_none());
        let _ = tokio::fs::remove_file(&p).await;
    }

    #[tokio::test]
    async fn read_capped_truncates_oversize_file_with_marker() {
        let dir = std::env::temp_dir().join("zero-agents-md-test-oversize");
        let _ = tokio::fs::create_dir_all(&dir).await;
        let p = dir.join("big.md");
        let big = "x".repeat(MAX_AGENTS_MD_BYTES + 50);
        tokio::fs::write(&p, &big).await.unwrap();
        let out = read_capped(&p)
            .await
            .expect("oversize file still loads (capped)");
        assert!(out.ends_with("[truncated]"));
        let _ = tokio::fs::remove_file(&p).await;
    }

    #[test]
    fn resolve_project_file_prefers_agents_md_over_claude_md() {
        let dir = std::env::temp_dir().join("zero-agents-md-resolve-test");
        let _ = std::fs::create_dir_all(&dir);
        let agents = dir.join("AGENTS.md");
        let claude = dir.join("CLAUDE.md");
        std::fs::write(&agents, "a").unwrap();
        std::fs::write(&claude, "c").unwrap();

        let (path, label) = resolve_project_file(&dir).expect("at least one candidate");
        assert_eq!(path, agents);
        assert_eq!(label, "the project's AGENTS.md");

        // Remove the preferred one — the next candidate (CLAUDE.md) wins
        // and the label changes to reflect which file was actually picked.
        std::fs::remove_file(&agents).unwrap();
        let (path, label) = resolve_project_file(&dir).expect("CLAUDE.md fallback");
        assert_eq!(path, claude);
        assert_eq!(label, "the project's CLAUDE.md");

        std::fs::remove_file(&claude).unwrap();
        assert!(resolve_project_file(&dir).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_project_file_falls_back_to_dotzero_agents_md() {
        let dir = std::env::temp_dir().join("zero-agents-md-dotzero-test");
        let _ = std::fs::create_dir_all(dir.join(".zero"));
        let dotzero = dir.join(".zero").join("AGENTS.md");
        std::fs::write(&dotzero, "n").unwrap();

        let (path, label) = resolve_project_file(&dir).expect(".zero/AGENTS.md fallback");
        assert_eq!(path, dotzero);
        assert_eq!(label, "the project's .zero/AGENTS.md");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
