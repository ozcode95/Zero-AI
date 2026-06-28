//! Knowledge-base documents for the embedding feature.
//!
//! A "document" is a single user-provided file copied into
//! `~/.zero/documents/`. The frontend hands us a *source* path (from the OS
//! file picker); we copy the bytes into a stable, de-duplicated location and
//! return the metadata the Embedding page renders.
//!
//! When the embedding feature is enabled (`settings.embedding.enabled`), the
//! text of every *enabled* document — i.e. every document whose id is not in
//! `settings.embedding.documents_disabled` — is injected into the system
//! prompt for every chat turn (see `chat::runner::render_documents_context`).
//! Because the runner re-reads settings + the documents directory every turn,
//! adding, removing, enabling, or disabling a document takes effect on the
//! very next message — across every conversation.

use crate::paths;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Hard cap on a single document. Keeps a stray multi-GB drop from instantly
/// OOMing the runner — the user gets a clear error instead.
pub const MAX_DOCUMENT_BYTES: u64 = 16 * 1024 * 1024; // 16 MiB

/// Maximum bytes of a single document we inline into the prompt. Documents
/// are concatenated into every system prompt, so an uncapped novella would
/// silently nuke the model's context window every turn.
pub const MAX_INLINE_TEXT_BYTES: usize = 96 * 1024;

/// One persisted document record. Mirrors the frontend `KbDocument`
/// interface in `src/stores/documents.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Stored filename (also the stable id used in
    /// `settings.embedding.documents_disabled`). URL/path-safe by
    /// construction — see [`unique_dest`].
    pub id: String,
    /// Human-friendly label. Defaults to the original filename the user
    /// picked; equals `id` for hand-dropped files.
    pub name: String,
    /// Size on disk in bytes. Surfaced in the UI.
    pub bytes: u64,
    /// Absolute path on disk under `~/.zero/documents/`.
    pub path: String,
}

/// Enumerate every document present on disk, sorted by name.
pub async fn list() -> Result<Vec<Document>> {
    let dir = paths::documents_dir()?;
    let mut out = Vec::new();
    let mut entries = match tokio::fs::read_dir(&dir).await {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e.into()),
    };
    while let Some(entry) = entries.next_entry().await? {
        let file_type = entry.file_type().await?;
        if !file_type.is_file() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        // Hidden / dotfiles (e.g. `.DS_Store`) are skipped so the list only
        // surfaces things the user actually added.
        if id.starts_with('.') {
            continue;
        }
        let bytes = entry.metadata().await.map(|m| m.len()).unwrap_or(0);
        out.push(Document {
            name: id.clone(),
            bytes,
            path: entry.path().to_string_lossy().to_string(),
            id,
        });
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(out)
}

/// Copy `source` into the documents directory and return the resulting
/// [`Document`] metadata. The destination filename is the (sanitised)
/// original name, with a numeric suffix appended on collision so re-adding a
/// file never clobbers an existing document.
pub async fn add(source: &Path) -> Result<Document> {
    let meta = tokio::fs::metadata(source)
        .await
        .with_context(|| format!("stat {}", source.display()))?;
    if !meta.is_file() {
        return Err(anyhow!("document is not a file: {}", source.display()));
    }
    if meta.len() > MAX_DOCUMENT_BYTES {
        return Err(anyhow!(
            "document is {} bytes; per-file cap is {} bytes",
            meta.len(),
            MAX_DOCUMENT_BYTES
        ));
    }

    let dir = paths::documents_dir()?;
    let original = source
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "document".into());
    let (dest, id) = unique_dest(&dir, &original)?;

    tokio::fs::copy(source, &dest)
        .await
        .with_context(|| format!("copy {} -> {}", source.display(), dest.display()))?;

    Ok(Document {
        name: id.clone(),
        bytes: meta.len(),
        path: dest.to_string_lossy().to_string(),
        id,
    })
}

/// Drop a document file from disk. Idempotent — a missing file is a
/// successful no-op so the UI can call this without checking first.
pub async fn delete(id: &str) -> Result<()> {
    let name = sanitize_name(id);
    if name.is_empty() {
        return Err(anyhow!("invalid document id"));
    }
    let path = paths::documents_dir()?.join(&name);
    if path.exists() {
        tokio::fs::remove_file(&path)
            .await
            .with_context(|| format!("remove {}", path.display()))?;
    }
    Ok(())
}

/// Read a document as UTF-8 text for prompt injection. Returns `Ok(None)`
/// when the bytes don't decode (binary blob, image, PDF, …) so the caller
/// can skip it rather than splice garbage into the prompt. Oversized text is
/// truncated with a `[truncated]` marker.
pub async fn load_text(id: &str) -> Result<Option<String>> {
    let name = sanitize_name(id);
    if name.is_empty() {
        return Ok(None);
    }
    let path = paths::documents_dir()?.join(&name);
    let bytes = tokio::fs::read(&path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    let Ok(text) = String::from_utf8(bytes) else {
        return Ok(None);
    };
    if text.len() > MAX_INLINE_TEXT_BYTES {
        let mut cut = MAX_INLINE_TEXT_BYTES;
        while !text.is_char_boundary(cut) && cut > 0 {
            cut -= 1;
        }
        let mut clipped = text[..cut].to_string();
        clipped.push_str("\n\n[truncated]");
        return Ok(Some(clipped));
    }
    Ok(Some(text))
}

/// Pick a destination path under `dir` for `original`, sanitising the name
/// and appending ` (n)` before the extension on collision. Returns the full
/// destination path plus its final filename (the document id).
fn unique_dest(dir: &Path, original: &str) -> Result<(std::path::PathBuf, String)> {
    let base = sanitize_name(original);
    let base = if base.is_empty() {
        "document".to_string()
    } else {
        base
    };
    let candidate = dir.join(&base);
    if !candidate.exists() {
        return Ok((candidate, base));
    }
    // Split `name.ext` so suffixes land before the extension: `report (2).pdf`.
    let (stem, ext) = match base.rsplit_once('.') {
        Some((s, e)) if !s.is_empty() => (s.to_string(), format!(".{e}")),
        _ => (base.clone(), String::new()),
    };
    for n in 2..10_000 {
        let name = format!("{stem} ({n}){ext}");
        let path = dir.join(&name);
        if !path.exists() {
            return Ok((path, name));
        }
    }
    Err(anyhow!("could not find a free filename for {original}"))
}

/// Reduce a filename to a single safe path component: strip directory
/// separators and anything that could escape the documents dir, collapse
/// control chars, and cap the length.
fn sanitize_name(name: &str) -> String {
    let trimmed = name.trim().trim_matches('.');
    let cleaned: String = trimmed
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '\0' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    let cleaned = cleaned.trim_matches('.').trim();
    if cleaned.is_empty() || cleaned == ".." || cleaned == "." {
        return String::new();
    }
    cleaned.chars().take(180).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_traversal_and_separators() {
        // Traversal collapses to a single safe component — no separators,
        // never a bare `..`, so it can't escape the documents dir.
        let s = sanitize_name("../../etc/passwd");
        assert!(!s.contains('/') && !s.contains('\\'));
        assert_ne!(s, "..");
        assert_eq!(sanitize_name("a/b\\c"), "a_b_c");
        assert_eq!(sanitize_name(".."), "");
        assert_eq!(sanitize_name("  notes.md  "), "notes.md");
    }

    #[test]
    fn unique_dest_keeps_extension_on_collision() {
        let dir = std::env::temp_dir().join(format!("zero-doc-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let (first, id1) = unique_dest(&dir, "report.pdf").unwrap();
        std::fs::write(&first, b"x").unwrap();
        assert_eq!(id1, "report.pdf");
        let (_second, id2) = unique_dest(&dir, "report.pdf").unwrap();
        assert_eq!(id2, "report (2).pdf");
        std::fs::remove_dir_all(&dir).ok();
    }
}
