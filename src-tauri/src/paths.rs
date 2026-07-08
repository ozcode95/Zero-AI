//! Storage roots for zero.
//!
//! Everything (DB, settings, model cache, llama.cpp runtime tree, logs, secrets)
//! lives under `~/.zero/`. The location is identical across platforms so
//! the same path shows up in docs, support tickets, and shell snippets.
//!
//! Prior versions used the platform-specific `ProjectDirs` layout
//! (`%LOCALAPPDATA%\zero\zero\data\` on Windows,
//! `~/Library/Application Support/com.zero.app/` on macOS,
//! `~/.local/share/zero/` on Linux). [`migrate_legacy_root_if_needed`] is
//! invoked once at startup so upgrading users keep their data without any
//! manual move.

use crate::llama::variant::LlamaVariant;
use anyhow::{Context, Result};
use directories::{BaseDirs, ProjectDirs};
use std::path::{Path, PathBuf};

/// Returns the root data directory for zero, creating it if missing.
///
/// All platforms: `<home>/.zero/`
pub fn root() -> Result<PathBuf> {
    let p = root_uncreated()?;
    std::fs::create_dir_all(&p).with_context(|| format!("creating {p:?}"))?;
    Ok(p)
}

/// Same as [`root`] but without the `create_dir_all` side effect. Used by
/// the migration helper so it can detect a fresh install vs an upgrade
/// before we've materialised the new directory.
fn root_uncreated() -> Result<PathBuf> {
    let home = BaseDirs::new()
        .context("could not determine user home directory")?
        .home_dir()
        .to_path_buf();
    Ok(home.join(".zero"))
}

/// Legacy storage root used before zero pinned everything under `~/.zero`.
/// Returns `None` only if the platform has no notion of a per-user data
/// dir (extremely unlikely; we just skip the migration in that case).
fn legacy_root() -> Option<PathBuf> {
    ProjectDirs::from("com", "zero", "zero").map(|d| d.data_local_dir().to_path_buf())
}

/// One-shot upgrade hook: if a pre-`.zero` layout exists on disk and the
/// new root hasn't been created yet, move the legacy tree across so the
/// user's models, settings, runtimes, and database survive the path
/// change. Idempotent — after a successful move (or if the user already
/// has data in the new location) this is a no-op.
///
/// Errors are returned so the caller can decide whether to surface them,
/// but [`state::AppState::init`] treats failures as non-fatal: a missing
/// migration just means the user starts with an empty profile, not that
/// the app fails to launch.
pub fn migrate_legacy_root_if_needed() -> Result<()> {
    let new = root_uncreated()?;
    let Some(legacy) = legacy_root() else {
        return Ok(());
    };
    if legacy == new || !legacy.exists() {
        return Ok(());
    }
    if new.exists() {
        // User already has data at the new location — never clobber it,
        // even if the legacy tree also still has files. Leave both in
        // place and let the user decide what to keep.
        tracing::info!(
            "skipping legacy migration: {} already exists; legacy data at {} is untouched",
            new.display(),
            legacy.display()
        );
        return Ok(());
    }

    if let Some(parent) = new.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {parent:?}"))?;
    }
    tracing::info!(
        "migrating zero data from legacy {} → {}",
        legacy.display(),
        new.display()
    );

    // Try a fast rename first. On Windows / same-volume this is atomic
    // and instant. If it fails (typically a cross-volume move on
    // Linux/macOS) we fall back to copy-then-remove.
    match std::fs::rename(&legacy, &new) {
        Ok(()) => {}
        Err(e) => {
            tracing::debug!("rename failed ({e}); falling back to copy-recursive");
            copy_dir_recursive(&legacy, &new)
                .with_context(|| format!("copy {} -> {}", legacy.display(), new.display()))?;
            if let Err(e) = std::fs::remove_dir_all(&legacy) {
                tracing::warn!(
                    "copied legacy data to {} but failed to clean up {}: {e}",
                    new.display(),
                    legacy.display()
                );
            }
        }
    }

    // Best-effort sweep of now-empty legacy parents (e.g.
    // `%LOCALAPPDATA%\zero\zero\` once `data\` is gone). `remove_dir`
    // only succeeds when the directory is empty, so this can't delete
    // anything else the user cares about.
    let mut cursor = legacy.parent().map(Path::to_path_buf);
    for _ in 0..3 {
        let Some(dir) = cursor else { break };
        if std::fs::remove_dir(&dir).is_err() {
            break;
        }
        cursor = dir.parent().map(Path::to_path_buf);
    }

    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let ft = entry.file_type()?;
        if ft.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if ft.is_file() {
            std::fs::copy(&from, &to)?;
        } else {
            // Skip symlinks / other oddities rather than guessing.
            return Err(std::io::Error::other(format!(
                "unsupported file type at {}",
                from.display()
            )));
        }
    }
    Ok(())
}

pub fn db_file() -> Result<PathBuf> {
    Ok(root()?.join("zero.db"))
}

pub fn system_cache() -> Result<PathBuf> {
    Ok(root()?.join("system.json"))
}

pub fn settings_file() -> Result<PathBuf> {
    Ok(root()?.join("settings.json"))
}

pub fn runtimes_dir() -> Result<PathBuf> {
    let p = root()?.join("runtimes");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// llama.cpp runtime install root. All variant installs live under
/// subdirectories here: `runtimes/llama.cpp/cuda/`, `runtimes/llama.cpp/openvino/`,
/// etc.
pub fn llama_dir() -> Result<PathBuf> {
    let p = runtimes_dir()?.join("llama.cpp");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Per-variant install directory. Each variant gets its own subdirectory
/// under `runtimes/llama.cpp/` so multiple builds can coexist on disk.
pub fn llama_variant_dir(variant: LlamaVariant) -> Result<PathBuf> {
    let p = llama_dir()?.join(variant.dir_name());
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// On-disk OpenVINO model/compile cache for the llama.cpp OpenVINO build.
///
/// Pointing `GGML_OPENVINO_CACHE_DIR` at a persistent directory lets
/// OpenVINO reuse its compiled device graphs across runs, so the costly
/// first-token graph compilation is paid once and subsequent model loads
/// are dramatically faster. Shared across all OpenVINO models (the cache is
/// keyed internally by model + device), so a single directory is enough.
pub fn llama_openvino_cache_dir() -> Result<PathBuf> {
    let p = llama_dir()?.join("ov_cache");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Working directory for the spawned `llama-server.exe`. Same root as
/// the variant's install dir so any auxiliary files the binary drops
/// (`.cache/`, crash dumps, ...) stay scoped.
pub fn llama_variant_state_dir(variant: LlamaVariant) -> Result<PathBuf> {
    llama_variant_dir(variant)
}

pub fn models_dir() -> Result<PathBuf> {
    let p = root()?.join("models");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// whisper.cpp runtime install root. The release binaries (`whisper-cli`
/// + backend DLLs) are extracted here. A single install serves the host;
/// re-installing wipes and replaces it.
pub fn whisper_dir() -> Result<PathBuf> {
    let p = runtimes_dir()?.join("whisper.cpp");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Directory holding downloaded whisper ggml model `.bin` files
/// (`models/whisper/ggml-base.en.bin`, ...). These are plain single-file
/// downloads, separate from the GGUF model cache under `models/`.
pub fn whisper_models_dir() -> Result<PathBuf> {
    let p = models_dir()?.join("whisper");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Path to the shared llama.cpp router model-preset INI file.
///
/// Every variant's router reads the same preset (they all serve the same
/// GGUF models, differing only in compute backend), so a single file under
/// `runtimes/llama.cpp/` is sufficient.
pub fn llama_models_preset() -> Result<PathBuf> {
    Ok(llama_dir()?.join("models-preset.ini"))
}

pub fn logs_dir() -> Result<PathBuf> {
    let p = root()?.join("logs");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Per-conversation attachments live under `<root>/attachments/<conv_id>/`.
/// Created lazily by [`crate::attachments::save`]. Kept under the same root
/// as everything else so backups / clean-up are a single `~/.zero/` move.
pub fn attachments_dir() -> Result<PathBuf> {
    let p = root()?.join("attachments");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Global user-level `AGENTS.md` (mirrors Claude Code's `~/.claude/CLAUDE.md`).
/// Injected into the system prompt on every chat turn by `agents_md::load`.
/// Best-effort: a missing file is normal (most users never author one).
pub fn agents_md_global() -> Result<PathBuf> {
    Ok(root()?.join("AGENTS.md"))
}

/// User-authored skills live under `<root>/skills/<skill_id>/SKILL.md`.
/// Mirrors the layout used by Anthropic-style agent skills so packs can be
/// dropped in by hand and picked up on next list.
pub fn skills_dir() -> Result<PathBuf> {
    let p = root()?.join("skills");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Knowledge-base documents live as flat files under `<root>/documents/`.
/// Each file is a single user-provided document copied in from the OS file
/// picker. When the embedding feature is enabled, the text of every enabled
/// document is injected into the system prompt for every chat turn so the
/// assistant can ground its answers in the user's own material.
pub fn documents_dir() -> Result<PathBuf> {
    let p = root()?.join("documents");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Optional project-level hooks config, living at
/// `<workspace_root>/.zero/hooks.json`. Returns the path for a given
/// workspace root; existence is decided by the caller. `None` here when
/// `root` is empty so we never end up materialising a `.zero/hooks.json`
/// at the filesystem root accidentally.
pub fn hooks_project_file(workspace_root: &Path) -> Option<PathBuf> {
    if workspace_root.as_os_str().is_empty() {
        return None;
    }
    Some(workspace_root.join(".zero").join("hooks.json"))
}
