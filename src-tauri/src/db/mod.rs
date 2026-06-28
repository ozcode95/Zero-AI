//! SQLite pool + schema.
//!
//! We avoid sqlx's compile-time macros (`query!`, `migrate!`) so the schema
//! can evolve without forcing a `DATABASE_URL` at build time. The schema is
//! applied as a single idempotent baseline on every boot — we don't track
//! per-bump migrations yet because zero is still pre-release and any change
//! lands directly in [`BASELINE`]. Once we ship, additive migrations should
//! be appended to a new `MIGRATIONS` ledger here.

pub mod llama_variant_state;
pub mod runtimes;

use anyhow::{Context, Result};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::path::Path;
use std::str::FromStr;

pub async fn open_pool(file: &Path) -> Result<SqlitePool> {
    let url = format!("sqlite://{}", file.display());
    let opts = SqliteConnectOptions::from_str(&url)?
        .create_if_missing(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
        .busy_timeout(std::time::Duration::from_secs(5));
    let pool = SqlitePoolOptions::new()
        .max_connections(8)
        .connect_with(opts)
        .await?;
    Ok(pool)
}

/// Apply the baseline schema, then additive column migrations for
/// pre-release schema bumps.  `CREATE TABLE IF NOT EXISTS` handles
/// new tables; `ALTER TABLE … ADD COLUMN` (with duplicate-column
/// errors silently ignored) handles new columns on existing tables.
pub async fn migrate(pool: &SqlitePool) -> Result<()> {
    sqlx::query(BASELINE)
        .execute(pool)
        .await
        .context("apply baseline schema")?;

    // Additive column migrations — run each one; SQLite returns
    // "duplicate column name" if the column already exists, which we
    // treat as a success (the schema is already up-to-date).
    for stmt in ADDITIVE_COLUMNS {
        if let Err(e) = sqlx::query(stmt).execute(pool).await {
            let msg = e.to_string();
            if msg.contains("duplicate column name") {
                continue;
            }
            return Err(e).context(format!("additive migration: {stmt}"));
        }
    }

    Ok(())
}

/// Baseline schema. The single source of truth for the on-disk shape.
/// `CREATE TABLE IF NOT EXISTS` keeps existing dbs intact across pre-release
/// schema bumps; once we ship, future additive changes should land as
/// per-version `Migration` entries above instead of being edited in place.
const BASELINE: &str = r#"
CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    model           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    disabled_tools  TEXT,            -- json array of `<server>::<tool>` keys
    sampling        TEXT             -- json SamplingConfig override
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    thinking        TEXT,
    attachments     TEXT,            -- json array
    turn_overrides  TEXT,            -- json TurnOverrides (user rows only)
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS local_models (
    id              TEXT PRIMARY KEY,
    hf_id           TEXT,
    path            TEXT NOT NULL,
    bytes           INTEGER NOT NULL,
    added_at        TEXT NOT NULL,
    revision        TEXT,
    files           INTEGER,
    verified_files  INTEGER,
    pipeline_tag    TEXT,
    metadata_json   TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    action_json     TEXT NOT NULL,              -- tagged TaskAction payload
    trigger_json    TEXT NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    last_run_at     TEXT,
    last_status     TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_facts (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,   -- 'fact' | 'preference' | 'pinned'
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Legacy table used by the removed OVMS runtime. Kept so existing
-- databases don't break on upgrade; no new rows are written.
CREATE TABLE IF NOT EXISTS served_models (
    model_id   TEXT PRIMARY KEY,
    added_at   TEXT NOT NULL
);

-- Per-variant state for each installed llama.cpp build. Each variant
-- (cuda, openvino, hip-radeon, cpu) can serve one model at a time
-- (the `--model <path>` arg), so we track a single loaded model per
-- variant. The variant column is the primary key.
CREATE TABLE IF NOT EXISTS llama_variant_state (
    variant         TEXT PRIMARY KEY,   -- 'cuda' | 'openvino' | 'hip-radeon' | 'cpu'
    loaded_model_id TEXT,
    updated_at      TEXT NOT NULL
);

-- Legacy: the old single-row llama_state table. Kept for migration
-- purposes — the startup init reads any row that exists and migrates
-- it to llama_variant_state.
CREATE TABLE IF NOT EXISTS llama_state (
    id              TEXT PRIMARY KEY,
    loaded_model_id TEXT,
    updated_at      TEXT NOT NULL
);

-- One row per installed runtime. Lets us survive restarts without re-probing
-- the filesystem, and gives a single audit trail for downloads/updates.
CREATE TABLE IF NOT EXISTS runtime_versions (
    name          TEXT PRIMARY KEY,   -- 'ovms' | future: 'ollama' | 'llama.cpp'
    version       TEXT NOT NULL,
    install_dir   TEXT NOT NULL,
    executable    TEXT NOT NULL,
    installed_at  TEXT NOT NULL,
    source_url    TEXT,
    metadata      TEXT                 -- json blob
);
"#;

/// Additive column migrations for tables that may already exist.
/// Each statement is an `ALTER TABLE … ADD COLUMN` that is safe to
/// re-run (duplicate-column errors are silently ignored in [`migrate`]).
const ADDITIVE_COLUMNS: &[&str] = &[
    "ALTER TABLE local_models ADD COLUMN metadata_json TEXT",
    // Generation throughput (tokens/s) for assistant turns, captured from
    // the upstream `timings` block. NULL for legacy rows / providers that
    // don't report it.
    "ALTER TABLE messages ADD COLUMN tokens_per_second REAL",
];
