//! Per-variant persistent state for llama.cpp instances.
//!
//! Each variant (cuda, openvino, hip-radeon, cpu) serves exactly one model
//! per process, so the state we need to remember across restarts is one
//! model id per variant (or nothing, when the variant's instance hasn't
//! loaded anything). The variant slug is the primary key.

use anyhow::Result;
use chrono::Utc;
use sqlx::{Row, SqlitePool};

/// Returns the persisted loaded-model id for a variant, or `None` when
/// nothing was loaded the last time the controller wrote state.
pub async fn get_loaded(pool: &SqlitePool, variant: &str) -> Result<Option<String>> {
    let row = sqlx::query("SELECT loaded_model_id FROM llama_variant_state WHERE variant = ?")
        .bind(variant)
        .fetch_optional(pool)
        .await?;
    Ok(row.and_then(|r| {
        r.try_get::<Option<String>, _>("loaded_model_id")
            .ok()
            .flatten()
    }))
}

/// Replace the persisted loaded-model id for a variant. Pass `None` to
/// clear it (e.g. after an explicit unload from the UI).
pub async fn set_loaded(pool: &SqlitePool, variant: &str, model_id: Option<&str>) -> Result<()> {
    sqlx::query(
        "INSERT INTO llama_variant_state (variant, loaded_model_id, updated_at)
         VALUES (?, ?, ?)
         ON CONFLICT(variant) DO UPDATE SET
            loaded_model_id = excluded.loaded_model_id,
            updated_at      = excluded.updated_at",
    )
    .bind(variant)
    .bind(model_id)
    .bind(Utc::now().to_rfc3339())
    .execute(pool)
    .await?;
    Ok(())
}

/// Migrate data from the legacy `llama_state` singleton table to the new
/// per-variant `llama_variant_state`. Reads the old row's model id and
/// writes it into the variant specified by `legacy_variant` (typically
/// "cuda" for the old default build). Safe to call on every boot — a
/// no-op when the old row is empty or already migrated.
pub async fn migrate_from_legacy(pool: &SqlitePool, legacy_variant: &str) -> Result<()> {
    let row = sqlx::query("SELECT loaded_model_id FROM llama_state WHERE id = 'singleton'")
        .fetch_optional(pool)
        .await?;

    if let Some(r) = row {
        let model_id: Option<String> = r.try_get("loaded_model_id").ok().flatten();
        if let Some(id) = model_id.filter(|s| !s.is_empty()) {
            set_loaded(pool, legacy_variant, Some(&id)).await?;
            // Clear the legacy row so we don't re-migrate on next boot.
            sqlx::query("UPDATE llama_state SET loaded_model_id = NULL WHERE id = 'singleton'")
                .execute(pool)
                .await?;
            tracing::info!(
                "migrated legacy llama_state loaded model to variant '{legacy_variant}': {id}"
            );
        }
    }

    Ok(())
}
