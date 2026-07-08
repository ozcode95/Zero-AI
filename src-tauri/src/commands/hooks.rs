//! Tauri commands for the hooks feature.
//!
//! Surfaces a CRUD layer over `Settings.hooks` (the global half only —
//! the project `<workspace>/.zero/hooks.json` override is file-managed
//! for v1). The runner re-resolves hooks on every turn so a save takes
//! effect on the next message without a restart.

use crate::error::IpcResult;
use crate::hooks::HooksConfig;
use crate::settings::Settings;

/// Return the global hooks configuration (`Settings.hooks`).
#[tauri::command]
pub async fn hooks_get() -> IpcResult<HooksConfig> {
    let s = Settings::load().await.map_err(|e| e.to_string())?;
    Ok(s.hooks)
}

/// Overwrite the global hooks configuration, persisting to
/// `~/.zero/settings.json`. Project-file overrides
/// (`<workspace>/.zero/hooks.json`) are untouched — they compose with
/// these at runtime.
#[tauri::command]
pub async fn hooks_set(config: HooksConfig) -> IpcResult<()> {
    let mut s = Settings::load().await.map_err(|e| e.to_string())?;
    s.hooks = config;
    s.save().await.map_err(|e| e.to_string().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::HookMatcher;

    #[test]
    fn hooks_set_round_trips_through_settings_serialisation() {
        let cfg = HooksConfig {
            pre_tool_use: vec![HookMatcher {
                matcher: Some("fs\\.write".into()),
                command: "echo hi".into(),
                timeout_secs: 5,
                enabled: true,
            }],
            ..Default::default()
        };
        let mut s = Settings::default();
        s.hooks = cfg;
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hooks.pre_tool_use.len(), 1);
        assert_eq!(back.hooks.pre_tool_use[0].command, "echo hi");
    }

    #[test]
    fn hooks_set_accepts_empty_config() {
        let s = Settings::default();
        assert!(s.hooks.is_empty());
    }
}
