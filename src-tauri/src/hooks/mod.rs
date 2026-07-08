//! Lifecycle + per-tool hooks (Claude-Code-style).
//!
//! A *hook* is a user-configured shell command that fires on a fixed set
//! of agent lifecycle events. The runner pipes a JSON description of the
//! event to the command's stdin; the command responds via **exit code**
//! and/or a structured JSON document on stdout to **observe, block, or
//! inject context**:
//!
//! | event             | when it fires               | can block? | can inject context?       |
//! |-------------------|-----------------------------|-----------|----------------------------|
//! | `PreToolUse`      | before a tool dispatches    | yes       | yes (appended to result)   |
//! | `PostToolUse`     | after a tool dispatches     | no*       | yes (replace_result / add) |
//! | `UserPromptSubmit`| start of a chat turn        | yes       | yes (extra context)        |
//! | `Stop`            | end of a chat turn (observe)| no**      | no                         |
//! | `SessionStart`    | first turn of a conversation| no        | yes (seed context)         |
//!
//! (`*` PostToolUse `blocked` becomes feedback to the model rather than
//! a hard stop. `**` Stop is observe-only in v1; the forced-continue
//! semantics from Claude Code are out of scope.)
//!
//! Configuration comes from two sources, merged per-event:
//!
//! 1. **Global** — `Settings.hooks` (lives in `~/.zero/settings.json`).
//! 2. **Project** — `<workspace_root>/.zero/hooks.json` (same shape).
//!
//! Project hooks run **after** global hooks so a repo can tighten the
//! policy its hooks enforce. The project file is only consulted when a
//! workspace is open; an unreadable or unparseable file logs a warning
//! and falls back to global-only — never a fatal load failure.
//!
//! The executor is **failure-isolated**: any spawn error, timeout, or
//! panic in a hook command degrades to "no decision, continue" and a
//! log line. A misconfigured hook can never crash a chat turn.

use crate::paths;
use crate::settings::Settings;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Default per-hook timeout in seconds. Matches Claude Code's default
/// and the `shell.exec` tool's 30 s budget so a hung hook can't freeze
/// the chat any longer than an explicit shell call would.
const DEFAULT_HOOK_TIMEOUT_SECS: u64 = 30;

/// Cap on captured stdout bytes from a hook command. The structured
/// JSON decision is always tiny; this bound just guards against a hook
/// that dumps a log to stdout instead of writing a decision document.
const MAX_STDOUT_BYTES: usize = 64 * 1024;

/// Which lifecycle event a [`HookMatcher`] is wired to. Kept as a plain
/// enum so the config can name a single event per matcher rather than
/// tagging matchers under event-keyed buckets in JSON.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    UserPromptSubmit,
    Stop,
    SessionStart,
}

/// One hook rule: a matcher + the command to run when it fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookMatcher {
    /// Regex matched against the resolved tool name (e.g. `fs.write`,
    /// `shell.exec`). `None` / empty = match every tool. Only meaningful
    /// for `PreToolUse` / `PostToolUse`; ignored on the other events.
    #[serde(default)]
    pub matcher: Option<String>,
    /// Shell command line to execute. Spawned through the OS default
    /// shell (PowerShell on Windows, `sh` elsewhere) — see
    /// [`build_command`].
    pub command: String,
    /// Maximum runtime in seconds before the child is killed. Defaults
    /// to [`DEFAULT_HOOK_TIMEOUT_SECS`].
    #[serde(default = "default_hook_timeout")]
    pub timeout_secs: u64,
    /// Whether the matcher is active. Defaults to `true` so a freshly
    /// authored hook fires immediately without an extra enable step.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_hook_timeout() -> u64 {
    DEFAULT_HOOK_TIMEOUT_SECS
}

fn default_true() -> bool {
    true
}

/// The full hooks configuration, bucketed by event. Stored both as the
/// `Settings.hooks` global knob and — in the same shape — as the
/// optional project override file at `<workspace>/.zero/hooks.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HooksConfig {
    #[serde(default)]
    pub pre_tool_use: Vec<HookMatcher>,
    #[serde(default)]
    pub post_tool_use: Vec<HookMatcher>,
    #[serde(default)]
    pub user_prompt_submit: Vec<HookMatcher>,
    #[serde(default)]
    pub stop: Vec<HookMatcher>,
    #[serde(default)]
    pub session_start: Vec<HookMatcher>,
}

impl HooksConfig {
    /// Returns `true` when every event bucket is empty — the no-op
    /// fast-path check the runner uses to skip hook resolution work on
    /// the (very common) default install that has no hooks at all.
    pub fn is_empty(&self) -> bool {
        self.pre_tool_use.is_empty()
            && self.post_tool_use.is_empty()
            && self.user_prompt_submit.is_empty()
            && self.stop.is_empty()
            && self.session_start.is_empty()
    }

    /// The matchers registered for a given event.
    pub fn for_event(&self, e: HookEvent) -> &[HookMatcher] {
        match e {
            HookEvent::PreToolUse => &self.pre_tool_use,
            HookEvent::PostToolUse => &self.post_tool_use,
            HookEvent::UserPromptSubmit => &self.user_prompt_submit,
            HookEvent::Stop => &self.stop,
            HookEvent::SessionStart => &self.session_start,
        }
    }
}

/// Aggregate result of running one or more hooks for an event. Soft,
/// never panic-inducing: every error path produces an "observe-only"
/// outcome (`blocked == false`, no context, no replacement).
#[derive(Debug, Clone, Default)]
pub struct HookOutcome {
    /// `true` → the event should be blocked. Only meaningful for
    /// `PreToolUse` and `UserPromptSubmit`; on `PostToolUse` and the
    /// observe-only events the runner surfaces it as feedback rather
    /// than a hard stop (see [`HookEvent`] table).
    pub blocked: bool,
    /// Short explanation shown to the model / user when `blocked` or
    /// surfaced as feedback.
    pub reason: Option<String>,
    /// Extra text injected into the conversation/turn. For
    /// `PreToolUse` it's appended to the tool-result note; for
    /// `UserPromptSubmit` / `SessionStart` it's seeded as context for
    /// the model on the firing turn.
    pub additional_context: Option<String>,
    /// `PostToolUse` only — replaces the tool's output text entirely.
    /// The first hook that sets this in a chain wins; later hooks are
    /// still run for their side effects (notifications, logging).
    pub replace_result: Option<String>,
}

/// Resolve the effective hooks config for the active workspace:
/// `Settings.hooks` (global) with each event bucket extended by the
/// optional `<workspace>/.zero/hooks.json` (project). Best-effort — a
/// missing or unreadable project file is logged and skipped.
pub async fn resolve() -> HooksConfig {
    let mut cfg = match Settings::load().await {
        Ok(s) => s.hooks,
        Err(e) => {
            tracing::warn!("hooks: failed to load settings ({e:#}); using empty config");
            HooksConfig::default()
        }
    };

    let Some(root) = crate::workspace::get() else {
        return cfg;
    };
    let Some(path) = paths::hooks_project_file(&root) else {
        return cfg;
    };
    if !path.is_file() {
        return cfg;
    }
    let bytes = match tokio::fs::read_to_string(&path).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("hooks: failed to read project file {}: {e}", path.display());
            return cfg;
        }
    };
    match serde_json::from_str::<HooksConfig>(&bytes) {
        Ok(proj) => {
            append(&mut cfg.pre_tool_use, proj.pre_tool_use);
            append(&mut cfg.post_tool_use, proj.post_tool_use);
            append(&mut cfg.user_prompt_submit, proj.user_prompt_submit);
            append(&mut cfg.stop, proj.stop);
            append(&mut cfg.session_start, proj.session_start);
        }
        Err(e) => {
            tracing::warn!(
                "hooks: project file {} did not parse ({e}); using global hooks only",
                path.display()
            );
        }
    }
    cfg
}

fn append(dst: &mut Vec<HookMatcher>, more: Vec<HookMatcher>) {
    dst.extend(more.into_iter().filter(|h| !h.command.trim().is_empty()));
}

/// Build the JSON payload piped to a hook command's stdin for the given
/// event. Mirrors Claude Code's contract: only the fields that make
/// sense for the event are populated, the rest are `null` so a single
/// parser can handle every event.
pub fn build_input(
    event: HookEvent,
    conversation_id: &str,
    message_id: &str,
    tool_name: Option<&str>,
    tool_input: Option<&Value>,
    tool_result: Option<&str>,
    prompt: Option<&str>,
    workspace_root: Option<&str>,
) -> Value {
    json!({
        "event": event,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_result": tool_result,
        "prompt": prompt,
        "workspace_root": workspace_root,
    })
}

/// Run a single hook command with `input` piped to its stdin. Enforces
/// the hook's timeout. Never panics: spawn failures, timeouts, and
/// non-UTF8 stderr all degrade to an observe-only outcome + a log line.
pub async fn run_hook(h: &HookMatcher, input: &Value) -> HookOutcome {
    if !h.enabled || h.command.trim().is_empty() {
        return HookOutcome::default();
    }

    let workspace_root = input
        .get("workspace_root")
        .and_then(|v| v.as_str())
        .map(std::path::PathBuf::from);

    let mut cmd = build_command(&h.command);
    if let Some(root) = &workspace_root {
        if !root.as_os_str().is_empty() {
            cmd.current_dir(root);
        }
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let input_bytes = serde_json::to_vec(input).unwrap_or_else(|_| b"{}".to_vec());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                "hooks: failed to spawn `{}`{}: {e}",
                h.command,
                h.matcher
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .map(|m| format!(" (matcher: {m})"))
                    .unwrap_or_default(),
            );
            return HookOutcome::default();
        }
    };

    // Pipe the JSON payload to stdin. Ignore write errors: a hook that
    // closes stdin early still gets a chance to decide via exit code.
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(&input_bytes).await;
        // dropping stdin closes the pipe so the child sees EOF.
    }

    let timeout = Duration::from_secs(h.timeout_secs.max(1));
    let run = async {
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        // Read stdout capped and stderr fully (stderr is small + advisory).
        let stdout_fut = read_capped_stdout(stdout);
        let stderr_fut = async {
            let mut buf = Vec::new();
            use tokio::io::AsyncReadExt;
            let mut r = stderr;
            let _ = r.read_to_end(&mut buf).await;
            String::from_utf8_lossy(&buf).into_owned()
        };
        let (out, err, status) = tokio::join!(stdout_fut, stderr_fut, child.wait());
        (out, err, status)
    };

    let (stdout_buf, stderr, status_res) = match tokio::time::timeout(timeout, run).await {
        Ok(v) => v,
        Err(_) => {
            // Best-effort kill. The Child is dropped on return either way.
            let _ = child.start_kill();
            let _ = child.wait().await;
            tracing::warn!(
                "hooks: `{}`{} timed out after {}s — treated as observe-only",
                h.command,
                h.matcher
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .map(|m| format!(" (matcher: {m})"))
                    .unwrap_or_default(),
                h.timeout_secs,
            );
            return HookOutcome::default();
        }
    };

    let status = match status_res {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("hooks: wait failed on `{}`: {e}", h.command);
            return HookOutcome::default();
        }
    };

    parse_outcome(&stdout_buf, &stderr, status.code(), h)
}

/// Read stdout from a hook, capped at [`MAX_STDOUT_BYTES`] so a chatty
/// hook can't blow the JSON parser or wedge the turn. Returns the raw
/// bytes (lossy-utf8 happens later in the parser).
async fn read_capped_stdout(mut r: tokio::process::ChildStdout) -> Vec<u8> {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::with_capacity(8 * 1024);
    let mut chunk = [0u8; 8 * 1024];
    loop {
        match r.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => {
                let take = (MAX_STDOUT_BYTES.saturating_sub(buf.len())).min(n);
                buf.extend_from_slice(&chunk[..take]);
                if buf.len() >= MAX_STDOUT_BYTES {
                    // keep draining so the child doesn't block on a full pipe
                    while r.read(&mut chunk).await.unwrap_or(0) > 0 {}
                    break;
                }
            }
            Err(_) => break,
        }
    }
    buf
}

/// Translate (stdout, stderr, exit_code) into a [`HookOutcome`].
///
/// Decision channels (in priority order):
/// 1. **Structured JSON on stdout** — `decision`, `reason`,
///    `additional_context`, `replace_result` override everything else.
///    Invalid JSON is treated as the stdout just being a log line.
/// 2. **Exit code 2** — block, with `reason` = stderr (or a generic
///    message when stderr is empty).
/// 3. **Exit code 0** (and no JSON) — observe-only allow.
/// 4. **Any other non-zero exit** — non-blocking error: log stderr, no
///    decision.
fn parse_outcome(
    stdout: &[u8],
    stderr: &str,
    exit_code: Option<i32>,
    h: &HookMatcher,
) -> HookOutcome {
    let stdout_str = String::from_utf8_lossy(stdout);
    let trimmed = stdout_str.trim();

    // Channel 1: structured JSON on stdout. Only parse when the trimmed
    // stdout looks like a JSON object — a hook that prints a log line
    // starting with `{` would otherwise be misread, but the explicit
    // `decision` field guard below keeps that risk tiny.
    if trimmed.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            let decision = v.get("decision").and_then(|d| d.as_str());
            let reason = v
                .get("reason")
                .and_then(|r| r.as_str())
                .map(|s| s.to_string());
            let additional_context = v
                .get("additional_context")
                .and_then(|r| r.as_str())
                .map(|s| s.to_string());
            let replace_result = v
                .get("replace_result")
                .and_then(|r| r.as_str())
                .map(|s| s.to_string());
            let blocked = matches!(decision, Some("block"));
            return HookOutcome {
                blocked,
                reason,
                additional_context,
                replace_result,
            };
        }
    }

    // Channel 2/3/4: exit code only.
    match exit_code {
        Some(0) => HookOutcome::default(),
        Some(2) => HookOutcome {
            blocked: true,
            reason: Some(if stderr.trim().is_empty() {
                format!("blocked by hook `{}`", h.command)
            } else {
                stderr.trim().to_string()
            }),
            additional_context: None,
            replace_result: None,
        },
        Some(code) => {
            tracing::warn!(
                "hooks: `{}` exited with non-zero/non-block code {code} (stderr: {stderr})",
                h.command
            );
            HookOutcome::default()
        }
        None => {
            // Killed by signal — treat as observe-only.
            tracing::warn!("hooks: `{}` killed by signal", h.command);
            HookOutcome::default()
        }
    }
}

/// Decide whether a matcher should fire for `tool_name`. `None`/empty
/// `matcher` matches every tool; an invalid regex is downgraded to a
/// literal substring match + a warning so a typo never silently drops
/// the hook. Disabling the matcher short-circuits to `false`.
pub fn matcher_fires(h: &HookMatcher, tool_name: Option<&str>) -> bool {
    if !h.enabled {
        return false;
    }
    let Some(p) = h.matcher.as_deref() else {
        return true;
    };
    let p = p.trim();
    if p.is_empty() {
        return true;
    }
    let Some(name) = tool_name else {
        // Non-tool events (UserPromptSubmit/Stop/SessionStart) always fire
        // — the matcher is meaningless there and the caller passes None.
        return true;
    };
    match Regex::new(p) {
        Ok(re) => re.is_match(name),
        Err(e) => {
            tracing::warn!(
                "hooks: matcher `{p}` is not a valid regex ({e}); treating as literal substring"
            );
            name.contains(p)
        }
    }
}

/// Run every hook whose matcher matches `tool_name`, short-circuiting
/// on the first `blocked`. Aggregate `additional_context` (concatenated,
/// each hook's contribution separated by a blank line), and take the
/// first `replace_result` (later hooks are still run for their side
/// effects but don't override the replacement once set).
pub async fn run_hooks(
    hooks: &[HookMatcher],
    tool_name: Option<&str>,
    input: &Value,
) -> HookOutcome {
    let mut out = HookOutcome::default();
    let mut ctxs: Vec<String> = Vec::new();
    for h in hooks {
        if !matcher_fires(h, tool_name) {
            continue;
        }
        let o = run_hook(h, input).await;
        if let Some(c) = o.additional_context {
            if !c.trim().is_empty() {
                ctxs.push(c);
            }
        }
        if out.replace_result.is_none() {
            if let Some(r) = o.replace_result {
                out.replace_result = Some(r);
            }
        }
        if o.blocked {
            out.blocked = true;
            out.reason = o.reason.or(out.reason);
            break;
        }
    }
    if !ctxs.is_empty() {
        out.additional_context = Some(ctxs.join("\n\n"));
    }
    out
}

/// Build a [`tokio::process::Command`] for the hook's command line,
/// routing through the OS default shell just like the built-in
/// `shell.exec` tool does. PowerShell `-NoProfile -NonInteractive` on
/// Windows (no profile for speed + determinism, no prompt to hang the
/// turn), `sh -c` elsewhere.
fn build_command(command_line: &str) -> Command {
    if cfg!(windows) {
        let mut c = Command::new("powershell");
        c.arg("-NoProfile")
            .arg("-NonInteractive")
            .arg("-Command")
            .arg(command_line);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command_line);
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(command: &str, matcher: Option<&str>) -> HookMatcher {
        HookMatcher {
            matcher: matcher.map(|s| s.to_string()),
            command: command.to_string(),
            timeout_secs: 5,
            enabled: true,
        }
    }

    fn input(tool_name: Option<&str>) -> Value {
        build_input(
            HookEvent::PreToolUse,
            "conv",
            "msg",
            tool_name,
            Some(&json!({})),
            None,
            None,
            None,
        )
    }

    #[test]
    fn hooksconfig_is_empty_on_default() {
        assert!(HooksConfig::default().is_empty());
    }

    #[test]
    fn hooksconfig_for_event_buckets_correctly() {
        let cfg = HooksConfig {
            pre_tool_use: vec![matcher("echo a", Some("fs.write"))],
            ..Default::default()
        };
        assert_eq!(cfg.for_event(HookEvent::PreToolUse).len(), 1);
        assert!(cfg.for_event(HookEvent::Stop).is_empty());
        assert!(!cfg.is_empty());
    }

    #[test]
    fn old_settings_json_without_hooks_key_loads_default() {
        // Simulate a legacy `settings.json`: round-trip a default Settings
        // through JSON, strip the `hooks` key, then deserialise back. The
        // hooks field must default to empty so an upgrade never trips a
        // missing-key error.
        let mut v: serde_json::Value = serde_json::to_value(Settings::default()).unwrap();
        if let Some(obj) = v.as_object_mut() {
            obj.remove("hooks");
        }
        let raw = serde_json::to_string(&v).unwrap();
        let s: Settings = serde_json::from_str(&raw).unwrap();
        assert!(
            s.hooks.is_empty(),
            "missing `hooks` key should default to empty, got: {s:?}"
        );
    }

    #[test]
    fn settings_json_with_hooks_block_round_trips() {
        let mut s = Settings::default();
        s.hooks = HooksConfig {
            pre_tool_use: vec![HookMatcher {
                matcher: Some("fs\\.write".into()),
                command: "echo hi".into(),
                timeout_secs: 5,
                enabled: true,
            }],
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hooks.pre_tool_use.len(), 1);
        assert_eq!(back.hooks.pre_tool_use[0].command, "echo hi");
        assert!(json.contains("fs.write"));
    }

    #[tokio::test]
    async fn exit_zero_is_observe_only_allow() {
        let cmd = if cfg!(windows) { "exit 0" } else { "true" };
        let o = run_hook(&matcher(cmd, None), &input(Some("fs.read"))).await;
        assert!(!o.blocked);
        assert!(o.reason.is_none());
        assert!(o.additional_context.is_none());
    }

    #[tokio::test]
    async fn exit_two_blocks_with_stderr_reason() {
        let cmd = if cfg!(windows) {
            "Write-Error 'nope'; exit 2"
        } else {
            "echo nope 1>&2; exit 2"
        };
        let o = run_hook(&matcher(cmd, None), &input(Some("fs.write"))).await;
        assert!(o.blocked);
        assert!(
            o.reason.as_deref().unwrap_or("").contains("nope"),
            "stderr should surface as reason: {o:?}"
        );
    }

    #[tokio::test]
    async fn stdout_json_decision_block_overrides_exit_code() {
        // A hook that prints a block decision AND exits 0: stdout wins.
        let cmd = if cfg!(windows) {
            r#"Write-Output '{"decision":"block","reason":"from-json"}'; exit 0"#
        } else {
            r#"printf '{"decision":"block","reason":"from-json"}'; exit 0"#
        };
        let o = run_hook(&matcher(cmd, None), &input(Some("fs.write"))).await;
        assert!(o.blocked);
        assert_eq!(o.reason.as_deref(), Some("from-json"));
    }

    #[tokio::test]
    async fn stdout_json_additional_context_is_captured() {
        let cmd = if cfg!(windows) {
            r#"Write-Output '{"additional_context":"seed-me"}'"#
        } else {
            r#"printf '{"additional_context":"seed-me"}'"#
        };
        let o = run_hook(&matcher(cmd, None), &input(None)).await;
        assert!(!o.blocked);
        assert_eq!(o.additional_context.as_deref(), Some("seed-me"));
    }

    #[tokio::test]
    async fn stdout_json_replace_result_is_captured() {
        let cmd = if cfg!(windows) {
            r#"Write-Output '{"replace_result":"rewritten"}'"#
        } else {
            r#"printf '{"replace_result":"rewritten"}'"#
        };
        let o = run_hook(&matcher(cmd, None), &input(Some("fs.read"))).await;
        assert_eq!(o.replace_result.as_deref(), Some("rewritten"));
    }

    #[tokio::test]
    async fn nonblocking_exit_does_not_block() {
        let cmd = if cfg!(windows) { "exit 1" } else { "false" };
        let o = run_hook(&matcher(cmd, None), &input(Some("fs.read"))).await;
        assert!(!o.blocked);
        assert!(o.reason.is_none());
    }

    #[tokio::test]
    async fn timeout_returns_observe_only_without_hang() {
        let cmd = if cfg!(windows) {
            "Start-Sleep -Seconds 30"
        } else {
            "sleep 30"
        };
        let mut h = matcher(cmd, None);
        h.timeout_secs = 1;
        let start = std::time::Instant::now();
        let o = run_hook(&h, &input(Some("fs.read"))).await;
        let elapsed = start.elapsed();
        assert!(!o.blocked);
        assert!(o.replace_result.is_none());
        // Should return well under a few seconds (1s timeout + kill grace).
        assert!(elapsed.as_secs() < 10, "hook hung: {elapsed:?}");
    }

    #[tokio::test]
    async fn matcher_regex_includes_and_excludes() {
        let cmd = if cfg!(windows) { "exit 0" } else { "true" };
        let write_hook = matcher(cmd, Some("^fs\\.(write|edit)$"));
        assert!(matcher_fires(&write_hook, Some("fs.write")));
        assert!(matcher_fires(&write_hook, Some("fs.edit")));
        assert!(!matcher_fires(&write_hook, Some("fs.read")));

        let o_match = run_hooks(
            &[write_hook.clone()],
            Some("fs.write"),
            &input(Some("fs.write")),
        )
        .await;
        assert!(!o_match.blocked);
        let o_no_match = run_hooks(&[write_hook], Some("fs.read"), &input(Some("fs.read"))).await;
        assert!(!o_no_match.blocked);
    }

    #[tokio::test]
    async fn run_hooks_short_circuits_on_first_block() {
        let block_cmd = if cfg!(windows) {
            "Write-Output '{\"decision\":\"block\",\"reason\":\"first\"}'; exit 0"
        } else {
            "printf '{\"decision\":\"block\",\"reason\":\"first\"}'; exit 0"
        };
        // Sentinel hook that would mark additional_context if it ran.
        // Should NOT run because the block short-circuits the chain.
        let never_cmd = if cfg!(windows) {
            r#"Write-Output '{"additional_context":"should-not-run"}'"#
        } else {
            r#"printf '{"additional_context":"should-not-run"}'"#
        };
        let hooks = vec![matcher(block_cmd, None), matcher(never_cmd, None)];
        let o = run_hooks(&hooks, Some("fs.write"), &input(Some("fs.write"))).await;
        assert!(o.blocked);
        assert!(
            o.additional_context
                .as_deref()
                .map(|c| !c.contains("should-not-run"))
                .unwrap_or(true),
            "hooks after a block must not run: {o:?}"
        );
    }

    #[test]
    fn matcher_empty_or_none_matches_everything() {
        assert!(matcher_fires(&matcher("x", None), Some("fs.read")));
        assert!(matcher_fires(&matcher("x", Some("")), Some("fs.write")));
        assert!(matcher_fires(&matcher("x", None), None));
    }

    #[test]
    fn matcher_invalid_regex_falls_back_to_substring() {
        let h = matcher("x", Some("(unclosed"));
        // `(unclosed` is not a valid regex — should NOT panic and should
        // fall back to literal substring matching.
        assert!(matcher_fires(&h, Some("(unclosed-tool-name")));
        assert!(!matcher_fires(&h, Some("fs.read")));
    }

    #[test]
    fn disabled_matcher_never_fires() {
        let mut h = matcher("x", None);
        h.enabled = false;
        assert!(!matcher_fires(&h, Some("fs.read")));
        assert!(!matcher_fires(&h, None));
    }

    #[test]
    fn build_input_shape_matches_contract() {
        let v = build_input(
            HookEvent::PostToolUse,
            "c",
            "m",
            Some("fs.write"),
            Some(&json!({"path": "/x"})),
            Some("ok"),
            Some("hello"),
            Some("/root"),
        );
        assert_eq!(v["event"], "PostToolUse");
        assert_eq!(v["tool_name"], "fs.write");
        assert_eq!(v["tool_result"], "ok");
        assert_eq!(v["prompt"], "hello");
        assert_eq!(v["workspace_root"], "/root");
    }
}
