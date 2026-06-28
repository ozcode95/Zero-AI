//! Built-in `skill` tool — the agent's hands-on access to procedural
//! memory (reusable, on-demand instruction packs under
//! `~/.zero/skills/<id>/SKILL.md`).
//!
//! Modelled on Nous Research's Hermes Agent skill system, which closes a
//! "learning loop": the agent doesn't just *follow* skills the user wrote,
//! it **authors new ones from experience**. After working out a non-trivial,
//! repeatable procedure it `save`s a skill so the next time the same task
//! comes up it can `load` the distilled steps instead of re-deriving them.
//!
//! Three actions:
//!
//! | Action | Required arguments        | Effect                                   |
//! | ------ | ------------------------- | ---------------------------------------- |
//! | `load` | `name` (the skill id)     | Return the skill's full `SKILL.md` body. |
//! | `save` | `body` (+ `id`/`name`)    | Create or update a skill, then enable it.|
//! | `list` | —                         | Enumerate every skill on disk.           |
//!
//! `action` is optional and inferred for backwards compatibility with the
//! `# Skills` system-prompt section, which instructs the model to load a
//! skill by calling this tool with just `{"name": "<id>"}`: when `action`
//! is omitted we treat a call carrying `body` as `save`, a call carrying
//! `name`/`id` as `load`, and a bare call as `list`.
//!
//! A freshly `save`d skill is added to `settings.skills_enabled` so it
//! shows up in the `# Skills` catalog (and can be `load`ed) on the very
//! next turn — the agent closes the loop without the user having to flip a
//! toggle in the UI.

use crate::mcp::{Tool, ToolResult, ToolSchema};
use crate::settings::Settings;
use crate::skills;
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

/// Canonical name of the built-in skill tool. Kept in a constant so the
/// chat runner's lazy-mode "always advertise" check, the system-prompt
/// skill-authoring hint, the `# Skills` catalog section (which tells the
/// model to call this tool by name), and the schema below all stay in
/// sync — a typo in any one of them would silently strand the model with
/// a catalog it's told to load from but can't reach.
pub const SKILL_TOOL_NAME: &str = "skill";

#[derive(Debug, Default)]
pub struct SkillInvoke;

#[derive(Debug, Deserialize)]
struct Args {
    #[serde(default)]
    action: Option<String>,
    /// For `load`: the skill id to read. Also accepted as the identifier
    /// for `save` when `id` is omitted (the `# Skills` section refers to
    /// skills by this field).
    #[serde(default)]
    name: Option<String>,
    /// For `save`: explicit skill id (folder name). Falls back to a
    /// slugified `title`/`name` when omitted.
    #[serde(default)]
    id: Option<String>,
    /// For `save`: human-friendly skill title written to the frontmatter
    /// `name:` field. Defaults to the id.
    #[serde(default)]
    title: Option<String>,
    /// For `save`: one-line summary written to frontmatter `description:`.
    /// Shown in the `# Skills` catalog so future turns can decide whether
    /// to load it.
    #[serde(default)]
    description: Option<String>,
    /// For `save`: the skill body (the actual instructions). Markdown.
    #[serde(default)]
    body: Option<String>,
}

#[async_trait]
impl Tool for SkillInvoke {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: SKILL_TOOL_NAME.into(),
            description: "Read and author reusable skills (procedural memory). A skill is a \
                 named pack of instructions stored on disk and surfaced in the \
                 `# Skills` catalog. Three actions: `load` returns a skill's full \
                 body so you can follow it (pass `name` = the skill id from the \
                 catalog); `save` creates or updates a skill from a `body` you \
                 supply (pass `id` and a short `description`) and enables it so it \
                 appears in the catalog next turn; `list` enumerates every skill. \
                 Author a skill with `save` after you work out a non-trivial, \
                 repeatable procedure — capture the durable steps (not this turn's \
                 specifics) so future sessions can `load` it instead of \
                 re-deriving the workflow."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["load", "save", "list"],
                        "description": "What to do. Optional: inferred as `save` when `body` is present, `load` when `name`/`id` is present, else `list`."
                    },
                    "name": {
                        "type": "string",
                        "description": "For `load`: the skill id to read (as shown in the `# Skills` catalog)."
                    },
                    "id": {
                        "type": "string",
                        "description": "For `save`: the skill id (folder name; letters, digits, `-`, `_`). Defaults to a slug of `title`."
                    },
                    "title": {
                        "type": "string",
                        "description": "For `save`: human-friendly skill title. Defaults to the id."
                    },
                    "description": {
                        "type": "string",
                        "description": "For `save`: one-line summary shown in the skills catalog so future turns can decide whether to load it."
                    },
                    "body": {
                        "type": "string",
                        "description": "For `save`: the skill instructions (Markdown). Capture durable, reusable steps — not this turn's one-off specifics."
                    }
                }
            }),
            // Writing a skill is a local, character-bounded file write that
            // only adds optional prompt text the user can inspect/disable on
            // the Skills page — no destructive-confirm gate, so the agent can
            // close its learning loop autonomously (as Hermes does).
            destructive: false,
        }
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let parsed: Args = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => {
                return Ok(err(format!("invalid arguments: {e:#}")));
            }
        };

        match resolve_action(&parsed).as_str() {
            "list" => list_skills().await,
            "load" => {
                let Some(id) = parsed.name.as_deref().or(parsed.id.as_deref()) else {
                    return Ok(err("`name` (the skill id) is required for action `load`"));
                };
                load_skill(id).await
            }
            "save" => save_skill(parsed).await,
            other => Ok(err(format!(
                "unknown action `{other}` (allowed: load, save, list)"
            ))),
        }
    }
}

pub fn all() -> Vec<Box<dyn Tool>> {
    vec![Box::new(SkillInvoke)]
}

// ─── action handlers ──────────────────────────────────────────────────

/// Infer the action when the caller didn't pass one explicitly. Keeps the
/// `# Skills` catalog's `{"name": "<id>"}` load convention working without
/// forcing every load call to spell out `"action": "load"`.
fn resolve_action(a: &Args) -> String {
    if let Some(action) = a.action.as_deref() {
        let action = action.trim().to_ascii_lowercase();
        if !action.is_empty() {
            return action;
        }
    }
    if a.body
        .as_deref()
        .map(|b| !b.trim().is_empty())
        .unwrap_or(false)
    {
        return "save".into();
    }
    if a.name.is_some() || a.id.is_some() {
        return "load".into();
    }
    "list".into()
}

async fn list_skills() -> Result<ToolResult> {
    let skills = match skills::list().await {
        Ok(s) => s,
        Err(e) => return Ok(err(format!("could not list skills: {e:#}"))),
    };
    let enabled = Settings::load()
        .await
        .map(|s| s.skills_enabled)
        .unwrap_or_default();
    if skills.is_empty() {
        return Ok(ok(
            "[skill: no skills saved yet. Use `save` to author one after \
             you work out a reusable procedure.]"
                .into(),
        ));
    }
    let mut out = format!("[skill: {} skill(s) on disk]\n", skills.len());
    for s in &skills {
        let mark = if enabled.iter().any(|e| e == &s.id) {
            "on "
        } else {
            "off"
        };
        match &s.description {
            Some(d) => out.push_str(&format!("- [{mark}] {}: {}\n", s.id, d)),
            None => out.push_str(&format!("- [{mark}] {}\n", s.id)),
        }
    }
    Ok(ok(out))
}

async fn load_skill(id: &str) -> Result<ToolResult> {
    match skills::load_with_body(id).await {
        Ok(s) => {
            let title = &s.meta.name;
            let body = s.body.trim();
            Ok(ok(format!("[skill: loaded `{id}` — {title}]\n\n{body}")))
        }
        Err(e) => Ok(err(format!(
            "could not load skill `{id}`: {e:#}. Call `skill` with \
             {{\"action\":\"list\"}} to see available skill ids."
        ))),
    }
}

async fn save_skill(a: Args) -> Result<ToolResult> {
    let Some(body) = a.body.as_deref().filter(|b| !b.trim().is_empty()) else {
        return Ok(err("`body` is required for action `save`"));
    };

    // Resolve an id: explicit `id` wins; otherwise slug the title/name.
    let title_src = a
        .title
        .as_deref()
        .or(a.name.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let id = match a.id.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(id) => id.to_string(),
        None => match title_src.map(slugify).filter(|s| !s.is_empty()) {
            Some(slug) => slug,
            None => {
                return Ok(err(
                    "provide an `id` (or a `title` to derive one from) for action `save`",
                ));
            }
        },
    };

    let name = title_src.unwrap_or(id.as_str());
    let description = a
        .description
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    // Create or update depending on whether the id already exists. We
    // probe with `list` rather than catching `create`'s "already exists"
    // error so a malformed-but-present skill still routes to `update`.
    let exists = skills::list()
        .await
        .map(|v| v.iter().any(|s| s.id == id))
        .unwrap_or(false);

    let saved = if exists {
        skills::update(&id, name, description, body).await
    } else {
        skills::create(&id, name, description, body).await
    };

    let meta = match saved {
        Ok(m) => m,
        Err(e) => return Ok(err(format!("could not save skill `{id}`: {e:#}"))),
    };

    // Auto-enable so the new skill joins the `# Skills` catalog next turn
    // — this is what closes the learning loop (mirrors Hermes).
    let enabled_note = match enable_skill(&meta.id).await {
        Ok(true) => " and enabled it",
        Ok(false) => " (already enabled)",
        Err(e) => {
            tracing::warn!("skill `{}` saved but enable failed: {e:#}", meta.id);
            " (could not auto-enable — toggle it on the Skills page)"
        }
    };

    let verb = if exists { "updated" } else { "created" };
    Ok(ok(format!(
        "[skill: {verb} `{id}`{enabled_note}. {bytes} bytes. It will appear in the \
         # Skills catalog on the next turn and can be loaded with \
         {{\"action\":\"load\",\"name\":\"{id}\"}}.]",
        bytes = meta.body_bytes,
    )))
}

/// Add a skill id to `settings.skills_enabled`. Returns `Ok(true)` when it
/// was newly enabled, `Ok(false)` when it was already on. Mirrors the
/// logic in [`crate::commands::skills::skills_set_enabled`].
async fn enable_skill(id: &str) -> Result<bool> {
    let mut s = Settings::load().await?;
    if s.skills_enabled.iter().any(|x| x == id) {
        return Ok(false);
    }
    s.skills_enabled.push(id.to_string());
    s.save().await?;
    Ok(true)
}

// ─── helpers ──────────────────────────────────────────────────────────

/// Turn a human title into a URL-safe skill id: lowercase, non-alphanumeric
/// runs collapse to a single `-`, trimmed of leading/trailing `-`, capped at
/// 48 chars. Kept deliberately conservative so the result always satisfies
/// `skills::is_valid_id`.
fn slugify(title: &str) -> String {
    let mut out = String::with_capacity(title.len());
    let mut prev_dash = false;
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-');
    let mut slug: String = trimmed.chars().take(48).collect();
    while slug.ends_with('-') {
        slug.pop();
    }
    slug
}

fn ok(content: String) -> ToolResult {
    ToolResult {
        content,
        is_error: false,
    }
}

fn err(msg: impl Into<String>) -> ToolResult {
    ToolResult {
        content: format!("[skill: {}]", msg.into()),
        is_error: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema_exposes_three_actions() {
        let s = SkillInvoke.schema();
        assert_eq!(s.name, "skill");
        assert!(!s.destructive);
        let actions = &s.input_schema["properties"]["action"]["enum"];
        assert_eq!(actions, &json!(["load", "save", "list"]));
    }

    #[test]
    fn action_is_inferred_from_payload_shape() {
        let load: Args = serde_json::from_value(json!({"name": "x"})).unwrap();
        assert_eq!(resolve_action(&load), "load");

        let save: Args = serde_json::from_value(json!({"id": "x", "body": "do things"})).unwrap();
        assert_eq!(resolve_action(&save), "save");

        let list: Args = serde_json::from_value(json!({})).unwrap();
        assert_eq!(resolve_action(&list), "list");

        // Explicit action wins over inference.
        let explicit: Args =
            serde_json::from_value(json!({"action": "list", "name": "x"})).unwrap();
        assert_eq!(resolve_action(&explicit), "list");
    }

    #[test]
    fn slugify_produces_valid_ids() {
        assert_eq!(slugify("Deploy to Fly.io"), "deploy-to-fly-io");
        assert_eq!(slugify("  Weird___Name!!  "), "weird-name");
        assert_eq!(slugify("CamelCase 123"), "camelcase-123");
        // Result must satisfy the skills module's id validator.
        assert!(skills::is_valid_id(&slugify("Set up Postgres + pgvector")));
    }

    #[tokio::test]
    async fn load_without_name_is_a_structured_error() {
        let out = SkillInvoke
            .call(json!({"action": "load"}))
            .await
            .expect("schema-valid call");
        assert!(out.is_error);
        assert!(out.content.contains("`name`"));
    }

    #[tokio::test]
    async fn save_without_body_is_a_structured_error() {
        let out = SkillInvoke
            .call(json!({"action": "save", "id": "x"}))
            .await
            .expect("schema-valid call");
        assert!(out.is_error);
        assert!(out.content.contains("`body`"));
    }

    #[tokio::test]
    async fn unknown_action_returns_structured_error() {
        let out = SkillInvoke
            .call(json!({"action": "delete", "name": "x"}))
            .await
            .expect("schema-valid call");
        assert!(out.is_error);
        assert!(out.content.contains("unknown action"));
    }
}
