//! Built-in `recall` tool — full-text search over the agent's own past
//! conversations.
//!
//! Adopts the cross-session recall that makes Nous Research's Hermes Agent
//! feel like it "grows with you": instead of treating every chat as a blank
//! slate, the agent can search everything it has ever discussed and pull a
//! relevant earlier exchange back into context. Where Hermes uses SQLite
//! FTS5, we keep it migration-free and dependency-light by ranking a `LIKE`
//! prefilter over the existing `messages` table in Rust — good enough for a
//! personal chat history and guaranteed to work on any SQLite build.
//!
//! One action, by design: `{"query": "...", "limit": 5}`. The tool returns
//! the best-matching user/assistant turns with their conversation title,
//! date, role, and a short snippet around the match, so the model can decide
//! whether to follow up — it never dumps whole conversations into context.

use crate::mcp::{Tool, ToolResult, ToolSchema};
use crate::state::AppStateExt;
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::{Row, SqlitePool};
use tauri::AppHandle;

/// Canonical tool name. A constant so the schema and any future
/// always-advertise gate stay in sync.
pub const RECALL_TOOL_NAME: &str = "recall";

/// How many candidate rows the SQL prefilter pulls back before Rust-side
/// ranking. Bounds the scan on large histories; the top `limit` survive.
const CANDIDATE_CAP: usize = 400;
/// Default and maximum number of hits returned to the model.
const DEFAULT_LIMIT: usize = 5;
const MAX_LIMIT: usize = 20;
/// Characters of context shown around the first matched term per hit.
const SNIPPET_WINDOW: usize = 240;

#[derive(Debug)]
pub struct Recall {
    db: SqlitePool,
}

impl Recall {
    pub fn new(app: &AppHandle) -> Self {
        Self {
            db: app.zero().db.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for Recall {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: RECALL_TOOL_NAME.into(),
            description: "Search your own past conversations and pull a relevant earlier \
                 exchange back into context. Use this when the user refers to \
                 something from before (\"like we set up last week\", \"that script I \
                 mentioned\"), when continuity across sessions would help, or before \
                 saying you don't know something the user may have already told you. \
                 Pass `query` with the keywords to look for; optionally `limit` (1–20, \
                 default 5). Returns the best-matching user/assistant turns with their \
                 conversation title, date, role, and a snippet — not whole \
                 conversations — so you can decide what to follow up on."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for across all past conversations. Multiple words rank a turn higher when more of them appear."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Max number of matching turns to return. Default 5."
                    }
                }
            }),
            // Read-only history search; never gated by the destructive prompt.
            destructive: false,
        }
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let parsed: Args = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return Ok(err(format!("invalid arguments: {e:#}"))),
        };

        let terms = tokenize(&parsed.query);
        if terms.is_empty() {
            return Ok(err(
                "`query` is empty or has no searchable terms (use 2+ character keywords)",
            ));
        }
        let limit = parsed.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);

        let candidates = match self.fetch_candidates(&terms).await {
            Ok(c) => c,
            Err(e) => return Ok(err(format!("search failed: {e:#}"))),
        };

        let mut hits = rank(candidates, &terms);
        hits.truncate(limit);

        if hits.is_empty() {
            return Ok(ok(format!(
                "[recall: no past conversation mentions {}.]",
                quoted_terms(&terms)
            )));
        }
        Ok(ok(render(&hits, &terms)))
    }
}

impl Recall {
    /// Pull candidate rows: any user/assistant message whose content
    /// matches at least one term. We build an `OR` of `LIKE` clauses and
    /// bind escaped patterns, newest first, capped at [`CANDIDATE_CAP`].
    async fn fetch_candidates(&self, terms: &[String]) -> Result<Vec<Candidate>> {
        let mut sql = String::from(
            "SELECT m.role AS role, m.content AS content, m.created_at AS created_at, \
                    c.title AS title \
             FROM messages m JOIN conversations c ON c.id = m.conversation_id \
             WHERE m.role IN ('user','assistant') AND (",
        );
        for i in 0..terms.len() {
            if i > 0 {
                sql.push_str(" OR ");
            }
            sql.push_str("m.content LIKE ? ESCAPE '\\'");
        }
        sql.push_str(") ORDER BY m.created_at DESC LIMIT ");
        sql.push_str(&CANDIDATE_CAP.to_string());

        let mut q = sqlx::query(&sql);
        for t in terms {
            q = q.bind(format!("%{}%", escape_like(t)));
        }
        let rows = q.fetch_all(&self.db).await?;

        Ok(rows
            .into_iter()
            .map(|r| Candidate {
                role: r.try_get("role").unwrap_or_default(),
                content: r.try_get("content").unwrap_or_default(),
                created_at: r.try_get("created_at").unwrap_or_default(),
                title: r.try_get("title").unwrap_or_default(),
            })
            .collect())
    }
}

pub fn all(app: &AppHandle) -> Vec<Box<dyn Tool>> {
    vec![Box::new(Recall::new(app))]
}

// ─── ranking + rendering ──────────────────────────────────────────────

#[derive(Debug)]
struct Candidate {
    role: String,
    content: String,
    created_at: String,
    title: String,
}

#[derive(Debug)]
struct Hit {
    score: usize,
    role: String,
    created_at: String,
    title: String,
    snippet: String,
}

/// Rank candidates by the number of distinct query terms they contain
/// (a cheap BM25 stand-in), breaking ties by recency (the SQL already
/// returned newest-first, so a stable sort preserves that order).
fn rank(candidates: Vec<Candidate>, terms: &[String]) -> Vec<Hit> {
    let mut hits: Vec<Hit> = candidates
        .into_iter()
        .filter_map(|c| {
            let lower = c.content.to_lowercase();
            let score = terms.iter().filter(|t| lower.contains(*t)).count();
            if score == 0 {
                return None;
            }
            let snippet = snippet_around(&c.content, &lower, terms);
            Some(Hit {
                score,
                role: c.role,
                created_at: c.created_at,
                title: c.title,
                snippet,
            })
        })
        .collect();
    hits.sort_by(|a, b| b.score.cmp(&a.score));
    hits
}

fn render(hits: &[Hit], terms: &[String]) -> String {
    let mut out = format!(
        "[recall: {} match(es) for {}]\n",
        hits.len(),
        quoted_terms(terms)
    );
    for h in hits {
        let date = h.created_at.split('T').next().unwrap_or(&h.created_at);
        out.push_str(&format!(
            "\n• {title} — {role}, {date}\n  {snippet}\n",
            title = if h.title.is_empty() {
                "(untitled)"
            } else {
                &h.title
            },
            role = h.role,
            date = date,
            snippet = h.snippet,
        ));
    }
    out
}

/// Extract a readable window of `content` centred on the first occurrence
/// of any term. Collapses whitespace and adds ellipses when truncated so a
/// hit reads as one tidy line.
fn snippet_around(content: &str, lower: &str, terms: &[String]) -> String {
    let first = terms
        .iter()
        .filter_map(|t| lower.find(t.as_str()))
        .min()
        .unwrap_or(0);

    // Centre the window on the match, clamped to char boundaries.
    let half = SNIPPET_WINDOW / 2;
    let raw_start = first.saturating_sub(half);
    let raw_end = (first + half).min(content.len());
    let start = floor_char_boundary(content, raw_start);
    let end = ceil_char_boundary(content, raw_end);

    let mut snippet = collapse_ws(&content[start..end]);
    if start > 0 {
        snippet.insert_str(0, "…");
    }
    if end < content.len() {
        snippet.push('…');
    }
    snippet
}

/// Split a query into lowercased search terms: 2+ chars, deduplicated,
/// capped so a pathological query can't blow up the SQL `OR` chain.
fn tokenize(query: &str) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for raw in query.split(|c: char| !c.is_alphanumeric()) {
        let t = raw.trim().to_lowercase();
        if t.chars().count() < 2 {
            continue;
        }
        if !seen.iter().any(|s| s == &t) {
            seen.push(t);
        }
        if seen.len() >= 12 {
            break;
        }
    }
    seen
}

fn quoted_terms(terms: &[String]) -> String {
    terms
        .iter()
        .map(|t| format!("`{t}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Escape SQLite `LIKE` metacharacters so a term containing `%` / `_` / `\`
/// is matched literally (paired with `ESCAPE '\'` in the query).
fn escape_like(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    for ch in term.chars() {
        if matches!(ch, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

fn ok(content: String) -> ToolResult {
    ToolResult {
        content,
        is_error: false,
    }
}

fn err(msg: impl Into<String>) -> ToolResult {
    ToolResult {
        content: format!("[recall: {}]", msg.into()),
        is_error: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn schema_requires_query_and_is_read_only() {
        let tool = Recall {
            db: SqlitePool::connect_lazy("sqlite::memory:").unwrap(),
        };
        let s = tool.schema();
        assert_eq!(s.name, "recall");
        assert!(!s.destructive);
        assert_eq!(s.input_schema["required"], json!(["query"]));
    }

    #[test]
    fn tokenize_drops_short_and_dedupes() {
        let t = tokenize("Deploy the the API to Fly.io a");
        assert_eq!(t, vec!["deploy", "the", "api", "to", "fly", "io"]);
    }

    #[test]
    fn tokenize_empty_when_no_real_terms() {
        assert!(tokenize("a ! ?").is_empty());
    }

    #[test]
    fn escape_like_escapes_metacharacters() {
        assert_eq!(escape_like("50%_off"), "50\\%\\_off");
        assert_eq!(escape_like("plain"), "plain");
    }

    #[test]
    fn rank_orders_by_distinct_term_count() {
        let terms = vec!["docker".to_string(), "compose".to_string()];
        let cands = vec![
            Candidate {
                role: "user".into(),
                content: "we used docker here".into(),
                created_at: "2025-01-01T00:00:00Z".into(),
                title: "one".into(),
            },
            Candidate {
                role: "assistant".into(),
                content: "the docker compose file lives in infra/".into(),
                created_at: "2025-01-02T00:00:00Z".into(),
                title: "two".into(),
            },
        ];
        let hits = rank(cands, &terms);
        assert_eq!(hits.len(), 2);
        // The two-term match must rank first regardless of recency order.
        assert_eq!(hits[0].title, "two");
        assert_eq!(hits[0].score, 2);
    }

    #[test]
    fn snippet_is_centred_and_collapsed() {
        let content = format!("{}DOCKER compose lives here{}", "x ".repeat(200), " tail");
        let lower = content.to_lowercase();
        let snip = snippet_around(&content, &lower, &["docker".to_string()]);
        assert!(snip.to_lowercase().contains("docker"));
        assert!(snip.starts_with('…'));
        // Whitespace collapsed: no double spaces.
        assert!(!snip.contains("  "));
    }
}
