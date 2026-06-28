//! UI-interaction built-ins: `ask_user_input` and `present_files`.
//!
//! Both are **placeholder** tools in the same vein as
//! [`crate::mcp::tools::discovery`]: the actual behaviour needs the live
//! chat context (the `AppHandle` plus the conversation / assistant-message
//! ids) so the chat runner intercepts dispatch and does the real work —
//! emitting a UI event, and (for `ask_user_input`) ending the turn so the
//! user can answer. The bodies here only run when a tool is invoked
//! outside a chat turn (the Tools-page "Test" button), where they return a
//! friendly explanation instead of doing nothing.

use crate::mcp::{Tool, ToolResult, ToolSchema};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

/// Tool name for the clarifying-question / elicitation tool. Kept in a
/// constant so the runner's intercept branch and the schema stay in sync.
pub const ASK_USER_INPUT_NAME: &str = "ask_user_input";

/// Tool name for the file-presentation tool.
pub const PRESENT_FILES_NAME: &str = "present_files";

#[derive(Debug, Default)]
pub struct AskUserInput;

#[async_trait]
impl Tool for AskUserInput {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: ASK_USER_INPUT_NAME.into(),
            description: "Ask the user one or more multiple-choice questions and let \
                 them answer by tapping an option — easier and less error-prone than \
                 making them type. Use this when a request is genuinely ambiguous and \
                 the answer isn't already in the conversation: prefer asking over \
                 guessing. Do NOT use it for open-ended questions, for things you can \
                 infer, or when the user already gave the constraint. Calling this \
                 ENDS your turn — stop after the call and wait; the user's choice \
                 arrives as their next message."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["questions"],
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "description": "1-3 questions to ask. Keep it to one where possible.",
                        "items": {
                            "type": "object",
                            "required": ["question", "options"],
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The question text shown to the user."
                                },
                                "options": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 4,
                                    "items": { "type": "string" },
                                    "description": "2-4 short, mutually-exclusive answer labels."
                                },
                                "multi": {
                                    "type": "boolean",
                                    "description": "Allow selecting more than one option. Default false."
                                }
                            }
                        }
                    }
                }
            }),
            destructive: false,
        }
    }

    async fn call(&self, _args: Value) -> Result<ToolResult> {
        Ok(ToolResult {
            content: "ask_user_input only works inside a chat turn — the chat runner \
                      presents the options to the user and pauses the turn until they \
                      choose. There is nothing to test in isolation."
                .into(),
            is_error: false,
        })
    }
}

#[derive(Debug, Default)]
pub struct PresentFiles;

#[async_trait]
impl Tool for PresentFiles {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: PRESENT_FILES_NAME.into(),
            description: "Surface one or more local files to the user as preview cards \
                 they can open or download. Use this right after you create or edit a \
                 file the user asked for (a report, script, image, …) so they have \
                 direct access to the deliverable instead of just a path in text. \
                 Pass the file paths you already wrote with fs.write / shell. This is \
                 a display action only — it does not read, move, or modify anything."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["paths"],
                "properties": {
                    "paths": {
                        "type": "array",
                        "minItems": 1,
                        "items": { "type": "string" },
                        "description": "Absolute or relative file paths to present. `~` expands to home."
                    }
                }
            }),
            destructive: false,
        }
    }

    async fn call(&self, _args: Value) -> Result<ToolResult> {
        Ok(ToolResult {
            content: "present_files only works inside a chat turn — the chat runner \
                      renders preview cards for the listed files in the conversation."
                .into(),
            is_error: false,
        })
    }
}

pub fn all() -> Vec<Box<dyn Tool>> {
    vec![Box::new(AskUserInput), Box::new(PresentFiles)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_advertise_expected_names_and_are_non_destructive() {
        let ask = AskUserInput.schema();
        assert_eq!(ask.name, ASK_USER_INPUT_NAME);
        assert!(!ask.destructive);
        assert!(ask.input_schema["properties"].get("questions").is_some());

        let present = PresentFiles.schema();
        assert_eq!(present.name, PRESENT_FILES_NAME);
        assert!(!present.destructive);
        assert!(present.input_schema["properties"].get("paths").is_some());
    }

    #[tokio::test]
    async fn placeholder_bodies_explain_rather_than_error() {
        let a = AskUserInput.call(json!({})).await.unwrap();
        assert!(!a.is_error);
        assert!(a.content.contains("chat turn"));

        let p = PresentFiles.call(json!({})).await.unwrap();
        assert!(!p.is_error);
        assert!(p.content.contains("chat turn"));
    }
}
