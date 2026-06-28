//! Router model-preset (`.ini`) generation for llama-server's **router mode**.
//!
//! Since the migration to llama.cpp, the bundled `llama-server` is launched
//! without a model (router mode). The router only knows about models that
//! were registered through one of its *sources* — we use a `--models-preset`
//! INI file. Each `[section]` is a model id; the keys map to `llama-server`
//! CLI arguments (without the leading dashes).
//!
//! The router routes every request by the `"model"` field, so the section id
//! must equal the local-model id the rest of the app uses (the Hugging Face
//! repo id, e.g. `unsloth/gemma-4-E4B-it-GGUF`).
//!
//! After regenerating this file the running router can be told to re-read it
//! with `GET /models?reload=1` — no restart required.

/// A single model entry destined for the router preset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresetModel {
    /// Router model id — must match the id used in chat requests.
    pub id: String,
    /// Absolute path to the main GGUF weight file (first shard for splits).
    pub model: String,
    /// Absolute path to the multimodal projector, if any.
    pub mmproj: Option<String>,
    /// Absolute path to the speculative-decoding draft model, if any.
    pub draft: Option<String>,
    /// Whether `draft` is an MTP draft (needs `spec-type=draft-mtp`).
    pub draft_is_mtp: bool,
}

/// Normalise a filesystem path for INI use.
///
/// The router accepts forward slashes on every platform, and using them
/// sidesteps any ambiguity around backslash escaping inside the INI parser.
fn ini_path(p: &str) -> String {
    p.replace('\\', "/")
}

/// Render the preset models into the INI text passed to `--models-preset`.
///
/// An empty model list still yields a valid (header-only) file so the router
/// has a stable source to reload from once models are downloaded.
///
/// `mtp_enabled` gates the **experimental** speculative-decoding / MTP draft
/// wiring. When `false` (the default) drafts are downloaded but never wired
/// into the preset, because MTP drafts crash / fail to load on some bundled
/// llama.cpp build + GPU combinations. When `true` the draft keys are emitted
/// so users who opt in get speculative decoding.
pub fn render_preset(models: &[PresetModel], mtp_enabled: bool) -> String {
    let mut out = String::from("version = 1\n");

    for m in models {
        out.push('\n');
        out.push_str(&format!("[{}]\n", m.id));
        out.push_str(&format!("model = {}\n", ini_path(&m.model)));

        if let Some(mmproj) = &m.mmproj {
            out.push_str(&format!("mmproj = {}\n", ini_path(mmproj)));
        }

        // Speculative-decoding drafts are experimental and gated behind the
        // `mtp_enabled` setting. MTP drafts crash / fail to load on some
        // llama.cpp build / GPU combinations, so they stay off unless the
        // user explicitly opts in from Settings → Local LLM.
        if mtp_enabled {
            if let Some(draft) = &m.draft {
                out.push_str(&format!("model-draft = {}\n", ini_path(draft)));
                if m.draft_is_mtp {
                    out.push_str("spec-type = draft-mtp\n");
                    // draft-mtp + flash-attention crashes the CUDA flash-attn
                    // kernel (fattn.cu) on at least some build/GPU combinations.
                    // Disabling FA for these models makes the load reliable while
                    // keeping speculative decoding working. Plain (non-MTP) drafts
                    // and models without a draft keep FA on its default (auto).
                    out.push_str("flash-attn = off\n");
                } else {
                    out.push_str("spec-type = draft-simple\n");
                }
            }
        }

        // We always stage models explicitly via POST /models/load, so the
        // router must not eagerly load everything when it boots.
        out.push_str("load-on-startup = false\n");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str) -> PresetModel {
        PresetModel {
            id: id.to_string(),
            model: format!("C:\\models\\{id}\\weights.gguf"),
            mmproj: None,
            draft: None,
            draft_is_mtp: false,
        }
    }

    #[test]
    fn empty_list_renders_header_only() {
        assert_eq!(render_preset(&[], false), "version = 1\n");
    }

    #[test]
    fn plain_model_has_no_spec_or_mmproj() {
        let out = render_preset(&[model("org/repo")], false);
        assert!(out.contains("[org/repo]"));
        assert!(out.contains("model = C:/models/org/repo/weights.gguf"));
        assert!(out.contains("load-on-startup = false"));
        assert!(!out.contains("mmproj"));
        assert!(!out.contains("spec-type"));
        assert!(!out.contains("flash-attn"));
    }

    #[test]
    fn backslashes_are_normalised_to_forward_slashes() {
        let out = render_preset(&[model("a/b")], false);
        assert!(
            !out.contains('\\'),
            "INI should not contain backslashes: {out}"
        );
    }

    // Draft (speculative-decoding) emission is experimental and gated behind
    // the `mtp_enabled` flag. With it off (the default) we download drafts but
    // emit no draft keys; mmproj (unrelated, and working) still renders. With
    // it on, the draft keys are emitted.

    #[test]
    fn mtp_draft_is_not_emitted_when_disabled_but_mmproj_is() {
        let mut m = model("org/repo");
        m.mmproj = Some("C:\\models\\org\\repo\\mmproj-F16.gguf".into());
        m.draft = Some("C:\\models\\org\\repo\\MTP\\repo-MTP.gguf".into());
        m.draft_is_mtp = true;
        let out = render_preset(&[m], false);
        // mmproj is unaffected by the draft gate and still rendered.
        assert!(out.contains("mmproj = C:/models/org/repo/mmproj-F16.gguf"));
        // Draft keys are suppressed while MTP is disabled.
        assert!(!out.contains("model-draft"));
        assert!(!out.contains("spec-type"));
        assert!(!out.contains("flash-attn"));
    }

    #[test]
    fn plain_draft_is_not_emitted_when_disabled() {
        let mut m = model("org/repo");
        m.draft = Some("C:\\models\\org\\repo\\repo-draft.gguf".into());
        m.draft_is_mtp = false;
        let out = render_preset(&[m], false);
        assert!(!out.contains("model-draft"));
        assert!(!out.contains("spec-type"));
        assert!(!out.contains("flash-attn"));
    }

    #[test]
    fn mtp_draft_is_emitted_when_enabled() {
        let mut m = model("org/repo");
        m.draft = Some("C:\\models\\org\\repo\\MTP\\repo-MTP.gguf".into());
        m.draft_is_mtp = true;
        let out = render_preset(&[m], true);
        assert!(out.contains("model-draft = C:/models/org/repo/MTP/repo-MTP.gguf"));
        assert!(out.contains("spec-type = draft-mtp"));
        assert!(out.contains("flash-attn = off"));
    }

    #[test]
    fn plain_draft_is_emitted_when_enabled() {
        let mut m = model("org/repo");
        m.draft = Some("C:\\models\\org\\repo\\repo-draft.gguf".into());
        m.draft_is_mtp = false;
        let out = render_preset(&[m], true);
        assert!(out.contains("model-draft = C:/models/org/repo/repo-draft.gguf"));
        assert!(out.contains("spec-type = draft-simple"));
        assert!(!out.contains("flash-attn"));
    }

    #[test]
    fn multiple_models_are_separated_by_blank_lines() {
        let out = render_preset(&[model("a/one"), model("b/two")], false);
        assert!(out.contains("[a/one]"));
        assert!(out.contains("[b/two]"));
        // header + two sections, each preceded by a blank line
        assert_eq!(out.matches("load-on-startup = false").count(), 2);
    }
}
