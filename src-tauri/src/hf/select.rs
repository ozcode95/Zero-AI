//! Quant-aware GGUF file selection.
//!
//! A single Hugging Face GGUF repo frequently ships *many* quantizations of
//! the same weights (`Q2_K`, `Q3_K_M`, … `Q8_0`, `F16`) — sometimes hundreds
//! of gigabytes in total — alongside a multimodal projector (`mmproj`) and a
//! speculative-decoding / MTP draft model. Downloading the whole repo is
//! wasteful and, for the larger repos, impractical.
//!
//! This module decides which `.gguf` files are actually worth pulling given a
//! desired quantization (the one `llmfit` recommends for the current machine,
//! surfaced as `bestQuant` in the per-model metadata). The policy is:
//!
//! * **Main weights** — keep exactly one quant. Prefer an exact match for the
//!   desired quant; otherwise fall back to the closest available quant that is
//!   no larger (so it still fits in memory), and only then to the smallest
//!   quant above. All shards of the chosen quant are kept together.
//! * **`mmproj`** — keep one projector (F16 preferred, then F32/BF16, then the
//!   largest). Multimodal models can't run without it, regardless of the main
//!   quant chosen.
//! * **draft / MTP** — keep one draft model, preferring the same quant as the
//!   main weights (again, all shards of that draft quant).
//!
//! Non-GGUF support files (`config.json`, tokenizer, chat templates, …) are
//! handled by the caller's existing `should_include` filter and are not the
//! concern of this module.

use once_cell::sync::Lazy;
use std::collections::BTreeMap;
use std::collections::HashSet;

/// A candidate GGUF file: its repo-relative path (forward slashes, possibly
/// inside a per-quant subdirectory) and its size in bytes (0 when unknown).
#[derive(Debug, Clone)]
pub struct GgufFile {
    pub name: String,
    pub size: u64,
}

/// The outcome of [`select_gguf`].
#[derive(Debug, Clone, Default)]
pub struct Selection {
    /// Every GGUF file (main shards + mmproj + draft shards) we want to keep.
    pub keep: Vec<String>,
    /// The quant token we settled on for the main weights, when one could be
    /// determined (e.g. `"Q4_K_M"`).
    pub target_quant: Option<String>,
    /// The multimodal projector chosen, if the repo has one.
    pub mmproj: Option<String>,
    /// The draft / MTP shards chosen, if the repo has any.
    pub drafts: Vec<String>,
    /// GGUF files deliberately skipped (other quants, extra projectors, …).
    pub skipped: Vec<String>,
}

/// Known quant tokens, longest-first so the most specific token wins
/// (`Q4_K_M` is matched before `Q4_K`, `BF16` before `F16`, …).
static SORTED_TOKENS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    let mut v: Vec<&'static str> = vec![
        // i-quants
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S",
        "IQ3_M", "IQ4_XS", "IQ4_NL", // k-quants with size suffixes
        "Q2_K_XL", "Q2_K_L", "Q2_K_S", "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_K_XL",
        "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q5_K_XL", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q6_K_XL", "Q6_K_L",
        "Q8_K_XL", // base k-quants and legacy quants
        "Q2_K", "Q6_K", "Q8_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", // ternary
        "TQ1_0", "TQ2_0", // float
        "BF16", "F16", "F32",
    ];
    v.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    v
});

/// Approximate bytes-per-weight for a quant token. Used only to rank quants
/// when an exact match for the desired quant isn't available, so coarse values
/// are fine. Monotonic within each family.
pub fn approx_bpp(token: &str) -> f64 {
    match token {
        "F32" => 4.0,
        "F16" | "BF16" => 2.0,
        "Q8_K_XL" => 1.15,
        "Q8_0" | "Q8_K" => 1.05,
        "Q6_K_XL" => 0.88,
        "Q6_K_L" => 0.84,
        "Q6_K" => 0.80,
        "Q5_K_XL" => 0.76,
        "Q5_K_L" => 0.72,
        "Q5_K_M" | "Q5_1" => 0.68,
        "Q5_K_S" | "Q5_0" => 0.65,
        "Q4_K_XL" => 0.65,
        "Q4_K_L" => 0.62,
        "Q4_1" => 0.59,
        "Q4_K_M" | "Q4_0" => 0.58,
        "Q4_K_S" => 0.54,
        "Q3_K_XL" => 0.55,
        "Q3_K_L" => 0.52,
        "Q3_K_M" => 0.48,
        "Q3_K_S" => 0.43,
        "Q2_K_XL" => 0.42,
        "Q2_K_L" => 0.40,
        "Q2_K" => 0.37,
        "Q2_K_S" => 0.35,
        "IQ4_XS" | "IQ4_NL" => 0.50,
        "IQ3_M" => 0.40,
        "IQ3_S" | "IQ3_XS" => 0.36,
        "IQ3_XXS" => 0.34,
        "IQ2_M" => 0.30,
        "IQ2_S" | "IQ2_XS" => 0.28,
        "IQ2_XXS" => 0.26,
        "IQ1_M" => 0.22,
        "IQ1_S" => 0.21,
        "TQ2_0" => 0.26,
        "TQ1_0" => 0.14,
        _ => 0.58,
    }
}

/// Extract the canonical quant token from a GGUF filename (or relative path).
///
/// Tolerant of the various ways quants appear in the wild: `Model-Q4_K_M.gguf`,
/// `model.q4_k_m.gguf`, `Q4_K_M/model-00001-of-00003.gguf`, and unsloth's
/// dynamic `UD-Q4_K_XL` (normalized to `Q4_K_XL`). Returns `None` for files
/// with no recognizable quant in their name.
pub fn extract_quant(name: &str) -> Option<String> {
    // Drop unsloth's "UD-" dynamic-quant prefix so `UD-Q4_K_XL` reduces to the
    // `Q4_K_XL` token we know how to rank.
    let up = name.to_uppercase().replace("UD-", "");
    for tok in SORTED_TOKENS.iter() {
        if token_at_boundary(&up, tok) {
            return Some((*tok).to_string());
        }
    }
    None
}

/// Normalize a user/llmfit-supplied quant string (`"q4_k_m"`, `"UD-Q4_K_M"`, …)
/// to the canonical uppercase token used for matching.
pub fn normalize_quant(q: &str) -> Option<String> {
    let t = q.trim().to_uppercase().replace("UD-", "");
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

/// True when `tok` occurs in `hay` not glued to surrounding alphanumerics, so
/// `Q4_0` doesn't match inside `IQ4_0` and `F16` doesn't match inside `BF16`.
/// Both arguments are expected to be uppercase. `_` counts as a boundary,
/// which is what we want for names like `model_Q4_K_M`.
fn token_at_boundary(hay: &str, tok: &str) -> bool {
    let bytes = hay.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = hay[from..].find(tok) {
        let i = from + rel;
        let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        let after = i + tok.len();
        let after_ok = after >= bytes.len() || !bytes[after].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        from = i + 1;
    }
    false
}

/// Strip a `-NNNNN-of-NNNNN` shard infix and the `.gguf` extension so all
/// shards of one (untokenized) model collapse to a single grouping key.
fn shard_base(name: &str) -> String {
    let lower = name.to_lowercase();
    let no_ext = lower.strip_suffix(".gguf").unwrap_or(&lower);
    // Look for the `-<5 digits>-of-<5 digits>` pattern and cut it off.
    let bytes = no_ext.as_bytes();
    let mut i = 0usize;
    while i + 13 <= bytes.len() {
        if bytes[i] == b'-'
            && bytes[i + 1..i + 6].iter().all(u8::is_ascii_digit)
            && &bytes[i + 6..i + 10] == b"-of-"
            && bytes[i + 10..i + 13].iter().all(u8::is_ascii_digit)
        {
            return no_ext[..i].to_string();
        }
        i += 1;
    }
    no_ext.to_string()
}

/// True when a filename is the *first* shard of a split GGUF
/// (`…-00001-of-00007.gguf`). llama.cpp loads the remaining shards itself when
/// handed the first one.
pub fn is_first_shard(name: &str) -> bool {
    name.to_lowercase().contains("-00001-of-")
}

#[derive(PartialEq)]
enum Kind {
    Main,
    Mmproj,
    Draft,
}

fn classify(name: &str) -> Kind {
    let lower = name.to_lowercase();
    if lower.contains("mmproj") {
        Kind::Mmproj
    } else if lower.contains("mtp") || lower.contains("draft") {
        Kind::Draft
    } else {
        Kind::Main
    }
}

/// Group GGUF files by quant token; files without a recognizable quant are
/// grouped by their shard base instead so an untokenized split model stays
/// together.
fn group_by_quant<'a>(
    files: &[&'a GgufFile],
) -> (
    BTreeMap<String, Vec<&'a GgufFile>>,
    BTreeMap<String, Vec<&'a GgufFile>>,
) {
    let mut tokened: BTreeMap<String, Vec<&GgufFile>> = BTreeMap::new();
    let mut untokened: BTreeMap<String, Vec<&GgufFile>> = BTreeMap::new();
    for f in files {
        match extract_quant(&f.name) {
            Some(q) => tokened.entry(q).or_default().push(*f),
            None => untokened.entry(shard_base(&f.name)).or_default().push(*f),
        }
    }
    (tokened, untokened)
}

/// Pick the quant token to keep from the available tokened groups, given the
/// desired quant. Exact match wins; otherwise the largest available quant that
/// doesn't exceed the desired size; otherwise the smallest available quant.
fn choose_target(tokened: &BTreeMap<String, Vec<&GgufFile>>, desired: &str) -> Option<String> {
    if tokened.is_empty() {
        return None;
    }
    if tokened.contains_key(desired) {
        return Some(desired.to_string());
    }
    let want = approx_bpp(desired);
    let mut ranked: Vec<(String, f64)> =
        tokened.keys().map(|k| (k.clone(), approx_bpp(k))).collect();
    // Deterministic ordering: by bpp, then name.
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    // Largest token that is <= desired (closest from below).
    if let Some((k, _)) = ranked.iter().rev().find(|(_, b)| *b <= want + 1e-9) {
        return Some(k.clone());
    }
    // Everything is bigger than desired — take the smallest.
    ranked.first().map(|(k, _)| k.clone())
}

/// Select the group (all shards) for the desired quant out of a set of files,
/// returning the chosen token plus the chosen files.
fn select_group<'a>(files: &[&'a GgufFile], desired: &str) -> (Option<String>, Vec<&'a GgufFile>) {
    if files.is_empty() {
        return (None, Vec::new());
    }
    let (tokened, untokened) = group_by_quant(files);
    if let Some(token) = choose_target(&tokened, desired) {
        let group = tokened.get(&token).cloned().unwrap_or_default();
        return (Some(token), group);
    }
    // No recognizable quant anywhere — keep the largest (likely the only)
    // untokenized model group.
    if let Some(group) = untokened
        .values()
        .max_by_key(|v| v.iter().map(|f| f.size).sum::<u64>())
    {
        return (None, group.clone());
    }
    (None, Vec::new())
}

/// Choose a single multimodal projector. F16 is the standard/preferred
/// representation; fall back to F32/BF16 and then the largest file.
fn pick_mmproj(files: &[&GgufFile]) -> Option<String> {
    if files.is_empty() {
        return None;
    }
    for pref in ["F16", "F32", "BF16"] {
        if let Some(f) = files
            .iter()
            .find(|f| extract_quant(&f.name).as_deref() == Some(pref))
        {
            return Some(f.name.clone());
        }
    }
    files.iter().max_by_key(|f| f.size).map(|f| f.name.clone())
}

/// Decide which GGUF files to download for a repo, given the desired quant
/// (defaults to `Q4_K_M` — the best all-round balance — when unknown).
///
/// `include_drafts` gates speculative-decoding / MTP draft shards. They're
/// only useful when MTP is enabled in Settings (off by default, since the
/// drafts can crash or fail to load on some llama.cpp build / GPU combos),
/// so when it's `false` we skip them entirely rather than waste bandwidth
/// and disk on a feature that won't be wired in at load time.
pub fn select_gguf(
    files: &[GgufFile],
    desired_quant: Option<&str>,
    include_drafts: bool,
) -> Selection {
    let desired = desired_quant
        .and_then(normalize_quant)
        .unwrap_or_else(|| "Q4_K_M".to_string());

    let mains: Vec<&GgufFile> = files
        .iter()
        .filter(|f| classify(&f.name) == Kind::Main)
        .collect();
    let mmprojs: Vec<&GgufFile> = files
        .iter()
        .filter(|f| classify(&f.name) == Kind::Mmproj)
        .collect();
    let drafts: Vec<&GgufFile> = if include_drafts {
        files
            .iter()
            .filter(|f| classify(&f.name) == Kind::Draft)
            .collect()
    } else {
        Vec::new()
    };

    let (target_quant, chosen_main) = select_group(&mains, &desired);

    // Prefer drafts matching the chosen main quant; otherwise let the draft
    // selection fall back on its own available quants against the same desire.
    let draft_desire = target_quant.clone().unwrap_or_else(|| desired.clone());
    let (_dt, chosen_drafts) = select_group(&drafts, &draft_desire);

    let mmproj = pick_mmproj(&mmprojs);

    let mut keep_set: HashSet<&str> = HashSet::new();
    for f in &chosen_main {
        keep_set.insert(f.name.as_str());
    }
    for f in &chosen_drafts {
        keep_set.insert(f.name.as_str());
    }
    if let Some(ref m) = mmproj {
        keep_set.insert(m.as_str());
    }

    let mut keep = Vec::new();
    let mut skipped = Vec::new();
    for f in files {
        if keep_set.contains(f.name.as_str()) {
            keep.push(f.name.clone());
        } else {
            skipped.push(f.name.clone());
        }
    }
    keep.sort();
    skipped.sort();

    let mut drafts_kept: Vec<String> = chosen_drafts.iter().map(|f| f.name.clone()).collect();
    drafts_kept.sort();

    Selection {
        keep,
        target_quant,
        mmproj,
        drafts: drafts_kept,
        skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(name: &str, size: u64) -> GgufFile {
        GgufFile {
            name: name.to_string(),
            size,
        }
    }

    #[test]
    fn extract_quant_handles_common_shapes() {
        assert_eq!(
            extract_quant("Qwen3-8B-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            extract_quant("model.q4_k_m.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            extract_quant("Q4_K_M/model-00001-of-00003.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            extract_quant("Qwen3-8B-UD-Q4_K_XL.gguf").as_deref(),
            Some("Q4_K_XL")
        );
        assert_eq!(extract_quant("model-BF16.gguf").as_deref(), Some("BF16"));
        assert_eq!(
            extract_quant("model-IQ4_XS.gguf").as_deref(),
            Some("IQ4_XS")
        );
        // No quant in the name.
        assert_eq!(extract_quant("model.gguf"), None);
        assert_eq!(extract_quant("tokenizer.gguf"), None);
    }

    #[test]
    fn f16_not_matched_inside_bf16() {
        assert_eq!(extract_quant("mmproj-BF16.gguf").as_deref(), Some("BF16"));
        assert_eq!(extract_quant("mmproj-F16.gguf").as_deref(), Some("F16"));
    }

    #[test]
    fn shard_base_collapses_splits() {
        assert_eq!(shard_base("model-00001-of-00003.gguf"), "model");
        assert_eq!(shard_base("model-00002-of-00003.gguf"), "model");
        assert_eq!(shard_base("model.gguf"), "model");
    }

    #[test]
    fn selects_only_the_desired_quant() {
        let files = vec![
            f("Model-Q2_K.gguf", 3_000),
            f("Model-Q4_K_M.gguf", 5_000),
            f("Model-Q5_K_M.gguf", 6_000),
            f("Model-Q8_0.gguf", 9_000),
            f("config.json.gguf", 1), // not really a thing, but must not be main-misclassified by size
        ];
        let sel = select_gguf(&files, Some("Q4_K_M"), true);
        assert_eq!(sel.target_quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(sel.keep, vec!["Model-Q4_K_M.gguf".to_string()]);
        assert!(sel.skipped.contains(&"Model-Q8_0.gguf".to_string()));
        assert!(sel.skipped.contains(&"Model-Q2_K.gguf".to_string()));
    }

    #[test]
    fn keeps_all_shards_of_chosen_quant() {
        let files = vec![
            f("Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf", 5_000),
            f("Q4_K_M/Model-Q4_K_M-00002-of-00002.gguf", 5_000),
            f("Q8_0/Model-Q8_0-00001-of-00003.gguf", 9_000),
            f("Q8_0/Model-Q8_0-00002-of-00003.gguf", 9_000),
            f("Q8_0/Model-Q8_0-00003-of-00003.gguf", 9_000),
        ];
        let sel = select_gguf(&files, Some("Q4_K_M"), true);
        assert_eq!(sel.keep.len(), 2);
        assert!(sel.keep.iter().all(|n| n.contains("Q4_K_M")));
    }

    #[test]
    fn keeps_mmproj_and_matching_draft() {
        let files = vec![
            f("Model-Q4_K_M.gguf", 5_000),
            f("Model-Q8_0.gguf", 9_000),
            f("mmproj-F16.gguf", 800),
            f("mmproj-Q8_0.gguf", 400),
            f("Model-MTP-Q4_K_M.gguf", 700),
            f("Model-MTP-Q8_0.gguf", 900),
        ];
        let sel = select_gguf(&files, Some("Q4_K_M"), true);
        assert_eq!(sel.mmproj.as_deref(), Some("mmproj-F16.gguf"));
        assert_eq!(sel.drafts, vec!["Model-MTP-Q4_K_M.gguf".to_string()]);
        assert!(sel.keep.contains(&"mmproj-F16.gguf".to_string()));
        assert!(sel.keep.contains(&"Model-MTP-Q4_K_M.gguf".to_string()));
        // The non-matching mmproj/draft are skipped.
        assert!(sel.skipped.contains(&"mmproj-Q8_0.gguf".to_string()));
        assert!(sel.skipped.contains(&"Model-MTP-Q8_0.gguf".to_string()));
    }

    #[test]
    fn excludes_drafts_when_mtp_disabled() {
        let files = vec![
            f("Model-Q4_K_M.gguf", 5_000),
            f("Model-MTP-Q4_K_M.gguf", 700),
        ];
        let sel = select_gguf(&files, Some("Q4_K_M"), false);
        assert!(sel.drafts.is_empty());
        assert!(!sel.keep.iter().any(|n| n.contains("MTP")));
        assert!(sel.keep.contains(&"Model-Q4_K_M.gguf".to_string()));
        // The draft shard is recorded as deliberately skipped.
        assert!(sel.skipped.contains(&"Model-MTP-Q4_K_M.gguf".to_string()));
    }

    #[test]
    fn falls_back_to_closest_smaller_quant() {
        // Desired Q4_K_M is absent; Q3_K_M (<=) wins over Q5_K_M.
        let files = vec![f("Model-Q5_K_M.gguf", 6_000), f("Model-Q3_K_M.gguf", 4_000)];
        let sel = select_gguf(&files, Some("Q4_K_M"), true);
        assert_eq!(sel.target_quant.as_deref(), Some("Q3_K_M"));
        assert_eq!(sel.keep, vec!["Model-Q3_K_M.gguf".to_string()]);
    }

    #[test]
    fn falls_back_to_smallest_above_when_nothing_smaller() {
        // Desired Q4_K_M absent and everything is bigger -> smallest above.
        let files = vec![f("Model-Q6_K.gguf", 8_000), f("Model-Q8_0.gguf", 9_000)];
        let sel = select_gguf(&files, Some("Q4_K_M"), true);
        assert_eq!(sel.target_quant.as_deref(), Some("Q6_K"));
    }

    #[test]
    fn single_untokenized_model_is_kept() {
        let files = vec![f("model.gguf", 5_000)];
        let sel = select_gguf(&files, Some("Q4_K_M"), true);
        assert_eq!(sel.keep, vec!["model.gguf".to_string()]);
        assert_eq!(sel.target_quant, None);
    }

    #[test]
    fn untokenized_sharded_model_keeps_all_shards() {
        let files = vec![
            f("model-00001-of-00002.gguf", 5_000),
            f("model-00002-of-00002.gguf", 5_000),
        ];
        let sel = select_gguf(&files, Some("Q4_K_M"), true);
        assert_eq!(sel.keep.len(), 2);
    }

    #[test]
    fn defaults_to_q4_k_m_when_no_desire() {
        let files = vec![
            f("Model-Q2_K.gguf", 3_000),
            f("Model-Q4_K_M.gguf", 5_000),
            f("Model-Q8_0.gguf", 9_000),
        ];
        let sel = select_gguf(&files, None, true);
        assert_eq!(sel.target_quant.as_deref(), Some("Q4_K_M"));
    }
}
