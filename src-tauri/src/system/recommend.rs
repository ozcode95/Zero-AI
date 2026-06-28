//! Hardware-aware model recommendations powered by llmfit-core.
//!
//! Uses `ModelDatabase` (embedded + online-cached model catalog),
//! `ModelFit::analyze_with_forced_runtime()` for per-model scoring against
//! the local machine (forced to LlamaCpp runtime), and `rank_models_by_fit()`
//! for quality-aware ranking. Only GGUF-compatible models are recommended
//! since the app uses llama.cpp exclusively.
//!
//! ## Hardware mode (GPU vs RAM)
//!
//! The UI exposes a `GPU` / `RAM` toggle that decides which memory pool and
//! compute path the models are scored against:
//!
//! * **GPU** — the detected discrete/primary GPU (VRAM-bound). Matches a
//!   CUDA / HIP llama.cpp build streaming weights from VRAM.
//! * **RAM** — CPU + iGPU + system RAM. We drop the discrete GPU from the
//!   probed specs so llmfit scores against the (much larger, but slower)
//!   system-RAM pool with CPU bandwidth. Matches a CPU / OpenVINO build.
//!
//! The same model therefore yields different `score`, tok/s, fit level, and
//! memory footprint depending on the active mode.
//!
//! ## Quant selection
//!
//! llmfit's analyzer picks the *highest-quality* quant that fits in memory
//! (`Q8_0` on a roomy machine), so its `best_quant`, `score`, and tok/s can
//! describe a quant we never actually download. To keep the surfaced numbers
//! honest — and to let the user explore the quality/speed trade-off — every
//! model is **re-scored against a caller-chosen quant** (default `Q4_K_M`,
//! the installer's default). Memory, fit level, tok/s, and the composite
//! score are all recomputed for that quant in [`pin_to_quant`], reusing
//! llmfit's public helpers so the math tracks the crate's own model.
//!
//! Results are cached per `(mode, quant)` under `~/.zero/` (refreshed every
//! 24 h).

use crate::paths;
use llmfit_core::fit::{FitLevel, InferenceRuntime, ModelFit, RunMode, ScoringWeights};
use llmfit_core::hardware::{GpuBackend, SystemSpecs};
use llmfit_core::models::{
    quant_bpp, quant_bytes_per_param, quant_quality_penalty, ModelDatabase, UseCase,
};
use llmfit_core::update::{update_model_cache, UpdateOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

/// The quant every recommendation is scored against by default. Matches the
/// quant the installer downloads by default, so `score`, tok/s, memory, and
/// `best_quant` all describe what the user will actually run.
pub const DEFAULT_QUANT: &str = "Q4_K_M";

/// Quants the UI offers in the quant filter, best quality → most compressed.
/// Mirrors llmfit's GGUF `QUANT_HIERARCHY`.
pub const SUPPORTED_QUANTS: &[&str] = &["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];

/// Validate a caller-supplied quant against [`SUPPORTED_QUANTS`], falling back
/// to [`DEFAULT_QUANT`] for unknown / missing values.
pub fn normalize_quant(quant: Option<&str>) -> String {
    match quant {
        Some(q) if SUPPORTED_QUANTS.iter().any(|s| s.eq_ignore_ascii_case(q)) => {
            // Canonicalize to the upper-case spelling we use internally.
            SUPPORTED_QUANTS
                .iter()
                .find(|s| s.eq_ignore_ascii_case(q))
                .unwrap()
                .to_string()
        }
        _ => DEFAULT_QUANT.to_string(),
    }
}

// ─── hardware mode ──────────────────────────────────────────────────────────

/// Which memory pool / compute path to score models against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwMode {
    /// Discrete / primary GPU (VRAM-bound).
    Gpu,
    /// CPU + iGPU + system RAM.
    Ram,
}

impl HwMode {
    /// Parse the wire string (`"gpu"` / `"ram"`); defaults to `Gpu`.
    pub fn from_opt(s: Option<&str>) -> Self {
        match s.map(|v| v.to_ascii_lowercase()) {
            Some(v) if v == "ram" => HwMode::Ram,
            _ => HwMode::Gpu,
        }
    }

    fn slug(self) -> &'static str {
        match self {
            HwMode::Gpu => "gpu",
            HwMode::Ram => "ram",
        }
    }
}

/// Probe the machine, then shape the specs for the requested mode.
///
/// `Gpu` mode uses the detected specs as-is (llmfit prefers the discrete GPU).
/// `Ram` mode strips the GPU so scoring falls back to the system-RAM pool and
/// CPU bandwidth — modelling a CPU / iGPU llama.cpp build.
fn specs_for_mode(mode: HwMode) -> SystemSpecs {
    let mut specs = SystemSpecs::detect();
    if mode == HwMode::Ram {
        specs.has_gpu = false;
        specs.gpu_vram_gb = None;
        specs.total_gpu_vram_gb = None;
        specs.gpu_name = None;
        specs.gpu_count = 0;
        specs.unified_memory = false;
        specs.gpus = Vec::new();
        specs.backend = if cfg!(target_arch = "aarch64") {
            GpuBackend::CpuArm
        } else {
            GpuBackend::CpuX86
        };
    }
    specs
}

// ─── wire types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRecommendations {
    /// ISO‑8601 timestamp of when the cache was populated.
    pub cached_at: String,
    /// Ranked model list (flat).
    pub models: Vec<RecommendedModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecommendedModel {
    /// HuggingFace repo id for download, e.g. `unsloth/Qwen3-8B-GGUF`.
    pub hf_id: String,
    /// Model family name, e.g. "Qwen3-8B".
    pub name: String,
    /// Organization / provider, e.g. "unsloth", "OpenVINO".
    pub provider: String,
    /// Parameter count string, e.g. "8B", "0.6B".
    pub parameter_count: String,
    /// Approximate on‑disk or RAM size, e.g. "~8.2 GB".
    pub size_hint: String,
    /// Quantization the metrics below were computed for, e.g. "Q4_K_M".
    pub best_quant: String,
    /// Context window length (tokens).
    pub context_length: u32,
    /// Capabilities: "vision", "tool_use".
    pub capabilities: Vec<String>,
    /// Input types the model accepts: "text", "image", "audio", "document".
    #[serde(rename = "inputTypes")]
    pub input_types: Vec<String>,
    /// Use case / task category for UI grouping.
    pub use_case: String,
    // ── enriched from llmfit scoring ──────────────────────────────────
    /// Memory fit level: "perfect", "good", "marginal", or "too_tight".
    pub fit_level: String,
    /// Composite fit score (0‑100, higher = better).
    pub score: f64,
    /// Estimated tokens per second for this model on this hardware.
    pub estimated_tps: f64,
    /// Estimated memory required in GB (VRAM in GPU mode, RAM in RAM mode).
    pub memory_required_gb: f64,
    /// Execution path: "GPU", "MoE offload", "CPU offload", "CPU only".
    pub run_mode: String,
    /// Model format (Debug representation of ModelFormat enum, e.g. "Gguf", "Awq").
    pub model_format: String,
    /// Inference runtime (Debug representation of InferenceRuntime enum, e.g. "LlamaCpp", "Mlx").
    pub inference_runtime: String,
}

// ─── quant pinning ──────────────────────────────────────────────────────────

/// The subset of [`ModelFit`] metrics that change once we force a specific
/// quant, recomputed from llmfit's dynamic-quant result.
struct QuantMetrics {
    quant: String,
    fit_level: FitLevel,
    score: f64,
    estimated_tps: f64,
    memory_required_gb: f64,
}

/// Re-derive the quant-dependent metrics of a `ModelFit` as if the model were
/// run at `quant` instead of the quant llmfit dynamically selected.
///
/// We keep the analyzer's `run_mode`, `use_case`, and memory *pool*
/// (`memory_available_gb`) — those are hardware/architecture facts — and only
/// swap out the pieces that depend on the weight quant:
///
/// * **memory** — adjust the weight portion by the chosen/dynamic bytes-per-
///   parameter delta, leaving KV-cache + overhead untouched (exact for dense
///   models; for MoE we re-quantize the active-expert weights so the value
///   stays anchored to the run-mode pool rather than ballooning to TooTight).
/// * **tok/s** — scale by the bandwidth-bound byte ratio (a token read is
///   dominated by streaming the weights, so halving the bytes ~doubles tok/s).
/// * **quality** — swap llmfit's quant penalty for the chosen quant's.
/// * **fit / speed / score** — recompute from the adjusted inputs with the
///   same formulas and use-case weights llmfit uses internally.
fn pin_to_quant(fit: &ModelFit, quant: &str, mode: HwMode) -> QuantMetrics {
    let model = &fit.model;
    let mem_available = fit.memory_available_gb;

    // Weight-byte delta: only the (active, for MoE) weights re-quantize.
    let weight_params_b = if model.is_moe {
        model
            .active_parameters
            .map(|p| p as f64 / 1_000_000_000.0)
            .unwrap_or_else(|| model.params_b())
    } else {
        model.params_b()
    };
    let mem_required = (fit.memory_required_gb
        + weight_params_b * (quant_bpp(quant) - quant_bpp(&fit.best_quant)))
    .max(0.1);

    // tok/s scales inversely with the bytes streamed per token.
    let dyn_bpp = quant_bytes_per_param(&fit.best_quant);
    let target_bpp = quant_bytes_per_param(quant);
    let estimated_tps = if dyn_bpp > 0.0 && target_bpp > 0.0 {
        fit.estimated_tps * dyn_bpp / target_bpp
    } else {
        fit.estimated_tps
    };

    let fit_level = score_fit_for(
        mem_required,
        mem_available,
        model.recommended_ram_gb,
        fit.run_mode,
        mode,
    );

    // Sub-scores (0-100), mirroring llmfit's private scorers.
    let quality = (fit.score_components.quality - quant_quality_penalty(&fit.best_quant)
        + quant_quality_penalty(quant))
    .clamp(0.0, 100.0);
    let speed = speed_score(estimated_tps, fit.use_case);
    let fit_sub = fit_score(mem_required, mem_available);
    let context = fit.score_components.context;

    let (wq, ws, wf, wc) = ScoringWeights::default().get(fit.use_case);
    let score = ((quality * wq + speed * ws + fit_sub * wf + context * wc) * 10.0).round() / 10.0;

    QuantMetrics {
        quant: quant.to_string(),
        fit_level,
        score,
        estimated_tps,
        memory_required_gb: mem_required,
    }
}

/// Memory headroom → fit level. Mirrors llmfit's private `score_fit`, with a
/// RAM-mode branch that scores on system-RAM headroom (CPU/iGPU inference
/// still "fits" perfectly when there's ample RAM, rather than being demoted to
/// Marginal the way a purely GPU-centric rule would).
fn score_fit_for(
    mem_required: f64,
    mem_available: f64,
    recommended: f64,
    run_mode: RunMode,
    mode: HwMode,
) -> FitLevel {
    if mem_required > mem_available {
        return FitLevel::TooTight;
    }

    if mode == HwMode::Ram {
        return if recommended <= mem_available {
            FitLevel::Perfect
        } else if mem_available >= mem_required * 1.2 {
            FitLevel::Good
        } else {
            FitLevel::Marginal
        };
    }

    match run_mode {
        RunMode::Gpu | RunMode::TensorParallel => {
            if recommended <= mem_available {
                FitLevel::Perfect
            } else if mem_available >= mem_required * 1.2 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        RunMode::MoeOffload | RunMode::CpuOffload => {
            if mem_available >= mem_required * 1.2 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        RunMode::CpuOnly => FitLevel::Marginal,
    }
}

/// Memory utilization sweet-spot score. Mirrors llmfit's private `fit_score`.
fn fit_score(required: f64, available: f64) -> f64 {
    if available <= 0.0 || required > available {
        return 0.0;
    }
    let ratio = required / available;
    if ratio <= 0.5 {
        60.0 + (ratio / 0.5) * 40.0
    } else if ratio <= 0.8 {
        100.0
    } else if ratio <= 0.9 {
        70.0
    } else {
        50.0
    }
}

/// tok/s normalized to a per-use-case target. Mirrors llmfit's `speed_score`.
fn speed_score(tps: f64, use_case: UseCase) -> f64 {
    let target = match use_case {
        UseCase::General | UseCase::Coding | UseCase::Multimodal | UseCase::Chat => 40.0,
        UseCase::Reasoning => 25.0,
        UseCase::Embedding => 200.0,
    };
    ((tps / target) * 100.0).clamp(0.0, 100.0)
}

// ─── ModelFit → RecommendedModel ────────────────────────────────────────────

fn fit_to_rec(fit: &ModelFit, qm: &QuantMetrics) -> RecommendedModel {
    let model = &fit.model;

    // Build the hf_id. Prefer the first GGUF source (the repo to download
    // from). Fallback to "provider/name"; if the name already contains a
    // slash it's already a full repo path — use it directly to avoid
    // double-prefixing (e.g. "unsloth/unsloth/model").
    let hf_id = match model.gguf_sources.first() {
        Some(src) => src.repo.clone(),
        None => {
            if model.name.contains('/') {
                model.name.clone()
            } else if model.provider.is_empty() {
                model.name.clone()
            } else {
                format!("{}/{}", model.provider, model.name)
            }
        }
    };

    // Size hint from the quant memory estimate.
    let mem_gb = qm.memory_required_gb;
    let size_hint = if mem_gb < 1.0 {
        format!("~{} MB", (mem_gb * 1024.0) as u32)
    } else {
        format!("{:.1} GB", mem_gb)
    };

    // Map capabilities.
    let capabilities: Vec<String> = model
        .capabilities
        .iter()
        .map(|c| match c {
            llmfit_core::models::Capability::Vision => "vision".to_string(),
            llmfit_core::models::Capability::ToolUse => "tool_use".to_string(),
        })
        .collect();

    let has_vision = capabilities.iter().any(|c| c == "vision");
    let name_lower = model.name.to_lowercase();

    let mut input_types = vec!["text".to_string()];
    if has_vision {
        input_types.push("image".to_string());
        input_types.push("document".to_string());
    }
    if name_lower.contains("whisper") || name_lower.contains("speech") {
        if !input_types.iter().any(|t| t == "audio") {
            input_types.push("audio".to_string());
        }
    }

    RecommendedModel {
        hf_id,
        name: model.name.clone(),
        provider: model.provider.clone(),
        parameter_count: model.parameter_count.clone(),
        size_hint,
        best_quant: qm.quant.clone(),
        context_length: model.context_length,
        capabilities,
        input_types,
        use_case: fit.use_case.label().to_string(),
        fit_level: fit_level_label(&qm.fit_level).to_string(),
        score: qm.score,
        estimated_tps: qm.estimated_tps,
        memory_required_gb: qm.memory_required_gb,
        run_mode: run_mode_label(&fit.run_mode).to_string(),
        model_format: format!("{:?}", fit.model.format),
        inference_runtime: format!("{:?}", fit.runtime),
    }
}

fn fit_level_label(level: &FitLevel) -> &'static str {
    match level {
        FitLevel::Perfect => "perfect",
        FitLevel::Good => "good",
        FitLevel::Marginal => "marginal",
        FitLevel::TooTight => "too_tight",
    }
}

fn run_mode_label(mode: &RunMode) -> &'static str {
    match mode {
        RunMode::Gpu => "GPU",
        RunMode::MoeOffload => "MoE offload",
        RunMode::CpuOffload => "CPU offload",
        RunMode::CpuOnly => "CPU only",
        RunMode::TensorParallel => "Tensor parallel",
    }
}

// ─── cache ───────────────────────────────────────────────────────────────────

const CACHE_MAX_AGE_HOURS: i64 = 24;

/// Per-`(mode, quant)` recommendation cache file.
fn cache_path(mode: HwMode, quant: &str) -> Option<PathBuf> {
    let root = paths::root().ok()?;
    Some(root.join(format!("recommended_models_{}_{}.json", mode.slug(), quant)))
}

fn load_cache(mode: HwMode, quant: &str) -> Option<Vec<RecommendedModel>> {
    let path = cache_path(mode, quant)?;
    let data = std::fs::read_to_string(&path).ok()?;
    let cached: CachedRecommendations = serde_json::from_str(&data).ok()?;

    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&cached.cached_at) {
        let age = chrono::Utc::now() - ts.to_utc();
        if age.num_hours() < CACHE_MAX_AGE_HOURS {
            tracing::info!(
                "recommend: using cached models ({} mode, {}, {} h old)",
                mode.slug(),
                quant,
                age.num_hours()
            );
            return Some(cached.models);
        }
    }

    tracing::info!(
        "recommend: cache expired ({} {}), will refresh",
        mode.slug(),
        quant
    );
    None
}

fn save_cache(mode: HwMode, quant: &str, models: &Vec<RecommendedModel>) {
    let path = match cache_path(mode, quant) {
        Some(p) => p,
        None => {
            tracing::warn!("recommend: cache path error");
            return;
        }
    };

    let cached = CachedRecommendations {
        cached_at: chrono::Utc::now().to_rfc3339(),
        models: models.clone(),
    };

    match serde_json::to_vec_pretty(&cached) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, &json) {
                tracing::warn!("recommend: cache write failed: {e:#}");
            }
        }
        Err(e) => tracing::warn!("recommend: cache serialization failed: {e:#}"),
    }
}

/// Remove every per-`(mode, quant)` recommendation cache file plus the legacy
/// single-file cache. Used by the explicit refresh path.
fn clear_all_caches() {
    let Ok(root) = paths::root() else { return };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("recommended_models") && name.ends_with(".json") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

// ─── catalogue sync marker ───────────────────────────────────────────────────
//
// `update_model_cache` always hits the network, so we gate it behind a single
// timestamp marker rather than firing it on every cold (mode, quant) cache —
// flipping the toggle or quant must not trigger an HTTP fetch.

fn catalogue_marker() -> Option<PathBuf> {
    Some(paths::root().ok()?.join("catalogue_synced_at.txt"))
}

fn catalogue_needs_sync() -> bool {
    let Some(path) = catalogue_marker() else {
        return true;
    };
    let Ok(s) = std::fs::read_to_string(&path) else {
        return true;
    };
    match chrono::DateTime::parse_from_rfc3339(s.trim()) {
        Ok(ts) => (chrono::Utc::now() - ts.to_utc()).num_hours() >= CACHE_MAX_AGE_HOURS,
        Err(_) => true,
    }
}

fn mark_catalogue_synced() {
    if let Some(path) = catalogue_marker() {
        let _ = std::fs::write(path, chrono::Utc::now().to_rfc3339());
    }
}

// ─── public API ──────────────────────────────────────────────────────────────

/// Maximum number of models to surface in the UI. Set high enough to include
/// the full catalogue; the user can filter client-side.
const MAX_PER_CATEGORY: usize = 5000;

/// Return spec-ranked recommendations for the given hardware mode and quant.
///
/// Results are disk-cached per `(mode, quant)` for 24 h. The online catalogue
/// is refreshed at most once per 24 h (gated by a marker file) so switching
/// mode/quant never triggers a network fetch.
pub fn recommend_all(mode: HwMode, quant: &str) -> Vec<RecommendedModel> {
    // Try cache first.
    if let Some(cached) = load_cache(mode, quant) {
        return cached;
    }

    // Best-effort online catalogue refresh, rate-limited to once per day.
    if catalogue_needs_sync() {
        match update_model_cache(&UpdateOptions::default(), |_| {}) {
            Ok((new, total)) => {
                tracing::info!("recommend: catalogue synced (+{new} new, {total} total)")
            }
            Err(e) => tracing::debug!("recommend: catalogue sync failed (non-fatal): {e}"),
        }
        mark_catalogue_synced();
    }

    let models = recommend_live(mode, quant);
    save_cache(mode, quant, &models);
    models
}

/// Force a fresh recommendation by clearing caches and re-fetching the
/// online model catalogue before scoring.
pub fn recommend_refresh(mode: HwMode, quant: &str) -> Vec<RecommendedModel> {
    clear_all_caches();
    tracing::info!("recommend: cleared all recommendation caches");

    // Force a fresh model catalogue update.
    match update_model_cache(&UpdateOptions::default(), |_| {}) {
        Ok((new, total)) => {
            tracing::info!("recommend: updated catalogue (+{new} new, {total} total)")
        }
        Err(e) => tracing::warn!("recommend: catalogue update failed: {e}"),
    }
    mark_catalogue_synced();

    let models = recommend_live(mode, quant);
    save_cache(mode, quant, &models);
    models
}

/// Compute recommendations from scratch, using llmfit-core.
///
/// Only GGUF-compatible models are included since the app uses llama.cpp as
/// its inference runtime. Non-GGUF formats (AWQ, GPTQ, MLX, Safetensors) are
/// filtered out. Only models with "perfect" or "good" fit level — at the
/// requested `quant` and `mode` — are returned.
fn recommend_live(mode: HwMode, quant: &str) -> Vec<RecommendedModel> {
    // Detect hardware specs (shaped for the requested mode).
    let specs = specs_for_mode(mode);
    tracing::info!(
        "recommend: {} mode @ {} — {} GB RAM, {} CPU cores, GPU: {} ({:.1} GB VRAM)",
        mode.slug(),
        quant,
        specs.total_ram_gb,
        specs.total_cpu_cores,
        specs.gpu_name.as_deref().unwrap_or("<none>"),
        specs.gpu_vram_gb.unwrap_or(0.0),
    );

    // Load model database (embedded + cached).
    let db = ModelDatabase::new();
    let all_models = db.get_all_models();
    tracing::info!(
        "recommend: loaded {} models from database",
        all_models.len()
    );

    // Filter to GGUF-compatible models only. llama.cpp can only run
    // GGUF quantizations, so we require at least one gguf_source repo
    // (the actual downloadable GGUF repository).
    let gguf_models: Vec<_> = all_models
        .iter()
        .filter(|m| !m.gguf_sources.is_empty())
        .collect();
    tracing::info!(
        "recommend: {} GGUF-compatible models (filtered from {})",
        gguf_models.len(),
        all_models.len()
    );

    // Analyze every GGUF model against the current hardware, forcing the
    // LlamaCpp runtime so the fit score reflects llama.cpp performance
    // (not MLX or vLLM).
    let mut fits: Vec<ModelFit> = gguf_models
        .iter()
        .map(|m| {
            ModelFit::analyze_with_forced_runtime(m, &specs, None, Some(InferenceRuntime::LlamaCpp))
        })
        .collect();

    tracing::info!("recommend: analyzed {} model fits", fits.len());

    // Rank: best score first, skipping models that don't fit at all.
    fits = llmfit_core::fit::rank_models_by_fit_opts(fits, true);

    tracing::info!("recommend: ranked {} fits", fits.len());

    // Convert to wire format.  Only include models with "perfect" or
    // "good" fit level — marginal / too_tight won't run acceptably.
    // Deduplicate by hfId so the same repo doesn't appear twice.
    let mut results: Vec<RecommendedModel> = Vec::new();
    let mut seen_hf_ids: HashSet<String> = HashSet::new();
    let mut skipped_runtime = 0usize;
    let mut skipped_fit = 0usize;
    let mut skipped_dup = 0usize;

    for fit in &fits {
        // Only skip models that the analyzer resolved to a non-LlamaCpp
        // runtime (should never happen since we forced it above, but
        // guard regardless).
        if fit.runtime != InferenceRuntime::LlamaCpp {
            skipped_runtime += 1;
            continue;
        }

        // Re-score everything at the requested quant before filtering, so the
        // perfect/good gate reflects the quant we actually download.
        let qm = pin_to_quant(fit, quant, mode);

        // Only include models that fit well on this hardware.
        match qm.fit_level {
            FitLevel::Perfect | FitLevel::Good => {}
            _ => {
                skipped_fit += 1;
                continue;
            }
        }

        let rec = fit_to_rec(fit, &qm);

        // Skip duplicate hfIds (e.g. different quant entries from the same repo).
        if !seen_hf_ids.insert(rec.hf_id.clone()) {
            skipped_dup += 1;
            continue;
        }

        if results.len() < MAX_PER_CATEGORY {
            results.push(rec);
        }
    }

    tracing::info!(
        "recommend: {skip_runtime} non-llamacpp, {skip_fit} poor-fit, {skip_dup} dup skipped → {total} models",
        skip_runtime = skipped_runtime,
        skip_fit = skipped_fit,
        skip_dup = skipped_dup,
        total = results.len(),
    );

    // Sort by estimated TPS descending so the fastest models appear first.
    results.sort_by(|a, b| {
        b.estimated_tps
            .partial_cmp(&a.estimated_tps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Search the model database for GGUF models matching `query` and return
/// the top 5 best-fitting models for the current hardware.
///
/// Scored against the discrete GPU at the default quant — the direct-download
/// search box doesn't expose the mode/quant toggle.
///
/// Only GGUF-compatible models are included (llama.cpp constraint).
pub fn search_models(query: &str) -> Vec<RecommendedModel> {
    let specs = specs_for_mode(HwMode::Gpu);
    let db = ModelDatabase::new();
    let found = db.find_model(query);

    // Filter to GGUF-compatible models only.
    let gguf_found: Vec<_> = found
        .into_iter()
        .filter(|m| !m.gguf_sources.is_empty())
        .collect();

    let mut fits: Vec<ModelFit> = gguf_found
        .iter()
        .map(|m| {
            ModelFit::analyze_with_forced_runtime(m, &specs, None, Some(InferenceRuntime::LlamaCpp))
        })
        .collect();

    fits = llmfit_core::fit::rank_models_by_fit_opts(fits, true);

    let mut results = Vec::new();
    for fit in &fits {
        // Only skip non-LlamaCpp runtime matches — the UI handles
        // fit-level and MLX-only filtering.
        if fit.runtime != InferenceRuntime::LlamaCpp {
            continue;
        }
        let qm = pin_to_quant(fit, DEFAULT_QUANT, HwMode::Gpu);
        results.push(fit_to_rec(fit, &qm));
        if results.len() >= 5 {
            break;
        }
    }

    results
}
