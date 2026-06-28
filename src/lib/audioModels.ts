/**
 * Curated audio model catalogue + hardware-aware recommendation.
 *
 * Audio runs through two local CLIs, not the chat server:
 *
 * * **Speech → text** uses `whisper-cli` (whisper.cpp) with a ggml `.bin`
 *   model downloaded from `ggerganov/whisper.cpp`. These are keyed by file
 *   name (`ggml-base.en.bin`), downloaded via the `whisper_download_model`
 *   command, and stored under `models/whisper/`.
 * * **Text → speech** uses `llama-tts` with an OuteTTS GGUF model plus the
 *   shared WavTokenizer vocoder ([`WAVTOKENIZER`]). These are HuggingFace
 *   repos pulled through the normal `models_download` flow.
 *
 * The recommenders pick the largest tier that comfortably fits the running
 * machine's memory budget (discrete-GPU VRAM when present, else system RAM).
 */

import type { SystemSpecs } from "@/stores/system";

// ─── Speech → text (whisper.cpp ggml models) ─────────────────────────────

export interface WhisperModelOption {
  /** ggml model file name (download id + on-disk name). */
  file: string;
  /** Short display name. */
  name: string;
  /** Human-readable on-disk footprint, e.g. "~142 MB". */
  sizeHint: string;
  /** Approximate resident memory the model needs, in GB. */
  needGb: number;
  /** `true` for the multilingual models; English-only otherwise. */
  multilingual: boolean;
  /** One-line note shown in the picker. */
  note: string;
}

/** Whisper tiers, smallest → largest. */
export const WHISPER_MODELS: readonly WhisperModelOption[] = [
  {
    file: "ggml-base.en.bin",
    name: "Base (English)",
    sizeHint: "~142 MB",
    needGb: 0.5,
    multilingual: false,
    note: "Fast, low memory. English only — a good laptop default.",
  },
  {
    file: "ggml-base.bin",
    name: "Base (multilingual)",
    sizeHint: "~142 MB",
    needGb: 0.5,
    multilingual: true,
    note: "Fast, low memory, 99 languages.",
  },
  {
    file: "ggml-small.bin",
    name: "Small (multilingual)",
    sizeHint: "~466 MB",
    needGb: 1.0,
    multilingual: true,
    note: "Noticeably more accurate than Base. Still light.",
  },
  {
    file: "ggml-large-v3-turbo-q5_0.bin",
    name: "Large v3 Turbo (quantized)",
    sizeHint: "~574 MB",
    needGb: 1.2,
    multilingual: true,
    note: "Near large-model accuracy, fast. Recommended when memory allows.",
  },
  {
    file: "ggml-large-v3-turbo.bin",
    name: "Large v3 Turbo",
    sizeHint: "~1.6 GB",
    needGb: 2.4,
    multilingual: true,
    note: "Highest accuracy. Best on a roomy GPU.",
  },
] as const;

// ─── Text → speech (OuteTTS + WavTokenizer vocoder) ──────────────────────

export interface TtsModelOption {
  /** HuggingFace repo id for download (== local-model id). */
  hfId: string;
  /** Short display name. */
  name: string;
  /** Human-readable on-disk footprint, e.g. "~0.5 GB". */
  sizeHint: string;
  /** Approximate resident memory the model needs, in GB. */
  needGb: number;
  /** One-line note shown in the picker. */
  note: string;
}

/**
 * The shared neural vocoder `llama-tts` needs alongside any OuteTTS model.
 * Auto-downloaded whenever a TTS model is set up — the user never picks it.
 */
export const WAVTOKENIZER = {
  hfId: "ggml-org/WavTokenizer",
  name: "WavTokenizer vocoder",
  sizeHint: "~125 MB",
} as const;

/** OuteTTS tiers, smallest → largest. */
export const TTS_MODELS: readonly TtsModelOption[] = [
  {
    hfId: "OuteAI/OuteTTS-0.2-500M-GGUF",
    name: "OuteTTS 0.2 (500M)",
    sizeHint: "~0.5 GB",
    needGb: 1.2,
    note: "Lightweight read-aloud. Natural enough for most replies.",
  },
] as const;

// ─── Recommendation ──────────────────────────────────────────────────────

/**
 * Memory budget (GB) to score audio models against. Prefer discrete-GPU
 * VRAM when a dGPU is present (that's where the CLIs offload); otherwise
 * fall back to total system RAM. Conservative default before specs probe.
 */
export function audioMemoryBudgetGb(specs: SystemSpecs | null): number {
  if (!specs) return 8;
  const dgpu = specs.gpus.find(
    (g) => g.kind === "discrete" && (g.vram_mb ?? 0) > 0,
  );
  if (dgpu && dgpu.vram_mb) return dgpu.vram_mb / 1024;
  return specs.ram_total_mb / 1024;
}

/**
 * Recommend the best-fitting whisper model: the largest tier whose resident
 * need stays under ~40% of the memory budget (leaving room for the chat
 * model). Falls back to the smallest tier.
 */
export function recommendWhisperModel(
  specs: SystemSpecs | null,
): WhisperModelOption {
  const headroom = audioMemoryBudgetGb(specs) * 0.4;
  let best = WHISPER_MODELS[0];
  for (const m of WHISPER_MODELS) {
    if (m.needGb <= headroom) best = m;
  }
  return best;
}

/** Recommend a text-to-speech model for this machine. */
export function recommendTtsModel(specs: SystemSpecs | null): TtsModelOption {
  const headroom = audioMemoryBudgetGb(specs) * 0.35;
  let best = TTS_MODELS[0];
  for (const m of TTS_MODELS) {
    if (m.needGb <= headroom) best = m;
  }
  return best;
}

/** Look up a whisper tier by file name. */
export function whisperModelByFile(
  file: string | null,
): WhisperModelOption | undefined {
  return file ? WHISPER_MODELS.find((m) => m.file === file) : undefined;
}

/** Look up a TTS tier by HuggingFace id. */
export function ttsModelByHfId(id: string | null): TtsModelOption | undefined {
  return id ? TTS_MODELS.find((m) => m.hfId === id) : undefined;
}

// ─── Audio model detection ─────────────────────────────────────

/**
 * Known audio / speech model families, matched against a model id as a
 * backstop for the pipeline-tag check. GGUF mirror repos (e.g.
 * `ggml-org/ultravox-…`) frequently ship with a missing or non-standard
 * `pipeline_tag`, so audio-language models like Ultravox would otherwise
 * leak into the text-chat model picker. Keep this list focused on
 * audio-specific families so a normal LLM is never misclassified.
 */
const AUDIO_MODEL_ID_PATTERN =
  /(ultravox|whisper|wavtokenizer|outetts|bark|musicgen|xtts|parler[-_]?tts|speech-?t5|seamless|moshi|[-_/]vits[-_/]|csm-|[-_](audio|omni|tts|stt|asr|voice|speech)([-_/]|$))/i;

/**
 * Heuristic: does this model id look like an audio / speech model (TTS,
 * STT, or an audio-language model such as Ultravox)? Used to keep such
 * models out of the text-chat model picker even when their HF
 * `pipeline_tag` is missing or ambiguous.
 */
export function isLikelyAudioModelId(id: string | null | undefined): boolean {
  if (!id) return false;
  return AUDIO_MODEL_ID_PATTERN.test(id);
}

// ─── Transcription language ──────────────────────────────────────────────

/**
 * Spoken-language choices for whisper transcription. `"auto"` lets whisper
 * detect, but it's unreliable on short dictation clips (it often flips
 * English to Japanese), so English is the default. Codes are the ISO
 * language codes whisper.cpp expects via `-l`.
 */
export const STT_LANGUAGES: readonly { code: string; label: string }[] = [
  { code: "en", label: "English" },
  { code: "auto", label: "Auto-detect" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "it", label: "Italian" },
  { code: "pt", label: "Portuguese" },
  { code: "nl", label: "Dutch" },
  { code: "ru", label: "Russian" },
  { code: "zh", label: "Chinese" },
  { code: "ja", label: "Japanese" },
  { code: "ko", label: "Korean" },
  { code: "hi", label: "Hindi" },
  { code: "ar", label: "Arabic" },
] as const;
