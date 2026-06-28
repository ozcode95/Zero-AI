import type { ModelTaskKind } from "./modelParams";

/**
 * Map a HuggingFace `pipeline_tag` to a `ModelTaskKind`.
 * Returns `null` when the tag doesn't cleanly map to a supported role —
 * callers should default to `text_generation` in those cases.
 */
export function pipelineTagToTaskKind(
  tag: string | null | undefined,
): ModelTaskKind | null {
  if (!tag) return null;
  const t = tag.toLowerCase().trim();

  // ── text generation (chat / completion / VLM) ────────────────
  if (
    [
      "text-generation",
      "image-text-to-text",
      "text2text-generation",
      "summarization",
      "translation",
      "question-answering",
      "visual-question-answering",
      "document-question-answering",
      "conversational",
      "chat",
    ].includes(t)
  ) {
    return "text_generation";
  }

  // ── embeddings ──────────────────────────────────────────────
  if (
    [
      "feature-extraction",
      "sentence-similarity",
      "fill-mask",
      "token-classification",
      "text-embedding",
      "image-feature-extraction",
    ].includes(t)
  ) {
    return "embeddings";
  }

  // ── rerank ──────────────────────────────────────────────────
  if (
    [
      "text-classification",
      "zero-shot-classification",
      "text-ranking",
      "sentence-similarity-reranking",
      "rerank",
    ].includes(t)
  ) {
    return "rerank";
  }

  // ── image generation ────────────────────────────────────────
  if (["text-to-image", "image-to-image", "inpainting"].includes(t)) {
    return "image_generation";
  }

  // ── speech → text (incl. audio-language models like Ultravox) ──
  if (
    [
      "automatic-speech-recognition",
      "audio-to-text",
      "whisper",
      "audio-text-to-text",
      "audio-classification",
      "audio-to-audio",
      "voice-activity-detection",
    ].includes(t)
  ) {
    return "speech2text";
  }

  // ── text → speech ───────────────────────────────────────────
  if (["text-to-speech", "text-to-audio"].includes(t)) {
    return "text2speech";
  }

  return null;
}

/**
 * Best-effort detection of audio / embedding / rerank models from a
 * model's id or repo name. Used as a fallback when the HuggingFace
 * `pipeline_tag` is missing or unhelpful — many GGUF repos (e.g.
 * `ggml-org/WavTokenizer`, `ggml-org/ultravox-…`) carry no tag at all,
 * so the tag-based mapping can't tell they aren't chat-servable.
 *
 * Returns the matched non-chat task kind, or `null` when the name looks
 * like an ordinary text / vision model.
 */
export function nonChatKindFromName(
  name: string | null | undefined,
): ModelTaskKind | null {
  if (!name) return null;
  const n = name.toLowerCase();

  // rerank / cross-encoder
  if (/rerank|cross-encoder/.test(n)) return "rerank";

  // embedding encoders
  if (
    /embed|bge-|gte-|e5-|nomic-embed|minilm|sentence-t5|sentence-transformers/.test(
      n,
    )
  ) {
    return "embeddings";
  }

  // audio: TTS / STT / audio tokenizers / vocoders / audio-language models
  if (
    /wavtokenizer|ultravox|whisper|outetts|parler|musicgen|encodec|vocoder|speecht5|kokoro|xtts|vall-?e|moshi|wav2vec|seamless|bark|snac|audio|\btts\b|\basr\b|\bstt\b|text-to-speech|speech-to-text/.test(
      n,
    )
  ) {
    return "speech2text";
  }

  return null;
}

// ─── Category metadata ──────────────────────────────────────────────

export interface CategoryMeta {
  kind: ModelTaskKind;
  /** Single‑word label for section headers, e.g. "Text Generation". */
  label: string;
  /** One‑liner shown below the header explaining the role. */
  blurb: string;
}

export const CATEGORY_META: Record<ModelTaskKind, CategoryMeta> = {
  text_generation: {
    kind: "text_generation",
    label: "Text Generation",
    blurb: "Powers chat, tool calls, and reasoning",
  },
  embeddings: {
    kind: "embeddings",
    label: "Embeddings",
    blurb: "Semantic search, document retrieval, and clustering",
  },
  rerank: {
    kind: "rerank",
    label: "Reranker",
    blurb: "Re‑ranks search results for relevance and accuracy",
  },
  image_generation: {
    kind: "image_generation",
    label: "Image Generation",
    blurb: "Text‑to‑image and image‑to‑image synthesis",
  },
  text2speech: {
    kind: "text2speech",
    label: "Text → Speech",
    blurb: "Natural‑sounding voice synthesis from text input",
  },
  speech2text: {
    kind: "speech2text",
    label: "Speech → Text",
    blurb: "Transcribes audio and voice into text",
  },
};

/** Display order for category sections (top‑to‑bottom). */
export const CATEGORY_ORDER: ModelTaskKind[] = [
  "text_generation",
  "embeddings",
  "rerank",
  "image_generation",
  "speech2text",
];
