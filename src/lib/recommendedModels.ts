/**
 * Hardware mode the recommendation table is scored against.
 *
 * * `"gpu"` — discrete / primary GPU (VRAM-bound). Matches a CUDA / HIP
 *   llama.cpp build.
 * * `"ram"` — CPU + iGPU + system RAM. Matches a CPU / OpenVINO build.
 */
export type HwMode = "gpu" | "ram";

/**
 * Quantizations the quant filter offers, best quality → most compressed.
 * Mirrors the backend `SUPPORTED_QUANTS` (llmfit's GGUF hierarchy).
 */
export const SUPPORTED_QUANTS = [
  "Q8_0",
  "Q6_K",
  "Q5_K_M",
  "Q4_K_M",
  "Q3_K_M",
  "Q2_K",
] as const;

/** The default quant the table loads with (matches the installer default). */
export const DEFAULT_QUANT = "Q4_K_M";

/**
 * A model recommendation returned by the backend `system_recommend_models`
 * command.  Ranked by hardware fit using llmfit-core's `ModelFit` scoring.
 *
 * Text-generation models come from the llmfit model database; all other
 * categories use a curated static list of OpenVINO IR models.
 */
export interface RecommendedModel {
  /** HuggingFace repo id for download, e.g. `unsloth/Qwen3-8B-GGUF`. */
  hfId: string;
  /** Model family name, e.g. "Qwen3-8B". */
  name: string;
  /** Organization / provider, e.g. "unsloth", "OpenVINO". */
  provider: string;
  /** Parameter count, e.g. "8B", "0.6B". */
  parameterCount: string;
  /** Approximate on-disk or RAM size, e.g. "~8.2 GB". */
  sizeHint: string;
  /** Best quantization that fits in memory, e.g. "Q4_K_M". */
  bestQuant: string;
  /** Context window length in tokens (0 if unknown). */
  contextLength: number;
  /** Capabilities: "vision" and/or "tool_use". */
  capabilities: string[];
  /** Use case / task category, e.g. "coding", "general". */
  useCase: string;
  /** Memory fit level: "perfect" | "good" | "marginal" | "too_tight". */
  fitLevel: string;
  /** Composite fit score (0–100, higher = better). */
  score: number;
  /** Estimated tokens per second on this hardware. */
  estimatedTps: number;
  /** Estimated RAM required in GB. */
  memoryRequiredGb: number;
  /** Execution path: "GPU" | "MoE offload" | "CPU offload" | "CPU only" | "Tensor parallel". */
  runMode: string;
  /** Supported input types: "text", "image", "document", "audio". */
  inputTypes: string[];
  /** Model weight format: "Gguf", "Awq", "Gptq", etc. */
  modelFormat: string;
  /** Inference runtime: "LlamaCpp", "Mlx", "Vllm". */
  inferenceRuntime: string;
}
