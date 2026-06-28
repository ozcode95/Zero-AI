import { create } from "zustand";
import { invoke, on, Events } from "@/lib/tauri";

export interface HfModelSummary {
  id: string;
  author: string;
  downloads: number;
  likes: number;
  updated_at: string;
  tags: string[];
  pipeline_tag?: string | null;
  has_openvino_ir: boolean;
  /**
   * Whether the repo has at least one `.gguf` sibling. Used by the
   * Models page when the llama.cpp provider is active to filter the
   * HF search to runnable repos.
   */
  has_gguf: boolean;
  total_size_bytes: number | null;
}

export interface LocalModel {
  id: string;
  path: string;
  bytes: number;
  added_at: string;
  hf_id: string | null;
  revision: string | null;
  files: number | null;
  /**
   * Number of files whose sha256 was recorded at download time (i.e.
   * verified against an HF-published LFS digest). `null` when the local
   * manifest pre-dates verification tracking — the next install/update will
   * backfill it. `0` means we have no integrity coverage at all.
   */
  verified_files?: number | null;
  /**
   * Upstream HuggingFace `pipeline_tag` captured at download time —
   * e.g. `text-generation`, `image-text-to-text`, `feature-extraction`.
   * The backend backfills this from the per-model sidecar for rows
   * that pre-date the column (final fallback: `text-generation`), so
   * callers can treat the absence of a value as "truly unknown" rather
   * than "legacy install".
   */
  pipeline_tag?: string | null;
  /**
   * JSON blob with llmfit recommendation metadata (use case, fit level,
   * score, best quant, capabilities, etc.). `null` for models installed
   * before this field existed or installed outside the recommendation flow.
   */
  metadata_json?: string | null;
}

/**
 * A single `.gguf` file in a HuggingFace repo, surfaced by the manual
 * download picker. Mirrors `GgufFileInfo` on the Rust side.
 */
export interface GgufFileInfo {
  /** Repo-relative path (forward slashes), e.g. `model-Q4_K_M.gguf`. */
  name: string;
  /** Size in bytes (LFS-aware); `0` when upstream didn't advertise one. */
  size: number;
  /** Canonical quant token when recognizable (e.g. `Q4_K_M`). */
  quant: string | null;
  /** `main` | `mmproj` | `draft`. */
  kind: string;
}

export interface DownloadProgress {
  model_id: string;
  bytes_done: number;
  bytes_total: number | null;
  files_done: number;
  files_total: number;
  state:
    | "pending"
    | "downloading"
    | "verifying"
    | "done"
    | "cancelled"
    | "error";
  error?: string | null;
}

interface ModelsState {
  query: string;
  results: HfModelSummary[];
  searching: boolean;
  local: LocalModel[];
  downloads: Record<string, DownloadProgress>;
  setQuery: (q: string) => void;
  /**
   * Search HuggingFace. When `providerKind === "llama.cpp"` the raw
   * query is forwarded as-is and results are filtered to repos with a
   * `.gguf` sibling; otherwise the search is scoped to the `OpenVINO`
   * namespace, filtered to repos that publish an OpenVINO IR.
   */
  search: (providerKind?: string) => Promise<void>;
  refreshLocal: () => Promise<void>;
  /**
   * Download a model from HuggingFace.
   * @param id — HuggingFace repo id (e.g. `unsloth/Qwen3-8B-GGUF`).
   * @param metadata — optional llmfit recommendation metadata to store
   *   alongside the model (use case, fit level, best quant, etc.).
   */
  download: (id: string, metadata?: Record<string, unknown>) => Promise<void>;
  /**
   * List the `.gguf` files in a HuggingFace repo for the manual download
   * picker. Throws on a network error or a repo with no GGUF files so the
   * caller can surface the message inline.
   */
  listGgufFiles: (id: string) => Promise<GgufFileInfo[]>;
  /**
   * Manual download: pull exactly the GGUF files the user hand-picked
   * (plus the repo's support files), bypassing the automatic quant picker.
   */
  downloadFiles: (id: string, files: string[]) => Promise<void>;
  cancel: (id: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  update: (id: string) => Promise<void>;
  /**
   * Forget a download entry from local state. Used by the Models card
   * grid to evict failed / cancelled downloads on sight so the user
   * isn't left staring at a dead card — the actual on-disk artifacts
   * (if any) are still cleaned up by the backend.
   */
  dismissDownload: (id: string) => void;
  bindEvents: () => Promise<() => void>;
}

export const useModelsStore = create<ModelsState>((set, get) => ({
  query: "",
  results: [],
  searching: false,
  local: [],
  downloads: {},

  setQuery: (q) => set({ query: q }),

  search: async (providerKind) => {
    set({ searching: true });
    try {
      const raw = get().query.trim();
      const isLlama = providerKind === "llama.cpp";
      // OVMS path: the runner can only load OpenVINO IRs, so we
      // always scope the HF query to the `OpenVINO` namespace and
      // filter on `has_openvino_ir`. The empty-query case becomes a
      // plain `OpenVINO` search so the popover stays useful on first
      // focus.
      //
      // llama.cpp path: GGUFs live under many community orgs
      // (`bartowski/*`, `TheBloke/*`, `ggml-org/*`, ...), so we
      // forward the raw query as-is and filter on `has_gguf` instead.
      // An empty query falls back to `gguf` so the user still sees
      // popular GGUF repos at a glance.
      const query = isLlama
        ? raw || "gguf"
        : raw
          ? `OpenVINO ${raw}`
          : "OpenVINO";
      const all =
        (await invoke<HfModelSummary[]>("models_search", {
          query,
        })) ?? [];
      const results = isLlama
        ? all.filter((m) => m.has_gguf)
        : all.filter((m) => m.has_openvino_ir);
      set({ results, searching: false });
    } catch (e) {
      console.error("models_search failed", e);
      set({ searching: false });
    }
  },

  refreshLocal: async () => {
    try {
      const local = (await invoke<LocalModel[]>("models_list_local")) ?? [];
      set({ local });
    } catch (e) {
      console.error("models_list_local failed", e);
    }
  },

  download: async (id, metadata) => {
    try {
      await invoke("models_download", {
        modelId: id,
        metadataJson: metadata ? JSON.stringify(metadata) : null,
      });
    } catch (e) {
      console.error("models_download failed", e);
    }
  },

  listGgufFiles: async (id) => {
    return (
      (await invoke<GgufFileInfo[]>("models_list_gguf_files", {
        modelId: id,
      })) ?? []
    );
  },

  downloadFiles: async (id, files) => {
    try {
      await invoke("models_download_files", { modelId: id, files });
    } catch (e) {
      console.error("models_download_files failed", e);
    }
  },

  cancel: async (id) => {
    try {
      await invoke<boolean>("models_cancel", { modelId: id });
    } catch (e) {
      console.error("models_cancel failed", e);
    }
  },

  remove: async (id) => {
    await invoke("models_delete", { modelId: id });
    await get().refreshLocal();
  },

  update: async (id) => {
    try {
      await invoke("models_update", { modelId: id });
    } catch (e) {
      console.error("models_update failed", e);
    }
  },

  dismissDownload: (id) => {
    set((s) => {
      if (!(id in s.downloads)) return s;
      const { [id]: _drop, ...rest } = s.downloads;
      return { downloads: rest };
    });
  },

  bindEvents: async () => {
    const off = await on<DownloadProgress>(
      Events.ModelDownloadProgress,
      (p) => {
        set((s) => {
          // Terminal failure states are evicted on arrival so the
          // Models card grid never has to render a dead "error" card.
          // The download invocation itself already surfaced the error
          // to the user via the result panel before it closed.
          if (p.state === "error" || p.state === "cancelled") {
            if (!(p.model_id in s.downloads)) return s;
            const { [p.model_id]: _drop, ...rest } = s.downloads;
            return { downloads: rest };
          }
          return { downloads: { ...s.downloads, [p.model_id]: p } };
        });
        if (p.state === "done") void get().refreshLocal();
      },
    );
    return off;
  },
}));
