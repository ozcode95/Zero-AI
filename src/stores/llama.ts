import { create } from "zustand";
import { invoke, on, Events } from "@/lib/tauri";

/**
 * Lifecycle of the bundled `llama-server` runtime. Mirrors
 * `crate::llama::LlamaStatus` on the Rust side; the snake-case wire
 * format is preserved verbatim so the UI never has to translate.
 */
export type LlamaStatus =
  | "not_installed"
  | "installing"
  | "installed"
  | "starting"
  | "running"
  | "stopping"
  | "stopped"
  | "error";

/** Per-variant instance info — mirrors `LlamaInstanceInfo` on the Rust side. */
export interface LlamaInstanceInfo {
  /** Which variant this info belongs to (`"cuda"`, `"openvino"`, etc.). */
  variant: string;
  installed_version: string | null;
  /** Latest upstream release tag, once an update check has run. */
  latest_version: string | null;
  /** True when `installed_version` differs from `latest_version`. */
  update_available: boolean;
  status: LlamaStatus;
  pid: number | null;
  /** Base URL for this variant's instance (e.g. `http://127.0.0.1:8081/v1`). */
  base_url: string;
  loaded_model: string | null;
  loaded_model_path: string | null;
  last_error: string | null;
}

/** Full orchestrator status — mirrors `OrchestratorInfo` on the Rust side. */
export interface OrchestratorInfo {
  /** The variant the chat runner should route to (`"cuda"`, `"openvino"`, etc.). */
  active_variant: string;
  /** Per-variant status. Keyed by variant slug. */
  instances: Record<string, LlamaInstanceInfo>;
  /**
   * Variant slugs that can actually run on the detected hardware
   * (accelerator builds the host supports, plus the universal CPU build).
   * The Settings UI hides variants not in this list. May be absent on
   * older payloads — treat `undefined` as "show everything".
   */
  applicable_variants: string[];
}

/** Log line shape emitted by `llama://log`. */
export interface LlamaLogLine {
  ts: string;
  level: string;
  line: string;
}

export type InstallStage =
  "fetch_release" | "download" | "extract" | "verify" | "done" | "error";

export interface LlamaInstallProgress {
  stage: InstallStage;
  message: string;
  bytes_done: number;
  bytes_total: number | null;
  percent: number;
  /** Variant slug the install pipeline is targeting. */
  variant: string;
}

/** All llama.cpp variant slugs, in priority order. */
export const VARIANT_SLUGS = ["cuda", "openvino", "hip-radeon", "cpu"] as const;
export type VariantSlug = (typeof VARIANT_SLUGS)[number];

export const VARIANT_PORTS: Record<VariantSlug, number> = {
  cuda: 8081,
  openvino: 8082,
  "hip-radeon": 8083,
  cpu: 8084,
};

export const VARIANT_DISPLAY: Record<VariantSlug, string> = {
  cuda: "CUDA (NVIDIA)",
  openvino: "OpenVINO (Intel)",
  "hip-radeon": "HIP (AMD Radeon)",
  cpu: "CPU",
};

/**
 * App-level readiness for the bundled llama.cpp runtime.
 *
 * The UI is "ready" once a usable runtime binary is installed and
 * operational — i.e. the active variant (or, when none is active yet,
 * any variant) is past the install stage and not errored. Until then
 * the chat composer, the model pickers, and every llama.cpp operation
 * (load model, send, etc.) are gated off so the user can't kick off
 * work the runtime can't yet serve.
 *
 * Note: "ready" here means the runtime is *available to operate*, not
 * that a model is currently loaded — loading a model is itself one of
 * the operations this gate enables. Install/update controls are
 * intentionally *not* gated by this (they're what bring the runtime to
 * a ready state).
 */
export function isLlamaReady(info: OrchestratorInfo | null): boolean {
  if (!info) return false;
  const ready = (inst: LlamaInstanceInfo | undefined): boolean =>
    !!inst &&
    inst.status !== "not_installed" &&
    inst.status !== "installing" &&
    inst.status !== "error";
  // Prefer the active variant the chat runner routes to; fall back to
  // any installed instance so a freshly installed-but-not-yet-activated
  // runtime still counts as ready.
  if (ready(info.instances[info.active_variant])) return true;
  return Object.values(info.instances).some(ready);
}

interface LlamaState {
  info: OrchestratorInfo | null;
  logs: LlamaLogLine[];
  installProgress: LlamaInstallProgress | null;
  /**
   * IDs of models whose `llama_load_model` invocation is currently
   * in-flight. Shared so the chat header banner, the Models page
   * cards, and the bottom bar can all reflect "loading" regardless
   * of which page kicked off the load.
   */
  loadingModelIds: Set<string>;
  refresh: () => Promise<void>;
  /** Install all applicable variants for the current hardware. */
  install: () => Promise<void>;
  /** Install a single variant by slug. */
  installVariant: (slug: string) => Promise<void>;
  /** Update (re-install) a specific variant. */
  updateVariant: (slug: string) => Promise<void>;
  /**
   * Check GitHub for the latest llama.cpp release and refresh per-variant
   * `update_available` flags. `force` bypasses the backend's TTL cache.
   */
  checkUpdates: (force?: boolean) => Promise<void>;
  /** Start (or restart) a variant's server, optionally loading a model. */
  start: (variant: string, modelId?: string | null) => Promise<void>;
  /** Stop a variant's server but keep the persisted model assignment. */
  stop: (variant: string) => Promise<void>;
  /** Load a model on the active variant. */
  loadModel: (id: string) => Promise<void>;
  /** Stop the active variant and clear the persisted model id. */
  unloadModel: () => Promise<void>;
  /** Stop a specific variant and clear its persisted model id. */
  unloadVariant: (variant: string) => Promise<void>;
  /** Switch the active variant. */
  switchVariant: (variant: string) => Promise<void>;
  bindEvents: () => Promise<() => void>;
  clearLogs: () => void;
}

export const useLlamaStore = create<LlamaState>((set, get) => ({
  info: null,
  logs: [],
  installProgress: null,
  loadingModelIds: new Set<string>(),

  refresh: async () => {
    try {
      const info = await invoke<OrchestratorInfo>("llama_info");
      if (info) set({ info });
    } catch (e) {
      console.error("llama_info failed", e);
    }
  },

  install: async () => {
    set({ installProgress: null });
    try {
      await invoke("llama_install");
    } catch (e) {
      console.error("llama_install failed", e);
    } finally {
      await get().refresh();
    }
  },

  installVariant: async (slug) => {
    try {
      await invoke("llama_install_variant", { variant: slug });
    } catch (e) {
      console.error("llama_install_variant failed", e);
    } finally {
      await get().refresh();
    }
  },

  updateVariant: async (slug) => {
    try {
      await invoke("llama_update_variant", { variant: slug });
    } catch (e) {
      console.error("llama_update_variant failed", e);
    } finally {
      await get().refresh();
    }
  },

  checkUpdates: async (force = false) => {
    try {
      const info = await invoke<OrchestratorInfo>("llama_check_updates", {
        force,
      });
      if (info) set({ info });
    } catch (e) {
      console.error("llama_check_updates failed", e);
    }
  },

  start: async (variant, modelId) => {
    await invoke("llama_start", { variant, modelId: modelId ?? null });
    await get().refresh();
  },

  stop: async (variant) => {
    await invoke("llama_stop", { variant });
    await get().refresh();
  },

  loadModel: async (id) => {
    set((s) => {
      const next = new Set(s.loadingModelIds);
      next.add(id);
      return { loadingModelIds: next };
    });
    try {
      await invoke("llama_load_model", { modelId: id });
      await get().refresh();
    } finally {
      set((s) => {
        if (!s.loadingModelIds.has(id)) return s;
        const next = new Set(s.loadingModelIds);
        next.delete(id);
        return { loadingModelIds: next };
      });
    }
  },

  unloadModel: async () => {
    await invoke("llama_unload_model");
    await get().refresh();
  },

  unloadVariant: async (variant) => {
    await invoke("llama_unload_variant", { variant });
    await get().refresh();
  },

  switchVariant: async (variant) => {
    await invoke("llama_switch_variant", { variant });
    await get().refresh();
  },

  bindEvents: async () => {
    const offLog = await on<LlamaLogLine>(Events.LlamaLog, (line) => {
      set((s) => ({ logs: [...s.logs.slice(-500), line] }));
    });
    const offStatus = await on<OrchestratorInfo>(Events.LlamaStatus, (info) => {
      set({ info });
    });
    const offProgress = await on<LlamaInstallProgress>(
      Events.LlamaInstallProgress,
      (p) => {
        set({ installProgress: p });
        if (p.stage === "done") {
          setTimeout(() => {
            const cur = get().installProgress;
            if (cur && cur.stage === "done") set({ installProgress: null });
          }, 2500);
        }
      },
    );
    // Opportunistic update check on startup. Best-effort and TTL-cached on
    // the backend, so this won't hammer the GitHub API across reloads.
    void get().checkUpdates();
    return () => {
      offLog();
      offStatus();
      offProgress();
    };
  },

  clearLogs: () => set({ logs: [] }),
}));
