import { create } from "zustand";
import { invoke } from "@/lib/tauri";
import type { RecommendedModel, HwMode } from "@/lib/recommendedModels";

export interface GpuInfo {
  name: string;
  vendor: string;
  vram_mb: number | null;
  kind: "integrated" | "discrete" | "unknown";
}

export interface NpuInfo {
  name: string;
  vendor: string;
}

export interface SystemSpecs {
  os: string;
  os_version: string;
  arch: string;
  cpu_brand: string;
  cpu_vendor: string;
  cpu_physical_cores: number;
  cpu_logical_cores: number;
  ram_total_mb: number;
  gpus: GpuInfo[];
  npus: NpuInfo[];
  probed_at: string;
}

interface SystemState {
  specs: SystemSpecs | null;
  loading: boolean;
  error: string | null;
  probe: (force?: boolean) => Promise<void>;
  /**
   * Fetch spec-ranked model recommendations as a flat list, scored for the
   * given hardware mode (`"gpu"` / `"ram"`) and quantization.
   * Returns models sorted best-fit-first.
   */
  recommendModels: (
    mode?: HwMode,
    quant?: string,
  ) => Promise<RecommendedModel[]>;
  /**
   * Force-refresh recommendations: clears caches, re-fetches the online
   * model catalogue, and re-scores against current hardware (for the given
   * mode and quant).
   */
  recommendRefresh: (
    mode?: HwMode,
    quant?: string,
  ) => Promise<RecommendedModel[]>;
  /**
   * Search the llmfit model database for models matching a query.
   * Returns up to 5 results ranked by fit score.
   */
  searchModels: (query: string) => Promise<RecommendedModel[]>;
}

export const useSystemStore = create<SystemState>((set) => ({
  specs: null,
  loading: false,
  error: null,
  probe: async (force = false) => {
    set({ loading: true, error: null });
    try {
      const specs = await invoke<SystemSpecs>("system_probe", { force });
      set({ specs, loading: false });
    } catch (e) {
      set({ error: String(e), loading: false });
    }
  },
  recommendModels: async (mode = "gpu", quant) => {
    try {
      const models = await invoke<RecommendedModel[]>(
        "system_recommend_models",
        { mode, quant },
      );
      return models ?? [];
    } catch (e) {
      console.error("system_recommend_models failed", e);
      return [];
    }
  },
  searchModels: async (query: string): Promise<RecommendedModel[]> => {
    try {
      const results = await invoke<RecommendedModel[]>("system_search_models", {
        query,
      });
      return results ?? [];
    } catch (e) {
      console.error("system_search_models failed", e);
      return [];
    }
  },
  recommendRefresh: async (mode = "gpu", quant) => {
    try {
      const models = await invoke<RecommendedModel[]>(
        "system_recommend_refresh",
        { mode, quant },
      );
      return models ?? [];
    } catch (e) {
      console.error("system_recommend_refresh failed", e);
      return [];
    }
  },
}));
