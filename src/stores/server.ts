import { create } from "zustand";
import { on, Events } from "@/lib/tauri";
import type { LlamaLogLine, LlamaStatus } from "@/stores/llama";

/**
 * Status type shared between the legacy OVMS surface (now removed) and
 * the llama.cpp orchestrator. Exported so the BottomBar and other
 * consumers can reference it without importing from either store.
 */
export type ServerStatus = LlamaStatus;

/**
 * Local-runtime log lines share a common `{ts, level, line}` shape
 * regardless of engine. Kept here as a convenience import site for
 * older code that still imports from this module.
 */
export type LogLine = LlamaLogLine;

/**
 * Minimal server store preserved for backward compatibility during the
 * OVMS → llama.cpp migration. All OVMS-specific functionality (multi-model
 * serving, TFS inspection, etc.) has been removed. New code should use
 * `useLlamaStore` from `@/stores/llama` instead.
 *
 * The store is kept non-empty so that existing imports don't break; it
 * simply mirrors the llama.cpp status via the shared event bus.
 */
interface ServerState {
  /** @deprecated Use `useLlamaStore` directly instead. */
  logs: LlamaLogLine[];
  bindEvents: () => Promise<() => void>;
  clearLogs: () => void;
}

export const useServerStore = create<ServerState>((set) => ({
  logs: [],

  bindEvents: async () => {
    // Re-broadcast llama.cpp log lines on the legacy channel so existing
    // event listeners that haven't been migrated yet still get data.
    const offLog = await on<LlamaLogLine>(Events.LlamaLog, (line) => {
      set((s) => ({ logs: [...s.logs.slice(-500), line] }));
    });
    return () => {
      offLog();
    };
  },

  clearLogs: () => set({ logs: [] }),
}));
