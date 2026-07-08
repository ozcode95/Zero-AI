import { create } from "zustand";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { exists } from "@tauri-apps/plugin-fs";
import { invoke } from "@/lib/tauri";
import { useSettingsStore } from "@/stores/settings";

/**
 * The active coding workspace (project root), mirrored from the Rust side.
 *
 * Opening a folder makes it the root the agent's `fs.*` tools resolve
 * relative paths against, shows file edits relative to it, and enables the
 * core file tools. The canonical value lives in `Settings.workspace_root`;
 * this store is the UI-facing view plus the folder-picker flow.
 */
export interface WorkspaceInfo {
  /** Absolute path to the workspace root. */
  path: string;
  /** Final path component — the workspace's display name. */
  name: string;
  /** Whether the path still resolves to a directory on disk. */
  exists: boolean;
}

interface WorkspaceState {
  /** The open workspace, or `null` when none is set. */
  workspace: WorkspaceInfo | null;
  loaded: boolean;
  /** True while a picker / set / clear round-trip is in flight. */
  busy: boolean;
  /** Hydrate from the backend (call once on app start). */
  load: () => Promise<void>;
  /**
   * Open the native folder picker and set the chosen directory as the
   * workspace. Resolves to the new workspace, or `null` if the user
   * cancelled or the call failed.
   */
  pick: () => Promise<WorkspaceInfo | null>;
  /** Set a specific path as the workspace (no picker). */
  setPath: (path: string) => Promise<WorkspaceInfo | null>;
  /** Close the active workspace. */
  clear: () => Promise<void>;
}

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  workspace: null,
  loaded: false,
  busy: false,

  load: async () => {
    try {
      const ws = await invoke<WorkspaceInfo | null>("workspace_get");
      set({ workspace: ws ?? null, loaded: true });
    } catch (e) {
      console.error("workspace_get failed", e);
      set({ loaded: true });
    }
  },

  pick: async () => {
    if (get().busy) return null;
    set({ busy: true });
    try {
      const picked = await openDialog({
        directory: true,
        multiple: false,
        title: "Select a project folder",
      });
      if (typeof picked !== "string") return null; // cancelled
      return await get().setPath(picked);
    } catch (e) {
      console.error("workspace pick failed", e);
      return null;
    } finally {
      set({ busy: false });
    }
  },

  setPath: async (path: string) => {
    set({ busy: true });
    try {
      // Verify the directory is reachable through the file-system plugin
      // before sending it to the backend. `workspace_set` checks is_dir on
      // the Rust side too; this catches a stale / inaccessible path early
      // so a dropped share doesn't clear the workspace silently.
      try {
        const ok = await exists(path);
        if (!ok) {
          console.error("workspace_set: path does not exist", path);
          return null;
        }
      } catch (e) {
        console.error("workspace_set: fs exists check failed", e);
      }
      const ws = await invoke<WorkspaceInfo>("workspace_set", { path });
      set({ workspace: ws });
      // Opening a workspace enables the core file tools on the backend;
      // refresh the settings store so the Tools page reflects that.
      void useSettingsStore.getState().load();
      return ws;
    } catch (e) {
      console.error("workspace_set failed", e);
      return null;
    } finally {
      set({ busy: false });
    }
  },

  clear: async () => {
    set({ busy: true });
    try {
      await invoke("workspace_clear");
      set({ workspace: null });
      // Keep the settings store in sync so a later settings save can't
      // write the stale `workspace_root` back to disk.
      void useSettingsStore.getState().load();
    } catch (e) {
      console.error("workspace_clear failed", e);
    } finally {
      set({ busy: false });
    }
  },
}));
