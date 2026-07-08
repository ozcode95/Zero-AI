import { create } from "zustand";

export type ViewId = "chat" | "models" | "tasks" | "embedding" | "settings";

/** Sub-section of the Settings view. Memory, Tools, and Skills used to be
 * top-level rail tabs; they now live as sections inside Settings. Keeping
 * the active section in the store (rather than local component state)
 * lets the command palette and the Chat header's "Manage" buttons
 * deep-link straight to one. */
export type SettingsSection =
  | "general"
  | "llama"
  | "audio"
  | "system"
  | "memory"
  | "tools"
  | "skills"
  | "hooks"
  | "agents_md";

interface UiState {
  view: ViewId;
  settingsSection: SettingsSection;
  commandPaletteOpen: boolean;
  setView: (v: ViewId) => void;
  setSettingsSection: (s: SettingsSection) => void;
  /** Jump to the Settings view and (optionally) a specific section. */
  openSettings: (section?: SettingsSection) => void;
  toggleCommandPalette: () => void;
  bindHotkeys: () => () => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  view: "chat",
  settingsSection: "general",
  commandPaletteOpen: false,

  setView: (v) => set({ view: v }),

  setSettingsSection: (s) => set({ settingsSection: s }),

  openSettings: (section) =>
    set((s) => ({
      view: "settings",
      settingsSection: section ?? s.settingsSection,
    })),

  toggleCommandPalette: () =>
    set((s) => ({ commandPaletteOpen: !s.commandPaletteOpen })),

  bindHotkeys: () => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      if (e.key === "k" || e.key === "K") {
        e.preventDefault();
        get().toggleCommandPalette();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  },
}));
