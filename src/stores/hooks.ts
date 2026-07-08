import { create } from "zustand";
import { invoke } from "@/lib/tauri";

// Hook event kinds. Mirrors the `HookEvent` serde enum on the Rust side:
// the chat runner fires these at well-defined points around tool calls and
// the session lifecycle. The matcher only constrains Pre/PostToolUse — for
// the other three events every hook fires regardless of `matcher`.
export type HookEvent =
  | "PreToolUse"
  | "PostToolUse"
  | "UserPromptSubmit"
  | "Stop"
  | "SessionStart";

/**
 * One shell command bound to an event. `matcher` is a regex tested against
 * the tool name (only meaningful for Pre/PostToolUse); empty/null means
 * "match every tool". `timeout_secs` bounds how long the runner waits for
 * the command before killing it.
 */
export interface HookMatcher {
  matcher: string | null;
  command: string;
  timeout_secs: number;
  enabled: boolean;
}

/**
 * Global hooks configuration persisted to `settings.json`. Each event maps
 * to a list of matchers; the runner executes them in order when the event
 * fires. Mirrors `HooksConfig` in Rust serde exactly — keep the snake_case
 * keys in sync.
 */
export interface HooksConfig {
  pre_tool_use: HookMatcher[];
  post_tool_use: HookMatcher[];
  user_prompt_submit: HookMatcher[];
  stop: HookMatcher[];
  session_start: HookMatcher[];
}

/** A fresh hook row as the UI creates it. The Rust side treats an empty
 * `matcher` (not null) as "match every tool", so we default to `""`. */
export function newHook(): HookMatcher {
  return { matcher: "", command: "", timeout_secs: 30, enabled: true };
}

/** An all-empty config — the value used for defaults and for "clear". */
export const EMPTY_HOOKS: HooksConfig = {
  pre_tool_use: [],
  post_tool_use: [],
  user_prompt_submit: [],
  stop: [],
  session_start: [],
};

interface HooksState {
  hooks: HooksConfig;
  loading: boolean;
  load: () => Promise<void>;
  save: (config: HooksConfig) => Promise<void>;
}

/**
 * Global lifecycle hooks store. Reloads the whole config on `load` and
 * replaces it wholesale on `save` — the backend persists it to
 * `settings.json`. There is no per-row update path; the UI keeps a local
 * dirty draft and commits the entire config on Save.
 */
export const useHooksStore = create<HooksState>((set) => ({
  hooks: { ...EMPTY_HOOKS },
  loading: false,
  load: async () => {
    set({ loading: true });
    try {
      const hooks = (await invoke<HooksConfig>("hooks_get")) ?? {
        ...EMPTY_HOOKS,
      };
      set({ hooks, loading: false });
    } catch (e) {
      console.error("hooks_get failed", e);
      set({ loading: false });
    }
  },
  save: async (config) => {
    await invoke("hooks_set", { config });
    set({ hooks: config });
  },
}));
