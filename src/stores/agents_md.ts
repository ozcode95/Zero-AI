import { create } from "zustand";
import { invoke } from "@/lib/tauri";

type Scope = "global" | "project";

/** State for a single AGENTS.md scope. The `path` is `null` for the project
 * scope when no workspace is open — that gates the editor in the UI. */
interface AgentsMdScopeState {
  content: string;
  path: string | null;
  exists: boolean;
  loading: boolean;
}

/** Shape returned by the `agents_md_get` IPC. */
interface AgentsMdGetResult {
  path: string | null;
  exists: boolean;
  content: string;
}

interface AgentsMdState {
  global: AgentsMdScopeState;
  project: AgentsMdScopeState & {
    /**
     * UI-only flag mirroring whether a workspace is currently open. Derived
     * from `useWorkspaceStore` in the component layer; the store itself
     * only stores what IPC returns (including a null `path`).
     */
    editable: boolean;
  };
  load: (scope: Scope) => Promise<void>;
  set: (scope: Scope, content: string) => Promise<void>;
}

const EMPTY_SCOPE: AgentsMdScopeState = {
  content: "",
  path: null,
  exists: false,
  loading: false,
};

/**
 * AGENTS.md context-file store. Two scopes: a global user file under
 * `~/.zero/AGENTS.md` and a per-project file at `<workspace>/AGENTS.md`
 * (the project scope also picks up `CLAUDE.md` and `.zero/AGENTS.md` on
 * read — first existing one wins — but writes always target
 * `<workspace>/AGENTS.md`).
 *
 * The project scope is only readable/writable when a workspace is open;
 * IPC returns `path: null` and rejects `set` otherwise. The component
 * layer surfaces this with an explanatory notice.
 */
export const useAgentsMdStore = create<AgentsMdState>((set) => ({
  global: { ...EMPTY_SCOPE },
  project: { ...EMPTY_SCOPE, editable: false },
  load: async (scope) => {
    set(
      (s) =>
        ({
          [scope]: { ...s[scope], loading: true },
        }) as Partial<AgentsMdState>,
    );
    try {
      const res = (await invoke<AgentsMdGetResult>("agents_md_get", {
        scope,
      })) ?? { path: null, exists: false, content: "" };
      set(
        (s) =>
          ({
            [scope]: {
              ...s[scope],
              content: res.content ?? "",
              path: res.path,
              exists: !!res.exists,
              loading: false,
            },
          }) as Partial<AgentsMdState>,
      );
    } catch (e) {
      console.error("agents_md_get failed", e);
      set(
        (s) =>
          ({
            [scope]: { ...s[scope], loading: false },
          }) as Partial<AgentsMdState>,
      );
    }
  },
  set: async (scope, content) => {
    await invoke("agents_md_set", { scope, content });
    set(
      (s) =>
        ({
          [scope]: {
            ...s[scope],
            content,
            exists: true,
          },
        }) as Partial<AgentsMdState>,
    );
  },
}));
