import { useEffect, useState } from "react";
import { Panel } from "@/components/tui/Panel";
import { TuiTextarea } from "@/components/tui/Input";
import { TuiButton } from "@/components/tui/Button";
import { useAgentsMdStore } from "@/stores/agents_md";
import { useWorkspaceStore } from "@/stores/workspace";

/**
 * AGENTS.md Settings sub-section. Two editor surfaces side-by-side: a
 * global user instructions file and a per-project one. The project editor
 * is gated on a workspace being open (IPC returns `path: null` and rejects
 * `set` otherwise).
 *
 * The content of these files is injected into every chat turn's system
 * prompt — global user instructions plus project instructions. The project
 * scope picks up the first existing of `AGENTS.md` / `CLAUDE.md` /
 * `.zero/AGENTS.md` on read, but writes always target `<workspace>/AGENTS.md`.
 */

type Scope = "global" | "project";

export function AgentsMdView() {
  const global = useAgentsMdStore((s) => s.global);
  const project = useAgentsMdStore((s) => s.project);
  const load = useAgentsMdStore((s) => s.load);
  const setScope = useAgentsMdStore((s) => s.set);

  const workspace = useWorkspaceStore((s) => s.workspace);

  // The project scope is editable iff a workspace is currently open. We
  // keep the store's `editable` flag in sync from here so other consumers
  // can read it, but locally we just recompute.
  const projectEditable = !!workspace;

  useEffect(() => {
    void load("global");
    void load("project");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-load the project scope whenever the workspace changes (open/close
  // or switch) so the editor reflects the file at the new root.
  useEffect(() => {
    void load("project");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace?.path]);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3">
      <p className="px-1 text-[12px] text-tui-fg-dim">
        These files are injected into every chat turn's system prompt:
        global user instructions plus project instructions. For the project
        scope the reader also picks up{" "}
        <code className="font-mono">CLAUDE.md</code> and{" "}
        <code className="font-mono">.zero/AGENTS.md</code> — the first
        existing one wins — but writes always target{" "}
        <code className="font-mono">AGENTS.md</code> at the workspace root.
      </p>

      <div className="flex min-h-0 min-w-0 flex-1 gap-3">
        <ScopeEditor
          scope="global"
          title="Global instructions"
          state={global}
          editable
          onSave={(content) => void setScope("global", content)}
        />
        <ScopeEditor
          scope="project"
          title={
            "Project instructions" +
            (workspace?.name ? ` · ${workspace.name}` : "")
          }
          state={project}
          editable={projectEditable}
          onSave={async (content) => {
            try {
              await setScope("project", content);
            } catch (e) {
              // No workspace open (or the write failed). Surface the same
              // notice the gate uses instead of silently dropping the click.
              console.error("agents_md_set(project) failed", e);
              window.alert(
                "Open a workspace to edit the project's AGENTS.md.",
              );
            }
          }}
        />
      </div>
    </div>
  );
}

interface ScopeEditorProps {
  scope: Scope;
  title: string;
  state: {
    content: string;
    path: string | null;
    exists: boolean;
    loading: boolean;
  };
  editable: boolean;
  onSave: (content: string) => void | Promise<void>;
}

function ScopeEditor({
  scope,
  title,
  state,
  editable,
  onSave,
}: ScopeEditorProps) {
  const { content, path, exists, loading } = state;
  const [draft, setDraft] = useState(content);
  const [saving, setSaving] = useState(false);

  // Re-sync the draft whenever the store content changes (initial load,
  // save we just committed, or a workspace switch for the project scope).
  useEffect(() => {
    setDraft(content);
  }, [content]);

  const gated = scope === "project" && !editable;

  async function save() {
    setSaving(true);
    try {
      await onSave(draft);
    } finally {
      setSaving(false);
    }
  }

  async function wipe() {
    if (!confirm("Clear this file's contents? This writes an empty file.")) {
      return;
    }
    setDraft("");
    setSaving(true);
    try {
      await onSave("");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Panel
      title={title}
      hint={path ?? undefined}
      className="flex-1"
      flush
      action={
        exists && !gated ? (
          <TuiButton size="sm" variant="ghost" onClick={() => void wipe()}>
            Wipe
          </TuiButton>
        ) : undefined
      }
    >
      <div className="flex min-h-0 min-w-0 flex-1 flex-col p-2.5">
        {/* Path + existence badge */}
        <div className="flex items-center gap-2 px-1 pb-2 text-[11px] text-tui-fg-muted">
          {path ? (
            <code className="truncate font-mono text-tui-fg-dim">{path}</code>
          ) : (
            <span className="italic">no path</span>
          )}
          {path && !exists && (
            <span className="shrink-0 rounded-[4px] border border-tui-border bg-[var(--fluent-bg-subtle)] px-1.5 py-[1px] text-[10.5px] text-tui-fg-muted">
              not yet created
            </span>
          )}
        </div>

        {gated ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 p-6 text-center text-[12px] text-tui-fg-muted">
            <p>Open a workspace to edit the project's AGENTS.md.</p>
          </div>
        ) : (
          <>
            <label className="flex min-h-0 flex-1 flex-col gap-1 text-[11px] font-medium text-tui-fg-dim">
              <span>content (markdown)</span>
              <TuiTextarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                rows={16}
                className="flex-1 font-mono text-[12px]"
                placeholder={
                  "# Instructions\n\nWrite the context, conventions, and\npersona the model should assume in this scope."
                }
                disabled={loading}
              />
            </label>
            <div className="flex items-center justify-end gap-2 pt-2">
              <TuiButton
                onClick={() => setDraft(content)}
                disabled={saving || draft === content}
              >
                Reset
              </TuiButton>
              <TuiButton
                variant="primary"
                onClick={() => void save()}
                disabled={saving || draft === content}
              >
                {saving ? "Saving…" : "Save"}
              </TuiButton>
            </div>
          </>
        )}
      </div>
    </Panel>
  );
}
