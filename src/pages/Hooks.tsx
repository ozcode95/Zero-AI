import { useEffect, useState } from "react";
import { Panel } from "@/components/tui/Panel";
import { TuiInput } from "@/components/tui/Input";
import { TuiButton } from "@/components/tui/Button";
import {
  EMPTY_HOOKS,
  newHook,
  useHooksStore,
  type HookEvent,
  type HookMatcher,
  type HooksConfig,
} from "@/stores/hooks";

/**
 * Hooks Settings sub-section. A CRUD editor for the global hooks config
 * persisted to `settings.json`. Keeps a local dirty draft synced from the
 * store on load and commits the whole config back on Save.
 */

// Static metadata for each event, in the order the UI lists them. The
// `key` matches the snake_case field on `HooksConfig`; `matcherMeaningful`
// is false for the three non-tool events where `matcher` is ignored.
interface EventMeta {
  key: keyof HooksConfig;
  event: HookEvent;
  label: string;
  caption: string;
  matcherMeaningful: boolean;
}

const EVENTS: EventMeta[] = [
  {
    key: "pre_tool_use",
    event: "PreToolUse",
    label: "PreToolUse",
    caption:
      "Fires before a tool call is dispatched. Use the matcher to scope it to specific tools.",
    matcherMeaningful: true,
  },
  {
    key: "post_tool_use",
    event: "PostToolUse",
    label: "PostToolUse",
    caption:
      "Fires after a tool call returns. Matcher constrains which tools trigger it.",
    matcherMeaningful: true,
  },
  {
    key: "user_prompt_submit",
    event: "UserPromptSubmit",
    label: "UserPromptSubmit",
    caption:
      "Fires when the user submits a prompt. There is no tool name to match — matcher is ignored.",
    matcherMeaningful: false,
  },
  {
    key: "stop",
    event: "Stop",
    label: "Stop",
    caption:
      "Fires when the agent stops. There is no tool name to match — matcher is ignored.",
    matcherMeaningful: false,
  },
  {
    key: "session_start",
    event: "SessionStart",
    label: "SessionStart",
    caption:
      "Fires once when a chat session starts. There is no tool name to match — matcher is ignored.",
    matcherMeaningful: false,
  },
];

export function HooksView() {
  const hooks = useHooksStore((s) => s.hooks);
  const load = useHooksStore((s) => s.load);
  const save = useHooksStore((s) => s.save);

  // Local dirty buffer — committed to the store only on Save.
  const [draft, setDraft] = useState<HooksConfig>({ ...EMPTY_HOOKS });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void load();
  }, [load]);

  // Pull the store's config into the local draft whenever it changes (load
  // completion, or a save we just triggered). We deep-clone so the user can
  // mutate freely without touching the store until Save.
  useEffect(() => {
    setDraft({
      pre_tool_use: hooks.pre_tool_use.map((h) => ({ ...h })),
      post_tool_use: hooks.post_tool_use.map((h) => ({ ...h })),
      user_prompt_submit: hooks.user_prompt_submit.map((h) => ({ ...h })),
      stop: hooks.stop.map((h) => ({ ...h })),
      session_start: hooks.session_start.map((h) => ({ ...h })),
    });
  }, [hooks]);

  function updateRow(
    key: keyof HooksConfig,
    index: number,
    patch: Partial<HookMatcher>,
  ) {
    setDraft((d) => {
      const next = { ...d };
      const list = next[key].slice();
      list[index] = { ...list[index], ...patch };
      next[key] = list;
      return next;
    });
  }

  function deleteRow(key: keyof HooksConfig, index: number) {
    setDraft((d) => {
      const next = { ...d };
      next[key] = d[key].filter((_, i) => i !== index);
      return next;
    });
  }

  function addRow(key: keyof HooksConfig) {
    setDraft((d) => {
      const next = { ...d };
      next[key] = [...d[key], newHook()];
      return next;
    });
  }

  async function commit() {
    setSaving(true);
    try {
      await save(draft);
    } finally {
      setSaving(false);
    }
  }

  const totalCount =
    draft.pre_tool_use.length +
    draft.post_tool_use.length +
    draft.user_prompt_submit.length +
    draft.stop.length +
    draft.session_start.length;

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3">
      {/* Safety copy */}
      <div className="fluent-mica rounded-[10px] border border-tui-border bg-[rgba(255,153,164,0.06)] px-3.5 py-2.5 text-[12px] text-tui-fg-dim shadow-[var(--fluent-shadow-2)]">
        <p>
          Hook commands run with the app's privileges using the OS default
          shell. Only register commands you trust. Per-project hooks (
          <code className="font-mono">&lt;workspace&gt;/.zero/hooks.json</code>)
          are file-managed in v1 and are not edited here — this page only edits
          the global config persisted to{" "}
          <code className="font-mono">settings.json</code>.
        </p>
      </div>

      <div className="flex items-center justify-between px-1">
        <span className="text-[12px] text-tui-fg-muted">
          {totalCount} hook{totalCount === 1 ? "" : "s"} configured
        </span>
        <TuiButton
          variant="primary"
          onClick={() => void commit()}
          disabled={saving}
        >
          {saving ? "Saving…" : "Save hooks"}
        </TuiButton>
      </div>

      {/* One collapsible group per event. The Panels render stacked so the
          whole editor scrolls with the section rather than each panel. */}
      {EVENTS.map((meta) => {
        const rows = draft[meta.key];
        return (
          <Panel
            key={meta.key}
            title={meta.label}
            hint={`${rows.length}`}
            flush
            action={
              <TuiButton
                size="sm"
                variant="subtle"
                onClick={() => addRow(meta.key)}
              >
                <span className="text-[13px] leading-none">+</span>
                Add hook
              </TuiButton>
            }
          >
            <div className="flex flex-col gap-2 p-2.5">
              <p className="px-1 text-[11px] text-tui-fg-muted">
                {meta.caption}
              </p>

              {rows.length === 0 && (
                <p className="px-1 py-2 text-[11px] text-tui-fg-muted">
                  No hooks for this event.
                </p>
              )}

              {rows.map((row, i) => (
                <HookRow
                  key={i}
                  row={row}
                  matcherMeaningful={meta.matcherMeaningful}
                  onChange={(patch) => updateRow(meta.key, i, patch)}
                  onDelete={() => deleteRow(meta.key, i)}
                />
              ))}
            </div>
          </Panel>
        );
      })}
    </div>
  );
}

function HookRow({
  row,
  matcherMeaningful,
  onChange,
  onDelete,
}: {
  row: HookMatcher;
  matcherMeaningful: boolean;
  onChange: (patch: Partial<HookMatcher>) => void;
  onDelete: () => void;
}) {
  return (
    <div className="flex flex-col gap-1.5 rounded-[6px] border border-tui-border bg-[var(--fluent-bg-subtle)] p-2">
      <div className="flex items-center gap-2">
        <label className="flex min-w-0 flex-1 flex-col gap-1 text-[11px] font-medium text-tui-fg-dim">
          <span>matcher (regex)</span>
          <TuiInput
            value={row.matcher ?? ""}
            onChange={(e) => onChange({ matcher: e.target.value })}
            placeholder="fs.write"
            disabled={!matcherMeaningful}
          />
        </label>
        {!matcherMeaningful && (
          <span className="mt-4 shrink-0 text-[10.5px] text-tui-fg-muted">
            n/a for this event
          </span>
        )}
      </div>

      <label className="flex flex-col gap-1 text-[11px] font-medium text-tui-fg-dim">
        <span>command</span>
        <TuiInput
          value={row.command}
          onChange={(e) => onChange({ command: e.target.value })}
          placeholder="echo hi"
          className="font-mono"
        />
      </label>

      <div className="flex items-end gap-3">
        <label className="flex w-28 flex-col gap-1 text-[11px] font-medium text-tui-fg-dim">
          <span>timeout (s)</span>
          <TuiInput
            type="number"
            min={1}
            max={3600}
            value={row.timeout_secs}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (!Number.isNaN(n)) {
                onChange({ timeout_secs: Math.max(1, Math.min(3600, n)) });
              }
            }}
          />
        </label>

        <label className="mb-[5px] flex items-center gap-1.5 text-[11px] font-medium text-tui-fg-dim">
          <input
            type="checkbox"
            checked={row.enabled}
            onChange={(e) => onChange({ enabled: e.target.checked })}
          />
          enabled
        </label>

        <div className="flex-1" />

        <TuiButton
          size="sm"
          variant="danger"
          onClick={onDelete}
          className="mb-[1px]"
          aria-label="Delete hook"
        >
          Delete
        </TuiButton>
      </div>
    </div>
  );
}
