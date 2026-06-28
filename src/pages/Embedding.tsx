import { useEffect, useMemo, useState } from "react";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { Panel } from "@/components/tui/Panel";
import { Toggle } from "@/components/tui/Toggle";
import { TuiButton } from "@/components/tui/Button";
import { Spinner } from "@/components/tui/Spinner";
import { useModelsStore } from "@/stores/models";
import { useSettingsStore } from "@/stores/settings";
import { useDocumentsStore } from "@/stores/documents";
import { useUiStore } from "@/stores/ui";
import { taskKindFromPipelineTag } from "@/lib/modelTasks";
import { bytes } from "@/lib/format";

/**
 * Embedding page.
 *
 * The feature lets the user ground every chat in their own reference
 * documents. The flow is:
 *
 *   1. An embedding model must be installed. If none is present we send the
 *      user to the Models page to download one.
 *   2. Once a model exists, a master toggle enables the feature for the app.
 *   3. Documents are added from the OS file picker. Every document is
 *      enabled by default; an enabled document's text is injected into the
 *      system prompt of *every* chat session (see the Rust
 *      `render_documents_context`). Toggling one off adds its id to
 *      `settings.embedding.documents_disabled`.
 */
export function EmbeddingView() {
  const local = useModelsStore((s) => s.local);
  const refreshLocal = useModelsStore((s) => s.refreshLocal);

  const embedding = useSettingsStore((s) => s.embedding);
  const saveSettings = useSettingsStore((s) => s.save);

  const documents = useDocumentsStore((s) => s.documents);
  const loadingDocs = useDocumentsStore((s) => s.loading);
  const listDocs = useDocumentsStore((s) => s.list);
  const addDocs = useDocumentsStore((s) => s.add);
  const removeDoc = useDocumentsStore((s) => s.remove);

  const setView = useUiStore((s) => s.setView);

  const [adding, setAdding] = useState(false);

  useEffect(() => {
    void refreshLocal();
    void listDocs();
  }, [refreshLocal, listDocs]);

  // An embedding model is one whose HuggingFace pipeline tag maps to the
  // `embeddings` task bucket (feature-extraction / sentence-similarity / …).
  const embeddingModels = useMemo(
    () =>
      local.filter(
        (m) => taskKindFromPipelineTag(m.pipeline_tag) === "embeddings",
      ),
    [local],
  );
  const hasEmbeddingModel = embeddingModels.length > 0;

  const disabled = useMemo(
    () => new Set(embedding.documents_disabled),
    [embedding.documents_disabled],
  );
  const enabledCount = documents.filter((d) => !disabled.has(d.id)).length;

  async function toggleFeature(on: boolean) {
    await saveSettings({ embedding: { ...embedding, enabled: on } });
  }

  async function toggleDoc(id: string, on: boolean) {
    const next = on
      ? embedding.documents_disabled.filter((x) => x !== id)
      : Array.from(new Set([...embedding.documents_disabled, id]));
    await saveSettings({ embedding: { ...embedding, documents_disabled: next } });
  }

  async function onAddDocuments() {
    if (adding) return;
    try {
      setAdding(true);
      const picked = await openDialog({
        multiple: true,
        filters: [
          {
            name: "Documents",
            extensions: [
              "md",
              "markdown",
              "txt",
              "json",
              "csv",
              "tsv",
              "log",
              "py",
              "rs",
              "ts",
              "tsx",
              "js",
              "jsx",
              "go",
              "java",
              "c",
              "cpp",
              "h",
              "html",
              "xml",
              "yaml",
              "yml",
              "toml",
            ],
          },
          { name: "All", extensions: ["*"] },
        ],
      });
      if (!picked) return;
      const paths = Array.isArray(picked) ? picked : [picked];
      await addDocs(paths);
    } finally {
      setAdding(false);
    }
  }

  async function onRemove(id: string) {
    await removeDoc(id);
    // Drop any stale disabled-entry so the list stays tidy.
    if (embedding.documents_disabled.includes(id)) {
      await saveSettings({
        embedding: {
          ...embedding,
          documents_disabled: embedding.documents_disabled.filter(
            (x) => x !== id,
          ),
        },
      });
    }
  }

  return (
    <div className="mx-auto flex min-h-0 w-full max-w-[860px] flex-1 flex-col gap-3">
      {/* ── header ─────────────────────────────────────────────── */}
      <header className="flex shrink-0 items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-[15px] font-semibold leading-tight text-tui-fg">
            Embedding
          </h1>
          <p className="mt-0.5 text-[11px] text-tui-fg-muted">
            Ground every chat in your own reference documents.
          </p>
        </div>
        {hasEmbeddingModel && (
          <label className="flex shrink-0 items-center gap-2 text-[12px] text-tui-fg-dim">
            <span className="font-medium">
              {embedding.enabled ? "Enabled" : "Disabled"}
            </span>
            <Toggle
              checked={embedding.enabled}
              onChange={(on) => void toggleFeature(on)}
              label="Enable embedding feature"
            />
          </label>
        )}
      </header>

      {!hasEmbeddingModel ? (
        <NoModelCard onBrowse={() => setView("models")} />
      ) : (
        <>
          <FeatureBanner enabled={embedding.enabled} model={embeddingModels[0]?.id ?? null} />

          <Panel
            title="Documents"
            hint={
              documents.length > 0
                ? `${documents.length} · ${enabledCount} active in every chat`
                : "none yet"
            }
            className="min-h-0 flex-1"
            action={
              <TuiButton
                variant="primary"
                size="sm"
                onClick={() => void onAddDocuments()}
                disabled={adding}
              >
                {adding ? "Adding…" : "Add documents"}
              </TuiButton>
            }
            flush
          >
            {loadingDocs && documents.length === 0 ? (
              <div className="flex flex-1 items-center justify-center gap-2 p-6 text-[12px] text-tui-fg-muted">
                <Spinner /> Loading documents…
              </div>
            ) : documents.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center text-[12px] text-tui-fg-muted">
                <DocIcon />
                <p className="font-medium text-tui-fg-dim">No documents yet</p>
                <p className="max-w-sm">
                  Add text, markdown, or code files. Every document you add is
                  enabled by default and becomes grounding context for every
                  chat session.
                </p>
              </div>
            ) : (
              <ul className="flex-1 overflow-auto p-1.5">
                {documents.map((d) => {
                  const on = !disabled.has(d.id);
                  return (
                    <li
                      key={d.id}
                      className="group flex items-center gap-3 rounded-md px-2 py-2 transition-colors hover:bg-[var(--fluent-bg-subtle-hover)]"
                    >
                      <span
                        aria-hidden="true"
                        className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                          on && embedding.enabled
                            ? "bg-tui-accent"
                            : "bg-tui-fg-muted"
                        }`}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-[12px] font-medium text-tui-fg">
                          {d.name}
                        </div>
                        <div className="truncate text-[11px] text-tui-fg-muted">
                          {bytes(d.bytes)}
                          {!on ? " · disabled" : ""}
                        </div>
                      </div>
                      <Toggle
                        checked={on}
                        onChange={(next) => void toggleDoc(d.id, next)}
                        label={on ? `Disable ${d.name}` : `Enable ${d.name}`}
                      />
                      <button
                        onClick={() => void onRemove(d.id)}
                        className="shrink-0 rounded p-1 text-tui-fg-muted opacity-0 transition-opacity duration-150 hover:bg-[var(--fluent-bg-subtle-hover)] hover:text-tui-err group-hover:opacity-100 focus-visible:opacity-100"
                        title="Remove document"
                        aria-label={`Remove ${d.name}`}
                      >
                        <svg
                          width="13"
                          height="13"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.8"
                          strokeLinecap="round"
                        >
                          <path d="M18 6 6 18M6 6l12 12" />
                        </svg>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}

/** Empty-state shown when no embedding model is installed — routes the
 * user to the Models page to download one. */
function NoModelCard({ onBrowse }: { onBrowse: () => void }) {
  return (
    <Panel className="flex-1">
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--fluent-bg-subtle)] text-tui-fg-muted">
          <DocIcon size={24} />
        </div>
        <h2 className="text-[14px] font-semibold text-tui-fg">
          An embedding model is required
        </h2>
        <p className="max-w-md text-[12px] text-tui-fg-muted">
          Embedding turns your documents into vectors so they can ground every
          chat. Download an embedding model first, then come back here to
          enable the feature and add documents.
        </p>
        <TuiButton variant="primary" size="sm" onClick={onBrowse}>
          Browse embedding models
        </TuiButton>
      </div>
    </Panel>
  );
}

/** Status banner explaining what the feature does, tuned to whether it's
 * currently on. */
function FeatureBanner({
  enabled,
  model,
}: {
  enabled: boolean;
  model: string | null;
}) {
  return (
    <div
      className={
        "shrink-0 rounded-[10px] border px-3.5 py-2.5 text-[12px] " +
        (enabled
          ? "border-tui-accent-dim bg-[var(--fluent-bg-subtle-selected)] text-tui-fg-dim"
          : "border-tui-border bg-[var(--fluent-bg-subtle)] text-tui-fg-muted")
      }
    >
      {enabled ? (
        <>
          Embedding is <span className="font-semibold text-tui-fg">on</span>.
          Every enabled document below is added as grounding context to every
          chat session{model ? <> using <code className="font-mono">{model}</code></> : null}.
        </>
      ) : (
        <>
          Embedding is off. Turn it on with the switch above to feed your
          enabled documents into every chat session.
        </>
      )}
    </div>
  );
}

function DocIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
      <path d="M14 3v5h5M9 13h6M9 17h6" />
    </svg>
  );
}
