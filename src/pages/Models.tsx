import { useEffect, useMemo, useState } from "react";
import { TuiButton } from "@/components/tui/Button";
import { TuiInput } from "@/components/tui/Input";
import { ProgressBar } from "@/components/tui/ProgressBar";
import { Spinner } from "@/components/tui/Spinner";

import {
  useModelsStore,
  type DownloadProgress,
  type GgufFileInfo,
  type LocalModel,
} from "@/stores/models";
import {
  useLlamaStore,
  isLlamaReady,
  type OrchestratorInfo,
} from "@/stores/llama";
import { useSettingsStore } from "@/stores/settings";
import { useSystemStore } from "@/stores/system";
import { bytes, relativeTime } from "@/lib/format";
import {
  pipelineTagToTaskKind,
  nonChatKindFromName,
} from "@/lib/modelCategory";
import {
  type RecommendedModel,
  type HwMode,
  SUPPORTED_QUANTS,
  DEFAULT_QUANT,
} from "@/lib/recommendedModels";

/**
 * Models page — single-view layout with recommended table and local
 * model-name search (filters the recommendation list client-side).
 *
 *   ┌────────────────────────────────────────────────────────┐
 *   │  [GPU|RAM] [ Search… ] [Use Case] [Fit] [Quant]         │
 *   ├────────────────────────────────────────────────────────┤
 *   │  Fit | Model | Param | Quant | VRAM | Ctx | Speed | …  │
 *   │  ──────────────────────────────────────────────────── │
 *   │  row …                                                 │
 *   └────────────────────────────────────────────────────────┘
 */
export function ModelsView() {
  // ── local library ────────────────────────────────────────────────
  const local = useModelsStore((s) => s.local);
  const downloads = useModelsStore((s) => s.downloads);
  const refreshLocal = useModelsStore((s) => s.refreshLocal);
  const download = useModelsStore((s) => s.download);
  const cancel = useModelsStore((s) => s.cancel);
  const remove = useModelsStore((s) => s.remove);
  const [installedModalOpen, setInstalledModalOpen] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // ── llama.cpp lifecycle (for loaded-model badge) ────────────────
  const llamaInfo = useLlamaStore((s) => s.info);
  const loadingModelIds = useLlamaStore((s) => s.loadingModelIds);
  const llamaLoadModel = useLlamaStore((s) => s.loadModel);
  const llamaUnloadModel = useLlamaStore((s) => s.unloadModel);
  const llamaLoadedModel = (() => {
    const active = llamaInfo?.active_variant;
    if (!active) return null;
    return llamaInfo?.instances[active]?.loaded_model ?? null;
  })();
  const isLlama = useSettingsStore((s) => {
    const id = s.active_provider_id;
    return s.providers.find((p) => p.id === id)?.kind === "llama.cpp";
  });

  const loadedModelIds = useMemo(() => {
    if (!isLlama || !llamaLoadedModel) return new Set<string>();
    return new Set([llamaLoadedModel]);
  }, [isLlama, llamaLoadedModel]);

  // Initial load
  useEffect(() => {
    void refreshLocal();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Local HF ids (for deduping). */
  const localHfIds = useMemo(
    () => new Set(local.map((l) => l.hf_id).filter((x): x is string => !!x)),
    [local],
  );

  async function startDownload(id: string, metadata?: Record<string, unknown>) {
    setInstalledModalOpen(true);
    await download(id, metadata);
  }

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      {/* ── Page header + custom download (pinned) ───────────── */}
      <div className="mx-auto w-full max-w-[1200px] shrink-0 px-4 pt-3">
        <header className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="text-[15px] font-semibold leading-tight text-tui-fg">
              Models
            </h1>
            <p className="mt-0.5 text-[11px] text-tui-fg-muted">
              Browse models tuned to your hardware, or download any GGUF repo by
              id.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <TuiButton
              variant="subtle"
              size="sm"
              onClick={() => setRefreshTrigger((n) => n + 1)}
              title="Clear caches and re-fetch the model catalogue"
            >
              Refresh
            </TuiButton>
            <TuiButton
              variant="primary"
              size="sm"
              onClick={() => setInstalledModalOpen(true)}
              title="View and manage downloaded models"
            >
              Installed{local.length > 0 ? ` · ${local.length}` : ""}
            </TuiButton>
          </div>
        </header>

        <div className="mt-3">
          <ManualDownloadPanel
            onDownloaded={() => setInstalledModalOpen(true)}
          />
        </div>
      </div>

      {/* ── Recommended table (scrollable) ─────────────────── */}
      <div className="mx-auto flex min-h-0 w-full max-w-[1200px] flex-1 flex-col overflow-auto px-4 pb-8 pt-4">
        <RecommendedView
          localHfIds={localHfIds}
          loadedModelIds={loadedModelIds}
          downloads={downloads}
          onDownload={(id, metadata) => void startDownload(id, metadata)}
          onCancel={cancel}
          refreshTrigger={refreshTrigger}
        />
      </div>

      {/* ── Installed Models Modal ────────────────────────────────── */}
      {installedModalOpen && (
        <InstalledModelsModal
          local={local}
          downloads={downloads}
          onCancel={cancel}
          onClose={() => setInstalledModalOpen(false)}
          onDelete={(id) => void remove(id)}
          llamaInfo={llamaInfo}
          loadingModelIds={loadingModelIds}
          onLoad={llamaLoadModel}
          onUnload={llamaUnloadModel}
        />
      )}
    </div>
  );
}

// ─── installed models modal ───────────────────────────────────────────

/** Parsed shape of metadata_json stored alongside installed models. */
interface ModelMetadata {
  useCase?: string;
  fitLevel?: string;
  score?: number;
  bestQuant?: string;
  contextLength?: number;
  estimatedTps?: number;
  memoryRequiredGb?: number;
  runMode?: string;
  inputTypes?: string[];
  capabilities?: string[];
  parameterCount?: string;
  sizeHint?: string;
  provider?: string;
  /** Model weight format (e.g. "Gguf", "Awq"). */
  modelFormat?: string;
  /** Inference runtime (e.g. "LlamaCpp", "Mlx"). */
  inferenceRuntime?: string;
}

function parseMetadata(json: string | null | undefined): ModelMetadata | null {
  if (!json) return null;
  try {
    return JSON.parse(json) as ModelMetadata;
  } catch {
    return null;
  }
}

function InstalledModelsModal({
  local,
  downloads,
  onCancel,
  onClose,
  onDelete,
  llamaInfo,
  loadingModelIds,
  onLoad,
  onUnload,
}: {
  local: LocalModel[];
  downloads: Record<string, DownloadProgress>;
  onCancel: (id: string) => Promise<void>;
  onClose: () => void;
  onDelete: (id: string) => void;
  llamaInfo: OrchestratorInfo | null;
  loadingModelIds: Set<string>;
  onLoad: (localModelId: string) => Promise<void>;
  onUnload: () => Promise<void>;
}) {
  const activeDownloads = Object.values(downloads).filter(
    (d) =>
      d.state === "pending" ||
      d.state === "downloading" ||
      d.state === "verifying",
  );

  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  /** Two-click delete: first click asks for confirmation, second executes. */
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

  const toggle = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Shared styled helpers (mirror RecommendedView detail modal)
  const Dt = ({ children }: { children: React.ReactNode }) => (
    <span className="text-tui-fg-dim">{children}</span>
  );
  const Dd = ({
    children,
    mono,
    className,
  }: {
    children: React.ReactNode;
    mono?: boolean;
    className?: string;
  }) => (
    <span
      className={`min-w-0 ${mono ? "font-mono text-[11px]" : "text-tui-fg"} ${className ?? ""}`}
    >
      {children}
    </span>
  );

  // Header summary: count + aggregate on-disk footprint.
  const totalBytes = local.reduce((sum, m) => sum + (m.bytes || 0), 0);

  const isModelLoaded = (m: LocalModel) =>
    !!llamaInfo &&
    llamaInfo.instances[llamaInfo.active_variant]?.loaded_model === m.id;
  // The Load button (and any other llama.cpp operation) is only usable
  // once the runtime is installed and operational. Until then loading a
  // model would fail, so gate it off.
  const runtimeReady = isLlamaReady(llamaInfo);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-[2px]"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={
          "fluent-mica relative flex max-h-[82vh] w-full max-w-[840px] flex-col overflow-hidden " +
          "rounded-[14px] border border-tui-border shadow-[var(--fluent-shadow-64)]"
        }
      >
        {/* Header */}
        <div className="flex items-center justify-between gap-3 border-b border-tui-border/60 px-5 py-4">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-[10px] border border-tui-border/60 bg-[var(--fluent-bg-subtle)] text-tui-accent">
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <path d="m3.3 7 8.7 5 8.7-5M12 22V12" />
              </svg>
            </span>
            <div>
              <h2
                className="text-[15px] font-semibold leading-tight text-tui-fg"
                style={{ fontFamily: "var(--font-display)" }}
              >
                Installed Models
              </h2>
              <p className="text-[11px] text-tui-fg-muted">
                {local.length} model{local.length === 1 ? "" : "s"}
                {totalBytes > 0 ? ` \u00b7 ${bytes(totalBytes)} on disk` : ""}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-[6px] p-1.5 text-tui-fg-muted transition-colors hover:bg-[var(--fluent-bg-subtle-hover)] hover:text-tui-fg"
            aria-label="Close"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto px-5 pb-5 pt-4">
          {/* ── In-progress downloads ─────────────────────────────── */}
          {activeDownloads.length > 0 && (
            <div className="mb-5">
              <h3 className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-tui-fg-muted">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-tui-accent opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-tui-accent" />
                </span>
                Downloading
              </h3>
              <div className="space-y-2">
                {activeDownloads.map((dl) => {
                  const pct = dl.bytes_total
                    ? Math.round((dl.bytes_done / dl.bytes_total) * 100)
                    : null;
                  return (
                    <div
                      key={dl.model_id}
                      className="rounded-[10px] border border-tui-accent/25 bg-tui-accent/[0.04] p-3"
                    >
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <span
                          className="truncate text-[13px] font-medium text-tui-fg"
                          title={dl.model_id}
                        >
                          {dl.model_id}
                        </span>
                        <div className="flex shrink-0 items-center gap-2">
                          <span className="font-mono text-[11px] tabular-nums text-tui-fg-muted">
                            {dl.files_done}/{dl.files_total} files{" \u00b7 "}
                            {bytes(dl.bytes_done)}
                            {dl.bytes_total
                              ? ` / ${bytes(dl.bytes_total)}`
                              : ""}
                            {pct != null ? ` \u00b7 ${pct}%` : ""}
                          </span>
                          <TuiButton
                            variant="ghost"
                            size="sm"
                            onClick={() => void onCancel(dl.model_id)}
                          >
                            Cancel
                          </TuiButton>
                        </div>
                      </div>
                      <ProgressBar
                        value={
                          dl.bytes_total ? dl.bytes_done / dl.bytes_total : 0
                        }
                        label={dl.state}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Installed models ─────────────────────────── */}
          <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-tui-fg-muted">
            Installed
          </h3>
          {local.length === 0 ? (
            <div className="flex flex-col items-center gap-2 rounded-[12px] border border-dashed border-tui-border/50 py-12 text-center">
              <svg
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-tui-fg-dim"
              >
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <path d="m3.3 7 8.7 5 8.7-5M12 22V12" />
              </svg>
              <p className="text-[13px] font-medium text-tui-fg-muted">
                No models installed yet
              </p>
              <p className="max-w-[20rem] text-[11px] leading-relaxed text-tui-fg-dim">
                Pick a model from the recommendation table and hit Download —
                only the files it needs are fetched, not the whole repo.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {local.map((m) => {
                const isLoaded = isModelLoaded(m);
                const isLoading = loadingModelIds.has(m.id);
                const expanded = expandedIds.has(m.id);
                const meta = parseMetadata(m.metadata_json);
                // Audio (speech↔text), embedding, and rerank models aren't
                // served through the llama.cpp chat orchestrator, so the
                // Load/Unload control is meaningless for them — hide it.
                // Fall back to a name-based check when the model carries no
                // pipeline tag (e.g. WavTokenizer / Ultravox GGUF repos).
                const taskKind =
                  pipelineTagToTaskKind(m.pipeline_tag) ??
                  nonChatKindFromName(m.hf_id ?? m.id);
                const isLoadable = !(
                  taskKind === "embeddings" ||
                  taskKind === "rerank" ||
                  taskKind === "speech2text" ||
                  taskKind === "text2speech"
                );
                const isMultimodal =
                  (meta?.inputTypes ?? []).some((t) =>
                    /image|vision|audio|video/i.test(t),
                  ) ||
                  /image|vision|audio|video|multimodal|mmproj/i.test(
                    m.pipeline_tag ?? "",
                  );

                return (
                  <div
                    key={m.id}
                    className={`overflow-hidden rounded-[10px] border transition-colors ${
                      isLoaded
                        ? "border-tui-accent/40 bg-tui-accent/[0.05]"
                        : "border-tui-border/40 bg-[var(--fluent-bg-subtle)] hover:border-tui-border/70"
                    }`}
                  >
                    {/* Summary row (always visible) */}
                    <div className="flex items-center gap-3 px-3 py-2.5">
                      {/* Type icon */}
                      <span
                        className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-[9px] border ${
                          isLoaded
                            ? "border-tui-accent/40 text-tui-accent"
                            : "border-tui-border/50 text-tui-fg-muted"
                        }`}
                      >
                        {isMultimodal ? (
                          <svg
                            width="17"
                            height="17"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.7"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <rect x="3" y="3" width="18" height="18" rx="2" />
                            <circle cx="9" cy="9" r="2" />
                            <path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21" />
                          </svg>
                        ) : (
                          <svg
                            width="17"
                            height="17"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.7"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                          </svg>
                        )}
                      </span>

                      {/* Model identity — click to expand */}
                      <button
                        type="button"
                        className="min-w-0 flex-1 text-left"
                        onClick={() => toggle(m.id)}
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className="truncate text-[13px] font-semibold text-tui-fg"
                            title={m.hf_id ?? m.id}
                          >
                            {m.hf_id ?? m.id}
                          </span>
                          {isLoaded && (
                            <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-tui-accent/40 bg-tui-accent/10 px-1.5 py-px text-[9px] font-semibold uppercase tracking-wide text-tui-accent">
                              <span className="h-1 w-1 rounded-full bg-tui-accent" />
                              Loaded
                            </span>
                          )}
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-tui-fg-muted">
                          <span className="font-mono tabular-nums">
                            {bytes(m.bytes)}
                          </span>
                          {meta?.parameterCount && (
                            <span className="text-tui-fg-dim">
                              {"\u00b7"} {meta.parameterCount}
                            </span>
                          )}
                          {meta?.bestQuant && (
                            <span className="rounded-[3px] border border-tui-border/40 bg-[var(--fluent-bg-subtle)] px-1.5 py-px font-mono text-[10px] text-tui-fg-muted">
                              {meta.bestQuant}
                            </span>
                          )}
                          {m.pipeline_tag && (
                            <span className="rounded-[3px] border border-tui-border/40 bg-[var(--fluent-bg-subtle)] px-1.5 py-px text-[10px] text-tui-fg-muted">
                              {m.pipeline_tag}
                            </span>
                          )}
                          {meta?.fitLevel && <FitBadge level={meta.fitLevel} />}
                        </div>
                      </button>

                      {/* Load / unload + delete controls */}
                      <div className="flex shrink-0 items-center gap-1">
                        {!isLoadable ? null : isLoading ? (
                          <div className="flex items-center gap-1.5 px-1 text-[11px] text-tui-fg-muted">
                            <Spinner size="sm" />
                            <span>Loading…</span>
                          </div>
                        ) : isLoaded ? (
                          <TuiButton
                            variant="ghost"
                            size="sm"
                            onClick={() => void onUnload()}
                          >
                            Unload
                          </TuiButton>
                        ) : (
                          <TuiButton
                            variant="primary"
                            size="sm"
                            disabled={!runtimeReady}
                            title={
                              runtimeReady
                                ? undefined
                                : "llama.cpp runtime not ready"
                            }
                            onClick={() => void onLoad(m.id)}
                          >
                            Load
                          </TuiButton>
                        )}
                        {confirmingId === m.id ? (
                          <>
                            <TuiButton
                              variant="danger"
                              size="sm"
                              onClick={() => {
                                onDelete(m.id);
                                setConfirmingId(null);
                              }}
                            >
                              Delete?
                            </TuiButton>
                            <TuiButton
                              variant="ghost"
                              size="sm"
                              onClick={() => setConfirmingId(null)}
                            >
                              Cancel
                            </TuiButton>
                          </>
                        ) : (
                          <TuiButton
                            variant="ghost"
                            size="sm"
                            onClick={() => setConfirmingId(m.id)}
                            title="Delete model"
                          >
                            <svg
                              width="13"
                              height="13"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                            >
                              <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                            </svg>
                          </TuiButton>
                        )}
                        {/* Expand toggle */}
                        <button
                          type="button"
                          onClick={() => toggle(m.id)}
                          className="rounded-[6px] p-1.5 text-tui-fg-muted transition-colors hover:bg-[var(--fluent-bg-subtle-hover)] hover:text-tui-fg"
                          aria-label={
                            expanded ? "Hide details" : "Show details"
                          }
                          aria-expanded={expanded}
                        >
                          <svg
                            width="13"
                            height="13"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                            className={`transition-transform ${expanded ? "rotate-90" : ""}`}
                          >
                            <path d="m9 18 6-6-6-6" />
                          </svg>
                        </button>
                      </div>
                    </div>

                    {/* ── Expanded detail grid ─────────────────── */}
                    {expanded && (
                      <div className="border-t border-tui-border/20 px-4 py-3">
                        <div className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-2 text-[12px]">
                          {/* Core fields */}
                          <Dt>Internal ID</Dt>
                          <Dd mono>{m.id}</Dd>

                          <Dt>HF Repo</Dt>
                          <Dd mono>{m.hf_id ?? "—"}</Dd>

                          <Dt>Path</Dt>
                          <Dd mono className="truncate">
                            {m.path}
                          </Dd>

                          <Dt>Size</Dt>
                          <Dd>{bytes(m.bytes)}</Dd>

                          <Dt>Added</Dt>
                          <Dd>{relativeTime(m.added_at)}</Dd>

                          <Dt>Revision</Dt>
                          <Dd mono>{m.revision ?? "—"}</Dd>

                          <Dt>Files</Dt>
                          <Dd>
                            {m.files != null ? (
                              <>
                                {m.files}
                                {m.verified_files != null &&
                                  m.verified_files > 0 && (
                                    <span className="ml-1 text-[10px] text-tui-fg-dim">
                                      ({m.verified_files} verified)
                                    </span>
                                  )}
                              </>
                            ) : (
                              "—"
                            )}
                          </Dd>

                          <Dt>Pipeline</Dt>
                          <Dd>{m.pipeline_tag ?? "—"}</Dd>

                          {/* Metadata from llmfit recommendation */}
                          {meta ? (
                            <>
                              {/* Thin separator */}
                              <div className="col-span-2 my-1 border-t border-tui-border/20" />

                              {meta.provider && (
                                <>
                                  <Dt>Provider</Dt>
                                  <Dd>{meta.provider}</Dd>
                                </>
                              )}

                              {meta.parameterCount && (
                                <>
                                  <Dt>Params</Dt>
                                  <Dd>{meta.parameterCount}</Dd>
                                </>
                              )}

                              {meta.sizeHint && (
                                <>
                                  <Dt>Size Hint</Dt>
                                  <Dd>{meta.sizeHint}</Dd>
                                </>
                              )}

                              {meta.bestQuant && (
                                <>
                                  <Dt>Best Quant</Dt>
                                  <Dd mono>{meta.bestQuant}</Dd>
                                </>
                              )}

                              {meta.contextLength != null &&
                                meta.contextLength > 0 && (
                                  <>
                                    <Dt>Context</Dt>
                                    <Dd>
                                      {meta.contextLength >= 1000
                                        ? `${Math.round(meta.contextLength / 1000)}K tokens`
                                        : `${meta.contextLength} tokens`}
                                    </Dd>
                                  </>
                                )}

                              {meta.useCase && (
                                <>
                                  <Dt>Use Case</Dt>
                                  <Dd className="capitalize">{meta.useCase}</Dd>
                                </>
                              )}

                              {meta.fitLevel && (
                                <>
                                  <Dt>Fit</Dt>
                                  <Dd>
                                    <span
                                      className={`inline-block rounded-[3px] px-1.5 py-px text-[11px] font-medium ${
                                        meta.fitLevel === "perfect"
                                          ? "bg-emerald-500/15 text-emerald-400"
                                          : meta.fitLevel === "good"
                                            ? "bg-sky-500/15 text-sky-400"
                                            : meta.fitLevel === "marginal"
                                              ? "bg-amber-500/15 text-amber-400"
                                              : "bg-red-500/15 text-red-400"
                                      }`}
                                    >
                                      {meta.fitLevel}
                                    </span>
                                  </Dd>
                                </>
                              )}

                              {meta.score != null && (
                                <>
                                  <Dt>Score</Dt>
                                  <Dd>{Math.round(meta.score)} / 100</Dd>
                                </>
                              )}

                              {meta.estimatedTps != null &&
                                meta.estimatedTps > 0 && (
                                  <>
                                    <Dt>Speed</Dt>
                                    <Dd>
                                      ~{Math.round(meta.estimatedTps)} t/s
                                    </Dd>
                                  </>
                                )}

                              {meta.memoryRequiredGb != null && (
                                <>
                                  <Dt>Memory</Dt>
                                  <Dd>{meta.memoryRequiredGb.toFixed(1)} GB</Dd>
                                </>
                              )}

                              {meta.runMode && (
                                <>
                                  <Dt>Run Mode</Dt>
                                  <Dd>{meta.runMode}</Dd>
                                </>
                              )}

                              {meta.modelFormat && (
                                <>
                                  <Dt>Format</Dt>
                                  <Dd mono>{meta.modelFormat}</Dd>
                                </>
                              )}

                              {meta.inferenceRuntime && (
                                <>
                                  <Dt>Runtime</Dt>
                                  <Dd mono>{meta.inferenceRuntime}</Dd>
                                </>
                              )}

                              {meta.inputTypes &&
                                meta.inputTypes.length > 0 && (
                                  <>
                                    <Dt>Inputs</Dt>
                                    <Dd>
                                      <div className="flex flex-wrap items-center gap-1">
                                        {meta.inputTypes.map((t) => (
                                          <span
                                            key={t}
                                            className="rounded-[3px] border border-tui-border/40 bg-[var(--fluent-bg-subtle)] px-1.5 py-px text-[10px] text-tui-fg-muted"
                                          >
                                            {t}
                                          </span>
                                        ))}
                                      </div>
                                    </Dd>
                                  </>
                                )}

                              {meta.capabilities &&
                                meta.capabilities.length > 0 && (
                                  <>
                                    <Dt>Capabilities</Dt>
                                    <Dd>
                                      <div className="flex flex-wrap items-center gap-1">
                                        {meta.capabilities.map((c) => (
                                          <span
                                            key={c}
                                            className="rounded-[3px] border border-tui-border/40 bg-[var(--fluent-bg-subtle)] px-1.5 py-px text-[10px] text-tui-fg-muted"
                                          >
                                            {c}
                                          </span>
                                        ))}
                                      </div>
                                    </Dd>
                                  </>
                                )}
                            </>
                          ) : (
                            <>
                              <div className="col-span-2 my-1 border-t border-tui-border/20" />
                              <Dt>Metadata</Dt>
                              <Dd className="text-tui-fg-dim italic">
                                None — install via the recommendation table to
                                populate
                              </Dd>
                            </>
                          )}

                          {/* Raw JSON (always visible when present) */}
                          {m.metadata_json && (
                            <>
                              <div className="col-span-2 mt-1 border-t border-tui-border/20 pt-2">
                                <Dt>Raw JSON</Dt>
                                <Dd
                                  mono
                                  className="select-all break-all text-[10px]"
                                >
                                  {m.metadata_json}
                                </Dd>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Column sort helpers
const SORT_FNS: Record<
  string,
  (a: RecommendedModel, b: RecommendedModel) => number
> = {
  name: (a, b) => a.name.localeCompare(b.name),
  provider: (a, b) => a.provider.localeCompare(b.provider),
  useCase: (a, b) => a.useCase.localeCompare(b.useCase),
  parameterCount: (a, b) =>
    parseFloat(a.parameterCount) - parseFloat(b.parameterCount),
  score: (a, b) => a.score - b.score,
  bestQuant: (a, b) => a.bestQuant.localeCompare(b.bestQuant),
  contextLength: (a, b) => a.contextLength - b.contextLength,
  fitLevel: (a, b) => fitRank(a.fitLevel) - fitRank(b.fitLevel),
  estimatedTps: (a, b) => a.estimatedTps - b.estimatedTps,
  memoryRequiredGb: (a, b) => a.memoryRequiredGb - b.memoryRequiredGb,
  modelFormat: (a, b) => a.modelFormat.localeCompare(b.modelFormat),
  inferenceRuntime: (a, b) =>
    a.inferenceRuntime.localeCompare(b.inferenceRuntime),
};

function fitRank(f: string): number {
  if (f === "perfect") return 0;
  if (f === "good") return 1;
  if (f === "marginal") return 2;
  return 3;
}

/**
 * Default the GPU/RAM toggle from the active llama.cpp variant: discrete-GPU
 * builds (CUDA, HIP/Radeon) score against VRAM; CPU and OpenVINO (iGPU) builds
 * score against system RAM. Falls back to GPU when no variant is active yet.
 */
function defaultHwMode(): HwMode {
  const active = useLlamaStore.getState().info?.active_variant;
  if (active === "cpu" || active === "openvino") return "ram";
  return "gpu";
}

/** Compact memory footprint, e.g. "5.5 GB" / "512 MB". */
function formatMem(gb: number): string {
  if (gb <= 0) return "—";
  return gb < 1 ? `${Math.round(gb * 1024)} MB` : `${gb.toFixed(1)} GB`;
}

/** tok/s with one decimal under 100, rounded above. */
function formatTps(tps: number): string {
  return tps >= 100 ? `${Math.round(tps)}` : tps.toFixed(1);
}

/** Friendly inference-engine label for the Mode column. */
function engineLabel(runtime: string): string {
  switch (runtime) {
    case "LlamaCpp":
      return "llama.cpp";
    case "Mlx":
      return "MLX";
    case "Vllm":
      return "vLLM";
    default:
      return runtime || "—";
  }
}

// ─── recommended view ─────────────────────────────────────────────────

function Sorter({
  label,
  column,
  sortKey,
  sortDir,
  onClick,
}: {
  label: string;
  column: string;
  sortKey: string;
  sortDir: "asc" | "desc";
  onClick: (col: string) => void;
}) {
  const active = sortKey === column;
  return (
    <button
      type="button"
      className={
        "inline-flex items-center gap-0.5 font-medium uppercase tracking-wider transition-colors " +
        (active ? "text-tui-fg" : "text-tui-fg-muted hover:text-tui-fg")
      }
      onClick={() => onClick(column)}
    >
      {label}
      {active && (
        <span className="text-[8px] leading-none">
          {sortDir === "asc" ? "\u25B2" : "\u25BC"}
        </span>
      )}
    </button>
  );
}

function RecommendedView({
  localHfIds,
  loadedModelIds,
  downloads,
  onDownload,
  onCancel,
  refreshTrigger,
}: {
  localHfIds: Set<string>;
  loadedModelIds: Set<string>;
  downloads: Record<string, DownloadProgress>;
  onDownload: (hfId: string, metadata?: Record<string, unknown>) => void;
  onCancel: (id: string) => Promise<void>;
  /** Increment to trigger a cache-clearing refresh from the parent. */
  refreshTrigger: number;
}) {
  const recommendModels = useSystemStore((s) => s.recommendModels);
  const recommendRefresh = useSystemStore((s) => s.recommendRefresh);
  const [models, setModels] = useState<RecommendedModel[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortKey, setSortKey] = useState<string>("score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  // Hardware mode + quant are *scoring* inputs sent to the backend (they
  // change every model's score / tok/s / fit), not client-side filters.
  // Default the mode from the active llama.cpp variant: GPU builds (CUDA,
  // HIP) score against VRAM; CPU / OpenVINO builds against system RAM.
  const [hwMode, setHwMode] = useState<HwMode>(() => defaultHwMode());
  const [quant, setQuant] = useState<string>(DEFAULT_QUANT);
  const [filterFit, setFilterFit] = useState("");
  const [filterUseCase, setFilterUseCase] = useState("");
  const [detailModel, setDetailModel] = useState<RecommendedModel | null>(null);

  const handleSort = (col: string) => {
    if (sortKey === col) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(col);
      setSortDir(col === "name" || col === "useCase" ? "asc" : "desc");
    }
  };

  // Load (and reload) whenever the hardware mode or quant changes — both are
  // scoring inputs, so the backend recomputes scores / tok/s / fit for them.
  // Results are disk-cached per (mode, quant), so toggling is cheap.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    recommendModels(hwMode, quant)
      .then((list) => {
        if (!cancelled) setModels(list);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hwMode, quant]);

  // Parent-triggered refresh (skip initial mount where refreshTrigger is 0).
  useEffect(() => {
    if (refreshTrigger === 0) return;
    handleRefresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshTrigger]);

  const handleRefresh = () => {
    setModels(null);
    setLoading(true);
    recommendRefresh(hwMode, quant)
      .then((list) => setModels(list))
      .finally(() => setLoading(false));
  };

  const useCases = useMemo(() => {
    const names = (models ?? []).map((r) => r.useCase);
    return [...new Set(names)].sort();
  }, [models]);

  const fitLevels = ["perfect", "good", "marginal", "too_tight"];

  // Client-side sort + filter (including local search by name)
  const allRecs = useMemo(() => {
    if (!models) return [];

    let filtered = models;

    // Local search — filter by model name (and provider, for convenience).
    const q = searchQuery.trim().toLowerCase();
    if (q) {
      filtered = filtered.filter(
        (r) =>
          r.name.toLowerCase().includes(q) ||
          r.provider.toLowerCase().includes(q) ||
          r.hfId.toLowerCase().includes(q),
      );
    }

    // Fit filter.
    if (filterFit) {
      filtered = filtered.filter((r) => r.fitLevel === filterFit);
    }
    // Use-case filter.
    if (filterUseCase) {
      filtered = filtered.filter((r) => r.useCase === filterUseCase);
    }

    // Sort.
    const fn = SORT_FNS[sortKey];
    if (fn) {
      filtered = [...filtered].sort((a, b) => {
        const cmp = fn(a, b);
        return sortDir === "desc" ? -cmp : cmp;
      });
    }
    return filtered;
  }, [models, searchQuery, sortKey, sortDir, filterFit, filterUseCase]);

  // ── tiny local helpers for the detail modal ────────────────────
  const Dt = ({ children }: { children: React.ReactNode }) => (
    <span className="text-tui-fg-dim">{children}</span>
  );
  const Dd = ({
    children,
    mono,
    className,
  }: {
    children: React.ReactNode;
    mono?: boolean;
    className?: string;
  }) => (
    <span
      className={`min-w-0 ${mono ? "font-mono text-[11px]" : "text-tui-fg"} ${className ?? ""}`}
    >
      {children}
    </span>
  );

  return (
    <div>
      {/* ── Hardware mode + Search + Filters ─────────────────────── */}
      <div className="mb-3 flex items-center gap-2">
        {/* GPU / RAM toggle — a scoring input, not a filter. GPU scores
            against the discrete GPU's VRAM; RAM against CPU + iGPU + system
            RAM. The same model gets a different score / tok/s / fit in each. */}
        <div
          className="inline-flex shrink-0 rounded-[4px] border border-tui-border/50 p-0.5"
          role="tablist"
          aria-label="Hardware mode"
        >
          {(
            [
              ["gpu", "GPU", "Score against the discrete GPU (dGPU + RAM)"],
              ["ram", "RAM", "Score against CPU + iGPU + system RAM"],
            ] as [HwMode, string, string][]
          ).map(([m, label, tip]) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={hwMode === m}
              title={tip}
              onClick={() => setHwMode(m)}
              className={
                "rounded-[3px] px-2.5 py-1 text-[11px] font-medium transition-colors " +
                (hwMode === m
                  ? "bg-tui-accent/20 text-tui-accent"
                  : "text-tui-fg-dim hover:text-tui-fg")
              }
            >
              {label}
            </button>
          ))}
        </div>

        {/* Search box — filters the recommendation list locally */}
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search model names…"
          className="h-7 min-w-0 flex-1 rounded-[4px] border border-tui-border/50 bg-transparent px-2.5 text-[12px] text-tui-fg outline-none placeholder:text-tui-fg-dim/60 focus:border-tui-accent/50"
        />
        <select
          value={filterUseCase}
          onChange={(e) => setFilterUseCase(e.target.value)}
          className="h-7 rounded-[4px] border border-tui-border/50 bg-transparent px-2 text-[11px] text-tui-fg outline-none focus:border-tui-accent/50"
        >
          <option value="">All Use Cases</option>
          {useCases.map((uc) => (
            <option key={uc} value={uc}>
              {uc}
            </option>
          ))}
        </select>
        <select
          value={filterFit}
          onChange={(e) => setFilterFit(e.target.value)}
          className="h-7 rounded-[4px] border border-tui-border/50 bg-transparent px-2 text-[11px] text-tui-fg outline-none focus:border-tui-accent/50"
        >
          <option value="">All Fit</option>
          {fitLevels.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
        {/* Quant — a scoring input. Lower quants free up memory and run
            faster, but at lower quality, so score / tok/s / fit all shift. */}
        <select
          value={quant}
          onChange={(e) => setQuant(e.target.value)}
          title="Quantization to score against — changes score, tok/s & fit"
          className="h-7 rounded-[4px] border border-tui-border/50 bg-transparent px-2 font-mono text-[11px] text-tui-fg outline-none focus:border-tui-accent/50"
        >
          {SUPPORTED_QUANTS.map((qq) => (
            <option key={qq} value={qq}>
              {qq}
            </option>
          ))}
        </select>
        {(filterFit ||
          filterUseCase ||
          searchQuery ||
          quant !== DEFAULT_QUANT) && (
          <button
            type="button"
            onClick={() => {
              setFilterFit("");
              setFilterUseCase("");
              setSearchQuery("");
              setQuant(DEFAULT_QUANT);
            }}
            className="text-[10px] text-tui-fg-dim hover:text-tui-fg"
          >
            Clear
          </button>
        )}
        <span className="ml-auto shrink-0 text-[10px] text-tui-fg-dim">
          {loading ? "…" : `${allRecs.length} models`}
        </span>
      </div>

      {/* ── Body ─────────────────────────────────────────────── */}
      {(() => {
        if (loading) {
          return (
            <div className="flex flex-col items-center justify-center gap-3 py-20 text-tui-fg-muted">
              <Spinner size="lg" />
              <p className="text-[13px]">
                Analyzing hardware & scanning model database…
              </p>
            </div>
          );
        }
        if (allRecs.length === 0) {
          return (
            <div className="flex flex-col items-center justify-center gap-2 py-20 text-tui-fg-muted">
              <p className="text-[13px]">
                {models
                  ? "No models match your filters."
                  : "No recommendations available yet."}
              </p>
              <p className="text-[11px]">
                Models are scored against your hardware — check back after the
                catalogue updates.
              </p>
            </div>
          );
        }
        return (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[12px]">
              <thead>
                <tr className="border-b border-tui-border/60 text-left text-[10px] uppercase tracking-wider text-tui-fg-muted">
                  <th className="pb-2 pr-3 font-medium">
                    <Sorter
                      label="Fit"
                      column="fitLevel"
                      sortKey={sortKey}
                      sortDir={sortDir}
                      onClick={handleSort}
                    />
                  </th>
                  <th className="pb-2 pr-3 font-medium">
                    <Sorter
                      label="Model"
                      column="name"
                      sortKey={sortKey}
                      sortDir={sortDir}
                      onClick={handleSort}
                    />
                  </th>
                  <th className="pb-2 pr-3 font-medium">
                    <Sorter
                      label="Param"
                      column="parameterCount"
                      sortKey={sortKey}
                      sortDir={sortDir}
                      onClick={handleSort}
                    />
                  </th>
                  <th className="pb-2 pr-3 font-medium">Quant</th>
                  <th className="pb-2 pr-3 font-medium">
                    <Sorter
                      label={hwMode === "gpu" ? "VRAM" : "RAM"}
                      column="memoryRequiredGb"
                      sortKey={sortKey}
                      sortDir={sortDir}
                      onClick={handleSort}
                    />
                  </th>
                  <th className="pb-2 pr-3 font-medium">
                    <Sorter
                      label="Ctx"
                      column="contextLength"
                      sortKey={sortKey}
                      sortDir={sortDir}
                      onClick={handleSort}
                    />
                  </th>
                  <th className="pb-2 pr-3 font-medium">
                    <Sorter
                      label="Speed"
                      column="estimatedTps"
                      sortKey={sortKey}
                      sortDir={sortDir}
                      onClick={handleSort}
                    />
                  </th>
                  <th className="pb-2 pr-3 font-medium">
                    <Sorter
                      label="Score"
                      column="score"
                      sortKey={sortKey}
                      sortDir={sortDir}
                      onClick={handleSort}
                    />
                  </th>
                  <th className="pb-2 pr-3 font-medium">Mode</th>
                  <th className="pb-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                {allRecs.map((rec) => {
                  const isInstalled = localHfIds.has(rec.hfId);
                  const isLoaded = loadedModelIds.has(rec.hfId);
                  const dl = downloads[rec.hfId];
                  const inFlight =
                    dl &&
                    (dl.state === "pending" ||
                      dl.state === "downloading" ||
                      dl.state === "verifying");

                  return (
                    <tr
                      key={rec.hfId}
                      onClick={() => setDetailModel(rec)}
                      className="cursor-pointer group border-b border-tui-border/30 transition-colors hover:bg-[var(--fluent-bg-subtle)]"
                    >
                      {/* Fit */}
                      <td className="whitespace-nowrap py-2 pr-3">
                        <FitBadge level={rec.fitLevel} />
                      </td>

                      {/* Model */}
                      <td className="max-w-[180px] py-2 pr-3">
                        <div className="flex items-center gap-1">
                          <span
                            className="block truncate font-medium text-tui-fg"
                            title={rec.name}
                          >
                            {rec.name.includes("/")
                              ? rec.name.split("/").slice(1).join("/")
                              : rec.name}
                          </span>
                        </div>
                        <div className="mt-0.5 flex items-center gap-1">
                          {rec.capabilities.includes("vision") && (
                            <span
                              className="shrink-0 rounded-[3px] border border-purple-400/50 bg-purple-500/10 px-1 py-px text-[9px] font-medium text-purple-400"
                              title="Vision-capable"
                            >
                              V
                            </span>
                          )}
                          {rec.capabilities.includes("tool_use") && (
                            <span
                              className="shrink-0 rounded-[3px] border border-amber-400/50 bg-amber-500/10 px-1 py-px text-[9px] font-medium text-amber-400"
                              title="Tool-use capable"
                            >
                              T
                            </span>
                          )}
                          {isInstalled && (
                            <span className="shrink-0 rounded-[3px] border border-tui-accent/40 bg-tui-accent/10 px-1 py-px text-[9px] font-medium text-tui-accent">
                              Local
                            </span>
                          )}
                          {isLoaded && (
                            <span className="inline-flex shrink-0 items-center gap-0.5 rounded-[3px] border border-tui-accent/50 bg-tui-selection px-1 py-px text-[9px] font-medium text-tui-accent">
                              <span className="h-1 w-1 rounded-full bg-tui-accent" />
                              Live
                            </span>
                          )}
                        </div>
                      </td>

                      {/* Param */}
                      <td className="whitespace-nowrap py-2 pr-3 text-tui-fg-muted">
                        {rec.parameterCount}
                      </td>

                      {/* Quant */}
                      <td className="whitespace-nowrap py-2 pr-3 font-mono text-[11px] text-tui-fg-muted">
                        {rec.bestQuant}
                      </td>

                      {/* VRAM / RAM (memory required in the active pool) */}
                      <td className="whitespace-nowrap py-2 pr-3 text-tui-fg-muted">
                        {formatMem(rec.memoryRequiredGb)}
                      </td>

                      {/* Ctx */}
                      <td className="whitespace-nowrap py-2 pr-3 text-tui-fg-muted">
                        {rec.contextLength > 0
                          ? rec.contextLength >= 1000
                            ? `${Math.round(rec.contextLength / 1000)}k`
                            : `${rec.contextLength}`
                          : "—"}
                      </td>

                      {/* Speed */}
                      <td className="whitespace-nowrap py-2 pr-3 text-[11px] text-tui-fg-dim">
                        {rec.estimatedTps > 0
                          ? `${formatTps(rec.estimatedTps)} t/s`
                          : "—"}
                      </td>

                      {/* Score */}
                      <td className="whitespace-nowrap py-2 pr-3 font-medium text-tui-fg">
                        {rec.score.toFixed(1)}
                      </td>

                      {/* Mode (inference engine) */}
                      <td className="whitespace-nowrap py-2 pr-3 text-[11px] text-tui-fg-muted">
                        {engineLabel(rec.inferenceRuntime)}
                      </td>

                      {/* Action */}
                      <td
                        className="py-2 text-right"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {inFlight ? (
                          <div className="flex items-center justify-end gap-1.5">
                            <Spinner size="sm" />
                            <span className="text-[10px] text-tui-fg-dim">
                              {dl.state === "downloading"
                                ? "downloading…"
                                : dl.state === "verifying"
                                  ? "verifying…"
                                  : "pending…"}
                            </span>
                            <TuiButton
                              variant="ghost"
                              size="sm"
                              onClick={() => void onCancel(dl.model_id)}
                            >
                              Cancel
                            </TuiButton>
                          </div>
                        ) : isInstalled ? (
                          <TuiButton
                            variant="ghost"
                            size="sm"
                            onClick={() =>
                              onDownload(
                                rec.hfId,
                                rec as unknown as Record<string, unknown>,
                              )
                            }
                            title="Re-download / update this model"
                          >
                            Re-download
                          </TuiButton>
                        ) : (
                          <TuiButton
                            variant="primary"
                            size="sm"
                            onClick={() =>
                              onDownload(
                                rec.hfId,
                                rec as unknown as Record<string, unknown>,
                              )
                            }
                          >
                            Download
                          </TuiButton>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        );
      })()}

      {/* ── Detail modal ──────────────────────────────────────── */}
      {detailModel && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={(e) => {
            if (e.target === e.currentTarget) setDetailModel(null);
          }}
        >
          <div className="max-h-[90vh] w-full max-w-[520px] overflow-y-auto rounded-[10px] border border-white/10 bg-[#1e1e1e] shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-tui-border/40 px-5 py-3">
              <div className="min-w-0">
                <h2 className="truncate text-[15px] font-semibold text-tui-fg">
                  {detailModel.name}
                </h2>
                <p className="truncate text-[11px] text-tui-fg-dim">
                  {detailModel.provider}
                </p>
              </div>
              <button
                onClick={() => setDetailModel(null)}
                className="ml-3 shrink-0 rounded-[4px] px-2 py-1 text-[11px] text-tui-fg-muted hover:bg-[var(--fluent-bg-subtle-hover)] hover:text-tui-fg"
              >
                ✕
              </button>
            </div>

            {/* Body */}
            <div className="space-y-3 px-5 py-4 text-[12px]">
              <div className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-2">
                <Dt>HF Repo</Dt>
                <Dd mono>{detailModel.hfId}</Dd>

                <Dt>Params</Dt>
                <Dd>{detailModel.parameterCount}</Dd>

                <Dt>Size</Dt>
                <Dd>{detailModel.sizeHint}</Dd>

                <Dt>Quant</Dt>
                <Dd>{detailModel.bestQuant}</Dd>

                <Dt>Context</Dt>
                <Dd>
                  {detailModel.contextLength > 0
                    ? detailModel.contextLength >= 1000
                      ? `${Math.round(detailModel.contextLength / 1000)}K tokens`
                      : `${detailModel.contextLength} tokens`
                    : "—"}
                </Dd>

                <Dt>Use Case</Dt>
                <Dd className="capitalize">{detailModel.useCase}</Dd>

                <Dt>Fit</Dt>
                <Dd>
                  <FitBadge level={detailModel.fitLevel} />
                </Dd>

                <Dt>Score</Dt>
                <Dd>{detailModel.score.toFixed(1)} / 100</Dd>

                <Dt>Speed</Dt>
                <Dd>
                  {detailModel.estimatedTps > 0
                    ? `~${formatTps(detailModel.estimatedTps)} t/s`
                    : "—"}
                </Dd>

                <Dt>{hwMode === "gpu" ? "VRAM" : "RAM"}</Dt>
                <Dd>{formatMem(detailModel.memoryRequiredGb)}</Dd>

                <Dt>Run Mode</Dt>
                <Dd>{detailModel.runMode}</Dd>

                <Dt>Engine</Dt>
                <Dd>{engineLabel(detailModel.inferenceRuntime)}</Dd>

                <Dt>Format</Dt>
                <Dd mono>{detailModel.modelFormat || "—"}</Dd>

                <Dt>Inputs</Dt>
                <Dd>
                  <div className="flex items-center gap-1">
                    {detailModel.inputTypes.length > 0
                      ? detailModel.inputTypes.map((t) => (
                          <span
                            key={t}
                            className="rounded-[3px] border border-tui-border/40 bg-[var(--fluent-bg-subtle)] px-1.5 py-px text-[10px] text-tui-fg-muted"
                          >
                            {t}
                          </span>
                        ))
                      : "—"}
                  </div>
                </Dd>

                {detailModel.capabilities.length > 0 && (
                  <>
                    <Dt>Capabilities</Dt>
                    <Dd>
                      <div className="flex items-center gap-1">
                        {detailModel.capabilities.map((c) => (
                          <span
                            key={c}
                            className="rounded-[3px] border border-tui-border/40 bg-[var(--fluent-bg-subtle)] px-1.5 py-px text-[10px] text-tui-fg-muted"
                          >
                            {c}
                          </span>
                        ))}
                      </div>
                    </Dd>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function FitBadge({ level }: { level: string }) {
  const color =
    level === "perfect"
      ? "text-emerald-400 border-emerald-500/40 bg-emerald-500/10"
      : level === "good"
        ? "text-tui-fg-muted border-tui-border/50 bg-tui-bg"
        : level === "marginal"
          ? "text-yellow-500 border-yellow-500/40 bg-yellow-500/10"
          : "text-red-400 border-red-500/40 bg-red-500/10";
  return (
    <span
      className={`rounded-[3px] border px-1.5 py-px text-[10px] font-medium ${color}`}
    >
      {level}
    </span>
  );
}

// ─── manual / custom model download ───────────────────────────────────

/**
 * Custom download section pinned to the top of the Models page. The repo-id
 * input and Fetch button are always visible; once a repo is fetched the
 * GGUF file checklist appears inline so the user can hand-pick file(s) and
 * download exactly those (bypassing the automatic quant picker).
 */
function ManualDownloadPanel({ onDownloaded }: { onDownloaded: () => void }) {
  const listGgufFiles = useModelsStore((s) => s.listGgufFiles);
  const downloadFiles = useModelsStore((s) => s.downloadFiles);

  const [modelId, setModelId] = useState("");
  const [files, setFiles] = useState<GgufFileInfo[] | null>(null);
  const [fetchedId, setFetchedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onFetch() {
    const id = modelId.trim();
    if (!id || fetching) return;
    setFetching(true);
    setError(null);
    setFiles(null);
    setSelected(new Set());
    try {
      const list = await listGgufFiles(id);
      setFiles(list);
      setFetchedId(id);
    } catch (e) {
      setError(typeof e === "string" ? e : String(e));
    } finally {
      setFetching(false);
    }
  }

  function toggleFile(name: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  async function onDownload() {
    if (!fetchedId || selected.size === 0) return;
    await downloadFiles(fetchedId, [...selected]);
    onDownloaded();
    // Reset back to a clean slate so the panel is ready for the next id.
    setFiles(null);
    setFetchedId(null);
    setSelected(new Set());
    setModelId("");
  }

  const selectedBytes = useMemo(() => {
    if (!files) return 0;
    return files
      .filter((f) => selected.has(f.name))
      .reduce((sum, f) => sum + (f.size || 0), 0);
  }, [files, selected]);

  return (
    <div className="rounded-[10px] border border-tui-border bg-[var(--fluent-bg-subtle)] px-3 py-3">
      <div className="mb-2 flex items-baseline gap-2">
        <span className="text-[12px] font-semibold text-tui-fg">
          Custom download
        </span>
        <span className="text-[11px] text-tui-fg-muted">
          Paste a HuggingFace repo id, then pick the GGUF file(s) to download.
        </span>
      </div>

      {/* Repo id + fetch — always visible */}
      <div className="flex items-center gap-2">
        <TuiInput
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void onFetch();
          }}
          placeholder="e.g. unsloth/Qwen3-8B-GGUF"
          spellCheck={false}
          className="flex-1"
        />
        <TuiButton
          variant="primary"
          onClick={() => void onFetch()}
          disabled={fetching || !modelId.trim()}
        >
          {fetching ? (
            <>
              <Spinner size="sm" /> Fetching…
            </>
          ) : (
            "Fetch"
          )}
        </TuiButton>
      </div>

      {error && (
        <div className="mt-3 rounded-[6px] border border-tui-err/30 bg-[rgba(255,153,164,0.06)] px-3 py-2 text-[11px] text-tui-err">
          {error}
        </div>
      )}

      {/* File picker */}
      {files && files.length > 0 && (
        <div className="mt-3 space-y-3">
          <div className="max-h-[280px] divide-y divide-tui-border overflow-y-auto rounded-[6px] border border-tui-border">
            {files.map((f) => {
              const checked = selected.has(f.name);
              return (
                <label
                  key={f.name}
                  className="flex cursor-pointer items-center gap-2.5 px-3 py-2 text-[11px] transition-colors hover:bg-[var(--fluent-bg-subtle-hover)]"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleFile(f.name)}
                    className="h-3.5 w-3.5 shrink-0 accent-tui-accent"
                  />
                  <span className="min-w-0 flex-1 truncate font-mono text-tui-fg">
                    {f.name}
                  </span>
                  {f.kind !== "main" && (
                    <span className="shrink-0 rounded-[3px] border border-tui-border px-1.5 py-px text-[10px] text-tui-fg-muted">
                      {f.kind}
                    </span>
                  )}
                  {f.quant && (
                    <span className="shrink-0 rounded-[3px] border border-tui-accent/40 bg-tui-accent/10 px-1.5 py-px text-[10px] text-tui-accent">
                      {f.quant}
                    </span>
                  )}
                  <span className="w-20 shrink-0 text-right tabular-nums text-tui-fg-muted">
                    {f.size > 0 ? bytes(f.size) : "—"}
                  </span>
                </label>
              );
            })}
          </div>

          <div className="flex items-center justify-between gap-3">
            <span className="text-[11px] text-tui-fg-muted">
              {selected.size === 0
                ? `${fetchedId} · select one or more files`
                : `${selected.size} file${
                    selected.size === 1 ? "" : "s"
                  } · ${bytes(selectedBytes)}`}
            </span>
            <TuiButton
              variant="primary"
              onClick={() => void onDownload()}
              disabled={selected.size === 0}
            >
              Download selected
            </TuiButton>
          </div>
        </div>
      )}

      {files && files.length === 0 && (
        <div className="mt-3 text-[11px] text-tui-fg-muted">
          No .gguf files found in this repo.
        </div>
      )}
    </div>
  );
}
