import { TuiButton } from "@/components/tui/Button";
import { ProgressBar } from "@/components/tui/ProgressBar";
import { Spinner } from "@/components/tui/Spinner";
import {
  useLlamaStore,
  VARIANT_DISPLAY,
  VARIANT_PORTS,
  VARIANT_SLUGS,
  type VariantSlug,
} from "@/stores/llama";
import { useSettingsStore } from "@/stores/settings";
import { useSystemStore } from "@/stores/system";
import { bytes } from "@/lib/format";
import { useState, type ReactNode } from "react";

/**
 * llama.cpp orchestrator lifecycle + host-system building blocks used by
 * the Settings page. Exported as separate pieces so the redesigned
 * Settings can place each block in the appropriate section.
 */

// ─── status tone helpers ─────────────────────────────────────────────

type StatusTone = "ok" | "warn" | "err" | "muted";

function toneClass(t: StatusTone): string {
  return {
    ok: "bg-tui-ok",
    warn: "bg-tui-warn",
    err: "bg-tui-err",
    muted: "bg-tui-fg-muted",
  }[t];
}

function statusTone(status: string): StatusTone {
  switch (status) {
    case "running":
      return "ok";
    case "starting":
    case "installing":
    case "stopping":
      return "warn";
    case "error":
      return "err";
    default:
      return "muted";
  }
}

/** Colored dot + status label. */
export function StatusDot({
  tone,
  label,
  className = "",
}: {
  tone: StatusTone;
  label?: ReactNode;
  className?: string;
}) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      <span
        className={`inline-block h-2 w-2 rounded-full ${toneClass(tone)}`}
      />
      {label != null && (
        <span className="text-[11px] text-tui-fg-dim">{label}</span>
      )}
    </span>
  );
}

// ─── variant status card ──────────────────────────────────────────────

function Fact({ label, value }: { label: string; value: ReactNode }) {
  return (
    <>
      <dt className="text-tui-fg-muted">{label}</dt>
      <dd className="min-w-0 truncate text-tui-fg">{value}</dd>
    </>
  );
}

/**
 * Collapsible status card for a single llama.cpp variant. The header strip
 * (always visible) carries the variant name, status dot/label and badges;
 * clicking it expands or hides the install state, version facts and
 * start/stop controls. Defaults to expanded for the active variant.
 */
function VariantCard({ slug }: { slug: VariantSlug }) {
  const orchInfo = useLlamaStore((s) => s.info);
  const storeInstall = useLlamaStore((s) => s.installVariant);
  const storeUpdate = useLlamaStore((s) => s.updateVariant);
  const storeStart = useLlamaStore((s) => s.start);
  const storeStop = useLlamaStore((s) => s.stop);
  const storeSwitch = useLlamaStore((s) => s.switchVariant);
  const defaultModel = useSettingsStore((s) => s.default_model);

  const instance = orchInfo?.instances[slug];
  const isActive = orchInfo?.active_variant === slug;

  const status = instance?.status ?? "not_installed";
  const running = status === "running";
  const installing = status === "installing";
  const starting = status === "starting";
  const stopping = status === "stopping";
  const installed = status !== "not_installed" && status !== "installing";
  const updateAvailable = installed && !!instance?.update_available;
  const displayName = VARIANT_DISPLAY[slug] ?? slug;
  const port = VARIANT_PORTS[slug] ?? 0;

  const startModel = instance?.loaded_model ?? defaultModel ?? "";

  // Expand the active variant by default so its controls are one glance
  // away; everything else starts collapsed to keep the list scannable.
  const [open, setOpen] = useState(isActive);

  return (
    <div className="overflow-hidden rounded-[8px] border border-tui-border bg-[var(--fluent-bg-subtle)]">
      {/* Header strip — always visible, toggles the body */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left transition-colors hover:bg-[var(--fluent-bg-subtle-hover)]"
      >
        <Chevron open={open} />
        <StatusDot tone={statusTone(status)} />
        <span className="text-[13px] font-semibold text-tui-fg">
          {displayName}
        </span>
        {isActive && (
          <span className="rounded-[3px] bg-tui-accent/20 px-1.5 py-0.5 text-[10px] font-medium text-tui-accent">
            active
          </span>
        )}
        {updateAvailable && (
          <span
            className="rounded-[3px] bg-tui-warn/20 px-1.5 py-0.5 text-[10px] font-medium text-tui-warn"
            title={`Update available: ${instance?.installed_version ?? "?"} → ${instance?.latest_version ?? "?"}`}
          >
            update
          </span>
        )}
        <span className="ml-auto text-[11px] capitalize text-tui-fg-muted">
          {status.replace("_", " ")}
        </span>
      </button>

      {/* Collapsible body */}
      {open && (
        <div className="space-y-3 border-t border-tui-border px-3 py-3">
          {/* Status hero */}
          <div className="flex items-center justify-between gap-3 rounded-[8px] border border-tui-border bg-tui-bg px-3 py-2.5">
            <div className="flex min-w-0 items-center gap-2.5">
              <div className="min-w-0">
                <div className="text-[13px] font-semibold capitalize text-tui-fg">
                  {status.replace("_", " ")}
                </div>
                <div className="truncate text-[11px] text-tui-fg-muted">
                  {instance?.base_url ?? "—"}
                </div>
              </div>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-1.5">
              {!installed && !installing && (
                <TuiButton
                  variant="primary"
                  onClick={() => void storeInstall(slug)}
                >
                  Install
                </TuiButton>
              )}
              {installing && (
                <TuiButton disabled>
                  <Spinner size="sm" /> Installing…
                </TuiButton>
              )}
              {installed && !isActive && (
                <TuiButton
                  variant="primary"
                  size="sm"
                  onClick={() => void storeSwitch(slug)}
                >
                  Set active
                </TuiButton>
              )}
              {updateAvailable && (
                <TuiButton
                  variant="primary"
                  size="sm"
                  disabled={installing}
                  onClick={() => void storeUpdate(slug)}
                  title={`Update to ${instance?.latest_version ?? "latest"}${
                    running ? " (stops the running server)" : ""
                  }`}
                >
                  Update → {instance?.latest_version ?? "latest"}
                </TuiButton>
              )}
              {!running && !stopping && installed && isActive && (
                <TuiButton
                  variant="primary"
                  size="sm"
                  disabled={starting}
                  onClick={() => void storeStart(slug, startModel || undefined)}
                  title={
                    starting
                      ? "llama-server is starting…"
                      : startModel
                        ? `Start ${displayName} with ${startModel}`
                        : "Start server idle (no model loaded)"
                  }
                >
                  {starting ? (
                    <>
                      <Spinner size="sm" /> Starting…
                    </>
                  ) : (
                    "Start"
                  )}
                </TuiButton>
              )}
              {(running || stopping) && isActive && (
                <TuiButton
                  variant="danger"
                  size="sm"
                  disabled={stopping}
                  onClick={() => void storeStop(slug)}
                >
                  {stopping ? (
                    <>
                      <Spinner size="sm" /> Stopping…
                    </>
                  ) : (
                    "Stop"
                  )}
                </TuiButton>
              )}
            </div>
          </div>

          {/* Facts grid */}
          {installed && (
            <dl className="grid grid-cols-[7.5rem_1fr] gap-x-3 gap-y-1.5 text-[11px]">
              <Fact
                label="Version"
                value={instance?.installed_version ?? "—"}
              />
              {updateAvailable && (
                <Fact
                  label="Latest"
                  value={
                    <span className="text-tui-warn">
                      {instance?.latest_version ?? "—"} (update available)
                    </span>
                  }
                />
              )}
              <Fact label="Port" value={String(port)} />
              <Fact label="PID" value={instance?.pid ?? "—"} />
              <Fact
                label="Loaded model"
                value={instance?.loaded_model ?? "—"}
              />
            </dl>
          )}

          {instance?.last_error && (
            <div className="rounded-[6px] border border-tui-err/30 bg-[rgba(255,153,164,0.06)] px-3 py-2 text-[11px] text-tui-err">
              {instance.last_error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Disclosure chevron that rotates when its row is expanded. */
function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      aria-hidden="true"
      className={`shrink-0 text-tui-fg-muted transition-transform duration-150 ${
        open ? "rotate-90" : ""
      }`}
    >
      <path
        d="M4 2l4 4-4 4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * Full orchestrator status surface: a collapsible card per llama.cpp
 * variant (all variants are listed so any can be installed on demand),
 * each showing status, version facts, and install/start/stop controls.
 */
export function LlamaStatusCard() {
  const orchInfo = useLlamaStore((s) => s.info);

  // Hide variants the host can't use (e.g. CUDA on a machine with no
  // NVIDIA GPU). The backend reports which builds are usable for the
  // detected hardware; until that's known we show everything. A variant
  // the user has already installed or activated is always kept visible
  // so they can still manage it even if the hardware wouldn't pick it.
  const applicable = orchInfo?.applicable_variants;
  const visibleSlugs = VARIANT_SLUGS.filter((slug) => {
    if (!applicable) return true;
    if (applicable.includes(slug)) return true;
    const inst = orchInfo?.instances[slug];
    return inst != null && inst.status !== "not_installed";
  });

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        {visibleSlugs.map((slug) => (
          <VariantCard key={slug} slug={slug} />
        ))}
      </div>
    </div>
  );
}

// ─── install progress ────────────────────────────────────────────────

/** Live download/extract progress shown while a llama.cpp install is running. */
export function LlamaInstallProgressCard() {
  const progress = useLlamaStore((s) => s.installProgress);
  if (!progress) return null;
  const variantLabel =
    VARIANT_DISPLAY[progress.variant as VariantSlug] ?? progress.variant;
  return (
    <div className="space-y-1.5 rounded-[8px] border border-tui-border bg-[var(--fluent-bg-subtle)] px-3 py-2.5 text-[11px]">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-tui-fg">
          Installing {variantLabel}
        </span>
        <span className="text-tui-fg-muted">
          {progress.stage.replace("_", " ")}
        </span>
      </div>
      <div className="truncate text-tui-fg-dim">{progress.message}</div>
      <ProgressBar
        value={progress.percent}
        max={1}
        label={
          progress.bytes_total
            ? `${bytes(progress.bytes_done)} / ${bytes(progress.bytes_total)}`
            : `${Math.round(progress.percent * 100)}%`
        }
        tone={
          progress.stage === "error"
            ? "err"
            : progress.stage === "done"
              ? "ok"
              : "default"
        }
      />
    </div>
  );
}

/** @deprecated Use LlamaInstallProgressCard instead. */
export const InstallProgressCard = LlamaInstallProgressCard;

// ─── host system specs ──────────────────────────────────────────

/** Read-only hardware snapshot for the System settings section. */
export function SystemSpecsCard() {
  const specs = useSystemStore((s) => s.specs);

  if (!specs) {
    return (
      <span className="inline-flex items-center gap-2 text-[12px] text-tui-fg-muted">
        <Spinner size="sm" /> Probing host…
      </span>
    );
  }

  // ISO-8601 → locale string. Falls back to the raw value if Date can't
  // parse (sysinfo on exotic hosts has been known to emit odd formats).
  const probedAt = (() => {
    const t = Date.parse(specs.probed_at);
    return Number.isNaN(t) ? specs.probed_at : new Date(t).toLocaleString();
  })();

  return (
    <dl className="grid grid-cols-[7.5rem_1fr] gap-x-3 gap-y-1.5 text-[12px]">
      <Fact label="OS" value={`${specs.os} ${specs.os_version}`} />
      <Fact label="Architecture" value={specs.arch} />
      <Fact label="CPU" value={specs.cpu_brand || "—"} />
      <Fact label="CPU vendor" value={specs.cpu_vendor || "—"} />
      <Fact
        label="Cores"
        value={`${specs.cpu_physical_cores} physical · ${specs.cpu_logical_cores} logical`}
      />
      <Fact label="RAM total" value={bytes(specs.ram_total_mb * 1024 * 1024)} />
      {specs.gpus.length === 0 && <Fact label="GPU" value="none detected" />}
      {specs.gpus.map((g, i) => (
        <Fact
          key={`gpu-${i}`}
          label={`GPU ${i}`}
          value={
            <span className="block min-w-0">
              <span className="block truncate">{g.name || "—"}</span>
              <span className="block truncate text-[11px] text-tui-fg-muted">
                {[
                  g.kind,
                  g.vendor || null,
                  g.vram_mb != null
                    ? `${bytes(g.vram_mb * 1024 * 1024)} VRAM`
                    : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            </span>
          }
        />
      ))}
      {specs.npus.length === 0 && <Fact label="NPU" value="none detected" />}
      {specs.npus.map((n, i) => (
        <Fact
          key={`npu-${i}`}
          label={`NPU ${i}`}
          value={
            <span className="block min-w-0">
              <span className="block truncate">{n.name || "—"}</span>
              {n.vendor && (
                <span className="block truncate text-[11px] text-tui-fg-muted">
                  {n.vendor}
                </span>
              )}
            </span>
          }
        />
      ))}
      <Fact label="Probed at" value={probedAt} />
    </dl>
  );
}
