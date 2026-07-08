import { useEffect, useState, type ReactNode } from "react";
import {
  enable as enableAutostart,
  disable as disableAutostart,
  isEnabled as isAutostartEnabled,
} from "@tauri-apps/plugin-autostart";
import { invoke, on } from "@/lib/tauri";
import { TuiInput, TuiTextarea } from "@/components/tui/Input";
import { Toggle } from "@/components/tui/Toggle";
import { Spinner } from "@/components/tui/Spinner";
import { useSettingsStore, type LlamaSettings } from "@/stores/settings";
import { useSystemStore } from "@/stores/system";
import { useLlamaStore } from "@/stores/llama";
import { useModelsStore } from "@/stores/models";
import { useUiStore, type SettingsSection } from "@/stores/ui";
import {
  WHISPER_MODELS,
  TTS_MODELS,
  WAVTOKENIZER,
  recommendWhisperModel,
  recommendTtsModel,
  whisperModelByFile,
  ttsModelByHfId,
  STT_LANGUAGES,
  type WhisperModelOption,
  type TtsModelOption,
} from "@/lib/audioModels";
import { MemoryView } from "@/pages/Memory";
import { ToolsView } from "@/pages/Tools";
import { SkillsView } from "@/pages/Skills";
import { HooksView } from "@/pages/Hooks";
import { AgentsMdView } from "@/pages/AgentsMd";
import {
  LlamaInstallProgressCard,
  LlamaStatusCard,
  SystemSpecsCard,
} from "@/components/server/ServerControls";

// ─── section navigation ──────────────────────────────────────────────

interface Section {
  id: SettingsSection;
  label: string;
  hint: string;
  icon: ReactNode;
}

const SECTIONS: Section[] = [
  {
    id: "general",
    label: "General",
    hint: "Agent behaviour and startup",
    icon: <SlidersIcon />,
  },
  {
    id: "llama",
    label: "Local LLM",
    hint: "llama.cpp server status",
    icon: <ServerIcon />,
  },
  {
    id: "audio",
    label: "Audio",
    hint: "Voice input & read-aloud",
    icon: <AudioIcon />,
  },
  {
    id: "memory",
    label: "Memory",
    hint: "Long-term knowledge store",
    icon: <MemoryIcon />,
  },
  {
    id: "tools",
    label: "Tools",
    hint: "MCP servers and built-ins",
    icon: <ToolsIcon />,
  },
  {
    id: "skills",
    label: "Skills",
    hint: "Reusable prompt packages",
    icon: <SkillsIcon />,
  },
  {
    id: "hooks",
    label: "Hooks",
    hint: "Lifecycle tool hooks",
    icon: <HookIcon />,
  },
  {
    id: "agents_md",
    label: "Instructions",
    hint: "AGENTS.md context file",
    icon: <DocIcon />,
  },
  {
    id: "system",
    label: "System",
    hint: "Host hardware snapshot",
    icon: <ChipIcon />,
  },
];

/** Sections that embed a former top-level page. These render full-width
 * and manage their own height/scroll, so we skip the narrow centered
 * column the native settings sections use. */
const PAGE_SECTIONS = new Set<SettingsSection>([
  "memory",
  "tools",
  "skills",
  "hooks",
  "agents_md",
]);

// ─── page shell ──────────────────────────────────────────────────────

export function SettingsView() {
  const section = useUiStore((s) => s.settingsSection);
  const setSection = useUiStore((s) => s.setSettingsSection);
  const isPageSection = PAGE_SECTIONS.has(section);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 gap-4">
      {/* Section rail */}
      <nav
        className="w-[210px] shrink-0 self-start"
        aria-label="Settings sections"
      >
        <ul className="space-y-0.5">
          {SECTIONS.map((s) => {
            const active = s.id === section;
            return (
              <li key={s.id}>
                <button
                  onClick={() => setSection(s.id)}
                  aria-current={active ? "page" : undefined}
                  className={
                    "group relative flex w-full items-center gap-2.5 rounded-[6px] py-2 pl-3 pr-2.5 " +
                    "text-left text-[12px] transition-colors duration-150 " +
                    (active
                      ? "bg-[var(--fluent-bg-subtle-selected)] text-tui-fg"
                      : "text-tui-fg-dim hover:bg-[var(--fluent-bg-subtle-hover)] hover:text-tui-fg")
                  }
                >
                  {active && (
                    <span className="absolute left-0 top-1/2 h-[55%] w-[3px] -translate-y-1/2 rounded-full bg-tui-accent" />
                  )}
                  <span
                    className={
                      "flex h-5 w-5 items-center justify-center " +
                      (active ? "text-tui-accent" : "text-tui-fg-muted")
                    }
                    aria-hidden
                  >
                    {s.icon}
                  </span>
                  <span className="flex min-w-0 flex-col leading-tight">
                    <span className="truncate font-medium">{s.label}</span>
                    <span className="truncate text-[10.5px] text-tui-fg-muted">
                      {s.hint}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Section content */}
      {isPageSection ? (
        // Embedded former-page sections own their full-width layout and
        // internal scrolling — don't wrap them in the narrow column.
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {section === "memory" && <MemoryView />}
          {section === "tools" && <ToolsView />}
          {section === "skills" && <SkillsView />}
          {section === "hooks" && <HooksView />}
          {section === "agents_md" && <AgentsMdView />}
        </div>
      ) : (
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto pr-1">
          <div className="mx-auto w-full max-w-[760px] space-y-4 pb-6">
            {section === "general" && <GeneralSection />}
            {section === "llama" && <LlamaSection />}
            {section === "audio" && <AudioSection />}
            {section === "system" && <SystemSection />}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── reusable section primitives ────────────────────────────────────

/** Card with a small section title strip and any children stacked inside. */
function SettingsGroup({
  title,
  description,
  action,
  children,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="fluent-mica overflow-hidden rounded-[10px] border border-tui-border shadow-[var(--fluent-shadow-2)]">
      <header className="flex items-baseline justify-between gap-3 border-b border-tui-border bg-[rgba(255,255,255,0.022)] px-4 py-2.5">
        <div className="min-w-0">
          <h3 className="text-[12px] font-semibold text-tui-fg">{title}</h3>
          {description && (
            <p className="mt-0.5 text-[11px] text-tui-fg-muted">
              {description}
            </p>
          )}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </header>
      <div className="px-4 py-1.5">{children}</div>
    </section>
  );
}

/** Horizontal label/control row with hairline separators between siblings. */
function Field({
  label,
  hint,
  control,
  align = "center",
}: {
  label: string;
  hint?: ReactNode;
  control: ReactNode;
  align?: "center" | "start";
}) {
  return (
    <div
      className={
        "flex justify-between gap-6 border-b border-tui-border py-3 last:border-b-0 " +
        (align === "start" ? "items-start" : "items-center")
      }
    >
      <div className="min-w-0 flex-1">
        <div className="text-[12px] text-tui-fg">{label}</div>
        {hint && (
          <div className="mt-0.5 text-[11px] leading-snug text-tui-fg-muted">
            {hint}
          </div>
        )}
      </div>
      <div className="flex shrink-0 items-center">{control}</div>
    </div>
  );
}

/** Centered banner / call-out box for non-actionable explanations. */
function Callout({
  tone = "info",
  children,
}: {
  tone?: "info" | "warn";
  children: ReactNode;
}) {
  const cls =
    tone === "warn"
      ? "border-tui-warn/30 bg-[rgba(252,225,0,0.04)] text-tui-warn"
      : "border-tui-border bg-[var(--fluent-bg-subtle)] text-tui-fg-dim";
  return (
    <div className={`rounded-[6px] border px-3 py-2 text-[11px] ${cls}`}>
      {children}
    </div>
  );
}

// ─── general section ─────────────────────────────────────────

/**
 * Reconcile the OS-level autostart registration with the desired state.
 * No-ops outside the Tauri runtime (e.g. a browser dev preview) and
 * swallows errors so a flaky registry/launch-agent write never blocks
 * the settings save.
 */
async function syncAutostart(enabled: boolean) {
  if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) {
    return;
  }
  try {
    const already = await isAutostartEnabled();
    if (enabled && !already) await enableAutostart();
    else if (!enabled && already) await disableAutostart();
  } catch (e) {
    console.error("autostart sync failed", e);
  }
}

/**
 * Plain-language summary of the hardware-driven startup policy that
 * runs in `AppState::init`. The toggles below it look like they fully
 * control auto-provisioning, but the policy short-circuits them when
 * a discrete GPU is detected (llama.cpp wins automatically). The
 * callout makes that override visible so users don't conclude the
 * toggles are broken when their dGPU host quietly downloads llama.cpp
 * even though `auto_provision_llama` is off.
 */
function StartupPolicyCallout() {
  const specs = useSystemStore((s) => s.specs);

  if (!specs) {
    return (
      <Callout>
        <span className="inline-flex items-center gap-2">
          <Spinner size="sm" /> Detecting hardware…
        </span>
      </Callout>
    );
  }

  const discrete = specs.gpus.filter((g) => g.kind === "discrete");
  const hasDgpu = discrete.length > 0;
  const primary = discrete[0];
  // Match the vendor → variant mapping in `crate::llama::variant`.
  // The user sees the human-friendly build name; the rust side picks
  // the matching release asset (`cuda-12.4` / `sycl` / `hip-radeon`).
  const variantHint = (() => {
    if (!primary) return null;
    const v = primary.vendor.toLowerCase();
    if (v.includes("nvidia")) return "CUDA";
    if (v.includes("intel")) return "SYCL";
    if (v.includes("amd") || v.includes("advanced micro")) return "HIP";
    return null;
  })();

  if (hasDgpu) {
    return (
      <Callout>
        <div className="flex items-start gap-2">
          <span
            className="mt-[3px] inline-block h-2 w-2 shrink-0 rounded-full bg-tui-accent"
            aria-hidden
          />
          <div className="min-w-0">
            <div className="font-semibold text-tui-fg">
              Discrete GPU detected
              {variantHint && (
                <span className="ml-1.5 text-tui-fg-muted">
                  · {variantHint} build
                </span>
              )}
            </div>
            <div className="mt-0.5">
              {primary && (
                <span className="truncate text-tui-fg">{primary.name}</span>
              )}
              <span className="block leading-snug">
                llama.cpp is auto-installed at startup, regardless of the
                toggles below.
              </span>
            </div>
          </div>
        </div>
      </Callout>
    );
  }

  return (
    <Callout>
      <div className="flex items-start gap-2">
        <span
          className="mt-[3px] inline-block h-2 w-2 shrink-0 rounded-full bg-tui-fg-muted"
          aria-hidden
        />
        <div className="min-w-0">
          <div className="font-semibold text-tui-fg">
            No discrete GPU detected
          </div>
          <div className="mt-0.5 leading-snug">
            llama.cpp can be auto-installed at startup by toggling its
            auto-provision option below.
          </div>
        </div>
      </div>
    </Callout>
  );
}

function GeneralSection() {
  const s = useSettingsStore();
  const loaded = useSettingsStore((st) => st.loaded);

  // Reconcile the OS-level autostart entry with the saved preference once
  // settings have loaded. Covers the case where the entry was removed
  // out-of-band (e.g. another install) but the toggle still reads "on".
  useEffect(() => {
    if (!loaded) return;
    void syncAutostart(s.autostart_enabled);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded]);

  function setAutostart(enabled: boolean) {
    void s.save({ autostart_enabled: enabled });
    void syncAutostart(enabled);
  }

  return (
    <>
      <SettingsGroup
        title="Window & startup"
        description="How zero launches and what the window's close button does."
      >
        <Field
          label="Start zero on system login"
          hint="Launch zero automatically when you sign in to your computer."
          control={
            <Toggle
              checked={s.autostart_enabled}
              onChange={setAutostart}
              label="Start on login"
            />
          }
        />
        <Field
          label="Start minimized"
          hint="Open straight to the taskbar instead of focusing the window on launch."
          control={
            <Toggle
              checked={s.minimize_on_startup}
              onChange={(v) => void s.save({ minimize_on_startup: v })}
              label="Start minimized"
            />
          }
        />
        <Field
          label="Close button minimizes to tray"
          hint="Keep zero running in the background when you press the window's close button instead of quitting. Restore it from the system-tray icon (click or Show menu)."
          control={
            <Toggle
              checked={s.close_to_taskbar}
              onChange={(v) => void s.save({ close_to_taskbar: v })}
              label="Minimize to tray on close"
            />
          }
        />
      </SettingsGroup>

      <SettingsGroup
        title="Agent behaviour"
        description="Controls how the agent plans, calls tools, and protects you from itself."
      >
        <Field
          label="Confirm destructive tools"
          hint="Prompt before any tool flagged as destructive (file deletes, shell writes…) runs."
          control={
            <Toggle
              checked={s.destructive_tool_confirm}
              onChange={(v) => void s.save({ destructive_tool_confirm: v })}
              label="Confirm destructive tools"
            />
          }
        />
        <Field
          label="Max iterations"
          hint="Upper bound on tool-call / response cycles per turn before the agent gives up."
          control={
            <TuiInput
              type="number"
              min={1}
              value={s.agent_max_iterations}
              onChange={(e) =>
                void s.save({
                  agent_max_iterations: Number(e.target.value) || 1,
                })
              }
              className="w-24"
            />
          }
        />
        <Field
          label="Lazy tool discovery"
          hint={
            <>
              Ship only the{" "}
              <code className="rounded bg-[var(--fluent-bg-subtle-pressed)] px-1 font-mono text-tui-fg">
                tools.list
              </code>{" "}
              built-in to the model on the first round of each turn. The model
              calls it on demand to learn what else is available; the rest of
              the catalogue is exposed automatically on the next round. Saves a
              lot of initial context when many MCP servers are wired up, at the
              cost of one extra round-trip on turns that need a tool.
            </>
          }
          control={
            <Toggle
              checked={s.lazy_tool_discovery}
              onChange={(v) => void s.save({ lazy_tool_discovery: v })}
              label="Lazy tool discovery"
            />
          }
        />
      </SettingsGroup>
    </>
  );
}

// ─── local LLM section ────────────────────────────────────────────

/**
 * Settings surface for the bundled `llama-server` runtime. Lifecycle
 * hero + facts on top, install progress strip, then the runtime knobs
 * that get baked into the next `llama-server` argv. Edits are saved
 * through `useSettingsStore` so they round-trip to disk.
 */
function LlamaSection() {
  const s = useSettingsStore();
  // Status of the active variant's instance, used to surface the
  // "restart to apply" hint while a server is live.
  const llamaStatus = useLlamaStore((st) =>
    st.info
      ? (st.info.instances[st.info.active_variant]?.status ?? null)
      : null,
  );
  const llamaRunning = llamaStatus === "running" || llamaStatus === "starting";

  function patchLlama(patch: Partial<LlamaSettings>) {
    void s.save({ llama: { ...s.llama, ...patch } });
  }

  const [extraDraft, setExtraDraft] = useState(() =>
    (s.llama.extra_args ?? []).join(" "),
  );

  // Keep the textarea in sync if another surface mutates `extra_args`
  // (e.g. a settings reload). Cheap join() so doing it on every render
  // is fine.
  useEffect(() => {
    setExtraDraft((s.llama.extra_args ?? []).join(" "));
  }, [s.llama.extra_args]);

  function commitExtraArgs() {
    const tokens = extraDraft
      .split(/\s+/)
      .map((t) => t.trim())
      .filter(Boolean);
    patchLlama({ extra_args: tokens });
  }

  return (
    <>
      <SettingsGroup
        title="Startup"
        description="What zero does with llama.cpp the first time you launch it on a fresh machine."
      >
        <StartupPolicyCallout />
        <Field
          label="Auto-install & start llama.cpp on launch"
          hint="Off by default. When a discrete GPU is detected at startup, zero auto-installs llama.cpp anyway since the GPU variants — CUDA / SYCL / HIP — are the right local runtime for that hardware. Turn this on to force the install on hosts without a dGPU too."
          control={
            <Toggle
              checked={s.auto_provision_llama}
              onChange={(v) => void s.save({ auto_provision_llama: v })}
              label="Auto-provision llama.cpp"
            />
          }
        />
      </SettingsGroup>
      <div className="py-2">
        <LlamaStatusCard />
      </div>

      <LlamaInstallProgressCard />

      <SettingsGroup
        title="Runtime"
        description="Bind address and core inference knobs applied the next time llama-server (re)starts."
        action={
          llamaRunning ? (
            <span className="text-[11px] text-tui-warn">restart to apply</span>
          ) : undefined
        }
      >
        <Field
          label="Host"
          hint="Interface llama-server binds to. 127.0.0.1 keeps it local-only."
          control={
            <TuiInput
              value={s.llama.host}
              onChange={(e) => patchLlama({ host: e.target.value })}
              className="w-40"
            />
          }
        />
        <Field
          label="GPU layers"
          hint={
            <>
              <code className="rounded bg-[var(--fluent-bg-subtle-pressed)] px-1 font-mono text-tui-fg">
                --n-gpu-layers
              </code>
              . −1 offloads every layer to the GPU (recommended for the
              GPU-enabled builds); 0 forces CPU-only inference.
            </>
          }
          control={
            <TuiInput
              type="number"
              min={-1}
              value={s.llama.n_gpu_layers}
              onChange={(e) =>
                patchLlama({ n_gpu_layers: Number(e.target.value) })
              }
              className="w-24"
            />
          }
        />
        <Field
          label="Context size"
          hint={
            <>
              <code className="rounded bg-[var(--fluent-bg-subtle-pressed)] px-1 font-mono text-tui-fg">
                --ctx-size
              </code>
              . 0 keeps the model's training context window.
            </>
          }
          control={
            <TuiInput
              type="number"
              min={0}
              value={s.llama.ctx_size}
              onChange={(e) =>
                patchLlama({ ctx_size: Number(e.target.value) || 0 })
              }
              className="w-28"
            />
          }
        />
        <Field
          label="Parallel slots"
          hint={
            <>
              <code className="rounded bg-[var(--fluent-bg-subtle-pressed)] px-1 font-mono text-tui-fg">
                --parallel
              </code>
              . Number of concurrent decode slots. Increase only if you actually
              run multiple chats at once — each slot eats VRAM.
            </>
          }
          control={
            <TuiInput
              type="number"
              min={1}
              value={s.llama.parallel}
              onChange={(e) =>
                patchLlama({ parallel: Number(e.target.value) || 1 })
              }
              className="w-24"
            />
          }
        />
        <Field
          label="Extra args"
          hint="Whitespace-separated argv tokens forwarded verbatim to llama-server. Escape hatch for any flag this UI doesn't model."
          control={
            <TuiTextarea
              value={extraDraft}
              onChange={(e) => setExtraDraft(e.target.value)}
              onBlur={commitExtraArgs}
              placeholder="--threads 8 --flash-attn"
              rows={2}
              className="w-full"
            />
          }
        />
      </SettingsGroup>

      <SettingsGroup
        title="Experimental"
        description="Bleeding-edge runtime features. These can break model loading — leave them off unless you know you want them."
      >
        <Field
          label="Multi-token prediction (MTP)"
          align="start"
          hint={
            <>
              <span className="mr-1.5 rounded-[3px] border border-tui-warn/40 bg-tui-warn/10 px-1.5 py-px text-[10px] font-medium text-tui-warn">
                experimental
              </span>
              Wire downloaded MTP / speculative-decoding draft models into the
              runtime to speed up generation. Disabled by default: MTP drafts
              can crash or cause a model to fail to load on some llama.cpp build
              / GPU combinations. Takes effect the next time a model is loaded.
            </>
          }
          control={
            <Toggle
              checked={s.llama.mtp_enabled}
              onChange={(v) => patchLlama({ mtp_enabled: v })}
              label="Enable MTP"
            />
          }
        />
      </SettingsGroup>
    </>
  );
}

// ─── audio section ───────────────────────────────────────────────────

/**
 * Speech-to-text + text-to-speech surface. Off by default.
 *
 * Speech-to-text uses an audio-input GGUF model loaded into the llama.cpp
 * router alongside the chat model. Flipping audio on with no STT model
 * downloaded opens a confirm modal that recommends a model sized for this
 * machine and (on confirm) downloads + loads it.
 *
 * Text-to-speech runs in the browser via the Web Speech API (llama.cpp has
 * no TTS endpoint), so it needs no model — only an optional voice pick.
 */
/** Mirror of the Rust `WhisperStatus`. */
interface WhisperStatus {
  runtime_installed: boolean;
  runtime_version: string | null;
  gpu: boolean;
  models: string[];
}

/** Mirror of the Rust `WhisperProgress` (whisper://progress event). */
interface WhisperProgress {
  stage: string;
  message: string;
  bytes_done: number;
  bytes_total: number | null;
  percent: number;
  target: string;
}

function AudioSection() {
  const audio = useSettingsStore((st) => st.audio);
  const save = useSettingsStore((st) => st.save);
  const specs = useSystemStore((st) => st.specs);
  const probe = useSystemStore((st) => st.probe);
  const local = useModelsStore((st) => st.local);
  const refreshLocal = useModelsStore((st) => st.refreshLocal);
  const download = useModelsStore((st) => st.download);

  const [whisper, setWhisper] = useState<WhisperStatus | null>(null);
  const [progress, setProgress] = useState<WhisperProgress | null>(null);
  const [confirm, setConfirm] = useState<{
    stt: WhisperModelOption;
    tts: TtsModelOption;
  } | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const refreshWhisper = async () => {
    try {
      setWhisper(await invoke<WhisperStatus>("whisper_status"));
    } catch (e) {
      console.error("whisper_status failed", e);
    }
  };

  useEffect(() => {
    void refreshLocal();
    void refreshWhisper();
    if (!specs) void probe();
    let un: (() => void) | undefined;
    void on<WhisperProgress>("whisper://progress", (p) => {
      setProgress(p.stage === "done" ? null : p);
      if (p.stage === "done") void refreshWhisper();
    }).then((u) => (un = u));
    return () => un?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const hfDownloaded = (id: string | null) =>
    !!id && local.some((m) => m.hf_id === id || m.id === id);
  const whisperDownloaded = (file: string | null) =>
    !!file && !!whisper?.models.includes(file);

  const runtimeInstalled = !!whisper?.runtime_installed;
  const sttReady = runtimeInstalled && whisperDownloaded(audio.stt_model);
  const ttsReady =
    hfDownloaded(audio.tts_model) && hfDownloaded(WAVTOKENIZER.hfId);

  // Ensure the whisper runtime is installed, then pull a ggml model.
  async function setUpWhisper(file: string) {
    try {
      if (!whisper?.runtime_installed) {
        setNote("Installing speech-to-text runtime…");
        await invoke("whisper_install");
        await refreshWhisper();
      }
      setNote(`Downloading ${file}…`);
      await invoke("whisper_download_model", { file });
      await refreshWhisper();
      setNote(null);
    } catch (e) {
      setNote(e instanceof Error ? e.message : "Speech-to-text setup failed");
    }
  }

  // Download an OuteTTS model + the shared vocoder it needs.
  function setUpTts(hfId: string) {
    if (!hfDownloaded(hfId)) void download(hfId);
    if (!hfDownloaded(WAVTOKENIZER.hfId)) void download(WAVTOKENIZER.hfId);
  }

  function onToggle(v: boolean) {
    if (!v) {
      void save({ audio: { ...audio, enabled: false } });
      return;
    }
    if (sttReady && ttsReady) {
      void save({ audio: { ...audio, enabled: true } });
      return;
    }
    const stt =
      whisperModelByFile(audio.stt_model) ?? recommendWhisperModel(specs);
    const tts = ttsModelByHfId(audio.tts_model) ?? recommendTtsModel(specs);
    setConfirm({ stt, tts });
  }

  async function confirmEnable() {
    if (!confirm) return;
    const { stt, tts } = confirm;
    await save({
      audio: {
        ...audio,
        enabled: true,
        stt_model: stt.file,
        tts_model: tts.hfId,
      },
    });
    setConfirm(null);
    void setUpWhisper(stt.file);
    setUpTts(tts.hfId);
  }

  async function selectStt(file: string) {
    await save({ audio: { ...audio, stt_model: file } });
    if (!whisperDownloaded(file) || !runtimeInstalled) void setUpWhisper(file);
  }

  async function selectTts(hfId: string) {
    await save({ audio: { ...audio, tts_model: hfId } });
    if (!hfDownloaded(hfId) || !hfDownloaded(WAVTOKENIZER.hfId)) setUpTts(hfId);
  }

  async function selectLanguage(lang: string) {
    await save({ audio: { ...audio, stt_language: lang } });
  }

  return (
    <>
      <SettingsGroup
        title="Audio"
        description="Talk to zero and have replies read back. Voice input runs on whisper.cpp; read-aloud runs on llama.cpp — both local and GPU-accelerated."
      >
        <Field
          label="Enable audio"
          hint="Adds a microphone button to the composer (speech → text) and a read-aloud button to assistant replies (text → speech). Off by default. The first time you turn it on, zero recommends and downloads models sized for this machine."
          control={
            <Toggle
              checked={audio.enabled}
              onChange={onToggle}
              label="Enable audio"
            />
          }
        />
      </SettingsGroup>

      {audio.enabled && (
        <SettingsGroup
          title="Voice input & read-aloud"
          description="Pick the speech-to-text and text-to-speech models. They download on demand and run as local GPU processes."
        >
          <WhisperModelField
            value={audio.stt_model}
            downloaded={whisper?.models ?? []}
            runtimeInstalled={runtimeInstalled}
            gpu={!!whisper?.gpu}
            onChange={(f) => void selectStt(f)}
          />
          <SttLanguageField
            value={audio.stt_language ?? "en"}
            onChange={(l) => void selectLanguage(l)}
          />
          <TtsModelField
            value={audio.tts_model}
            onChange={(id) => void selectTts(id)}
          />
          {progress && (
            <div className="py-2">
              <Callout>
                {progress.message}
                {progress.bytes_total
                  ? ` — ${Math.round(progress.percent * 100)}%`
                  : ""}
              </Callout>
            </div>
          )}
          {note && !progress && (
            <div className="py-2">
              <Callout>{note}</Callout>
            </div>
          )}
        </SettingsGroup>
      )}

      {confirm && (
        <AudioDownloadModal
          stt={confirm.stt}
          tts={confirm.tts}
          sttReady={sttReady}
          ttsReady={ttsReady}
          gpu={!!whisper?.gpu}
          runtimeInstalled={runtimeInstalled}
          onCancel={() => setConfirm(null)}
          onConfirm={() => void confirmEnable()}
        />
      )}
    </>
  );
}

/** Speech-to-text model picker: a select over the whisper tiers with a
 *  download-state badge. Whisper runs as a standalone CLI, so picking a tier
 *  that isn't on disk installs the runtime (if needed) and the ggml model. */
function WhisperModelField({
  value,
  downloaded,
  runtimeInstalled,
  gpu,
  onChange,
}: {
  value: string | null;
  downloaded: string[];
  runtimeInstalled: boolean;
  gpu: boolean;
  onChange: (file: string) => void;
}) {
  const isDown = !!value && downloaded.includes(value);
  return (
    <Field
      label="Speech → text"
      hint={`Transcribes the composer's voice input with whisper.cpp (${
        gpu ? "GPU" : "CPU"
      }). Switching to a model you haven't downloaded starts the download automatically.`}
      align="start"
      control={
        <div className="flex flex-col items-end gap-1">
          <select
            value={value ?? ""}
            onChange={(e) => onChange(e.target.value)}
            className="w-56 rounded-[6px] border border-tui-border bg-[var(--fluent-bg-subtle)] px-2 py-1 text-[12px] text-tui-fg"
          >
            <option value="" disabled>
              Select a model…
            </option>
            {WHISPER_MODELS.map((m) => (
              <option key={m.file} value={m.file}>
                {m.name} · {m.sizeHint}
              </option>
            ))}
          </select>
          {value && (
            <span
              className={
                "text-[11px] " +
                (isDown ? "text-emerald-400" : "text-tui-fg-muted")
              }
            >
              {isDown
                ? "downloaded"
                : runtimeInstalled
                  ? "not downloaded"
                  : "runtime + model needed"}
            </span>
          )}
        </div>
      }
    />
  );
}

/** Spoken-language picker for transcription. Auto-detect is offered but
 *  English is the default: whisper mis-detects short dictation clips (often
 *  flipping English to Japanese), so a fixed language is far more reliable. */
function SttLanguageField({
  value,
  onChange,
}: {
  value: string;
  onChange: (lang: string) => void;
}) {
  return (
    <Field
      label="Spoken language"
      hint="The language you'll dictate in. Auto-detect is unreliable on short clips, so pick your language for accurate transcription."
      align="start"
      control={
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-56 rounded-[6px] border border-tui-border bg-[var(--fluent-bg-subtle)] px-2 py-1 text-[12px] text-tui-fg"
        >
          {STT_LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.label}
            </option>
          ))}
        </select>
      }
    />
  );
}

/** Text-to-speech model picker: a select over the OuteTTS tiers. Selecting
 *  a tier downloads it (and the shared WavTokenizer vocoder) through the
 *  normal model-download flow. */
function TtsModelField({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (id: string) => void;
}) {
  const local = useModelsStore((st) => st.local);
  const downloads = useModelsStore((st) => st.downloads);

  const isHf = (id: string) => local.some((m) => m.hf_id === id || m.id === id);
  const downloaded = !!value && isHf(value);
  const vocoderDown = isHf(WAVTOKENIZER.hfId);
  const dl = value ? downloads[value] : undefined;
  const vdl = downloads[WAVTOKENIZER.hfId];
  const downloading =
    (!!dl && (dl.state === "downloading" || dl.state === "pending")) ||
    (!!vdl && (vdl.state === "downloading" || vdl.state === "pending"));

  let badge: ReactNode = null;
  if (downloading) {
    const active = dl ?? vdl;
    const pct =
      active && active.bytes_total && active.bytes_total > 0
        ? Math.round((active.bytes_done / active.bytes_total) * 100)
        : null;
    badge = (
      <span className="inline-flex items-center gap-1 text-[11px] text-tui-fg-muted">
        <Spinner size="sm" />
        {pct === null ? "Downloading…" : `Downloading ${pct}%`}
      </span>
    );
  } else if (downloaded) {
    badge = (
      <span
        className={
          "text-[11px] " + (vocoderDown ? "text-emerald-400" : "text-tui-warn")
        }
      >
        {vocoderDown ? "downloaded" : "vocoder pending"}
      </span>
    );
  } else if (value) {
    badge = (
      <span className="text-[11px] text-tui-fg-muted">not downloaded</span>
    );
  }

  return (
    <Field
      label="Text → speech"
      hint="Reads assistant replies aloud with llama-tts. The voice model and a small shared vocoder download together."
      align="start"
      control={
        <div className="flex flex-col items-end gap-1">
          <select
            value={value ?? ""}
            onChange={(e) => onChange(e.target.value)}
            className="w-56 rounded-[6px] border border-tui-border bg-[var(--fluent-bg-subtle)] px-2 py-1 text-[12px] text-tui-fg"
          >
            <option value="" disabled>
              Select a model…
            </option>
            {TTS_MODELS.map((m) => (
              <option key={m.hfId} value={m.hfId}>
                {m.name} · {m.sizeHint}
              </option>
            ))}
          </select>
          {badge}
        </div>
      }
    />
  );
}

/** Confirm modal shown the first time audio is enabled: lists the
 *  machine-sized speech-to-text + text-to-speech models (and the whisper
 *  runtime / vocoder) that will be downloaded. */
function AudioDownloadModal({
  stt,
  tts,
  sttReady,
  ttsReady,
  gpu,
  runtimeInstalled,
  onCancel,
  onConfirm,
}: {
  stt: WhisperModelOption;
  tts: TtsModelOption;
  sttReady: boolean;
  ttsReady: boolean;
  gpu: boolean;
  runtimeInstalled: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const allReady = sttReady && ttsReady;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onCancel}
    >
      <div
        className="fluent-mica w-full max-w-[460px] rounded-[10px] border border-tui-border shadow-[var(--fluent-shadow-4)]"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="border-b border-tui-border px-4 py-3">
          <h3 className="text-[13px] font-semibold text-tui-fg">
            Set up audio
          </h3>
          <p className="mt-0.5 text-[11px] text-tui-fg-muted">
            {allReady
              ? "These local models are already on disk and will be used for voice input and read-aloud."
              : `These local models — sized for this machine — will be downloaded and run on the ${
                  gpu ? "GPU" : "CPU"
                }. They stay on disk for next time.`}
          </p>
        </header>
        <div className="px-4 py-2">
          {!runtimeInstalled && (
            <AudioModalRow
              tag="Speech engine"
              name={`whisper.cpp runtime (${gpu ? "CUDA" : "CPU"})`}
              note="One-time download of the whisper-cli binary."
              sizeHint={gpu ? "~680 MB" : "~8 MB"}
              downloaded={false}
            />
          )}
          <AudioModalRow
            tag="Speech → text"
            name={stt.name}
            note={stt.note}
            sizeHint={stt.sizeHint}
            downloaded={sttReady}
          />
          <AudioModalRow
            tag="Text → speech"
            name={tts.name}
            note={tts.note}
            sizeHint={tts.sizeHint}
            downloaded={ttsReady}
          />
          <AudioModalRow
            tag="Vocoder"
            name={WAVTOKENIZER.name}
            note="Shared neural vocoder llama-tts needs to render audio."
            sizeHint={WAVTOKENIZER.sizeHint}
            downloaded={ttsReady}
          />
        </div>
        <footer className="flex justify-end gap-2 border-t border-tui-border px-4 py-3">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-[6px] border border-tui-border bg-[var(--fluent-bg-subtle)] px-3 py-1.5 text-[12px] text-tui-fg-dim transition-colors hover:bg-[var(--fluent-bg-subtle-hover)] hover:text-tui-fg"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="rounded-[6px] border border-transparent bg-tui-accent-dim px-3 py-1.5 text-[12px] text-white shadow-[var(--fluent-shadow-2)] transition-colors hover:bg-[var(--fluent-accent-hover)]"
          >
            {allReady ? "Enable" : "Download & enable"}
          </button>
        </footer>
      </div>
    </div>
  );
}

function AudioModalRow({
  tag,
  name,
  note,
  sizeHint,
  downloaded,
}: {
  tag: string;
  name: string;
  note: string;
  sizeHint: string;
  downloaded: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-tui-border py-2.5 last:border-b-0">
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-wide text-tui-fg-muted">
          {tag}
        </div>
        <div className="text-[12px] text-tui-fg">{name}</div>
        <div className="mt-0.5 text-[11px] leading-snug text-tui-fg-muted">
          {note}
        </div>
      </div>
      <div className="shrink-0 text-right">
        <div className="text-[11px] text-tui-fg-dim">{sizeHint}</div>
        <div
          className={
            "text-[11px] " + (downloaded ? "text-emerald-400" : "text-tui-warn")
          }
        >
          {downloaded ? "on disk" : "will download"}
        </div>
      </div>
    </div>
  );
}

// ─── system section ──────────────────────────────────────────────────

function SystemSection() {
  return (
    <SettingsGroup
      title="Host"
      description="Hardware zero can see on this machine. Read-only — useful for picking a device or filing bug reports."
    >
      <div className="py-2">
        <SystemSpecsCard />
      </div>
    </SettingsGroup>
  );
}

// ─── icons (inline SVG — match the sidebar's stroke weight) ─────────

function SlidersIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
    >
      <path d="M4 6h10M18 6h2M4 12h6M14 12h6M4 18h12M20 18h0" />
      <circle cx="16" cy="6" r="2" />
      <circle cx="12" cy="12" r="2" />
      <circle cx="18" cy="18" r="2" />
    </svg>
  );
}

function ServerIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="2" y="2" width="20" height="8" rx="2" />
      <rect x="2" y="14" width="20" height="8" rx="2" />
      <path d="M6 6v0M6 18v0" />
    </svg>
  );
}

function AudioIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" />
    </svg>
  );
}

function ChipIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
      <path d="M9 9h6v6H9z" />
      <path d="M3 9h3M3 12h3M3 15h3M18 9h3M18 12h3M18 15h3M9 3v3M12 3v3M15 3v3M9 18v3M12 18v3M15 18v3" />
    </svg>
  );
}

function MemoryIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M9 5a4 4 0 1 0-4 4v6a4 4 0 1 0 4 4 4 4 0 1 0 6 0 4 4 0 1 0 4-4V9a4 4 0 1 0-4-4 4 4 0 1 0-6 0Z" />
    </svg>
  );
}

function ToolsIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-2.4 2.6-2.6Z" />
    </svg>
  );
}

function SkillsIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 3 14.5 9l6 .5-4.6 4 1.5 6L12 16l-5.4 3.5 1.5-6L3.5 9.5 9.5 9 12 3Z" />
    </svg>
  );
}

function HookIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7 4v6a5 5 0 0 0 10 0V4" />
      <path d="M12 15v3a3 3 0 0 0 6 0" />
    </svg>
  );
}

function DocIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      <path d="M9 13h6M9 17h4" />
    </svg>
  );
}
