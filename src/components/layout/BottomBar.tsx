import { useLlamaStore } from "@/stores/llama";
import type { LlamaStatus } from "@/stores/llama";
import { VARIANT_DISPLAY } from "@/stores/llama";
import { useChatStore } from "@/stores/chat";
import { StatusSpacer } from "@/components/tui/StatusBar";
import { Spinner } from "@/components/tui/Spinner";
import { ServerLogsPopover } from "@/components/server/ServerLogsPopover";

/**
 * Persistent status strip pinned to the bottom of the shell. Surfaces
 * the always-relevant llama.cpp orchestrator state so the user can see
 * at a glance which variant is running and whether it's ready.
 */
export function BottomBar() {
  const orchInfo = useLlamaStore((s) => s.info);
  const llamaLoadingIds = useLlamaStore((s) => s.loadingModelIds);
  const streaming = useChatStore((s) => s.streamingMessageId);
  const activeChat = useChatStore((s) =>
    s.conversations.find((c) => c.id === s.activeId),
  );

  // Resolve the active variant's status.  When no variant is explicitly
  // active ("unset"), look for any variant that is installing, starting,
  // or running — so the bottom bar shows progress during auto-provision.
  const activeSlug = (() => {
    const slug = orchInfo?.active_variant ?? null;
    if (slug && slug !== "unset" && orchInfo?.instances[slug]) return slug;
    // Fallback: pick the first instance with a transitional or running status.
    if (orchInfo) {
      for (const [key, inst] of Object.entries(orchInfo.instances)) {
        if (
          inst.status === "installing" ||
          inst.status === "starting" ||
          inst.status === "running" ||
          inst.status === "installed"
        ) {
          return key;
        }
      }
    }
    return null;
  })();
  const activeInstance = activeSlug
    ? (orchInfo?.instances[activeSlug] ?? null)
    : null;
  const activeStatus: LlamaStatus | null = activeInstance?.status ?? null;
  const loadedModel = activeInstance?.loaded_model ?? null;

  const loadingId = llamaLoadingIds.values().next().value as string | undefined;
  const isLoadingModel = !streaming && !!loadingId;

  const activeTransitional =
    activeStatus === "starting" ||
    activeStatus === "stopping" ||
    activeStatus === "installing";

  const runtimeLabel = (() => {
    const name = activeSlug
      ? (VARIANT_DISPLAY[activeSlug as keyof typeof VARIANT_DISPLAY] ??
        activeSlug)
      : "llama.cpp";
    switch (activeStatus) {
      case "starting":
        return `Starting llama.cpp - ${name}…`;
      case "stopping":
        return `Stopping llama.cpp - ${name}…`;
      case "installing":
        return `Installing llama.cpp - ${name}…`;
      case "stopped":
        return `llama.cpp - ${name} stopped`;
      case "installed":
        return `llama.cpp - ${name} idle`;
      case "not_installed":
        return `llama.cpp not installed`;
      case "error":
        return `llama.cpp - ${name} error`;
      case "running":
      default:
        if (activeStatus === null) return "No variant selected";
        return "Ready";
    }
  })();

  const isReady = activeStatus === "running" && !streaming && !isLoadingModel;
  const showSpinner = isLoadingModel || activeTransitional;

  return (
    <footer
      className={
        "flex shrink-0 items-center gap-3 border-t border-tui-border " +
        "bg-tui-bg-elev px-4 py-1.5 text-[11px]"
      }
    >
      <div className="inline-flex items-center gap-2 text-tui-fg-dim">
        {showSpinner ? (
          <Spinner size="sm" />
        ) : (
          <span className="relative flex h-2 w-2">
            <span
              className={
                "absolute inline-flex h-full w-full rounded-full bg-tui-accent " +
                (streaming ? "animate-ping opacity-75" : "opacity-0")
              }
            />
            <span
              className={
                "relative inline-flex h-2 w-2 rounded-full " +
                (isReady
                  ? "bg-tui-accent"
                  : activeStatus === "error"
                    ? "bg-tui-err"
                    : "bg-tui-fg-muted")
              }
            />
          </span>
        )}
        <span className="font-semibold text-tui-fg">
          {streaming
            ? "Generating…"
            : isLoadingModel
              ? "Loading model"
              : runtimeLabel}
        </span>
        {isLoadingModel && loadingId && (
          <span className="text-tui-fg-muted">· {loadingId}</span>
        )}
        {loadedModel &&
          activeStatus === "running" &&
          !isLoadingModel &&
          !streaming && (
            <span className="text-tui-fg-muted">· {loadedModel}</span>
          )}
        {activeChat && !streaming && !isLoadingModel && !activeTransitional && (
          <span className="text-tui-fg-muted">
            · {activeChat.title || "Untitled"}
          </span>
        )}
      </div>
      <StatusSpacer />
      <ServerLogsPopover
        label="llama"
        status={activeStatus ?? "—"}
        tone={
          activeStatus === "running"
            ? "ok"
            : activeStatus === "error"
              ? "err"
              : "default"
        }
      />
    </footer>
  );
}
