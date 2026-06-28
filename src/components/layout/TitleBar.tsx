import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { getVersion } from "@tauri-apps/api/app";
import { useSettingsStore } from "@/stores/settings";

/**
 * Custom Win11-style title bar.
 *
 * Replaces the native window chrome (the Tauri window is created with
 * `decorations: false`). The whole strip is a drag region; the three
 * caption buttons on the right (minimize / maximize-restore / close)
 * call the equivalent webview window APIs. Layout follows the Windows
 * 11 spec: 48px tall, 46×32 caption buttons, accent-color hover for
 * the close button.
 */
export function TitleBar() {
  const [maximized, setMaximized] = useState(false);
  const [isTauri, setIsTauri] = useState(false);
  const [version, setVersion] = useState("");

  useEffect(() => {
    if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) {
      return;
    }
    setIsTauri(true);
    void getVersion()
      .then(setVersion)
      .catch((e) => console.warn("[titlebar] failed to read app version", e));
    const w = getCurrentWindow();
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    void (async () => {
      try {
        setMaximized(await w.isMaximized());
        const off = await w.onResized(async () => {
          if (cancelled) return;
          try {
            setMaximized(await w.isMaximized());
          } catch {
            /* window may be closing */
          }
        });
        if (cancelled) {
          off();
        } else {
          unlisten = off;
        }
      } catch (e) {
        console.warn("[titlebar] failed to bind window events", e);
      }
    })();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const win = isTauri ? getCurrentWindow() : null;
  const closeToTaskbar = useSettingsStore((s) => s.close_to_taskbar);

  // "Close button minimizes to taskbar": when enabled, the caption close
  // button keeps zero running in the background instead of quitting.
  const handleClose = () => {
    if (closeToTaskbar) void win?.minimize();
    else void win?.close();
  };

  return (
    <header
      data-tauri-drag-region
      className={
        "relative flex h-[40px] shrink-0 select-none items-center gap-3 " +
        "border-b border-tui-border bg-tui-bg pl-3 pr-0 text-[12px] " +
        "text-tui-fg-dim"
      }
    >
      {/* brand */}
      <div
        data-tauri-drag-region
        className="pointer-events-none flex items-center gap-2"
      >
        <div className="flex h-5 w-5 items-center justify-center rounded-[4px] text-white shadow-[var(--fluent-shadow-2)]">
          <img src="/icon.png" width="18" height="18" alt="ZerØ icon" />
        </div>
        <span className="text-[12px] font-semibold tracking-wide text-tui-fg">
          ZerØ
        </span>
        {version && (
          <span className="text-[10px] text-tui-fg-muted">v{version}</span>
        )}
      </div>

      {/* drag spacer */}
      <div data-tauri-drag-region className="flex-1" />

      {/* caption buttons */}
      <div className="flex h-full items-stretch">
        <CaptionButton
          ariaLabel="Minimize"
          onClick={() => void win?.minimize()}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <path d="M0 5h10" stroke="currentColor" strokeWidth="1" />
          </svg>
        </CaptionButton>
        <CaptionButton
          ariaLabel={maximized ? "Restore" : "Maximize"}
          onClick={() => void win?.toggleMaximize()}
        >
          {maximized ? (
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
              <rect
                x="0.5"
                y="2.5"
                width="6"
                height="6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
              <path
                d="M2.5 2.5V1.5h6v6h-1"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
            </svg>
          ) : (
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
              <rect
                x="0.5"
                y="0.5"
                width="9"
                height="9"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
            </svg>
          )}
        </CaptionButton>
        <CaptionButton
          ariaLabel={closeToTaskbar ? "Minimize to taskbar" : "Close"}
          onClick={handleClose}
          danger
        >
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <path
              d="M0 0l10 10M10 0L0 10"
              stroke="currentColor"
              strokeWidth="1"
            />
          </svg>
        </CaptionButton>
      </div>
    </header>
  );
}

function CaptionButton({
  ariaLabel,
  onClick,
  children,
  danger = false,
}: {
  ariaLabel: string;
  onClick: () => void;
  children: React.ReactNode;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      title={ariaLabel}
      onClick={onClick}
      className={
        "flex h-full w-[46px] items-center justify-center text-tui-fg-dim " +
        "transition-colors duration-100 ease-out " +
        (danger
          ? "hover:bg-[#c42b1c] hover:text-white active:bg-[#b9261a]"
          : "hover:bg-[var(--fluent-bg-subtle-hover)] hover:text-tui-fg active:bg-[var(--fluent-bg-subtle-pressed)]")
      }
    >
      {children}
    </button>
  );
}
