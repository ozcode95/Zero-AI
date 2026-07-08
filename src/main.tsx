import ReactDOM from "react-dom/client";
import { trace, debug, info, warn, error } from "@tauri-apps/plugin-log";
import App from "./App";
import "./index.css";

// Forward the webview's console into tauri-plugin-log. The plugin emits the
// records on the backend as `log` calls, which LogTracer pipes into the same
// tracing subscriber + rolling log file the Rust side uses — so frontend and
// backend logs share one stream. Outside the Tauri runtime (e.g. a plain
// browser dev preview) the import is inert and we leave console untouched.
//
// Guarded so a missing plugin never breaks the UI: each forwarded method
// preserves the original browser behaviour, then best-effort ships the line
// to the backend.
function forwardConsole(
  fnName: "log" | "debug" | "info" | "warn" | "error",
  logger: (message: string) => Promise<unknown>,
) {
  const original = console[fnName].bind(console);
  console[fnName] = ((...args: unknown[]) => {
    original(...args);
    const message = args
      .map((a) =>
        typeof a === "string"
          ? a
          : (() => {
              try {
                return JSON.stringify(a);
              } catch {
                return String(a);
              }
            })(),
      )
      .join(" ");
    void logger(message).catch(() => {
      /* plugin unavailable — swallow */
    });
  }) as never;
}

if (typeof window !== "undefined" && "__TAURI_INTERNALS__" in window) {
  forwardConsole("log", trace);
  forwardConsole("debug", debug);
  forwardConsole("info", info);
  forwardConsole("warn", warn);
  forwardConsole("error", error);
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  // <React.StrictMode>
  <App />,
  // </React.StrictMode>
);
