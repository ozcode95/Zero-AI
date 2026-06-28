import { create } from "zustand";
import { invoke, on, Events } from "@/lib/tauri";
import { useLlamaStore } from "@/stores/llama";
import { useSettingsStore, type SamplingConfig } from "@/stores/settings";
import { EMPTY_SAMPLING } from "@/stores/settings";

/// Title used for freshly-created conversations until either the user
/// renames the chat manually or `send()` auto-derives one from the first
/// user message. Kept in a single place so the auto-rename check stays in
/// sync with `create()` and the backend default.
export const DEFAULT_CONVERSATION_TITLE = "New chat";

/// Maximum length (in characters) of an auto-derived conversation title.
/// Tuned to fit the sidebar row without truncation on a typical window
/// width while still being recognisable.
const AUTO_TITLE_MAX_CHARS = 48;

/// Derive a human-friendly conversation title from arbitrary message text.
/// Collapses whitespace, strips control characters, and truncates with a
/// trailing ellipsis when needed. Returns an empty string when the input
/// is blank — callers should treat that as "don't rename".
export function deriveConversationTitle(text: string): string {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return "";
  if (cleaned.length <= AUTO_TITLE_MAX_CHARS) return cleaned;
  // Prefer cutting at the last word boundary inside the budget so titles
  // don't end mid-word; fall back to a hard cut for pathological input.
  const slice = cleaned.slice(0, AUTO_TITLE_MAX_CHARS);
  const lastSpace = slice.lastIndexOf(" ");
  const trimmed =
    lastSpace > AUTO_TITLE_MAX_CHARS / 2 ? slice.slice(0, lastSpace) : slice;
  return `${trimmed.trimEnd()}\u2026`;
}

export type Role = "system" | "user" | "assistant" | "tool";

export interface Message {
  id: string;
  conversation_id: string;
  role: Role;
  content: string;
  thinking?: string | null;
  created_at: string;
  attachments?: Attachment[];
  /**
   * Per-turn capability flags the composer attached to this user
   * message (web search, deep research, thinking opt-in, autonomous
   * loop). Only ever set on user rows; absent / `null` on every
   * other role. Legacy rows persisted before the column existed
   * also surface as missing here — the Rust runner falls back to
   * slash-prefix parsing in that case so old chats keep working.
   */
  turn_overrides?: TurnOverrides | null;
  /**
   * Generation throughput (tokens/second) for an assistant turn, taken
   * from the upstream `timings` block when the stream finished. Only set
   * on assistant rows from a provider that reports timings (llama.cpp);
   * `null`/absent on user rows, legacy rows, and providers without
   * timings (e.g. OVMS).
   */
  tokens_per_second?: number | null;
}

/**
 * Per-turn capability flags forwarded to the chat runner. Each field
 * maps to a previously slash-gated behaviour:
 *   - `web` → unlock `web.search` + `web.read_page` (was `/web`).
 *   - `research` → unlock `web.deep_research` + `web.read_page`
 *     (was `/research`).
 *   - `think` → opt **into** the model's reasoning trace for this
 *     turn. Default for every chat is *no thinking*; this flag is
 *     the per-turn opt-in (Gemma 4 gets its `<|think|>` control
 *     token; other families have `enable_thinking: false` dropped).
 *
 * All fields are optional on the wire because the Rust side defaults
 * each missing flag to `false` — callers only need to send the keys
 * they want set.
 */
export interface TurnOverrides {
  web?: boolean;
  research?: boolean;
  think?: boolean;
}

export interface Attachment {
  kind: "image" | "audio" | "document";
  path: string;
  mime: string;
  bytes: number;
  /** Original filename the user picked. Surfaced in chips + bubbles. */
  name: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  model: string | null;
}

interface ChatDeltaPayload {
  conversation_id: string;
  message_id: string;
  delta: string;
  thinking?: boolean;
}

/// Payload for `chat://rewrite`. The handler *replaces* the message's
/// `content` with `content` rather than appending like a delta. Emitted
/// by the runner after it consumes a legacy ```tool_use``` fenced-JSON
/// block so the un-trimmed text already streamed into the bubble gets
/// scrubbed before the tool banners append on top.
interface ChatRewritePayload {
  conversation_id: string;
  message_id: string;
  content: string;
}

interface ChatDonePayload {
  conversation_id: string;
  message_id: string;
  cancelled?: boolean;
  /** Generation throughput (tokens/s) from the upstream `timings` block. */
  tokens_per_second?: number | null;
}

/// Stable string tags emitted by `chat://error`. Keep in sync with
/// `ChatErrorKind` in `src-tauri/src/chat/runner.rs`.
export type ChatErrorKind =
  | "no_active_provider"
  | "unsupported_provider"
  | "no_model_selected"
  | "ovms_not_running"
  | "llama_not_running"
  | "upstream_unreachable"
  | "upstream_http"
  | "other";

interface ChatErrorPayload {
  conversation_id: string;
  message_id: string;
  error: string;
  kind?: ChatErrorKind;
  hint?: string;
  provider_kind?: string;
  base_url?: string;
  retryable?: boolean;
}

/// Diagnostics stored against a failed assistant message. Surfaced in the
/// chat bubble by `Chat.tsx` so the user can see a one-line hint + retry
/// without digging through the raw error text.
export interface ChatErrorInfo {
  message_id: string;
  error: string;
  kind: ChatErrorKind;
  hint?: string;
  provider_kind?: string;
  base_url?: string;
  retryable: boolean;
}

/// In-flight destructive-tool confirmation prompt. The runner blocks the
/// stream waiting for `chat_tool_confirm(call_id, allow)` so we render a
/// modal until the user makes a choice (or the chat is cancelled).
export interface ToolConfirmRequest {
  conversation_id: string;
  message_id: string;
  call_id: string;
  server_id: string;
  server_name: string;
  tool: string;
  description: string;
  arguments: unknown;
  destructive: boolean;
}

/// A single question surfaced by the `ask_user_input` tool. `options`
/// are the tappable choices; `multi` (default false) lets the user pick
/// more than one before submitting.
export interface AskUserInputQuestion {
  question: string;
  options: string[];
  multi?: boolean;
}

/// Outstanding `ask_user_input` prompt attached to a *finished* assistant
/// turn. Unlike a tool-confirm, the runner has already stopped generating
/// — the user's pick is sent back through the normal composer send flow as
/// a fresh user message (which starts a new turn with the answer in
/// context). Doubles as the `chat://ask-user-input` payload. Keyed by
/// `message_id` so the buttons render beneath the right bubble.
export interface AskUserInputRequest {
  conversation_id: string;
  message_id: string;
  questions: AskUserInputQuestion[];
}

/// A deliverable surfaced by the `present_files` tool. `kind` is a coarse
/// bucket ("image" | "audio" | "document" | "other") the UI maps to an
/// icon; `exists` / `size` come from the backend's stat at emit time.
export interface PresentedFile {
  path: string;
  name: string;
  kind: string;
  exists: boolean;
  size?: number;
}

/// Payload for `chat://present-files`. Unlike ask-user-input this does
/// *not* end the turn — the assistant keeps talking — so we just stash the
/// cards keyed by `message_id`.
interface PresentFilesPayload {
  conversation_id: string;
  message_id: string;
  files: PresentedFile[];
}

interface SendResult {
  user: Message;
  assistant: Message;
}

interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
  messages: Record<string, Message[]>;
  streamingMessageId: string | null;
  loading: boolean;
  /// Per-conversation map of the *latest* error for each assistant message.
  /// Cleared when a retry succeeds (first delta lands) or when the user
  /// dismisses it.
  errors: Record<string, Record<string, ChatErrorInfo>>;
  /// Outstanding destructive-tool confirm prompts, keyed by call_id.
  /// Insertion order is preserved by the underlying Record (each new
  /// confirm gets a fresh key) so the modal can render the oldest one.
  toolConfirms: Record<string, ToolConfirmRequest>;
  /// Outstanding `ask_user_input` prompts, keyed by `message_id`. The
  /// assistant's turn already ended; answering sends a fresh user message
  /// and clears the entry. Ephemeral — not persisted.
  askInputs: Record<string, AskUserInputRequest>;
  /// Files surfaced by `present_files`, keyed by `message_id`. Rendered as
  /// preview cards beneath the assistant bubble. Ephemeral — not persisted.
  presentedFiles: Record<string, PresentedFile[]>;
  /// Per-conversation tool disable list. Entries are catalog keys of the
  /// form `<server_id>::<tool_name>` and represent tools the user turned
  /// off in the chat-header tools popover (overriding the global enable
  /// on the Tools page). Loaded lazily on demand and persisted via
  /// `chat_set_disabled_tools`.
  disabledTools: Record<string, string[]>;
  /// Per-conversation sampling override. `null` means "loaded and empty"
  /// (the runner will fall through to the provider + profile defaults);
  /// `undefined` means "not loaded yet". Loaded lazily on first popover
  /// open and persisted via `chat_set_sampling`.
  sampling: Record<string, SamplingConfig>;
  /// Model the user picked from the chat header *before* a conversation
  /// exists (the new "+ New chat" flow defers the DB write until the
  /// first message). Materialized onto the freshly-created conversation
  /// inside `create()` and cleared on `newChat` / `open` / first apply.
  pendingModel: string | null;

  list: () => Promise<void>;
  open: (id: string) => Promise<void>;
  /// Drop the active conversation (without touching the DB) so the chat
  /// panel renders the empty "new chat" state. The actual DB-backed
  /// conversation is created lazily on the first message send.
  newChat: () => void;
  create: (title?: string) => Promise<string>;
  remove: (id: string) => Promise<void>;
  rename: (id: string, title: string) => Promise<void>;
  setModel: (model: string | null) => Promise<void>;
  setPendingModel: (model: string | null) => Promise<void>;
  send: (
    text: string,
    attachments?: Attachment[],
    overrides?: TurnOverrides,
  ) => Promise<void>;
  cancel: () => Promise<void>;
  retry: (messageId: string) => Promise<void>;
  dismissError: (conversationId: string, messageId: string) => void;
  resolveToolConfirm: (callId: string, allow: boolean) => Promise<void>;
  /// Answer an `ask_user_input` prompt: send `text` back through the
  /// normal send flow (starting a fresh assistant turn) and drop the
  /// stored prompt for `messageId` so the buttons disappear.
  answerAskInput: (messageId: string, text: string) => Promise<void>;
  loadDisabledTools: (conversationId: string) => Promise<void>;
  setToolDisabled: (
    conversationId: string,
    key: string,
    disabled: boolean,
  ) => Promise<void>;
  loadSampling: (conversationId: string) => Promise<void>;
  setSampling: (
    conversationId: string,
    sampling: SamplingConfig,
  ) => Promise<void>;

  bindEvents: () => Promise<() => void>;
}

/// Eagerly load `model` on the active llama.cpp variant so picking a
/// model from the chat header starts loading it immediately — the same
/// affordance the Models page "Load" button gives. Without this, the
/// header pick only updates the conversation's pinned model and the
/// actual load is deferred to the first `send()`, leaving the bottom-bar
/// pill stale and the user wondering whether their pick took effect.
///
/// Best-effort: the chat runner stages the model again at send-time, so
/// any failure here just logs and lets the runner surface the real error.
async function swapLlamaTextGen(model: string): Promise<void> {
  const settings = useSettingsStore.getState();
  const provider = settings.providers.find(
    (p) => p.id === settings.active_provider_id,
  );
  if (!provider || provider.kind !== "llama.cpp") return;

  const llama = useLlamaStore.getState();
  const info = llama.info;
  // Wait for the orchestrator to report in before touching it. Loading
  // can't be staged against an instance we don't know about yet.
  if (!info) return;
  const instance = info.instances[info.active_variant];
  // Don't poke a runtime that can't load anything right now — the Server
  // page is where the user resolves install/setup. `loadModel` will start
  // a stopped server on its own, so those states are still allowed.
  const status = instance?.status;
  if (status === "not_installed" || status === "installing") return;

  // Already serving the requested model: no work, no spinner.
  if (instance?.loaded_model === model) return;

  try {
    await llama.loadModel(model);
  } catch (e) {
    console.warn("eager llama.cpp model load failed (will retry on send)", e);
  }
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  activeId: null,
  messages: {},
  streamingMessageId: null,
  loading: false,
  errors: {},
  toolConfirms: {},
  askInputs: {},
  presentedFiles: {},
  disabledTools: {},
  sampling: {},
  pendingModel: null,

  list: async () => {
    set({ loading: true });
    try {
      const conversations =
        (await invoke<Conversation[]>("chat_list_conversations")) ?? [];
      set({ conversations, loading: false });
    } catch (e) {
      console.error("chat_list_conversations failed", e);
      set({ loading: false });
    }
  },

  open: async (id) => {
    // Drop the ephemeral, message-keyed UI prompts from whatever chat we
    // were just in — ask_user_input options and present_files cards are
    // in-memory only and shouldn't bleed across conversations.
    set({
      activeId: id,
      pendingModel: null,
      askInputs: {},
      presentedFiles: {},
    });
    if (get().messages[id]) return;
    try {
      const msgs =
        (await invoke<Message[]>("chat_list_messages", {
          conversationId: id,
        })) ?? [];
      set((s) => ({ messages: { ...s.messages, [id]: msgs } }));
    } catch (e) {
      console.error("chat_list_messages failed", e);
    }
  },

  newChat: () => {
    // Reset to the empty "about to start a new chat" state without
    // hitting the backend. A real Conversation row only gets created on
    // the first `send()` (or any other action that needs a conv id),
    // which keeps abandoned drafts out of the sidebar and the DB.
    set({ activeId: null, pendingModel: null });
  },

  create: async (title) => {
    const resolvedTitle = title ?? DEFAULT_CONVERSATION_TITLE;
    const id = await invoke<string>("chat_create_conversation", {
      title: resolvedTitle,
    });
    if (!id) {
      // Backend refused — still reconcile the list so the UI doesn't go
      // stale, then bail without flipping `activeId`.
      await get().list();
      return id;
    }
    // Eagerly seed local state so the sidebar updates and the chat
    // panel snaps to an empty fresh view *before* the follow-up
    // `chat_list_conversations` round-trip resolves. Without this, the
    // user can click "+ New chat" and keep staring at the previous
    // conversation's messages until the reconciliation lands — and if
    // anything in that chain slows down (or a concurrent `send` is
    // mid-flight), the new chat never visibly takes over.
    const now = new Date().toISOString();
    set((s) => {
      const exists = s.conversations.some((c) => c.id === id);
      const conv: Conversation = {
        id,
        title: resolvedTitle,
        model: null,
        created_at: now,
        updated_at: now,
      };
      return {
        activeId: id,
        conversations: exists ? s.conversations : [conv, ...s.conversations],
        // Force an empty buffer for the new id so the messages selector
        // can never fall back to a stale entry from a previous chat.
        messages: { ...s.messages, [id]: [] },
      };
    });
    // Materialize any model the user picked from the header *before*
    // the conversation existed. Done here (rather than in every caller
    // that lazy-creates) so the pin survives whichever path triggered
    // the create — send, attach, tool toggle, etc.
    const pending = get().pendingModel;
    if (pending) {
      try {
        await get().setModel(pending);
        // Only clear pendingModel after setModel succeeds so the bottom
        // bar can display it during the async update.
        set({ pendingModel: null });
      } catch (e) {
        console.warn("applying pending model on create failed", e);
        // Still clear on error to avoid stale state.
        set({ pendingModel: null });
      }
    }

    // Reconcile with server-truth in the background. We don't await it
    // because the eager seed above is already enough to make the UI
    // correct; the refresh just fills in the canonical `updated_at` /
    // ordering once the backend has had a chance to commit.
    //
    // This MUST run *after* the `setModel` await above. Otherwise the
    // `chat_list_conversations` SELECT can race against the
    // `chat_set_model` UPDATE and ship back a stale row whose `model`
    // is still `null`, clobbering the just-pinned value and leaving
    // the bottom-bar "model" pill blank for the rest of the chat.
    void get().list();

    return id;
  },

  remove: async (id) => {
    await invoke("chat_delete_conversation", { conversationId: id });
    set((s) => {
      const { [id]: _drop, ...rest } = s.messages;
      return {
        messages: rest,
        activeId: s.activeId === id ? null : s.activeId,
      };
    });
    await get().list();
  },

  rename: async (id, title) => {
    const trimmed = title.trim();
    const normalized =
      trimmed.length > 0 ? trimmed : DEFAULT_CONVERSATION_TITLE;
    // Mirror locally first so the sidebar updates instantly; the backend
    // round-trip just persists the change.
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === id ? { ...c, title: normalized } : c,
      ),
    }));
    try {
      await invoke("chat_set_title", {
        conversationId: id,
        title: normalized,
      });
    } catch (e) {
      console.error("chat_set_title failed", e);
      // Don't roll back — the user sees the desired title in the sidebar.
      // A follow-up `list()` (e.g. on next create) will reconcile if the
      // backend really did reject the change.
    }
  },

  setPendingModel: async (model) => {
    set({ pendingModel: model });
    // Trigger the same eager swap that `setModel` runs once a
    // conversation exists. The user picked a model from the chat
    // header — they expect it to start loading immediately, the same
    // way the Models page "Load" button behaves. Without this, a fresh
    // "+ New chat" would silently defer the load until the first
    // `send()`, leaving the bottom-bar pill blank in the meantime.
    if (model) await swapLlamaTextGen(model);
  },

  setModel: async (model) => {
    const convId = get().activeId;
    if (!convId) return;
    await invoke("chat_set_model", { conversationId: convId, model });
    // Reflect the change locally without a round-trip.
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === convId ? { ...c, model } : c,
      ),
    }));
    if (model) await swapLlamaTextGen(model);
  },

  send: async (text, attachments, overrides) => {
    const convId = get().activeId ?? (await get().create());
    if (!convId) return;

    // Capture pre-send state so we can decide whether to auto-title the
    // chat after the user's first message lands. We deliberately read
    // *before* the splice below so a freshly-created chat (zero messages,
    // default title) is correctly recognised as eligible.
    const stateBefore = get();
    const convBefore = stateBefore.conversations.find((c) => c.id === convId);
    const msgsBefore = stateBefore.messages[convId] ?? [];
    const isFirstUserMessage = !msgsBefore.some((m) => m.role === "user");
    const titleIsDefault =
      (convBefore?.title ?? "").trim() === DEFAULT_CONVERSATION_TITLE;

    // Only ship `overrides` to the backend when at least one flag is
    // actually on so the wire stays clean for the common case (every
    // plain prompt). The Rust side treats a missing payload as the
    // all-false default.
    const hasOverride =
      !!overrides &&
      (overrides.web === true ||
        overrides.research === true ||
        overrides.think === true);

    const res = await invoke<SendResult>("chat_send_message", {
      conversationId: convId,
      content: text,
      attachments: attachments ?? [],
      overrides: hasOverride ? overrides : null,
    });
    if (!res) return;
    // Optimistically splice both turns in. The runner will then stream deltas
    // into the assistant row via `chat://delta`.
    set((s) => {
      const list = s.messages[convId] ?? [];
      return {
        messages: {
          ...s.messages,
          [convId]: [...list, res.user, res.assistant],
        },
        streamingMessageId: res.assistant.id,
      };
    });

    // Auto-rename the conversation from its first user message so the
    // sidebar stops showing a generic "New chat" row the moment the chat
    // actually has a topic. We only do this when the title is still the
    // default placeholder — a user who already renamed the chat (or one
    // seeded with a custom title via `create(title)`) is left alone.
    if (isFirstUserMessage && titleIsDefault) {
      const derived = deriveConversationTitle(text);
      if (derived) void get().rename(convId, derived);
    }
  },

  cancel: async () => {
    const id = get().streamingMessageId;
    if (!id) return;
    await invoke("chat_cancel", { messageId: id });
  },

  retry: async (messageId) => {
    // Optimistically clear the error + content so the bubble looks fresh
    // before the backend has had a chance to spin a new stream.
    set((s) => {
      const next: Record<string, Record<string, ChatErrorInfo>> = {
        ...s.errors,
      };
      for (const convId of Object.keys(next)) {
        if (next[convId][messageId]) {
          const { [messageId]: _drop, ...rest } = next[convId];
          next[convId] = rest;
        }
      }
      const updatedMessages: Record<string, Message[]> = { ...s.messages };
      for (const convId of Object.keys(updatedMessages)) {
        const list = updatedMessages[convId];
        const idx = list.findIndex((m) => m.id === messageId);
        if (idx !== -1) {
          const copy = list.slice();
          copy[idx] = { ...copy[idx], content: "", thinking: null };
          updatedMessages[convId] = copy;
        }
      }
      return {
        errors: next,
        messages: updatedMessages,
        streamingMessageId: messageId,
      };
    });
    try {
      await invoke("chat_retry", { messageId });
    } catch (e) {
      console.error("chat_retry failed", e);
      set((s) =>
        s.streamingMessageId === messageId ? { streamingMessageId: null } : s,
      );
    }
  },

  dismissError: (conversationId, messageId) => {
    set((s) => {
      const conv = s.errors[conversationId];
      if (!conv || !conv[messageId]) return s;
      const { [messageId]: _drop, ...rest } = conv;
      return { errors: { ...s.errors, [conversationId]: rest } };
    });
  },

  resolveToolConfirm: async (callId, allow) => {
    // Drop the prompt optimistically so the modal closes immediately even
    // if the IPC round-trip takes a moment. Backend resolution is
    // idempotent (returns `false` on unknown ids) so a duplicate click is
    // harmless.
    set((s) => {
      if (!s.toolConfirms[callId]) return s;
      const { [callId]: _drop, ...rest } = s.toolConfirms;
      return { toolConfirms: rest };
    });
    try {
      await invoke("chat_tool_confirm", { callId, allow });
    } catch (e) {
      console.error("chat_tool_confirm failed", e);
    }
  },

  answerAskInput: async (messageId, text) => {
    const req = get().askInputs[messageId];
    if (!req) return;
    // Drop the prompt first so the option buttons disappear immediately.
    set((s) => {
      if (!s.askInputs[messageId]) return s;
      const { [messageId]: _drop, ...rest } = s.askInputs;
      return { askInputs: rest };
    });
    // Reuse the exact send flow the composer uses so attachment / override
    // defaults stay consistent. The answer is for `req.conversation_id`;
    // in the common case that's the active conversation, which is what
    // `send` targets.
    await get().send(text);
  },

  loadDisabledTools: async (conversationId) => {
    if (get().disabledTools[conversationId]) return;
    try {
      const keys =
        (await invoke<string[]>("chat_get_disabled_tools", {
          conversationId,
        })) ?? [];
      set((s) => ({
        disabledTools: { ...s.disabledTools, [conversationId]: keys },
      }));
    } catch (e) {
      console.error("chat_get_disabled_tools failed", e);
      // Treat failures as "no overrides" so the popover still renders.
      set((s) =>
        s.disabledTools[conversationId]
          ? s
          : {
              disabledTools: { ...s.disabledTools, [conversationId]: [] },
            },
      );
    }
  },

  setToolDisabled: async (conversationId, key, disabled) => {
    const current = get().disabledTools[conversationId] ?? [];
    const next = disabled
      ? current.includes(key)
        ? current
        : [...current, key]
      : current.filter((k) => k !== key);
    if (next === current) return;
    set((s) => ({
      disabledTools: { ...s.disabledTools, [conversationId]: next },
    }));
    try {
      await invoke("chat_set_disabled_tools", {
        conversationId,
        keys: next,
      });
    } catch (e) {
      console.error("chat_set_disabled_tools failed", e);
      // Roll back the optimistic update on failure so the UI stays
      // consistent with what the backend will actually apply.
      set((s) => ({
        disabledTools: { ...s.disabledTools, [conversationId]: current },
      }));
    }
  },

  loadSampling: async (conversationId) => {
    if (get().sampling[conversationId]) return;
    try {
      const cfg = (await invoke<SamplingConfig>("chat_get_sampling", {
        conversationId,
      })) ?? { ...EMPTY_SAMPLING };
      set((s) => ({
        sampling: { ...s.sampling, [conversationId]: cfg },
      }));
    } catch (e) {
      console.error("chat_get_sampling failed", e);
      // Treat failures as "no override" so the popover renders the
      // provider/profile defaults instead of erroring out.
      set((s) =>
        s.sampling[conversationId]
          ? s
          : {
              sampling: {
                ...s.sampling,
                [conversationId]: { ...EMPTY_SAMPLING },
              },
            },
      );
    }
  },

  setSampling: async (conversationId, sampling) => {
    const current = get().sampling[conversationId] ?? { ...EMPTY_SAMPLING };
    // Optimistic local update so the popover slider feels live; rolled
    // back on backend failure (same pattern as `setToolDisabled`).
    set((s) => ({
      sampling: { ...s.sampling, [conversationId]: sampling },
    }));
    try {
      await invoke("chat_set_sampling", { conversationId, sampling });
    } catch (e) {
      console.error("chat_set_sampling failed", e);
      set((s) => ({
        sampling: { ...s.sampling, [conversationId]: current },
      }));
    }
  },

  bindEvents: async () => {
    const offDelta = await on<ChatDeltaPayload>(Events.ChatDelta, (p) => {
      set((s) => {
        const list = s.messages[p.conversation_id] ?? [];
        const idx = list.findIndex((m) => m.id === p.message_id);
        let next: Message[];
        if (idx === -1) {
          // No placeholder yet (e.g. another window opened the chat after
          // streaming started) — synthesize one.
          next = [
            ...list,
            {
              id: p.message_id,
              conversation_id: p.conversation_id,
              role: "assistant",
              content: p.thinking ? "" : p.delta,
              thinking: p.thinking ? p.delta : null,
              created_at: new Date().toISOString(),
            },
          ];
        } else {
          next = list.slice();
          const cur = next[idx];
          next[idx] = p.thinking
            ? { ...cur, thinking: (cur.thinking ?? "") + p.delta }
            : { ...cur, content: cur.content + p.delta };
        }
        // First delta after a retry clears any stale error for the same
        // message id so the hint banner disappears on its own.
        const convErrs = s.errors[p.conversation_id];
        const errors =
          convErrs && convErrs[p.message_id]
            ? (() => {
                const { [p.message_id]: _drop, ...rest } = convErrs;
                return { ...s.errors, [p.conversation_id]: rest };
              })()
            : s.errors;
        return {
          messages: { ...s.messages, [p.conversation_id]: next },
          streamingMessageId: p.message_id,
          errors,
        };
      });
    });

    const offDone = await on<ChatDonePayload>(Events.ChatDone, (p) => {
      set((s) => {
        // Record the throughput on the just-finished assistant row so the
        // bubble footer can render `XX.X tok/s`. The runner only sends a
        // value for providers that report timings; leave the row untouched
        // otherwise so a missing stat doesn't blank an existing one.
        let messages = s.messages;
        if (p.tokens_per_second != null) {
          const list = s.messages[p.conversation_id];
          if (list) {
            const idx = list.findIndex((m) => m.id === p.message_id);
            if (idx !== -1) {
              const next = list.slice();
              next[idx] = {
                ...next[idx],
                tokens_per_second: p.tokens_per_second,
              };
              messages = { ...s.messages, [p.conversation_id]: next };
            }
          }
        }
        return {
          messages,
          streamingMessageId:
            s.streamingMessageId === p.message_id ? null : s.streamingMessageId,
        };
      });
    });

    // `chat://rewrite` replaces the live streaming buffer with a
    // canonical version. Used by the runner to scrub legacy fenced
    // tool-call JSON that already streamed into the bubble before it
    // was recognised as a protocol marker, and to splice per-round
    // reasoning into `content` as an inline `[thinking] … [/thinking]`
    // block. Either way the live `thinking` field is reset to null:
    // for the legacy-fence case there was no separate reasoning to
    // begin with, and for the inline-thinking case the text has just
    // been moved into `content` (leaving it in `thinking` too would
    // render the same paragraph twice). If the message isn't in the
    // local cache (rare — same window must be open) we synthesise a
    // row with the canonical content.
    const offRewrite = await on<ChatRewritePayload>(Events.ChatRewrite, (p) => {
      set((s) => {
        const list = s.messages[p.conversation_id] ?? [];
        const idx = list.findIndex((m) => m.id === p.message_id);
        let next: Message[];
        if (idx === -1) {
          next = [
            ...list,
            {
              id: p.message_id,
              conversation_id: p.conversation_id,
              role: "assistant",
              content: p.content,
              thinking: null,
              created_at: new Date().toISOString(),
            },
          ];
        } else {
          next = list.slice();
          next[idx] = { ...next[idx], content: p.content, thinking: null };
        }
        return { messages: { ...s.messages, [p.conversation_id]: next } };
      });
    });

    const offErr = await on<ChatErrorPayload>(Events.ChatError, (p) => {
      console.error("chat error", p);
      set((s) => {
        const list = s.messages[p.conversation_id] ?? [];
        const idx = list.findIndex((m) => m.id === p.message_id);
        let next = list;
        if (idx !== -1) {
          next = list.slice();
          const cur = next[idx];
          next[idx] = {
            ...cur,
            content: cur.content || `[error] ${p.error}`,
          };
        }
        const info: ChatErrorInfo = {
          message_id: p.message_id,
          error: p.error,
          kind: p.kind ?? "other",
          hint: p.hint,
          provider_kind: p.provider_kind,
          base_url: p.base_url,
          retryable: p.retryable ?? false,
        };
        const convErrs = s.errors[p.conversation_id] ?? {};
        return {
          messages: { ...s.messages, [p.conversation_id]: next },
          streamingMessageId:
            s.streamingMessageId === p.message_id ? null : s.streamingMessageId,
          errors: {
            ...s.errors,
            [p.conversation_id]: { ...convErrs, [p.message_id]: info },
          },
        };
      });
    });

    const offConfirm = await on<ToolConfirmRequest>(
      Events.ChatToolConfirm,
      (p) => {
        set((s) => ({
          toolConfirms: { ...s.toolConfirms, [p.call_id]: p },
        }));
      },
    );

    // `chat://ask-user-input` arrives *after* the assistant's turn ended.
    // Stash the prompt keyed by message_id so Chat.tsx renders the option
    // buttons beneath that bubble; the answer goes back out as a fresh
    // user message via `answerAskInput`.
    const offAskInput = await on<AskUserInputRequest>(
      Events.ChatAskUserInput,
      (p) => {
        set((s) => ({
          askInputs: { ...s.askInputs, [p.message_id]: p },
        }));
      },
    );

    // `chat://present-files` surfaces deliverables mid-turn (the assistant
    // keeps talking). Stash the file list keyed by message_id for the
    // preview cards.
    const offPresentFiles = await on<PresentFilesPayload>(
      Events.ChatPresentFiles,
      (p) => {
        set((s) => ({
          presentedFiles: { ...s.presentedFiles, [p.message_id]: p.files },
        }));
      },
    );

    // When a chat-level cancellation clears the streaming flag, drop any
    // confirms tied to that message so we don't leave a stale modal
    // around — the backend has already torn down the pending sender.
    const offDoneClearConfirms = await on<ChatDonePayload>(
      Events.ChatDone,
      (p) => {
        set((s) => {
          const next = { ...s.toolConfirms };
          let changed = false;
          for (const [key, req] of Object.entries(next)) {
            if (req.message_id === p.message_id) {
              delete next[key];
              changed = true;
            }
          }
          return changed ? { toolConfirms: next } : s;
        });
      },
    );

    return () => {
      offDelta();
      offDone();
      offRewrite();
      offErr();
      offConfirm();
      offAskInput();
      offPresentFiles();
      offDoneClearConfirms();
    };
  },
}));
