import { create } from "zustand";
import { invoke } from "@/lib/tauri";

/**
 * One knowledge-base document. Mirrors the Rust `Document` struct returned
 * by the `documents_*` commands. The `id` is the stored filename under
 * `~/.zero/documents/` and is what `settings.embedding.documents_disabled`
 * references.
 */
export interface KbDocument {
  id: string;
  name: string;
  bytes: number;
  path: string;
}

interface DocumentsState {
  documents: KbDocument[];
  loading: boolean;
  /** Re-read the documents directory. */
  list: () => Promise<void>;
  /**
   * Copy OS-picker file paths into the knowledge base. New documents are
   * enabled by default (the enabled state lives in settings, not here).
   * Returns the freshly added documents.
   */
  add: (paths: string[]) => Promise<KbDocument[]>;
  /** Delete a document from disk, then refresh the list. */
  remove: (id: string) => Promise<void>;
}

export const useDocumentsStore = create<DocumentsState>((set, get) => ({
  documents: [],
  loading: false,

  list: async () => {
    set({ loading: true });
    try {
      const documents = (await invoke<KbDocument[]>("documents_list")) ?? [];
      set({ documents, loading: false });
    } catch (e) {
      console.error("documents_list failed", e);
      set({ loading: false });
    }
  },

  add: async (paths) => {
    const added: KbDocument[] = [];
    for (const p of paths) {
      try {
        const doc = await invoke<KbDocument>("documents_add", {
          sourcePath: p,
        });
        if (doc) added.push(doc);
      } catch (e) {
        console.error("documents_add failed", p, e);
      }
    }
    await get().list();
    return added;
  },

  remove: async (id) => {
    try {
      await invoke("documents_delete", { id });
    } catch (e) {
      console.error("documents_delete failed", e);
    }
    await get().list();
  },
}));
