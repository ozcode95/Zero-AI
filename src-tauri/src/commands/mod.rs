//! Tauri command handlers, grouped by domain. Each command is a thin shim
//! that adapts an IPC call into a backend function, returning `IpcResult<T>`.

pub mod attachments;
pub mod audio;
pub mod chat;
pub mod documents;
pub mod llama;
pub mod mcp;
pub mod memory;
pub mod models;

pub mod settings;
pub mod skills;
pub mod system;
pub mod tasks;
