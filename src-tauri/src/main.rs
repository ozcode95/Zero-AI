#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Route every allocation in the desktop executable through mimalloc. Declared
// in the binary crate so it governs the final linked image.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    zero_lib::run()
}
