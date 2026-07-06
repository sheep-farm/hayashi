//! Hayashi language library crate.
//!
//! Exposes the interpreter for use as a library (including WebAssembly).
//! The binary target (`hay`) lives in `src/main.rs`.

pub mod io;
pub mod lang;

pub use lang::interpreter::Interpreter;
pub use lang::run_source;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;
    use crate::lang::interpreter::Interpreter;

    #[wasm_bindgen]
    pub fn run_hayashi(source: &str) -> String {
        let mut interp = Interpreter::new();
        match crate::lang::run_source(source, &mut interp) {
            Ok(()) => String::from("OK"),
            Err(e) => format!("Error: {e}"),
        }
    }
}
