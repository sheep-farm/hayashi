//! Hayashi language library crate.
//!
//! Exposes the interpreter for use as a library (including WebAssembly).
//! The binary target (`hay`) lives in `src/main.rs`.

// On wasm32, shadow the standard `print!` and `println!` macros BEFORE
// any module declarations so that all downstream code uses our versions
// that route output through `print_output` → `__hayashi_print` (JS callback).
#[cfg(target_arch = "wasm32")]
macro_rules! print {
    ($($arg:tt)*) => {{
        $crate::print_output(&format!($($arg)*))
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! println {
    () => {{
        $crate::print_output("\n")
    }};
    ($($arg:tt)*) => {{
        $crate::print_output(&format!("{}\n", format_args!($($arg)*)))
    }};
}

pub mod io;
pub mod lang;

pub use lang::interpreter::Interpreter;
pub use lang::run_source;

/// Print a string to the output stream. On native targets this calls `print!`.
/// On wasm32, this calls a JavaScript function `__hayashi_print` that
/// the playground page defines to capture output.
#[inline]
pub fn print_output(text: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        print!("{}", text);
    }
    #[cfg(target_arch = "wasm32")]
    {
        self::wasm::__hayashi_print(text);
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    /// JavaScript callback that receives each line of output.
    /// The JS side sets `window.__hayashi_print` to capture text.
    #[wasm_bindgen]
    extern "C" {
        pub fn __hayashi_print(text: &str);
    }

    #[wasm_bindgen]
    pub fn run_hayashi(source: &str) -> String {
        let mut interp = crate::lang::interpreter::Interpreter::new();
        match crate::lang::run_source(source, &mut interp) {
            Ok(()) => String::new(),
            Err(e) => format!("{e}"),
        }
    }
}
