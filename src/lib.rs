//! Hayashi language library crate.
//!
//! Exposes the interpreter for use as a library (including WebAssembly).
//! The binary target (`hay`) lives in `src/main.rs`.

// On wasm32, shadow the standard `print!` and `println!` macros BEFORE
// any module declarations so that all downstream code uses our versions
// that route output through `print_output` → JS callback.
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
/// On wasm32, this calls a JavaScript callback stored via `set_print_callback`.
#[inline]
pub fn print_output(text: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        print!("{}", text);
    }
    #[cfg(target_arch = "wasm32")]
    {
        PRINT_FN.with(|f| {
            if let Some(ref cb) = *f.borrow() {
                let _ = cb.call1(&wasm_bindgen::JsValue::NULL, &wasm_bindgen::JsValue::from_str(text));
            }
        });
    }
}

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static PRINT_FN: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    /// Set the JavaScript callback that receives all interpreter output.
    /// Must be called before `run_hayashi`.
    #[wasm_bindgen]
    pub fn set_print_callback(cb: js_sys::Function) {
        super::PRINT_FN.with(|f| *f.borrow_mut() = Some(cb));
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
