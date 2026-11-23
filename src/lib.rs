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
        let captured = CAPTURE.with(|capture| {
            if let Some(buffer) = capture.borrow_mut().as_mut() {
                buffer.push_str(text);
                true
            } else {
                false
            }
        });
        if !captured {
            unsafe {
                if let Some(ref cb) = PRINT_FN {
                    let _ = cb.call1(
                        &wasm_bindgen::JsValue::NULL,
                        &wasm_bindgen::JsValue::from_str(text),
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
static mut PRINT_FN: Option<js_sys::Function> = None;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static CAPTURE: std::cell::RefCell<Option<String>> =
        std::cell::RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
pub fn start_capture() {
    CAPTURE.with(|capture| *capture.borrow_mut() = Some(String::new()));
}

#[cfg(target_arch = "wasm32")]
pub fn take_capture() -> String {
    CAPTURE.with(|capture| capture.borrow_mut().take().unwrap_or_default())
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    /// Set the JavaScript callback that receives all interpreter output.
    /// Must be called before `run_hayashi`.
    #[wasm_bindgen]
    pub fn set_print_callback(cb: js_sys::Function) {
        unsafe {
            super::PRINT_FN = Some(cb);
        }
    }

    #[wasm_bindgen]
    pub fn run_hayashi(source: &str) -> String {
        let mut interp = crate::lang::interpreter::Interpreter::new();
        match crate::lang::run_source(source, &mut interp) {
            Ok(()) => String::new(),
            Err(e) => format!("{e}"),
        }
    }

    /// Persistent session for host-driven execution.
    #[wasm_bindgen]
    pub struct WasmSession {
        inner: crate::lang::host_api::Session,
    }

    #[wasm_bindgen]
    impl WasmSession {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                inner: crate::lang::host_api::Session::new(),
            }
        }

        /// Bind a JSON-encoded value. Returns an error string or "".
        pub fn set_input_json(&mut self, name: &str, json: &str) -> String {
            match serde_json::from_str::<serde_json::Value>(json) {
                Ok(value) => match crate::lang::host_api::value_from_json(&value) {
                    Ok(value) => self.inner.set_value(name, value).err().unwrap_or_default(),
                    Err(error) => error,
                },
                Err(error) => format!("invalid JSON: {error}"),
            }
        }

        /// Alias an existing session value (models/DataFrames without copy).
        pub fn set_input_ref(&mut self, name: &str, from: &str) -> String {
            self.inner.set_ref(name, from).err().unwrap_or_default()
        }

        /// Run a source snippet. Returns JSON {output, error}.
        pub fn run(&mut self, source: &str) -> String {
            crate::start_capture();
            let result = crate::lang::run_source(source, &mut self.inner.interp);
            let output = crate::take_capture();
            serde_json::json!({
                "output": output,
                "error": result.err().map(|error| error.to_string()),
            })
            .to_string()
        }

        /// JSON envelope for one output variable.
        pub fn get_output_json(&self, name: &str) -> String {
            match self.inner.get_value(name) {
                Some(value) => crate::lang::host_api::value_to_json(&value).to_string(),
                None => serde_json::json!({
                    "kind": "error",
                    "message": format!("unknown variable '{name}'"),
                })
                .to_string(),
            }
        }

        pub fn get_summary_json(&self, name: &str) -> String {
            match self.inner.get_value(name) {
                Some(value) => serde_json::json!({
                    "kind": crate::lang::host_api::value_kind(&value),
                    "summary": crate::lang::host_api::value_summary(&value),
                })
                .to_string(),
                None => "{}".into(),
            }
        }

        pub fn list_variables_json(&self) -> String {
            let variables: Vec<serde_json::Value> = self
                .inner
                .list_variables()
                .into_iter()
                .map(|(name, kind, summary)| {
                    serde_json::json!({"name": name, "kind": kind, "summary": summary})
                })
                .collect();
            serde_json::json!({"variables": variables}).to_string()
        }

        pub fn reset(&mut self) {
            self.inner.reset();
        }
    }
}
