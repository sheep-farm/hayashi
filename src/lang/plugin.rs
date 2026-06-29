use super::interpreter::Value;
use std::collections::HashMap;
use std::rc::Rc;

/// Unified Hayashi Plugin Trait
#[allow(dead_code)]
pub trait HayashiPlugin {
    fn name(&self) -> &str;
    fn call(&mut self, func_name: &str, args: &[Value]) -> Result<Value, String>;
}

/// Helper to serialize Value into JSON for WASM/FFI exchanges
pub fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::Float(f) => serde_json::json!(f),
        Value::Int(i) => serde_json::json!(i),
        Value::Bool(b) => serde_json::json!(b),
        Value::Str(s) => serde_json::json!(s),
        Value::Nil => serde_json::Value::Null,
        Value::List(lst) => {
            let arr: Vec<serde_json::Value> = lst.iter().map(value_to_json).collect();
            serde_json::Value::Array(arr)
        }
        Value::Dict(dct) => {
            let mut map = serde_json::Map::new();
            for (k, v) in dct.iter() {
                map.insert(k.clone(), value_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        Value::DataFrame(df) => {
            // Serialize DataFrame as a dictionary of column arrays
            let mut map = serde_json::Map::new();
            for col in df.column_names() {
                if let Ok(c) = df.get_column(&col) {
                    match c {
                        greeners::Column::Float(arr) => {
                            let vals: Vec<serde_json::Value> =
                                arr.iter().map(|&x| serde_json::json!(x)).collect();
                            map.insert(col.to_string(), serde_json::Value::Array(vals));
                        }
                        greeners::Column::Int(arr) => {
                            let vals: Vec<serde_json::Value> =
                                arr.iter().map(|&x| serde_json::json!(x)).collect();
                            map.insert(col.to_string(), serde_json::Value::Array(vals));
                        }
                        greeners::Column::Bool(arr) => {
                            let vals: Vec<serde_json::Value> =
                                arr.iter().map(|&x| serde_json::json!(x)).collect();
                            map.insert(col.to_string(), serde_json::Value::Array(vals));
                        }
                        greeners::Column::String(arr) => {
                            let vals: Vec<serde_json::Value> =
                                arr.iter().map(|s| serde_json::json!(s)).collect();
                            map.insert(col.to_string(), serde_json::Value::Array(vals));
                        }
                        _ => {}
                    }
                }
            }
            serde_json::Value::Object(map)
        }
        _ => serde_json::Value::Null,
    }
}

/// Helper to deserialize JSON back into Value
pub fn json_to_value(jval: &serde_json::Value) -> Value {
    match jval {
        serde_json::Value::Null => Value::Nil,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::Str(s.clone()),
        serde_json::Value::Array(arr) => {
            let lst: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::List(Rc::new(lst))
        }
        serde_json::Value::Object(obj) => {
            let mut map = HashMap::new();
            for (k, v) in obj.iter() {
                map.insert(k.clone(), json_to_value(v));
            }
            Value::Dict(Rc::new(map))
        }
    }
}

// =============================================================================
// Rust Native Plugin Implementation (using libloading)
// =============================================================================

pub struct RustNativePlugin {
    #[allow(dead_code)]
    name: String,
    lib: libloading::Library,
}

impl RustNativePlugin {
    pub fn new(path: &str, name: &str) -> Result<Self, String> {
        let lib = unsafe { libloading::Library::new(path).map_err(|e| e.to_string())? };
        Ok(Self {
            name: name.to_string(),
            lib,
        })
    }
}

impl HayashiPlugin for RustNativePlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn call(&mut self, func_name: &str, args: &[Value]) -> Result<Value, String> {
        unsafe {
            // Native plugins export functions with signature:
            // extern "C" fn(*const c_char) -> *mut c_char
            let func: libloading::Symbol<
                unsafe extern "C" fn(*const std::os::raw::c_char) -> *mut std::os::raw::c_char,
            > = self
                .lib
                .get(func_name.as_bytes())
                .map_err(|e| e.to_string())?;

            // 1. Serialize args to JSON
            let json_args: Vec<serde_json::Value> = args.iter().map(value_to_json).collect();
            let payload = serde_json::Value::Array(json_args).to_string();
            let c_payload = std::ffi::CString::new(payload).map_err(|e| e.to_string())?;

            // 2. Call the function
            let res_ptr = func(c_payload.as_ptr());
            if res_ptr.is_null() {
                return Err(format!(
                    "Native plugin function '{func_name}' returned NULL pointer"
                ));
            }

            // 3. Convert return pointer back to string and Value
            let c_res = std::ffi::CStr::from_ptr(res_ptr);
            let res_str = c_res.to_string_lossy().to_string();

            // 4. Optionally deallocate the returned C string if a deallocator is exported
            if let Ok(free_func) = self
                .lib
                .get::<unsafe extern "C" fn(*mut std::os::raw::c_char)>(b"free_string")
            {
                free_func(res_ptr);
            }

            let ret_json: serde_json::Value =
                serde_json::from_str(&res_str).map_err(|e| e.to_string())?;
            Ok(json_to_value(&ret_json))
        }
    }
}

// =============================================================================
// WebAssembly Plugin Implementation (using wasmi)
// =============================================================================

pub struct WasmPlugin {
    #[allow(dead_code)]
    name: String,
    store: wasmi::Store<()>,
    instance: wasmi::Instance,
}

impl WasmPlugin {
    pub fn new(path: &str, name: &str) -> Result<Self, String> {
        let wasm_bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let engine = wasmi::Engine::default();
        let module = wasmi::Module::new(&engine, &wasm_bytes[..]).map_err(|e| e.to_string())?;
        let mut store = wasmi::Store::new(&engine, ());

        // Create empty Linker for wasmi imports (can be extended in the future)
        let linker = <wasmi::Linker<()>>::new(&engine);
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| e.to_string())?
            .start(&mut store)
            .map_err(|e| e.to_string())?;

        Ok(Self {
            name: name.to_string(),
            store,
            instance,
        })
    }
}

impl HayashiPlugin for WasmPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn call(&mut self, func_name: &str, args: &[Value]) -> Result<Value, String> {
        // Expose standard WASM interface:
        // Guest exports:
        //   alloc(size: i32) -> i32
        //   dealloc(ptr: i32, size: i32)
        //   [func_name](args_json_ptr: i32, args_json_len: i32) -> i64
        //
        // High 32 bits of returning i64 encode the pointer, low 32 bits encode length of return JSON.

        let alloc = self
            .instance
            .get_export(&self.store, "alloc")
            .and_then(|e| e.into_func())
            .ok_or_else(|| "WASM plugin missing 'alloc' function export".to_string())?
            .typed::<i32, i32>(&self.store)
            .map_err(|e| e.to_string())?;

        let dealloc = self
            .instance
            .get_export(&self.store, "dealloc")
            .and_then(|e| e.into_func())
            .ok_or_else(|| "WASM plugin missing 'dealloc' function export".to_string())?
            .typed::<(i32, i32), ()>(&self.store)
            .map_err(|e| e.to_string())?;

        let run_func = self
            .instance
            .get_export(&self.store, func_name)
            .and_then(|e| e.into_func())
            .ok_or_else(|| format!("WASM plugin missing export function '{func_name}'"))?
            .typed::<(i32, i32), i64>(&self.store)
            .map_err(|e| e.to_string())?;

        let memory = self
            .instance
            .get_export(&self.store, "memory")
            .and_then(|e| e.into_memory())
            .ok_or_else(|| "WASM plugin missing 'memory' export".to_string())?;

        // 1. Serialize args to JSON string
        let json_args: Vec<serde_json::Value> = args.iter().map(value_to_json).collect();
        let payload = serde_json::Value::Array(json_args).to_string();
        let payload_bytes = payload.as_bytes();
        let len = payload_bytes.len() as i32;

        // 2. Allocate memory on the Guest side
        let ptr = alloc
            .call(&mut self.store, len)
            .map_err(|e| e.to_string())?;

        // 3. Write payload into Guest memory
        memory
            .write(&mut self.store, ptr as usize, payload_bytes)
            .map_err(|e| e.to_string())?;

        // 4. Run the function
        let ret_encoded = run_func
            .call(&mut self.store, (ptr, len))
            .map_err(|e| e.to_string())?;

        // 5. Destructure returning i64 (high 32 bits = return ptr, low 32 bits = return len)
        let ret_ptr = (ret_encoded >> 32) as i32;
        let ret_len = (ret_encoded & 0xFFFFFFFF) as i32;

        // 6. Read returned JSON from Guest memory
        let mut ret_buf = vec![0u8; ret_len as usize];
        memory
            .read(&self.store, ret_ptr as usize, &mut ret_buf)
            .map_err(|e| e.to_string())?;

        // Deallocate arguments payload on Guest
        let _ = dealloc.call(&mut self.store, (ptr, len));

        // 7. Parse returned JSON and map back to Value
        let ret_str = String::from_utf8(ret_buf).map_err(|e| e.to_string())?;
        let ret_json: serde_json::Value =
            serde_json::from_str(&ret_str).map_err(|e| e.to_string())?;

        // Deallocate returned JSON buffer on Guest
        let _ = dealloc.call(&mut self.store, (ret_ptr, ret_len));

        Ok(json_to_value(&ret_json))
    }
}
