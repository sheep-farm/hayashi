use super::interpreter::Value;
use arrow::array::{
    make_array, Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray,
};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use greeners::Column;
use ndarray::Array1;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

/// Unified Hayashi Plugin Trait
#[allow(dead_code)]
pub trait HayashiPlugin {
    fn name(&self) -> &str;
    fn call(&mut self, func_name: &str, args: &[Value]) -> Result<Value, String>;
}

/// Converte uma coluna do Greeners em um ArrayRef do Arrow.
pub fn column_to_arrow(col: &Column) -> ArrayRef {
    match col {
        Column::Float(arr) => {
            let vec = arr.to_vec();
            Arc::new(Float64Array::from(vec)) as ArrayRef
        }
        Column::Int(arr) => {
            let vec = arr.to_vec();
            Arc::new(Int64Array::from(vec)) as ArrayRef
        }
        Column::Bool(arr) => {
            let vec = arr.to_vec();
            Arc::new(BooleanArray::from(vec)) as ArrayRef
        }
        Column::String(arr) => {
            let vec = arr.to_vec();
            Arc::new(StringArray::from(vec)) as ArrayRef
        }
        Column::Categorical(cat) => {
            let vec = cat.to_strings();
            Arc::new(StringArray::from(vec)) as ArrayRef
        }
        Column::DateTime(arr) => {
            let vec: Vec<String> = arr.iter().map(|dt| dt.to_string()).collect();
            Arc::new(StringArray::from(vec)) as ArrayRef
        }
    }
}

/// Converte um DataFrame do Greeners em um StructArray do Arrow (retornado como ArrayRef).
pub fn dataframe_to_arrow(df: &greeners::DataFrame) -> ArrayRef {
    use arrow::array::StructArray;
    use arrow::datatypes::{Field, Fields};

    let mut fields = Vec::new();
    let mut arrays = Vec::new();

    for col_name in df.column_names() {
        if let Ok(col) = df.get_column(&col_name) {
            let array = column_to_arrow(col);
            fields.push(Field::new(&col_name, array.data_type().clone(), true));
            arrays.push(array);
        }
    }

    let struct_array = StructArray::try_new(Fields::from(fields), arrays, None).unwrap();
    Arc::new(struct_array) as ArrayRef
}

/// Converte um StructArray do Arrow em um DataFrame do Greeners.
pub fn arrow_to_dataframe(array: &ArrayRef) -> Result<greeners::DataFrame, String> {
    use arrow::array::StructArray;
    use arrow::datatypes::DataType;

    match array.data_type() {
        DataType::Struct(fields) => {
            let struct_array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| "failed to downcast StructArray".to_string())?;

            let mut columns = HashMap::new();
            for (i, field) in fields.iter().enumerate() {
                let col_name = field.name().clone();
                let col_array = struct_array.column(i);
                let col = arrow_to_column(col_array)?;
                columns.insert(col_name, col);
            }

            greeners::DataFrame::from_columns(columns)
                .map_err(|e| format!("failed to build DataFrame: {e}"))
        }
        other => Err(format!("expected DataType::Struct, got {:?}", other)),
    }
}

/// Converte um ArrayRef do Arrow em uma coluna do Greeners.
pub fn arrow_to_column(array: &ArrayRef) -> Result<Column, String> {
    use arrow::datatypes::DataType;

    let len = array.len();
    match array.data_type() {
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| "failed to downcast Float64Array".to_string())?;
            let vec: Vec<f64> = (0..len)
                .map(|i| {
                    if arr.is_null(i) {
                        f64::NAN
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Ok(Column::Float(Array1::from(vec)))
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| "failed to downcast Int64Array".to_string())?;
            let vec: Vec<i64> = (0..len)
                .map(|i| if arr.is_null(i) { 0 } else { arr.value(i) })
                .collect();
            Ok(Column::Int(Array1::from(vec)))
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| "failed to downcast BooleanArray".to_string())?;
            let vec: Vec<bool> = (0..len)
                .map(|i| if arr.is_null(i) { false } else { arr.value(i) })
                .collect();
            Ok(Column::Bool(Array1::from(vec)))
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| "failed to downcast StringArray".to_string())?;
            let vec: Vec<String> = (0..len)
                .map(|i| {
                    if arr.is_null(i) {
                        "".to_string()
                    } else {
                        arr.value(i).to_string()
                    }
                })
                .collect();
            Ok(Column::String(Array1::from(vec)))
        }
        other => Err(format!("unsupported Arrow type for Column: {:?}", other)),
    }
}

/// Converte uma coluna do Greeners em um Value::List do Hayashi.
pub fn column_to_value(col: &Column) -> Value {
    match col {
        Column::Float(arr) => Value::List(Rc::new(arr.iter().map(|&x| Value::Float(x)).collect())),
        Column::Int(arr) => Value::List(Rc::new(arr.iter().map(|&x| Value::Int(x)).collect())),
        Column::Bool(arr) => Value::List(Rc::new(arr.iter().map(|&x| Value::Bool(x)).collect())),
        Column::String(arr) => {
            Value::List(Rc::new(arr.iter().map(|s| Value::Str(s.clone())).collect()))
        }
        Column::Categorical(cat) => Value::List(Rc::new(
            cat.to_strings().into_iter().map(Value::Str).collect(),
        )),
        Column::DateTime(arr) => Value::List(Rc::new(
            arr.iter().map(|dt| Value::Str(dt.to_string())).collect(),
        )),
    }
}

/// Converts a Hayashi list into a Greeners column if it is homogeneous and primitive.
pub fn list_to_column(lst: &[Value]) -> Option<Column> {
    if lst.is_empty() {
        return None;
    }

    match &lst[0] {
        Value::Float(_) => {
            let mut vec = Vec::with_capacity(lst.len());
            for v in lst {
                match v {
                    Value::Float(f) => vec.push(*f),
                    Value::Int(i) => vec.push(*i as f64),
                    _ => return None,
                }
            }
            Some(Column::Float(Array1::from(vec)))
        }
        Value::Int(_) => {
            let mut vec = Vec::with_capacity(lst.len());
            for v in lst {
                match v {
                    Value::Int(i) => vec.push(*i),
                    Value::Float(f) => vec.push(*f as i64),
                    _ => return None,
                }
            }
            Some(Column::Int(Array1::from(vec)))
        }
        Value::Bool(_) => {
            let mut vec = Vec::with_capacity(lst.len());
            for v in lst {
                match v {
                    Value::Bool(b) => vec.push(*b),
                    _ => return None,
                }
            }
            Some(Column::Bool(Array1::from(vec)))
        }
        Value::Str(_) => {
            let mut vec = Vec::with_capacity(lst.len());
            for v in lst {
                match v {
                    Value::Str(s) => vec.push(s.clone()),
                    _ => return None,
                }
            }
            Some(Column::String(Array1::from(vec)))
        }
        _ => None,
    }
}

/// Helper to serialize Value into JSON for WASM/FFI exchanges
pub fn value_to_json(
    val: &Value,
    use_arrow: bool,
    temp_boxes: &mut Vec<(usize, usize)>,
) -> serde_json::Value {
    match val {
        Value::Float(f) => serde_json::json!(f),
        Value::Int(i) => serde_json::json!(i),
        Value::Bool(b) => serde_json::json!(b),
        Value::Str(s) => serde_json::json!(s),
        Value::Nil => serde_json::Value::Null,
        Value::List(lst) => {
            if use_arrow && !lst.is_empty() {
                if let Some(col) = list_to_column(lst) {
                    let arrow_array = column_to_arrow(&col);
                    if let Ok((ffi_array, ffi_schema)) =
                        arrow::ffi::to_ffi(&arrow_array.into_data())
                    {
                        let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;
                        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
                        temp_boxes.push((array_ptr, schema_ptr));

                        let mut col_map = serde_json::Map::new();
                        col_map.insert(
                            "__arrow_array_ptr__".to_string(),
                            serde_json::json!(array_ptr),
                        );
                        col_map.insert(
                            "__arrow_schema_ptr__".to_string(),
                            serde_json::json!(schema_ptr),
                        );
                        return serde_json::Value::Object(col_map);
                    }
                }
            }
            let arr: Vec<serde_json::Value> = lst
                .iter()
                .map(|v| value_to_json(v, use_arrow, temp_boxes))
                .collect();
            serde_json::Value::Array(arr)
        }
        Value::Dict(dct) => {
            let mut map = serde_json::Map::new();
            for (k, v) in dct.iter() {
                map.insert(k.clone(), value_to_json(v, use_arrow, temp_boxes));
            }
            serde_json::Value::Object(map)
        }
        Value::DataFrame(df) => {
            if use_arrow {
                let arrow_array = dataframe_to_arrow(df);
                if let Ok((ffi_array, ffi_schema)) = arrow::ffi::to_ffi(&arrow_array.into_data()) {
                    let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;
                    let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
                    temp_boxes.push((array_ptr, schema_ptr));

                    let mut df_map = serde_json::Map::new();
                    df_map.insert(
                        "__arrow_array_ptr__".to_string(),
                        serde_json::json!(array_ptr),
                    );
                    df_map.insert(
                        "__arrow_schema_ptr__".to_string(),
                        serde_json::json!(schema_ptr),
                    );
                    return serde_json::Value::Object(df_map);
                }
            }
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
pub fn json_to_value(
    jval: &serde_json::Value,
    returned_arrow_ptrs: &mut Vec<(usize, usize)>,
    host_allocated: &std::collections::HashSet<usize>,
) -> Value {
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
            let lst: Vec<Value> = arr
                .iter()
                .map(|v| json_to_value(v, returned_arrow_ptrs, host_allocated))
                .collect();
            Value::List(Rc::new(lst))
        }
        serde_json::Value::Object(obj) => {
            if let (Some(arr_val), Some(sch_val)) = (
                obj.get("__arrow_array_ptr__"),
                obj.get("__arrow_schema_ptr__"),
            ) {
                if let (Some(arr_ptr), Some(sch_ptr)) = (arr_val.as_u64(), sch_val.as_u64()) {
                    let array_ptr = arr_ptr as *mut FFI_ArrowArray;
                    let schema_ptr = sch_ptr as *mut FFI_ArrowSchema;
                    let is_host = host_allocated.contains(&(arr_ptr as usize));
                    unsafe {
                        if let Ok(array_data) =
                            arrow::ffi::from_ffi(std::ptr::read(array_ptr), &*schema_ptr)
                        {
                            let array_ref = make_array(array_data);
                            if let arrow::datatypes::DataType::Struct(_) = array_ref.data_type() {
                                if let Ok(df) = arrow_to_dataframe(&array_ref) {
                                    if !is_host {
                                        returned_arrow_ptrs
                                            .push((arr_ptr as usize, sch_ptr as usize));
                                    }
                                    return Value::DataFrame(Rc::new(df));
                                }
                            } else {
                                if let Ok(col) = arrow_to_column(&array_ref) {
                                    if !is_host {
                                        returned_arrow_ptrs
                                            .push((arr_ptr as usize, sch_ptr as usize));
                                    }
                                    return column_to_value(&col);
                                }
                            }
                        }
                    }
                }
            }

            let mut map = HashMap::new();
            for (k, v) in obj.iter() {
                map.insert(
                    k.clone(),
                    json_to_value(v, returned_arrow_ptrs, host_allocated),
                );
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
            let func: libloading::Symbol<
                unsafe extern "C" fn(*const std::os::raw::c_char) -> *mut std::os::raw::c_char,
            > = self
                .lib
                .get(func_name.as_bytes())
                .map_err(|e| e.to_string())?;

            // 1. Serialize args to JSON (collecting host FFI boxes in temp_boxes)
            let mut temp_boxes = Vec::new();
            let json_args: Vec<serde_json::Value> = args
                .iter()
                .map(|v| value_to_json(v, true, &mut temp_boxes))
                .collect();
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

            // 4. Deallocate the returned C string
            if let Ok(free_func) = self
                .lib
                .get::<unsafe extern "C" fn(*mut std::os::raw::c_char)>(b"free_string")
            {
                free_func(res_ptr);
            }

            // 5. Deserialize JSON, collecting any guest-allocated Arrow pointers
            let ret_json: serde_json::Value =
                serde_json::from_str(&res_str).map_err(|e| e.to_string())?;

            let host_allocated_set: std::collections::HashSet<usize> =
                temp_boxes.iter().map(|(arr, _)| *arr).collect();
            let mut returned_arrow_ptrs = Vec::new();
            let val = json_to_value(&ret_json, &mut returned_arrow_ptrs, &host_allocated_set);

            // 6. Free host-allocated FFI boxes after deserialization has reconstructed the data
            for &(arr_ptr, sch_ptr) in &temp_boxes {
                let mut arr_box = Box::from_raw(arr_ptr as *mut FFI_ArrowArray);
                arr_box.release = None;
                drop(arr_box);
                let mut sch_box = Box::from_raw(sch_ptr as *mut FFI_ArrowSchema);
                sch_box.release = None;
                drop(sch_box);
            }

            // 7. Clean up guest-allocated Arrow pointers using plugin's free_arrow_pointers hook
            if !returned_arrow_ptrs.is_empty() {
                if let Ok(free_arrow_func) = self
                    .lib
                    .get::<unsafe extern "C" fn(*mut FFI_ArrowArray, *mut FFI_ArrowSchema)>(
                        b"free_arrow_pointers",
                    )
                {
                    for (arr_ptr, sch_ptr) in returned_arrow_ptrs {
                        let arr = arr_ptr as *mut FFI_ArrowArray;
                        let sch = sch_ptr as *mut FFI_ArrowSchema;
                        if !arr.is_null() {
                            (*arr).release = None;
                        }
                        if !sch.is_null() {
                            (*sch).release = None;
                        }
                        free_arrow_func(arr, sch);
                    }
                }
            }

            Ok(val)
        }
    }
}

// =============================================================================
// WebAssembly Plugin Implementation (using wasmi)
// =============================================================================

#[cfg(feature = "wasm")]
pub struct WasmPlugin {
    #[allow(dead_code)]
    name: String,
    store: wasmi::Store<()>,
    instance: wasmi::Instance,
}

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
impl HayashiPlugin for WasmPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn call(&mut self, func_name: &str, args: &[Value]) -> Result<Value, String> {
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

        // 1. Serialize args to JSON string (Arrow is not used for WASM sandbox)
        let mut temp_boxes = Vec::new();
        let json_args: Vec<serde_json::Value> = args
            .iter()
            .map(|v| value_to_json(v, false, &mut temp_boxes))
            .collect();
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

        // 5. Destructure returning i64
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

        let mut returned_arrow_ptrs = Vec::new();
        Ok(json_to_value(
            &ret_json,
            &mut returned_arrow_ptrs,
            &std::collections::HashSet::new(),
        ))
    }
}
