//! Host-facing API for embedding Hayashi.
//!
//! Provides a persistent interpreter session and a JSON representation of
//! Hayashi values (scalars, text, lists, dicts, Series, DataFrame and model
//! results). This is the data plane used by external hosts such as QNesto;
//! the DAP adapter remains the control/debug plane.
//!
//! The representation is intentionally additive: existing `run_hayashi` and
//! CLI behaviour are untouched.

use crate::lang::interpreter::model::Model;
use crate::lang::interpreter::model_view::ModelView;
use crate::lang::interpreter::value::{Series, Value};
use crate::lang::Interpreter;
use greeners::{Column, DataFrame};
use indexmap::IndexMap;
use ndarray::Array1;
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use std::sync::Arc;

/// Result of executing a source snippet in a session.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub output: String,
    pub stderr: String,
    pub error: Option<String>,
}

/// A persistent Hayashi interpreter plus value helpers.
pub struct Session {
    pub interp: Interpreter,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        let mut interp = Interpreter::new();
        interp.load_plugins();
        interp.set_auto_display(false);
        Self { interp }
    }

    /// Bind (or replace) a variable in the session.
    pub fn set_value(&mut self, name: &str, value: Value) -> Result<(), String> {
        self.interp
            .env
            .set(name, value)
            .map_err(|error| error.to_string())
    }

    /// Alias an existing variable without serializing it.
    pub fn set_ref(&mut self, name: &str, from: &str) -> Result<(), String> {
        let value = self
            .interp
            .env
            .get(from)
            .cloned()
            .ok_or_else(|| format!("unknown variable '{from}'"))?;
        self.set_value(name, value)
    }

    pub fn get_value(&self, name: &str) -> Option<Value> {
        self.interp.env.get(name).cloned()
    }

    pub fn remove(&mut self, name: &str) {
        self.interp.env.remove(name);
    }

    pub fn clear(&mut self) {
        for name in self.interp.env.all_names() {
            self.interp.env.remove(&name);
        }
    }

    pub fn list_variables(&self) -> Vec<(String, String, String)> {
        self.interp
            .env
            .all_names()
            .into_iter()
            .filter_map(|name| {
                self.interp
                    .env
                    .get(&name)
                    .map(|value| (name, value_kind(value).to_string(), value_summary(value)))
            })
            .collect()
    }

    /// Execute a source snippet, preserving the session state.
    pub fn run(&mut self, source: &str) -> RunResult {
        self.interp.set_auto_display(false);
        self.run_captured(source)
    }

    pub fn reset(&mut self) {
        let plugins = std::mem::take(&mut self.interp.plugins);
        let mut interp = Interpreter::new();
        interp.plugins = plugins;
        interp.set_auto_display(false);
        self.interp = interp;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run_captured(&mut self, source: &str) -> RunResult {
        use gag::BufferRedirect;
        use std::io::Read;

        let stdout_redirect = BufferRedirect::stdout().ok();
        let stderr_redirect = BufferRedirect::stderr().ok();
        let result = crate::lang::run_source(source, &mut self.interp);

        let mut output = String::new();
        let mut stderr = String::new();
        if let Some(mut redirect) = stdout_redirect {
            let _ = redirect.read_to_string(&mut output);
        }
        if let Some(mut redirect) = stderr_redirect {
            let _ = redirect.read_to_string(&mut stderr);
        }
        RunResult {
            output,
            stderr,
            error: result.err().map(|error| error.to_string()),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn run_captured(&mut self, source: &str) -> RunResult {
        crate::start_capture();
        let result = crate::lang::run_source(source, &mut self.interp);
        let output = crate::take_capture();
        RunResult {
            output,
            stderr: String::new(),
            error: result.err().map(|error| error.to_string()),
        }
    }
}

/// Short kind label used by hosts to select widgets/ports.
pub fn value_kind(value: &Value) -> &'static str {
    if is_model_value(value) {
        return "model";
    }
    match value {
        Value::Float(_) => "number",
        Value::Int(_) => "int",
        Value::Bool(_) => "bool",
        Value::Str(_) => "text",
        Value::DataFrame(_) => "dataframe",
        Value::Series(_) => "series",
        Value::List(_) => "list",
        Value::Dict(_) => "dict",
        Value::Nil => "nil",
        Value::Geometry(_) => "geometry",
        Value::Plot { .. } => "plot",
        _ => "opaque",
    }
}

/// One-line summary for lists, panels and tooltips.
pub fn value_summary(value: &Value) -> String {
    if let Some(view) = model_view(value) {
        return view.summary;
    }
    match value {
        Value::Float(number) => number.to_string(),
        Value::Int(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Str(text) => text.clone(),
        Value::DataFrame(frame) => format!(
            "DataFrame {}x{} ({})",
            frame.n_rows(),
            frame.column_names().len(),
            frame.column_names().join(", ")
        ),
        Value::Series(series) => format!("Series {} ({})", series.name, series.len()),
        Value::List(items) => format!("list({})", items.len()),
        Value::Dict(items) => format!("dict({})", items.len()),
        Value::Nil => "nil".into(),
        Value::Geometry(_) => "geometry".into(),
        Value::Plot { format, .. } => format!("plot({format})"),
        other => {
            let text = other.to_string();
            if text.len() > 120 {
                format!("{}…", &text[..120])
            } else {
                text
            }
        }
    }
}

fn is_model_value(value: &Value) -> bool {
    matches!(value, Value::OlsResult(_)) || value.to_model_view().is_some()
}

fn model_view(value: &Value) -> Option<ModelView> {
    match value {
        Value::OlsResult(model) => Some(model.to_model_view()),
        other => other.to_model_view(),
    }
}

/// Serialize any Hayashi value into the host JSON envelope.
pub fn value_to_json(value: &Value) -> Json {
    if let Some(view) = model_view(value) {
        return model_view_to_json(&view);
    }
    match value {
        Value::Float(number) => json!({"kind": "number", "value": number}),
        Value::Int(number) => json!({"kind": "int", "value": number}),
        Value::Bool(flag) => json!({"kind": "bool", "value": flag}),
        Value::Str(text) => json!({"kind": "text", "value": text}),
        Value::Nil => json!({"kind": "nil"}),
        Value::DataFrame(frame) => dataframe_to_json(frame),
        Value::Series(series) => json!({
            "kind": "series",
            "name": series.name,
            "values": series.values.iter().map(value_to_json).collect::<Vec<_>>(),
        }),
        Value::List(items) => json!({
            "kind": "list",
            "items": items.iter().map(value_to_json).collect::<Vec<_>>(),
        }),
        Value::Dict(items) => {
            let map: serde_json::Map<String, Json> = items
                .iter()
                .map(|(key, item)| (key.clone(), value_to_json(item)))
                .collect();
            json!({"kind": "dict", "items": map})
        }
        Value::Geometry(wkt) => json!({"kind": "geometry", "value": wkt}),
        Value::Plot { spec, format } => {
            json!({"kind": "plot", "format": format, "spec": spec})
        }
        Value::Model(view) => model_view_to_json(&view.to_model_view()),
        other => {
            // OLS and several estimator results are not covered by
            // `to_model_view`; use the display form as a safe fallback.
            json!({
                "kind": "opaque",
                "summary": value_summary(other),
            })
        }
    }
}

fn model_view_to_json(view: &ModelView) -> Json {
    let numbers = |values: &Array1<f64>| {
        values
            .iter()
            .map(|value| {
                if value.is_finite() {
                    json!(value)
                } else {
                    Json::Null
                }
            })
            .collect::<Vec<_>>()
    };
    let mut stats = serde_json::Map::new();
    for (key, value) in &view.fit {
        stats.insert(key.clone(), value_to_json(value));
    }
    let mut body = json!({
        "kind": "model",
        "modelType": view.type_name,
        "summary": view.summary,
        "terms": view.variable_names,
        "estimate": numbers(&view.params),
        "stdError": numbers(&view.std_errors),
        "statistic": numbers(&view.test_values),
        "pValue": numbers(&view.p_values),
        "stats": stats,
    });
    if let Some(lower) = &view.conf_lower {
        body["confLow"] = json!(numbers(lower));
    }
    if let Some(upper) = &view.conf_upper {
        body["confHigh"] = json!(numbers(upper));
    }
    if let Some(residuals) = &view.residuals {
        body["residuals"] = json!(numbers(residuals));
    }
    if let Some(fitted) = &view.fitted_values {
        body["fitted"] = json!(numbers(fitted));
    }
    if let Some(n) = view.n_obs() {
        body["n"] = json!(n);
    }
    body
}

fn dataframe_to_json(frame: &DataFrame) -> Json {
    let columns: Vec<Json> = frame
        .column_names()
        .into_iter()
        .filter_map(|name| {
            let column = frame.get_column(&name).ok()?;
            Some(column_to_json(&name, column))
        })
        .collect();
    json!({
        "kind": "dataframe",
        "n": frame.n_rows(),
        "columns": columns,
    })
}

fn column_to_json(name: &str, column: &Column) -> Json {
    match column {
        Column::Float(values) => json!({
            "name": name,
            "type": "float",
            "values": values.iter().map(|v| {
                if v.is_finite() { json!(v) } else { Json::Null }
            }).collect::<Vec<_>>(),
        }),
        Column::Int(values) => json!({
            "name": name,
            "type": "int",
            "values": values.iter().map(|v| json!(v)).collect::<Vec<_>>(),
        }),
        Column::Bool(values) => json!({
            "name": name,
            "type": "bool",
            "values": values.iter().map(|v| json!(v)).collect::<Vec<_>>(),
        }),
        Column::String(values) => json!({
            "name": name,
            "type": "string",
            "values": values.iter().map(|v| json!(v)).collect::<Vec<_>>(),
        }),
        Column::Categorical(values) => {
            let items: Vec<Json> = values
                .codes
                .iter()
                .map(|code| {
                    values
                        .get_level(*code)
                        .map(|level| json!(level))
                        .unwrap_or(Json::Null)
                })
                .collect();
            json!({"name": name, "type": "categorical", "values": items})
        }
        Column::DateTime(values) => json!({
            "name": name,
            "type": "datetime",
            "values": values.iter().map(|v| json!(v.to_string())).collect::<Vec<_>>(),
        }),
    }
}

/// Parse the host JSON envelope back into a Hayashi value.
///
/// Models are deliberately not reconstructed here: pass them by reference
/// (`set_ref`) inside the session. DataFrames, scalars, lists and dicts are
/// accepted.
pub fn value_from_json(value: &Json) -> Result<Value, String> {
    let kind = value
        .get("kind")
        .and_then(Json::as_str)
        .ok_or_else(|| "value.kind is required".to_string())?;
    match kind {
        "number" => value
            .get("value")
            .and_then(Json::as_f64)
            .map(Value::Float)
            .ok_or_else(|| "number.value is required".to_string()),
        "int" => value
            .get("value")
            .and_then(Json::as_i64)
            .map(Value::Int)
            .ok_or_else(|| "int.value is required".to_string()),
        "bool" => value
            .get("value")
            .and_then(Json::as_bool)
            .map(Value::Bool)
            .ok_or_else(|| "bool.value is required".to_string()),
        "text" => value
            .get("value")
            .and_then(Json::as_str)
            .map(|text| Value::Str(text.to_string()))
            .ok_or_else(|| "text.value is required".to_string()),
        "nil" => Ok(Value::Nil),
        "list" => {
            let items = value
                .get("items")
                .and_then(Json::as_array)
                .ok_or_else(|| "list.items is required".to_string())?;
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(value_from_json(item)?);
            }
            Ok(Value::List(Arc::new(values)))
        }
        "dict" => {
            let items = value
                .get("items")
                .and_then(Json::as_object)
                .ok_or_else(|| "dict.items is required".to_string())?;
            let mut map = HashMap::new();
            for (key, item) in items {
                map.insert(key.clone(), value_from_json(item)?);
            }
            Ok(Value::Dict(Arc::new(map)))
        }
        "dataframe" => Ok(Value::DataFrame(Arc::new(dataframe_from_json(value)?))),
        "series" => {
            let name = value.get("name").and_then(Json::as_str).unwrap_or("series");
            let items = value
                .get("values")
                .and_then(Json::as_array)
                .ok_or_else(|| "series.values is required".to_string())?;
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(value_from_json(item)?);
            }
            Ok(Value::Series(Arc::new(Series::new(name, values))))
        }
        "model" => Err("models must be passed by reference (set_ref)".into()),
        other => Err(format!("unsupported value kind '{other}'")),
    }
}

fn dataframe_from_json(value: &Json) -> Result<DataFrame, String> {
    let columns = value
        .get("columns")
        .and_then(Json::as_array)
        .ok_or_else(|| "dataframe.columns is required".to_string())?;
    let mut typed: IndexMap<String, Column> = IndexMap::new();
    for column in columns {
        let name = column
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| "column.name is required".to_string())?
            .to_string();
        let kind = column.get("type").and_then(Json::as_str).unwrap_or("float");
        let values = column
            .get("values")
            .and_then(Json::as_array)
            .ok_or_else(|| format!("column '{name}' has no values"))?;
        let built = match kind {
            "float" | "number" => {
                let mut data = Vec::with_capacity(values.len());
                for item in values {
                    match item.as_f64() {
                        Some(number) => data.push(number),
                        None => data.push(f64::NAN),
                    }
                }
                Column::Float(Array1::from(data))
            }
            "int" => {
                let mut data = Vec::with_capacity(values.len());
                for item in values {
                    data.push(item.as_i64().unwrap_or(0));
                }
                Column::Int(Array1::from(data))
            }
            "bool" => {
                let mut data = Vec::with_capacity(values.len());
                for item in values {
                    data.push(item.as_bool().unwrap_or(false));
                }
                Column::Bool(Array1::from(data))
            }
            "categorical" => {
                let data: Vec<String> = values
                    .iter()
                    .map(|item| item.as_str().unwrap_or("").to_string())
                    .collect();
                Column::from_strings(data)
            }
            "string" | "text" => {
                let data: Vec<String> = values
                    .iter()
                    .map(|item| item.as_str().unwrap_or("").to_string())
                    .collect();
                Column::String(Array1::from(data))
            }
            other => return Err(format!("unsupported column type '{other}'")),
        };
        typed.insert(name, built);
    }
    DataFrame::from_columns(typed).map_err(|error| error.to_string())
}

/// Convenience for hosts that only need the model view.
pub fn model_json(value: &Value) -> Option<Json> {
    model_view(value).map(|view| model_view_to_json(&view))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_session_roundtrip() {
        let mut session = Session::new();
        session.set_value("x", Value::Float(10.0)).unwrap();
        let run = session.run("let y = x * 2\nprint(y)");
        assert!(run.error.is_none(), "{:?}", run.error);
        assert!(run.output.contains("20"));
        assert!(
            matches!(session.get_value("y"), Some(Value::Float(v)) if (v - 20.0).abs() < 1e-12)
        );
    }

    #[test]
    fn dataframe_roundtrip() {
        let json = serde_json::json!({
            "kind": "dataframe",
            "n": 3,
            "columns": [
                {"name": "x", "type": "float", "values": [1, 2, 3]},
                {"name": "y", "type": "float", "values": [3, 5, 7]}
            ]
        });
        let value = value_from_json(&json).unwrap();
        let Value::DataFrame(frame) = &value else {
            panic!("not a dataframe")
        };
        assert_eq!(frame.n_rows(), 3);
        assert_eq!(frame.column_names(), vec!["x", "y"]);
        let exported = value_to_json(&value);
        assert_eq!(exported["kind"], "dataframe");
        assert_eq!(exported["columns"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn ols_model_is_exported_typed() {
        let mut session = Session::new();
        let json = serde_json::json!({
            "kind": "dataframe",
            "n": 5,
            "columns": [
                {"name": "x", "type": "float", "values": [1, 2, 3, 4, 5]},
                {"name": "y", "type": "float", "values": [2, 4, 6, 8, 10]}
            ]
        });
        session
            .set_value("df", value_from_json(&json).unwrap())
            .unwrap();
        let run = session.run("let m = ols(y ~ x, df)");
        assert!(run.error.is_none(), "{:?}", run.error);
        let model = session.get_value("m").expect("model");
        assert_eq!(value_kind(&model), "model");
        let exported = value_to_json(&model);
        assert_eq!(exported["kind"], "model");
        assert_eq!(exported["modelType"], "OlsResult");
        assert!(exported["estimate"].as_array().unwrap().len() >= 1);
    }
}
