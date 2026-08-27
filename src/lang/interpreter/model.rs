use crate::lang::error::{HayashiError, Result};
use crate::lang::interpreter::model_view::ModelView;
use crate::lang::interpreter::value::Value;
use ndarray::Array1;
use serde_json;
use std::any::Any;

/// Common interface for all Hayashi model results.
///
/// This trait replaces the scattered `match` over `Value::*Result` variants.
/// Each adapter wraps a Greeners result and exposes the operations needed by
/// the interpreter: display, summary, coefficient extraction, prediction,
/// residual access, field lookup, and DAP/JSON serialization.
pub trait Model: std::fmt::Display + Any {
    /// Estimator name shown to the user, e.g. "OlsResult".
    fn type_name(&self) -> &str;

    /// Downcasting support for estimator-specific logic (margins, DAP, diagnostics).
    fn as_any(&self) -> &dyn Any;

    /// One-line summary for DAP hover and `summary` field access.
    fn summary(&self) -> String;

    /// Canonical view used by `tidy()`, `glance()`, `esttab`, DAP, etc.
    fn to_model_view(&self) -> ModelView;

    /// `tidy()` output: a Dict with columns `variable`, `coef`, `std_err`,
    /// `t`, `p_value`, `conf_low`, `conf_high`.
    fn tidy(&self) -> Value {
        let mv = self.to_model_view();
        Value::Dict(std::sync::Arc::new(mv.to_tidy_map()))
    }

    /// `glance()` output: a Dict of fit statistics.
    fn glance(&self) -> Value {
        let mv = self.to_model_view();
        Value::Dict(std::sync::Arc::new(mv.to_glance_map()))
    }

    /// In-sample prediction or residual extraction.
    ///
    /// `kind` is one of the supported predict keys (e.g. "xb", "fitted",
    /// "residuals"). `newdata` is `None` for in-sample; some adapters may
    /// support out-of-sample prediction when a DataFrame is supplied.
    fn predict(&self, kind: &str, _newdata: Option<&greeners::DataFrame>) -> Result<Vec<f64>> {
        Err(HayashiError::Runtime(format!(
            "predict: kind '{kind}' not supported for {}",
            self.type_name()
        )))
    }

    /// Residuals, when available.
    fn residuals(&self) -> Option<Array1<f64>> {
        None
    }

    /// Fitted values, when available or cheaply computable.
    fn fitted_values(&self) -> Option<Array1<f64>> {
        None
    }

    /// Field access for `model.field` and `model[\"field\"]`.
    fn field(&self, name: &str) -> Result<Value> {
        Err(HayashiError::Runtime(format!(
            "{} has no field '{name}'",
            self.type_name()
        )))
    }

    /// JSON serialization for plugins and `json(model)`.
    fn to_json(&self) -> serde_json::Value {
        serde_json::Value::Null
    }
}

/// Generic model result backed by a precomputed `ModelView`.
///
/// Used for estimators that only need display, `tidy`/`glance`, and DAP
/// expansion, without type-specific `predict` or field logic. It allows
/// wrapping any `ModelView` behind the `Model` trait so the interpreter can
/// treat it uniformly.
#[derive(Clone)]
pub struct GenericModel {
    view: ModelView,
    fields: std::collections::HashMap<String, Value>,
}

impl GenericModel {
    pub fn new(view: ModelView, fields: std::collections::HashMap<String, Value>) -> Self {
        Self { view, fields }
    }

    pub fn view(&self) -> &ModelView {
        &self.view
    }

    pub fn fields(&self) -> &std::collections::HashMap<String, Value> {
        &self.fields
    }
}

impl std::fmt::Display for GenericModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.view.summary)
    }
}

impl Model for GenericModel {
    fn type_name(&self) -> &str {
        &self.view.type_name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn summary(&self) -> String {
        self.view.summary.clone()
    }

    fn to_model_view(&self) -> ModelView {
        self.view.clone()
    }

    fn field(&self, name: &str) -> Result<Value> {
        self.fields
            .get(name)
            .cloned()
            .ok_or_else(|| HayashiError::Runtime(format!("field '{name}' not found")))
    }
}
