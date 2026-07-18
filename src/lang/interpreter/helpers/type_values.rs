// ── Conversão de tipos e valores ─────────────────────────────────────────────
use super::super::*;
use std::cmp::Ordering;

/// Comparator for `f64` that treats `NaN` as greater than any finite value
/// (matching Stata's convention where missing sorts last in ascending order).
/// This avoids panics from `partial_cmp(...).unwrap()` when data contains NaN.
pub(in crate::lang::interpreter) fn nan_last_cmp(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (true, true) => Ordering::Equal,
    }
}

/// Builds a rendered diagnostic value.
pub(in crate::lang::interpreter) fn diag(rendered: String) -> Value {
    Value::DiagResult(Rc::new(DiagResult {
        rendered,
        fields: HashMap::new(),
    }))
}

/// Builds a rendered diagnostic value with structured fields for DAP/debug.
pub(in crate::lang::interpreter) fn diag_with(
    rendered: String,
    fields: HashMap<String, Value>,
) -> Value {
    Value::DiagResult(Rc::new(DiagResult { rendered, fields }))
}

/// Converts `Value` to boolean permissively.
pub(in crate::lang::interpreter) fn value_as_bool(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Int(i) => *i != 0,
        Value::Float(f) => *f != 0.0 && !f.is_nan(),
        Value::Nil => false,
        _ => true,
    }
}

/// Extracts estimated coefficients from a model result.
pub(in crate::lang::interpreter) fn extract_params(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::OlsResult(m) => Some(m.result.params.to_vec()),
        Value::BinaryResult(m) => Some(m.result.params.to_vec()),
        Value::PenalizedResult(m) => Some(m.params.to_vec()),
        Value::PoissonResult(r) => Some(r.params.to_vec()),
        Value::NegBinResult(r) => Some(r.params.to_vec()),
        Value::QuantileResult(r) => Some(r.params.to_vec()),
        Value::PanelResult(r) => Some(r.params.to_vec()),
        Value::TobitResult(r) => Some(r.params.to_vec()),
        _ => None,
    }
}

/// Extracts standard errors from a model result.
pub(in crate::lang::interpreter) fn extract_se(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::OlsResult(m) => Some(m.result.std_errors.to_vec()),
        Value::BinaryResult(m) => Some(m.result.std_errors.to_vec()),
        Value::PenalizedResult(m) => Some(m.std_errors.to_vec()),
        Value::PoissonResult(r) => Some(r.std_errors.to_vec()),
        Value::NegBinResult(r) => Some(r.std_errors.to_vec()),
        Value::QuantileResult(r) => Some(r.std_errors.to_vec()),
        Value::PanelResult(r) => Some(r.std_errors.to_vec()),
        Value::TobitResult(r) => Some(r.std_errors.to_vec()),
        _ => None,
    }
}

/// Extracts coefficient names from a model result.
pub(in crate::lang::interpreter) fn extract_var_names(v: &Value) -> Vec<String> {
    match v {
        Value::OlsResult(m) => m.result.variable_names.clone().unwrap_or_default(),
        Value::BinaryResult(m) => m.coef_names.clone(),
        Value::PenalizedResult(m) => m.variable_names.clone(),
        Value::PoissonResult(r) => r.variable_names.clone().unwrap_or_default(),
        Value::NegBinResult(r) => r.variable_names.clone().unwrap_or_default(),
        Value::QuantileResult(r) => r.variable_names.clone().unwrap_or_default(),
        Value::PanelResult(r) => r.variable_names.clone().unwrap_or_default(),
        Value::TobitResult(r) => r.variable_names.clone().unwrap_or_default(),
        _ => vec![],
    }
}

/// Converts `Value` to `f64`.
pub(in crate::lang::interpreter) fn value_as_f64(v: &Value) -> Result<f64> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => Err(HayashiError::Type("expected numeric value".into())),
    }
}
