// ── Manipulação de DataFrame ─────────────────────────────────────────────────
use super::super::*;

/// Extracts a column as Array1<f64>; accepts Float, Int, Bool, Categorical, etc.
pub(in crate::lang::interpreter) fn get_col_f64(
    df: &DataFrame,
    name: &str,
) -> Result<ndarray::Array1<f64>> {
    let col = df
        .get_column(name)
        .map_err(|_| HayashiError::Runtime(format!("column '{name}' not found")))?;
    Ok(col.to_float())
}

/// Rebuilds X from the model's variable name list.
/// `_cons`/`const`/`Intercept` → column of 1s; others → columns from df.
pub(in crate::lang::interpreter) fn build_x_from_varnames(
    df: &DataFrame,
    names: &[String],
) -> Result<ndarray::Array2<f64>> {
    let n = df.n_rows();
    let k = names.len();
    let mut x = ndarray::Array2::<f64>::zeros((n, k));
    for (j, name) in names.iter().enumerate() {
        match name.as_str() {
            "_cons" | "const" | "Intercept" | "(Intercept)" => {
                x.column_mut(j).fill(1.0);
            }
            other => {
                let col = get_col_f64(df, other).map_err(|_| {
                    HayashiError::Runtime(format!(
                        "predict: column '{other}' not found in DataFrame"
                    ))
                })?;
                x.column_mut(j).assign(&col);
            }
        }
    }
    Ok(x)
}
