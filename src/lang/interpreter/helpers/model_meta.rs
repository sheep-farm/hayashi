// ── Extração de coeficientes e metadados de modelos ──────────────────────────
use super::super::*;

/// Generates coefficient names from the formula and observed categories.
pub(in crate::lang::interpreter) fn coef_names_from_formula(
    formula_ast: &Formula,
    df: &DataFrame,
    n_cols: usize,
) -> Vec<String> {
    let mut names: Vec<String> = vec!["_cons".into()];
    for term in &formula_ast.rhs {
        match term {
            RhsTerm::Categorical(e) => {
                // Para C(Var(v)) simples extraímos os níveis do df
                if let Expr::Var(v) = e.as_ref() {
                    let raw = col_to_strings(df, v).unwrap_or_default();
                    let mut unique: Vec<String> = raw
                        .into_iter()
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    sort_maybe_numeric_strings(&mut unique);
                    for val in unique.into_iter().skip(1) {
                        names.push(format!("{v}={val}"));
                    }
                } else {
                    names.push(term.display_name());
                }
            }
            other => names.push(other.display_name()),
        }
    }
    names.truncate(n_cols);
    while names.len() < n_cols {
        names.push(format!("x{}", names.len() + 1));
    }
    names
}
