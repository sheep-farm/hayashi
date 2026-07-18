// ── Ordenação de strings ──────────────────────────────────────────────────────

/// Converts a finite numeric string, if possible.
pub(in crate::lang::interpreter) fn finite_numeric_string(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Sorts strings numerically if all are finite numeric; otherwise alphabetically.
pub(in crate::lang::interpreter) fn sort_maybe_numeric_strings(values: &mut [String]) {
    if values
        .iter()
        .all(|value| finite_numeric_string(value).is_some())
    {
        values.sort_by(
            |a, b| match (finite_numeric_string(a), finite_numeric_string(b)) {
                (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                _ => a.cmp(b),
            },
        );
    } else {
        values.sort();
    }
}
