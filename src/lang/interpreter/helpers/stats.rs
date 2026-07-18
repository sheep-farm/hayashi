// ── Funções estatísticas ──────────────────────────────────────────────────────

/// Φ(x) normal CDF — Abramowitz & Stegun 26.2.17 (error < 7.5e-8).
pub(in crate::lang::interpreter) fn norm_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let phi = 1.0 - greeners::norm_pdf(x) * poly;
    if x >= 0.0 {
        phi
    } else {
        1.0 - phi
    }
}
