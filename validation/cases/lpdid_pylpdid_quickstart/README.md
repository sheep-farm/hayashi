# LP-DiD quickstart validation (pylpdid)

This case validates Hayashi's `lpdid` command against the Python `pylpdid`
reference on the quickstart example from `pylpdid/examples/01_quickstart.py`.

## DGP

An absorbing staggered-adoption panel with 200 units and 15 periods. Cohorts
are `g ∈ {0, 6, 9, 12}` (0 = never treated). The outcome is

```
y_it = unit_fe_i + 0.3 * t + 2.0 * treated_it + ε_it
```

where `unit_fe_i ~ N(0, 1)` and `ε_it ~ N(0, 0.5)`.

## Files

- `data/gen.hay` — generates `data/panel.csv` from the DGP above.
- `data/panel.csv` — the generated panel (committed for reproducibility).
- `hayashi/run.hay` — runs Hayashi's `lpdid` and exports the event-study
coefficients as a CSV.
- `reference/run.py` — runs `pylpdid` on the same panel and emits JSON with
coefficients and standard errors.

## Reference

- Python: `pylpdid` (Dube, Girardi, Jordà & Taylor 2025).

## Tolerances

- `coefficients`: `1e-6`
- `standard_errors`: `1e-4`

These tolerances are tight because both sides use the same OLS point estimate
and a cluster-robust covariance estimator grouped by unit.

## Attribution

The reference implementation `pylpdid` is copyright (c) 2026 Daniel de Abreu
Pereira Uhr and is used here under the MIT License. The DGP and quickstart
example are taken from `pylpdid/examples/01_quickstart.py`.
