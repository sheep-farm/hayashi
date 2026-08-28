# Experimental estimators

A subset of estimators in Hayashi is gated behind the `experimental` Cargo
feature. They are not compiled into the default binary because they are still
being validated or their API is not yet stable.

## Enabling experimental commands

Build Hayashi with the `experimental` feature:

```bash
cargo build --release --features experimental
```

The `hayashi` validation CI uses this flag (`--features experimental` in
`.github/workflows/validation.yml`) so that experimental validation cases can be
run.

## Gated commands

The commands below require `experimental` to be enabled:

| Command | Description |
|---------|-------------|
| `bsc` | Bayesian structural-causal approximation |
| `bvar` | Bayesian VAR with Minnesota prior |
| `mfvar` | Mixed-frequency VAR |
| `bayes_sfa_production` / `bayes_sfa_cost` | Bayesian stochastic frontier analysis |
| `fcoef` / `functional_coef` | Functional-coefficient model |
| `fapanel` / `fa_panel` | Factor-augmented panel |
| `fmols` | Fully Modified OLS (cointegration) |
| `johansen_break` | Johansen cointegration with structural breaks |
| `lstm` | Long short-term memory network for time series |
| `transformer` / `transformer_ts` | Transformer for time series |
| `spectral` | Spectral time-series analysis |
| `tvcopula` / `tv_copula` | Time-varying copula |
| `tvp_var` | Time-varying parameter VAR |
| `spatial_sar` | Spatial SAR (cross-section) |
| `spatial_sem` | Spatial SEM (cross-section) |
| `spatial_durbin` / `sdm` | Spatial Durbin model |
| `spatial_panel_sar` | Spatial panel SAR |
| `spatial_panel_sem` | Spatial panel SEM |
| `spatial_durbin_error` / `sdem` | Spatial Durbin error model |

## Validation

Validation cases for experimental estimators are only run when the runner is
also told to include them:

```bash
python validation/run.py --experimental
```

Without `--experimental`, the runner skips cases whose `case.yml` has
`experimental: true`.

## Status

- `fmols` and `spatial_panel_sar` are validated (`pass`) but remain
  experimental.
- The remaining experimental estimators are listed in the validation matrix as
  `not-supported` while their reference implementations are being assembled.
