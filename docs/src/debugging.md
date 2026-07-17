# Debugging Hayashi Scripts

Hayashi ships with a built-in Debug Adapter Protocol (DAP) server, so you can debug `.hay` scripts directly from VS Code.

## Starting a debug session

1. Install the [Hayashi VS Code extension](https://github.com/sheep-farm/hayashi/tree/dev/editors/vscode).
2. Make sure the `hay` binary is in your `PATH`.
3. Open a `.hay` file, go to the **Run and Debug** panel, and select **"Debug Hayashi script"**.
4. Set breakpoints and start debugging (`F5`).

Behind the scenes, VS Code launches:

```bash
hay dap path/to/script.hay
```

The DAP server supports breakpoints, single stepping (`step over`, `step in`, `step out`, `continue`), scopes (`Locals` / `Globals`), and variable inspection.

## Inspecting model results

When you pause on a breakpoint after fitting a model, model objects expand in the **Variables** panel with a concise summary plus structured children. For example, an OLS result appears as:

```
result: OLS(k=2, n=10000), R2=1.0000
  coefficients    DataFrame(2 rows, 7 cols)
  fit             Dict(13 entries)
  residuals       Series(residuals: 10000 values)
  fitted_values   Series(fitted_values: 10000 values)
  params          Series(params: 2 values)
  std_errors      Series(std_errors: 2 values)
  test_values     Series(test_values: 2 values)
  p_values        Series(p_values: 2 values)
  conf_lower      Series(conf_lower: 2 values)
  conf_upper      Series(conf_upper: 2 values)
```

### `coefficients`

A `DataFrame` containing the coefficient table:

| variable | coef | std_err | t / z | p_value | conf_low | conf_high |
|---|---|---|---|---|---|---|
| x1 | ... | ... | ... | ... | ... | ... |
| x2 | ... | ... | ... | ... | ... | ... |

The test statistic column is labelled `t` for t-based inference and `z` for z-based inference; both map to the same `test_values` Series in the debugger.

### `fit`

A `Dict` with model-specific fit statistics:

- **OLS / IV / panel OLS-style**: `r2`, `adj_r2` (where available), `f_stat`, `prob_f`, `aic`, `bic`, `log_lik`, `sigma`, `n_obs`, `df_model`, `df_resid`, `cov_type`, `inference`.
- **Binary choice (`logit`, `probit`)**: `model_name`, `pseudo_r2`, `log_lik`, `iterations`, `inference`.
- **Count data (`poisson`, `nbreg`, `zip`, `zinb`)**: `log_lik`, `deviance`, `null_deviance`, `aic`, `bic`, `pseudo_r2`, `pearson_chi2`, `n_obs`, `df_resid`, `df_model`, `iterations`, `converged`, plus model-specific entries such as `alpha` (NegBin / ZINB) and `dispersion` (GLM).
- **Quantile (`qreg`)**: `tau`, `r2`, `iterations`.
- **Tobit**: `sigma`, `log_lik`, `n_obs`, `n_censored`, `df_resid`, `iterations`.
- **Random effects (`re`)**: `r2_overall`, `sigma_u`, `sigma_e`, `theta`, `inference`.
- **GMM**: `j_stat`, `j_p_value`, `n_obs`, `df_model`, `df_overid`.
- **Arellano-Bond / System GMM**: `sargan_stat`, `sargan_pvalue`, `sargan_df`, `n_obs`, `n_entities`, `n_instruments`, `max_lags`, `step`, `m1_stat`, `m1_pval`, `m2_stat`, `m2_pval`.
- **PCSE / PanelGLS**: `r2`, `n_obs`, `n_entities`, `t_periods`, `df_resid`, `sigma` (and `panels` for PanelGLS).
- **GLSAR**: `r2`, `n_obs`, `df_resid`.
- **Mixed**: `log_lik`, `aic`, `bic`, `n_obs`, `n_groups`, `var_resid`, `iterations`, `converged`.
- **Beta**: `precision_param`, `log_lik`, `aic`, `bic`, `pseudo_r2`, `n_obs`, `iterations`, `converged`.
- **Zero-inflated (`zip` / `zinb`)**: separate `count_coefficients` and `inflate_coefficients` DataFrames plus a `fit` Dict.

### `params`, `std_errors`, `test_values`, `p_values`

These `Series` expose the raw coefficient vectors. They are useful when you want to inspect one vector at a time or copy a value from the debugger.

### `conf_lower` / `conf_upper`

Confidence interval bounds for each parameter. Only shown when the underlying model stores them.

## Model-specific views

### Panel models (`fe`, `re`, `pcse`, `xtgls`, `feiv`)

```
result: Panel(k=2, n=1000, N=50), R2=0.8123
  coefficients  DataFrame(2 rows, 7 cols)
  fit           Dict(...)
  params        Series(params: 2 values)
  ...
```

`Panel` and `FE2SLS` report `n` and `N` (entities). `RandomEffects` reports `sigma_u`, `sigma_e`, and `theta`.

### Binary choice

```
result: Logit(k=2), pseudoR2=0.3124
  coefficients  DataFrame(2 rows, 5 cols)
  fit           Dict(5 entries)
  ...
```

### Count data

```
result: Poisson(k=2, n=100), pseudoR2=0.8890
  coefficients  DataFrame(2 rows, 7 cols)
  fit           Dict(13 entries)
  ...
```

### Zero-inflated models (`zip`, `zinb`)

```
result: ZeroInflated(count=2, inflate=2, n=100), logLik=-134.2
  count_coefficients   DataFrame(2 rows, 5 cols)
  inflate_coefficients DataFrame(2 rows, 5 cols)
  fit                  Dict(...)
```

### Mixed models (`mixed`)

```
result: Mixed(fixed=2, n=100, groups=10), logLik=-123.4
  fixed_effects    DataFrame(2 rows, 5 cols)
  random_effects   DataFrame(...)
  fit              Dict(...)
```

The `fixed_effects` DataFrame contains the population-level coefficients; `random_effects` contains one column per group.

### SUR and 3SLS

For system estimators, each equation is exposed as a child variable:

```
result: SUR(eqs=2), sysR2=0.8543
  system_r2  Float
  sigma_cross  DataFrame(...)
  equation_0 / gnp   Dict { coefficients, fit }
  equation_1 / inv   Dict { coefficients, fit }
```

`3SLS` follows the same pattern but omits the system-level `system_r2` and `sigma_cross` nodes.

## Limitations

- Not all estimators are debuggable yet. The expansion covers the most common cross-sectional, panel, count, binary, and system models.
- Time-series models (ARIMA, GARCH, VAR, etc.) currently display as plain text without structured children.
- Very large `Series`/`DataFrame` children are truncated by VS Code (`namedVariables: 100`), but you can request more values through the Variables panel.

## Command-line debugging

You can also start the DAP server manually for testing or integration with other editors:

```bash
hay dap script.hay
```

It reads DAP messages from `stdin` and writes responses to `stdout`. The VS Code extension handles the protocol automatically.
