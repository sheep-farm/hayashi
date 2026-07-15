# Changelog

All notable changes to Hayashi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — dev

### Added

- **`acf()` / `pacf()`**: return autocorrelation / partial autocorrelation values as a list. Accept a DataFrame + column name, or a model (OLS/GARCH/ARIMA — uses residuals). Complements the existing `acfplot`/`pacfplot` ASCII visualizations.
- **`cusumtest(model)`**: CUSUM test for structural stability (Brown, Durbin, Evans 1975). Uses recursive residuals and checks if the cumulative sum stays within 5% significance bounds. Supports OLS models.
- **`akaike_weights(m1, m2, ...)`**: returns Akaike weights as a dict `{model_name: weight}` for programmatic model comparison. Supports OLS, logit/probit, Poisson, NegBin, Tobit, Ordered, Mixed, and Zero-Inflated models. Prints a summary table with AIC, ΔAIC, and weights.
- **`lrtest(m_restricted, m_unrestricted)`**: likelihood-ratio test for nested models. Computes LR = -2(ln L_r - ln L_u) ~ chi²(df) and prints statistic, p-value, df, and verdict. Supports any estimator that reports a log-likelihood: OLS, logit/probit, Poisson, NegBin, Tobit, Ordered, Mixed, Zero-Inflated, GLM, GARCH, ARIMA. Errors if models are not nested in the expected direction (unrestricted must have more parameters than restricted).
- **`estat_overid(endog_formula, instrument_formula, df)`**: Sargan / Hansen J overidentification test for IV/2SLS. Tests H0: instruments are exogenous. Computes Sargan = n × R² from regression of IV residuals on instruments, distributed as chi²(L - K). Only applicable when overidentified (L > K). Aliases: `sargan`, `overid`.
- **`estat_endog(endog_formula, instrument_formula, df)`**: Durbin-Wu-Hausman endogeneity test. Tests H0: endogenous regressors are exogenous (OLS is consistent). Uses the augmented regression approach (adds first-stage residuals to structural equation, F-test on their significance). If rejected, IV is needed; if not, OLS is preferred. Aliases: `dwh`, `endog_test`.
- **`estat_classification(model, threshold=0.5)`**: classification table for logit/probit. Computes the 2×2 confusion matrix at the given threshold and reports sensitivity (recall), specificity, and overall correct rate. Alias: `classification`.
- **`lroc(model)`**: ROC curve and AUC for logit/probit. Computes the area under the ROC curve using the rank-based Wilcoxon-Mann-Whitney statistic. Also reports the Gini coefficient (2·AUC − 1). Aliases: `roc`, `estat_roc`.
- **`estat_gof(model, groups=10)`**: Hosmer-Lemeshow goodness-of-fit test for logit/probit. Divides observations into `groups` deciles based on predicted probabilities and compares observed vs expected counts via chi²(g − 2). H0: model fits adequately. Aliases: `hosmer_lemeshow`, `hltest`.
- **`linktest(model)`**: specification error detection for logit/probit (Stata's `linktest`). Re-estimates the model using the linear predictor ŷ = Xβ and ŷ² as the only regressors. If the coefficient on ŷ² is significant (p < 0.05), there is a specification error — wrong link function or functional form. H0: model is correctly specified.
- **`xtlogit(y ~ x1 + x2, df, id="group_col" [, corr=exchangeable])`**: panel logit via GEE (population-averaged). Supports correlation structures: independence, exchangeable (default), ar1, unstructured. Wrapper over `gee()` with `family=binomial, link=logit`.
- **`xtprobit(y ~ x1 + x2, df, id="group_col" [, corr=exchangeable])`**: panel probit via GEE. Wrapper over `gee()` with `family=binomial, link=probit`.
- **`xtpoisson(y ~ x1 + x2, df, id="group_col" [, corr=exchangeable])`**: panel Poisson via GEE. Wrapper over `gee()` with `family=poisson, link=log`.
- **`eventstudy(y ~ event_time + controls, df [, ref=-1, min=-5, max=5, cov=HC1])`**: event study (dynamic DiD with leads and lags). Constructs event-time dummies for each period relative to the event, excluding the reference period, and runs OLS. Reports coefficients, standard errors, t-stats, and p-values for each event time. Aliases: `event_study`, `es`.
- **`lroc` enhanced**: now includes ASCII ROC curve plot in addition to AUC and Gini. The curve shows TPR vs FPR at all thresholds, with the diagonal (random classifier) for reference.
- **`nls_exp` / `nls_power` / `nls_logistic` / `nls_cobb_douglas` / `nls_ces`**: nonlinear least squares estimation via Levenberg-Marquardt with numerical gradients. Supports common functional forms: exponential (y = a·exp(b·x)), power (y = a·x^b), logistic (y = a/(1+exp(-b·(x-c)))), Cobb-Douglas production (y = a·x1^b1·x2^b2), and CES production (y = a·(b1·x1^ρ + b2·x2^ρ)^(1/ρ)). Requires `start=[...]` option with starting values. Optional `max_iter` and `tol` control convergence.
- **`marginsplot(model)`**: ASCII plot of average marginal effects (AME) for logit/probit models. Horizontal bar chart with 95% confidence intervals and zero reference line. Alias: `margins_plot`.
- **`spatial_sar(y ~ x, df, w=W)`**: Spatial Autoregressive model (y = ρWy + Xβ + ε). Maximum likelihood estimation with grid search + golden section for ρ. W is a row-standardized spatial weights matrix provided as a list of lists. Reports ρ, β, standard errors, t-stats, p-values, R², and log-likelihood.
- **`spatial_sem(y ~ x, df, w=W)`**: Spatial Error Model (y = Xβ + u, u = λWu + ε). FGLS estimation with spatial error autocorrelation. Same W format as SAR.
- **`double_ml(y ~ d + x1 + x2, df [, folds=5, poly=2])`**: Double/debiased machine learning (Chernozhukov et al. 2018). Partially linear model Y = θ·D + g(X) + ε. K-fold cross-fitting with polynomial expansion nuisance models. First RHS variable is the treatment, rest are controls. Reports θ, SE, t, p, 95% CI. Alias: `dml`.
- **`sfa_production(y ~ x1 + x2, df)`**: Stochastic production frontier (y = α + β'x + v − u). Half-normal inefficiency (u ≥ 0). MLE via grid search + golden section over λ = σ_u/σ_v. Reports σ_v, σ_u, λ, γ, mean technical efficiency, and per-observation TE (Jondrow et al. 1982). Alias: `frontier`.
- **`sfa_cost(y ~ x1 + x2, df)`**: Stochastic cost frontier (y = α + β'x + v + u). Same estimation as production but with reversed skewness.
- **`panel_tobit(y ~ x1 + x2, df, id="firm" [, censor=0])`**: Panel Tobit with random effects. y_it = max(censor, x_it'β + α_i + ε_it). EM-style MLE with variance components (σ_α, σ_ε) estimated from panel structure. Reports β, SE, t, p, ρ (ICC), log-likelihood.
- **`panel_heckman(y ~ x1 + x2, df, sel="z ~ w1 + w2", id="firm")`**: Panel Heckman (selection with random effects). Two-step: probit selection + OLS outcome with inverse Mills ratio (IMR). Cluster-robust SE. Reports ρ (corr between selection and outcome errors), σ_α, σ_ν.
- **`spatial_panel_sar(y ~ x1 + x2, df, w=W, id="entity")`**: Spatial panel SAR with fixed effects. y_it = ρ·W·y_it + x_it'β + μ_i + ε_it. Within transformation (demean by entity) + grid search for ρ. W can be n×n or n_entities×n_entities (auto-expanded to block diagonal).
- **`spatial_panel_sem(y ~ x1 + x2, df, w=W, id="entity")`**: Spatial panel SEM with fixed effects. y_it = x_it'β + μ_i + u_it, u_it = λ·W·u_it + ε_it. Same W format as SAR panel.
- **`bayes_sfa_production(y ~ x1 + x2, df [, burn=1000, draws=2000])`**: Bayesian stochastic production frontier via MCMC (Gibbs sampler). y = α + β'x + v − u. Half-normal inefficiency. Priors: β ~ N(0,100), σ_v² ~ IG(2,2), σ_u² ~ IG(2,2). Reports posterior means, SDs, 95% credible intervals. Alias: `bayes_frontier`.
- **`bayes_sfa_cost(y ~ x1 + x2, df [, burn=1000, draws=2000])`**: Bayesian stochastic cost frontier via MCMC. y = α + β'x + v + u. Same priors and output as production.
- **`midas(y ~ x, df [, freq=3, lags=12, poly=2])`**: Mixed Data Sampling regression (Ghysels et al. 2004). Regresses a low-frequency variable (e.g., quarterly GDP) on a high-frequency variable (e.g., monthly industrial production) via Almon polynomial weighting. NLS estimation with grid search + Newton refinement on γ. freq = frequency ratio, lags = number of high-frequency lags, poly = Almon polynomial degree.
- **`tvp(y ~ x1 + x2, df)`**: Time-Varying Parameter regression via Kalman filter. y_t = x_t'β_t + ε_t, β_t = β_{t-1} + η_t. MLE for σ_ε and σ_η via grid search + golden section. Reports smoothed β_t estimates (first, middle, last period).
- **`setar(y ~ 1, df [, order=2, delay=1])`**: Self-Exciting Threshold Autoregressive model. Two regimes split by y_{t-delay} threshold (estimated by minimizing RSS via grid search). order = AR order per regime, delay = threshold lag. Reports separate coefficients, SE, t, p for each regime.
- **`panel_qreg(y ~ x1 + x2, df, id="firm" [, tau=0.5])`**: Panel quantile regression with fixed effects (Koenker 2004). Within transformation (demean by entity) + standard quantile regression on demeaned data. tau = quantile level. Alias: `panel_quantile`.
- **`msvar(y1 ~ y2, df [, regimes=2, lags=1])`**: Markov-Switching VAR with K regimes. y_t = μ_{s_t} + A·y_{t-1} + ε_t, ε_t ~ N(0, Σ_{s_t}). EM (Baum-Welch) estimation with forward-backward algorithm. Reports transition matrix, regime-specific intercepts/covariances, filtered and smoothed regime probabilities. Alias: `ms_var`.
- **`favar(y1 ~ y2 + y3, df, observed="rate" [, factors=2, lags=1, irf=0])`**: Factor-Augmented VAR (Bernanke et al. 2005). Two-step: PCA factors from large panel + VAR on [F, R] where R is observed policy variable. Reports factor loadings, variance explained, VAR coefficients, and optional IRF.
- **`spatial_durbin(y ~ x1 + x2, df, w=W, id="entity")`**: Spatial Durbin Model (panel, fixed effects). y = ρ·W·y + X·β + W·X·θ + FE + ε. Nests SAR (θ=0) and SLX (ρ=0). Within transformation + grid search for ρ. Reports direct effects (β) and indirect/spillover effects (θ). Alias: `sdm`.
- **`johansen_break(y1 ~ y2, df [, lags=1, breaks=[50]])`**: Johansen cointegration test with structural breaks. Shift dummies at specified break points. Trace and Lambda-max statistics with break-adjusted critical values. Reports eigenvalues, cointegrating vectors, and selected cointegration rank.
- **`tvp_var(y1 ~ y2, df [, lags=1])`**: Time-Varying Parameter VAR via Kalman filter. All VAR coefficients evolve as random walks: y_t = Z_t'·B_t + ε_t, B_t = B_{t-1} + η_t. MLE for Σ and Q via grid search + golden section. Reports smoothed time-varying coefficients B_t.
- **`spatial_durbin_error(y ~ x1 + x2, df, w=W, id="entity")`**: Spatial Durbin Error Model (panel, fixed effects). y = X·β + W·X·θ + FE + u, u = λ·W·u + ε. Combines SLX (spatially lagged X) with SEM (spatial error autocorrelation). Reports direct (β) and indirect (θ) effects, λ. Alias: `sdem`.
- **`fmols(y ~ x1 + x2, df)`**: Fully Modified OLS (Phillips & Hansen 1990). Nonparametric correction for endogeneity and serial correlation in cointegrating regressions. Bartlett kernel long-run covariance with Newey-West automatic bandwidth. Reports α, β, SE, t, p, Ω.
- **`qvar(y1 ~ y2, df [, lags=1, tau=0.5, boot=100])`**: Quantile VAR — equation-by-equation quantile regression. Captures quantile-specific dynamics (e.g., recession vs expansion). tau = quantile level, boot = bootstrap replications for SE. Reports per-equation coefficients, SE, t, p, pseudo R². Alias: `quantile_var`.
- **`pstr(y ~ x1 + x2, df, q="transition_var", id="entity")`**: Panel Smooth Transition Regression (fixed effects). y = μ_i + β₀'x + β₁'x·g(q;γ,c) + ε where g(q) = 1/(1+exp(-γ(q-c))) is the logistic transition function. Grid search over (γ, c). Reports β₀ (linear regime), β₁ (nonlinear regime), γ, c.
- **`modwt(df, var [, scales=4])`**: Maximal Overlap Discrete Wavelet Transform (Haar wavelet). Decomposes a time series into multi-scale components (time-frequency analysis). Reports wavelet/scaling coefficients, energy per scale, detail and smooth components. Unlike DWT, MODWT does not decimate.
- **`copula(y1 ~ y2, df [, type="gaussian"])`**: Copula-based dependence modeling via Sklar's theorem. Supports Gaussian, Clayton, Gumbel, and Frank copulas. Two-step estimation: empirical CDF margins + copula parameter MLE. Reports Kendall's tau, Spearman's rho, copula parameter, AIC/BIC.
- **`nardl(y ~ x, df [, lags=1])`**: Nonlinear ARDL (Shin, Yu & Greenwood-Nimmo 2014). Asymmetric cointegration: decomposes x into positive/negative partial sums. Conditional ECM with long-run multipliers (β⁺, β⁻) and short-run asymmetric coefficients. Wald tests for long-run and short-run asymmetry.
- **`pvar(y1 ~ y2, df, id="entity" [, lags=1])`**: Panel VAR via GMM (Holtz-Eakin, Newey & Rosen 1988). Arellano-Bond style: lagged levels as instruments for first-differenced equations. Per-equation coefficients, J-test for overidentifying restrictions. Alias: `panel_var`.
- **`fcoef(y ~ x1 + x2, df, z="moderator" [, points=20])`**: Functional coefficient model (Hastie & Tibshirani 1993). Local linear kernel regression: coefficients vary smoothly with moderator z. Gaussian kernel, Silverman bandwidth. Reports coefficients at evaluation points. Alias: `functional_coef`.
- **`dcc_garch(y1 ~ y2, df)`**: DCC-GARCH (Engle 2002). Dynamic Conditional Correlation. Two-step: univariate GARCH(1,1) per series + DCC for time-varying correlation matrix. Reports DCC α/β, GARCH params, conditional volatilities, dynamic correlation matrices. Alias: `dcc`.
- **`tvar(y1 ~ y2, df, q="threshold_var" [, lags=1, delay=1])`**: Threshold VAR (Tong 1978, Tsay 1998). Regime switching via threshold variable: different VAR coefficients in low/high regimes. Grid search over threshold c (percentiles of q) and delay d. Reports per-regime coefficients, residual covariances, RSS, AIC/BIC. Alias: `threshold_var`.
- **`bvar(y1 ~ y2, df [, lags=1, lambda1=0.1, lambda2=0.2, lambda3=1.0])`**: Bayesian VAR with Minnesota prior (Litterman 1986). Random walk prior: shrinks own lags toward 1, cross lags toward 0. Conjugate Normal prior + Normal likelihood → closed-form posterior. λ₁=own lag tightness, λ₂=cross lag, λ₃=higher lag decay. Alias: `bayesian_var`.
- **`mfvar(df_low, y_low1, ..., df_high, y_high1, ... [, agg=3, lags=1])`**: Mixed-Frequency VAR (Foroni, Ghysels & Marcellino 2013). Combines low-freq + high-freq variables via MIDAS exponential Almon aggregation. Grid search over Almon θ to minimize variance. agg=high-freq periods per low-freq period. Alias: `mixed_freq_var`.
- **`tvcopula(y1 ~ y2, df [, type="gaussian"])`**: Time-varying copula (Patton 2006). GARCH-like dynamics for copula parameter: θₜ = g(a + b·θₜ₋₁ + c·ψₜ₋₁). type: gaussian, clayton, gumbel. Grid search over (a, b, c). Reports θ path, Kendall's τ path, dynamics parameters. Alias: `tv_copula`.

- **`tidy()` extended to all model types**: now supports IV, logit/probit, panel FE/RE, GMM, Poisson, NegBin, GLM, Quantile, Tobit, Heckman, Ordered, Arellano-Bond, Penalized (ridge/lasso/elasticnet), RLM, Beta, GEE, ARIMA, and GARCH — in addition to the existing OLS and Rolling support. Returns a DataFrame with `variable`, `coef`, `std_err`, `t` (or `z`), `p_value`, `conf_low`, `conf_high`.
- **`glance()` extended to all model types**: returns model fit statistics as a DataFrame. Available keys vary by model type: `r2`, `adj_r2`, `pseudo_r2`, `n`, `f_stat`, `prob_f`, `aic`, `bic`, `log_lik`, `sigma`, `j_stat`, `j_p_value`, `df_overid`, `sigma_u`, `sigma_e`, `theta`, `tau`, `alpha`, `rho`, `delta`, `deviance`, `qic`, `n_entities`, `n_groups`, `n_censored`, `sigma2`.
- **`names(df)` builtin**: returns DataFrame column names as a list of strings.
- **Model serialization for native plugins**: `value_to_json` now serializes model results (`OlsResult`, `IvResult`, `BinaryResult`, `PanelResult`, `ReResult`, `GmmResult`, `PoissonResult`, `NegBinResult`, `GlmResult`, `QuantileResult`, `TobitResult`, `HeckmanResult`, `OrderedResult`, `AbResult`, `PenalizedResult`, `RlmResult`, `BetaResult`, `GeeResult`, `ArimaResult`, `GarchResult`) as JSON dicts with `__model_type__`, `variable`, `coef`, `std_err`, `p_value`, and fit statistics — instead of `null`. Enables native plugins (e.g. haytex) to consume model data directly.
- **Safe modes for `hay dist-update`**:
  - `hay dist-update --help` prints subcommand-specific help without network access.
  - `hay dist-update --check` reports whether a newer release is available without downloading or replacing the binary.
  - `hay dist-update --nightly` downloads and installs the latest nightly build from the `dev` branch (pre-release, may be unstable). Nightly builds are generated daily via GitHub Actions for Linux, macOS, and Windows.
  - Unknown flags and unexpected positional arguments fail fast.
  - Argument parser covered by focused unit tests.
- **Plugin compatibility check**: plugins can declare a minimum Hayashi version in a `hayashi.toml` file at the repo root (`min_version = "0.2.9"`). During `hay install`, the file is fetched and the version is compared. If the current Hayashi version is lower, installation is refused with a clear message. Pre-release suffixes (`-dev`, `-rc`) are ignored in the comparison, so `0.2.9-dev` satisfies `min_version = "0.2.9"`.
- **English-only user-facing output**: all comments, error messages, and printed strings in the Rust source tree translated to English. Mathematical notation (`×`, `ŷ`, `Ŷ`, `H₀`, `κ`, etc.) is preserved.
- **Interpreter decomposition**: `src/lang/interpreter.rs` split into focused submodules:
  - `execution.rs` — statement execution
  - `eval_expr.rs` — expression evaluation
  - `dispatch.rs` — function-call dispatcher
  - `helpers.rs` — shared static utilities
  - `value.rs` — `Value` type
  - `models.rs` — model wrappers
  - `panel_diagnostics.rs`, `rolling_recursive.rs`, `aggregation.rs`, `timeseries_models.rs` — grouped estimator logic
  - `interpreter.rs` reduced from ~4,800 lines to ~680 lines.
- **`for` loop index/value binding**: `for i, v in list { ... }` binds the element index to `i` and the value to `v`. `for k, v in dict { ... }` binds each key/value pair. Dict iteration requires two variables.
- **`parallel for` construct**: concurrent variant of `for` that runs iterations across threads via `std::thread::scope`. Each iteration's return value (last expression or explicit `return`) is collected into a list. Can be used as an expression (`let results = parallel for t in tickers, threads=8 { ... }`) or as a statement (`parallel for t in tickers { ... }` — result stored in the iteration variable). Optional `threads=N` parameter limits the number of worker threads (default: `available_parallelism()`). Each thread gets its own interpreter with a snapshot of the outer environment (send-safe values only).
- **`rbind()` builtin**: concatenates a list of DataFrames vertically in a single pass in Rust. `nil` entries are silently skipped — useful when combining results from `parallel for` where some iterations return `nil`. Example: `let all = rbind(results)`.
- **`dataframe()` accepts `Series`**: the `dataframe()` constructor now accepts `Value::Series` directly as column values (in addition to `Value::List`), extracting the underlying values automatically.
- **`try/catch/finally`**: `finally { ... }` block now runs regardless of whether the try succeeded, failed, or executed `return`/`break`/`continue`. Errors or control flow inside `finally` take precedence.
- **`hay install` with version**: `hay install user/repo [version]` installs a specific release. `version` may be `latest`, `v1.2.3`, or `1.2.3`.
- **`hay list` shows versions**: installed plugins now display their version from `.metadata.json`.
- **`match` as contextual keyword**: `match` now works as a regular identifier (`let match = 1`) and still starts a match expression when followed by a scrutinee and brace (`let r = match x { ... }`).
- **README smoke test**: `scripts/readme_smoke.hay` exercises the main features documented in `README.md` and is run by the test suite.
- **`list_files()` builtin**: `list_files(dir)` and `list_files(dir, pattern)` return a sorted list of file paths, enabling dynamic batch processing of datasets.
- **`columns=` and `where=` options for `load`**: push column projection and row filtering down to the data source, avoiding loading the full dataset into RAM. Supported by Parquet (Arrow `ProjectionMask` + `RowFilter` — filter evaluated during row-group scan), SQLite and ODBC (`SELECT cols FROM t WHERE pred`, validated against `PRAGMA table_info`), CSV/TSV (projection in read loop, row-by-row predicate), DTA (projection in `read_record`, row-by-row predicate), and Excel (projection after `worksheet_range`, row filter on cells). JSON is not yet supported. `where=` accepts a Hayashi expression of the form `column OP literal` combined with `&&`, `||`, `!`, and `in [...]` — parsed by the Hayashi parser and normalized into a `RowPredicate` enum (`src/lang/predicate.rs`). `query=` cannot be combined with `columns=` or `where=`. On a 837 MB / 30 M-row Parquet file, `columns=[ticker, date, adj_close], where="ticker == \"AAPL\""` reduced peak RAM from ~7.5 GB (eager full load) to ~4 MB.
- **Row group pruning by statistics in Parquet**: before applying `RowFilter`, the loader reads per-row-group min/max statistics from the Parquet metadata and skips row groups where the `where=` predicate cannot possibly match. This is done via `RowPredicate::can_match(&dyn GroupBounds)`, which evaluates the predicate conservatively against `(min, max)` bounds per column. On a 799 MB / 30 M-row / 8 292-row-group Parquet file sorted by ticker, `where="ticker == \"AAPL\""` pruned 8 291 of 8 292 row groups, reducing load time from ~62 s (full scan) to ~312 ms (212× faster) with ~60 MB peak RSS. SQLite with a `(ticker, date)` B-tree index remains faster for point lookups (~42 ms, ~26 MB RSS) due to direct seek without metadata overhead, but Parquet with pruning is superior for full-column analytics.

### Changed

- **`quietly(expr)` deprecated**: the functional form is marked deprecated and will be removed in a future release. Use `quietly on` / `quietly off` instead. README and `help(quietly)` updated to reflect this.
- **Validation runner exit semantics**: `validation/run.py` now returns a non-zero exit code when cases are `blocked`, unless `--allow-blocked` is passed.
- **`data_source` field added to validation cases**: book-based simulated cases (e.g. `var_book`) are explicitly tagged as `dgp`.

### Fixed

- **Parquet timestamp/date columns rendered as `PrimitiveArray<Timestamp(µs)>`**: Arrow temporal types (`Timestamp(s|ms|µs|ns, tz)`, `Date32`, `Date64`) loaded from Parquet were falling through the catch-all branch of `append_as_string`, which formatted the whole Arrow array via `{:?}` instead of each row's value. The loader now converts each value via `arrow::temporal_conversions` to a `NaiveDateTime` and formats it as ISO 8601 (`YYYY-MM-DD` when the time component is midnight, otherwise `YYYY-MM-DDTHH:MM:SS`). Date/time columns are stored as Hayashi strings; to extract components use `generate df ano = substr(date, 0, 4)` (vectorial, works in `generate`), or use the scalar builtin `date("YYYY-MM-DD")` to convert a single ISO date string to a Unix timestamp.
- **CSV export column order**: `DataFrame::to_csv` no longer sorts columns alphabetically. Column insertion order is preserved via `IndexMap` (replacing `HashMap`). Affects all CSV/JSON exports and display functions.
- **`append()` losing string columns**: `get_string()` now handles `Categorical` columns (previously returned error, causing `append()` to produce empty strings for any column with repeated values — e.g. dates, tickers, sectors).
- **Plugin resolution on Windows**: `HOME` now falls back to `USERPROFILE`. `resolve_import` also searches the executable's directory (`exe_dir/`, `exe_dir/plugins/`, `exe_dir/.hay/plugins/`). `HAYASHI_PATH` uses `;` as separator on Windows.
- **Validation workflow**: repaired malformed `.github/workflows/validation.yml`, added `../Greeners` checkout, and switched R dependency installation to use `validation/DESCRIPTION`.
- **Clippy warnings**: fixed `empty_line_after_doc_comments`, `too_many_arguments`, and `needless_range_loop` warnings.
- **`tobit_mroz` tracking**: marked as needing isolated intercept-difference investigation and linked to issue #42.

### Removed

### Internal / CI

- `cargo fmt` run across the Rust source tree.
- Validation workflow temporarily set to `workflow_dispatch` only until the baseline is clean.

## [0.2.6] — 2026-08-25

### Added

- **Hybrid plugin system** (`import`): Hayashi now supports three plugin tiers in a single unified `HayashiPlugin` trait:
  - **Native Rust** (`.so`/`.dll` via `libloading`): plugins export `extern "C"` functions, args/return values are exchanged as JSON strings
  - **WebAssembly** (`.wasm` via `wasmi`): sandboxed plugins expose `alloc`/`dealloc`/function exports; args serialized to JSON written into guest memory, result packed as `i64` (`high 32 bits = ptr`, `low 32 bits = len`)
  - **Script** (`.hay`): existing interpreted plugin tier, unchanged
  - Bidirectional `value_to_json` / `json_to_value` helpers for all Hayashi value types (Float, Int, Bool, Str, List, Dict, DataFrame)
  - Book chapters updated in EN and PT-BR to document the new import model
- **Pipe placeholder `_`**: `df |> ols(lw ~ yos, _)` passes the piped value into an arbitrary argument position, not just the first. Works in any expression context.
- **`ttest` option `unequal=false`**: explicitly request a pooled (equal-variance) t-test. Default remains Welch (unequal variances). Documented in book chapters and command reference.
- Expanded empirical validation programme to 40 cases: activated 21 existing cases and added 6 new cases for ridge, elasticnet, nbreg, oprobit, mlogit, and SUR.
- Added parametric-bootstrap standard errors for VECM and enabled inference in the Hayashi VECM handler.
- Added validation matrix section to Appendix C of both English and Portuguese books.
- Added empirical validation subsections to Chapters 33, 35, 38, and 39 in both languages.

### Fixed

- **Non-numeric columns in `generate` and estimators**: `get_col_f64` and `eval_col_expr` now call `col.to_float()` instead of hard-failing on Boolean and Categorical columns. Previously, running `ols` or `generate` on a DataFrame loaded from JSON/CSV with boolean columns would panic or raise a type error.
- **Hardened URL downloads** (security, contributed by Charles Shaw):
  - Reject `localhost`, `ip6-localhost`, `ip6-loopback`, and `.localhost` hostnames before connecting
  - Reject all private, loopback, link-local, and unspecified IPv4/IPv6 addresses (SSRF prevention)
  - Custom resolver validates resolved IPs against the same allowlist (DNS rebinding prevention)
  - `redirects(0)` prevents redirect-based bypass
  - 30-second connection timeout and 100 MB download size limit enforced
- **EGARCH/GJRGARCH function signatures** corrected in the quick reference appendix (EN and PT-BR books)
- Fixed all Clippy warnings in Hayashi and Greeners; `cargo clippy -- -D warnings` now passes in both repos.
- Updated `argmin` and `argmin-math` in Greeners to resolve `RUSTSEC-2024-0384` (unmaintained `instant` dependency).
- Installed missing R packages (`glmnet`, `systemfit`, `jsonlite`, `MatchIt`, `rdrobust`, `sampleSelection`) so the validation runner now exercises both R and Python references where available.

### Changed

- **`ttest` calculations delegated to Greeners**: the interpreter no longer implements the t-statistic, degrees of freedom, and p-value arithmetic inline. All computation now goes through `greeners::ttest`, keeping the interpreter as a thin dispatcher.

### Removed

- **Unused `anyhow` dependency** removed from `Cargo.toml` (contributed by Charles Shaw)

### Internal / CI

- Format check (`cargo fmt --check`) and Windows smoke test restored to CI pipeline (contributed by Charles Shaw)
- Dead code warnings silenced in `ttest` dispatch path
- Empirical validation runner now uses both R and Python references; documentation updated to reflect the change.
- All 40 validation cases pass; `hay validate` reports overall status `pass`.

## [0.2.3] — 2026-06-25

### Added

- **`codebook(df)`**: detailed variable description — type, unique values, missing, range, percentiles
- **`swilk(df, var)`**: Shapiro-Wilk normality test (Royston 1995, 3 ≤ n ≤ 5000)
- **`sfrancia(df, var)`**: Shapiro-Francia normality test (Royston 1993, 5 ≤ n ≤ 5000)
- **`sktest(df, var)`**: Skewness/Kurtosis tests (Jarque-Bera + D'Agostino)
- **`mutate(df, col1=expr1, col2=expr2)`**: generate multiple columns at once, pipe-friendly
- **`group_by(df, by, stat, vars...)`**: pipe-friendly aggregation by group
- **`pivot_longer(df, stubs=[], i=, j=)`**: expressive wide-to-long reshape
- **`pivot_wider(df, i=, j=, values=)`**: expressive long-to-wide reshape
- **`select()`**: alias for `keep`, natural in pipe chains
- **`generate()` as function call**: `generate(df, col=expr)` works in pipes, modifies in-place
- **`print()` multi-arg**: `print(a, b, c, sep=", ", end="")` with separator and line ending
- **`summarize()` returns dict**: silent when captured (`let s = summarize(df, x)`), prints when standalone
- **`resolve_var_list()`**: 16 commands now accept bare, string, variable, and list-of-strings for column names
- **Pipe assigns back**: standalone `df |> f()` modifies `df`; `let r = df |> f()` preserves `df`
- **Error messages**: source line preview with `^`, Levenshtein "did you mean?", stack traces for nested functions, `expected X, got Y` type mismatch
- **`Expr::Pipe`**: AST node preserves pipe source for assign-back semantics

### Fixed

- Keywords followed by `(` now parse as function calls (e.g. `generate(df, col=expr)`)
- Parser line numbers: correctly track lines after newlines
- `MomentHelpers` now exported from Greeners (was missing from lib.rs)

## [0.2.2] — 2026-06-24

### Added

- **`%` (modulo operator)**: `10 % 3` → `1`, works on int and float, including inside `generate`
- **`**` (power alias)**: alternative to `^` for users coming from Python/JS (`2 ** 10` → `1024`)
- **Compound assignment**: `+=`, `-=`, `*=`, `/=`, `%=` desugar to `x = x op expr`
- **`gmm()`**: generic two-step efficient GMM with Hansen J-test for overidentification
- **`cmnlogit()`**: conditional multinomial logit (McFadden's choice model)
- **`help(about)`**: project info derived from Cargo.toml (version, license, author, repo)
- **`help(license)`**: GPL-3.0 notice derived from Cargo.toml metadata
- **README badges**: license, Rust edition, version, crates.io, CI status

### Fixed

- **Formula parsing unified**: all 53 estimators now accept both bare formulas (`y ~ x1 + x2`) and string formulas (`"y ~ x1 + x2"`), enabling dynamic formula composition

## [0.2.1] — 2026-06-24

### Fixed

- **Scalar math functions**: `sqrt()`, `abs()`, `ln()`, `exp()`, `pow()`, `log2()`, `log10()`, `sin()`, `cos()`, `tan()`, `ceil()`, `floor()`, `round()`, `sign()`, `factorial()`, `normalden()`, `invnormal()`, `mod()`, `atan2()`, `max()`, `min()`, `comb()` now work as standalone expressions (previously only worked inside `generate`)

## [0.2.0] — 2026-06-24

### Breaking Changes

- **Extension**: `.hy` → `.hay` (avoids conflict with [hylang.org](https://hylang.org))
- **Binary**: `hayashi` → `hay`
- **Directory**: `exemplos/` → `examples/`, filenames in English
- **`push`/`pop`**: now mutate in-place (like Python/JS/Rust), previously returned new list
- **`import`**: now namespaced by default (`import("mod")` → `mod::func()`)
- **History file**: `.hayashi_history` → `.hay_history`

### Added

- **CI/CD** (GitHub Actions): `cargo fmt` + `cargo clippy` + `cargo test` on Linux/macOS/Windows; release workflow builds stripped binaries for 4 targets on tag push
- **Namespaces**: `import("mod")` → `mod::func()`, `import("mod", as=alias)` → `alias::func()`, `import("mod", only=["f"])` → `f()` direct access. Parser supports `::` (ColonColon token)
- **REPL improvements**: tab completion for keywords + env variables, syntax highlighting (keywords blue, strings green, numbers yellow, comments gray), colored `hay>` prompt, fish-style history hints
- **Date/time**: `date("YYYY-MM-DD")` and `datetime("YYYY-MM-DD HH:MM:SS")` return Unix timestamps; `generate df Y = year(col)`, `month()`, `day()`, `hour()`, `minute()`, `second()`, `dow()` extract components from DateTime columns or float timestamps
- **Collinearity detection**: `linalg::drop_collinear` (QR-based) across 9 estimators (OLS, IV, Logit, Probit, GLM, Poisson, NegBin, MNLogit, GMM). Stata-style display: `(omitted)` inline in coefficient table + `note:` in footer
- **Pipe with closures**: `value |> |x| x * 3` works in expressions and inside `generate`
- **`eval_col_expr` accesses env**: `generate` and `filter` resolve user-defined functions and scalar variables (e.g., `filter(df, ts >= cutoff)` where `cutoff` is a `let`)
- **Opts accept variables**: `cov=my_var` resolves variable; falls back to string if undefined (so `cov=robust` still works)
- **`esttab` accepts lists**: `esttab(models)` where `models` is built with `push` in a loop
- **`Expr::Apply`** AST node for closure application via pipe
- **VS Code extension 0.2.0**: `.hay` extension, f-string highlighting, 46 estimators, datetime/namespace/pipe/`::` operator support
- **Author metadata**: Cargo.toml and README

### Fixed

- **Stack overflow on Windows/macOS**: main thread spawns with 32 MB stack (debug builds use ~10x more stack per frame)
- **Cross-platform tests**: `std::env::temp_dir()` replaces hardcoded `/tmp/` paths (Windows `os error 3`)
- **Package manager**: `hay install user/repo` uses correct `user/repo/` directory structure

### Changed

- **Greeners** (1.4.5-dev): `Arc<Column>` COW for zero-copy `select`/`drop`/`keep`/`rename`; `omitted_vars` now `Vec<(usize, String)>` for positional display; `x_clean` field on `OlsResult`
- **Parser**: opt values parsed as full expressions (not forced to `Expr::Str`); bare identifiers fall back to string on undefined variable
- **`maybe_filter_df`**: now `&mut self` (eval_col_expr needs env access)
- Test count: 387 → 389
- Example count: 59 → 60

## [0.1.0] — 2026-06-15

Initial release.

- 46 estimators: OLS, IV, Logit, Probit, Poisson, NegBin, Tobit, Heckman, FE, RE, Arellano-Bond, System GMM, ARIMA, GARCH, VAR, VECM, Lasso, Ridge, DID, RD, Cox, Fama-MacBeth, and more
- Stata-like syntax with modern language features (closures, pipe, match, f-strings, try/catch)
- 8 I/O formats: CSV, TSV, JSON, DTA, Excel, Parquet, SQLite, ODBC
- Plugin system: auto-load, package manager (`hay install user/repo`), URL import
- Post-estimation: test, nlcom, margins, bootstrap, esttab, vif, hausman, predict
- VS Code extension with syntax highlighting
- 387 tests, 59 examples, ~110 help topics
- 100% Rust — no C, no Fortran, no system BLAS/LAPACK
