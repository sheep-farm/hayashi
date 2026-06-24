# Command Reference

Quick reference for all Hayashi commands and functions. For detailed usage, type `help(command)` in the REPL.

---

## Data I/O

| Command | Syntax | Description |
|---|---|---|
| `load` | `load "path" as alias [, opts]` | Load CSV, TSV, JSON, DTA, XLSX, Parquet, SQLite, ODBC |
| `export` | `export(value, "fmt", "path")` | Export DataFrame or model (csv, json, tsv, xlsx, parquet, sqlite, latex, html) |
| `input` | `input alias` ... `end` | Create DataFrame from inline numeric data |

Load options: `sheet=`, `table=`, `query=`, `sep=`. URLs are downloaded automatically.

---

## Data Manipulation

| Command | Syntax | Description |
|---|---|---|
| `generate` | `generate df var = expr` | Create new column |
| `replace` | `replace df var = expr [if cond]` | Replace column values conditionally |
| `drop` | `drop(df, col1, col2, ...)` | Remove columns |
| `keep` | `keep(df, col1, col2, ...)` | Keep only specified columns |
| `dropna` | `dropna(df [, col1, ...])` | Remove rows with missing values |
| `rename` | `rename(df, old, new)` | Rename a column |
| `sort` | `sort(df, var1 [, desc=true])` | Sort by columns |
| `filter` | `filter(df, condition)` | Filter rows by condition |
| `merge` | `merge(df1, df2, key=col [, type=])` | Merge two DataFrames (inner, left, right, outer) |
| `append` | `append(df1, df2)` | Stack DataFrames vertically |
| `collapse` | `collapse(df, stat, vars, by=group)` | Aggregate by group (mean, sd, min, max, median, count, sum) |
| `reshape` | `reshape(df, id=col, stubs=[...])` | Reshape wide to long |
| `encode` | `encode(df, col [, gen=new])` | String to numeric encoding |
| `decode` | `decode(df, col)` | Numeric to string |
| `winsor` | `winsor(df, var, p=0.01 [, gen=])` | Winsorize at percentiles |
| `tabgen` | `tabgen(df, var [, prefix=])` | Generate dummy variables |
| `recode` | `recode(df, var, from=[], to=[])` | Recode values |
| `destring` | `destring(df, var)` | Convert string column to numeric |
| `duplicates` | `duplicates(df, var [, action=])` | Report, drop, or tag duplicates |
| `label` | `label(df, var, "desc")` | Attach variable label |
| `preserve` | `preserve(df)` | Snapshot DataFrame |
| `restore` | `restore(df)` | Restore to last snapshot |

---

## Descriptive Statistics

| Command | Syntax | Description |
|---|---|---|
| `summarize` | `summarize(df [, vars, detail=true])` | Summary statistics |
| `tabulate` | `tabulate(df, var1 [, var2, chi2=true])` | Frequency / cross-tabulation |
| `tabstat` | `tabstat(df, vars, stats=[...] [, by=])` | Customizable summary table |
| `correlate` | `correlate(df, var1, var2, ...)` | Correlation matrix |
| `pwcorr` | `pwcorr(df, var1, var2, ...)` | Pairwise correlations with stars |
| `ci` | `ci(df, var [, level=0.95])` | Confidence interval for mean |
| `centile` | `centile(df, var [, percentiles=[]])` | Arbitrary percentiles |
| `ttest` | `ttest(df, var [, mu= \| by= \| paired])` | T-test (one-sample, two-sample, paired) |
| `xtsum` | `xtsum(df, var)` | Panel summary (between/within) |
| `count` | `count df [if cond]` | Count observations |
| `describe` | `describe(df)` | Variable names, types, labels |
| `list` | `list(df [, vars=[], n=10])` | Show observations |
| `anova` | `anova(df, var, by="group")` | One-way ANOVA |
| `manova` | `manova(df, vars, by="group")` | Multivariate ANOVA |

---

## Estimation

| Command | Syntax | Description |
|---|---|---|
| `ols` / `reg` | `ols(Y ~ X1 + X2, df [, opts])` | OLS with HC0-HC4, cluster, NW |
| `iv` | `iv(Y ~ X1 + Xendog, ~ Z1 + Z2, df)` | IV / 2SLS |
| `logit` | `logit(Y ~ X, df)` | Logistic regression |
| `probit` | `probit(Y ~ X, df)` | Probit regression |
| `ologit` | `ologit(Y ~ X, df)` | Ordered logit |
| `oprobit` | `oprobit(Y ~ X, df)` | Ordered probit |
| `mlogit` | `mlogit(Y ~ X, df)` | Multinomial logit |
| `cloglog` | `cloglog(Y ~ X, df)` | Complementary log-log |
| `poisson` | `poisson(Y ~ X, df)` | Poisson regression |
| `nbreg` | `nbreg(Y ~ X, df)` | Negative binomial |
| `zip` | `zip(Y ~ X, df [, inflate=])` | Zero-inflated Poisson |
| `zinb` | `zinb(Y ~ X, df [, inflate=])` | Zero-inflated negative binomial |
| `tobit` | `tobit(Y ~ X, df [, ll=, ul=])` | Tobit censored regression |
| `heckman` | `heckman(Y ~ X, df, select=formula)` | Heckman selection model |
| `qreg` | `qreg(Y ~ X, df [, q=0.5])` | Quantile regression |
| `fe` | `fe(Y ~ X, df [, id=col])` | Fixed effects (within) |
| `re` | `re(Y ~ X, df [, id=col])` | Random effects (GLS) |
| `ab` | `ab(Y ~ X, df, id=, time=)` | Arellano-Bond |
| `sysgmm` | `sysgmm(Y ~ X, df, id=, time=)` | System GMM (Blundell-Bond) |
| `pcse` | `pcse(Y ~ X, df, id=, time=)` | Panel-corrected SEs |
| `xtgls` | `xtgls(Y ~ X, df, id=, time=)` | Feasible GLS for panels |
| `glsar` | `glsar(Y ~ X, df [, lags=])` | GLS with AR errors |
| `mixed` | `mixed(Y ~ X, df, id=var)` | Mixed-effects / HLM |
| `lasso` | `lasso(Y ~ X, df [, lambda=])` | Lasso (L1) |
| `ridge` | `ridge(Y ~ X, df [, lambda=])` | Ridge (L2) |
| `elasticnet` | `elasticnet(Y ~ X, df [, alpha=])` | Elastic net |
| `glm` | `glm(Y ~ X, df, family=, link=)` | Generalized linear model |
| `rlm` | `rlm(Y ~ X, df [, method=])` | Robust M-estimation |
| `gee` | `gee(Y ~ X, df, id= [, family=, corr=])` | Generalized estimating equations |
| `betareg` | `betareg(Y ~ X, df)` | Beta regression (proportions) |
| `arima` | `arima(df, var, p=, d=, q= [, SARIMA])` | ARIMA / SARIMA |
| `garch` | `garch(df, var [, p=, q=, dist=])` | GARCH volatility |
| `egarch` | `egarch(df, var [, p=, q=])` | EGARCH (asymmetric) |
| `var` | `var(df, var1, var2 [, lags=])` | Vector autoregression |
| `vecm` | `vecm(df, var1, var2 [, lags=, rank=])` | Vector error correction |
| `svar` | `svar(df, var1, var2 [, lags=, type=])` | Structural VAR |
| `cox` | `cox(Y ~ X, df [, time=])` | Cox proportional hazards |
| `km` | `km(df, time=, event= [, by=])` | Kaplan-Meier survival |
| `did` | `did(Y ~ X, df, treat=, post=)` | Difference-in-differences |
| `rd` | `rd(Y ~ X, df, running=, cutoff=)` | Regression discontinuity |
| `synth` | `synth(df, outcome=, treat_unit=, ...)` | Synthetic control |
| `psm` | `psm(Y ~ X, df [, k=, caliper=])` | Propensity score matching |
| `fmb` | `fmb(Y ~ X, df, time= [, nw=])` | Fama-MacBeth |
| `lowess` | `lowess(df, y, x [, frac=, it=])` | Local polynomial smoothing |
| `pca` | `pca(df, vars [, ncomp=])` | Principal component analysis |
| `factor` | `factor(df, vars [, nfactors=])` | Factor analysis |

Panel setup: `xtset(df, id_col, time_col)` -- after declaring, panel estimators infer `id=` and `time=`.

Time series setup: `tsset df time_col` -- enables `L.x` (lag), `F.x` (lead), `D.x` (diff) operators.

Covariance options (where applicable): `cov=nonrobust|HC1|HC2|HC3|HC4|robust`, `cluster=var`, `cluster2=var`, `nw=lags`.

---

## Post-Estimation

| Command | Syntax | Description |
|---|---|---|
| `test` | `test(m, "var")` or `test(m, "X1 = X2")` | Wald / F-test |
| `test` | `test(m, "white")`, `test(m, "bp")`, `test(m, "dw")` | Diagnostic tests |
| `nlcom` | `nlcom(m, expr)` | Nonlinear combination (delta method) |
| `margins` | `margins(m [, dydx=[], at_var=])` | Average marginal effects |
| `predict` | `predict df var = m [, "kind"]` | Predicted values (xb, residuals, pr, count) |
| `esttab` | `esttab(m1, m2 [, fmt=, path=])` | Model comparison table |
| `eststo` | `eststo(expr)` | Store model for `esttab` |
| `estat` / `ic` | `estat(m1, m2)` | AIC/BIC comparison |
| `hausman` | `hausman(m_fe, m_re)` | Hausman specification test |
| `lincom` | `lincom(m, expr)` | Linear combination of coefficients |
| `bootstrap` | `bootstrap(est, formula, df, n=)` | Generic bootstrap |
| `bootse` | `bootse(est, formula, df, n=)` | Bootstrap standard errors |
| `vif` | `vif(m)` | Variance inflation factors |
| `influence` | `influence(m)` | DFFITS, Cook's D, leverage |
| `irf` | `irf(m [, periods=])` | Impulse response function |
| `fevd` | `fevd(m [, periods=])` | Forecast error variance decomposition |
| `coefplot` | `coefplot(m [, width=])` | ASCII coefficient plot |

---

## Statistical Tests

| Command | Syntax | Description |
|---|---|---|
| `adf` | `adf(df, var [, lags=])` | Augmented Dickey-Fuller |
| `kpss` | `kpss(df, var)` | KPSS stationarity test |
| `pp` | `pp(df, var)` | Phillips-Perron |
| `ljungbox` | `ljungbox(df, var [, lags=])` | Ljung-Box autocorrelation |
| `archtest` | `archtest(df, var [, lags=])` | Engle's ARCH test |
| `granger` | `granger(df, y, x [, lags=])` | Granger causality |
| `johansen` | `johansen(df, var1, var2 [, lags=])` | Johansen cointegration |
| `bptest` | `bptest(df, formula, id=)` | Breusch-Pagan LM |
| `jb` | `jb(df, var)` | Jarque-Bera normality |
| `reset` | `reset(m)` | Ramsey RESET |
| `white` | `white(m)` | White heteroskedasticity |
| `dw` | `test(m, "dw")` | Durbin-Watson |

---

## Finance

| Command | Syntax | Description |
|---|---|---|
| `fmb` | `fmb(Y ~ X, df, time= [, nw=])` | Fama-MacBeth with optional NW |
| `portsort` | `portsort(df, ret, sortvar, n= [, time=])` | Portfolio sort (quintile spreads) |
| `doublesort` | `doublesort(df, ret, v1, v2, n1=, n2=)` | Double portfolio sort |

---

## Graphs

| Command | Output | Description |
|---|---|---|
| `scatter(df, x, y)` | ASCII | Scatter plot |
| `histogram(df, var [, bins=])` | ASCII | Histogram |
| `boxplot(df, var)` | ASCII | Box plot |
| `kdensity(df, var)` | ASCII | Kernel density |
| `qqplot(df, var)` | ASCII | Q-Q plot |
| `corrplot(df, vars)` | ASCII | Correlation heatmap |
| `acfplot(df, var [, lags=])` | ASCII | ACF correlogram |
| `pacfplot(df, var [, lags=])` | ASCII | PACF correlogram |
| `graph_scatter(df, x, y, path=)` | SVG | Scatter plot |
| `graph_line(df, x, y, path=)` | SVG | Line plot |
| `graph_hist(df, var, path=)` | SVG | Histogram |
| `graph_coef(m, path=)` | SVG | Coefficient plot |

---

## Math Functions

Available inside `generate` expressions and general arithmetic:

`abs`, `ceil`, `floor`, `round`, `log`, `log10`, `exp`, `sqrt`, `min`, `max`, `mean`, `sd`, `sum`, `mod`.

---

## String Functions

| Function | Description |
|---|---|
| `upper(s)` | Convert to uppercase |
| `lower(s)` | Convert to lowercase |
| `trim(s)` | Remove leading/trailing whitespace |
| `substr(s, start, len)` | Substring extraction |
| `split(s, delim)` | Split into list |
| `str_replace(s, from, to)` | Replace all occurrences |
| `contains(s, sub)` | Substring test |
| `regexm(s, pat)` | Regex match test |
| `regexr(s, pat, rep)` | Replace first regex match |
| `regexra(s, pat, rep)` | Replace all regex matches |
| `regexs(s, pat)` | Extract first capture group |
| `format(val, fmt)` | Format number as string |
| `len(s)` | String length |

---

## Date/Time Functions

| Function | Description |
|---|---|
| `date("YYYY-MM-DD")` | Parse date to timestamp |
| `datetime("YYYY-MM-DD HH:MM:SS")` | Parse datetime to timestamp |
| `year(col)` | Extract year |
| `month(col)` | Extract month |
| `day(col)` | Extract day |
| `hour(col)` | Extract hour |
| `minute(col)` | Extract minute |
| `second(col)` | Extract second |
| `dow(col)` | Day of week (0=Mon, 6=Sun) |

---

## Language & Utility

| Command | Description |
|---|---|
| `let x = expr` | Declare mutable variable |
| `const X = expr` | Declare immutable variable |
| `display expr` | Print scalar value |
| `print(expr)` | Print any value |
| `source("file.hay")` | Execute script (always re-runs) |
| `import("module")` | Load module once (namespaced) |
| `plugin_path("dir")` | Register plugin search directory |
| `quietly(expr)` | Suppress output |
| `capture(expr)` | Ignore errors (returns nil) |
| `assert(cond, "msg")` | Error if condition is false |
| `timer(expr)` | Time an expression |
| `set_seed(n)` | Set RNG seed |
| `help(topic)` | Built-in help (~110 topics) |
| `type(x)` | Get type name |
| `int()` `float()` `str()` `bool()` | Type conversions |
