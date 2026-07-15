# Command Reference

Quick reference for all Hayashi commands and functions. For detailed usage, type `help(command)` in the REPL.

---

## Data I/O

| Command | Syntax | Description |
|---|---|---|
| `load` | `load "path" as alias [, opts]` | Load CSV, TSV, JSON, DTA, XLSX, Parquet, SQLite, ODBC |
| `export` | `export(value, "fmt", "path")` | Export DataFrame or model (csv, json, tsv, xlsx, parquet, sqlite, latex, html) |
| `input` | `input alias` ... `end` | Create DataFrame from inline numeric data |

Load options: `sheet=`, `table=`, `query=`, `sep=`, `columns=`, `where=`. URLs are downloaded automatically. `columns=` and `where=` push projection and filtering down to the source (Parquet, SQLite, ODBC, CSV/TSV, DTA, Excel) — see [Loading Data](../data/loading.md#column-projection-and-row-filtering-columns-where).

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
| `append` | `append(df1, df2)` | Stack two DataFrames vertically |
| `rbind` | `rbind(list_of_dfs)` | Concatenate a list of DataFrames vertically (skips nil) |
| `collapse` | `collapse(df, stat, vars, by=group)` | Aggregate by group (mean, sd, min, max, median, count, sum) |
| `reshape` | `reshape(df, id=col, stubs=[...])` | Reshape wide to long |
| `encode` | `encode(df, col [, gen=new])` | String to numeric encoding |
| `decode` | `decode(df, col, labels=[...])` | Numeric to string |
| `winsor` | `winsor(df, var, p=0.01 [, gen=])` | Winsorize at percentiles |
| `tabgen` | `tabgen(df, var [, prefix=])` | Generate dummy variables |
| `recode` | `recode(df, var, from=[], to=[])` | Recode values |
| `destring` | `destring(df, var)` | Convert string column to numeric |
| `duplicates` | `duplicates(df, var [, action=])` | Report, drop, or tag duplicates |
| `label` | `label(df, var, "desc")` | Attach variable label |
| `drop_collinear` | `drop_collinear(df [, vars=[...]])` | Remove perfectly collinear columns |
| `preserve` | `preserve(df)` | Snapshot DataFrame |
| `restore` | `restore(df)` | Restore to last snapshot |
| `dataframe` | `dataframe({"x": [1, 2]})` | Build DataFrame from dict of lists |

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
| `ttest` | `ttest(df, var [, mu= \| by= [, unequal=false] \| paired])` | T-test (one-sample, two-sample, paired) |
| `xtsum` | `xtsum(df, var)` | Panel summary (between/within) |
| `count` | `count df [if cond]` | Count observations |
| `median` | `median(list)` or `median(df, x)` | Median |
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
| `cmnlogit` | `cmnlogit(Y ~ X, df, group=id, alts=3)` | Conditional multinomial logit |
| `clogit` | `clogit(Y ~ X, df, group=id)` | Conditional logit |
| `cpoisson` | `cpoisson(Y ~ X, df, group=id)` | Conditional Poisson / PPML |
| `poisson` | `poisson(Y ~ X, df)` | Poisson regression |
| `nbreg` | `nbreg(Y ~ X, df)` | Negative binomial |
| `zip` | `zip(Y ~ X, df [, inflate=])` | Zero-inflated Poisson |
| `zinb` | `zinb(Y ~ X, df [, inflate=])` | Zero-inflated negative binomial |
| `tobit` | `tobit(Y ~ X, df [, ll=, ul=])` | Tobit censored regression |
| `heckman` | `heckman(Y ~ X, S ~ Z, df)` | Heckman selection model |
| `qreg` | `qreg(Y ~ X, df [, q=0.5])` | Quantile regression |
| `wls` | `wls(Y ~ X, df, weights="w")` | Weighted least squares |
| `fe` | `fe(Y ~ X, df [, id=col])` | Fixed effects (within) |
| `re` | `re(Y ~ X, df [, id=col])` | Random effects (GLS) |
| `be` | `be(Y ~ X, df [, id=col])` | Between estimator |
| `feiv` | `feiv(Y ~ Xendog + X, ~ Z, df, id=col)` | Fixed-effects IV |
| `ab` | `ab(Y ~ X, df, id=, time=)` | Arellano-Bond |
| `sysgmm` | `sysgmm(Y ~ X, df, id=, time=)` | System GMM (Blundell-Bond) |
| `pcse` | `pcse(Y ~ X, df, id=, time=)` | Panel-corrected SEs |
| `xtgls` | `xtgls(Y ~ X, df, id=, time=)` | Feasible GLS for panels |
| `pthresh` | `pthresh(Y ~ X, df, id=, q=)` | Panel threshold model |
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
| `autoreg` | `autoreg(df, var [, lags=])` | Autoregression |
| `ardl` | `ardl(df, y, x [, p=, q=])` | Autoregressive distributed lag model |
| `kalman` | `kalman(df, var [, model=])` | State-space Kalman smoothing |
| `garch` | `garch(df, var [, p=, q=, dist=])` | GARCH volatility |
| `egarch` | `egarch(df, var [, p=, q=])` | EGARCH (asymmetric) |
| `gjrgarch` | `gjrgarch(df, var [, p=, q=])` | GJR-GARCH volatility |
| `var` | `var(df, var1, var2 [, lags=])` | Vector autoregression |
| `vecm` | `vecm(df, var1, var2 [, lags=, rank=])` | Vector error correction |
| `varma` | `varma(df, vars [, p=, q=])` | VARMA / VARMAX |
| `svar` | `svar(df, var1, var2 [, lags=, type=])` | Structural VAR |
| `ucm` | `ucm(df, var)` | Unobserved components model |
| `ets` | `ets(df, var)` | Exponential smoothing |
| `msauto` | `msauto(df, var [, regimes=])` | Markov-switching autoregression |
| `cox` | `cox(Y ~ X, df [, time=])` | Cox proportional hazards |
| `km` | `km(df, time=, event= [, by=])` | Kaplan-Meier survival |
| `did` | `did(Y ~ X, df, treat=, post=)` | Difference-in-differences |
| `rd` | `rd(Y ~ running, cutoff, df [, bw=, poly=])` | Regression discontinuity |
| `fuzzy_rd` | `fuzzy_rd(Y ~ X, "treat", cutoff, df)` | Fuzzy regression discontinuity |
| `synth` | `synth("Y", "treated_id", t0, df, id=, time=)` | Synthetic control |
| `psm` | `psm(Y ~ treat + X, df [, k=, caliper=])` | Propensity score matching |
| `fmb` | `fmb(Y ~ X, df, time= [, nw=])` | Fama-MacBeth |
| `lowess` | `lowess(df, y, x [, frac=, it=])` | Local polynomial smoothing |
| `sur` | `sur(df, y1 ~ x1, y2 ~ x2)` | Seemingly unrelated regressions |
| `three_sls` | `threesl(df, y1 ~ x1, y2 ~ x2, instruments=[...])` | Three-stage least squares |
| `pca` | `pca(df, vars [, ncomp=])` | Principal component analysis |
| `factor` | `factor(df, vars [, nfactors=])` | Factor analysis |
| `dfm` | `dfm(df, var1, var2 [, factors=])` | Dynamic factor model |
| `gam` | `gam(Y ~ X, df)` | Generalized additive model |
| `mice` | `mice(df, vars=["Y", "X1"])` | Multiple imputation |

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
| `estclear` | `estclear()` | Clear stored models |
| `testparm` | `testparm(m, ["x1", "x2"])` | Joint F-test |
| `estat` / `ic` | `estat(m1, m2)` | AIC/BIC comparison |
| `akaike_weights` | `akaike_weights(m1, m2)` | Akaike weights (dict) |
| `lrtest` | `lrtest(m_restricted, m_unrestricted)` | Likelihood-ratio test |
| `weak_iv` | `weak_iv(endog_formula, instr_formula, df)` | Weak instrument test (1st-stage F, Cragg-Donald) |
| `estat_overid` | `estat_overid(endog_formula, instr_formula, df)` | Sargan/Hansen J overidentification test |
| `estat_endog` | `estat_endog(endog_formula, instr_formula, df)` | Durbin-Wu-Hausman endogeneity test |
| `estat_classification` | `estat_classification(model, threshold=0.5)` | Classification table (logit/probit) |
| `lroc` | `lroc(model)` | ROC curve and AUC (logit/probit) |
| `estat_gof` | `estat_gof(model, groups=10)` | Hosmer-Lemeshow goodness-of-fit |
| `linktest` | `linktest(model)` | Specification test (logit/probit) |
| `xtlogit` | `xtlogit(y ~ x, df, id="g")` | Panel logit (GEE) |
| `xtprobit` | `xtprobit(y ~ x, df, id="g")` | Panel probit (GEE) |
| `xtpoisson` | `xtpoisson(y ~ x, df, id="g")` | Panel Poisson (GEE) |
| `eventstudy` | `eventstudy(y ~ etime + x, df, ref=-1, min=-5, max=5)` | Event study (dynamic DiD) |
| `nls_exp` | `nls_exp(y ~ x, df, start=[a,b])` | NLS exponential: y = a*exp(b*x) |
| `nls_power` | `nls_power(y ~ x, df, start=[a,b])` | NLS power: y = a*x^b |
| `nls_logistic` | `nls_logistic(y ~ x, df, start=[a,b,c])` | NLS logistic: y = a/(1+exp(-b*(x-c))) |
| `nls_cobb_douglas` | `nls_cobb_douglas(y ~ x1+x2, df, start=[a,b0,b1])` | Cobb-Douglas production |
| `nls_ces` | `nls_ces(y ~ x1+x2, df, start=[a,b1,b2,rho])` | CES production |
| `marginsplot` | `marginsplot(m)` | AME plot for logit/probit |
| `spatial_sar` | `spatial_sar(y ~ x, df, w=W)` | Spatial autoregressive (SAR) |
| `spatial_sem` | `spatial_sem(y ~ x, df, w=W)` | Spatial error model (SEM) |
| `double_ml` | `double_ml(y ~ d + x1 + x2, df, folds=5)` | Double/debiased ML (Chernozhukov) |
| `sfa_production` | `sfa_production(y ~ x1 + x2, df)` | Stochastic production frontier |
| `sfa_cost` | `sfa_cost(y ~ x1 + x2, df)` | Stochastic cost frontier |
| `panel_tobit` | `panel_tobit(y ~ x, df, id="firm", censor=0)` | Panel Tobit (random effects) |
| `panel_heckman` | `panel_heckman(y ~ x, df, sel="z ~ w", id="firm")` | Panel Heckman (selection) |
| `spatial_panel_sar` | `spatial_panel_sar(y ~ x, df, w=W, id="entity")` | Spatial panel SAR (FE) |
| `spatial_panel_sem` | `spatial_panel_sem(y ~ x, df, w=W, id="entity")` | Spatial panel SEM (FE) |
| `bayes_sfa_production` | `bayes_sfa_production(y ~ x, df, burn=, draws=)` | Bayesian SFA production |
| `bayes_sfa_cost` | `bayes_sfa_cost(y ~ x, df, burn=, draws=)` | Bayesian SFA cost |
| `midas` | `midas(y ~ x, df, freq=3, lags=12, poly=2)` | Mixed Data Sampling regression |
| `tvp` | `tvp(y ~ x1 + x2, df)` | Time-Varying Parameter (Kalman filter) |
| `setar` | `setar(y ~ 1, df, order=2, delay=1)` | Self-Exciting Threshold AR |
| `panel_qreg` | `panel_qreg(y ~ x, df, id="firm", tau=0.5)` | Panel quantile (FE) |
| `msvar` | `msvar(y1 ~ y2, df, regimes=2, lags=1)` | Markov-Switching VAR |
| `favar` | `favar(y1 ~ y2 + y3, df, observed="r", factors=2)` | Factor-Augmented VAR |
| `spatial_durbin` | `spatial_durbin(y ~ x, df, w=W, id="e")` | Spatial Durbin (panel FE) |
| `johansen_break` | `johansen_break(y1 ~ y2, df, lags=1, breaks=[50])` | Johansen with breaks |
| `tvp_var` | `tvp_var(y1 ~ y2, df, lags=1)` | Time-Varying Parameter VAR |
| `spatial_durbin_error` | `spatial_durbin_error(y ~ x, df, w=W, id="e")` | Spatial Durbin Error (panel FE) |
| `fmols` | `fmols(y ~ x, df)` | Fully Modified OLS (cointegration) |
| `qvar` | `qvar(y1 ~ y2, df, lags=1, tau=0.5)` | Quantile VAR |
| `pstr` | `pstr(y ~ x, df, q="var", id="e")` | Panel Smooth Transition |
| `modwt` | `modwt(df, var, scales=4)` | Wavelet decomposition (MODWT) |
| `copula` | `copula(y1 ~ y2, df, type="gaussian")` | Copula dependence |
| `hausman` | `hausman(m_fe, m_re)` | Hausman specification test |
| `lincom` | `lincom(m, expr)` | Linear combination of coefficients |
| `bootstrap` | `bootstrap(est, formula, df, n=)` | Generic bootstrap |
| `bootse` | `bootse(est, formula, df, n=)` | Bootstrap standard errors |
| `vif` | `vif(m)` | Variance inflation factors |
| `influence` | `influence(m)` | DFFITS, Cook's D, leverage |
| `cusumtest` | `cusumtest(m)` | CUSUM structural stability test |
| `acf` | `acf(df, var [, lags=])` or `acf(m [, lags=])` | ACF values (list) |
| `pacf` | `pacf(df, var [, lags=])` or `pacf(m [, lags=])` | PACF values (list) |
| `gqtest` | `gqtest(m [, split=])` | Goldfeld-Quandt heteroskedasticity |
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
| `bptest` | `bptest(df, formula, id=)` | Breusch-Pagan LM (RE vs OLS) |
| `ftest_fe` | `ftest_fe(df, formula, id=)` | F-test (FE vs OLS) |
| `wooldridge` | `wooldridge(df, formula, id=, time=)` | Wooldridge serial correlation |
| `pesaran` | `pesaran(df, formula, id=, time=)` | Pesaran CD cross-sectional dependence |
| `abtest` | `abtest(df, formula, id=, time=)` | Arellano-Bond m1/m2 |
| `mundlak` | `mundlak(df, formula, id=)` | Mundlak (RE vs FE) |
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
