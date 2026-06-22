# Hayashi

An interpreted language for applied econometrics. Named after [Fumio Hayashi](https://en.wikipedia.org/wiki/Fumio_Hayashi).

Single binary, no dependencies, Stata-like syntax. Built in Rust on top of [Greeners](https://github.com/sheep-farm/Greeners).

## Install

Requires only the Rust toolchain. No system dependencies.

```bash
git clone https://github.com/sheep-farm/hayashi.git
cd hayashi
cargo build --release
# Binary at target/release/hayashi
```

## Usage

```bash
hayashi                  # interactive REPL (multi-line)
hayashi script.hy        # run a script
hayashi -                # read from stdin
hayashi --help           # list commands
```

## Quick start

```stata
load "data.dta" as df

// OLS with clustered standard errors
let m1 = reg(Y ~ X1 + X2, df, cluster=firm)
let m2 = reg(Y ~ X1 + X2 + X3, df, cluster=firm)

// Comparison table (ASCII or LaTeX)
esttab(m1, m2)
esttab(m1, m2, fmt=latex, path="table1.tex")

// Post-estimation
test(m2, X2, X3)              // joint F-test
test(m2, "X2 = X3")           // Wald: beta_X2 = beta_X3
nlcom(m2, X2 / X3)            // delta method
coefplot(m2)                  // ASCII coefficient plot

// Bootstrap (works with any estimator)
bootstrap(ols, Y ~ X1 + X2, df, n=1000)
```

## Estimators

| Category | Commands |
|---|---|
| Linear | `ols` `reg` `iv` `wls` `glsar` |
| Panel | `fe` `re` `ab` `sysgmm` `pcse` `xtgls` |
| Binary | `logit` `probit` `cloglog` `clogit` |
| Count | `poisson` `nbreg` `zip` `zinb` |
| Ordinal | `ologit` `oprobit` |
| Censored | `tobit` `heckman` `truncreg` |
| Survival | `cox` `km` |
| Quantile | `qreg` |
| Regularization | `lasso` `ridge` `elasticnet` |
| Time series | `arima` `sarima` `garch` `egarch` `gjrgarch` |
| VAR | `var` `vecm` `svar` `irf` `fevd` |
| Causal | `did` `rd` `synth` `psmatch` |
| Finance | `fmb` (Fama-MacBeth + Newey-West) |
| Robust | `rlm` `gee` `glm` `betareg` |

All estimators support `if=` for subsamples and `cov=`/`cluster=` for robust SEs.

## Post-estimation

```stata
test(m, X1, X2)              // joint F-test H0: beta=0
test(m, "X1 = X2")           // linear restriction
test(m, "X1 = 0.5")          // value test
test(m, white)               // White heteroskedasticity
test(m, bp)                  // Breusch-Pagan
test(m, dw)                  // Durbin-Watson
nlcom(m, X1 / X2)            // nonlinear combination (delta method)
margins(m)                   // average marginal effects
margins(m, dydx=[X1])        // specific variables
margins(m, at_X2=0)          // at fixed values
coefplot(m)                  // ASCII coefficient plot with 95% CI
estat(m1, m2, m3)            // AIC/BIC comparison
hausman(m_fe, m_re)          // Hausman test
predict df yhat = m          // fitted values
predict df e = m, residuals  // residuals
bootstrap(ols, Y ~ X, df, n=1000)
```

## Data

```stata
load "file.csv" as df
load "file.dta" as df          // Stata .dta files
load "https://...csv" as df    // URLs

generate df lnY = log(Y)
generate df D = (X == 1)       // dummy from condition
generate df D = (col == "val") // string comparison
replace df Y = 0 if X > 10
drop(df, col)
keep(df, col1, col2)
winsor(df, Y, p=0.01)          // winsorize at 1%/99%
encode(df, str_col)             // string -> numeric
tabgen(df, group)               // generate dummies

summarize(df)
tabulate(df, group)
correlate(df, X1, X2, X3)
pwcorr(df, X1, X2, X3)         // with significance stars
list(df, vars=[X1, X2], n=10)
ttest(df, Y, by=group)
```

## Panel

```stata
xtset(df, firm, year)

let m_fe = fe(Y ~ X1 + X2, df)
let m_re = re(Y ~ X1 + X2, df)
hausman(m_fe, m_re)
esttab(m_fe, m_re)
```

After `xtset`, panel estimators use stored id/time automatically.

## Finance

```stata
// Fama-MacBeth with Newey-West
fmb(ret ~ beta + size + bm, df, time=month, nw=4)

// Portfolio sorts
portsort(df, ret, beta, n=5)
doublesort(df, ret, size, bm, n1=5, n2=5)
```

## Automation

```stata
// Dynamic formulas + eststo loop
for spec in ["X1", "X1 + X2", "X1 + X2 + X3"] {
    eststo(ols("Y ~ " + spec, df, cluster=firm))
}
esttab(fmt=latex, path="table1.tex")
estclear()
```

## Scoping

Variables are block-scoped. `{}` controls lifetime, `Rc` drop is deterministic -- no GC.

```stata
let x = 10
{
    let temp = 42   // dies at }
    x = x + temp    // modifies outer x (assign without let)
}
// temp is gone, x = 52
```

`let` declares in current scope. Assignment without `let` modifies the nearest enclosing scope.

## Language

```stata
let x = 5
scalar mu = mean(df, Y)
display mu

if x > 3 { ... } else { ... }
for i in 1..10 { ... }
for v in ["a", "b"] { ... }
while cond { ... }
fn square(x) { return x * x }

set_seed(42)
timer(ols(Y ~ X, df))
source("other_script.hy")
help(ols)
```

## Build & Test

```bash
cargo build --release      # optimized binary
cargo test                 # 208 tests, ~3s
```

## License

MIT
