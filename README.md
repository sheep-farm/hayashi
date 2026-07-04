# 林 Hayashi Language

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Docs: CC BY-SA 4.0](https://img.shields.io/badge/Docs-CC%20BY--SA%204.0-lightgrey.svg)](LICENSE-BOOKS.md)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-0.2.7--dev-yellow.svg)](Cargo.toml)
[![crates.io](https://img.shields.io/crates/v/hayashi-lang.svg)](https://crates.io/crates/hayashi-lang)
[![CI](https://github.com/sheep-farm/hayashi/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/sheep-farm/hayashi/actions/workflows/ci.yml?query=branch%3Amaster)

An interpreted language for applied econometrics. Named after [Fumio Hayashi](https://en.wikipedia.org/wiki/Fumio_Hayashi).

Stata-like syntax, modern language features, zero cost. Built in Rust on top of [Greeners](https://github.com/sheep-farm/Greeners).

## Install

```bash
git clone https://github.com/sheep-farm/hayashi.git
cd hayashi
cargo build --release
# Binary at target/release/hay

# Optional: ODBC support (requires unixodbc)
cargo build --release --features odbc
```

## Usage

```bash
hay                  # interactive REPL (tab completion, syntax highlighting)
hay script.hay       # run a script
hay -                # read from stdin
hay --help           # list commands
```

REPL features: tab completion for keywords + variables, syntax highlighting (keywords blue, strings green, numbers yellow), history hints (fish-style).

## Quick start

```
load "data.dta" as df

let m1 = reg(Y ~ X1 + X2, df, cluster=firm)
let m2 = reg(Y ~ X1 + X2 + X3, df, cluster=firm)

esttab(m1, m2)
esttab(m1, m2, fmt=latex, path="table.tex")

export(df, "csv", "data.csv")
export(df, "xlsx", "data.xlsx")
export(df, "parquet", "data.parquet")
export(m1, "latex", "table.tex")
```

## Language

Hayashi is a dynamically-typed, block-scoped interpreted language. It combines Stata-like econometrics syntax with modern programming constructs.

### Type system

| Type | Literal | Notes |
|---|---|---|
| `int` | `42` | 64-bit signed integer |
| `float` | `3.14` | 64-bit IEEE 754 |
| `bool` | `true` / `false` | |
| `string` | `"hello"` | UTF-8, immutable |
| `list` | `[1, 2, 3]` | Heterogeneous; `push`/`pop` mutate, rest COW |
| `dict` | `{"key": value}` | String keys, any values, immutable (COW) |
| `nil` | | Absence of value |
| `dataframe` | via `load` / `input` | Tabular data, copy-on-write (`Rc`) |
| `function` | `fn name(x) { }` or `\|x\| expr` | Named or anonymous (closure) |

Explicit conversions: `int(x)`, `float(x)`, `str(x)`, `bool(x)`.
Introspection: `type(x)` returns the type name as a string.

### Variables and mutability

```
let x = 10              // mutable — can be reassigned
const PI = 3.14159       // immutable — error on reassign or redeclare
x = 20                   // assign without let — searches outer scopes
```

No variable shadowing: `let x` over an existing `const x` is an error, even in an inner scope. This prevents subtle bugs common in C/C++.

### Scoping

Block-scoped with deterministic destruction. No garbage collector.

```
let x = 10
if true {
    let temp = 42        // lives only in this block
    x = x + temp         // modifies outer x
}
// temp is gone, x = 52
```

Function parameters are `const` by default — data enters immutable, result exits via `return`. DataFrames use `Rc<DataFrame>` for zero-copy passing and copy-on-write mutation.

### Control flow

| Construct | Syntax | Returns value? |
|---|---|---|
| If statement | `if cond { } else if { } else { }` | No |
| If expression | `if cond { a } else { b }` | Yes |
| Match | `match expr { pat => result, _ => default }` | Yes |
| Block expression | `{ stmt; ...; expr }` | Yes |
| For loop | `for i in 1..10 { }` / `for v in list { }` | No |
| While loop | `while cond { }` | No |
| Try/catch | `try { } catch e { }` | No |
| Break/continue | `break` / `continue` | — |
| Return | `return expr` | — |

### Block expressions

A block `{ stmt; ...; expr }` evaluates a sequence of statements and returns the value of the last expression. Variables declared inside the block are local to it.

```
let df = {
    let raw = load("data.csv")
    generate raw y = log(x)
    keep(raw, ["date", "y"])
    raw
}
// df is available; raw and the temporary columns are gone
```

### Output control

`quietly on` suppresses automatic output from statements and estimators. `print(...)` and `display ...` still appear. `quietly off` restores normal output. The flag is scope-aware: a toggle inside a block reverts when the block ends.

```
quietly on

let df = {
    let a = load("a.csv")
    let b = load("b.csv")
    let m = merge(a, b, key=id, type=inner)
    generate m z = x - y
    m
}

quietly off

ols(z ~ x, df)
print("done")
```

Both `quietly on` and `quietly()` (function form) share the same suppression mechanism, so new commands need no special handling — they just use the internal output channel.

### Functions and closures

```
// Named function — parameters are const
fn add(a, b) {
    let result = a + b
    return result
}

// Closure — anonymous, captures outer scope
let double = |x| x * 2
let big = filter(list, |x| x > 10)
```

### Operators

| Category | Operators |
|---|---|
| Arithmetic | `+` `-` `*` `/` `^` `**` `%` |
| Assignment | `=` `+=` `-=` `*=` `/=` `%=` |
| Comparison | `==` `!=` `>` `<` `>=` `<=` |
| Logical | `&&` (or `&`) `\|\|` `!` |
| Membership | `in` — works with list, dict (key), string (substring) |
| Pipe | `\|>` — passes left side as first argument (or replacing `_` placeholder) |
| Index | `list[i]` `dict["key"]` |
| String | `+` for concatenation |

### String interpolation

```
let msg = f"mean = {mu:.2f}, n = {n}, p-value = {p:.4e}"
```

F-strings support any expression inside `{}` and format specifiers: `.Nf` (decimal places), `.Ne` (scientific notation). Escape braces with `{{` and `}}`.

### Collections

`push(list, item)` and `pop(list)` mutate in-place (like Python/JS). Other list operations return new lists, leaving the original unchanged.

**List operations:** `push` `pop` `insert` `remove` `clear` `reverse` `index` `slice` `join` `map` `filter` `unique` `flatten` `sort` `range` `len`

**Dict operations:** `keys` `values` `has_key` `dict_set` `dict_remove` `dict_merge` `len`

**Pipe chaining:**
```
[5, 3, 1, 4, 2]
    |> filter(|x| x > 2)
    |> sort
    |> map(|x| x * 10)
    |> reverse

value |> |x| x * 3           // pipe with inline closure
exper |> dobro                // pipe with user function
df |> ols(lw ~ yos, _)        // pipe using '_' as placeholder for specific argument positions
```

## Data I/O

```
// Load — 8 formats + URL
load "file.csv" as df
load "file.tsv" as df
load "file.json" as df
load "file.dta" as df                    // Stata
load "file.xlsx" as df, sheet=Plan1      // Excel (xlsx/xls/ods)
load "file.parquet" as df                // Apache Parquet
load "file.db" as df, table=prices       // SQLite
load "file.db" as df, query="SELECT * FROM prices WHERE year > 2020"
load "odbc://DSN=mydb" as df, query="SELECT * FROM t"  // ODBC (feature flag)
load "https://...data.csv" as df         // URL (auto-download)
load "data.csv" as df, sep=";"           // custom delimiter

// Export — 8 formats
export(df, "csv", "out.csv")
export(df, "json", "out.json")
export(df, "tsv", "out.tsv")
export(df, "xlsx", "out.xlsx")
export(df, "parquet", "out.parquet")
export(df, "sqlite", "out.db")
export(m, "latex", "table.tex")
export(m, "html", "table.html")
```

`query=` is raw SQL executed by SQLite or the configured ODBC database. Remote `load` downloads untrusted input even with URL validation and size/time limits. ODBC support is optional and requires system ODBC drivers. See the [Trust Model](docs/src/trust-model.md).

## Estimators

| Category | Commands |
|---|---|
| Linear | `ols` `reg` `iv` `wls` `glsar` |
| Panel | `fe` `re` `be` `feiv` `ab` `sysgmm` `pcse` `xtgls` |
| Binary | `logit` `probit` `clogit` |
| Count | `poisson` `nbreg` `zip` `zinb` |
| Ordinal | `ologit` `oprobit` `mlogit` `cmnlogit` |
| GMM | `gmm` |
| Censored | `tobit` `heckman` |
| Survival | `cox` `km` |
| Quantile | `qreg` |
| Regularization | `lasso` `ridge` `elasticnet` |
| Time series | `arima` `sarima` `autoreg` `ardl` `kalman` `garch` `egarch` `gjrgarch` |
| VAR | `var` `vecm` `varma` `svar` `irf` `fevd` |
| Causal | `did` `rd` `fuzzy_rd` `synth` `psm` |
| Finance | `fmb` `portsort` `doublesort` |
| Robust / flexible | `rlm` `gee` `glm` `betareg` `mixed` `lowess` `gam` |
| Systems / factors | `sur` `three_sls` `pca` `factor` `dfm` |

Common options include `if=` for subsamples and `cov=`/`cluster=`/`nw=` where supported. Core regression estimators auto-detect and drop perfectly collinear variables (Stata-style `(omitted)` display).

## Post-estimation

```
test(m, "X1", "X2")          // joint F-test
test(m, "X1 = X2")           // linear restriction
test(m, "white")             // White heteroskedasticity
test(m, "bp")                // Breusch-Pagan
test(m, "dw")                // Durbin-Watson
nlcom(m, X1 / X2)            // nonlinear combination (delta method)
margins(m)                   // AME with SEs, z-values, p-values
coefplot(m)                  // ASCII coefficient plot with 95% CI
estat(m1, m2, m3)            // AIC/BIC comparison
hausman(m_fe, m_re)          // Hausman test
predict df yhat = m              // fitted values
predict df e = m, "residuals"    // residuals
bootstrap(ols, Y ~ X, df, n=1000)
influence(m)                 // DFFITS, Cook's D, leverage
vif(m)                       // variance inflation factors

// Store and compare models
eststo(m1)
eststo(m2)
esttab()                     // model comparison table
estclear()                   // clear stored models

// Joint F-test
 testparm(m, ["X1", "X2"])  // H0: selected coefficients = 0
```

## Data manipulation

```
// Generate (statement — modifies in-place)
generate df lnY = log(Y)
generate df D = (X == 1)
generate df row = _n

// Mutate (function — multi-column, pipe-friendly)
let df2 = mutate(df, z = x^2, w = ln(y), ratio = x / y)
let df2 = df |> mutate(z = x * 2) |> filter(z > 5) |> sort(z)

// Pipe semantics: standalone modifies source, captured preserves it
df |> mutate(z = x^2)               // modifies df
let result = df |> mutate(z = x^2)  // df unchanged, result has z

// Selection and filtering
select(df, col1, col2)              // alias for keep
drop(df, col)
filter(df, mpg > 25 & foreign == 1)
sort(df, price)

// Aggregation
group_by(df, setor, mean, ret, vol)  // pipe-friendly
collapse(df, mean, price, mpg, by=foreign)

// Reshape
pivot_longer(df, stubs=["gdp"], i=country, j=year)
pivot_wider(df, i=id, j=year, values=gdp)

// Other
replace df Y = 0 if X > 10
merge(df1, df2, key=id, type=left)
append(df1, df2)
encode(df, region)               // string -> numeric
decode(df, region_num, labels=["north", "south", "east", "west"])
winsor(df, Y, p=0.01)
dropna(df, price, mpg)
ffill(df)                           // forward-fill NaN em colunas float
rename(df, old, new)
label(df, Y, "GDP per capita")
duplicates(df, id, action=drop)
drop_collinear(df)               // remove perfectly collinear columns
preserve(df) / restore(df)

// Time-series declaration (required for L.x, F.x, D.x operators)
tsset df year
xtset(df, firm, year)            // panel structure
```

## Date/time

```
// Parsing
let t = date("2024-06-15")              // -> Unix timestamp
let dt = datetime("2024-06-15 14:30:00")

// Extraction in generate
generate df Y = year(date_col)
generate df M = month(date_col)
generate df D = day(date_col)
generate df H = hour(date_col)
generate df W = dow(date_col)           // 0=Monday

// Filtering with scalar variables
let cutoff = date("2020-01-01")
let sub = filter(df, ts >= cutoff)
```

## Descriptive statistics

```
// summarize returns dict when captured, prints when standalone
let s = summarize(df, price, detail=true)
display s["mean"]

// All accept bare, string, variable, or list for column names
let cols = ["price", "mpg"]
summarize(df, cols)

// Descriptive commands
codebook(df)                         // detailed variable description
tabulate(df, group)
tabulate(df, row, col, chi2=true)
correlate(df, X1, X2, X3)
pwcorr(df, X1, X2, X3)
ttest(df, Y, by=group)
ci(df, Y, level=0.99)
centile(df, Y, percentiles=[10, 50, 90])
describe(df)

// Panel summary
xtsum(df, wage, hours, id=firm)       // within/between decomposition

// Normality tests
swilk(df, Y)                         // Shapiro-Wilk
sfrancia(df, Y)                      // Shapiro-Francia
sktest(df, Y)                        // Skewness/Kurtosis (JB + D'Agostino)
```

## Validation programme

Hayashi includes a reproducible, automated empirical validation programme in
`validation/`. It compares Hayashi output against reference implementations
(R and Python/statsmodels) on real datasets and on simulated DGPs taken from
the Hayashi book chapters:

```bash
python -m venv validation/.venv
validation/.venv/bin/pip install -r validation/requirements.txt
Rscript -e 'install.packages(c("wooldridge", "jsonlite"))'
hay validate
```

See `validation/README.md` for the full protocol and `validation/MATRIX.md` for
the current status of every case.

## Graphs

```
// SVG (publishable)
graph_scatter(df, X, Y, path="fig.svg")
graph_line(df, X, Y, path="fig.svg")
graph_hist(df, Y, path="fig.svg", bins=30)
graph_coef(m, path="fig.svg")

// ASCII (terminal)
scatter(df, X, Y)
histogram(df, Y)
coefplot(m)
boxplot(df, Y)
kdensity(df, Y)
acfplot(df, Y, lags=20)
qqplot(df, Y)
corrplot(df, X1, X2, X3)
```

## Types and collections

```
// Scalars
let x = 42              // int
let pi = 3.14           // float
let name = "Hayashi"    // string
let ok = true           // bool

// Constants (immutable)
const N = 1000
const TAX = 0.15

// Lists (push/pop mutate in-place, other ops return new list)
let nums = [1, 2, 3]
let doubled = nums |> map(|x| x * 2)    // [2, 4, 6]
push(nums, 4)  pop(nums)  insert(nums, 0, 99)  remove(nums, 1)
sort(nums)  reverse(nums)  unique(nums)  flatten(nested)
slice(nums, 1, 3)  index(nums, 2)  join(nums, ", ")  len(nums)

// Dicts (immutable — operations return new dict)
let d = {"name": "Alice", "age": 30}
display d["name"]
keys(d)  values(d)  has_key(d, "name")
dict_set(d, "city", "SP")  dict_remove(d, "age")  dict_merge(d1, d2)

// Build DataFrame from dict of lists
let df = dataframe({"x": [1, 2, 3], "y": [4, 5, 6]})

// Type predicates
is_int(42)       // true
is_str("hello")  // true
is_df(df)        // true

// Type conversions
int(3.9)  float(42)  str(true)  bool(0)  type(x)

// Median
median([1, 3, 2])
median(df, price)
```

## Control flow

```
// If statement
if x > 0 {
    display "positive"
} else if x == 0 {
    display "zero"
} else {
    display "negative"
}

// If expression (ternary — returns value)
let label = if x > 0 { "positive" } else { "negative" }

// Match (pattern matching — returns value)
let name = match code {
    1 => "one",
    2 => "two",
    _ => "other"
}

// Loops
for i in 1..10 { display i }
for v in ["X1", "X2"] { eststo(ols("Y ~ " + v, df)) }
while cond { ... }

// Try/catch
try {
    load "data.csv" as df
} catch e {
    display f"Error: {e}"
}
```

## Functions and closures

```
// Named functions (parameters are const by default)
fn square(x) { return x * x }
fn add(a, b) {
    let result = a + b
    return result
}

// Closures (anonymous, capture outer scope)
let doubled = map([1, 2, 3], |x| x * 2)
let big = filter(nums, |x| x > 10)
let add = |a, b| a + b
```

## Namespaces

```
// Module-based namespacing
import("finance")                    // finance::sharpe(), finance::sortino()
import("finance", as=fin)           // fin::sharpe()
import("finance", only=["sharpe"])  // sharpe() directly

// Qualified calls
let ratio = finance::sharpe(ret, rf)
```

## F-strings and operators

```
// String interpolation
let msg = f"mean = {mu:.2f}, n = {n}, p = {p:.4e}"

// Pipe operator (|>)
[5, 3, 1, 4, 2] |> sort |> reverse |> map(|x| x * 10)
df |> ols(lw ~ yos, _)             // passes df to the '_' placeholder position

// In operator (membership test)
if 3 in [1, 2, 3] { ... }
if "key" in dict { ... }
if "lo" in "hello" { ... }

// Substring / membership
contains("hello", "ell")         // true

// Regex
regexm(s, "[0-9]+")              // match → bool
regexr(s, "[0-9]+", "NUM")       // replace first
regexra(s, "[0-9]+", "NUM")      // replace all
regexs(s, "([0-9]+\\.[0-9]+)")   // extract capture
generate df is_dr = regexm(name, "^Dr")
```

## Panel and finance

```
// Panel
xtset(df, firm, year)
let m_fe = fe(Y ~ X1 + X2, df)
let m_re = re(Y ~ X1 + X2, df)
hausman(m_fe, m_re)

// Finance
fmb(ret ~ beta + size + bm, df, time=month, nw=4)
portsort(df, ret, beta, n=5)
doublesort(df, ret, size, bm, n1=5, n2=5)
```

## Scoping and mutability

```
let x = 10           // mutable
const PI = 3.14      // immutable — error on reassign

// Block scoping — variables die at }
if true {
    let temp = 42    // dies here
    x = x + temp     // modifies outer x (assign without let)
}

// Function parameters are const (immutable input, result out)
fn f(n) {
    // n = 99       // ERROR: cannot reassign const 'n'
    let result = n + 1
    return result
}

// DataFrames use copy-on-write (Rc) — zero-copy in functions
```

## Misc

```
quietly on                    // suppress automatic output from here
quietly off                   // restore automatic output
quietly(ols(Y ~ X, df))       // suppress one expression
capture(ols(Y ~ X, df))       // ignore errors
assert(n > 0, "empty data")
timer(ols(Y ~ X, df))         // time execution
set_seed(42)                   // reproducibility
source("other_script.hay")     // run another script
help(ols)                      // help() has ~210 topics with examples
help(about)                    // project info (version, license, author)
print("x =", x, "y =", y)    // multi-arg with sep= and end=
file_exists("cache/data.csv")  // bool
ensure_dir("cache")            // create directory if missing
write("text", "note.txt")      // write string to file
print("a", "b", sep=", ")     // a, b
```

## Extensibility

```
// Native & Script plugins — installed to ~/.hay/packages/
import("finance")                    // finance::sharpe(), finance::sortino()
import("finance", as=fin)            // fin::sharpe()

// Install script or native plugin from GitHub (-y to bypass overwrite prompt)
// $ hay install user/repo [-y]

// Uninstall a package (successfully deletes native plugin files, dirs, and metadata)
// $ hay remove user/repo

// List installed packages
// $ hay list

// Check integrity/version of installed packages with remote GitHub repository
// $ hay check-plugin [user/repo]

// Update one or all packages to their latest versions (-y to bypass prompt)
// $ hay update [user/repo] [-y]

// Plugin search paths
plugin_path("/shared/plugins", "/team/lib")
```

Packages, imports, and auto-loaded plugins execute Hayashi/native code in your session. Install and import only code you trust; see the [Trust Model](docs/src/trust-model.md).

Native plugins (`.so`/`.dll`/`.dylib` / `.wasm`) are fully supported, enabling third parties to ship optimized estimators, spatial packages, and data connectors via Hayashi's namespace system (using the `hayashi-plugin-sdk`). Closed-source proprietary plugins are legally permitted through Hayashi's GPL-3.0 Linking Exception.

## Build & test

```bash
cargo build --release      # optimized binary -> target/release/hay
cargo test                 # 428 tests, <1s
```

60 example scripts in `examples/`, all passing.

## Error messages

Hayashi provides rich error diagnostics:

```
error: line 3: undefined variable 'preco_total'
  3 │ display preco_total
    │ ^^^^^^^^^^^^^^^^^^^

error: line 6: undefined function 'sumarize' — did you mean 'summarize'?
  6 │ sumarize(df)
    │ ^^^^^^^^^^^^

error: line 2: undefined variable 'factor'
Stack trace:
  in calculate() at line 2
  in process() at line 5

error: line 1: expected DataFrame, got Int
  1 │ summarize(42)
    │ ^^^^^^^^^^^^^
```

## Author

Flávio de Vasconcellos Corrêa — [@sheep-farm](https://github.com/sheep-farm) — flavio.vcorrea@ufpel.edu.br

## License

GPL-3.0 with **Plugin Exception** — see [LICENSE](LICENSE).

This exception explicitly allows linking and loading proprietary/closed-source plugins developed using `hayashi-plugin-sdk` into Hayashi without triggering copyleft requirements.

