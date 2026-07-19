# Hayashi Benchmarks

Honest Hayashi/Greeners benchmarks against R and Python for common
econometric estimators.

## Goal

Measure execution time reproducibly, showing both Hayashi wins and losses.
No cherry-picking.

## Covered estimators

- `ols` — Ordinary Least Squares
- `logit` — Binary logit
- `probit` — Binary probit
- `iv` — IV / 2SLS
- `qreg` — Quantile regression (median, `tau=0.5`, no bootstrap)
- `arima` — AR(1) via `arima(df, y, p=1, d=0, q=0)`
- `garch` — GARCH(1,1)
- `var` — Vector Autoregression, 2-equation VAR(1)
- `panel` — Fixed-effects panel (`plm` / `linearmodels`)

## Competitors

- **R:** `lm`, `glm` (logit/probit), `arima`, `rugarch`, `plm`, `ivreg`, `quantreg`, `vars`
- **Python:** `statsmodels` (OLS, logit/probit, ARIMA, VAR, quantile), `linearmodels` (IV, panel), `arch`
- **Hayashi:** `ols`, `logit`, `probit`, `iv`, `qreg`, `arima`, `garch`, `var`, `fe`

## Methodology

1. Each estimator runs on synthetic datasets of increasing size.
2. Each implementation runs `warmup` untimed iterations to warm caches,
   then `iters` timed iterations inside a single process.
3. The timed run is repeated `runs` times; each timed iteration emits a
   `  elapsed: X.XXXXs` line.
4. The runner parses all `elapsed:` lines and reports mean, std, min and max
   **per estimator call**.
5. Peak RSS is sampled via `/proc/<pid>/status` during each run and averaged.
6. Datasets and scripts are versioned; raw results are kept in `results/`
   and are not committed.

## Usage

Run everything (estimators + DataFrame operations + Greeners Rust
microbenchmarks) with sensible defaults:

```bash
cd benchmarks
./run.sh
```

Options:

```bash
./run.sh --estimators          # only cross-language estimator benchmarks
./run.sh --ops                 # only DataFrame/language operation benchmarks
./run.sh --rust                # only Greeners Criterion microbenchmarks
./run.sh --rust --full         # full Criterion statistics (slower)
./run.sh --estimators --ops    # estimators + ops, skipping Rust
```

Environment variables for tuning:

```bash
SIZES=1000,10000,100000 ITERS=30 RUNS=5 WARMUP=3 ./run.sh
```

Or call the individual runners directly:

```bash
python scripts/run.py --estimator ols --sizes 1000,10000,100000 \
    --iters 30 --runs 5 --warmup 3

python scripts/benchmark_ops.py --op load_csv,filter,sort --sizes 10000,100000
```

- `--iters`: timed iterations per subprocess run.
- `--runs`: how many times the timed subprocess is launched.
- `--warmup`: untimed iterations before the timed loop.

## DataFrame / language operation benchmarks

`scripts/benchmark_ops.py` benchmarks individual Hayashi operations, with
optional pandas comparison for the same task:

```bash
python scripts/benchmark_ops.py
python scripts/benchmark_ops.py --op load_csv,filter,sort --sizes 10000,100000
python scripts/benchmark_ops.py --op load_csv --sizes 100000,1000000
```

Covered operations:

- `load_csv` — CSV parse/load (uses wall-clock time per run)
- `generate_random` — add a random-normal column
- `generate_expr` — add a derived column `(x + y) * 2`
- `filter` — row filter `x > 0`
- `sort` — sort by a single column
- `groupby_mean` — group and compute mean
- `merge` — inner join on `id`
- `loop` — integer `for` loop overhead
- `parallel_loop` — `parallel for` loading CSVs and `rbind` with ordered output (vs Python multiprocessing.Pool + pandas.concat)
- `function_call` — user function call overhead

## Honest interpretation / caveats

- Hayashi may lose on small datasets because of binary load / script parsing time.
- Hayashi tends to win on large datasets and repeated loops thanks to Rust/LLVM.
- The reported time is **per estimator call**; startup is amortized by the
  inner iteration count.
- Competitors compute more by default (covariance matrix, tests, influence).
  This benchmark measures the default command time, not a minimally
  equivalent implementation.
- `statsmodels` in particular does a lot of extra work in the default
  `fit()`, so it may look slower than it really is for an equivalent task.
- `qreg` is benchmarked with `boot=0` in Hayashi to measure only the
  IRLS fit; R/Python compute their standard errors by default.
- `var` uses a 2-equation VAR(1) with intercept on stationary synthetic
  data; the absolute time is small, so focus on relative speedup.
- R and Python have mature ecosystems; this benchmark measures raw
  estimation speed, not overall productivity.

## Generate summary table and plots

After running benchmarks:

```bash
python scripts/summarize.py
```

Generates:

- `results/summary.md` — Markdown table with speedups
- `results/summary.png` — log-log plot per estimator

Requires `matplotlib` installed (optional).

## Results

- Hayashi raw results are written to `results/<estimator>_YYYYMMDD_HHMMSS.json`
  and `results/ops_YYYYMMDD_HHMMSS.json`.
- Greeners Criterion reports (HTML + JSON) are written to
  `target/criterion/` inside `../Greeners`.
- All result files are git-ignored and should be regenerated locally.
