# Hayashi Benchmarks

Honest Hayashi/Greeners benchmarks against R and Python for common
econometric estimators.

## Goal

Measure execution time reproducibly, showing both Hayashi wins and losses.
No cherry-picking.

## Covered estimators

- `ols` — Ordinary Least Squares
- `logit` — Binary logit
- `arima` — AR(1) via `arima(df, y, p=1, d=0, q=0)`
- `garch` — GARCH(1,1)
- `panel` — Fixed-effects panel (`plm` / `linearmodels`)

## Competitors

- **R:** `lm`, `glm`, `arima`, `rugarch`, `plm`
- **Python:** `statsmodels`, `linearmodels`, `arch`
- **Hayashi:** `ols`, `logit`, `arima`, `garch`, `fe`

## Methodology

1. Each estimator runs on synthetic datasets of increasing size.
2. Each combination (estimator × language × size) runs several times.
3. The first run is discarded as warmup when applicable.
4. Mean and standard deviation of wall-clock time are reported.
5. Datasets and scripts are versioned; raw results are kept in `results/`
   and are not committed.

## Usage

```bash
cd benchmarks
./run.sh
```

Or, with fine control:

```bash
python scripts/run.py --estimator ols --sizes 1000,10000,100000 --reps 10
```

## Honest interpretation / caveats

- Hayashi may lose on small datasets because of compilation / binary load time.
- Hayashi tends to win on large datasets and repeated loops thanks to
  Rust/LLVM.
- Competitors compute more by default (covariance matrix, tests, influence).
  This benchmark measures the default command time, not a minimally
  equivalent implementation.
- `statsmodels` in particular does a lot of extra work in the default
  `fit()`, so it may look slower than it really is for an equivalent task.
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

Raw results are written to `results/<estimator>_YYYYMMDD_HHMMSS.json`.
They are git-ignored and should be regenerated locally.
