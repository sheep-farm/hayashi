# Known validation and documentation gaps for the 0.2.10 release

This document tracks estimator families that are implemented and exposed to users but are not yet covered by the empirical validation programme in `validation/`, or where the existing coverage is thin enough that release notes should be careful.

## Status of the validation programme

- As of this update, `validation/matrix.yml` contains 230 cases.
- 217 cases currently `pass`; 15 remain `not-supported` because a stable
  comparable reference is unavailable.
- The cases compare Hayashi output against R and/or Python reference implementations (statsmodels, linearmodels, wooldridge, etc.) with family-specific tolerances.

## Implemented estimators not yet in the validation matrix

These commands exist in the interpreter dispatch and are listed in user-facing docs, but have no corresponding case under `validation/cases/` (no numerical validation against a reference implementation):

- `bplm` — (`src/lang/interpreter/estimators_panel.rs`)
- `cffilter` / `cf_filter` / `christiano_fitzgerald` — Christiano-Fitzgerald band-pass filter (`src/lang/interpreter/estimators_misc.rs`)
- `chamberlain` — Chamberlain panel estimator (`src/lang/interpreter/estimators_panel.rs`)
- `cmnlogit` / `cmlogit` / `conditional_mlogit` — conditional multinomial logit (`src/lang/interpreter/estimators_panel.rs`)
- `gam` / `gamfit` — generalized additive model (`src/lang/interpreter/estimators_timeseries.rs`)
- `markov` / `msar` / `markovswitching` — Markov-switching model (`src/lang/interpreter/estimators_panel.rs`)
- `mice` / `mi` / `multiple_imputation` — MICE imputation, simple/one-shot variant (`src/lang/interpreter/estimators_timeseries.rs`; the `mice_chained` variant is validated)
- `mstl` / `stl` — seasonal-trend decomposition (`src/lang/interpreter/estimators_timeseries.rs`)
- `msauto` / `markov_ar` / `ms_ar` / `hamilton` — Markov-switching autoregression (`src/lang/interpreter/estimators_timeseries.rs`)
- `pthresh` / `xtthresh` / `panel_threshold` / `threshold` — panel threshold regression (`src/lang/interpreter/estimators_misc.rs`)

The following estimators were previously in this list but are now covered by validation cases (`be_simulated`, `feiv_simulated`, `clogit_simulated`, `cpoisson_simulated`, `varma_simulated`, `threesl_simulated`, `svec_simulated`, `decompose_simulated`, `ucm_simulated`, `cancorr_simulated`):

- `be` — between estimator (`src/lang/interpreter/estimators_panel.rs`)
- `cancorr` — canonical correlation (`src/lang/interpreter/estimators_misc.rs`)
- `clogit` — conditional logit (`src/lang/interpreter/estimators_panel.rs`)
- `cpoisson` — conditional Poisson / PPML (`src/lang/interpreter/estimators_panel.rs`)
- `decompose` — seasonal decomposition (`src/lang/interpreter/estimators_timeseries.rs`)
- `feiv` — fixed-effects IV (`src/lang/interpreter/estimators_panel.rs`)
- `ucm` / `uc` / `structural_ts` — unobserved components model (`src/lang/interpreter/estimators_timeseries.rs`)
- `varma` / `varmax` — vector ARMA (`src/lang/interpreter/estimators_timeseries.rs`)
- `three_sls` / `threesl` / `3sls` / `reg3` — three-stage least squares (`src/lang/interpreter/estimators_timeseries.rs`)

## Implemented estimators with not-supported validation placeholders

These estimators have a directory under `validation/cases/`, but the case is marked `not-supported` because no stable R/Python reference is currently available (the `run.hay` is a placeholder). They are therefore implemented and callable, but not numerically validated:

- `bsc` / `bayesian_sc` — Bayesian synthetic control
- `bvar` / `bayesian_var` — Bayesian VAR
- `bayes_sfa_production` / `bayes_sfa_cost` / `bayes_frontier` — Bayesian stochastic frontier
- `fapanel` / `fa_panel` — factor-analysis panel
- `fcoef` / `functional_coef` — functional coefficients
- `fmols` — fully modified OLS
- `johansen_break` — Johansen cointegration with structural break
- `lstm`
- `mfvar` / `mixed_freq_var` — mixed-frequency VAR
- `spatial_durbin_error` / `sdem`
- `spatial_panel_sar` / `spatial_panel_sem` — `spatial_panel_sar` has a `not-supported` placeholder; `spatial_panel_sem` has no case yet
- `spectral` / `spectral_clustering`
- `transformer` / `transformer_ts`
- `tvcopula` / `tv_copula` — time-varying copula
- `tvp_var` — time-varying parameter VAR

## Validated but with thin or convention-sensitive coverage

These families appear in the matrix, but the coverage should be described carefully in release notes:

- `kalman` — one R-only local-level case (`kalman_nyse`).
- `svar` — one Cholesky-identified SVAR case. Broader SVAR/SVEC identification and export/parse support may still need work.
- `ab` (Arellano-Bond), `sysgmm` (Blundell-Bond), `pcse`, `xtgls` — dynamic-panel/GMM and panel-GLS cases are present, but the references implement the same two-step procedures used by Hayashi/Greeners; other packages (e.g. `plm::pgmm`) use different instrument and weighting conventions, so results may not be portable across implementations.
- `km` (Kaplan-Meier), `cox`, `psm`, `synth`, `did`, `rd`, `fuzzy_rd`, `qreg`, `rlm`, `gee`, `gmm`, `iv` — have at least one case, but often rely on simulated or selected real datasets; broader coverage is desirable.