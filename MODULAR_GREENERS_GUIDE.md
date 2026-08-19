# Modular Greeners guide

This branch makes Greeners sub-crates optional in Hayashi.

## Local greeners facade

`crates/greeners/` re-exports Greeners sub-crates behind Cargo features:
- `core` (always enabled)
- `ols`, `glm`, `panel`, `timeseries`, `bayesian`, `causal`, `ml`, `spatial`, `survival`, `imputation`, `diagnostics`

## Hayashi features

Forwarded from `Cargo.toml`:
- `greeners-ols` -> `greeners/ols`
- `greeners-glm` -> `greeners/glm`
- `greeners-panel` -> `greeners/panel`
- `greeners-timeseries` -> `greeners/timeseries`
- `greeners-bayesian` -> `greeners/bayesian`
- `greeners-causal` -> `greeners/causal`
- `greeners-ml` -> `greeners/ml`
- `greeners-spatial` -> `greeners/spatial`
- `greeners-survival` -> `greeners/survival`
- `greeners-imputation` -> `greeners/imputation`
- `greeners-diagnostics` -> `greeners/diagnostics`
- `full` = all of the above
- `minimal` = `ols`

## Module -> feature map

Generated from `Greeners/crates/*/src/lib.rs`:

| module | feature |
|--------|---------|
| arima | timeseries |
| autoreg | timeseries |
| bart | ml |
| bayesian_linear | bayesian |
| bayesian_sc | bayesian |
| bayesian_sfa | bayesian |
| beta_model | glm |
| binary_diagnostics | diagnostics |
| biplot | core |
| bootstrap | core |
| bspline | core |
| bvar | bayesian |
| causal_forest | causal |
| causal_impact | causal |
| column | core |
| conditional | glm |
| conformal | causal |
| copula | core |
| cuped | causal |
| dataframe | core |
| datasets | core |
| dbscan | ml |
| dcc_garch | timeseries |
| decomposition | timeseries |
| descrstatsw | core |
| dfm | timeseries |
| diagnostics | diagnostics |
| did | causal |
| discrete | glm |
| distributions | core |
| dml_crossfit | causal |
| double_ml | causal |
| dr_learner | causal |
| dynamic_factor | timeseries |
| dynamic_panel | panel |
| error | core |
| ets | timeseries |
| event_study | ols |
| fa_panel | panel |
| fama_macbeth | diagnostics |
| favar | bayesian |
| fmols | ols |
| formula | core |
| functional_coef | core |
| garch | timeseries |
| gee | glm |
| glm | glm |
| glmgam | glm |
| gls | ols |
| glsar | ols |
| gmm | ols |
| gmm_clustering | core |
| gp | ml |
| gradient_boosting | ml |
| grf | ml |
| hausman | panel |
| hawkes | timeseries |
| heckman | ols |
| hierarchical | ml |
| imputation | imputation |
| influence | diagnostics |
| isotonic | core |
| iv | ols |
| johansen_break | timeseries |
| kmeans | ml |
| linalg | core |
| lp_did | causal |
| lstm | timeseries |
| margins | core |
| markov | timeseries |
| markov_autoreg | timeseries |
| mfvar | bayesian |
| mice | imputation |
| midas | timeseries |
| mixed | bayesian |
| mlp | ml |
| mnlogit | glm |
| model_selection | diagnostics |
| moment_helpers | core |
| ms_var | timeseries |
| mstl | timeseries |
| multipletests | core |
| multivariate | core |
| nardl | timeseries |
| negbin | glm |
| nls | ols |
| nonparametric | core |
| ols | ols |
| ordered | glm |
| orthogonal_forest | ml |
| panel | panel |
| panel_heckman | panel |
| panel_quantile | panel |
| panel_robust | panel |
| panel_tobit | panel |
| panel_var | panel |
| poisson | glm |
| predicate | core |
| proportion | core |
| psm | causal |
| pstr | panel |
| qrf | ml |
| qrf_inference | ml |
| quantile | ols |
| quantile_var | timeseries |
| random_forest | ml |
| rd | causal |
| reg_path | ols |
| rlm | ols |
| rolling | ols |
| setar | timeseries |
| spatial | spatial |
| spatial_durbin | spatial |
| spatial_durbin_error | spatial |
| spatial_panel | spatial |
| specification_tests | diagnostics |
| spectral | timeseries |
| statespace | timeseries |
| stats | core |
| stochastic_frontier | timeseries |
| summary_col | core |
| sur | ols |
| survival | survival |
| sv | timeseries |
| svar | timeseries |
| synth | causal |
| synth_did | causal |
| three_sls | ols |
| threshold | panel |
| timeseries | timeseries |
| tmle | causal |
| tobit | ols |
| transformer | ml |
| transforms | core |
| tsne | ml |
| tv_copula | timeseries |
| tvar | timeseries |
| tvp | timeseries |
| tvp_var | timeseries |
| types | core |
| umap | ml |
| unobserved_components | timeseries |
| var | timeseries |
| varma | timeseries |
| vecm | timeseries |
| wavelet | timeseries |
| wls | ols |
| xgboost | ml |
| zero_inflated | glm |
