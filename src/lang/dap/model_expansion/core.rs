use super::*;
use crate::lang::interpreter::{DiagResult, Series, Value};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;

pub struct RegressionCtx<'a> {
    pub names: Vec<String>,
    pub params: &'a Array1<f64>,
    pub std_errors: &'a Array1<f64>,
    pub test_values: &'a Array1<f64>,
    pub p_values: &'a Array1<f64>,
    pub conf_lower: Option<&'a Array1<f64>>,
    pub conf_upper: Option<&'a Array1<f64>>,
    pub fit: Value,
    pub residuals: Option<&'a Array1<f64>>,
    pub fitted_values: Option<&'a Array1<f64>>,
    pub x: Option<&'a Array2<f64>>,
}

/// Returns (summary string, type name, expected child count) for a value.
pub fn value_summary(v: &Value) -> (String, String, usize) {
    let (summary, type_name) = value_summary_and_type(v);
    (summary, type_name.to_string(), value_children(v).len())
}

/// Returns the named children of a value for DAP variable expansion.
pub fn value_children(v: &Value) -> Vec<(String, Value)> {
    match v {
        Value::DataFrame(df) => df
            .column_names()
            .iter()
            .filter_map(|name| {
                df.get_column(name)
                    .ok()
                    .map(|col| (name.clone(), column_to_value(name, col)))
            })
            .collect(),
        Value::List(lst) => lst
            .iter()
            .take(100)
            .enumerate()
            .map(|(i, item): (usize, &Value)| (format!("[{i}]"), item.clone()))
            .collect(),
        Value::Dict(d) => d
            .iter()
            .take(100)
            .map(|(k, item): (&String, &Value)| (k.clone(), item.clone()))
            .collect(),
        Value::Series(s) => s
            .values
            .iter()
            .take(100)
            .enumerate()
            .map(|(i, item): (usize, &Value)| (format!("[{i}]"), item.clone()))
            .collect(),
        Value::OlsResult(m) => regression_children(RegressionCtx {
            names: m.result.variable_names.clone().unwrap_or_default(),
            params: &m.result.params,
            std_errors: &m.result.std_errors,
            test_values: &m.result.t_values,
            p_values: &m.result.p_values,
            conf_lower: Some(&m.result.conf_lower),
            conf_upper: Some(&m.result.conf_upper),
            fit: ols_fit_dict(&m.result),
            residuals: Some(&m.residuals),
            fitted_values: None,
            x: Some(&m.x),
        }),
        Value::IvResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: iv_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::PanelResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: panel_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::ReResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: re_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::BinaryResult(m) => regression_children(RegressionCtx {
            names: m.coef_names.clone(),
            params: &m.result.params,
            std_errors: &m.result.std_errors,
            test_values: &m.result.z_values,
            p_values: &m.result.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: binary_fit_dict(&m.result),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::QuantileResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: quantile_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::TobitResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: tobit_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::PoissonResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.z_values,
            p_values: &r.p_values,
            conf_lower: Some(&r.conf_lower),
            conf_upper: Some(&r.conf_upper),
            fit: poisson_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::NegBinResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.z_values,
            p_values: &r.p_values,
            conf_lower: Some(&r.conf_lower),
            conf_upper: Some(&r.conf_upper),
            fit: negbin_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::GlmResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.z_values,
            p_values: &r.p_values,
            conf_lower: Some(&r.conf_lower),
            conf_upper: Some(&r.conf_upper),
            fit: glm_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::RlmResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: Some(&r.conf_lower),
            conf_upper: Some(&r.conf_upper),
            fit: rlm_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::BetaResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.z_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: beta_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::GmmResult(r) => regression_children(RegressionCtx {
            names: (0..r.params.len()).map(|i| format!("x{i}")).collect(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: gmm_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::AbResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: ab_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::SysGmmResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: sysgmm_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::PcseResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: pcse_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::PanelGlsResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: panel_gls_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::FE2SLSResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: fe2sls_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::OrderedResult(r) => regression_children(RegressionCtx {
            names: r.variable_names.clone().unwrap_or_default(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.z_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: ordered_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::ZeroInflatedResult(r) => zero_inflated_children(r),
        Value::MixedResult(r) => mixed_children(r),
        Value::GlsarResult(r) => regression_children(RegressionCtx {
            names: (0..r.params.len()).map(|i| format!("x{i}")).collect(),
            params: &r.params,
            std_errors: &r.std_errors,
            test_values: &r.t_values,
            p_values: &r.p_values,
            conf_lower: None,
            conf_upper: None,
            fit: glsar_fit_dict(r),
            residuals: None,
            fitted_values: None,
            x: None,
        }),
        Value::SurResult(m) => sur_children(m),
        Value::ThreeSLSResult(m) => three_sls_children(m),
        Value::MNLogitResult(r) => mnlogit_children(r),
        Value::ArimaResult(r) => arima_children(r),
        Value::GarchResult(r) => garch_children(r),
        Value::EtsResult(r) => ets_children(r),
        Value::MstlResult(r) => mstl_children(r),
        Value::UCResult(r) => uc_children(r),
        Value::LocalLevelResult(r) => local_level_children(r),
        Value::AutoRegResult(r) => autoreg_children(r),
        Value::ArdlResult(r) => ardl_children(r),
        Value::ThresholdResult(r) => threshold_children(r),
        Value::VarResult(r) => var_children(r),
        Value::VecmResult(r) => vecm_children(r),
        Value::VarmaResult(r) => varma_children(r),
        Value::SVarResult(r) => svar_children(r),
        Value::MSARResult(r) => msar_children(r),
        Value::DFMResult(r) => dfm_children(r),
        Value::MarkovResult(r) => markov_children(r),
        Value::RdResult(r) => rd_children(r),
        Value::SynthResult(r) => synth_children(r),
        Value::PsmResult(r) => psm_children(r),
        Value::DidResult(r) => did_children(r),
        Value::KMResult(r) => km_children(r),
        Value::CoxResult(r) => cox_children(r),
        Value::HeckmanResult(r) => heckman_children(r),
        Value::GeeResult(r) => gee_children(r),
        Value::LowessResult(r) => lowess_children(r),
        Value::PenalizedResult(m) => penalized_children(m),
        Value::PcaResult(m) => pca_children(m),
        Value::FactorResult(m) => factor_children(m),
        Value::MiceResult(r) => mice_children(r),
        Value::GamResult(r) => gam_children(r),
        Value::ConditionalResult(r) => conditional_children(r),
        Value::RollingResult(r) => rolling_children(r),
        Value::RecursiveLSResult(r) => recursive_ls_children(r),
        Value::DecompResult(r) => decomp_children(r),
        Value::DiagResult(r) => diag_children(r),
        Value::KmeansResult(r) => kmeans_children(r),
        Value::DbscanResult(r) => dbscan_children(r),
        Value::IsotonicResult(r) => isotonic_children(r),
        Value::KdeResult(r) => kde_children(r),
        Value::BartResult(r) => bart_children(r),
        Value::GpResult(r) => gp_children(r),
        Value::GmmClusteringResult(r) => gmm_clustering_children(r),
        Value::HierarchicalResult(r) => hierarchical_children(r),
        Value::SpectralResult(r) => spectral_children(r),
        Value::ModelResult { fields, .. } => {
            fields.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        }
        _ => Vec::new(),
    }
}

/// Looks up a named child/field on any value. Returns `None` if the field does
/// not exist. This makes DAP-style expansion available for `.field` / `["field"]`
/// access on model results, not only on dicts/dataframes. If the field is not a
/// top-level child, it is also searched inside any nested `fit` dict, so scalar
/// summaries like `m.r2` or `m.inertia` work without duplicating every scalar
/// at the top level.
pub fn value_field(v: &Value, field: &str) -> Option<Value> {
    let direct = match v {
        Value::Dict(d) => d.get(field).cloned(),
        Value::ModelResult { fields, .. } => fields.get(field).cloned(),
        Value::DataFrame(df) => df
            .get_column(field)
            .ok()
            .map(|col| column_to_value(field, col)),
        Value::List(_) | Value::Series(_) => None,
        _ => value_children(v)
            .into_iter()
            .find(|(name, _)| name == field)
            .map(|(_, child)| child),
    };
    if direct.is_some() {
        return direct;
    }

    // Fall back to looking inside the model's `fit` dict, if present.
    if let Value::Dict(d) = value_children(v)
        .into_iter()
        .find(|(name, _)| name == "fit")
        .map(|(_, child)| child)
        .unwrap_or(Value::Nil)
    {
        return d.get(field).cloned();
    }

    None
}

pub fn value_summary_and_type(v: &Value) -> (String, &'static str) {
    match v {
        Value::Float(f) => (format!("{f}"), "Float"),
        Value::Int(i) => (format!("{i}"), "Int"),
        Value::Bool(b) => (format!("{b}"), "Bool"),
        Value::Str(s) => (s.clone(), "String"),
        Value::Nil => ("nil".into(), "Nil"),
        Value::DataFrame(df) => (
            format!(
                "DataFrame({} rows, {} cols)",
                df.n_rows(),
                df.column_names().len()
            ),
            "DataFrame",
        ),
        Value::List(lst) => (format!("List({} items)", lst.len()), "List"),
        Value::Dict(d) => (format!("Dict({} entries)", d.len()), "Dict"),
        Value::Series(s) => (format!("Series({}: {} values)", s.name, s.len()), "Series"),
        Value::OlsResult(m) => {
            let r = &m.result;
            (
                format!(
                    "OLS(k={}, n={}), R2={:.4}",
                    r.params.len(),
                    r.n_obs,
                    r.r_squared
                ),
                "OlsResult",
            )
        }
        Value::IvResult(r) => (
            format!(
                "IV(k={}, n={}), R2={:.4}",
                r.params.len(),
                r.n_obs,
                r.r_squared
            ),
            "IvResult",
        ),
        Value::PanelResult(r) => (
            format!(
                "Panel(k={}, n={}, N={}), R2={:.4}",
                r.params.len(),
                r.n_obs,
                r.n_entities,
                r.r_squared
            ),
            "PanelResult",
        ),
        Value::ReResult(r) => (
            format!(
                "RE(k={}, N={}), R2={:.4}",
                r.params.len(),
                r.variable_names
                    .as_ref()
                    .map(|v: &Vec<String>| v.len())
                    .unwrap_or(r.params.len()),
                r.r_squared_overall
            ),
            "ReResult",
        ),
        Value::BinaryResult(m) => (
            format!(
                "{}(k={}), pseudoR2={:.4}",
                m.result.model_name,
                m.result.params.len(),
                m.result.pseudo_r2
            ),
            "BinaryResult",
        ),
        Value::QuantileResult(r) => (
            format!(
                "Quantile(tau={:.2}, k={}), R2={:.4}",
                r.tau,
                r.params.len(),
                r.r_squared
            ),
            "QuantileResult",
        ),
        Value::TobitResult(r) => (
            format!(
                "Tobit(k={}, n={}), sigma={:.4}",
                r.params.len(),
                r.n_obs,
                r.sigma
            ),
            "TobitResult",
        ),
        Value::PoissonResult(r) => (
            format!(
                "Poisson(k={}, n={}), pseudoR2={:.4}",
                r.params.len(),
                r.n_obs,
                r.pseudo_r2
            ),
            "PoissonResult",
        ),
        Value::NegBinResult(r) => (
            format!(
                "NegBin(k={}, n={}), alpha={:.4}",
                r.params.len(),
                r.n_obs,
                r.alpha
            ),
            "NegBinResult",
        ),
        Value::GlmResult(r) => (
            format!(
                "GLM(k={}, n={}), pseudoR2={:.4}",
                r.params.len(),
                r.n_obs,
                r.pseudo_r2
            ),
            "GlmResult",
        ),
        Value::RlmResult(r) => (
            format!(
                "RLM(k={}, n={}), scale={:.4}",
                r.params.len(),
                r.n_obs,
                r.scale
            ),
            "RlmResult",
        ),
        Value::BetaResult(r) => (
            format!(
                "Beta(k={}, n={}), phi={:.4}",
                r.params.len(),
                r.n_obs,
                r.precision_param
            ),
            "BetaResult",
        ),
        Value::GmmResult(r) => (
            format!(
                "GMM(k={}, n={}), J={:.4}",
                r.params.len(),
                r.n_obs,
                r.j_stat
            ),
            "GmmResult",
        ),
        Value::AbResult(r) => (
            format!(
                "ArellanoBond(k={}, n={}), m1_p={:.4}",
                r.params.len(),
                r.n_obs,
                r.m1_pval
            ),
            "AbResult",
        ),
        Value::SysGmmResult(r) => (
            format!(
                "SysGMM(k={}, n={}), Sargan={:.4}",
                r.params.len(),
                r.n_obs_fd,
                r.sargan_stat
            ),
            "SysGmmResult",
        ),
        Value::PcseResult(r) => (
            format!(
                "PCSE(k={}, n={}), R2={:.4}",
                r.params.len(),
                r.n_obs,
                r.r_squared
            ),
            "PcseResult",
        ),
        Value::PanelGlsResult(r) => (
            format!(
                "PanelGLS(k={}, n={}), R2={:.4}",
                r.params.len(),
                r.n_obs,
                r.r_squared
            ),
            "PanelGlsResult",
        ),
        Value::FE2SLSResult(r) => (
            format!(
                "FE2SLS(k={}, n={}), R2={:.4}",
                r.params.len(),
                r.n_obs,
                r.r_squared
            ),
            "FE2SLSResult",
        ),
        Value::OrderedResult(r) => (
            format!(
                "Ordered(k={}, n={}), pseudoR2={:.4}",
                r.params.len(),
                r.n_obs,
                r.pseudo_r2
            ),
            "OrderedResult",
        ),
        Value::ZeroInflatedResult(r) => (
            format!(
                "ZeroInflated(count={}, inflate={}, n={}), logLik={:.4}",
                r.count_params.len(),
                r.inflate_params.len(),
                r.n_obs,
                r.log_likelihood
            ),
            "ZeroInflatedResult",
        ),
        Value::MixedResult(r) => (
            format!(
                "Mixed(fixed={}, n={}, groups={}), logLik={:.4}",
                r.fixed_effects.len(),
                r.n_obs,
                r.n_groups,
                r.log_likelihood
            ),
            "MixedResult",
        ),
        Value::GlsarResult(r) => (
            format!(
                "GLSAR(k={}, n={}), R2={:.4}",
                r.params.len(),
                r.n_obs,
                r.r_squared
            ),
            "GlsarResult",
        ),
        Value::SurResult(m) => (
            format!(
                "SUR(eqs={}), sysR2={:.4}",
                m.result.equations.len(),
                m.result.system_r2
            ),
            "SurResult",
        ),
        Value::ThreeSLSResult(m) => (
            format!("3SLS(eqs={})", m.result.equations.len()),
            "ThreeSLSResult",
        ),
        Value::MNLogitResult(r) => (
            format!(
                "MNLogit(k={}, n={}), pseudoR2={:.4}",
                r.params.nrows(),
                r.n_obs,
                r.pseudo_r2
            ),
            "MNLogitResult",
        ),
        Value::ArimaResult(r) => (
            format!(
                "ARIMA(p={},d={},q={}), n={}, logLik={:.4}",
                r.order.p, r.order.d, r.order.q, r.n_obs, r.log_likelihood
            ),
            "ArimaResult",
        ),
        Value::GarchResult(r) => (
            format!(
                "GARCH(p={},q={}), n={}, logLik={:.4}",
                r.p, r.q, r.n_obs, r.log_likelihood
            ),
            "GarchResult",
        ),
        Value::EtsResult(r) => (
            format!(
                "ETS({},{}), n={}, sse={:.4}",
                r.trend_type, r.seasonal_type, r.n_obs, r.sse
            ),
            "EtsResult",
        ),
        Value::MstlResult(r) => (
            format!("MSTL(periods={:?}), n={}", r.periods, r.n_obs),
            "MstlResult",
        ),
        Value::UCResult(r) => (
            format!(
                "UC(level={:?}, seasonal={:?}), n={}, logLik={:.4}",
                r.level_type, r.seasonal_type, r.n_obs, r.log_likelihood
            ),
            "UCResult",
        ),
        Value::LocalLevelResult(r) => (
            format!(
                "LocalLevel(n={}, sigma_obs={:.4}, sigma_state={:.4})",
                r.n_obs, r.sigma_obs, r.sigma_state
            ),
            "LocalLevelResult",
        ),
        Value::AutoRegResult(r) => (
            format!(
                "AutoReg(lags={}), n={}, R2={:.4}",
                r.lags, r.n_obs, r.r_squared
            ),
            "AutoRegResult",
        ),
        Value::ArdlResult(r) => (
            format!(
                "ARDL(y_lags={}, x_lags={}), n={}, R2={:.4}",
                r.y_lags, r.x_lags, r.n_obs, r.r_squared
            ),
            "ArdlResult",
        ),
        Value::ThresholdResult(r) => (
            format!(
                "PanelThreshold(gamma={:.4}), R2={:.4}",
                r.threshold_gamma, r.r_squared
            ),
            "ThresholdResult",
        ),
        Value::VarResult(r) => (
            format!(
                "VAR(lags={}, k={}), n={}, AIC={:.4}",
                r.lags, r.n_vars, r.n_obs, r.aic
            ),
            "VarResult",
        ),
        Value::VecmResult(r) => (
            format!("VECM(rank={}, lags={}), n={}", r.rank, r.lags, r.n_obs),
            "VecmResult",
        ),
        Value::VarmaResult(r) => (
            format!(
                "VARMA({},{}), k={}, n={}",
                r.p_lags, r.q_lags, r.n_vars, r.n_obs
            ),
            "VarmaResult",
        ),
        Value::SVarResult(r) => (
            format!(
                "SVAR(k={}, lags={}), id={}",
                r.var_result.n_vars, r.var_result.lags, r.identification
            ),
            "SVarResult",
        ),
        Value::MSARResult(r) => (
            format!(
                "MSAR(ar={}, regimes={}), n={}",
                r.ar_order, r.k_regimes, r.n_obs
            ),
            "MSARResult",
        ),
        Value::DFMResult(m) => (
            format!(
                "DFM(factors={}, series={}), n={}",
                m.result.n_factors, m.result.n_vars, m.result.n_obs
            ),
            "DFMResult",
        ),
        Value::MarkovResult(r) => (
            format!(
                "MarkovSwitching(ar={}, regimes={}), n={}",
                r.ar_order, r.n_regimes, r.n_obs
            ),
            "MarkovResult",
        ),
        Value::RdResult(r) => (
            format!(
                "RD(tau={:.4}, bw={:.4}), n={}",
                r.tau, r.bandwidth, r.n_total
            ),
            "RdResult",
        ),
        Value::SynthResult(r) => (
            format!(
                "Synth(unit={}, donors={}), T_pre={}, T_post={}",
                r.treated_unit, r.n_donors, r.t_pre, r.t_post
            ),
            "SynthResult",
        ),
        Value::PsmResult(r) => (
            format!(
                "PSM(ATT={:.4}), n_treat={}, n_matched={}",
                r.att, r.n_treated, r.n_matched_treated
            ),
            "PsmResult",
        ),
        Value::DidResult(r) => (
            format!(
                "DiD(ATT={:.4}), n={}, R2={:.4}",
                r.att, r.n_obs, r.r_squared
            ),
            "DidResult",
        ),
        Value::KMResult(r) => (
            format!(
                "KM(n={}, events={}), median={:.4}",
                r.n_obs, r.n_events, r.median_survival
            ),
            "KMResult",
        ),
        Value::CoxResult(r) => (
            format!(
                "Cox(k={}, events={}), C={:.4}",
                r.params.len(),
                r.n_events,
                r.concordance
            ),
            "CoxResult",
        ),
        Value::HeckmanResult(r) => (
            format!(
                "Heckman(n={}, selected={}), rho={:.4}",
                r.n_obs, r.n_selected, r.rho
            ),
            "HeckmanResult",
        ),
        Value::GeeResult(r) => (
            format!(
                "GEE(k={}, groups={}), QIC={:.4}",
                r.params.len(),
                r.n_groups,
                r.qic
            ),
            "GeeResult",
        ),
        Value::LowessResult(r) => (
            format!("Lowess(n={}, frac={:.4})", r.n_obs, r.frac),
            "LowessResult",
        ),
        Value::PenalizedResult(m) => (
            format!(
                "{}(n={}, alpha={:.4}), R2={:.4}",
                capitalize(&m.kind),
                m.n_obs,
                m.alpha,
                m.r_squared
            ),
            "PenalizedResult",
        ),
        Value::PcaResult(m) => (
            format!(
                "PCA(n={}, components={})",
                m.result.n_obs, m.result.n_components
            ),
            "PcaResult",
        ),
        Value::FactorResult(m) => (
            format!(
                "Factor(n={}, factors={})",
                m.result.n_obs, m.result.n_factors
            ),
            "FactorResult",
        ),
        Value::MiceResult(r) => (
            format!(
                "MICE(n={}, vars={}, m={})",
                r.n_obs, r.n_vars, r.n_imputations
            ),
            "MiceResult",
        ),
        Value::GamResult(r) => (
            format!(
                "GAM(n={}, linear={}, smooth={}), GCV={:.4}",
                r.n_obs, r.n_linear, r.n_smooth, r.gcv_score
            ),
            "GamResult",
        ),
        Value::ConditionalResult(r) => (
            format!("{}(n={}, groups={})", r.model_name, r.n_obs, r.n_groups),
            "ConditionalResult",
        ),
        Value::RollingResult(r) => (
            format!(
                "RollingOLS(n={}, window={}), k={}",
                r.n_obs,
                r.window,
                r.params_history.ncols()
            ),
            "RollingResult",
        ),
        Value::RecursiveLSResult(r) => (
            format!("RecursiveLS(n={}, k={})", r.n_obs, r.params_history.ncols()),
            "RecursiveLSResult",
        ),
        Value::DecompResult(r) => (
            format!("Decomp({}), n={}", r.model, r.observed.len()),
            "DecompResult",
        ),
        Value::DiagResult(r) => (
            format!("Diagnostic({} fields)", r.fields.len()),
            "DiagResult",
        ),
        Value::KmeansResult(r) => (
            format!(
                "KMeans(k={}, n={}), inertia={:.4}",
                r.n_clusters, r.n_obs, r.inertia
            ),
            "KmeansResult",
        ),
        Value::DbscanResult(r) => (
            format!(
                "DBSCAN(clusters={}, noise={}), n={}",
                r.n_clusters, r.n_noise, r.n_obs
            ),
            "DbscanResult",
        ),
        Value::IsotonicResult(r) => (
            format!(
                "Isotonic(n={}, increasing={}), R2={:.4}",
                r.n_obs, r.increasing, r.r_squared
            ),
            "IsotonicResult",
        ),
        Value::KdeResult(r) => (
            format!(
                "KDE(n={}, bandwidth={:.4}, points={})",
                r.n_obs,
                r.bandwidth,
                r.support.len()
            ),
            "KdeResult",
        ),
        Value::BartResult(r) => (
            format!(
                "BART(trees={}, depth={}), n={}, R2={:.4}",
                r.n_trees, r.max_depth, r.n_obs, r.r_squared
            ),
            "BartResult",
        ),
        Value::GpResult(r) => (
            format!(
                "GP(n={}, l={:.4}, sigma_f={:.4}), R2={:.4}",
                r.n_obs,
                r.length_scale,
                r.signal_variance.sqrt(),
                r.r_squared
            ),
            "GpResult",
        ),
        Value::GmmClusteringResult(r) => (
            format!(
                "GMM(k={}, n={}), loglik={:.4}, converged={}",
                r.n_clusters, r.n_obs, r.log_likelihood, r.converged
            ),
            "GmmClusteringResult",
        ),
        Value::HierarchicalResult(r) => (
            format!(
                "HClust(k={}, n={}), linkage={:?}",
                r.n_clusters, r.n_obs, r.linkage
            ),
            "HierarchicalResult",
        ),
        Value::SpectralResult(r) => (
            format!(
                "Spectral(k={}, n={}), inertia={:.4}",
                r.n_clusters, r.n_obs, r.inertia
            ),
            "SpectralResult",
        ),
        Value::ModelResult {
            summary, type_name, ..
        } => (summary.clone(), *type_name),
        Value::UserFn(f) => (format!("<fn({})>", f.params.join(", ")), "Function"),
        _ => (v.to_string(), "Model"),
    }
}

pub fn regression_children<'a>(ctx: RegressionCtx<'a>) -> Vec<(String, Value)> {
    let mut vars = Vec::new();

    let coef_df = coef_dataframe(
        &ctx.names,
        ctx.params,
        ctx.std_errors,
        ctx.test_values,
        ctx.p_values,
        ctx.conf_lower,
        ctx.conf_upper,
    );
    vars.push(("coefficients".into(), coef_df));
    vars.push(("fit".into(), ctx.fit));

    if let Some(resid) = ctx.residuals {
        if !resid.is_empty() {
            vars.push(("residuals".into(), array1_to_series("residuals", resid)));
        }
    }
    if let Some(fitted) = ctx.fitted_values {
        if !fitted.is_empty() {
            vars.push((
                "fitted_values".into(),
                array1_to_series("fitted_values", fitted),
            ));
        }
    } else if let Some(x) = ctx.x {
        if !x.is_empty() {
            let fitted = x.dot(ctx.params);
            vars.push((
                "fitted_values".into(),
                array1_to_series("fitted_values", &fitted),
            ));
        }
    }

    vars.push(("params".into(), array1_to_series("params", ctx.params)));
    vars.push((
        "std_errors".into(),
        array1_to_series("std_errors", ctx.std_errors),
    ));
    vars.push((
        "test_values".into(),
        array1_to_series("test_values", ctx.test_values),
    ));
    vars.push((
        "p_values".into(),
        array1_to_series("p_values", ctx.p_values),
    ));
    if let Some(cl) = ctx.conf_lower {
        vars.push(("conf_lower".into(), array1_to_series("conf_lower", cl)));
    }
    if let Some(cu) = ctx.conf_upper {
        vars.push(("conf_upper".into(), array1_to_series("conf_upper", cu)));
    }

    vars
}

pub fn coef_dataframe(
    names: &[String],
    params: &Array1<f64>,
    std_errors: &Array1<f64>,
    test_values: &Array1<f64>,
    p_values: &Array1<f64>,
    conf_lower: Option<&Array1<f64>>,
    conf_upper: Option<&Array1<f64>>,
) -> Value {
    let n = params.len();
    let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();

    let name_col: Vec<String> = (0..n)
        .map(|i| names.get(i).cloned().unwrap_or_else(|| format!("x{i}")))
        .collect();
    columns.insert("variable".into(), greeners::Column::from_strings(name_col));
    columns.insert("coef".into(), f64_array_column(params));
    columns.insert("std_err".into(), f64_array_column(std_errors));
    columns.insert("t".into(), f64_array_column(test_values));
    columns.insert("p_value".into(), f64_array_column(p_values));
    if let Some(cl) = conf_lower {
        columns.insert("conf_low".into(), f64_array_column(cl));
    }
    if let Some(cu) = conf_upper {
        columns.insert("conf_high".into(), f64_array_column(cu));
    }

    Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(columns)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ))
}

pub fn array1_to_series(name: &str, arr: &Array1<f64>) -> Value {
    let values: Vec<Value> = arr.iter().map(|&v| Value::Float(v)).collect();
    Value::Series(Arc::new(Series::new(name, values)))
}

pub fn series_from_vec(name: &str, v: &[f64]) -> Value {
    let values: Vec<Value> = v.iter().map(|&x| Value::Float(x)).collect();
    Value::Series(Arc::new(Series::new(name, values)))
}

pub fn array2_to_dataframe(_name: &str, arr: &Array2<f64>) -> Value {
    let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();
    for j in 0..arr.ncols() {
        let col: Vec<f64> = arr.column(j).iter().copied().collect();
        columns.insert(
            format!("col{j}"),
            greeners::Column::Float(Array1::from(col)),
        );
    }
    Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(columns)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ))
}

pub fn f64_array_column(arr: &Array1<f64>) -> greeners::Column {
    greeners::Column::Float(Array1::from(arr.iter().copied().collect::<Vec<_>>()))
}

pub fn fit_dict(entries: &[(&str, Value)]) -> Value {
    let mut map = HashMap::new();
    for (k, v) in entries {
        map.insert((*k).to_string(), v.clone());
    }
    Value::Dict(Arc::new(map))
}

pub fn array2_to_dataframe_named(arr: &Array2<f64>, col_names: &[String]) -> Value {
    let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();
    for j in 0..arr.ncols() {
        let name = col_names
            .get(j)
            .cloned()
            .unwrap_or_else(|| format!("col{j}"));
        let col: Vec<f64> = arr.column(j).iter().copied().collect();
        columns.insert(name, greeners::Column::Float(Array1::from(col)));
    }
    Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(columns)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ))
}

/// Generic first-class model result constructor.  Stores the formatted Greeners
/// output plus a named dict of children so DAP and `var.field` can inspect it
/// without requiring a dedicated `Value` variant for every estimator.
pub fn model_result(
    display: impl Into<String>,
    summary: impl Into<String>,
    type_name: &'static str,
    fields: Vec<(String, Value)>,
) -> Value {
    let map: HashMap<String, Value> = fields.into_iter().collect();
    Value::ModelResult {
        display: display.into(),
        summary: summary.into(),
        type_name,
        fields: Arc::new(map),
    }
}

/// Series of integer values from a `&[usize]` (e.g. cluster labels).
pub fn int_series(name: &str, values: &[usize]) -> Value {
    let vals: Vec<Value> = values.iter().map(|&v| Value::Int(v as i64)).collect();
    Value::Series(Arc::new(Series::new(name, vals)))
}

/// DataFrame with `variable` and `importance` columns from a variable-name list
/// and an importance vector.
pub fn feature_importance_df(names: &[String], importance: &Array1<f64>) -> Value {
    let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();
    let var_col: Vec<String> = (0..importance.len())
        .map(|i| names.get(i).cloned().unwrap_or_else(|| format!("x{i}")))
        .collect();
    columns.insert(
        "variable".into(),
        greeners::Column::String(ndarray::Array1::from(var_col)),
    );
    let imp_col: Vec<f64> = importance.iter().copied().collect();
    columns.insert(
        "importance".into(),
        greeners::Column::Float(ndarray::Array1::from(imp_col)),
    );
    Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(columns)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ))
}

/// DataFrame with `variable` and `coefficient` columns for named coefficient
/// vectors (e.g. OLS, conformal base predictor, DR-learner CATE).
pub fn coefficients_df(names: &[String], params: &Array1<f64>) -> Value {
    let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();
    let var_col: Vec<String> = (0..params.len())
        .map(|i| names.get(i).cloned().unwrap_or_else(|| format!("x{i}")))
        .collect();
    columns.insert(
        "variable".into(),
        greeners::Column::String(ndarray::Array1::from(var_col)),
    );
    let coef_col: Vec<f64> = params.iter().copied().collect();
    columns.insert(
        "coefficient".into(),
        greeners::Column::Float(ndarray::Array1::from(coef_col)),
    );
    Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(columns)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ))
}

pub fn var_param_names(k: usize, p: usize, var_names: &[String]) -> Vec<String> {
    let mut names = Vec::with_capacity(1 + k * p);
    names.push("const".into());
    for lag in 1..=p {
        for name in var_names.iter().take(k) {
            names.push(format!("L{lag}.{name}"));
        }
    }
    names
}

pub fn diag_children(r: &DiagResult) -> Vec<(String, Value)> {
    let mut vars: Vec<(String, Value)> = r
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    vars.push(("rendered".into(), Value::Str(r.rendered.clone())));
    vars
}

pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

pub fn column_to_value(name: &str, column: &greeners::Column) -> Value {
    use chrono::Timelike;
    let values: Vec<Value> = match column {
        greeners::Column::Float(arr) => arr.iter().map(|&v| Value::Float(v)).collect(),
        greeners::Column::Int(arr) => arr.iter().map(|&v| Value::Int(v)).collect(),
        greeners::Column::Bool(arr) => arr.iter().map(|&v| Value::Bool(v)).collect(),
        greeners::Column::String(arr) => arr.iter().cloned().map(Value::Str).collect(),
        greeners::Column::DateTime(arr) => arr
            .iter()
            .map(|dt| {
                Value::Str(format!(
                    "{} {:02}:{:02}:{:02}",
                    dt.date(),
                    dt.hour(),
                    dt.minute(),
                    dt.second()
                ))
            })
            .collect(),
        greeners::Column::Categorical(cat) => (0..cat.len())
            .map(|i| {
                cat.get_string(i)
                    .map(|s| Value::Str(s.to_string()))
                    .unwrap_or(Value::Nil)
            })
            .collect(),
    };
    Value::Series(Arc::new(Series::new(name, values)))
}
