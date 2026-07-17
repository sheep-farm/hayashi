use crate::lang::interpreter::models::{DFMModel, SurModel, ThreeSLSModel};
use crate::lang::interpreter::{Series, Value};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;

/// Context describing a generic regression-like result for DAP expansion.
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
        _ => Vec::new(),
    }
}

fn value_summary_and_type(v: &Value) -> (String, &'static str) {
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
        Value::UserFn(f) => (format!("<fn({})>", f.params.join(", ")), "Function"),
        _ => (v.to_string(), "Model"),
    }
}

fn regression_children<'a>(ctx: RegressionCtx<'a>) -> Vec<(String, Value)> {
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

fn coef_dataframe(
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

fn array1_to_series(name: &str, arr: &Array1<f64>) -> Value {
    let values: Vec<Value> = arr.iter().map(|&v| Value::Float(v)).collect();
    Value::Series(Arc::new(Series::new(name, values)))
}

fn array2_to_dataframe(_name: &str, arr: &Array2<f64>) -> Value {
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

fn f64_array_column(arr: &Array1<f64>) -> greeners::Column {
    greeners::Column::Float(Array1::from(arr.iter().copied().collect::<Vec<_>>()))
}

fn fit_dict(entries: &[(&str, Value)]) -> Value {
    let mut map = HashMap::new();
    for (k, v) in entries {
        map.insert((*k).to_string(), v.clone());
    }
    Value::Dict(Arc::new(map))
}

fn ols_fit_dict(r: &greeners::OlsResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("adj_r2", Value::Float(r.adj_r_squared)),
        ("f_stat", Value::Float(r.f_statistic)),
        ("prob_f", Value::Float(r.prob_f)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("sigma", Value::Float(r.sigma)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("cov_type", Value::Str(format!("{:?}", r.cov_type))),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn iv_fit_dict(r: &greeners::iv::IvResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("cov_type", Value::Str(format!("{:?}", r.cov_type))),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn panel_fit_dict(r: &greeners::panel::PanelResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn re_fit_dict(r: &greeners::panel::RandomEffectsResult) -> Value {
    fit_dict(&[
        ("r2_overall", Value::Float(r.r_squared_overall)),
        ("sigma_u", Value::Float(r.sigma_u)),
        ("sigma_e", Value::Float(r.sigma_e)),
        ("theta", Value::Float(r.theta)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn binary_fit_dict(r: &greeners::discrete::BinaryModelResult) -> Value {
    fit_dict(&[
        ("model_name", Value::Str(r.model_name.clone())),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn quantile_fit_dict(r: &greeners::QuantileResult) -> Value {
    fit_dict(&[
        ("tau", Value::Float(r.tau)),
        ("r2", Value::Float(r.r_squared)),
        ("iterations", Value::Int(r.iterations as i64)),
    ])
}

fn tobit_fit_dict(r: &greeners::TobitResult) -> Value {
    fit_dict(&[
        ("sigma", Value::Float(r.sigma)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_censored", Value::Int(r.n_censored as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
    ])
}

fn poisson_fit_dict(r: &greeners::PoissonResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("deviance", Value::Float(r.deviance)),
        ("null_deviance", Value::Float(r.null_deviance)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("pearson_chi2", Value::Float(r.pearson_chi2)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn negbin_fit_dict(r: &greeners::NegBinResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("deviance", Value::Float(r.deviance)),
        ("null_deviance", Value::Float(r.null_deviance)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("pearson_chi2", Value::Float(r.pearson_chi2)),
        ("alpha", Value::Float(r.alpha)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn glm_fit_dict(r: &greeners::GlmResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("deviance", Value::Float(r.deviance)),
        ("null_deviance", Value::Float(r.null_deviance)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("pearson_chi2", Value::Float(r.pearson_chi2)),
        ("dispersion", Value::Float(r.dispersion)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("family", Value::Str(format!("{:?}", r.family))),
    ])
}

fn rlm_fit_dict(r: &greeners::RlmResult) -> Value {
    fit_dict(&[
        ("scale", Value::Float(r.scale)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

fn beta_fit_dict(r: &greeners::BetaResult) -> Value {
    fit_dict(&[
        ("precision_param", Value::Float(r.precision_param)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

fn gmm_fit_dict(r: &greeners::GmmResult) -> Value {
    fit_dict(&[
        ("j_stat", Value::Float(r.j_stat)),
        ("j_p_value", Value::Float(r.j_p_value)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("df_overid", Value::Int(r.df_overid as i64)),
    ])
}

fn ab_fit_dict(r: &greeners::ArellanoBondResult) -> Value {
    fit_dict(&[
        ("sargan_stat", Value::Float(r.sargan_stat)),
        ("sargan_pvalue", Value::Float(r.sargan_pvalue)),
        ("sargan_df", Value::Int(r.sargan_df as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("t_bar", Value::Float(r.t_bar)),
        ("n_instruments", Value::Int(r.n_instruments as i64)),
        ("max_lags", Value::Int(r.max_lags as i64)),
        ("step", Value::Int(r.step as i64)),
        ("m1_stat", Value::Float(r.m1_stat)),
        ("m1_pval", Value::Float(r.m1_pval)),
        ("m2_stat", Value::Float(r.m2_stat)),
        ("m2_pval", Value::Float(r.m2_pval)),
    ])
}

fn sysgmm_fit_dict(r: &greeners::SystemGmmResult) -> Value {
    fit_dict(&[
        ("sargan_stat", Value::Float(r.sargan_stat)),
        ("sargan_pvalue", Value::Float(r.sargan_pvalue)),
        ("sargan_df", Value::Int(r.sargan_df as i64)),
        ("n_obs_fd", Value::Int(r.n_obs_fd as i64)),
        ("n_obs_lev", Value::Int(r.n_obs_lev as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("n_instruments", Value::Int(r.n_instruments as i64)),
        ("max_lags", Value::Int(r.max_lags as i64)),
        ("step", Value::Int(r.step as i64)),
        ("m1_stat", Value::Float(r.m1_stat)),
        ("m1_pval", Value::Float(r.m1_pval)),
        ("m2_stat", Value::Float(r.m2_stat)),
        ("m2_pval", Value::Float(r.m2_pval)),
    ])
}

fn pcse_fit_dict(r: &greeners::PcseResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("t_periods", Value::Int(r.t_periods as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
    ])
}

fn panel_gls_fit_dict(r: &greeners::PanelGlsResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("t_periods", Value::Int(r.t_periods as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("panels", Value::Str(format!("{:?}", r.panels))),
    ])
}

fn fe2sls_fit_dict(r: &greeners::PanelIvResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn ordered_fit_dict(r: &greeners::OrderedResult) -> Value {
    fit_dict(&[
        ("model_name", Value::Str(r.model_name.clone())),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_categories", Value::Int(r.n_categories as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("converged", Value::Bool(r.converged)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

fn zero_inflated_children(r: &greeners::ZeroInflatedResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let count_names = r.count_var_names.clone().unwrap_or_default();
    let inflate_names = r.inflate_var_names.clone().unwrap_or_default();

    let count_coef = coef_dataframe(
        &count_names,
        &r.count_params,
        &r.count_std_errors,
        &r.count_z_values,
        &r.count_p_values,
        None,
        None,
    );
    vars.push(("count_coefficients".into(), count_coef));

    let inflate_coef = coef_dataframe(
        &inflate_names,
        &r.inflate_params,
        &r.inflate_std_errors,
        &r.inflate_z_values,
        &r.inflate_p_values,
        None,
        None,
    );
    vars.push(("inflate_coefficients".into(), inflate_coef));

    vars.push(("fit".into(), zero_inflated_fit_dict(r)));
    vars
}

fn zero_inflated_fit_dict(r: &greeners::ZeroInflatedResult) -> Value {
    let mut entries: Vec<(&str, Value)> = vec![
        ("model_name", Value::Str(r.model_name.clone())),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("converged", Value::Bool(r.converged)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ];
    if let Some(alpha) = r.alpha {
        entries.push(("alpha", Value::Float(alpha)));
    }
    fit_dict(&entries)
}

fn mixed_children(r: &greeners::MixedResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let names = r.variable_names.clone().unwrap_or_default();
    let fixed = coef_dataframe(
        &names,
        &r.fixed_effects,
        &r.fixed_se,
        &r.z_values,
        &r.p_values,
        None,
        None,
    );
    vars.push(("fixed_effects".into(), fixed));

    let mut re_cols: IndexMap<String, greeners::Column> = IndexMap::new();
    for (group, vals) in r.random_effects.iter() {
        re_cols.insert(
            format!("group_{group}"),
            greeners::Column::Float(Array1::from(vals.iter().copied().collect::<Vec<_>>())),
        );
    }
    let re_df = Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(re_cols)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ));
    vars.push(("random_effects".into(), re_df));

    vars.push(("fit".into(), mixed_fit_dict(r)));
    vars
}

fn mixed_fit_dict(r: &greeners::MixedResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_groups", Value::Int(r.n_groups as i64)),
        ("var_resid", Value::Float(r.var_resid)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

fn glsar_fit_dict(r: &greeners::GlsarResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
    ])
}

fn sur_children(m: &SurModel) -> Vec<(String, Value)> {
    let r = &m.result;
    let mut vars = Vec::new();
    vars.push(("system_r2".into(), Value::Float(r.system_r2)));
    vars.push((
        "sigma_cross".into(),
        array2_to_dataframe("sigma", &r.sigma_cross),
    ));
    for (i, eq) in r.equations.iter().enumerate() {
        let name = if eq.name.is_empty() {
            format!("equation_{i}")
        } else {
            eq.name.clone()
        };
        let eq_fit = fit_dict(&[("r2", Value::Float(eq.r_squared))]);
        let eq_val = coef_dataframe(
            &(0..eq.params.len())
                .map(|j| format!("x{j}"))
                .collect::<Vec<_>>(),
            &eq.params,
            &eq.std_errors,
            &eq.t_values,
            &eq.p_values,
            None,
            None,
        );
        let mut wrap = HashMap::new();
        wrap.insert("coefficients".into(), eq_val);
        wrap.insert("fit".into(), eq_fit);
        vars.push((name, Value::Dict(Arc::new(wrap))));
    }
    vars
}

fn three_sls_children(m: &ThreeSLSModel) -> Vec<(String, Value)> {
    let r = &m.result;
    let mut vars = Vec::new();
    for (i, eq) in r.equations.iter().enumerate() {
        let name = m
            .eq_var_names
            .get(i)
            .and_then(|v| v.first())
            .cloned()
            .unwrap_or_else(|| format!("equation_{i}"));
        let eq_val = coef_dataframe(
            &(0..eq.params.len())
                .map(|j| format!("x{j}"))
                .collect::<Vec<_>>(),
            &eq.params,
            &eq.std_errors,
            &eq.t_values,
            &eq.p_values,
            None,
            None,
        );
        let mut wrap = HashMap::new();
        wrap.insert("coefficients".into(), eq_val);
        wrap.insert(
            "fit".into(),
            fit_dict(&[("r2", Value::Float(eq.r_squared))]),
        );
        vars.push((name, Value::Dict(Arc::new(wrap))));
    }
    vars
}

fn mnlogit_children(r: &greeners::MNLogitResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let names = r.variable_names.clone().unwrap_or_default();
    let categories = &r.category_labels;
    for (j, cat) in categories.iter().enumerate().skip(1) {
        let cat_name = format!("category_{cat:.0}");
        let cat_params = r.params.column(j - 1).to_owned();
        let cat_se = r.std_errors.column(j - 1).to_owned();
        let cat_z = r.z_values.column(j - 1).to_owned();
        let cat_p = r.p_values.column(j - 1).to_owned();
        let coef = coef_dataframe(&names, &cat_params, &cat_se, &cat_z, &cat_p, None, None);
        vars.push((cat_name, coef));
    }
    vars.push(("fit".into(), mnlogit_fit_dict(r)));
    vars
}

fn mnlogit_fit_dict(r: &greeners::MNLogitResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_categories", Value::Int(r.n_categories as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

fn arima_children(r: &greeners::ArimaResult) -> Vec<(String, Value)> {
    let mut params = vec![r.intercept];
    params.extend(r.ar_params.iter());
    params.extend(r.ma_params.iter());
    params.extend(r.seasonal_ar_params.iter());
    params.extend(r.seasonal_ma_params.iter());
    if let Some(exog) = &r.exog_params {
        params.extend(exog.iter());
    }
    let params = Array1::from_vec(params);
    let names = r.param_names.clone();
    regression_children(RegressionCtx {
        names,
        params: &params,
        std_errors: &r.std_errors,
        test_values: &r.t_values,
        p_values: &r.p_values,
        conf_lower: Some(&r.conf_lower),
        conf_upper: Some(&r.conf_upper),
        fit: arima_fit_dict(r),
        residuals: Some(&r.residuals),
        fitted_values: None,
        x: None,
    })
}

fn arima_fit_dict(r: &greeners::ArimaResult) -> Value {
    let mut entries = vec![
        ("sigma2", Value::Float(r.sigma2)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("p", Value::Int(r.order.p as i64)),
        ("d", Value::Int(r.order.d as i64)),
        ("q", Value::Int(r.order.q as i64)),
        ("method", Value::Str(r.estimation_method.clone())),
    ];
    if let Some(so) = &r.seasonal_order {
        entries.push(("seasonal_p", Value::Int(so.p as i64)));
        entries.push(("seasonal_d", Value::Int(so.d as i64)));
        entries.push(("seasonal_q", Value::Int(so.q as i64)));
        entries.push(("seasonal_period", Value::Int(so.s as i64)));
    }
    fit_dict(&entries)
}

fn garch_children(r: &greeners::GarchResult) -> Vec<(String, Value)> {
    let mut vars = regression_children(RegressionCtx {
        names: r.variable_names.clone(),
        params: &r.params,
        std_errors: &r.std_errors,
        test_values: &r.z_values,
        p_values: &r.p_values,
        conf_lower: Some(&r.conf_lower),
        conf_upper: Some(&r.conf_upper),
        fit: garch_fit_dict(r),
        residuals: Some(&r.residuals),
        fitted_values: None,
        x: None,
    });
    vars.push((
        "conditional_variance".into(),
        array1_to_series("conditional_variance", &r.conditional_variance),
    ));
    vars.push((
        "standardized_residuals".into(),
        array1_to_series("standardized_residuals", &r.standardized_residuals),
    ));
    vars
}

fn garch_fit_dict(r: &greeners::GarchResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("p", Value::Int(r.p as i64)),
        ("q", Value::Int(r.q as i64)),
        ("model_type", Value::Str(format!("{:?}", r.model_type))),
        ("dist", Value::Str(format!("{:?}", r.dist))),
    ])
}

fn ets_children(r: &greeners::ETSResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    vars.push(("level".into(), array1_to_series("level", &r.level)));
    if !r.trend.is_empty() {
        vars.push(("trend".into(), array1_to_series("trend", &r.trend)));
    }
    if !r.seasonal.is_empty() {
        vars.push(("seasonal".into(), array1_to_series("seasonal", &r.seasonal)));
    }
    if !r.fitted_values.is_empty() {
        vars.push((
            "fitted_values".into(),
            array1_to_series("fitted_values", &r.fitted_values),
        ));
    }
    if !r.residuals.is_empty() {
        vars.push((
            "residuals".into(),
            array1_to_series("residuals", &r.residuals),
        ));
    }
    vars.push(("fit".into(), ets_fit_dict(r)));
    vars
}

fn ets_fit_dict(r: &greeners::ETSResult) -> Value {
    let mut entries = vec![
        ("alpha", Value::Float(r.alpha)),
        ("sse", Value::Float(r.sse)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("last_level", Value::Float(r.last_level)),
        ("seasonal_periods", Value::Int(r.seasonal_periods as i64)),
        ("trend_type", Value::Str(r.trend_type.clone())),
        ("seasonal_type", Value::Str(r.seasonal_type.clone())),
        ("damped", Value::Bool(r.damped)),
    ];
    if let Some(beta) = r.beta {
        entries.push(("beta", Value::Float(beta)));
    }
    if let Some(gamma) = r.gamma {
        entries.push(("gamma", Value::Float(gamma)));
    }
    if let Some(phi) = r.phi {
        entries.push(("phi", Value::Float(phi)));
    }
    if !r.last_trend.is_nan() {
        entries.push(("last_trend", Value::Float(r.last_trend)));
    }
    fit_dict(&entries)
}

fn mstl_children(r: &greeners::MSTLResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    vars.push(("trend".into(), array1_to_series("trend", &r.trend)));
    for (i, s) in r.seasonal.iter().enumerate() {
        let period = r.periods.get(i).copied().unwrap_or(i);
        let name = format!("seasonal_{period}");
        vars.push((name.clone(), array1_to_series(&name, s)));
    }
    if !r.resid.is_empty() {
        vars.push(("residuals".into(), array1_to_series("residuals", &r.resid)));
    }
    vars.push((
        "observed".into(),
        array1_to_series("observed", &r.observed()),
    ));
    vars.push(("fit".into(), mstl_fit_dict(r)));
    vars
}

fn mstl_fit_dict(r: &greeners::MSTLResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        (
            "periods",
            Value::List(Arc::new(
                r.periods.iter().map(|&p| Value::Int(p as i64)).collect(),
            )),
        ),
    ])
}

fn uc_children(r: &greeners::UCResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let n = r.params.len();
    let params = if n > 0 {
        Array1::from_vec(r.params.clone())
    } else {
        Array1::zeros(0)
    };
    let names = r.param_names.clone();
    if !params.is_empty() {
        let coef = coef_dataframe(
            &names,
            &params,
            &Array1::zeros(n),
            &Array1::zeros(n),
            &Array1::zeros(n),
            None,
            None,
        );
        vars.push(("coefficients".into(), coef));
    }
    if !r.level.is_empty() {
        vars.push(("level".into(), array1_to_series("level", &r.level)));
    }
    if let Some(trend) = &r.trend {
        if !trend.is_empty() {
            vars.push(("trend".into(), array1_to_series("trend", trend)));
        }
    }
    if let Some(seasonal) = &r.seasonal {
        if !seasonal.is_empty() {
            vars.push(("seasonal".into(), array1_to_series("seasonal", seasonal)));
        }
    }
    if !r.residuals.is_empty() {
        vars.push((
            "residuals".into(),
            array1_to_series("residuals", &r.residuals),
        ));
    }
    vars.push(("fit".into(), uc_fit_dict(r)));
    vars
}

fn uc_fit_dict(r: &greeners::UCResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("level_type", Value::Str(format!("{:?}", r.level_type))),
        (
            "seasonal_type",
            Value::Str(format!("{:?}", r.seasonal_type)),
        ),
    ])
}

fn local_level_children(r: &greeners::LocalLevelResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let filtered: Vec<f64> = r
        .filtered_states
        .iter()
        .map(|s| s.first().copied().unwrap_or(f64::NAN))
        .collect();
    let smoothed: Vec<f64> = r
        .smoothed_states
        .iter()
        .map(|s| s.first().copied().unwrap_or(f64::NAN))
        .collect();
    if !filtered.is_empty() {
        vars.push((
            "filtered".into(),
            array1_to_series("filtered", &Array1::from_vec(filtered)),
        ));
    }
    if !smoothed.is_empty() {
        vars.push((
            "smoothed".into(),
            array1_to_series("smoothed", &Array1::from_vec(smoothed)),
        ));
    }
    vars.push(("fit".into(), local_level_fit_dict(r)));
    vars
}

fn local_level_fit_dict(r: &greeners::LocalLevelResult) -> Value {
    fit_dict(&[
        ("sigma_obs", Value::Float(r.sigma_obs)),
        ("sigma_state", Value::Float(r.sigma_state)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("n_obs", Value::Int(r.n_obs as i64)),
    ])
}

fn autoreg_children(r: &greeners::AutoRegResult) -> Vec<(String, Value)> {
    regression_children(RegressionCtx {
        names: r.param_names.clone(),
        params: &r.params,
        std_errors: &r.std_errors,
        test_values: &r.t_values,
        p_values: &r.p_values,
        conf_lower: None,
        conf_upper: None,
        fit: autoreg_fit_dict(r),
        residuals: Some(&r.residuals),
        fitted_values: Some(&r.fitted_values),
        x: None,
    })
}

fn autoreg_fit_dict(r: &greeners::AutoRegResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("adj_r2", Value::Float(r.adj_r_squared)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("lags", Value::Int(r.lags as i64)),
        ("trend", Value::Str(r.trend.clone())),
    ])
}

fn ardl_children(r: &greeners::ARDLResult) -> Vec<(String, Value)> {
    regression_children(RegressionCtx {
        names: r.param_names.clone(),
        params: &r.params,
        std_errors: &r.std_errors,
        test_values: &r.t_values,
        p_values: &r.p_values,
        conf_lower: None,
        conf_upper: None,
        fit: ardl_fit_dict(r),
        residuals: Some(&r.residuals),
        fitted_values: Some(&r.fitted_values),
        x: None,
    })
}

fn ardl_fit_dict(r: &greeners::ARDLResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("adj_r2", Value::Float(r.adj_r_squared)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("y_lags", Value::Int(r.y_lags as i64)),
        ("x_lags", Value::Int(r.x_lags as i64)),
    ])
}

fn threshold_children(r: &greeners::threshold::ThresholdResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let regime1_names: Vec<String> = (0..r.params_regime1.len())
        .map(|i| format!("x{i}"))
        .collect();
    let regime2_names: Vec<String> = (0..r.params_regime2.len())
        .map(|i| format!("x{i}"))
        .collect();
    let r1 = coef_dataframe(
        &regime1_names,
        &r.params_regime1,
        &Array1::zeros(r.params_regime1.len()),
        &Array1::zeros(r.params_regime1.len()),
        &Array1::zeros(r.params_regime1.len()),
        None,
        None,
    );
    let r2 = coef_dataframe(
        &regime2_names,
        &r.params_regime2,
        &Array1::zeros(r.params_regime2.len()),
        &Array1::zeros(r.params_regime2.len()),
        &Array1::zeros(r.params_regime2.len()),
        None,
        None,
    );
    vars.push(("regime1".into(), r1));
    vars.push(("regime2".into(), r2));
    vars.push(("fit".into(), threshold_fit_dict(r)));
    vars
}

fn threshold_fit_dict(r: &greeners::threshold::ThresholdResult) -> Value {
    fit_dict(&[
        ("threshold_gamma", Value::Float(r.threshold_gamma)),
        ("r2", Value::Float(r.r_squared)),
        ("ssr_min", Value::Float(r.ssr_min)),
        ("n_search", Value::Int(r.n_search as i64)),
    ])
}

fn array2_to_dataframe_named(arr: &Array2<f64>, col_names: &[String]) -> Value {
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

fn var_param_names(k: usize, p: usize, var_names: &[String]) -> Vec<String> {
    let mut names = Vec::with_capacity(1 + k * p);
    names.push("const".into());
    for lag in 1..=p {
        for name in var_names.iter().take(k) {
            names.push(format!("L{lag}.{name}"));
        }
    }
    names
}

fn var_children(r: &greeners::var::VarResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let k = r.n_vars;
    let p = r.lags;
    let param_names = var_param_names(k, p, &r.var_names);
    let total_rows = r.params.nrows();
    for i in 0..k {
        let dep = r
            .var_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("var{i}"));
        let params_col = r.params.column(i).to_owned();
        let se_col = r.std_errors.column(i).to_owned();
        let zeros = Array1::zeros(total_rows);
        let coef = coef_dataframe(
            &param_names,
            &params_col,
            &se_col,
            &zeros,
            &zeros,
            None,
            None,
        );
        let mut wrap = HashMap::new();
        wrap.insert("coefficients".into(), coef);
        vars.push((dep, Value::Dict(Arc::new(wrap))));
    }
    vars.push((
        "sigma_u".into(),
        array2_to_dataframe_named(&r.sigma_u, &r.var_names),
    ));
    vars.push(("fit".into(), var_fit_dict(r)));
    vars
}

fn var_fit_dict(r: &greeners::var::VarResult) -> Value {
    fit_dict(&[
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("lags", Value::Int(r.lags as i64)),
        ("n_vars", Value::Int(r.n_vars as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
    ])
}

fn vecm_children(r: &greeners::vecm::VecmResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let row_labels: Vec<String> = (0..r.beta.nrows()).map(|i| format!("eq{i}")).collect();
    let col_labels = r.variable_names.clone();
    vars.push((
        "beta".into(),
        array2_to_dataframe_named(&r.beta, &col_labels),
    ));
    vars.push((
        "alpha".into(),
        array2_to_dataframe_named(&r.alpha, &col_labels),
    ));
    vars.push((
        "gamma".into(),
        array2_to_dataframe_named(&r.gamma, &row_labels),
    ));
    vars.push((
        "std_errors_beta".into(),
        array2_to_dataframe_named(&r.std_errors_beta, &col_labels),
    ));
    vars.push((
        "std_errors_alpha".into(),
        array2_to_dataframe_named(&r.std_errors_alpha, &col_labels),
    ));
    vars.push((
        "std_errors_gamma".into(),
        array2_to_dataframe_named(&r.std_errors_gamma, &row_labels),
    ));
    vars.push((
        "residuals".into(),
        array2_to_dataframe_named(&r.residuals, &col_labels),
    ));
    vars.push((
        "eigenvalues".into(),
        array1_to_series("eigenvalues", &r.eigenvalues),
    ));
    vars.push(("fit".into(), vecm_fit_dict(r)));
    vars
}

fn vecm_fit_dict(r: &greeners::vecm::VecmResult) -> Value {
    fit_dict(&[
        ("rank", Value::Int(r.rank as i64)),
        ("n_vars", Value::Int(r.n_vars as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("lags", Value::Int(r.lags as i64)),
    ])
}

fn varma_children(r: &greeners::varma::VarmaResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let col_names: Vec<String> = (0..r.n_vars).map(|i| format!("var{i}")).collect();
    let ar_col_names: Vec<String> = (0..r.ar_params.ncols())
        .map(|i| format!("col{i}"))
        .collect();
    let ma_col_names: Vec<String> = (0..r.ma_params.ncols())
        .map(|i| format!("col{i}"))
        .collect();
    vars.push((
        "ar_params".into(),
        array2_to_dataframe_named(&r.ar_params, &ar_col_names),
    ));
    vars.push((
        "ma_params".into(),
        array2_to_dataframe_named(&r.ma_params, &ma_col_names),
    ));
    if let Some(exog) = &r.exog_params {
        let exog_names: Vec<String> = (0..exog.ncols()).map(|i| format!("x{i}")).collect();
        vars.push((
            "exog_params".into(),
            array2_to_dataframe_named(exog, &exog_names),
        ));
    }
    vars.push((
        "sigma_u".into(),
        array2_to_dataframe_named(&r.sigma_u, &col_names),
    ));
    vars.push(("fit".into(), varma_fit_dict(r)));
    vars
}

fn varma_fit_dict(r: &greeners::varma::VarmaResult) -> Value {
    fit_dict(&[
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("p_lags", Value::Int(r.p_lags as i64)),
        ("q_lags", Value::Int(r.q_lags as i64)),
        ("n_vars", Value::Int(r.n_vars as i64)),
        ("n_exog", Value::Int(r.n_exog as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
    ])
}

fn svar_children(r: &greeners::svar::SVarResult) -> Vec<(String, Value)> {
    let mut vars = var_children(&r.var_result);
    let k = r.var_result.n_vars;
    let names: Vec<String> = (0..k).map(|i| format!("var{i}")).collect();
    vars.push((
        "a_matrix".into(),
        array2_to_dataframe_named(&r.a_matrix, &names),
    ));
    vars.push((
        "b_matrix".into(),
        array2_to_dataframe_named(&r.b_matrix, &names),
    ));
    vars.push((
        "identification".into(),
        Value::Str(r.identification.clone()),
    ));
    vars
}

fn msar_children(r: &greeners::markov_autoreg::MarkovAutoregResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    for j in 0..r.k_regimes {
        let regime_name = format!("regime_{j}");
        let mut map = HashMap::new();
        map.insert("mean".into(), Value::Float(r.regime_means[j]));
        map.insert("sigma".into(), Value::Float(r.regime_sigmas[j]));
        let ar_names: Vec<String> = (0..r.ar_order).map(|l| format!("AR.L{}", l + 1)).collect();
        let ar = r.ar_params.row(j).to_owned();
        let zeros = Array1::zeros(ar.len());
        let ar_df = coef_dataframe(&ar_names, &ar, &zeros, &zeros, &zeros, None, None);
        map.insert("ar_coefficients".into(), ar_df);
        vars.push((regime_name, Value::Dict(Arc::new(map))));
    }
    let regime_labels: Vec<String> = (0..r.k_regimes).map(|i| format!("regime_{i}")).collect();
    vars.push((
        "transition_matrix".into(),
        array2_to_dataframe_named(&r.transition_matrix, &regime_labels),
    ));
    vars.push((
        "smoothed_probs".into(),
        array2_to_dataframe_named(&r.smoothed_probs, &regime_labels),
    ));
    vars.push((
        "filtered_probs".into(),
        array2_to_dataframe_named(&r.filtered_probs, &regime_labels),
    ));
    vars.push(("fit".into(), msar_fit_dict(r)));
    vars
}

fn msar_fit_dict(r: &greeners::markov_autoreg::MarkovAutoregResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("k_regimes", Value::Int(r.k_regimes as i64)),
        ("ar_order", Value::Int(r.ar_order as i64)),
    ])
}

fn dfm_children(m: &DFMModel) -> Vec<(String, Value)> {
    let r = &*m.result;
    let mut vars = Vec::new();
    let factor_names: Vec<String> = (0..r.n_factors).map(|i| format!("F{}", i + 1)).collect();
    vars.push((
        "factors".into(),
        array2_to_dataframe_named(&r.factors, &factor_names),
    ));
    vars.push((
        "loadings".into(),
        array2_to_dataframe_named(&r.factor_loadings, &factor_names),
    ));
    let ar_names: Vec<String> = (0..r.factor_order)
        .map(|i| format!("AR{}", i + 1))
        .collect();
    for (i, ar) in r.factor_ar_params.iter().enumerate() {
        let name = ar_names.get(i).cloned().unwrap_or_else(|| format!("AR{i}"));
        vars.push((name.clone(), array2_to_dataframe_named(ar, &factor_names)));
    }
    vars.push((
        "factor_cov".into(),
        array2_to_dataframe_named(&r.sigma_factor, &factor_names),
    ));
    vars.push((
        "obs_variances".into(),
        array1_to_series("obs_variances", &r.sigma_obs),
    ));
    vars.push(("fit".into(), dfm_fit_dict(r)));
    vars
}

fn dfm_fit_dict(r: &greeners::DynamicFactorResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_series", Value::Int(r.n_vars as i64)),
        ("n_factors", Value::Int(r.n_factors as i64)),
        ("factor_order", Value::Int(r.factor_order as i64)),
    ])
}

fn markov_children(r: &greeners::markov::MarkovSwitchingResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    for j in 0..r.n_regimes {
        let regime_name = format!("regime_{j}");
        let mut map = HashMap::new();
        let params = &r.regime_params[j];
        let ar_names: Vec<String> = (0..r.ar_order).map(|l| format!("AR.L{}", l + 1)).collect();
        let zeros = Array1::zeros(params.len());
        let coef = coef_dataframe(&ar_names, params, &zeros, &zeros, &zeros, None, None);
        map.insert("coefficients".into(), coef);
        map.insert("variance".into(), Value::Float(r.regime_variances[j]));
        vars.push((regime_name, Value::Dict(Arc::new(map))));
    }
    let regime_labels: Vec<String> = (0..r.n_regimes).map(|i| format!("regime_{i}")).collect();
    vars.push((
        "transition_matrix".into(),
        array2_to_dataframe_named(&r.transition_matrix, &regime_labels),
    ));
    vars.push((
        "smoothed_probs".into(),
        array2_to_dataframe_named(&r.smoothed_probs, &regime_labels),
    ));
    vars.push((
        "filtered_probs".into(),
        array2_to_dataframe_named(&r.filtered_probs, &regime_labels),
    ));
    vars.push(("fit".into(), markov_fit_dict(r)));
    vars
}

fn markov_fit_dict(r: &greeners::markov::MarkovSwitchingResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_regimes", Value::Int(r.n_regimes as i64)),
        ("ar_order", Value::Int(r.ar_order as i64)),
    ])
}

fn column_to_value(name: &str, column: &greeners::Column) -> Value {
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
