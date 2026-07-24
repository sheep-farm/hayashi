use super::models::{BinaryModel, DFMModel, FactorModel, OlsModel, PcaModel, PenalizedModel, SurModel, ThreeSLSModel};
use super::Value;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;

/// Canonical, Hayashi-side view of any model result.
///
/// `ModelView` decouples consumers (DAP, `predict`, `esttab`, `tidy`,
/// `glance`, `export`) from the heterogeneous `Value::*Result` variants and
/// from the `Greeners` result types.
///
/// Optional fields are `None` when the underlying estimator does not
/// provide them.  Extra estimator-specific data lives in `extras`.
#[derive(Clone)]
pub struct ModelView {
    /// Type name shown to the user, e.g. "OlsResult".
    pub type_name: String,
    /// Short summary line for DAP / hover.
    pub summary: String,
    /// Names of the right-hand-side variables (including "_cons" when an
    /// intercept is present).  May be generated as "x0", "x1", ... when the
    /// result does not store names.
    pub variable_names: Vec<String>,
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    /// t-statistics or z-statistics, depending on the estimator.
    pub test_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Option<Array1<f64>>,
    pub conf_upper: Option<Array1<f64>>,
    /// Fit / summary statistics such as `r2`, `aic`, `bic`, `log_lik`.
    pub fit: HashMap<String, Value>,
    pub residuals: Option<Array1<f64>>,
    pub fitted_values: Option<Array1<f64>>,
    /// Design matrix, kept for post-estimation diagnostics and `predict`.
    pub x: Option<Array2<f64>>,
    /// Estimator-specific extras.  Examples:
    /// - `kind`: "logit" | "probit" for binary models.
    /// - `y`: response vector for binary models.
    /// - `eq_var_names`: variable names per equation for SUR/3SLS.
    /// - `var_names`: original variable names for PCA/Factor/DFM.
    pub extras: HashMap<String, Value>,
}

impl ModelView {
    /// Number of observations, when available in `fit`.
    pub fn n_obs(&self) -> Option<usize> {
        match self.fit.get("n_obs") {
            Some(Value::Int(n)) => Some(*n as usize),
            Some(Value::Float(n)) => Some(*n as usize),
            _ => None,
        }
    }

    /// Pseudo / adjusted / within R-squared, when available.
    pub fn r_squared(&self) -> Option<f64> {
        match self.fit.get("r_squared") {
            Some(Value::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Access an extra field, typed as a list of floats, if present.
    pub fn extra_vec(&self, key: &str) -> Option<Vec<f64>> {
        match self.extras.get(key) {
            Some(Value::List(v)) => v
                .iter()
                .map(|x| match x {
                    Value::Float(f) => Some(*f),
                    Value::Int(i) => Some(*i as f64),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>(),
            Some(Value::Series(s)) => s.values.iter().map(|v| match v {
                Value::Float(f) => Some(*f),
                Value::Int(i) => Some(*i as f64),
                _ => None,
            }).collect::<Option<Vec<_>>>(),
            _ => None,
        }
    }
}

impl std::fmt::Debug for ModelView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelView")
            .field("type_name", &self.type_name)
            .field("summary", &self.summary)
            .field("variable_names", &self.variable_names)
            .field("params_len", &self.params.len())
            .finish()
    }
}

/// Convert any `Value` model result into a `ModelView`.
///
/// Returns `None` for non-model values.
impl Value {
    pub fn to_model_view(&self) -> Option<ModelView> {
        match self {
            Value::OlsResult(m) => Some(model_view_from_ols(m)),
            Value::BinaryResult(m) => Some(model_view_from_binary(m)),
            Value::SurResult(m) => Some(model_view_from_sur(m)),
            Value::PcaResult(m) => Some(model_view_from_pca(m)),
            Value::FactorResult(m) => Some(model_view_from_factor(m)),
            Value::DFMResult(m) => Some(model_view_from_dfm(m)),
            Value::ThreeSLSResult(m) => Some(model_view_from_three_sls(m)),
            Value::PenalizedResult(m) => Some(model_view_from_penalized(m)),
            Value::IvResult(r) => Some(model_view_from_iv(r)),
            Value::PanelResult(r) => Some(model_view_from_panel(r)),
            Value::ReResult(r) => Some(model_view_from_random_effects(r)),
            Value::QuantileResult(r) => Some(model_view_from_quantile(r)),
            Value::TobitResult(r) => Some(model_view_from_tobit(r)),
            Value::PoissonResult(r) => Some(model_view_from_poisson(r)),
            Value::NegBinResult(r) => Some(model_view_from_negbin(r)),
            Value::GlmResult(r) => Some(model_view_from_glm(r)),
            Value::RlmResult(r) => Some(model_view_from_rlm(r)),
            Value::BetaResult(r) => Some(model_view_from_beta(r)),
            Value::GmmResult(r) => Some(model_view_from_gmm(r)),
            Value::ModelResult { display, summary, type_name, fields } => {
                Some(model_view_from_model_result(display, summary, type_name, fields))
            }
            _ => None,
        }
    }
}

fn names_or_x(n: Option<&Vec<String>>, len: usize) -> Vec<String> {
    n.cloned().unwrap_or_else(|| (0..len).map(|i| format!("x{i}")).collect())
}

fn model_view_from_ols(m: &OlsModel) -> ModelView {
    let r = &m.result;
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("df_resid".into(), Value::Int(r.df_resid as i64));
    fit.insert("r_squared".into(), Value::Float(r.r_squared));
    fit.insert("adj_r_squared".into(), Value::Float(r.adj_r_squared));
    fit.insert("f_statistic".into(), Value::Float(r.f_statistic));
    fit.insert("f_p_value".into(), Value::Float(r.prob_f));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("sigma".into(), Value::Float(r.sigma));

    ModelView {
        type_name: "OlsResult".into(),
        summary: format!("OLS(k={}, n={}), R2={:.4}", r.params.len(), r.n_obs, r.r_squared),
        variable_names: r.variable_names.clone().unwrap_or_default(),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: Some(r.conf_lower.clone()),
        conf_upper: Some(r.conf_upper.clone()),
        fit,
        residuals: Some(m.residuals.clone()),
        fitted_values: None,
        x: Some(m.x.clone()),
        extras: HashMap::new(),
    }
}

fn model_view_from_binary(m: &BinaryModel) -> ModelView {
    let r = &m.result;
    let mut fit = HashMap::new();
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));

    let mut extras = HashMap::new();
    extras.insert("kind".into(), Value::Str(m.kind.clone()));

    ModelView {
        type_name: "BinaryResult".into(),
        summary: format!("{}(k={}), pseudo-R2={:.4}",
            r.model_name, r.params.len(), r.pseudo_r2),
        variable_names: m.coef_names.clone(),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.z_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: Some(m.x.clone()),
        extras,
    }
}

#[allow(clippy::type_complexity)]
fn flatten_equations(equations: &[greeners::sur::SurEquationResult]) -> (Vec<String>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let total_len: usize = equations.iter().map(|eq| eq.params.len()).sum();
    let mut names = Vec::with_capacity(total_len);
    let mut params = Vec::with_capacity(total_len);
    let mut std_errors = Vec::with_capacity(total_len);
    let mut t_values = Vec::with_capacity(total_len);
    let mut p_values = Vec::with_capacity(total_len);

    for eq in equations {
        for i in 0..eq.params.len() {
            let vname = if i == 0 { format!("{}:_cons", eq.name) } else { format!("{}:x{i}", eq.name) };
            names.push(vname);
            params.push(eq.params[i]);
            std_errors.push(eq.std_errors[i]);
            t_values.push(eq.t_values[i]);
            p_values.push(eq.p_values[i]);
        }
    }

    (names, params.into(), std_errors.into(), t_values.into(), p_values.into())
}

fn model_view_from_sur(m: &SurModel) -> ModelView {
    let r = &m.result;
    let n_obs = r.equations.first().map(|eq| eq.params.len()).unwrap_or(0);

    let mut fit = HashMap::new();
    fit.insert("n_equations".into(), Value::Int(r.equations.len() as i64));
    fit.insert("system_r2".into(), Value::Float(r.system_r2));

    let mut extras = HashMap::new();
    extras.insert(
        "eq_var_names".into(),
        Value::List(Arc::new(
            m.eq_var_names
                .iter()
                .map(|v| Value::List(Arc::new(v.iter().map(|s| Value::Str(s.clone())).collect())))
                .collect(),
        )),
    );

    let (names, params, std_errors, t_values, p_values) = flatten_equations(&r.equations);

    ModelView {
        type_name: "SurResult".into(),
        summary: format!("SUR(equations={}, n≈{}), system-R2={:.4}", r.equations.len(), n_obs, r.system_r2),
        variable_names: names,
        params,
        std_errors,
        test_values: t_values,
        p_values,
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}

fn model_view_from_pca(m: &PcaModel) -> ModelView {
    let r = &m.result;
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_components".into(), Value::Int(r.n_components as i64));

    let mut extras = HashMap::new();
    extras.insert("var_names".into(), Value::List(Arc::new(
        m.var_names.iter().map(|s| Value::Str(s.clone())).collect()
    )));
    extras.insert("explained_variance".into(), Value::List(Arc::new(
        r.explained_variance.iter().map(|&v| Value::Float(v)).collect()
    )));
    extras.insert("explained_variance_ratio".into(), Value::List(Arc::new(
        r.explained_variance_ratio.iter().map(|&v| Value::Float(v)).collect()
    )));

    // PCA does not have regression-style coefficients in the usual sense.
    ModelView {
        type_name: "PcaResult".into(),
        summary: format!("PCA(components={}, variables={}, n={})", r.n_components, m.var_names.len(), r.n_obs),
        variable_names: (0..r.n_components).map(|i| format!("PC{}", i + 1)).collect(),
        params: Array1::zeros(r.n_components),
        std_errors: Array1::zeros(r.n_components),
        test_values: Array1::zeros(r.n_components),
        p_values: Array1::ones(r.n_components),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}

fn model_view_from_factor(m: &FactorModel) -> ModelView {
    let r = &m.result;
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_factors".into(), Value::Int(r.n_factors as i64));

    let mut extras = HashMap::new();
    extras.insert("var_names".into(), Value::List(Arc::new(
        m.var_names.iter().map(|s| Value::Str(s.clone())).collect()
    )));
    extras.insert("eigenvalues".into(), Value::List(Arc::new(
        r.eigenvalues.iter().map(|&v| Value::Float(v)).collect()
    )));

    ModelView {
        type_name: "FactorResult".into(),
        summary: format!("Factor Analysis(factors={}, variables={}, n={})", r.n_factors, m.var_names.len(), r.n_obs),
        variable_names: (0..r.n_factors).map(|i| format!("F{}", i + 1)).collect(),
        params: Array1::zeros(r.n_factors),
        std_errors: Array1::zeros(r.n_factors),
        test_values: Array1::zeros(r.n_factors),
        p_values: Array1::ones(r.n_factors),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}

fn model_view_from_dfm(m: &DFMModel) -> ModelView {
    let r = &m.result;
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_factors".into(), Value::Int(r.n_factors as i64));
    fit.insert("n_vars".into(), Value::Int(r.n_vars as i64));

    let mut extras = HashMap::new();
    extras.insert("var_names".into(), Value::List(Arc::new(
        m.var_names.iter().map(|s| Value::Str(s.clone())).collect()
    )));

    ModelView {
        type_name: "DFMResult".into(),
        summary: format!("Dynamic Factor Model(factors={}, n={})", r.n_factors, r.n_obs),
        variable_names: (0..r.n_factors).map(|i| format!("Factor{}", i + 1)).collect(),
        params: Array1::zeros(r.n_factors),
        std_errors: Array1::zeros(r.n_factors),
        test_values: Array1::zeros(r.n_factors),
        p_values: Array1::ones(r.n_factors),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}

#[allow(clippy::type_complexity)]
fn flatten_three_sls_equations(equations: &[greeners::three_sls::EquationResult]) -> (Vec<String>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let total_len: usize = equations.iter().map(|eq| eq.params.len()).sum();
    let mut names = Vec::with_capacity(total_len);
    let mut params = Vec::with_capacity(total_len);
    let mut std_errors = Vec::with_capacity(total_len);
    let mut t_values = Vec::with_capacity(total_len);
    let mut p_values = Vec::with_capacity(total_len);

    for eq in equations {
        for i in 0..eq.params.len() {
            let vname = if i == 0 { format!("{}:_cons", eq.name) } else { format!("{}:x{i}", eq.name) };
            names.push(vname);
            params.push(eq.params[i]);
            std_errors.push(eq.std_errors[i]);
            t_values.push(eq.t_values[i]);
            p_values.push(eq.p_values[i]);
        }
    }

    (names, params.into(), std_errors.into(), t_values.into(), p_values.into())
}

fn model_view_from_three_sls(m: &ThreeSLSModel) -> ModelView {
    let r = &m.result;
    let n_obs = r.equations.first().map(|eq| eq.params.len()).unwrap_or(0);

    let mut fit = HashMap::new();
    fit.insert("n_equations".into(), Value::Int(r.equations.len() as i64));
    fit.insert("system_r2".into(), Value::Float(r.system_r2));

    let mut extras = HashMap::new();
    extras.insert(
        "eq_var_names".into(),
        Value::List(Arc::new(
            m.eq_var_names
                .iter()
                .map(|v| Value::List(Arc::new(v.iter().map(|s| Value::Str(s.clone())).collect())))
                .collect(),
        )),
    );

    let (names, params, std_errors, t_values, p_values) = flatten_three_sls_equations(&r.equations);

    ModelView {
        type_name: "ThreeSLSResult".into(),
        summary: format!("3SLS(equations={}, n≈{}), system-R2={:.4}", r.equations.len(), n_obs, r.system_r2),
        variable_names: names,
        params,
        std_errors,
        test_values: t_values,
        p_values,
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}

fn model_view_from_penalized(m: &PenalizedModel) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(m.n_obs as i64));
    fit.insert("r_squared".into(), Value::Float(m.r_squared));
    fit.insert("alpha".into(), Value::Float(m.alpha));
    if let Some(l1r) = m.l1_ratio {
        fit.insert("l1_ratio".into(), Value::Float(l1r));
    }

    let mut extras = HashMap::new();
    extras.insert("kind".into(), Value::Str(m.kind.clone()));

    ModelView {
        type_name: "PenalizedResult".into(),
        summary: format!("{}(k={}, n={}), R2={:.4}",
            match m.kind.as_str() {
                "ridge" => "Ridge",
                "lasso" => "Lasso",
                "elasticnet" => "ElasticNet",
                _ => "Penalized Regression",
            },
            m.params.len(), m.n_obs, m.r_squared),
        variable_names: m.variable_names.clone(),
        params: m.params.clone(),
        std_errors: m.std_errors.clone(),
        test_values: Array1::zeros(m.params.len()),
        p_values: Array1::ones(m.params.len()),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}

// ── Rc<GreenersResult> converters ───────────────────────────────────────────

fn model_view_from_iv(r: &std::rc::Rc<greeners::iv::IvResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("df_resid".into(), Value::Int(r.df_resid as i64));
    fit.insert("r_squared".into(), Value::Float(r.r_squared));
    fit.insert("sigma".into(), Value::Float(r.sigma));

    ModelView {
        type_name: "IvResult".into(),
        summary: format!("IV/2SLS(k={}, n={}), R2={:.4}", r.params.len(), r.n_obs, r.r_squared),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_panel(r: &std::rc::Rc<greeners::panel::PanelResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_entities".into(), Value::Int(r.n_entities as i64));
    fit.insert("r_squared".into(), Value::Float(r.r_squared));
    fit.insert("sigma".into(), Value::Float(r.sigma));

    ModelView {
        type_name: "PanelResult".into(),
        summary: format!("Fixed Effects(k={}, n={}, panels={}), R2={:.4}",
            r.params.len(), r.n_obs, r.n_entities, r.r_squared),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_random_effects(r: &std::rc::Rc<greeners::panel::RandomEffectsResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("r_squared_overall".into(), Value::Float(r.r_squared_overall));
    fit.insert("sigma_u".into(), Value::Float(r.sigma_u));
    fit.insert("sigma_e".into(), Value::Float(r.sigma_e));
    fit.insert("theta".into(), Value::Float(r.theta));

    ModelView {
        type_name: "ReResult".into(),
        summary: format!("Random Effects(k={}), R2={:.4}",
            r.params.len(), r.r_squared_overall),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_quantile(r: &std::rc::Rc<greeners::QuantileResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("tau".into(), Value::Float(r.tau));
    fit.insert("r_squared".into(), Value::Float(r.r_squared));
    fit.insert("iterations".into(), Value::Int(r.iterations as i64));

    ModelView {
        type_name: "QuantileResult".into(),
        summary: format!("Quantile Regression(tau={:.2}, k={})", r.tau, r.params.len()),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_tobit(r: &std::rc::Rc<greeners::TobitResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));

    ModelView {
        type_name: "TobitResult".into(),
        summary: format!("Tobit(k={}, n={}), logLik={:.4}", r.params.len(), r.n_obs, r.log_likelihood),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_poisson(r: &std::rc::Rc<greeners::PoissonResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));
    fit.insert("deviance".into(), Value::Float(r.deviance));

    ModelView {
        type_name: "PoissonResult".into(),
        summary: format!("Poisson(k={}, n={}), pseudo-R2={:.4}", r.params.len(), r.n_obs, r.pseudo_r2),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.z_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: Some(r.conf_lower.clone()),
        conf_upper: Some(r.conf_upper.clone()),
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_negbin(r: &std::rc::Rc<greeners::NegBinResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));
    fit.insert("alpha".into(), Value::Float(r.alpha));

    ModelView {
        type_name: "NegBinResult".into(),
        summary: format!("Negative Binomial(k={}, n={}), alpha={:.4}", r.params.len(), r.n_obs, r.alpha),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.z_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: Some(r.conf_lower.clone()),
        conf_upper: Some(r.conf_upper.clone()),
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_glm(r: &std::rc::Rc<greeners::GlmResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));
    fit.insert("deviance".into(), Value::Float(r.deviance));
    fit.insert("dispersion".into(), Value::Float(r.dispersion));

    ModelView {
        type_name: "GlmResult".into(),
        summary: format!("GLM(k={}, n={}), pseudo-R2={:.4}", r.params.len(), r.n_obs, r.pseudo_r2),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.z_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: Some(r.conf_lower.clone()),
        conf_upper: Some(r.conf_upper.clone()),
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_rlm(r: &std::rc::Rc<greeners::RlmResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("scale".into(), Value::Float(r.scale));
    fit.insert("n_iter".into(), Value::Int(r.n_iter as i64));

    ModelView {
        type_name: "RlmResult".into(),
        summary: format!("Robust Linear Model(k={}, n={})", r.params.len(), r.n_obs),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: Some(r.conf_lower.clone()),
        conf_upper: Some(r.conf_upper.clone()),
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_beta(r: &std::rc::Rc<greeners::BetaResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));
    fit.insert("precision_param".into(), Value::Float(r.precision_param));

    ModelView {
        type_name: "BetaResult".into(),
        summary: format!("Beta Regression(k={}, n={}), pseudo-R2={:.4}", r.params.len(), r.n_obs, r.pseudo_r2),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.z_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_gmm(r: &std::rc::Rc<greeners::GmmResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("j_stat".into(), Value::Float(r.j_stat));
    fit.insert("j_p_value".into(), Value::Float(r.j_p_value));
    fit.insert("df_overid".into(), Value::Int(r.df_overid as i64));

    ModelView {
        type_name: "GmmResult".into(),
        summary: format!("GMM(k={}, n={}), J={:.4}", r.params.len(), r.n_obs, r.j_stat),
        variable_names: (0..r.params.len()).map(|i| format!("x{i}")).collect(),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

fn model_view_from_model_result(
    _display: &str,
    summary: &str,
    type_name: &str,
    fields: &std::sync::Arc<HashMap<String, Value>>,
) -> ModelView {
    // Heuristic: try to extract common fields from the generic ModelResult.
    let empty = Array1::zeros(0);
    let params = match fields.get("params") {
        Some(Value::List(v)) => v.iter().map(|x| match x {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            _ => f64::NAN,
        }).collect::<Vec<_>>().into(),
        _ => empty.clone(),
    };
    let names = match fields.get("variable_names") {
        Some(Value::List(v)) => v.iter().map(|x| match x {
            Value::Str(s) => s.clone(),
            _ => "?".into(),
        }).collect(),
        _ => (0..params.len()).map(|i| format!("x{i}")).collect(),
    };

    ModelView {
        type_name: type_name.to_string(),
        summary: summary.to_string(),
        variable_names: names,
        params,
        std_errors: Array1::zeros(0),
        test_values: Array1::zeros(0),
        p_values: Array1::zeros(0),
        conf_lower: None,
        conf_upper: None,
        fit: fields.as_ref().clone(),
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
