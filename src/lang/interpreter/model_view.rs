use super::{Series, Value};
use indexmap::IndexMap;
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
            Some(Value::Series(s)) => s
                .values
                .iter()
                .map(|v| match v {
                    Value::Float(f) => Some(*f),
                    Value::Int(i) => Some(*i as f64),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>(),
            _ => None,
        }
    }

    /// Build the `tidy()` output map: variable, coef, std_err, t, p_value,
    /// conf_low, conf_high.
    pub fn to_tidy_map(&self) -> HashMap<String, Value> {
        let n = self.params.len();
        let names: Vec<Value> = (0..n)
            .map(|i| {
                Value::Str(
                    self.variable_names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("x{i}")),
                )
            })
            .collect();
        let coefs: Vec<Value> = self.params.iter().map(|&v| Value::Float(v)).collect();
        let ses: Vec<Value> = self.std_errors.iter().map(|&v| Value::Float(v)).collect();
        let tests: Vec<Value> = self.test_values.iter().map(|&v| Value::Float(v)).collect();
        let ps: Vec<Value> = self.p_values.iter().map(|&v| Value::Float(v)).collect();

        let (cl, cu): (Vec<Value>, Vec<Value>) = match (&self.conf_lower, &self.conf_upper) {
            (Some(l), Some(u)) if l.len() == n && u.len() == n => (
                l.iter().map(|&v| Value::Float(v)).collect(),
                u.iter().map(|&v| Value::Float(v)).collect(),
            ),
            _ => {
                let nan = vec![Value::Float(f64::NAN); n];
                (nan.clone(), nan)
            }
        };

        let mut map = HashMap::new();
        map.insert("variable".into(), Value::List(Arc::new(names)));
        map.insert("coef".into(), Value::List(Arc::new(coefs)));
        map.insert("std_err".into(), Value::List(Arc::new(ses)));
        map.insert("t".into(), Value::List(Arc::new(tests)));
        map.insert("p_value".into(), Value::List(Arc::new(ps)));
        map.insert("conf_low".into(), Value::List(Arc::new(cl)));
        map.insert("conf_high".into(), Value::List(Arc::new(cu)));
        map
    }

    /// Build coefficient rows for `esttab` and similar table builders.
    /// Tuple is `(variable, coef, std_err, p_value)`.
    pub fn to_coef_rows(&self) -> Vec<(String, f64, Option<f64>, Option<f64>)> {
        self.variable_names
            .iter()
            .zip(self.params.iter())
            .zip(self.std_errors.iter())
            .zip(self.p_values.iter())
            .map(|(((n, &c), &s), &p)| (n.clone(), c, Some(s), Some(p)))
            .collect()
    }

    /// Build the `glance()` output: a copy of `fit` as a one-row Dict,
    /// wrapping each scalar in a single-element `Value::List` so that
    /// `dict_to_dataframe` can materialise it.
    pub fn to_glance_map(&self) -> HashMap<String, Value> {
        self.fit
            .iter()
            .map(|(k, v)| {
                let wrapped = match v {
                    Value::List(_) | Value::Series(_) => v.clone(),
                    _ => Value::List(Arc::new(vec![v.clone()])),
                };
                (k.clone(), wrapped)
            })
            .collect()
    }

    /// Export coefficients as CSV (tidy format).
    pub fn to_csv(&self) -> String {
        let mut out = String::from("variable,coef,std_err,t,p_value,conf_low,conf_high\n");
        let n = self.params.len();
        let se = &self.std_errors;
        let t = &self.test_values;
        let p = &self.p_values;
        let cl = self.conf_lower.as_ref();
        let cu = self.conf_upper.as_ref();
        for i in 0..n {
            let name = self
                .variable_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{i}"));
            out.push_str(&format!(
                "{},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7}\n",
                name,
                self.params[i],
                se.get(i).copied().unwrap_or(f64::NAN),
                t.get(i).copied().unwrap_or(f64::NAN),
                p.get(i).copied().unwrap_or(f64::NAN),
                cl.map(|v| v[i]).unwrap_or(f64::NAN),
                cu.map(|v| v[i]).unwrap_or(f64::NAN),
            ));
        }
        out
    }

    fn fmt_row(&self, i: usize, use_html: bool) -> String {
        let name = self
            .variable_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("x{i}"));
        let c = self.params[i];
        let se = self.std_errors.get(i).copied().unwrap_or(f64::NAN);
        let p = self.p_values.get(i).copied().unwrap_or(1.0);
        let stars = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        if use_html {
            format!(
                "<tr><td>{}</td><td>{:.4} {}</td><td>({:.4})</td><td>{:.4}</td></tr>\n",
                name, c, stars, se, p
            )
        } else {
            format!(
                "{} & {:.4} {} & ({:.4}) & {:.4} \\\\\n",
                name, c, stars, se, p
            )
        }
    }

    /// Export coefficients as a LaTeX table fragment.
    pub fn to_latex(&self) -> String {
        let mut out = String::new();
        out.push_str("\\begin{tabular}{lccc}\n");
        out.push_str("\\hline\n");
        out.push_str("Variable & Coef. & Std. Err. & p-value \\\\\n");
        out.push_str("\\hline\n");
        for i in 0..self.params.len() {
            out.push_str(&self.fmt_row(i, false));
        }
        out.push_str("\\hline\n");
        out.push_str("\\end{tabular}\n");
        out
    }

    /// Export coefficients as an HTML table fragment.
    pub fn to_html(&self) -> String {
        let mut out = String::new();
        out.push_str("<table>\n");
        out.push_str(
            "<tr><th>Variable</th><th>Coef.</th><th>Std. Err.</th><th>p-value</th></tr>\n",
        );
        for i in 0..self.params.len() {
            out.push_str(&self.fmt_row(i, true));
        }
        out.push_str("</table>\n");
        out
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
            #[cfg(feature = "greeners-ols")]
            Value::IvResult(r) => Some(model_view_from_iv(r)),
            #[cfg(feature = "greeners-panel")]
            Value::PanelResult(r) => Some(model_view_from_panel(r)),
            #[cfg(feature = "greeners-panel")]
            Value::ReResult(r) => Some(model_view_from_random_effects(r)),
            #[cfg(feature = "greeners-ols")]
            Value::QuantileResult(r) => Some(model_view_from_quantile(r)),
            #[cfg(feature = "greeners-ols")]
            Value::TobitResult(r) => Some(model_view_from_tobit(r)),
            #[cfg(feature = "greeners-glm")]
            Value::PoissonResult(r) => Some(model_view_from_poisson(r)),
            #[cfg(feature = "greeners-glm")]
            Value::NegBinResult(r) => Some(model_view_from_negbin(r)),
            #[cfg(feature = "greeners-glm")]
            Value::GlmResult(r) => Some(model_view_from_glm(r)),
            #[cfg(feature = "greeners-ols")]
            Value::RlmResult(r) => Some(model_view_from_rlm(r)),
            #[cfg(feature = "greeners-glm")]
            Value::BetaResult(r) => Some(model_view_from_beta(r)),
            #[cfg(feature = "greeners-ols")]
            Value::GmmResult(r) => Some(model_view_from_gmm(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::GarchResult(r) => Some(model_view_from_garch(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::AutoRegResult(r) => Some(model_view_from_autoreg(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::ArdlResult(r) => Some(model_view_from_ardl(r)),
            #[cfg(feature = "greeners-ols")]
            Value::GlsarResult(r) => Some(model_view_from_glsar(r)),
            #[cfg(feature = "greeners-glm")]
            Value::OrderedResult(r) => Some(model_view_from_ordered(r)),
            #[cfg(feature = "greeners-survival")]
            Value::CoxResult(r) => Some(model_view_from_cox(r)),
            #[cfg(feature = "greeners-glm")]
            Value::GeeResult(r) => Some(model_view_from_gee(r)),
            #[cfg(feature = "greeners-bayesian")]
            Value::MixedResult(r) => Some(model_view_from_mixed(r)),
            #[cfg(feature = "greeners-glm")]
            Value::ZeroInflatedResult(r) => Some(model_view_from_zero_inflated(r)),
            #[cfg(feature = "greeners-panel")]
            Value::ThresholdResult(r) => Some(model_view_from_threshold(r)),
            #[cfg(feature = "greeners-causal")]
            Value::DidResult(r) => Some(model_view_from_did(r)),
            #[cfg(feature = "greeners-causal")]
            Value::RdResult(r) => Some(model_view_from_rd(r)),
            #[cfg(feature = "greeners-ols")]
            Value::RecursiveLSResult(r) => Some(model_view_from_recursive_ls(r)),
            #[cfg(feature = "greeners-glm")]
            Value::ConditionalResult(r) => Some(model_view_from_conditional(r)),
            #[cfg(feature = "greeners-glm")]
            Value::GamResult(r) => Some(model_view_from_gam(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::EtsResult(r) => Some(model_view_from_ets(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::MarkovResult(r) => Some(model_view_from_markov_switching(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::MSARResult(r) => Some(model_view_from_markov_autoreg(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::VarResult(r) => Some(model_view_from_var(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::VecmResult(r) => Some(model_view_from_vecm(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::SVarResult(r) => Some(model_view_from_svar(r)),
            #[cfg(feature = "greeners-timeseries")]
            Value::VarmaResult(r) => Some(model_view_from_varma(r)),
            #[cfg(feature = "greeners-panel")]
            Value::AbResult(r) => Some(model_view_from_ab(r)),
            #[cfg(feature = "greeners-panel")]
            Value::SysGmmResult(r) => Some(model_view_from_sys_gmm(r)),
            #[cfg(feature = "greeners-panel")]
            Value::PcseResult(r) => Some(model_view_from_pcse(r)),
            #[cfg(feature = "greeners-panel")]
            Value::PanelGlsResult(r) => Some(model_view_from_panel_gls(r)),
            #[cfg(feature = "greeners-ols")]
            Value::RollingResult(r) => Some(model_view_from_rolling(r)),
            Value::ModelResult {
                display,
                summary,
                type_name,
                variable_names,
                params,
                std_errors,
                test_values,
                p_values,
                conf_lower,
                conf_upper,
                fit,
                residuals,
                fitted_values,
                x,
                extras,
                fields: _,
            } => Some(ModelView::from_model_result_fields(
                display.clone(),
                summary.clone(),
                type_name,
                variable_names.clone(),
                params.clone(),
                std_errors.clone(),
                test_values.clone(),
                p_values.clone(),
                conf_lower.clone(),
                conf_upper.clone(),
                fit.clone(),
                residuals.clone(),
                fitted_values.clone(),
                x.clone(),
                extras.clone(),
            )),
            Value::Model(m) => Some(m.to_model_view()),
            _ => None,
        }
    }
}

fn names_or_x(n: Option<&Vec<String>>, len: usize) -> Vec<String> {
    n.cloned()
        .unwrap_or_else(|| (0..len).map(|i| format!("x{i}")).collect())
}

/// Create a ModelView directly from ModelResult fields.
/// This is the unified constructor that replaces 50+ individual `model_view_from_*` functions.
impl ModelView {
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_result_fields(
        _display: String,
        summary: String,
        type_name: &'static str,
        variable_names: Vec<String>,
        params: Option<Array1<f64>>,
        std_errors: Option<Array1<f64>>,
        test_values: Option<Array1<f64>>,
        p_values: Option<Array1<f64>>,
        conf_lower: Option<Array1<f64>>,
        conf_upper: Option<Array1<f64>>,
        fit: HashMap<String, Value>,
        residuals: Option<Array1<f64>>,
        fitted_values: Option<Array1<f64>>,
        x: Option<Array2<f64>>,
        extras: HashMap<String, Value>,
    ) -> ModelView {
        ModelView {
            type_name: type_name.to_string(),
            summary,
            variable_names,
            params: params.unwrap_or_else(|| Array1::zeros(0)),
            std_errors: std_errors.unwrap_or_else(|| Array1::zeros(0)),
            test_values: test_values.unwrap_or_else(|| Array1::zeros(0)),
            p_values: p_values.unwrap_or_else(|| Array1::zeros(0)),
            conf_lower,
            conf_upper,
            fit,
            residuals,
            fitted_values,
            x,
            extras,
        }
    }
}

// ── Rc<GreenersResult> converters ───────────────────────────────────────────
#[cfg(feature = "greeners-ols")]
#[cfg(feature = "greeners-ols")]
fn model_view_from_iv(r: &std::rc::Rc<greeners::iv::IvResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("df_resid".into(), Value::Int(r.df_resid as i64));
    fit.insert("r_squared".into(), Value::Float(r.r_squared));
    fit.insert("sigma".into(), Value::Float(r.sigma));

    ModelView {
        type_name: "IvResult".into(),
        summary: format!(
            "IV/2SLS(k={}, n={}), R2={:.4}",
            r.params.len(),
            r.n_obs,
            r.r_squared
        ),
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
#[cfg(feature = "greeners-panel")]
#[cfg(feature = "greeners-panel")]
fn model_view_from_panel(r: &std::rc::Rc<greeners::panel::PanelResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_entities".into(), Value::Int(r.n_entities as i64));
    fit.insert("r_squared".into(), Value::Float(r.r_squared));
    fit.insert("sigma".into(), Value::Float(r.sigma));

    ModelView {
        type_name: "PanelResult".into(),
        summary: format!(
            "Fixed Effects(k={}, n={}, panels={}), R2={:.4}",
            r.params.len(),
            r.n_obs,
            r.n_entities,
            r.r_squared
        ),
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
#[cfg(feature = "greeners-panel")]
#[cfg(feature = "greeners-panel")]
fn model_view_from_random_effects(
    r: &std::rc::Rc<greeners::panel::RandomEffectsResult>,
) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert(
        "r_squared_overall".into(),
        Value::Float(r.r_squared_overall),
    );
    fit.insert("sigma_u".into(), Value::Float(r.sigma_u));
    fit.insert("sigma_e".into(), Value::Float(r.sigma_e));
    fit.insert("theta".into(), Value::Float(r.theta));

    ModelView {
        type_name: "ReResult".into(),
        summary: format!(
            "Random Effects(k={}), R2={:.4}",
            r.params.len(),
            r.r_squared_overall
        ),
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
#[cfg(feature = "greeners-ols")]
#[cfg(feature = "greeners-ols")]
fn model_view_from_quantile(r: &std::rc::Rc<greeners::quantile::QuantileResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("tau".into(), Value::Float(r.tau));
    fit.insert("r_squared".into(), Value::Float(r.r_squared));
    fit.insert("iterations".into(), Value::Int(r.iterations as i64));

    ModelView {
        type_name: "QuantileResult".into(),
        summary: format!(
            "Quantile Regression(tau={:.2}, k={})",
            r.tau,
            r.params.len()
        ),
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
#[cfg(feature = "greeners-ols")]
#[cfg(feature = "greeners-ols")]
fn model_view_from_tobit(r: &std::rc::Rc<greeners::tobit::TobitResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));

    ModelView {
        type_name: "TobitResult".into(),
        summary: format!(
            "Tobit(k={}, n={}), logLik={:.4}",
            r.params.len(),
            r.n_obs,
            r.log_likelihood
        ),
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
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_poisson(r: &std::rc::Rc<greeners::poisson::PoissonResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));
    fit.insert("deviance".into(), Value::Float(r.deviance));

    ModelView {
        type_name: "PoissonResult".into(),
        summary: format!(
            "Poisson(k={}, n={}), pseudo-R2={:.4}",
            r.params.len(),
            r.n_obs,
            r.pseudo_r2
        ),
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
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_negbin(r: &std::rc::Rc<greeners::negbin::NegBinResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));
    fit.insert("alpha".into(), Value::Float(r.alpha));

    ModelView {
        type_name: "NegBinResult".into(),
        summary: format!(
            "Negative Binomial(k={}, n={}), alpha={:.4}",
            r.params.len(),
            r.n_obs,
            r.alpha
        ),
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
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_glm(r: &std::rc::Rc<greeners::glm::GlmResult>) -> ModelView {
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
        summary: format!(
            "GLM(k={}, n={}), pseudo-R2={:.4}",
            r.params.len(),
            r.n_obs,
            r.pseudo_r2
        ),
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
#[cfg(feature = "greeners-ols")]
#[cfg(feature = "greeners-ols")]
fn model_view_from_rlm(r: &std::rc::Rc<greeners::rlm::RlmResult>) -> ModelView {
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
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_beta(r: &std::rc::Rc<greeners::beta_model::BetaResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));
    fit.insert("precision_param".into(), Value::Float(r.precision_param));

    ModelView {
        type_name: "BetaResult".into(),
        summary: format!(
            "Beta Regression(k={}, n={}), pseudo-R2={:.4}",
            r.params.len(),
            r.n_obs,
            r.pseudo_r2
        ),
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
#[cfg(feature = "greeners-ols")]
#[cfg(feature = "greeners-ols")]
fn model_view_from_gmm(r: &std::rc::Rc<greeners::gmm::GmmResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("j_stat".into(), Value::Float(r.j_stat));
    fit.insert("j_p_value".into(), Value::Float(r.j_p_value));
    fit.insert("df_overid".into(), Value::Int(r.df_overid as i64));

    ModelView {
        type_name: "GmmResult".into(),
        summary: format!(
            "GMM(k={}, n={}), J={:.4}",
            r.params.len(),
            r.n_obs,
            r.j_stat
        ),
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
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_garch(r: &std::rc::Rc<greeners::garch::GarchResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("p".into(), Value::Int(r.p as i64));
    fit.insert("q".into(), Value::Int(r.q as i64));

    let mut extras = HashMap::new();
    extras.insert(
        "model_type".into(),
        Value::Str(format!("{:?}", r.model_type)),
    );
    extras.insert("dist".into(), Value::Str(format!("{:?}", r.dist)));

    ModelView {
        type_name: "GarchResult".into(),
        summary: format!(
            "{}({}, p={}, q={}), n={}",
            r.model_type, r.dist, r.p, r.q, r.n_obs
        ),
        variable_names: r.variable_names.clone(),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.z_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: Some(r.conf_lower.clone()),
        conf_upper: Some(r.conf_upper.clone()),
        fit,
        residuals: Some(r.residuals.clone()),
        fitted_values: None,
        x: None,
        extras,
    }
}
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_autoreg(r: &std::rc::Rc<greeners::autoreg::AutoRegResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("r2".into(), Value::Float(r.r_squared));
    fit.insert("adj_r2".into(), Value::Float(r.adj_r_squared));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("lags".into(), Value::Int(r.lags as i64));

    let mut extras = HashMap::new();
    extras.insert("trend".into(), Value::Str(r.trend.clone()));

    ModelView {
        type_name: "AutoRegResult".into(),
        summary: format!("AR(lags={}, n={}), R2={:.4}", r.lags, r.n_obs, r.r_squared),
        variable_names: r.param_names.clone(),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: Some(r.residuals.clone()),
        fitted_values: Some(r.fitted_values.clone()),
        x: None,
        extras,
    }
}
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_ardl(r: &std::rc::Rc<greeners::autoreg::ARDLResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("r2".into(), Value::Float(r.r_squared));
    fit.insert("adj_r2".into(), Value::Float(r.adj_r_squared));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("y_lags".into(), Value::Int(r.y_lags as i64));
    fit.insert("x_lags".into(), Value::Int(r.x_lags as i64));

    ModelView {
        type_name: "ArdlResult".into(),
        summary: format!(
            "ARDL(y_lags={}, x_lags={}, n={}), R2={:.4}",
            r.y_lags, r.x_lags, r.n_obs, r.r_squared
        ),
        variable_names: r.param_names.clone(),
        params: r.params.clone(),
        std_errors: r.std_errors.clone(),
        test_values: r.t_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: Some(r.residuals.clone()),
        fitted_values: Some(r.fitted_values.clone()),
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-ols")]
#[cfg(feature = "greeners-ols")]
fn model_view_from_glsar(r: &std::rc::Rc<greeners::glsar::GlsarResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("df_resid".into(), Value::Int(r.df_resid as i64));
    fit.insert("r2".into(), Value::Float(r.r_squared));
    fit.insert(
        "rho".into(),
        Value::List(Arc::new(r.rho.iter().map(|&v| Value::Float(v)).collect())),
    );

    ModelView {
        type_name: "GlsarResult".into(),
        summary: format!(
            "GLS-AR(rho_len={}, n={}), R2={:.4}",
            r.rho.len(),
            r.n_obs,
            r.r_squared
        ),
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
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_ordered(r: &std::rc::Rc<greeners::ordered::OrderedResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));
    fit.insert("n_categories".into(), Value::Int(r.n_categories as i64));

    let mut names = r.variable_names.clone().unwrap_or_default();
    for i in 0..r.thresholds.len() {
        names.push(format!("_cut{}", i + 1));
    }
    let mut params = r.params.to_vec();
    params.extend(&r.thresholds);
    let mut se = r.std_errors.to_vec();
    se.extend(&r.threshold_std_errors);

    ModelView {
        type_name: "OrderedResult".into(),
        summary: format!(
            "{}(k={}, n={}), pseudo-R2={:.4}",
            r.model_name,
            r.params.len(),
            r.n_obs,
            r.pseudo_r2
        ),
        variable_names: names,
        params: Array1::from(params),
        std_errors: Array1::from(se),
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
#[cfg(feature = "greeners-survival")]
#[cfg(feature = "greeners-survival")]
fn model_view_from_cox(r: &std::rc::Rc<greeners::survival::CoxResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_events".into(), Value::Int(r.n_events as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("concordance".into(), Value::Float(r.concordance));

    ModelView {
        type_name: "CoxResult".into(),
        summary: format!(
            "Cox PH(k={}, n={}, events={}), C={:.4}",
            r.params.len(),
            r.n_obs,
            r.n_events,
            r.concordance
        ),
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
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_gee(r: &std::rc::Rc<greeners::gee::GeeResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_groups".into(), Value::Int(r.n_groups as i64));
    fit.insert("scale".into(), Value::Float(r.scale));
    fit.insert("qic".into(), Value::Float(r.qic));

    ModelView {
        type_name: "GeeResult".into(),
        summary: format!(
            "GEE(k={}, n={}, groups={})",
            r.params.len(),
            r.n_obs,
            r.n_groups
        ),
        variable_names: names_or_x(r.variable_names.as_ref(), r.params.len()),
        params: r.params.clone(),
        std_errors: r.robust_se.clone(),
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
#[cfg(feature = "greeners-bayesian")]
#[cfg(feature = "greeners-bayesian")]
fn model_view_from_mixed(r: &std::rc::Rc<greeners::mixed::MixedResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_groups".into(), Value::Int(r.n_groups as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("var_resid".into(), Value::Float(r.var_resid));

    let mut extras = HashMap::new();
    extras.insert(
        "random_effects".into(),
        Value::Dict(Arc::new(
            r.random_effects
                .iter()
                .map(|(k, v)| {
                    (
                        k.to_string(),
                        Value::List(Arc::new(v.iter().map(|&x| Value::Float(x)).collect())),
                    )
                })
                .collect(),
        )),
    );

    ModelView {
        type_name: "MixedResult".into(),
        summary: format!(
            "Mixed LM(k={}, n={}, groups={})",
            r.fixed_effects.len(),
            r.n_obs,
            r.n_groups
        ),
        variable_names: names_or_x(r.variable_names.as_ref(), r.fixed_effects.len()),
        params: r.fixed_effects.clone(),
        std_errors: r.fixed_se.clone(),
        test_values: r.z_values.clone(),
        p_values: r.p_values.clone(),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_zero_inflated(
    r: &std::rc::Rc<greeners::zero_inflated::ZeroInflatedResult>,
) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    if let Some(alpha) = r.alpha {
        fit.insert("alpha".into(), Value::Float(alpha));
    }

    let mut names = Vec::new();
    for i in 0..r.count_params.len() {
        names.push(format!("count_x{i}"));
    }
    for i in 0..r.inflate_params.len() {
        names.push(format!("inflate_x{i}"));
    }
    let mut params = r.count_params.to_vec();
    params.extend(r.inflate_params.iter());
    let mut se = r.count_std_errors.to_vec();
    se.extend(r.inflate_std_errors.iter());
    let mut z = r.count_z_values.to_vec();
    z.extend(r.inflate_z_values.iter());
    let mut p = r.count_p_values.to_vec();
    p.extend(r.inflate_p_values.iter());

    ModelView {
        type_name: "ZeroInflatedResult".into(),
        summary: format!(
            "{}(count={}, inflate={}, n={})",
            r.model_name,
            r.count_params.len(),
            r.inflate_params.len(),
            r.n_obs
        ),
        variable_names: names,
        params: Array1::from(params),
        std_errors: Array1::from(se),
        test_values: Array1::from(z),
        p_values: Array1::from(p),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-panel")]
#[cfg(feature = "greeners-panel")]
fn model_view_from_threshold(r: &std::rc::Rc<greeners::threshold::ThresholdResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_search".into(), Value::Int(r.n_search as i64));
    fit.insert("r2".into(), Value::Float(r.r_squared));
    fit.insert("ssr_min".into(), Value::Float(r.ssr_min));

    let mut params = r.params_regime1.to_vec();
    params.extend(r.params_regime2.iter());
    params.push(r.threshold_gamma);
    let n = params.len();
    let names: Vec<String> = (0..r.params_regime1.len())
        .map(|i| format!("regime1_x{i}"))
        .chain((0..r.params_regime2.len()).map(|i| format!("regime2_x{i}")))
        .chain(std::iter::once("threshold".into()))
        .collect();

    ModelView {
        type_name: "ThresholdResult".into(),
        summary: format!(
            "Threshold(gamma={:.4}, n_search={}), R2={:.4}",
            r.threshold_gamma, r.n_search, r.r_squared
        ),
        variable_names: names,
        params: Array1::from(params),
        std_errors: Array1::zeros(n),
        test_values: Array1::zeros(n),
        p_values: Array1::ones(n),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-causal")]
#[cfg(feature = "greeners-causal")]
fn model_view_from_did(r: &std::rc::Rc<greeners::did::DidResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("r2".into(), Value::Float(r.r_squared));
    fit.insert("att".into(), Value::Float(r.att));
    fit.insert("control_pre_mean".into(), Value::Float(r.control_pre_mean));
    fit.insert(
        "control_post_mean".into(),
        Value::Float(r.control_post_mean),
    );
    fit.insert("treated_pre_mean".into(), Value::Float(r.treated_pre_mean));
    fit.insert(
        "treated_post_mean".into(),
        Value::Float(r.treated_post_mean),
    );

    ModelView {
        type_name: "DidResult".into(),
        summary: format!("DiD(ATT={:.4}, n={})", r.att, r.n_obs),
        variable_names: r.variable_names.clone(),
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
#[cfg(feature = "greeners-causal")]
#[cfg(feature = "greeners-causal")]
fn model_view_from_rd(r: &std::rc::Rc<greeners::rd::RdResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_left".into(), Value::Int(r.n_left as i64));
    fit.insert("n_right".into(), Value::Int(r.n_right as i64));
    fit.insert("n_total".into(), Value::Int(r.n_total as i64));
    fit.insert("bandwidth".into(), Value::Float(r.bandwidth));
    fit.insert("poly_order".into(), Value::Int(r.poly_order as i64));
    fit.insert("cutoff".into(), Value::Float(r.cutoff));

    let mut extras = HashMap::new();
    extras.insert("is_fuzzy".into(), Value::Int(r.is_fuzzy as i64));
    if let Some(t) = r.first_stage_tau {
        extras.insert("first_stage_tau".into(), Value::Float(t));
    }
    if let Some(se) = r.first_stage_se {
        extras.insert("first_stage_se".into(), Value::Float(se));
    }

    let params = Array1::from_vec(vec![r.tau]);
    let se = Array1::from_vec(vec![r.se]);
    let t = Array1::from_vec(vec![r.z]);
    let p = Array1::from_vec(vec![r.p_value]);

    ModelView {
        type_name: "RdResult".into(),
        summary: format!(
            "RD(tau={:.4}, n={}), bw={:.4}",
            r.tau, r.n_total, r.bandwidth
        ),
        variable_names: vec!["tau".into()],
        params,
        std_errors: se,
        test_values: t,
        p_values: p,
        conf_lower: Some(Array1::from_vec(vec![r.ci_lower])),
        conf_upper: Some(Array1::from_vec(vec![r.ci_upper])),
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}
#[cfg(feature = "greeners-ols")]
#[cfg(feature = "greeners-ols")]
fn model_view_from_recursive_ls(
    r: &std::rc::Rc<greeners::rolling::RecursiveLSResult>,
) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));

    ModelView {
        type_name: "RecursiveLSResult".into(),
        summary: format!("Recursive LS(k={}, n={})", r.params.len(), r.n_obs),
        variable_names: (0..r.params.len()).map(|i| format!("x{i}")).collect(),
        params: r.params.clone(),
        std_errors: Array1::zeros(r.params.len()),
        test_values: Array1::zeros(r.params.len()),
        p_values: Array1::ones(r.params.len()),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: Some(r.residuals.clone()),
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_conditional(
    r: &std::rc::Rc<greeners::conditional::ConditionalResult>,
) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_groups".into(), Value::Int(r.n_groups as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("iterations".into(), Value::Int(r.iterations as i64));

    ModelView {
        type_name: "ConditionalResult".into(),
        summary: format!(
            "{}(k={}, n={}, groups={})",
            r.model_name,
            r.params.len(),
            r.n_obs,
            r.n_groups
        ),
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
#[cfg(feature = "greeners-glm")]
#[cfg(feature = "greeners-glm")]
fn model_view_from_gam(r: &std::rc::Rc<greeners::glmgam::GamResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_linear".into(), Value::Int(r.n_linear as i64));
    fit.insert("n_smooth".into(), Value::Int(r.n_smooth as i64));
    fit.insert("edf".into(), Value::Float(r.edf));
    fit.insert("gcv_score".into(), Value::Float(r.gcv_score));
    fit.insert("scale".into(), Value::Float(r.scale));

    ModelView {
        type_name: "GamResult".into(),
        summary: format!(
            "GAM(k={}, n={}), GCV={:.4}",
            r.params.len(),
            r.n_obs,
            r.gcv_score
        ),
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
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_ets(r: &std::rc::Rc<greeners::ets::ETSResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("sse".into(), Value::Float(r.sse));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));
    fit.insert("alpha".into(), Value::Float(r.alpha));
    fit.insert("phi".into(), Value::Float(r.phi.unwrap_or(f64::NAN)));
    fit.insert(
        "seasonal_periods".into(),
        Value::Int(r.seasonal_periods as i64),
    );
    fit.insert("damped".into(), Value::Int(r.damped as i64));

    let mut extras = HashMap::new();
    extras.insert("trend_type".into(), Value::Str(r.trend_type.clone()));
    extras.insert("seasonal_type".into(), Value::Str(r.seasonal_type.clone()));

    ModelView {
        type_name: "EtsResult".into(),
        summary: format!(
            "ETS({}, trend={}, n={})",
            r.seasonal_type, r.trend_type, r.n_obs
        ),
        variable_names: (0..3).map(|i| format!("comp{i}")).collect(),
        params: Array1::zeros(3),
        std_errors: Array1::zeros(3),
        test_values: Array1::zeros(3),
        p_values: Array1::ones(3),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: Some(r.residuals.clone()),
        fitted_values: Some(r.fitted_values.clone()),
        x: None,
        extras,
    }
}
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_markov_switching(
    r: &std::rc::Rc<greeners::markov::MarkovSwitchingResult>,
) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_regimes".into(), Value::Int(r.n_regimes as i64));
    fit.insert("ar_order".into(), Value::Int(r.ar_order as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));

    let mut params = Vec::new();
    let mut names = Vec::new();
    for (i, rp) in r.regime_params.iter().enumerate() {
        for (j, &v) in rp.iter().enumerate() {
            params.push(v);
            names.push(format!("regime{}_x{}", i + 1, j));
        }
    }
    let n = params.len();

    ModelView {
        type_name: "MarkovSwitchingResult".into(),
        summary: format!(
            "Markov Switching(regimes={}, ar={}, n={})",
            r.n_regimes, r.ar_order, r.n_obs
        ),
        variable_names: names,
        params: Array1::from(params),
        std_errors: Array1::zeros(n),
        test_values: Array1::zeros(n),
        p_values: Array1::ones(n),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_markov_autoreg(
    r: &std::rc::Rc<greeners::markov_autoreg::MarkovAutoregResult>,
) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_regimes".into(), Value::Int(r.k_regimes as i64));
    fit.insert("ar_order".into(), Value::Int(r.ar_order as i64));
    fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));

    let mut params = Vec::new();
    let mut names = Vec::new();
    for i in 0..r.k_regimes {
        params.push(r.regime_means[i]);
        names.push(format!("regime{}_intercept", i + 1));
        for p in 0..r.ar_order {
            params.push(r.ar_params[[i, p]]);
            names.push(format!("regime{}_ar{}", i + 1, p + 1));
        }
        params.push(r.regime_sigmas[i]);
        names.push(format!("regime{}_sigma", i + 1));
    }
    let n = params.len();

    ModelView {
        type_name: "MSARResult".into(),
        summary: format!(
            "MSAR(regimes={}, ar={}, n={})",
            r.k_regimes, r.ar_order, r.n_obs
        ),
        variable_names: names,
        params: Array1::from(params),
        std_errors: Array1::zeros(n),
        test_values: Array1::zeros(n),
        p_values: Array1::ones(n),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-timeseries")]
fn flatten_var_params(r: &greeners::var::VarResult) -> (Vec<String>, Array1<f64>, Array1<f64>) {
    let k = r.n_vars;
    let p = r.lags;
    let n_coef = (1 + p * k) * k;
    let mut params = Array1::<f64>::zeros(n_coef);
    let mut ses = Array1::<f64>::zeros(n_coef);
    let mut names: Vec<String> = Vec::with_capacity(n_coef);
    let mut idx = 0;
    for eq in 0..k {
        for row in 0..(1 + p * k) {
            params[idx] = r.params[(row, eq)];
            ses[idx] = r.std_errors[(row, eq)];
            names.push(if row == 0 {
                format!("const_{}", r.var_names[eq])
            } else {
                let lag = (row - 1) / k;
                let src = (row - 1) % k;
                format!("L{}.{}/{}", lag + 1, r.var_names[src], r.var_names[eq])
            });
            idx += 1;
        }
    }
    (names, params, ses)
}
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_var(r: &std::rc::Rc<greeners::var::VarResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_vars".into(), Value::Int(r.n_vars as i64));
    fit.insert("lags".into(), Value::Int(r.lags as i64));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));

    let (names, params, std_errors) = flatten_var_params(r);
    let n = params.len();

    ModelView {
        type_name: "VarResult".into(),
        summary: format!("VAR(vars={}, lags={}, n={})", r.n_vars, r.lags, r.n_obs),
        variable_names: names,
        params,
        std_errors,
        test_values: Array1::zeros(n),
        p_values: Array1::ones(n),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_svar(r: &std::rc::Rc<greeners::svar::SVarResult>) -> ModelView {
    let mut fit = HashMap::new();
    let vr = &r.var_result;
    fit.insert("n_obs".into(), Value::Int(vr.n_obs as i64));
    fit.insert("n_vars".into(), Value::Int(vr.n_vars as i64));
    fit.insert("lags".into(), Value::Int(vr.lags as i64));
    fit.insert("aic".into(), Value::Float(vr.aic));
    fit.insert("bic".into(), Value::Float(vr.bic));

    let mut extras = HashMap::new();
    extras.insert(
        "identification".into(),
        Value::Str(r.identification.clone()),
    );

    let (names, params, std_errors) = flatten_var_params(vr);
    let n = params.len();

    ModelView {
        type_name: "SVarResult".into(),
        summary: format!(
            "SVAR(vars={}, lags={}, n={}), id={}",
            vr.n_vars, vr.lags, vr.n_obs, r.identification
        ),
        variable_names: names,
        params,
        std_errors,
        test_values: Array1::zeros(n),
        p_values: Array1::ones(n),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras,
    }
}
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_varma(r: &std::rc::Rc<greeners::varma::VarmaResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_vars".into(), Value::Int(r.n_vars as i64));
    fit.insert("p_lags".into(), Value::Int(r.p_lags as i64));
    fit.insert("q_lags".into(), Value::Int(r.q_lags as i64));
    fit.insert("aic".into(), Value::Float(r.aic));
    fit.insert("bic".into(), Value::Float(r.bic));

    let n_ar = r.ar_params.len();
    let n_ma = r.ma_params.len();
    let n_exog = r.exog_params.as_ref().map(|x| x.len()).unwrap_or(0);
    let n = n_ar + n_ma + n_exog;
    let mut names = Vec::with_capacity(n);
    for i in 0..n_ar {
        names.push(format!("ar{i}"));
    }
    for i in 0..n_ma {
        names.push(format!("ma{i}"));
    }
    for i in 0..n_exog {
        names.push(format!("exog{i}"));
    }

    let mut params = Vec::with_capacity(n);
    params.extend(r.ar_params.iter());
    params.extend(r.ma_params.iter());
    if let Some(ex) = r.exog_params.as_ref() {
        params.extend(ex.iter());
    }

    ModelView {
        type_name: "VarmaResult".into(),
        summary: format!(
            "VARMA(vars={}, p={}, q={}, n={})",
            r.n_vars, r.p_lags, r.q_lags, r.n_obs
        ),
        variable_names: names,
        params: Array1::from(params),
        std_errors: Array1::zeros(n),
        test_values: Array1::zeros(n),
        p_values: Array1::ones(n),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-timeseries")]
#[cfg(feature = "greeners-timeseries")]
fn model_view_from_vecm(r: &std::rc::Rc<greeners::vecm::VecmResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_vars".into(), Value::Int(r.n_vars as i64));
    fit.insert("rank".into(), Value::Int(r.rank as i64));
    fit.insert("lags".into(), Value::Int(r.lags as i64));

    let k = r.n_vars;
    let rank = r.rank;
    let p_vecm = r.lags.saturating_sub(1);
    let n_alpha = rank * k;
    let n_beta = rank * k;
    // gamma has columns [intercept | Delta y_{t-1} ... Delta y_{t-p}], so
    // short-run coefficient count is k * p_vecm * k (excluding intercept column 0).
    let n_gamma = k * p_vecm * k;
    let n_total = n_alpha + n_beta + n_gamma;
    let mut params = Array1::<f64>::zeros(n_total);
    let mut ses = Array1::<f64>::zeros(n_total);
    let mut names: Vec<String> = Vec::with_capacity(n_total);
    let mut idx = 0;
    for i in 0..rank {
        for j in 0..k {
            params[idx] = r.alpha[(j, i)];
            ses[idx] = r.std_errors_alpha[(j, i)];
            names.push(format!(
                "alpha_{}_{}",
                i + 1,
                r.variable_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("x{j}"))
            ));
            idx += 1;
        }
    }
    for i in 0..rank {
        for j in 0..k {
            params[idx] = r.beta[(j, i)];
            ses[idx] = r.std_errors_beta[(j, i)];
            names.push(format!(
                "beta_{}_{}",
                i + 1,
                r.variable_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("x{j}"))
            ));
            idx += 1;
        }
    }
    for eq in 0..k {
        for lag in 1..=p_vecm {
            for src in 0..k {
                let col = 1 + (lag - 1) * k + src;
                params[idx] = r.gamma[(eq, col)];
                ses[idx] = r.std_errors_gamma[(eq, (lag - 1) * k + src)];
                names.push(format!(
                    "gamma_L{lag}_{}_{}",
                    r.variable_names[src], r.variable_names[eq]
                ));
                idx += 1;
            }
        }
    }

    ModelView {
        type_name: "VecmResult".into(),
        summary: format!(
            "VECM(vars={}, rank={}, lags={}, n={})",
            r.n_vars, r.rank, r.lags, r.n_obs
        ),
        variable_names: names,
        params,
        std_errors: ses,
        test_values: Array1::zeros(n_total),
        p_values: Array1::ones(n_total),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: None,
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}
#[cfg(feature = "greeners-panel")]
#[cfg(feature = "greeners-panel")]
fn model_view_from_ab(r: &std::rc::Rc<greeners::dynamic_panel::ArellanoBondResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_entities".into(), Value::Int(r.n_entities as i64));
    fit.insert("sargan_stat".into(), Value::Float(r.sargan_stat));
    fit.insert("sargan_pvalue".into(), Value::Float(r.sargan_pvalue));
    fit.insert("n_instruments".into(), Value::Int(r.n_instruments as i64));

    ModelView {
        type_name: "AbResult".into(),
        summary: format!(
            "Arellano-Bond(k={}, n={}, entities={})",
            r.params.len(),
            r.n_obs,
            r.n_entities
        ),
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
#[cfg(feature = "greeners-panel")]
#[cfg(feature = "greeners-panel")]
fn model_view_from_sys_gmm(r: &std::rc::Rc<greeners::dynamic_panel::SystemGmmResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs_fd".into(), Value::Int(r.n_obs_fd as i64));
    fit.insert("n_obs_lev".into(), Value::Int(r.n_obs_lev as i64));
    fit.insert("n_entities".into(), Value::Int(r.n_entities as i64));
    fit.insert("sargan_stat".into(), Value::Float(r.sargan_stat));
    fit.insert("sargan_pvalue".into(), Value::Float(r.sargan_pvalue));
    fit.insert("n_instruments".into(), Value::Int(r.n_instruments as i64));

    ModelView {
        type_name: "SysGmmResult".into(),
        summary: format!(
            "System GMM(k={}, n_fd={}, n_lev={})",
            r.params.len(),
            r.n_obs_fd,
            r.n_obs_lev
        ),
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
#[cfg(feature = "greeners-panel")]
#[cfg(feature = "greeners-panel")]
fn model_view_from_pcse(r: &std::rc::Rc<greeners::panel::PcseResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_entities".into(), Value::Int(r.n_entities as i64));
    fit.insert("t_periods".into(), Value::Int(r.t_periods as i64));
    fit.insert("df_resid".into(), Value::Int(r.df_resid as i64));
    fit.insert("r2".into(), Value::Float(r.r_squared));
    fit.insert("sigma".into(), Value::Float(r.sigma));

    ModelView {
        type_name: "PcseResult".into(),
        summary: format!(
            "PCSE(k={}, n={}, entities={}), R2={:.4}",
            r.params.len(),
            r.n_obs,
            r.n_entities,
            r.r_squared
        ),
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
#[cfg(feature = "greeners-panel")]
#[cfg(feature = "greeners-panel")]
fn model_view_from_panel_gls(r: &std::rc::Rc<greeners::panel::PanelGlsResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("n_entities".into(), Value::Int(r.n_entities as i64));
    fit.insert("t_periods".into(), Value::Int(r.t_periods as i64));
    fit.insert("df_resid".into(), Value::Int(r.df_resid as i64));
    fit.insert("r2".into(), Value::Float(r.r_squared));
    fit.insert("sigma".into(), Value::Float(r.sigma));

    let mut extras = HashMap::new();
    extras.insert("panels".into(), Value::Str(format!("{:?}", r.panels)));

    ModelView {
        type_name: "PanelGlsResult".into(),
        summary: format!(
            "Panel GLS(k={}, n={}, entities={}), R2={:.4}",
            r.params.len(),
            r.n_obs,
            r.n_entities,
            r.r_squared
        ),
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
        extras,
    }
}
#[cfg(feature = "greeners-ols")]
#[cfg(feature = "greeners-ols")]
fn model_view_from_rolling(r: &std::rc::Rc<greeners::rolling::RollingResult>) -> ModelView {
    let mut fit = HashMap::new();
    fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
    fit.insert("window".into(), Value::Int(r.window as i64));

    // Use final parameter estimates as the canonical coefficients.
    let k = r.params_history.ncols();
    let names = r.variable_names.clone().unwrap_or_default();

    ModelView {
        type_name: "RollingResult".into(),
        summary: format!("Rolling OLS(k={}, window={}, n={})", k, r.window, r.n_obs),
        variable_names: names,
        params: r.params_history.row(r.n_obs.saturating_sub(1)).to_owned(),
        std_errors: Array1::zeros(k),
        test_values: Array1::zeros(k),
        p_values: Array1::ones(k),
        conf_lower: None,
        conf_upper: None,
        fit,
        residuals: Some(r.residuals.clone()),
        fitted_values: None,
        x: None,
        extras: HashMap::new(),
    }
}

// ── DAP / field expansion helpers ──────────────────────────────────────────

fn array1_to_series(name: &str, arr: &Array1<f64>) -> Value {
    let values: Vec<Value> = arr.iter().map(|&v| Value::Float(v)).collect();
    Value::Series(Arc::new(Series::new(name, values)))
}

fn f64_array_column(arr: &Array1<f64>) -> greeners::Column {
    greeners::Column::Float(Array1::from(arr.iter().copied().collect::<Vec<_>>()))
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

/// Generate DAP-style `(name, Value)` children from a `ModelView`.
pub fn model_view_to_children(mv: &ModelView) -> Vec<(String, Value)> {
    let mut vars = Vec::new();

    let coef_df = coef_dataframe(
        &mv.variable_names,
        &mv.params,
        &mv.std_errors,
        &mv.test_values,
        &mv.p_values,
        mv.conf_lower.as_ref(),
        mv.conf_upper.as_ref(),
    );
    vars.push(("coefficients".into(), coef_df));

    let fit = Value::Dict(Arc::new(mv.fit.clone()));
    vars.push(("fit".into(), fit));

    if let Some(resid) = mv.residuals.as_ref() {
        if !resid.is_empty() {
            vars.push(("residuals".into(), array1_to_series("residuals", resid)));
        }
    }
    if let Some(fitted) = mv.fitted_values.as_ref() {
        if !fitted.is_empty() {
            vars.push((
                "fitted_values".into(),
                array1_to_series("fitted_values", fitted),
            ));
        }
    } else if let Some(x) = mv.x.as_ref() {
        if !x.is_empty() {
            let fitted = x.dot(&mv.params);
            vars.push((
                "fitted_values".into(),
                array1_to_series("fitted_values", &fitted),
            ));
        }
    }

    vars.push(("params".into(), array1_to_series("params", &mv.params)));
    vars.push((
        "std_errors".into(),
        array1_to_series("std_errors", &mv.std_errors),
    ));
    vars.push((
        "test_values".into(),
        array1_to_series("test_values", &mv.test_values),
    ));
    vars.push((
        "p_values".into(),
        array1_to_series("p_values", &mv.p_values),
    ));
    if let Some(cl) = mv.conf_lower.as_ref() {
        vars.push(("conf_lower".into(), array1_to_series("conf_lower", cl)));
    }
    if let Some(cu) = mv.conf_upper.as_ref() {
        vars.push(("conf_upper".into(), array1_to_series("conf_upper", cu)));
    }

    // Expose extras at the top level so `m.kind`, `m.var_names`, etc. work.
    for (k, v) in &mv.extras {
        vars.push((k.clone(), v.clone()));
    }

    vars
}

#[allow(dead_code)]
fn model_view_from_model_result(
    _display: &str,
    summary: &str,
    type_name: &str,
    fields: &std::sync::Arc<HashMap<String, Value>>,
) -> ModelView {
    // Heuristic: try to extract common fields from the generic ModelResult.
    let empty = Array1::zeros(0);
    let params = match fields.get("params") {
        Some(Value::List(v)) => v
            .iter()
            .map(|x| match x {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => f64::NAN,
            })
            .collect::<Vec<_>>()
            .into(),
        _ => empty.clone(),
    };
    let names = match fields.get("variable_names") {
        Some(Value::List(v)) => v
            .iter()
            .map(|x| match x {
                Value::Str(s) => s.clone(),
                _ => "?".into(),
            })
            .collect(),
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
