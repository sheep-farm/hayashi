use crate::lang::error::{HayashiError, Result};
use crate::lang::interpreter::model::Model;
use crate::lang::interpreter::model_view::ModelView;
use crate::lang::interpreter::value::{Series, Value};
use greeners::{Column, DataFrame};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use std::rc::Rc;
use std::sync::Arc;

// ── Wrappers that preserve the X matrix for diagnostics and predict ─────────

#[derive(Clone)]
pub struct OlsModel {
    pub result: Rc<greeners::OlsResult>,
    pub residuals: Array1<f64>,
    pub x: Array2<f64>,
}

impl std::fmt::Display for OlsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

#[derive(Clone)]
#[cfg(feature = "greeners-ols")]
pub struct PenalizedModel {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub variable_names: Vec<String>,
    pub r_squared: f64,
    pub n_obs: usize,
    pub alpha: f64,
    pub l1_ratio: Option<f64>,
    pub kind: String,
}

#[cfg(feature = "greeners-ols")]
impl std::fmt::Display for PenalizedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let title = match self.kind.as_str() {
            "ridge" => "Ridge Regression",
            "lasso" => "Lasso Regression",
            "elasticnet" => "ElasticNet Regression",
            _ => "Penalized Regression",
        };
        writeln!(f, "\n{:=^60}", format!(" {title} "))?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.6}", "Alpha:", self.alpha)?;
        if let Some(l1r) = self.l1_ratio {
            writeln!(f, "{:<20} {:>10.6}", "L1 ratio:", l1r)?;
        }
        writeln!(f, "{:<20} {:>10.4}", "R-squared:", self.r_squared)?;

        writeln!(f, "\n{:-^60}", " Coefficients ")?;
        writeln!(
            f,
            "{:<15} {:>12} {:>12} {:>12} {:>12}",
            "Variable", "coef", "std err", "t", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(60))?;
        for i in 0..self.params.len() {
            writeln!(
                f,
                "{:<15} {:>12.6} {:>12.6} {:>12.4} {:>12.4}",
                self.variable_names[i], self.params[i], self.std_errors[i], 0.0, 0.0
            )?;
        }
        writeln!(f, "{:=^60}", "")
    }
}

#[derive(Clone)]
#[cfg(feature = "greeners-glm")]
pub struct BinaryModel {
    pub result: Rc<greeners::discrete::BinaryModelResult>,
    pub y: Array1<f64>,
    pub x: Array2<f64>,
    pub kind: String,            // "logit" | "probit"
    pub coef_names: Vec<String>, // coefficient names for margins
}

#[cfg(feature = "greeners-glm")]
impl std::fmt::Display for BinaryModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── SUR wrapper (preserves variable names per equation) ─────────────────────

#[derive(Clone)]
#[cfg(feature = "greeners-ols")]
pub struct SurModel {
    pub result: Rc<greeners::sur::SurResult>,
    pub eq_var_names: Vec<Vec<String>>, // names per equation
}

#[cfg(feature = "greeners-ols")]
impl std::fmt::Display for SurModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = &self.result;
        let thick = "═".repeat(78);
        let thin = "─".repeat(78);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^78}", " Seemingly Unrelated Regressions (SUR) ")?;
        writeln!(f, "{:^78}", "Zellner's Efficient Estimator")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " Cross-Equation Error Correlation (Σ):")?;
        for row in r.sigma_cross.rows() {
            write!(f, "  [")?;
            for v in row {
                write!(f, " {:>8.4}", v)?;
            }
            writeln!(f, " ]")?;
        }
        for (eq, vnames) in r.equations.iter().zip(self.eq_var_names.iter()) {
            writeln!(f, "\n{:-^78}", format!(" Equation: {} ", eq.name))?;
            writeln!(
                f,
                "{:<20} {:>10} {:>10} {:>8} {:>8}",
                "Variable", "Coef", "Std Err", "t", "P>|t|"
            )?;
            writeln!(f, "{thin}")?;
            for i in 0..eq.params.len() {
                let vname: &str = vnames.get(i).map(|s| s.as_str()).unwrap_or("?");
                writeln!(
                    f,
                    "{:<20} {:>10.4} {:>10.4} {:>8.3} {:>8.3}",
                    vname, eq.params[i], eq.std_errors[i], eq.t_values[i], eq.p_values[i]
                )?;
            }
            writeln!(f, " R² = {:.4}", eq.r_squared)?;
        }
        writeln!(f, "{thick}")
    }
}

// ── PCA wrapper (adds variable names to PCAResult) ───────────────────────────
#[derive(Clone)]
pub struct PcaModel {
    pub result: Rc<greeners::multivariate::PCAResult>,
    pub var_names: Vec<String>,
}

impl std::fmt::Display for PcaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = &self.result;
        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^62}", " Principal Component Analysis ")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " {:>20}  {:>10}", "Observations:", r.n_obs)?;
        writeln!(f, " {:>20}  {:>10}", "Components:", r.n_components)?;
        writeln!(f, " {:>20}  {:>10}", "Variables:", self.var_names.len())?;
        writeln!(
            f,
            "\n{:^12} {:>12} {:>12} {:>10}",
            "Component", "Var Expl.", "% Cum.", "Eigenvalue"
        )?;
        writeln!(f, "{thin}")?;
        let mut cum = 0.0;
        for i in 0..r.n_components {
            cum += r.explained_variance_ratio[i];
            writeln!(
                f,
                " PC{:<9} {:>12.4} {:>12.4} {:>10.4}",
                i + 1,
                r.explained_variance_ratio[i],
                cum,
                r.explained_variance[i]
            )?;
        }
        writeln!(f, "\n{:^62}", " Loadings ")?;
        writeln!(f, "{thin}")?;
        let hdr: String = (0..r.n_components)
            .map(|i| format!(" {:>8}", format!("PC{}", i + 1)))
            .collect();
        writeln!(f, "{:<18}{hdr}", "Variable")?;
        for (j, vname) in self.var_names.iter().enumerate() {
            let row: String = (0..r.n_components)
                .map(|i| format!(" {:>8.4}", r.loadings[[j, i]]))
                .collect();
            writeln!(f, "{:<18}{row}", vname)?;
        }
        writeln!(f, "{thick}")
    }
}

// ── Factor Analysis wrapper ───────────────────────────────────────────────────
#[derive(Clone)]
pub struct FactorModel {
    pub result: Rc<greeners::multivariate::FactorResult>,
    pub var_names: Vec<String>,
}

impl std::fmt::Display for FactorModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = &self.result;
        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^62}", " Factor Analysis (Principal Axis) ")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " {:>20}  {:>10}", "Observations:", r.n_obs)?;
        writeln!(f, " {:>20}  {:>10}", "Factors:", r.n_factors)?;
        writeln!(f, "\n{:^62}", " Factor Loadings ")?;
        writeln!(f, "{thin}")?;
        let hdr: String = (0..r.n_factors)
            .map(|i| format!(" {:>8}", format!("F{}", i + 1)))
            .collect();
        writeln!(f, "{:<18}{hdr}  {:>10}", "Variable", "Communality")?;
        for (j, vname) in self.var_names.iter().enumerate() {
            let row: String = (0..r.n_factors)
                .map(|i| format!(" {:>8.4}", r.loadings[[j, i]]))
                .collect();
            writeln!(f, "{:<18}{row}  {:>10.4}", vname, r.communalities[j])?;
        }
        writeln!(f, "\n{:<12} {:>10}", "Eigenvalues:", "")?;
        for (i, &ev) in r.eigenvalues.iter().enumerate() {
            writeln!(f, "  F{:<10} {:>10.4}", i + 1, ev)?;
        }
        writeln!(f, "{thick}")
    }
}

// ── DFM wrapper ───────────────────────────────────────────────────────────────
#[derive(Clone)]
#[cfg(feature = "greeners-timeseries")]
pub struct DFMModel {
    pub result: Rc<greeners::dynamic_factor::DynamicFactorResult>,
    #[allow(dead_code)]
    pub var_names: Vec<String>,
}

#[cfg(feature = "greeners-timeseries")]
impl std::fmt::Display for DFMModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── 3SLS wrapper ──────────────────────────────────────────────────────────────
#[derive(Clone)]
#[cfg(feature = "greeners-ols")]
pub struct ThreeSLSModel {
    pub result: Rc<greeners::three_sls::ThreeSLSResult>,
    #[allow(dead_code)]
    pub eq_var_names: Vec<Vec<String>>,
}

#[cfg(feature = "greeners-ols")]
impl std::fmt::Display for ThreeSLSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── Model trait implementation for OLS ──────────────────────────────────────

impl Model for OlsModel {
    fn type_name(&self) -> &str {
        "OlsResult"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn summary(&self) -> String {
        let r = &self.result;
        format!(
            "OLS(k={}, n={}), R2={:.4}",
            r.params.len(),
            r.n_obs,
            r.r_squared
        )
    }

    fn to_model_view(&self) -> ModelView {
        let r = &self.result;
        let mut fit = std::collections::HashMap::new();
        fit.insert("n".into(), Value::Int(r.n_obs as i64));
        fit.insert("df_resid".into(), Value::Int(r.df_resid as i64));
        fit.insert("r2".into(), Value::Float(r.r_squared));
        fit.insert("adj_r2".into(), Value::Float(r.adj_r_squared));
        fit.insert("f_stat".into(), Value::Float(r.f_statistic));
        fit.insert("prob_f".into(), Value::Float(r.prob_f));
        fit.insert("log_lik".into(), Value::Float(r.log_likelihood));
        fit.insert("aic".into(), Value::Float(r.aic));
        fit.insert("bic".into(), Value::Float(r.bic));
        fit.insert("sigma".into(), Value::Float(r.sigma));

        ModelView {
            type_name: "OlsResult".into(),
            summary: self.summary(),
            variable_names: r.variable_names.clone().unwrap_or_default(),
            params: r.params.clone(),
            std_errors: r.std_errors.clone(),
            test_values: r.t_values.clone(),
            p_values: r.p_values.clone(),
            conf_lower: Some(r.conf_lower.clone()),
            conf_upper: Some(r.conf_upper.clone()),
            fit,
            residuals: Some(self.residuals.clone()),
            fitted_values: None,
            x: Some(self.x.clone()),
            extras: std::collections::HashMap::new(),
        }
    }

    fn predict(&self, kind: &str, _newdata: Option<&DataFrame>) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Ok(self.x.dot(&self.result.params).to_vec()),
            "residuals" | "resid" | "e" => Ok(self.residuals.to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict OLS: kind '{k}' unknown — use: xb, residuals"
            ))),
        }
    }

    fn residuals(&self) -> Option<Array1<f64>> {
        Some(self.residuals.clone())
    }

    fn fitted_values(&self) -> Option<Array1<f64>> {
        Some(self.x.dot(&self.result.params))
    }

    fn field(&self, name: &str) -> Result<Value> {
        let r = &self.result;
        let names = r.variable_names.clone().unwrap_or_default();

        let vec_to_series = |v: &[f64], label: &str| {
            let vals: Vec<Value> = v.iter().map(|&x| Value::Float(x)).collect();
            Value::Series(Arc::new(Series::new(label, vals)))
        };

        let vec_to_dataframe = |v: &ndarray::Array1<f64>, col: &str| {
            let mut columns: IndexMap<String, Column> = IndexMap::new();
            let var_col: Vec<String> = (0..v.len())
                .map(|i| names.get(i).cloned().unwrap_or_else(|| format!("x{i}")))
                .collect();
            let val_col: Vec<f64> = v.iter().copied().collect();
            columns.insert(
                "variable".into(),
                Column::String(ndarray::Array1::from(var_col)),
            );
            columns.insert(col.into(), Column::Float(ndarray::Array1::from(val_col)));
            DataFrame::from_columns(columns)
                .map_or_else(|_e| Value::Nil, |df| Value::DataFrame(Arc::new(df)))
        };

        match name {
            "params" | "coef" | "coefficients" => Ok(vec_to_dataframe(&r.params, "coef")),
            "std_errors" | "se" => Ok(vec_to_dataframe(&r.std_errors, "std_err")),
            "t_values" | "t" => Ok(vec_to_dataframe(&r.t_values, "t")),
            "p_values" | "p" => Ok(vec_to_dataframe(&r.p_values, "p_value")),
            "conf_lower" => Ok(vec_to_dataframe(&r.conf_lower, "conf_low")),
            "conf_upper" => Ok(vec_to_dataframe(&r.conf_upper, "conf_high")),
            "residuals" => Ok(vec_to_series(&self.residuals.to_vec(), "residuals")),
            "fitted" | "fitted_values" => {
                Ok(vec_to_series(&self.x.dot(&r.params).to_vec(), "fitted"))
            }
            "r_squared" | "r2" => Ok(Value::Float(r.r_squared)),
            "adj_r_squared" | "adj_r2" => Ok(Value::Float(r.adj_r_squared)),
            "f_statistic" | "f" => Ok(Value::Float(r.f_statistic)),
            "prob_f" => Ok(Value::Float(r.prob_f)),
            "log_lik" | "log_likelihood" => Ok(Value::Float(r.log_likelihood)),
            "aic" => Ok(Value::Float(r.aic)),
            "bic" => Ok(Value::Float(r.bic)),
            "sigma" => Ok(Value::Float(r.sigma)),
            "n" | "n_obs" => Ok(Value::Int(r.n_obs as i64)),
            "df_resid" => Ok(Value::Int(r.df_resid as i64)),
            "df_model" => Ok(Value::Int(r.df_model as i64)),
            "cov_type" => Ok(Value::Str(format!("{:?}", r.cov_type))),
            "inference_type" => Ok(Value::Str(format!("{:?}", r.inference_type))),
            "variable_names" => {
                let lst: Vec<Value> = names.into_iter().map(Value::Str).collect();
                Ok(Value::List(Arc::new(lst)))
            }
            "summary" => Ok(Value::Str(format!("{}", r))),
            _ => Err(HayashiError::Runtime(format!(
                "OLS result has no field '{name}'"
            ))),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let r = &self.result;
        serde_json::json!({
            "__model_type__": "ols",
            "variable": r.variable_names.clone().unwrap_or_default(),
            "coef": r.params.to_vec(),
            "std_err": r.std_errors.to_vec(),
            "t": r.t_values.to_vec(),
            "p_value": r.p_values.to_vec(),
            "conf_low": r.conf_lower.to_vec(),
            "conf_high": r.conf_upper.to_vec(),
            "r2": r.r_squared,
            "adj_r2": r.adj_r_squared,
            "n": r.n_obs,
            "f_stat": r.f_statistic,
            "prob_f": r.prob_f,
            "aic": r.aic,
            "bic": r.bic,
            "log_lik": r.log_likelihood,
            "sigma": r.sigma,
        })
    }
}

// ── Model trait implementation for Binary (logit/probit) ─────────────────────

#[cfg(feature = "greeners-glm")]
impl Model for BinaryModel {
    fn type_name(&self) -> &str {
        "BinaryResult"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn summary(&self) -> String {
        let r = &self.result;
        format!(
            "{}(k={}), pseudo-R2={:.4}",
            r.model_name,
            r.params.len(),
            r.pseudo_r2
        )
    }

    fn to_model_view(&self) -> ModelView {
        let r = &self.result;
        let mut fit = std::collections::HashMap::new();
        fit.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
        fit.insert("pseudo_r2".into(), Value::Float(r.pseudo_r2));

        let mut extras = std::collections::HashMap::new();
        extras.insert("kind".into(), Value::Str(self.kind.clone()));

        ModelView {
            type_name: "BinaryResult".into(),
            summary: self.summary(),
            variable_names: self.coef_names.clone(),
            params: r.params.clone(),
            std_errors: r.std_errors.clone(),
            test_values: r.z_values.clone(),
            p_values: r.p_values.clone(),
            conf_lower: None,
            conf_upper: None,
            fit,
            residuals: None,
            fitted_values: Some(self.result.predict_proba(&self.x)),
            x: Some(self.x.clone()),
            extras,
        }
    }

    fn predict(&self, kind: &str, _newdata: Option<&DataFrame>) -> Result<Vec<f64>> {
        match kind {
            "pr" | "xb" | "fitted" => Ok(self.result.predict_proba(&self.x).to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict logit/probit: kind '{k}' unknown — use: pr"
            ))),
        }
    }

    fn residuals(&self) -> Option<Array1<f64>> {
        None
    }

    fn fitted_values(&self) -> Option<Array1<f64>> {
        Some(self.result.predict_proba(&self.x))
    }

    fn field(&self, name: &str) -> Result<Value> {
        let r = &self.result;
        match name {
            "summary" => Ok(Value::Str(format!("{}", r))),
            "kind" => Ok(Value::Str(self.kind.clone())),
            "n" | "n_obs" => Ok(Value::Int(self.y.len() as i64)),
            "pseudo_r2" => Ok(Value::Float(r.pseudo_r2)),
            "log_lik" | "log_likelihood" => Ok(Value::Float(r.log_likelihood)),
            "params" | "coef" | "coefficients" => Ok(Value::List(Arc::new(
                r.params.iter().map(|&v| Value::Float(v)).collect(),
            ))),
            "std_errors" | "se" => Ok(Value::List(Arc::new(
                r.std_errors.iter().map(|&v| Value::Float(v)).collect(),
            ))),
            "z_values" | "z" => Ok(Value::List(Arc::new(
                r.z_values.iter().map(|&v| Value::Float(v)).collect(),
            ))),
            "p_values" | "p" => Ok(Value::List(Arc::new(
                r.p_values.iter().map(|&v| Value::Float(v)).collect(),
            ))),
            "variable_names" => {
                let lst: Vec<Value> = self.coef_names.iter().cloned().map(Value::Str).collect();
                Ok(Value::List(Arc::new(lst)))
            }
            _ => Err(HayashiError::Runtime(format!(
                "Binary result has no field '{name}'"
            ))),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let r = &self.result;
        serde_json::json!({
            "__model_type__": self.kind.as_str(),
            "variable": self.coef_names.clone(),
            "coef": r.params.to_vec(),
            "std_err": r.std_errors.to_vec(),
            "z": r.z_values.to_vec(),
            "p_value": r.p_values.to_vec(),
            "pseudo_r2": r.pseudo_r2,
            "log_lik": r.log_likelihood,
        })
    }
}

// ── SUR helpers and Model trait implementation ────────────────────────────────

#[allow(clippy::type_complexity)]
#[cfg(feature = "greeners-ols")]
fn flatten_equations(
    equations: &[greeners::sur::SurEquationResult],
) -> (
    Vec<String>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let total_len: usize = equations.iter().map(|eq| eq.params.len()).sum();
    let mut names = Vec::with_capacity(total_len);
    let mut params = Vec::with_capacity(total_len);
    let mut std_errors = Vec::with_capacity(total_len);
    let mut t_values = Vec::with_capacity(total_len);
    let mut p_values = Vec::with_capacity(total_len);

    for eq in equations {
        for i in 0..eq.params.len() {
            let vname = if i == 0 {
                format!("{}:_cons", eq.name)
            } else {
                format!("{}:x{i}", eq.name)
            };
            names.push(vname);
            params.push(eq.params[i]);
            std_errors.push(eq.std_errors[i]);
            t_values.push(eq.t_values[i]);
            p_values.push(eq.p_values[i]);
        }
    }

    (
        names,
        params.into(),
        std_errors.into(),
        t_values.into(),
        p_values.into(),
    )
}

#[cfg(feature = "greeners-ols")]
impl Model for SurModel {
    fn type_name(&self) -> &str {
        "SurResult"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn summary(&self) -> String {
        let r = &self.result;
        let n_obs = r.equations.first().map(|eq| eq.params.len()).unwrap_or(0);
        format!(
            "SUR(equations={}, n≈{}), system-R2={:.4}",
            r.equations.len(),
            n_obs,
            r.system_r2
        )
    }

    fn to_model_view(&self) -> ModelView {
        let r = &self.result;

        let mut fit = std::collections::HashMap::new();
        fit.insert("n_equations".into(), Value::Int(r.equations.len() as i64));
        fit.insert("system_r2".into(), Value::Float(r.system_r2));

        let mut extras = std::collections::HashMap::new();
        extras.insert(
            "eq_var_names".into(),
            Value::List(Arc::new(
                self.eq_var_names
                    .iter()
                    .map(|v| {
                        Value::List(Arc::new(v.iter().map(|s| Value::Str(s.clone())).collect()))
                    })
                    .collect(),
            )),
        );

        let (names, params, std_errors, t_values, p_values) = flatten_equations(&r.equations);

        ModelView {
            type_name: "SurResult".into(),
            summary: self.summary(),
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
}

// ── PCA, Factor, DFM, 3SLS and Penalized implementations ──────────────────────

impl Model for PcaModel {
    fn type_name(&self) -> &str {
        "PcaResult"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn summary(&self) -> String {
        let r = &self.result;
        format!(
            "PCA(components={}, variables={}, n={})",
            r.n_components,
            self.var_names.len(),
            r.n_obs
        )
    }

    fn to_model_view(&self) -> ModelView {
        let r = &self.result;
        let mut fit = std::collections::HashMap::new();
        fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
        fit.insert("n_components".into(), Value::Int(r.n_components as i64));

        let mut extras = std::collections::HashMap::new();
        extras.insert(
            "var_names".into(),
            Value::List(Arc::new(
                self.var_names
                    .iter()
                    .map(|s| Value::Str(s.clone()))
                    .collect(),
            )),
        );
        extras.insert(
            "explained_variance".into(),
            Value::List(Arc::new(
                r.explained_variance
                    .iter()
                    .map(|&v| Value::Float(v))
                    .collect(),
            )),
        );
        extras.insert(
            "explained_variance_ratio".into(),
            Value::List(Arc::new(
                r.explained_variance_ratio
                    .iter()
                    .map(|&v| Value::Float(v))
                    .collect(),
            )),
        );

        ModelView {
            type_name: "PcaResult".into(),
            summary: self.summary(),
            variable_names: (0..r.n_components)
                .map(|i| format!("PC{}", i + 1))
                .collect(),
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
}

impl Model for FactorModel {
    fn type_name(&self) -> &str {
        "FactorResult"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn summary(&self) -> String {
        let r = &self.result;
        format!(
            "Factor Analysis(factors={}, variables={}, n={})",
            r.n_factors,
            self.var_names.len(),
            r.n_obs
        )
    }

    fn to_model_view(&self) -> ModelView {
        let r = &self.result;
        let mut fit = std::collections::HashMap::new();
        fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
        fit.insert("n_factors".into(), Value::Int(r.n_factors as i64));

        let mut extras = std::collections::HashMap::new();
        extras.insert(
            "var_names".into(),
            Value::List(Arc::new(
                self.var_names
                    .iter()
                    .map(|s| Value::Str(s.clone()))
                    .collect(),
            )),
        );
        extras.insert(
            "eigenvalues".into(),
            Value::List(Arc::new(
                r.eigenvalues.iter().map(|&v| Value::Float(v)).collect(),
            )),
        );

        ModelView {
            type_name: "FactorResult".into(),
            summary: self.summary(),
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
}

#[cfg(feature = "greeners-timeseries")]
impl Model for DFMModel {
    fn type_name(&self) -> &str {
        "DFMResult"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn summary(&self) -> String {
        let r = &self.result;
        format!(
            "Dynamic Factor Model(factors={}, n={})",
            r.n_factors, r.n_obs
        )
    }

    fn to_model_view(&self) -> ModelView {
        let r = &self.result;
        let mut fit = std::collections::HashMap::new();
        fit.insert("n_obs".into(), Value::Int(r.n_obs as i64));
        fit.insert("n_factors".into(), Value::Int(r.n_factors as i64));
        fit.insert("n_vars".into(), Value::Int(r.n_vars as i64));

        let mut extras = std::collections::HashMap::new();
        extras.insert(
            "var_names".into(),
            Value::List(Arc::new(
                self.var_names
                    .iter()
                    .map(|s| Value::Str(s.clone()))
                    .collect(),
            )),
        );

        let n = r.n_vars;
        let params = r.sigma_obs.mapv(|x| 1.0 - x);
        let std_errors = Array1::from_elem(n, f64::NAN);
        let test_values = Array1::zeros(n);
        let p_values = Array1::ones(n);

        ModelView {
            type_name: "DFMResult".into(),
            summary: self.summary(),
            variable_names: self.var_names.clone(),
            params,
            std_errors,
            test_values,
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
}

#[allow(clippy::type_complexity)]
#[cfg(feature = "greeners-ols")]
fn flatten_three_sls_equations(
    equations: &[greeners::three_sls::EquationResult],
) -> (
    Vec<String>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let total_len: usize = equations.iter().map(|eq| eq.params.len()).sum();
    let mut names = Vec::with_capacity(total_len);
    let mut params = Vec::with_capacity(total_len);
    let mut std_errors = Vec::with_capacity(total_len);
    let mut t_values = Vec::with_capacity(total_len);
    let mut p_values = Vec::with_capacity(total_len);

    for eq in equations {
        for i in 0..eq.params.len() {
            let base = eq
                .var_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{i}"));
            let vname = format!("{}:{}", eq.name, base);
            names.push(vname);
            params.push(eq.params[i]);
            std_errors.push(eq.std_errors[i]);
            t_values.push(eq.t_values[i]);
            p_values.push(eq.p_values[i]);
        }
    }

    (
        names,
        params.into(),
        std_errors.into(),
        t_values.into(),
        p_values.into(),
    )
}

#[cfg(feature = "greeners-ols")]
impl Model for ThreeSLSModel {
    fn type_name(&self) -> &str {
        "ThreeSLSResult"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn summary(&self) -> String {
        let r = &self.result;
        let n_obs = r.equations.first().map(|eq| eq.params.len()).unwrap_or(0);
        format!(
            "3SLS(equations={}, n≈{}), system-R2={:.4}",
            r.equations.len(),
            n_obs,
            r.system_r2
        )
    }

    fn to_model_view(&self) -> ModelView {
        let r = &self.result;

        let mut fit = std::collections::HashMap::new();
        fit.insert("n_equations".into(), Value::Int(r.equations.len() as i64));
        fit.insert("system_r2".into(), Value::Float(r.system_r2));

        let mut extras = std::collections::HashMap::new();
        extras.insert(
            "eq_var_names".into(),
            Value::List(Arc::new(
                self.eq_var_names
                    .iter()
                    .map(|v| {
                        Value::List(Arc::new(v.iter().map(|s| Value::Str(s.clone())).collect()))
                    })
                    .collect(),
            )),
        );

        let (names, params, std_errors, t_values, p_values) =
            flatten_three_sls_equations(&r.equations);

        ModelView {
            type_name: "ThreeSLSResult".into(),
            summary: self.summary(),
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
}

#[cfg(feature = "greeners-ols")]
impl Model for PenalizedModel {
    fn type_name(&self) -> &str {
        "PenalizedResult"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn summary(&self) -> String {
        format!(
            "{}(k={}, n={}), R2={:.4}",
            match self.kind.as_str() {
                "ridge" => "Ridge",
                "lasso" => "Lasso",
                "elasticnet" => "ElasticNet",
                _ => "Penalized Regression",
            },
            self.params.len(),
            self.n_obs,
            self.r_squared
        )
    }

    fn to_model_view(&self) -> ModelView {
        let mut fit = std::collections::HashMap::new();
        fit.insert("n_obs".into(), Value::Int(self.n_obs as i64));
        fit.insert("r_squared".into(), Value::Float(self.r_squared));
        fit.insert("alpha".into(), Value::Float(self.alpha));
        if let Some(l1r) = self.l1_ratio {
            fit.insert("l1_ratio".into(), Value::Float(l1r));
        }

        let mut extras = std::collections::HashMap::new();
        extras.insert("kind".into(), Value::Str(self.kind.clone()));

        ModelView {
            type_name: "PenalizedResult".into(),
            summary: self.summary(),
            variable_names: self.variable_names.clone(),
            params: self.params.clone(),
            std_errors: self.std_errors.clone(),
            test_values: Array1::zeros(self.params.len()),
            p_values: Array1::ones(self.params.len()),
            conf_lower: None,
            conf_upper: None,
            fit,
            residuals: None,
            fitted_values: None,
            x: None,
            extras,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "__model_type__": self.kind.as_str(),
            "variable": self.variable_names.clone(),
            "coef": self.params.to_vec(),
            "std_err": self.std_errors.to_vec(),
            "r2": self.r_squared,
            "n": self.n_obs,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
        })
    }
}
