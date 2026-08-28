use crate::lang::ast::{Expr, Spanned};
use crate::lang::error::HayashiError;
use crate::lang::interpreter::Model;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

// ── User-defined function ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UserFn {
    pub params: Vec<String>,
    pub defaults: Vec<Option<Expr>>,
    pub doc: Option<String>,
    pub body: Vec<Spanned>,
}

// ── Structured error ─────────────────────────────────────────────────────────

/// Structured error exposed to the user in `try { ... } catch e { ... }`.
/// `e.kind`, `e.msg` and `e.line` are accessible as fields of a dict.
#[derive(Debug, Clone)]
pub struct ErrorValue {
    pub kind: String,
    pub msg: String,
    pub line: i64,
}

impl ErrorValue {
    pub fn from_hayashi_error(e: &HayashiError, current_line: usize) -> Self {
        let (kind, msg) = match e {
            HayashiError::Lex { msg, .. } => ("lex", msg.clone()),
            HayashiError::Parse { msg, .. } => ("parse", msg.clone()),
            HayashiError::Type(m) => ("type", m.clone()),
            HayashiError::Runtime(m) => ("runtime", m.clone()),
            HayashiError::Annotated(m) => ("annotated", m.clone()),
            HayashiError::Io(m) => ("io", m.clone()),
            HayashiError::Return | HayashiError::Break | HayashiError::Continue => {
                ("control", e.to_string())
            }
        };
        let line = match Self::extract_line(&msg) {
            0 => current_line as i64,
            n => n,
        };
        let msg = Self::strip_line_prefix(&msg);
        Self {
            kind: kind.into(),
            msg,
            line,
        }
    }

    fn extract_line(msg: &str) -> i64 {
        // formats: "line N: ..." or "Lexer error at line N: ..."
        if let Some(pos) = msg.find("line ") {
            let rest = &msg[pos + 5..];
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !num.is_empty() {
                return num.parse().unwrap_or(0);
            }
        }
        0
    }

    fn strip_line_prefix(msg: &str) -> String {
        if let Some(pos) = msg.find("line ") {
            let rest = &msg[pos + 5..];
            if let Some(colon) = rest.find(": ") {
                let line_num_len = rest[..colon]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .count();
                if line_num_len == colon {
                    return rest[colon + 2..].to_string();
                }
            }
        }
        msg.to_string()
    }
}

// ── Diagnostic test result ───────────────────────────────────────────────────

#[derive(Clone)]
pub struct DiagResult {
    pub rendered: String,               // pre-rendered output by the test
    pub fields: HashMap<String, Value>, // structured fields for DAP/debug
}

impl std::fmt::Debug for DiagResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiagResult")
            .field("rendered", &self.rendered)
            .field("fields", &self.fields.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl std::fmt::Display for DiagResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.rendered)
    }
}

// ── Series ───────────────────────────────────────────────────────────────────

/// Series: DataFrame column as a first-class citizen.
#[derive(Clone)]
pub struct Series {
    pub name: String,
    pub values: Vec<Value>,
}

impl Series {
    pub fn new(name: impl Into<String>, values: Vec<Value>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn first(&self) -> Option<Value> {
        self.values.first().cloned()
    }

    pub fn last(&self) -> Option<Value> {
        self.values.last().cloned()
    }

    pub fn numeric_values(&self) -> Vec<f64> {
        self.values
            .iter()
            .filter_map(|v| match v {
                Value::Float(x) => Some(*x),
                Value::Int(x) => Some(*x as f64),
                _ => None,
            })
            .collect()
    }

    pub fn mean(&self) -> f64 {
        let v = self.numeric_values();
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }

    pub fn sd(&self) -> f64 {
        let v = self.numeric_values();
        if v.len() < 2 {
            f64::NAN
        } else {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let ss = v.iter().map(|x| (x - m).powi(2)).sum::<f64>();
            (ss / (v.len() - 1) as f64).sqrt()
        }
    }

    pub fn min(&self) -> f64 {
        let v = self.numeric_values();
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        }
    }

    pub fn max(&self) -> f64 {
        let v = self.numeric_values();
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        }
    }

    pub fn shift(&self, n: i64) -> Series {
        let len = self.values.len();
        let n_abs = n.unsigned_abs() as usize;
        let fill = Value::Nil;
        let mut shifted = Vec::with_capacity(len);
        if n > 0 {
            shifted.extend(std::iter::repeat_n(fill, n_abs));
            shifted.extend(self.values[..len.saturating_sub(n_abs)].iter().cloned());
        } else if n < 0 {
            shifted.extend(self.values[n_abs.min(len)..].iter().cloned());
            shifted.extend(std::iter::repeat_n(fill, n_abs.min(len)));
        } else {
            shifted = self.values.clone();
        }
        Series::new(self.name.clone(), shifted)
    }
}

// ── Value ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    DataFrame(Arc<greeners::DataFrame>),

    OlsResult(super::models::OlsModel),
    #[cfg(feature = "greeners-ols")]
    IvResult(Rc<greeners::iv::IvResult>),
    #[cfg(feature = "greeners-glm")]
    BinaryResult(super::models::BinaryModel),
    PanelResult(Rc<greeners::panel::PanelResult>),
    #[cfg(feature = "greeners-panel")]
    BetweenResult(Rc<greeners::panel::BetweenResult>),
    #[cfg(feature = "greeners-panel")]
    ReResult(Rc<greeners::panel::RandomEffectsResult>),
    #[cfg(feature = "greeners-timeseries")]
    ArimaResult(Rc<greeners::arima::ArimaResult>),
    #[cfg(feature = "greeners-timeseries")]
    VarResult(Rc<greeners::var::VarResult>),
    #[cfg(feature = "greeners-timeseries")]
    VecmResult(Rc<greeners::vecm::VecmResult>),
    #[cfg(feature = "greeners-timeseries")]
    GarchResult(Rc<greeners::garch::GarchResult>),
    DiagResult(Rc<DiagResult>),
    #[cfg(feature = "greeners-panel")]
    AbResult(Rc<greeners::dynamic_panel::ArellanoBondResult>),
    #[cfg(feature = "greeners-ols")]
    GmmResult(Rc<greeners::gmm::GmmResult>),
    #[cfg(feature = "greeners-panel")]
    SysGmmResult(Rc<greeners::dynamic_panel::SystemGmmResult>),
    #[cfg(feature = "greeners-panel")]
    FE2SLSResult(Rc<greeners::panel::PanelIvResult>),
    #[cfg(feature = "greeners-panel")]
    PcseResult(Rc<greeners::panel::PcseResult>),
    #[cfg(feature = "greeners-panel")]
    PanelGlsResult(Rc<greeners::panel::PanelGlsResult>),
    #[cfg(feature = "greeners-ols")]
    TobitResult(Rc<greeners::tobit::TobitResult>),
    #[cfg(feature = "greeners-ols")]
    HeckmanResult(Rc<greeners::heckman::HeckmanResult>),
    #[cfg(feature = "greeners-causal")]
    RdResult(Rc<greeners::rd::RdResult>),
    #[cfg(feature = "greeners-causal")]
    SynthResult(Rc<greeners::synth::SynthResult>),
    #[cfg(feature = "greeners-causal")]
    PsmResult(Rc<greeners::psm::PsmResult>),
    #[cfg(feature = "greeners-glm")]
    PoissonResult(Rc<greeners::poisson::PoissonResult>),
    #[cfg(feature = "greeners-glm")]
    NegBinResult(Rc<greeners::negbin::NegBinResult>),
    #[cfg(feature = "greeners-glm")]
    OrderedResult(Rc<greeners::ordered::OrderedResult>),
    #[cfg(feature = "greeners-glm")]
    MNLogitResult(Rc<greeners::mnlogit::MNLogitResult>),
    #[cfg(feature = "greeners-causal")]
    DidResult(Rc<greeners::did::DidResult>),
    #[cfg(feature = "greeners-ols")]
    QuantileResult(Rc<greeners::quantile::QuantileResult>),
    #[cfg(feature = "greeners-survival")]
    KMResult(Rc<greeners::survival::KMResult>),
    #[cfg(feature = "greeners-survival")]
    CoxResult(Rc<greeners::survival::CoxResult>),
    #[cfg(feature = "greeners-ols")]
    RlmResult(Rc<greeners::rlm::RlmResult>),
    #[cfg(feature = "greeners-glm")]
    GeeResult(Rc<greeners::gee::GeeResult>),
    #[cfg(feature = "greeners-glm")]
    ZeroInflatedResult(Rc<greeners::zero_inflated::ZeroInflatedResult>),
    #[cfg(feature = "greeners-bayesian")]
    MixedResult(Rc<greeners::mixed::MixedResult>),
    #[cfg(feature = "greeners-glm")]
    BetaResult(Rc<greeners::beta_model::BetaResult>),
    #[cfg(feature = "greeners-ols")]
    GlsarResult(Rc<greeners::glsar::GlsarResult>),

    #[cfg(feature = "greeners-ols")]
    SurResult(super::models::SurModel),
    #[cfg(feature = "greeners-ols")]
    RollingResult(Rc<greeners::rolling::RollingResult>),
    #[cfg(feature = "greeners-ols")]
    RecursiveLSResult(Rc<greeners::rolling::RecursiveLSResult>),
    PcaResult(super::models::PcaModel),
    FactorResult(super::models::FactorModel),
    #[cfg(feature = "greeners-glm")]
    GlmResult(Rc<greeners::glm::GlmResult>),
    LowessResult(Rc<greeners::nonparametric::LowessResult>),

    #[cfg(feature = "greeners-timeseries")]
    MarkovResult(Rc<greeners::markov::MarkovSwitchingResult>),
    #[cfg(feature = "greeners-glm")]
    ConditionalResult(Rc<greeners::conditional::ConditionalResult>),
    #[cfg(feature = "greeners-timeseries")]
    VarmaResult(Rc<greeners::varma::VarmaResult>),
    #[cfg(feature = "greeners-timeseries")]
    DecompResult(Rc<greeners::decomposition::DecompositionResult>),
    #[cfg(feature = "greeners-timeseries")]
    MstlResult(Rc<greeners::mstl::MSTLResult>),
    #[cfg(feature = "greeners-timeseries")]
    UCResult(Rc<greeners::unobserved_components::UCResult>),
    #[cfg(feature = "greeners-glm")]
    GamResult(Rc<greeners::glmgam::GamResult>),
    #[cfg(feature = "greeners-imputation")]
    MiceResult(Rc<greeners::imputation::MICEResult>),
    #[cfg(feature = "greeners-timeseries")]
    MSARResult(Rc<greeners::markov_autoreg::MarkovAutoregResult>),
    #[cfg(feature = "greeners-timeseries")]
    SVarResult(Rc<greeners::svar::SVarResult>),

    #[cfg(feature = "greeners-timeseries")]
    EtsResult(Rc<greeners::ets::ETSResult>),

    #[cfg(feature = "greeners-panel")]
    ThresholdResult(Rc<greeners::threshold::ThresholdResult>),
    #[cfg(feature = "greeners-timeseries")]
    AutoRegResult(Rc<greeners::autoreg::AutoRegResult>),
    #[cfg(feature = "greeners-timeseries")]
    ArdlResult(Rc<greeners::autoreg::ARDLResult>),
    #[cfg(feature = "greeners-timeseries")]
    LocalLevelResult(Rc<greeners::statespace::LocalLevelResult>),
    #[cfg(feature = "greeners-ml")]
    KmeansResult(Rc<greeners::kmeans::KmeansResult>),
    #[cfg(feature = "greeners-ml")]
    DbscanResult(Rc<greeners::dbscan::DbscanResult>),
    IsotonicResult(Rc<greeners::isotonic::IsotonicResult>),
    KdeResult(Rc<greeners::nonparametric::KDEResult>),
    #[cfg(feature = "greeners-ml")]
    BartResult(Rc<greeners::bart::BartResult>),
    #[cfg(feature = "greeners-ml")]
    GpResult(Rc<greeners::gp::GpResult>),
    GmmClusteringResult(Rc<greeners::gmm_clustering::GmmResult>),
    #[cfg(feature = "greeners-ml")]
    HierarchicalResult(Rc<greeners::hierarchical::HierarchicalResult>),
    #[cfg(feature = "greeners-timeseries")]
    #[cfg(feature = "experimental")]
    SpectralResult(Rc<greeners::spectral::SpectralResult>),

    #[cfg(feature = "greeners-timeseries")]
    DFMResult(super::models::DFMModel),
    #[cfg(feature = "greeners-ols")]
    ThreeSLSResult(super::models::ThreeSLSModel),
    #[cfg(feature = "greeners-ols")]
    PenalizedResult(super::models::PenalizedModel),

    /// First-class model result trait object. Used for unified model
    /// access, replacing the scattered `Value::*Result` variants over time.
    Model(Rc<dyn Model>),
    /// Generic first-class model result: a display string plus a dict of
    /// named children.  Used for estimators that do not yet have a dedicated
    /// `Value` variant, while still exposing every field to DAP and to the
    /// `var.field` / `var["field"]` syntax.
    #[allow(clippy::large_enum_variant)]
    ModelResult {
        display: String,
        summary: String,
        type_name: &'static str,
        /// Variable names (RHS + intercept if present)
        variable_names: Vec<String>,
        /// Coefficient estimates
        params: Option<Array1<f64>>,
        /// Standard errors
        std_errors: Option<Array1<f64>>,
        /// Test statistics (t or z)
        test_values: Option<Array1<f64>>,
        /// P-values
        p_values: Option<Array1<f64>>,
        /// Confidence intervals
        conf_lower: Option<Array1<f64>>,
        conf_upper: Option<Array1<f64>>,
        /// Fit statistics (r2, aic, bic, log_lik, n_obs, etc.)
        fit: HashMap<String, Value>,
        /// Residuals (when available)
        residuals: Option<Array1<f64>>,
        /// Fitted values (when available)
        fitted_values: Option<Array1<f64>>,
        /// Design matrix (for diagnostics and predict)
        x: Option<Array2<f64>>,
        /// Estimator-specific extras (kind, y, eq_var_names, var_names, etc.)
        extras: HashMap<String, Value>,
        /// Structured fields for DAP/debug
        fields: Arc<HashMap<String, Value>>,
    },
    List(Arc<Vec<Value>>),
    Dict(Arc<HashMap<String, Value>>),
    Series(Arc<Series>),
    UserFn(Arc<UserFn>),
    Error(Rc<ErrorValue>),
    /// Geometria vetorial em WKT. Produzida por plugins geoespaciais.
    Geometry(String),
    /// Output visual composável. Produzido por plugins de visualização.
    Plot {
        spec: String,
        format: String,
    },
    Nil,
}

impl Value {
    /// Factory method to create a ModelResult from Greeners result fields.
    /// This is the single entry point for converting any Greeners estimator result
    /// into a unified Hayashi ModelResult.
    ///
    /// # Arguments
    /// * `params` - Coefficient estimates
    /// * `std_errors` - Standard errors
    /// * `t_values` - Test statistics (t or z)
    /// * `p_values` - P-values
    /// * `conf_lower` - Confidence interval lower bounds (optional)
    /// * `conf_upper` - Confidence interval upper bounds (optional)
    /// * `r_squared` - R-squared (optional)
    /// * `adj_r_squared` - Adjusted R-squared (optional)
    /// * `f_statistic` - F-statistic (optional)
    /// * `prob_f` - F-test p-value (optional)
    /// * `log_likelihood` - Log-likelihood (optional)
    /// * `aic` - AIC (optional)
    /// * `bic` - BIC (optional)
    /// * `sigma` - Residual standard error (optional)
    /// * `n_obs` - Number of observations
    /// * `n_vars` - Number of variables (params)
    /// * `variable_names` - Variable names (optional)
    /// * `type_name` - Estimator type name for display
    /// * `display` - Full display string
    /// * `residuals` - Residuals (optional)
    /// * `fitted_values` - Fitted values (optional)
    /// * `x` - Design matrix (optional)
    #[allow(clippy::too_many_arguments)]
    pub fn model_result(
        params: Array1<f64>,
        std_errors: Array1<f64>,
        t_values: Array1<f64>,
        p_values: Array1<f64>,
        conf_lower: Option<Array1<f64>>,
        conf_upper: Option<Array1<f64>>,
        r_squared: Option<f64>,
        adj_r_squared: Option<f64>,
        f_statistic: Option<f64>,
        prob_f: Option<f64>,
        log_likelihood: Option<f64>,
        aic: Option<f64>,
        bic: Option<f64>,
        sigma: Option<f64>,
        n_obs: usize,
        n_vars: usize,
        variable_names: Option<Vec<String>>,
        type_name: &'static str,
        display: String,
        residuals: Option<Array1<f64>>,
        fitted_values: Option<Array1<f64>>,
        x: Option<Array2<f64>>,
    ) -> Value {
        let summary = format!(
            "{} (k={}, n={}), pseudo-R2={:.4}",
            type_name,
            n_vars,
            n_obs,
            r_squared.unwrap_or(f64::NAN)
        );

        let variable_names =
            variable_names.unwrap_or_else(|| (0..n_vars).map(|i| format!("x{i}")).collect());

        let params_opt = Some(params);
        let std_errors_opt = Some(std_errors);
        let test_values_opt = Some(t_values);
        let p_values_opt = Some(p_values);

        let mut fit = HashMap::new();
        if let Some(r2) = r_squared {
            fit.insert("r2".into(), Value::Float(r2));
        }
        if let Some(adj_r2) = adj_r_squared {
            fit.insert("adj_r2".into(), Value::Float(adj_r2));
        }
        if let Some(f_stat) = f_statistic {
            fit.insert("f_stat".into(), Value::Float(f_stat));
        }
        if let Some(pf) = prob_f {
            fit.insert("prob_f".into(), Value::Float(pf));
        }
        if let Some(log_lik) = log_likelihood {
            fit.insert("log_lik".into(), Value::Float(log_lik));
        }
        if let Some(aic_val) = aic {
            fit.insert("aic".into(), Value::Float(aic_val));
        }
        if let Some(bic_val) = bic {
            fit.insert("bic".into(), Value::Float(bic_val));
        }
        if let Some(sig) = sigma {
            fit.insert("sigma".into(), Value::Float(sig));
        }
        fit.insert("n_obs".into(), Value::Int(n_obs as i64));

        let extras = HashMap::new();

        let fields: Arc<HashMap<String, Value>> = {
            let mut m = HashMap::new();
            m.insert("display".into(), Value::Str(display.clone()));
            m.insert("summary".into(), Value::Str(summary.clone()));
            m.insert("type_name".into(), Value::Str(type_name.to_string()));
            m.insert(
                "variable_names".into(),
                Value::List(Arc::new(
                    variable_names
                        .iter()
                        .map(|s| Value::Str(s.clone()))
                        .collect(),
                )),
            );
            if let Some(p) = &params_opt {
                m.insert(
                    "params".into(),
                    Value::List(Arc::new(p.iter().map(|&v| Value::Float(v)).collect())),
                );
            }
            if let Some(se) = &std_errors_opt {
                m.insert(
                    "std_errors".into(),
                    Value::List(Arc::new(se.iter().map(|&v| Value::Float(v)).collect())),
                );
            }
            m.insert("fit".into(), Value::Dict(Arc::new(fit.clone())));
            Arc::new(m)
        };

        Value::ModelResult {
            display,
            summary,
            type_name,
            variable_names,
            params: params_opt,
            std_errors: std_errors_opt,
            test_values: test_values_opt,
            p_values: p_values_opt,
            conf_lower,
            conf_upper,
            fit,
            residuals,
            fitted_values,
            x,
            extras,
            fields,
        }
    }
}

// ── Send-safety for parallel for ───────────────────────────────────────────

impl Value {
    /// Returns `true` if this value contains no `Rc`-backed model results,
    /// meaning it can safely cross a thread boundary inside `parallel for`.
    ///
    /// `DataFrame`, `List`, `Dict`, `Series`, `UserFn` use `Arc` (Send).
    /// Primitive variants (`Float`, `Int`, `Bool`, `Str`, `Nil`, `Geometry`,
    /// `Plot`) are inherently Send.  All `*Result` variants wrap `Rc` and are
    /// NOT Send — they cannot be captured into a `parallel for` block.
    pub fn is_send_safe(&self) -> bool {
        match self {
            Value::Float(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Str(_)
            | Value::Nil
            | Value::Geometry(_) => true,
            Value::DataFrame(_) => true,
            Value::List(lst) => lst.iter().all(|v| v.is_send_safe()),
            Value::Dict(d) => d.values().all(|v| v.is_send_safe()),
            Value::Series(_) => true,
            Value::UserFn(_) => true,
            Value::Plot { .. } => true,
            Value::ModelResult { fields, .. } => fields.values().all(|v| v.is_send_safe()),
            Value::Model(_) => false,
            // All model result variants use Rc — not Send.
            _ => false,
        }
    }
}

/// Wrapper that guarantees the inner `Value` is `Send`-safe.
/// Constructed only via `SendValue::new` which checks `is_send_safe`.
#[derive(Clone)]
pub struct SendValue(pub Value);

// SAFETY: SendValue is only constructed from values that pass `is_send_safe`,
// meaning they contain no `Rc`-backed model results. Arc and primitives are Send.
unsafe impl Send for SendValue {}
unsafe impl Sync for SendValue {}

impl SendValue {
    /// Wraps a `Value` if it is send-safe, otherwise returns an error.
    pub fn new(v: Value) -> std::result::Result<Self, String> {
        if v.is_send_safe() {
            Ok(Self(v))
        } else {
            Err(format!(
                "parallel for: captured value is not thread-safe (contains model result with Rc): {v}"
            ))
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Float(v) => write!(f, "{v}"),
            Value::Int(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Str(v) => write!(f, "{v}"),
            Value::DataFrame(df) => write!(f, "{df}"),
            Value::OlsResult(m) => write!(f, "{m}"),
            #[cfg(feature = "greeners-ols")]
            Value::IvResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::BinaryResult(m) => write!(f, "{m}"),
            #[cfg(feature = "greeners-panel")]
            Value::PanelResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-panel")]
            Value::BetweenResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-panel")]
            Value::ReResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::ArimaResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::VarResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::VecmResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::GarchResult(r) => write!(f, "{r}"),
            Value::DiagResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-panel")]
            Value::AbResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::GmmResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-panel")]
            Value::SysGmmResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-panel")]
            Value::FE2SLSResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-panel")]
            Value::PcseResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-panel")]
            Value::PanelGlsResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::TobitResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::HeckmanResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-causal")]
            Value::RdResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-causal")]
            Value::SynthResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-causal")]
            Value::PsmResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::PoissonResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::NegBinResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::OrderedResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::MNLogitResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-causal")]
            Value::DidResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::QuantileResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-survival")]
            Value::KMResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-survival")]
            Value::CoxResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::RlmResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::GeeResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::ZeroInflatedResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-bayesian")]
            Value::MixedResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::BetaResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::GlsarResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::SurResult(m) => write!(f, "{m}"),
            #[cfg(feature = "greeners-ols")]
            Value::RollingResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::RecursiveLSResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::GlmResult(r) => write!(f, "{r}"),
            Value::LowessResult(r) => write!(f, "{r}"),
            Value::PcaResult(m) => write!(f, "{m}"),
            Value::FactorResult(m) => write!(f, "{m}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::MarkovResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::ConditionalResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::VarmaResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::DecompResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::MstlResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::UCResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-glm")]
            Value::GamResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-imputation")]
            Value::MiceResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::MSARResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::SVarResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::ThreeSLSResult(m) => write!(f, "{m}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::DFMResult(m) => write!(f, "{m}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::EtsResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ols")]
            Value::PenalizedResult(m) => write!(f, "{m}"),
            #[cfg(feature = "greeners-panel")]
            Value::ThresholdResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::AutoRegResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::ArdlResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            Value::LocalLevelResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ml")]
            Value::KmeansResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ml")]
            Value::DbscanResult(r) => write!(f, "{r}"),
            Value::IsotonicResult(r) => write!(f, "{r}"),
            Value::KdeResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ml")]
            Value::BartResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ml")]
            Value::GpResult(r) => write!(f, "{r}"),
            Value::GmmClusteringResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-ml")]
            Value::HierarchicalResult(r) => write!(f, "{r}"),
            #[cfg(feature = "greeners-timeseries")]
            #[cfg(feature = "experimental")]
            Value::SpectralResult(r) => write!(f, "{r}"),
            Value::Model(m) => write!(f, "{m}"),
            Value::ModelResult { display, .. } => write!(f, "{display}"),
            Value::List(v) => {
                write!(f, "[")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Value::Dict(m) => {
                write!(f, "{{")?;
                let mut sorted: Vec<_> = m.iter().collect();
                sorted.sort_by_key(|(k, _)| (*k).clone());
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{k}\": {v}")?;
                }
                write!(f, "}}")
            }
            Value::Series(s) => {
                write!(f, "Series({}: [", s.name)?;
                for (i, v) in s.values.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if i >= 5 && s.values.len() > 10 {
                        write!(f, "... ({} items)", s.values.len() - 10)?;
                        break;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "])")
            }
            Value::UserFn(f_) => write!(f, "<fn({})>", f_.params.join(", ")),
            Value::Error(e) => {
                write!(f, "Error({}: {}", e.kind, e.msg)?;
                if e.line > 0 {
                    write!(f, " at line {}", e.line)?;
                }
                write!(f, ")")
            }
            Value::Geometry(wkt) => {
                let preview = if wkt.len() > 60 {
                    &wkt[..60]
                } else {
                    wkt.as_str()
                };
                write!(f, "Geometry({preview}...)")
            }
            Value::Plot { spec, format } => {
                if format == "latex" || format == "html" || format == "markdown" {
                    write!(f, "{spec}")
                } else {
                    write!(f, "Plot({format})")
                }
            }
            Value::Nil => write!(f, "nil"),
        }
    }
}
