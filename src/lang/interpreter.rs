use crate::lang::ast::*;
use crate::lang::error::{HayashiError, Result};
use greeners::diagnostics::Diagnostics;
use greeners::linalg::UPLO;
use greeners::linalg::{LinalgEigh as _, LinalgInverse as _};
use greeners::specification_tests::SpecificationTests;
use greeners::{chi2_pvalue, f_pvalue, logistic, norm_pdf, t_pvalue_two, t_quantile};
use greeners::{
    CovarianceType, DataFrame, FixedEffects, Formula as GFormula, Logit, Probit, RandomEffects, IV,
    OLS,
};
use ndarray::{Array1, Array2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

// ── eval_call() dividido por categoria (ver src/lang/interpreter/) ──────────
// Cada submódulo implementa `impl Interpreter { fn eval_call_X(...) -> Result<Option<Value>> }`
// Retorna `Ok(None)` quando `func` não pertence à categoria, para o dispatcher tentar a próxima.
mod data_manipulation;
mod estimators_misc;
mod estimators_timeseries;
mod post_estimation_ts;
mod visualization;

fn t_critical_95(df: f64) -> f64 {
    t_quantile(0.975, df)
}

fn rd_kernel_opt(opt: Option<&Value>) -> std::result::Result<greeners::RdKernel, String> {
    match opt {
        None => Ok(greeners::RdKernel::Triangular),
        Some(Value::Str(s)) => match s.as_str() {
            "triangular" | "tri" => Ok(greeners::RdKernel::Triangular),
            "uniform" | "uni" => Ok(greeners::RdKernel::Uniform),
            "epanechnikov" | "epa" => Ok(greeners::RdKernel::Epanechnikov),
            other => Err(format!(
                "kernel '{other}' unknown (triangular|uniform|epanechnikov)"
            )),
        },
        _ => Err("kernel must be string".into()),
    }
}

fn standard_normal_draw<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = 1.0 - rng.gen::<f64>();
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ── Wrappers que preservam a matriz X para diagnósticos e predict ────────────

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
pub struct BinaryModel {
    pub result: Rc<greeners::discrete::BinaryModelResult>,
    pub y: Array1<f64>,
    pub x: Array2<f64>,
    pub kind: String,            // "logit" | "probit"
    pub coef_names: Vec<String>, // nomes dos coeficientes para margins
}

impl std::fmt::Display for BinaryModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── SUR wrapper (preserva nomes de variáveis por equação) ────────────────────

#[derive(Clone)]
pub struct SurModel {
    pub result: Rc<greeners::sur::SurResult>,
    pub eq_var_names: Vec<Vec<String>>, // nomes por equação
}

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

// ── PCA wrapper (adiciona nomes de variáveis ao PCAResult) ───────────────────
#[derive(Clone)]
pub struct PcaModel {
    pub result: Rc<greeners::PCAResult>,
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
        writeln!(f, " {:>20}  {:>10}", "Observações:", r.n_obs)?;
        writeln!(f, " {:>20}  {:>10}", "Componentes:", r.n_components)?;
        writeln!(f, " {:>20}  {:>10}", "Variáveis:", self.var_names.len())?;
        writeln!(
            f,
            "\n{:^12} {:>12} {:>12} {:>10}",
            "Componente", "Var Expl.", "% Acum.", "Eigenvalue"
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
        writeln!(f, "{:<18}{hdr}", "Variável")?;
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
    pub result: Rc<greeners::FactorResult>,
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
        writeln!(f, " {:>20}  {:>10}", "Observações:", r.n_obs)?;
        writeln!(f, " {:>20}  {:>10}", "Fatores:", r.n_factors)?;
        writeln!(f, "\n{:^62}", " Cargas Fatoriais (Loadings) ")?;
        writeln!(f, "{thin}")?;
        let hdr: String = (0..r.n_factors)
            .map(|i| format!(" {:>8}", format!("F{}", i + 1)))
            .collect();
        writeln!(f, "{:<18}{hdr}  {:>10}", "Variável", "Comunalit.")?;
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

// ── Função definida pelo usuário ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UserFn {
    pub params: Vec<String>,
    pub body: Vec<Spanned>,
}

// ── Resultado de testes de diagnóstico (print-on-demand) ─────────────────────

#[derive(Debug, Clone)]
pub struct DiagResult {
    pub rendered: String, // output pré-renderizado pelo teste
}

impl std::fmt::Display for DiagResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.rendered)
    }
}

// ── DFM wrapper ───────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct DFMModel {
    pub result: Rc<greeners::DynamicFactorResult>,
    #[allow(dead_code)]
    pub var_names: Vec<String>,
}

impl std::fmt::Display for DFMModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── 3SLS wrapper ──────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct ThreeSLSModel {
    pub result: Rc<greeners::three_sls::ThreeSLSResult>,
    #[allow(dead_code)]
    pub eq_var_names: Vec<Vec<String>>,
}

impl std::fmt::Display for ThreeSLSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── Valores em runtime ────────────────────────────────────────────────────────

#[derive(Clone)]
pub enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    DataFrame(Rc<DataFrame>),
    OlsResult(OlsModel),
    IvResult(Rc<greeners::iv::IvResult>),
    BinaryResult(BinaryModel),
    PanelResult(Rc<greeners::panel::PanelResult>),
    ReResult(Rc<greeners::panel::RandomEffectsResult>),
    ArimaResult(Rc<greeners::ArimaResult>),
    VarResult(Rc<greeners::var::VarResult>),
    VecmResult(Rc<greeners::vecm::VecmResult>),
    GarchResult(Rc<greeners::GarchResult>),
    DiagResult(Rc<DiagResult>),
    AbResult(Rc<greeners::ArellanoBondResult>),
    GmmResult(Rc<greeners::GmmResult>),
    SysGmmResult(Rc<greeners::SystemGmmResult>),
    FE2SLSResult(Rc<greeners::PanelIvResult>),
    PcseResult(Rc<greeners::PcseResult>),
    PanelGlsResult(Rc<greeners::PanelGlsResult>),
    TobitResult(Rc<greeners::TobitResult>),
    HeckmanResult(Rc<greeners::HeckmanResult>),
    RdResult(Rc<greeners::RdResult>),
    SynthResult(Rc<greeners::SynthResult>),
    PsmResult(Rc<greeners::PsmResult>),
    PoissonResult(Rc<greeners::PoissonResult>),
    NegBinResult(Rc<greeners::NegBinResult>),
    OrderedResult(Rc<greeners::OrderedResult>),
    MNLogitResult(Rc<greeners::MNLogitResult>),
    DidResult(Rc<greeners::DidResult>),
    QuantileResult(Rc<greeners::QuantileResult>),
    KMResult(Rc<greeners::KMResult>),
    CoxResult(Rc<greeners::CoxResult>),
    RlmResult(Rc<greeners::RlmResult>),
    GeeResult(Rc<greeners::GeeResult>),
    ZeroInflatedResult(Rc<greeners::ZeroInflatedResult>),
    MixedResult(Rc<greeners::MixedResult>),
    BetaResult(Rc<greeners::BetaResult>),
    GlsarResult(Rc<greeners::GlsarResult>),
    SurResult(SurModel),
    RollingResult(Rc<greeners::RollingResult>),
    RecursiveLSResult(Rc<greeners::RecursiveLSResult>),
    GlmResult(Rc<greeners::GlmResult>),
    LowessResult(Rc<greeners::LowessResult>),
    PcaResult(PcaModel),
    FactorResult(FactorModel),
    MarkovResult(Rc<greeners::MarkovSwitchingResult>),
    ConditionalResult(Rc<greeners::ConditionalResult>),
    VarmaResult(Rc<greeners::varma::VarmaResult>),
    DecompResult(Rc<greeners::DecompositionResult>),
    MstlResult(Rc<greeners::MSTLResult>),
    UCResult(Rc<greeners::UCResult>),
    GamResult(Rc<greeners::GamResult>),
    MiceResult(Rc<greeners::MICEResult>),
    MSARResult(Rc<greeners::MarkovAutoregResult>),
    SVarResult(Rc<greeners::SVarResult>),
    ThreeSLSResult(ThreeSLSModel),
    DFMResult(DFMModel),
    EtsResult(Rc<greeners::ETSResult>),
    ThresholdResult(Rc<greeners::threshold::ThresholdResult>),
    AutoRegResult(Rc<greeners::AutoRegResult>),
    ArdlResult(Rc<greeners::ARDLResult>),
    List(Rc<Vec<Value>>),
    Dict(Rc<std::collections::HashMap<String, Value>>),
    UserFn(Rc<UserFn>),
    Nil,
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
            Value::IvResult(r) => write!(f, "{r}"),
            Value::BinaryResult(m) => write!(f, "{m}"),
            Value::PanelResult(r) => write!(f, "{r}"),
            Value::ReResult(r) => write!(f, "{r}"),
            Value::ArimaResult(r) => write!(f, "{r}"),
            Value::VarResult(r) => write!(f, "{r}"),
            Value::VecmResult(r) => write!(f, "{r}"),
            Value::GarchResult(r) => write!(f, "{r}"),
            Value::DiagResult(r) => write!(f, "{r}"),
            Value::AbResult(r) => write!(f, "{r}"),
            Value::GmmResult(r) => write!(f, "{r}"),
            Value::SysGmmResult(r) => write!(f, "{r}"),
            Value::FE2SLSResult(r) => write!(f, "{r}"),
            Value::PcseResult(r) => write!(f, "{r}"),
            Value::PanelGlsResult(r) => write!(f, "{r}"),
            Value::TobitResult(r) => write!(f, "{r}"),
            Value::HeckmanResult(r) => write!(f, "{r}"),
            Value::RdResult(r) => write!(f, "{r}"),
            Value::SynthResult(r) => write!(f, "{r}"),
            Value::PsmResult(r) => write!(f, "{r}"),
            Value::PoissonResult(r) => write!(f, "{r}"),
            Value::NegBinResult(r) => write!(f, "{r}"),
            Value::OrderedResult(r) => write!(f, "{r}"),
            Value::MNLogitResult(r) => write!(f, "{r}"),
            Value::DidResult(r) => write!(f, "{r}"),
            Value::QuantileResult(r) => write!(f, "{r}"),
            Value::KMResult(r) => write!(f, "{r}"),
            Value::CoxResult(r) => write!(f, "{r}"),
            Value::RlmResult(r) => write!(f, "{r}"),
            Value::GeeResult(r) => write!(f, "{r}"),
            Value::ZeroInflatedResult(r) => write!(f, "{r}"),
            Value::MixedResult(r) => write!(f, "{r}"),
            Value::BetaResult(r) => write!(f, "{r}"),
            Value::GlsarResult(r) => write!(f, "{r}"),
            Value::SurResult(m) => write!(f, "{m}"),
            Value::RollingResult(r) => write!(f, "{r}"),
            Value::RecursiveLSResult(r) => write!(f, "{r}"),
            Value::GlmResult(r) => write!(f, "{r}"),
            Value::LowessResult(r) => write!(f, "{r}"),
            Value::PcaResult(m) => write!(f, "{m}"),
            Value::FactorResult(m) => write!(f, "{m}"),
            Value::MarkovResult(r) => write!(f, "{r}"),
            Value::ConditionalResult(r) => write!(f, "{r}"),
            Value::VarmaResult(r) => write!(f, "{r}"),
            Value::DecompResult(r) => write!(f, "{r}"),
            Value::MstlResult(r) => write!(f, "{r}"),
            Value::UCResult(r) => write!(f, "{r}"),
            Value::GamResult(r) => write!(f, "{r}"),
            Value::MiceResult(r) => write!(f, "{r}"),
            Value::MSARResult(r) => write!(f, "{r}"),
            Value::SVarResult(r) => write!(f, "{r}"),
            Value::ThreeSLSResult(m) => write!(f, "{m}"),
            Value::DFMResult(m) => write!(f, "{m}"),
            Value::EtsResult(r) => write!(f, "{r}"),
            Value::ThresholdResult(r) => write!(f, "{r}"),
            Value::AutoRegResult(r) => write!(f, "{r}"),
            Value::ArdlResult(r) => write!(f, "{r}"),
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
            Value::UserFn(f_) => write!(f, "<fn({})>", f_.params.join(", ")),
            Value::Nil => write!(f, "nil"),
        }
    }
}

// ── Ambiente de variáveis ─────────────────────────────────────────────────────

struct Scope {
    vars: HashMap<String, Value>,
    consts: HashSet<String>,
}

impl Scope {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
            consts: HashSet::new(),
        }
    }
}

pub struct Env {
    scopes: Vec<Scope>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    pub fn declare(&mut self, name: &str, val: Value) -> Result<()> {
        for scope in self.scopes.iter().rev() {
            if scope.consts.contains(name) {
                return Err(HayashiError::Runtime(format!(
                    "cannot redeclare const '{name}'"
                )));
            }
        }
        self.scopes
            .last_mut()
            .unwrap()
            .vars
            .insert(name.to_string(), val);
        Ok(())
    }

    pub fn declare_const(&mut self, name: &str, val: Value) {
        let scope = self.scopes.last_mut().unwrap();
        scope.vars.insert(name.to_string(), val);
        scope.consts.insert(name.to_string());
    }

    pub fn set(&mut self, name: &str, val: Value) -> Result<()> {
        for scope in self.scopes.iter().rev() {
            if scope.consts.contains(name) {
                return Err(HayashiError::Runtime(format!(
                    "cannot reassign const '{name}'"
                )));
            }
        }
        for scope in self.scopes.iter_mut().rev() {
            if scope.vars.contains_key(name) {
                scope.vars.insert(name.to_string(), val);
                return Ok(());
            }
        }
        self.scopes
            .last_mut()
            .unwrap()
            .vars
            .insert(name.to_string(), val);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.vars.get(name) {
                return Some(v);
            }
        }
        None
    }

    pub fn all_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .scopes
            .iter()
            .flat_map(|s| s.vars.keys().cloned())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    pub fn remove(&mut self, name: &str) {
        for scope in self.scopes.iter_mut().rev() {
            if scope.vars.remove(name).is_some() {
                scope.consts.remove(name);
                return;
            }
        }
    }

    pub fn var_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for scope in self.scopes.iter().rev() {
            for key in scope.vars.keys() {
                if !names.contains(key) {
                    names.push(key.clone());
                }
            }
        }
        names
    }
}

// ── Interpetador ──────────────────────────────────────────────────────────────

const BUILTIN_NAMES: &[&str] = &[
    "mean",
    "sd",
    "min",
    "max",
    "sum",
    "total",
    "median",
    "variance",
    "quantile",
    "cov",
    "corr_pair",
    "abs",
    "sqrt",
    "ln",
    "log",
    "exp",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "ceil",
    "floor",
    "round",
    "sign",
    "factorial",
    "comb",
    "int",
    "float",
    "str",
    "bool",
    "len",
    "typeof",
    "ols",
    "iv",
    "logit",
    "probit",
    "poisson",
    "nbreg",
    "tobit",
    "heckman",
    "fe",
    "re",
    "be",
    "fe2sls",
    "ab",
    "sysgmm",
    "pcse",
    "xtgls",
    "qreg",
    "rlm",
    "lasso",
    "ridge",
    "elasticnet",
    "cox",
    "arima",
    "autoreg",
    "ardl",
    "kalman",
    "var",
    "vecm",
    "varma",
    "svar",
    "garch",
    "glm",
    "gee",
    "mixed",
    "mlogit",
    "ologit",
    "oprobit",
    "clogit",
    "cpoisson",
    "gmm",
    "sur",
    "three_sls",
    "fmb",
    "did",
    "rd",
    "psm",
    "synth",
    "summarize",
    "tabulate",
    "tabstat",
    "correlate",
    "corr",
    "pwcorr",
    "describe",
    "codebook",
    "ttest",
    "ci",
    "centile",
    "count",
    "nrow",
    "filter",
    "sort",
    "drop",
    "keep",
    "select",
    "dropna",
    "rename",
    "merge",
    "append",
    "collapse",
    "group_by",
    "reshape",
    "mutate",
    "generate",
    "pivot_longer",
    "pivot_wider",
    "anova",
    "pca",
    "factor",
    "manova",
    "cancorr",
    "kde",
    "lowess",
    "swilk",
    "sfrancia",
    "sktest",
    "omnibus",
    "dagostino",
    "vif",
    "predict",
    "esttab",
    "eststo",
    "margins",
    "test",
    "lincom",
    "nlcom",
    "bootstrap",
    "bootse",
    "histogram",
    "boxplot",
    "kdensity",
    "qqplot",
    "scatter",
    "recode",
    "destring",
    "winsor",
    "label",
    "format",
    "print",
    "display",
    "source",
    "import",
    "assert",
    "timer",
    "push",
    "pop",
    "reverse",
    "unique",
    "flatten",
    "chain",
    "join",
    "split",
    "contains",
    "starts_with",
    "ends_with",
    "lower",
    "upper",
    "trim",
    "substr",
    "replace",
    "regexm",
    "regexr",
    "regexs",
    "input",
    "load",
    "export",
    "write",
];

pub struct Interpreter {
    pub env: Env,
    ts_info: HashMap<String, String>,
    panel_info: HashMap<String, (String, String)>,
    rng_seed: Option<u64>,
    rng: rand::rngs::StdRng,
    preserved: HashMap<String, Value>,
    stored_models: Vec<Value>,
    return_value: Option<Value>,
    labels: HashMap<String, HashMap<String, String>>,
    current_line: usize,
    imported: HashSet<String>,
    plugin_paths: Vec<String>,
    pub plugins: HashMap<String, Box<dyn super::plugin::HayashiPlugin>>,
    capturing: bool,
    call_stack: Vec<(String, usize)>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            env: Env::new(),
            ts_info: HashMap::new(),
            panel_info: HashMap::new(),
            rng_seed: None,
            rng: {
                use rand::SeedableRng;
                rand::rngs::StdRng::from_entropy()
            },
            preserved: HashMap::new(),
            stored_models: Vec::new(),
            return_value: None,
            labels: HashMap::new(),
            current_line: 0,
            imported: HashSet::new(),
            plugin_paths: Vec::new(),
            plugins: HashMap::new(),
            capturing: false,
            call_stack: Vec::new(),
        }
    }

    fn levenshtein(a: &str, b: &str) -> usize {
        let (_m, n) = (a.len(), b.len());
        let mut prev: Vec<usize> = (0..=n).collect();
        let mut curr = vec![0; n + 1];
        for (i, ca) in a.chars().enumerate() {
            curr[0] = i + 1;
            for (j, cb) in b.chars().enumerate() {
                let cost = if ca == cb { 0 } else { 1 };
                curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[n]
    }

    fn suggest(name: &str, candidates: &[String]) -> Option<String> {
        let max_dist = match name.len() {
            0..=2 => 1,
            3..=5 => 2,
            _ => 3,
        };
        candidates
            .iter()
            .filter_map(|c| {
                let d = Self::levenshtein(name, c);
                if d > 0 && d <= max_dist {
                    Some((d, c.clone()))
                } else {
                    None
                }
            })
            .min_by_key(|(d, _)| *d)
            .map(|(_, c)| c)
    }

    #[allow(dead_code)]
    fn format_stack_trace(&self, innermost: &str, line: usize) -> String {
        let mut frames = Vec::new();
        for (name, ln) in self.call_stack.iter().rev() {
            frames.push(format!("  in {name}() at line {ln}"));
        }
        frames.push(format!("  in {innermost}() at line {line}"));
        format!("Stack trace:\n{}", frames.join("\n"))
    }

    fn rt_err(&self, msg: impl Into<String>) -> HayashiError {
        let m = msg.into();
        if self.current_line > 0 {
            HayashiError::Runtime(format!("line {}: {}", self.current_line, m))
        } else {
            HayashiError::Runtime(m)
        }
    }

    fn type_name(v: &Value) -> &'static str {
        match v {
            Value::Float(_) => "Float",
            Value::Int(_) => "Int",
            Value::Bool(_) => "Bool",
            Value::Str(_) => "String",
            Value::Nil => "Nil",
            Value::List(_) => "List",
            Value::Dict(_) => "Dict",
            Value::DataFrame(_) => "DataFrame",
            Value::UserFn(_) => "Function",
            Value::OlsResult(_) => "OlsResult",
            Value::IvResult(_) => "IvResult",
            _ => "Object",
        }
    }

    fn type_mismatch(&self, expected: &str, got: &Value) -> HayashiError {
        self.type_err(format!("expected {expected}, got {}", Self::type_name(got)))
    }

    fn binary_mle_vcov(
        kind: &str,
        params: &Array1<f64>,
        y: &Array1<f64>,
        x: &Array2<f64>,
    ) -> Option<Array2<f64>> {
        if x.ncols() != params.len() || x.nrows() != y.len() {
            return None;
        }

        let xb = x.dot(params);
        let mut x_weighted = x.to_owned();
        let normal_dist = match kind {
            "logit" => None,
            "probit" => Some(Normal::new(0.0, 1.0).ok()?),
            _ => return None,
        };

        for (i, mut row) in x_weighted.axis_iter_mut(Axis(0)).enumerate() {
            let weight = if kind == "logit" {
                let p = logistic(xb[i]);
                p * (1.0 - p)
            } else {
                // Observed-information probit Hessian, matching statsmodels:
                // -H = X' diag(lambda_i * (lambda_i + x_i'beta)) X.
                let q = if y[i] > 0.5 { 1.0 } else { -1.0 };
                let qxb = q * xb[i];
                let p = normal_dist
                    .as_ref()
                    .map(|dist| dist.cdf(qxb))
                    .unwrap_or(f64::NAN)
                    .clamp(1e-10, 1.0 - 1e-10);
                let lambda = q * norm_pdf(qxb) / p;
                lambda * (lambda + xb[i])
            };

            if !weight.is_finite() || weight <= 0.0 {
                return None;
            }
            row *= weight;
        }

        x.t().dot(&x_weighted).inv().ok()
    }

    fn eval_as_int(&mut self, expr: &Expr, ctx: &str) -> Result<i64> {
        match self.eval_expr(expr)? {
            Value::Int(i) => Ok(i),
            Value::Float(f) => Ok(f as i64),
            v => Err(self.type_err(format!(
                "{ctx} must be integer, got {}",
                Self::type_name(&v)
            ))),
        }
    }

    fn resolve_var_list(&mut self, args: &[Expr], df: &greeners::DataFrame) -> Result<Vec<String>> {
        let col_names = df.column_names();
        let mut names = Vec::new();
        for a in args {
            match a {
                Expr::Str(s) => names.push(s.clone()),
                Expr::Var(name) if col_names.contains(name) => {
                    names.push(name.clone());
                }
                other => match self.eval_expr(other)? {
                    Value::Str(s) => names.push(s),
                    Value::List(lst) => {
                        for v in lst.iter() {
                            match v {
                                Value::Str(s) => names.push(s.clone()),
                                _ => {
                                    return Err(self.type_err("variable list items must be strings"))
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(
                            self.type_err("expected column name, string, or list of strings")
                        )
                    }
                },
            }
        }
        Ok(names)
    }

    fn type_err(&self, msg: impl Into<String>) -> HayashiError {
        let m = msg.into();
        if self.current_line > 0 {
            HayashiError::Type(format!("line {}: {}", self.current_line, m))
        } else {
            HayashiError::Type(m)
        }
    }

    fn call_value_fn(&mut self, f: &Value, args: &[Value]) -> Result<Value> {
        match f {
            Value::UserFn(uf) => {
                self.env.push_scope();
                for (param, val) in uf.params.iter().zip(args.iter()) {
                    self.env.declare_const(param, val.clone());
                }
                let body = uf.body.clone();
                let mut ret = Value::Nil;
                for s in &body {
                    match self.exec(s) {
                        Ok(()) => {}
                        Err(HayashiError::Return) => break,
                        Err(e) => {
                            self.env.pop_scope();
                            return Err(e);
                        }
                    }
                }
                if let Some(rv) = self.return_value.take() {
                    ret = rv;
                }
                self.env.pop_scope();
                Ok(ret)
            }
            _ => Err(self.rt_err("expected a function or closure")),
        }
    }

    fn dict_to_dataframe(&self, map: &HashMap<String, Value>) -> Result<greeners::DataFrame> {
        let mut columns = HashMap::new();
        let mut expected_len: Option<usize> = None;

        for (col_name, val) in map {
            let list = match val {
                Value::List(lst) => lst,
                _ => return Err(self.type_err(format!("column '{col_name}' must be a list"))),
            };

            let len = list.len();
            if let Some(expected) = expected_len {
                if len != expected {
                    return Err(self.rt_err(format!(
                        "all columns must have the same length (column '{}' has length {}, expected {})",
                        col_name, len, expected
                    )));
                }
            } else {
                expected_len = Some(len);
            }

            if len == 0 {
                columns.insert(
                    col_name.clone(),
                    greeners::Column::Float(ndarray::Array1::from(vec![])),
                );
                continue;
            }

            let first = &list[0];
            let col = match first {
                Value::Float(_) => {
                    let mut data = Vec::with_capacity(len);
                    for (i, v) in list.iter().enumerate() {
                        match v {
                            Value::Float(f) => data.push(*f),
                            Value::Int(i_val) => data.push(*i_val as f64),
                            other => {
                                return Err(self.type_err(format!(
                                    "element at index {} of column '{}' is not numeric (got {})",
                                    i, col_name, other
                                )))
                            }
                        }
                    }
                    greeners::Column::Float(ndarray::Array1::from(data))
                }
                Value::Int(_) => {
                    let mut data = Vec::with_capacity(len);
                    for (i, v) in list.iter().enumerate() {
                        match v {
                            Value::Int(i_val) => data.push(*i_val),
                            Value::Float(f) => data.push(*f as i64),
                            other => {
                                return Err(self.type_err(format!(
                                    "element at index {} of column '{}' is not an integer (got {})",
                                    i, col_name, other
                                )))
                            }
                        }
                    }
                    greeners::Column::Int(ndarray::Array1::from(data))
                }
                Value::Bool(_) => {
                    let mut data = Vec::with_capacity(len);
                    for (i, v) in list.iter().enumerate() {
                        match v {
                            Value::Bool(b) => data.push(*b),
                            other => {
                                return Err(self.type_err(format!(
                                    "element at index {} of column '{}' is not boolean (got {})",
                                    i, col_name, other
                                )))
                            }
                        }
                    }
                    greeners::Column::Bool(ndarray::Array1::from(data))
                }
                Value::Str(_) => {
                    let mut data = Vec::with_capacity(len);
                    for (i, v) in list.iter().enumerate() {
                        match v {
                            Value::Str(s) => data.push(s.clone()),
                            other => {
                                return Err(self.type_err(format!(
                                    "element at index {} of column '{}' is not a string (got {})",
                                    i, col_name, other
                                )))
                            }
                        }
                    }
                    greeners::Column::from_strings(data)
                }
                other => {
                    return Err(self.type_err(format!(
                        "unsupported type for column '{}': {}",
                        col_name, other
                    )))
                }
            };

            columns.insert(col_name.clone(), col);
        }

        if expected_len.is_none() {
            return Err(self.rt_err("cannot create empty dataframe (no columns)"));
        }

        greeners::DataFrame::from_columns(columns)
            .map_err(|e| self.rt_err(format!("failed to create dataframe: {e}")))
    }

    pub fn load_plugins(&mut self) {
        let home = match std::env::var_os("HOME") {
            Some(h) => h,
            None => return,
        };
        let plugin_dir = std::path::Path::new(&home).join(".hay").join("plugins");
        if !plugin_dir.is_dir() {
            return;
        }

        let mut entries: Vec<_> = match std::fs::read_dir(&plugin_dir) {
            Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
            Err(_) => return,
        };
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("hay") {
                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                if self.imported.contains(&name) {
                    continue;
                }
                if let Ok(src) = std::fs::read_to_string(&path) {
                    self.imported.insert(name);
                    let _ = crate::lang::run_source(&src, self);
                }
            }
        }
    }

    fn resolve_import(&self, name: &str) -> Result<String> {
        let has_ext = name.ends_with(".hay")
            || name.ends_with(".wasm")
            || name.ends_with(".so")
            || name.ends_with(".dll")
            || name.ends_with(".dylib");
        let candidates = if has_ext {
            vec![name.to_string()]
        } else {
            vec![
                format!("{name}.hay"),
                format!("{name}.wasm"),
                format!("{name}.so"),
                format!("{name}.dll"),
                format!("{name}.dylib"),
            ]
        };

        for cand in &candidates {
            let is_native_or_wasm = cand.ends_with(".wasm")
                || cand.ends_with(".so")
                || cand.ends_with(".dll")
                || cand.ends_with(".dylib");

            // No perfil de release (produção), restringimos plugins nativos/WASM
            // a serem carregados exclusivamente de ~/.hay/packages/.
            let restrict_to_packages = is_native_or_wasm && !cfg!(debug_assertions);

            // 1. Current directory
            if !restrict_to_packages && std::path::Path::new(cand).exists() {
                return Ok(cand.to_string());
            }

            // 2. ~/.hay/plugins/
            if !restrict_to_packages {
                if let Some(home) = std::env::var_os("HOME") {
                    let plugin_path = std::path::Path::new(&home)
                        .join(".hay")
                        .join("plugins")
                        .join(cand);
                    if plugin_path.exists() {
                        return Ok(plugin_path.to_string_lossy().to_string());
                    }
                }
            }

            // 3. ~/.hay/packages/ (installed packages)
            if let Some(home) = std::env::var_os("HOME") {
                let pkg_path = std::path::Path::new(&home)
                    .join(".hay")
                    .join("packages")
                    .join(cand);
                if pkg_path.exists() {
                    return Ok(pkg_path.to_string_lossy().to_string());
                }
            }

            // 4. User-declared plugin_paths
            if !restrict_to_packages {
                for dir in &self.plugin_paths {
                    let p = std::path::Path::new(dir).join(cand);
                    if p.exists() {
                        return Ok(p.to_string_lossy().to_string());
                    }
                }
            }

            // 5. HAYASHI_PATH env var (colon-separated)
            if !restrict_to_packages {
                if let Ok(paths) = std::env::var("HAYASHI_PATH") {
                    for dir in paths.split(':') {
                        let p = std::path::Path::new(dir).join(cand);
                        if p.exists() {
                            return Ok(p.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        Err(HayashiError::Runtime(format!(
            "import: module '{}' not found (searched: ./, ~/.hay/plugins/, plugin_path, $HAYASHI_PATH)",
            name
        )))
    }

    pub fn get_rng(&mut self) -> Box<dyn rand::RngCore> {
        use rand::{RngCore, SeedableRng};
        let derived_seed = self.rng.next_u64();
        Box::new(rand::rngs::StdRng::seed_from_u64(derived_seed))
    }

    // ── Avalia expressão ──────────────────────────────────────────────────────

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Float(v) => Ok(Value::Float(*v)),
            Expr::Int(v) => Ok(Value::Int(*v)),
            Expr::Bool(v) => Ok(Value::Bool(*v)),
            Expr::Str(v) => Ok(Value::Str(v.clone())),
            Expr::Nil => Ok(Value::Nil),

            Expr::FString(template) => {
                let mut result = String::new();
                let mut chars = template.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '{' {
                        if chars.peek() == Some(&'{') {
                            chars.next();
                            result.push('{');
                            continue;
                        }
                        let mut expr_str = String::new();
                        let mut fmt_spec = String::new();
                        let mut in_fmt = false;
                        let mut depth = 1;
                        for c2 in chars.by_ref() {
                            if c2 == '{' {
                                depth += 1;
                            }
                            if c2 == '}' {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            if c2 == ':' && depth == 1 && !in_fmt {
                                in_fmt = true;
                                continue;
                            }
                            if in_fmt {
                                fmt_spec.push(c2);
                            } else {
                                expr_str.push(c2);
                            }
                        }
                        let mut lexer = crate::lang::lexer::Lexer::new(&expr_str);
                        let tokens = lexer.tokenize()?;
                        let mut parser = crate::lang::parser::Parser::new(tokens);
                        let expr = parser.parse_expr()?;
                        let val = self.eval_expr(&expr)?;
                        if fmt_spec.is_empty() {
                            result.push_str(&format!("{val}"));
                        } else {
                            let num = match &val {
                                Value::Float(f) => *f,
                                Value::Int(i) => *i as f64,
                                _ => {
                                    result.push_str(&format!("{val}"));
                                    continue;
                                }
                            };
                            let formatted = match fmt_spec.as_str() {
                                s if s.starts_with('.') && s.ends_with('f') => {
                                    let prec: usize = s[1..s.len() - 1].parse().unwrap_or(2);
                                    format!("{num:.prec$}")
                                }
                                s if s.starts_with('.') && s.ends_with('e') => {
                                    let prec: usize = s[1..s.len() - 1].parse().unwrap_or(2);
                                    format!("{num:.prec$e}")
                                }
                                _ => format!("{val}"),
                            };
                            result.push_str(&formatted);
                        }
                    } else if c == '}' {
                        if chars.peek() == Some(&'}') {
                            chars.next();
                        }
                        result.push('}');
                    } else {
                        result.push(c);
                    }
                }
                Ok(Value::Str(result))
            }

            Expr::Var(name) => self.env.get(name).cloned().ok_or_else(|| {
                let known = self.env.all_names();
                let hint = Self::suggest(name, &known)
                    .map(|s| format!(" — did you mean '{s}'?"))
                    .unwrap_or_default();
                self.rt_err(format!("undefined variable '{name}'{hint}"))
            }),

            Expr::Formula(_f) => Err(HayashiError::Runtime(
                "formula must be used inside an estimator call".into(),
            )),

            Expr::Closure { params, body } => Ok(Value::UserFn(Rc::new(UserFn {
                params: params.clone(),
                body: vec![(Stmt::Return(Some(*body.clone())), 0)],
            }))),

            Expr::Apply { func, args } => {
                let closure_val = self.eval_expr(func)?;
                let uf = match closure_val {
                    Value::UserFn(f) => f,
                    _ => return Err(self.rt_err("apply: expected function or closure")),
                };
                let arg_vals: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<_>>()?;

                self.env.push_scope();
                for (param, val) in uf.params.iter().zip(arg_vals) {
                    self.env.declare_const(param, val);
                }
                let body = uf.body.clone();
                let mut exec_err: Option<HayashiError> = None;
                for s in &body {
                    match self.exec(s) {
                        Ok(()) => {}
                        Err(HayashiError::Return) => break,
                        Err(e) => {
                            exec_err = Some(e);
                            break;
                        }
                    }
                }
                self.env.pop_scope();
                if let Some(e) = exec_err {
                    return Err(e);
                }
                Ok(self.return_value.take().unwrap_or(Value::Nil))
            }

            Expr::Pipe { expr, .. } => self.eval_expr(expr),

            Expr::Match { expr, arms } => {
                let scrutinee = self.eval_expr(expr)?;
                let scrutinee_str = format!("{scrutinee}");
                for (pattern, result) in arms {
                    let is_wildcard = matches!(pattern, Expr::Var(n) if n == "_");
                    if is_wildcard {
                        return self.eval_expr(result);
                    }
                    let pat_val = self.eval_expr(pattern)?;
                    let pat_str = format!("{pat_val}");
                    if scrutinee_str == pat_str {
                        return self.eval_expr(result);
                    }
                }
                Err(self.rt_err("match: no arm matched"))
            }

            Expr::IfExpr {
                cond,
                then_expr,
                else_expr,
            } => {
                let cond_val = self.eval_expr(cond)?;
                if Self::value_as_bool(&cond_val) {
                    self.eval_expr(then_expr)
                } else {
                    self.eval_expr(else_expr)
                }
            }

            // ── Aritmética / lógica escalar ───────────────────────────────────
            Expr::BinOp { op, lhs, rhs } => {
                // Short-circuit para And/Or
                match op {
                    BinOp::And => {
                        let l = self.eval_expr(lhs)?;
                        if !Self::value_as_bool(&l) {
                            return Ok(Value::Bool(false));
                        }
                        let r = self.eval_expr(rhs)?;
                        return Ok(Value::Bool(Self::value_as_bool(&r)));
                    }
                    BinOp::Or => {
                        let l = self.eval_expr(lhs)?;
                        if Self::value_as_bool(&l) {
                            return Ok(Value::Bool(true));
                        }
                        let r = self.eval_expr(rhs)?;
                        return Ok(Value::Bool(Self::value_as_bool(&r)));
                    }
                    BinOp::In => {
                        let l = self.eval_expr(lhs)?;
                        let r = self.eval_expr(rhs)?;
                        let found = match &r {
                            Value::List(lst) => {
                                let needle = format!("{l}");
                                lst.iter().any(|item| format!("{item}") == needle)
                            }
                            Value::Dict(m) => match &l {
                                Value::Str(s) => m.contains_key(s),
                                _ => m.contains_key(&format!("{l}")),
                            },
                            Value::Str(s) => match &l {
                                Value::Str(sub) => s.contains(sub.as_str()),
                                _ => s.contains(&format!("{l}")),
                            },
                            _ => {
                                return Err(self
                                    .type_err("'in' requires list, dict, or string on right side"))
                            }
                        };
                        return Ok(Value::Bool(found));
                    }
                    _ => {}
                }
                let l = self.eval_expr(lhs)?;
                let r = self.eval_expr(rhs)?;
                Self::eval_scalar_binop(op, l, r)
            }

            Expr::Neg(inner) => match self.eval_expr(inner)? {
                Value::Int(v) => Ok(Value::Int(-v)),
                Value::Float(v) => Ok(Value::Float(-v)),
                _ => Err(HayashiError::Type("negação unária requires number".into())),
            },

            Expr::Not(inner) => {
                let v = self.eval_expr(inner)?;
                Ok(Value::Bool(!Self::value_as_bool(&v)))
            }

            // ── Lista literal ─────────────────────────────────────────────────
            Expr::List(items) => {
                let vals: Vec<Value> = items
                    .iter()
                    .map(|e| self.eval_expr(e))
                    .collect::<Result<_>>()?;
                Ok(Value::List(Rc::new(vals)))
            }

            // ── Dict literal ─────────────────────────────────────────────────
            Expr::Dict(pairs) => {
                let mut map = std::collections::HashMap::new();
                for (k_expr, v_expr) in pairs {
                    let key = match self.eval_expr(k_expr)? {
                        Value::Str(s) => s,
                        Value::Int(i) => format!("{i}"),
                        Value::Float(f) => format!("{f}"),
                        other => {
                            return Err(HayashiError::Type(format!(
                                "dict key must be string, got {other}"
                            )))
                        }
                    };
                    let val = self.eval_expr(v_expr)?;
                    map.insert(key, val);
                }
                Ok(Value::Dict(Rc::new(map)))
            }

            // ── Indexação: lista[idx] ou dict["key"] ─────────────────────────
            Expr::Index { obj, idx } => {
                let obj_val = self.eval_expr(obj)?;
                let idx_val = self.eval_expr(idx)?;
                match (&obj_val, &idx_val) {
                    (Value::Dict(m), Value::Str(key)) => m.get(key).cloned().ok_or_else(|| {
                        HayashiError::Runtime(format!("key '{key}' not found in dict"))
                    }),
                    (Value::Dict(_), _) => {
                        Err(HayashiError::Type("dict index must be a string".into()))
                    }
                    (Value::DataFrame(df), Value::Str(key)) => {
                        let col = df.get_column(key).map_err(|_| {
                            HayashiError::Runtime(format!("column '{key}' not found in DataFrame"))
                        })?;
                        use greeners::Column;
                        let vals: Vec<Value> = match col {
                            Column::Float(arr) => arr.iter().map(|&x| Value::Float(x)).collect(),
                            Column::Int(arr) => arr.iter().map(|&x| Value::Int(x)).collect(),
                            Column::Bool(arr) => arr.iter().map(|&x| Value::Bool(x)).collect(),
                            Column::String(arr) => {
                                arr.iter().map(|s| Value::Str(s.clone())).collect()
                            }
                            Column::Categorical(c) => c
                                .codes
                                .iter()
                                .map(|&code| {
                                    let level = c
                                        .levels
                                        .get(code as usize)
                                        .map(|s| s.clone())
                                        .unwrap_or_else(|| "".to_string());
                                    Value::Str(level)
                                })
                                .collect(),
                            Column::DateTime(arr) => {
                                arr.iter().map(|dt| Value::Str(dt.to_string())).collect()
                            }
                        };
                        Ok(Value::List(Rc::new(vals)))
                    }
                    (Value::DataFrame(_), _) => Err(HayashiError::Type(
                        "DataFrame column index must be a string".into(),
                    )),
                    (Value::List(v), _) => {
                        let i = match idx_val {
                            Value::Int(i) => i,
                            Value::Float(f) => f as i64,
                            _ => {
                                return Err(HayashiError::Type("list index must be integer".into()))
                            }
                        };
                        let len = v.len() as i64;
                        let real = if i < 0 { len + i } else { i };
                        if real < 0 || real >= len {
                            return Err(HayashiError::Runtime(format!(
                                "index out of range (len={len})"
                            )));
                        }
                        Ok(v[real as usize].clone())
                    }
                    _ => Err(HayashiError::Type("indexação requires list ou dict".into())),
                }
            }

            Expr::Call { func, args, opts } => self.eval_call(func, args, opts),

            Expr::Field {
                obj,
                field,
                args,
                opts,
            } => self.eval_field(obj, field, args, opts),

            Expr::TsOp { .. } => Err(HayashiError::Runtime(
                "operadores L./F./D. só são válidos dentro de generate".into(),
            )),

            Expr::Range(start_expr, end_expr) => {
                let start = self.eval_as_int(start_expr, "range start")?;
                let end = self.eval_as_int(end_expr, "range end")?;
                let step: i64 = if start <= end { 1 } else { -1 };
                let mut v = Vec::new();
                let mut cur = start;
                while if step > 0 { cur < end } else { cur > end } {
                    v.push(Value::Int(cur));
                    cur += step;
                }
                Ok(Value::List(Rc::new(v)))
            }

            Expr::RangeInclusive(start_expr, end_expr) => {
                let start = self.eval_as_int(start_expr, "range start")?;
                let end = self.eval_as_int(end_expr, "range end")?;
                let step: i64 = if start <= end { 1 } else { -1 };
                let mut v = Vec::new();
                let mut cur = start;
                while if step > 0 { cur <= end } else { cur >= end } {
                    v.push(Value::Int(cur));
                    cur += step;
                }
                Ok(Value::List(Rc::new(v)))
            }
        }
    }

    // ── Converte fórmula AST → string Greeners ────────────────────────────────

    fn formula_to_string(f: &Formula) -> String {
        let rhs_parts: Vec<String> = f
            .rhs
            .iter()
            .map(|t| match t {
                RhsTerm::Var(v) => v.clone(),
                RhsTerm::Categorical(v) => format!("C({v})"),
                RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
            })
            .collect();

        let mut formula_str = if f.lhs.is_empty() {
            format!("~ {}", rhs_parts.join(" + "))
        } else {
            format!("{} ~ {}", f.lhs, rhs_parts.join(" + "))
        };
        if !f.fe.is_empty() {
            formula_str.push_str(" | ");
            formula_str.push_str(&f.fe.join(" + "));
        }
        formula_str
    }

    /// Extrai coluna como Array1<f64>; aceita Float, Int, Bool, Categorical, etc. convertendo dinamicamente.
    fn get_col_f64(df: &DataFrame, name: &str) -> Result<ndarray::Array1<f64>> {
        let col = df
            .get_column(name)
            .map_err(|_| HayashiError::Runtime(format!("column '{name}' not found")))?;
        Ok(col.to_float())
    }

    /// Reconstrói X a partir da lista de nomes de variáveis do modelo.
    /// `_cons`/`const`/`Intercept` → coluna de 1s; demais → colunas do df.
    fn build_x_from_varnames(df: &DataFrame, names: &[String]) -> Result<ndarray::Array2<f64>> {
        let n = df.n_rows();
        let k = names.len();
        let mut x = ndarray::Array2::<f64>::zeros((n, k));
        for (j, name) in names.iter().enumerate() {
            match name.as_str() {
                "_cons" | "const" | "Intercept" | "(Intercept)" => {
                    x.column_mut(j).fill(1.0);
                }
                other => {
                    let col = Self::get_col_f64(df, other).map_err(|_| {
                        HayashiError::Runtime(format!(
                            "predict: column '{other}' not found no DataFrame"
                        ))
                    })?;
                    x.column_mut(j).assign(&col);
                }
            }
        }
        Ok(x)
    }

    fn resolve_cov(opt_val: Option<&Value>) -> Result<CovarianceType> {
        match opt_val {
            None => Ok(CovarianceType::NonRobust),
            Some(Value::Str(s)) => match s.as_str() {
                "nonrobust" | "ols" => Ok(CovarianceType::NonRobust),
                "robust" => Ok(CovarianceType::HC1),
                "HC1" => Ok(CovarianceType::HC1),
                "HC2" => Ok(CovarianceType::HC2),
                "HC3" => Ok(CovarianceType::HC3),
                "HC4" => Ok(CovarianceType::HC4),
                other => Err(HayashiError::Type(format!(
                    "unknown covariance type '{other}'"
                ))),
            },
            _ => Err(HayashiError::Type("cov= must be a string".into())),
        }
    }

    fn col_to_cluster_ids(df: &DataFrame, col: &str) -> Result<Vec<usize>> {
        let mut map: HashMap<i64, usize> = HashMap::new();
        let mut next = 0usize;
        if let Ok(arr) = df.get_int(col) {
            Ok(arr
                .iter()
                .map(|&v| {
                    *map.entry(v).or_insert_with(|| {
                        let id = next;
                        next += 1;
                        id
                    })
                })
                .collect())
        } else if let Ok(arr) = df.get(col) {
            Ok(arr
                .iter()
                .map(|&v| {
                    let key = v as i64;
                    *map.entry(key).or_insert_with(|| {
                        let id = next;
                        next += 1;
                        id
                    })
                })
                .collect())
        } else if let Ok(arr) = df.get_string(col) {
            let mut smap: HashMap<String, usize> = HashMap::new();
            Ok(arr
                .iter()
                .map(|v| {
                    *smap.entry(v.clone()).or_insert_with(|| {
                        let id = next;
                        next += 1;
                        id
                    })
                })
                .collect())
        } else {
            Err(HayashiError::Runtime(format!(
                "cluster column '{col}' not found"
            )))
        }
    }

    fn resolve_cov_full(
        opt_map: &HashMap<String, Value>,
        df: &DataFrame,
    ) -> Result<CovarianceType> {
        if let Some(Value::Str(cluster_col)) = opt_map.get("cluster") {
            let ids = Self::col_to_cluster_ids(df, cluster_col)?;
            if let Some(Value::Str(cluster2_col)) = opt_map.get("cluster2") {
                let ids2 = Self::col_to_cluster_ids(df, cluster2_col)?;
                Ok(CovarianceType::ClusteredTwoWay(ids, ids2))
            } else {
                Ok(CovarianceType::Clustered(ids))
            }
        } else if let Some(Value::Str(nw)) = opt_map.get("nw") {
            let lags: usize = nw
                .parse()
                .unwrap_or_else(|_| (df.n_rows() as f64).powf(0.25) as usize);
            Ok(CovarianceType::NeweyWest(lags))
        } else if let Some(Value::Int(nw)) = opt_map.get("nw") {
            Ok(CovarianceType::NeweyWest(*nw as usize))
        } else {
            Self::resolve_cov(opt_map.get("cov"))
        }
    }

    fn filter_df_by_mask(df: &DataFrame, mask: &[f64]) -> Result<Rc<DataFrame>> {
        let keep: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m != 0.0)
            .map(|(i, _)| i)
            .collect();
        df.iloc(Some(&keep), None)
            .map(Rc::new)
            .map_err(|e| HayashiError::Runtime(e.to_string()))
    }

    fn maybe_filter_df(&mut self, df: &Rc<DataFrame>, opts: &[Opt]) -> Result<Rc<DataFrame>> {
        if let Some(if_opt) = opts.iter().find(|o| o.name == "if") {
            let mask = self.eval_col_expr(&if_opt.value, df)?;
            Self::filter_df_by_mask(df, &mask)
        } else {
            Ok(df.clone())
        }
    }

    // ── Funções built-in ──────────────────────────────────────────────────────

    fn eval_call(&mut self, func: &str, args: &[Expr], opts: &[Opt]) -> Result<Value> {
        if let Some(pos) = func.find("::") {
            let ns = &func[..pos];
            let member = &func[pos + 2..];
            if self.plugins.contains_key(ns) {
                let mut evaluated_args = Vec::new();
                for arg in args {
                    evaluated_args.push(self.eval_expr(arg)?);
                }
                let mut plugin = self.plugins.remove(ns).unwrap();
                let res = plugin
                    .call(member, &evaluated_args)
                    .map_err(|e| HayashiError::Runtime(format!("plugin '{ns}' error: {e}")));
                self.plugins.insert(ns.to_string(), plugin);
                return res;
            }
        }

        let is_mutate = func == "mutate" || func == "generate";
        let opt_map: HashMap<String, Value> = opts
            .iter()
            .filter(|o| o.name != "if" && o.name != "vars" && o.name != "dydx" && !is_mutate)
            .map(|o| {
                let val = self.eval_expr(&o.value).or_else(|e| {
                    if let Expr::Var(name) = &o.value {
                        Ok(Value::Str(name.clone()))
                    } else {
                        Err(e)
                    }
                })?;
                Ok((o.name.clone(), val))
            })
            .collect::<Result<_>>()?;

        macro_rules! try_group {
            ($m:ident) => {
                if let Some(v) = self.$m(func, args, opts, &opt_map)? {
                    return Ok(v);
                }
            };
        }
        try_group!(eval_call_visualization);
        try_group!(eval_call_estimators_misc);
        try_group!(eval_call_estimators_timeseries);
        try_group!(eval_call_data_manipulation);
        try_group!(eval_call_post_estimation_ts);

        match func {
            // ── Type conversions ─────────────────────────────────────────────
            "int" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("int(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Int(match v {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    Value::Bool(b) => {
                        if b {
                            1
                        } else {
                            0
                        }
                    }
                    Value::Str(s) => s
                        .trim()
                        .parse::<i64>()
                        .or_else(|_| s.trim().parse::<f64>().map(|f| f as i64))
                        .map_err(|_| self.type_err(format!("cannot convert '{s}' to int")))?,
                    other => return Err(self.type_err(format!("cannot convert {other} to int"))),
                }))
            }

            "float" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("float(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Float(match v {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    Value::Bool(b) => {
                        if b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Value::Str(s) => s
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| self.type_err(format!("cannot convert '{s}' to float")))?,
                    other => return Err(self.type_err(format!("cannot convert {other} to float"))),
                }))
            }

            "str" | "string" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("str(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Str(format!("{v}")))
            }

            "bool" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("bool(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(Self::value_as_bool(&v)))
            }
            "is_nil" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_nil(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Nil)))
            }
            "is_int" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_int(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Int(_))))
            }
            "is_float" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_float(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Float(_))))
            }
            "is_bool" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_bool(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Bool(_))))
            }
            "is_str" | "is_string" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_str(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Str(_))))
            }
            "is_list" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_list(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::List(_))))
            }
            "is_dict" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_dict(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Dict(_))))
            }
            "is_df" | "is_dataframe" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_dataframe(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::DataFrame(_))))
            }
            "is_fn" | "is_function" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_function(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::UserFn(_))))
            }

            "type" | "typeof" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("type(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Str(
                    match v {
                        Value::Float(_) => "float",
                        Value::Int(_) => "int",
                        Value::Bool(_) => "bool",
                        Value::Str(_) => "string",
                        Value::List(_) => "list",
                        Value::Dict(_) => "dict",
                        Value::DataFrame(_) => "dataframe",
                        Value::UserFn(_) => "function",
                        Value::Nil => "nil",
                        _ => "model",
                    }
                    .to_string(),
                ))
            }

            // ── Date/time ─────────────────────────────────────────────────────
            "date" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("date(\"YYYY-MM-DD\")".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("date() requires a string".into())),
                };
                let nd = chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d")
                    .map_err(|e| HayashiError::Runtime(format!("date parse error: {e}")))?;
                let dt = nd.and_hms_opt(0, 0, 0).unwrap();
                Ok(Value::Float(dt.and_utc().timestamp() as f64))
            }

            "datetime" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "datetime(\"YYYY-MM-DD HH:MM:SS\")".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("datetime() requires a string".into())),
                };
                let dt = chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                    .or_else(|_| chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%dT%H:%M:%S"))
                    .map_err(|e| HayashiError::Runtime(format!("datetime parse error: {e}")))?;
                Ok(Value::Float(dt.and_utc().timestamp() as f64))
            }

            // ── Builtins de lista ─────────────────────────────────────────────
            "len" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "len() requires exactly 1 argument".into(),
                    ));
                }
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::List(lst) => Ok(Value::Int(lst.len() as i64)),
                    Value::Dict(m) => Ok(Value::Int(m.len() as i64)),
                    Value::Str(s) => Ok(Value::Int(s.chars().count() as i64)),
                    _ => Err(HayashiError::Type(
                        "len() requires list, dict, or string".into(),
                    )),
                }
            }

            "keys" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("keys(dict)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::Dict(m) => {
                        let mut ks: Vec<String> = m.keys().cloned().collect();
                        ks.sort();
                        Ok(Value::List(Rc::new(
                            ks.into_iter().map(Value::Str).collect(),
                        )))
                    }
                    _ => Err(HayashiError::Type("keys() requires dict".into())),
                }
            }

            "values" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("values(dict)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::Dict(m) => {
                        let mut pairs: Vec<_> = m.iter().collect();
                        pairs.sort_by_key(|(k, _)| (*k).clone());
                        Ok(Value::List(Rc::new(
                            pairs.into_iter().map(|(_, v)| v.clone()).collect(),
                        )))
                    }
                    _ => Err(HayashiError::Type("values() requires dict".into())),
                }
            }

            "has_key" => {
                if args.len() != 2 {
                    return Err(self.rt_err("has_key(dict, \"key\")"));
                }
                let d = self.eval_expr(&args[0])?;
                let k = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("has_key: key must be string".into())),
                };
                match d {
                    Value::Dict(m) => Ok(Value::Bool(m.contains_key(&k))),
                    _ => Err(HayashiError::Type("has_key() requires dict".into())),
                }
            }

            "dict_merge" | "dmerge" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("dict_merge(dict1, dict2)".into()));
                }
                let d1 = self.eval_expr(&args[0])?;
                let d2 = self.eval_expr(&args[1])?;
                match (d1, d2) {
                    (Value::Dict(m1), Value::Dict(m2)) => {
                        let mut merged = (*m1).clone();
                        for (k, v) in m2.iter() {
                            merged.insert(k.clone(), v.clone());
                        }
                        Ok(Value::Dict(Rc::new(merged)))
                    }
                    _ => Err(HayashiError::Type("dict_merge() requires two dicts".into())),
                }
            }

            "dict_set" | "dset" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "dict_set(dict, \"key\", value)".into(),
                    ));
                }
                let d = self.eval_expr(&args[0])?;
                let k = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("dict_set: key must be string".into())),
                };
                let v = self.eval_expr(&args[2])?;
                match d {
                    Value::Dict(m) => {
                        let mut new_m = (*m).clone();
                        new_m.insert(k, v);
                        Ok(Value::Dict(Rc::new(new_m)))
                    }
                    _ => Err(HayashiError::Type("dict_set() requires dict".into())),
                }
            }

            "dict_remove" | "dremove" => {
                if args.len() != 2 {
                    return Err(self.rt_err("dict_remove(dict, \"key\")"));
                }
                let d = self.eval_expr(&args[0])?;
                let k = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("dict_remove: key must be string".into())),
                };
                match d {
                    Value::Dict(m) => {
                        let mut new_m = (*m).clone();
                        new_m.remove(&k);
                        Ok(Value::Dict(Rc::new(new_m)))
                    }
                    _ => Err(HayashiError::Type("dict_remove() requires dict".into())),
                }
            }

            "dataframe" => {
                if args.len() != 1 {
                    return Err(self.rt_err("dataframe(dict)"));
                }
                let d = self.eval_expr(&args[0])?;
                match d {
                    Value::Dict(m) => {
                        let df = self.dict_to_dataframe(&m)?;
                        Ok(Value::DataFrame(Rc::new(df)))
                    }
                    _ => Err(HayashiError::Type("dataframe() requires dict".into())),
                }
            }

            // ── Funções de string ─────────────────────────────────────────────
            "upper" | "lower" | "trim" => {
                let s =
                    match self
                        .eval_expr(args.first().ok_or_else(|| {
                            self.rt_err(format!("{func}() requires 1 argument"))
                        })?)? {
                        Value::Str(s) => s,
                        v => {
                            return Err(HayashiError::Type(format!(
                                "{func}() requires string, recebeu {v}"
                            )))
                        }
                    };
                Ok(Value::Str(match func {
                    "upper" => s.to_uppercase(),
                    "lower" => s.to_lowercase(),
                    "trim" => s.trim().to_string(),
                    _ => unreachable!(),
                }))
            }

            "write" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "write(content, path) requires 2 arguments".into(),
                    ));
                }
                let content = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(self.type_err(format!("write: content must be string, got {v}")))
                    }
                };
                let path = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("write: path must be string, got {v}"))),
                };
                std::fs::write(&path, &content)
                    .map_err(|e| HayashiError::Io(format!("Failed to write file '{path}': {e}")))?;
                Ok(Value::Nil)
            }

            "contains" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "contains(s, padrão) requires 2 arguments".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("contains: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("contains: pattern must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Bool(s.contains(pat.as_str())))
            }

            "starts_with" | "ends_with" => {
                if args.len() != 2 {
                    return Err(self.rt_err(format!("{func}(s, padrão) requires 2 arguments")));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("{func}: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("{func}: pattern must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Bool(match func {
                    "starts_with" => s.starts_with(pat.as_str()),
                    "ends_with" => s.ends_with(pat.as_str()),
                    _ => unreachable!(),
                }))
            }

            // substr(s, início [, comprimento]) — índice 0-based em chars
            "substr" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime(
                        "substr(s, start [, length]) requires 2 or 3 arguments".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("substr: expected string, got {v}"))),
                };
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len() as i64;
                let start = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    v => {
                        return Err(self.type_err(format!("substr: start must be integer, got {v}")))
                    }
                };
                let real_start =
                    (if start < 0 { len + start } else { start }).clamp(0, len) as usize;
                let count = if args.len() == 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Int(i) => i.max(0) as usize,
                        Value::Float(f) => f.max(0.0) as usize,
                        v => {
                            return Err(
                                self.type_err(format!("substr: length must be integer, got {v}"))
                            )
                        }
                    }
                } else {
                    chars.len().saturating_sub(real_start)
                };
                let end = (real_start + count).min(chars.len());
                Ok(Value::Str(chars[real_start..end].iter().collect()))
            }

            // split(s, delimitador) → List de Str
            "split" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "split(s, delimitador) requires 2 arguments".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("split: expected string, got {v}"))),
                };
                let delim = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("split: delimiter must be string, got {v}"))
                        )
                    }
                };
                let parts: Vec<Value> = s
                    .split(delim.as_str())
                    .map(|p| Value::Str(p.to_string()))
                    .collect();
                Ok(Value::List(Rc::new(parts)))
            }

            // str_replace(s, de, para) — "replace" é palavra-chave
            "str_replace" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "str_replace(s, de, para) requires 3 arguments".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(self.type_err(format!("str_replace: expected string, got {v}")))
                    }
                };
                let from = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("str_replace: 'from' must be string, got {v}"))
                        )
                    }
                };
                let to = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("str_replace: 'to' must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Str(s.replace(from.as_str(), to.as_str())))
            }

            // ── Regex ─────────────────────────────────────────────────────────
            // regexm(s, pattern)            → 1 se match, 0 se não
            // regexr(s, pattern, replace)   → substitui primeira ocorrência
            // regexra(s, pattern, replace)  → substitui todas
            // regexs(s, pattern)            → extrai primeiro grupo de captura
            "regexm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("regexm(string, pattern)".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("regexm: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexm: pattern must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Bool(greeners::Transforms::regexm(&s, &pat)))
            }

            "regexr" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "regexr(string, pattern, replacement)".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("regexr: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexr: pattern must be string, got {v}"))
                        )
                    }
                };
                let rep = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexr: replacement must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Str(greeners::Transforms::regexr(&s, &pat, &rep)))
            }

            "regexra" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "regexra(string, pattern, replacement)".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("regexra: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexra: pattern must be string, got {v}"))
                        )
                    }
                };
                let rep = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexra: replacement must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Str(greeners::Transforms::regexra(&s, &pat, &rep)))
            }

            "regexs" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("regexs(string, pattern)".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("regexs: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexs: pattern must be string, got {v}"))
                        )
                    }
                };
                match greeners::Transforms::regexs(&s, &pat) {
                    Some(m) => Ok(Value::Str(m)),
                    None => Ok(Value::Str(String::new())),
                }
            }

            // ── Agregações sobre List ─────────────────────────────────────────
            // "sum" fica para summarize(df) — Stata-style
            // "total" é a soma de uma lista numérica
            "sum" | "mean" | "sd" | "std" | "min" | "max" | "total" => {
                // Forma 1: mean(list)  /  sd(list)  /  std(list)  etc.
                // Forma 2: mean(df, var)  ou  mean(df, var, if=cond)
                let nums: Vec<f64> = if args.len() >= 2 {
                    // forma DataFrame
                    let df_name = match &args[0] {
                        Expr::Var(n) => n.clone(),
                        _ => {
                            return Err(self
                                .type_err(format!("{func}: first argument must be a DataFrame")))
                        }
                    };
                    let df = match self.env.get(&df_name) {
                        Some(Value::DataFrame(d)) => d.clone(),
                        _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                    };
                    let var_name = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(self.type_err(format!(
                                "{func}: second argument must be a variable name"
                            )))
                        }
                    };
                    let col = Self::get_col_f64(&df, &var_name)?;
                    // filtro opcional: if=cond
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        col.iter()
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|(&v, _)| v)
                            .collect()
                    } else {
                        col.to_vec()
                    }
                } else if args.len() == 1 {
                    let v = self.eval_expr(&args[0])?;
                    match v {
                        Value::List(lst) => {
                            lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?
                        }
                        other => {
                            return Err(self
                                .type_err(format!("{func}() requires numeric list, got {other}")))
                        }
                    }
                } else {
                    return Err(self.rt_err(format!("{func}() requires at least 1 argument")));
                };
                if nums.is_empty() {
                    return Err(self.rt_err(format!(
                        "{func}(): nenhum valor (empty list ou filtro excluiu tudo)"
                    )));
                }
                let result = match func {
                    "sum" | "total" => nums.iter().sum::<f64>(),
                    "mean" => nums.iter().sum::<f64>() / nums.len() as f64,
                    "min" => nums.iter().cloned().fold(f64::INFINITY, f64::min),
                    "max" => nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    "sd" | "std" => {
                        let n = nums.len() as f64;
                        let m = nums.iter().sum::<f64>() / n;
                        (nums.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
                    }
                    _ => unreachable!(),
                };
                Ok(Value::Float(result))
            }

            // ── Novas agregações escalares (todas suportam if = cond) ────────
            "median" => {
                // median(lista) | median(df, x) | median(df, x, if = cond)
                let nums: Vec<f64> = if args.len() >= 2 {
                    let df_name = match &args[0] {
                        Expr::Var(n) => n.clone(),
                        _ => return Err(self.rt_err("median: first argument must be a DataFrame")),
                    };
                    let df = match self.env.get(&df_name) {
                        Some(Value::DataFrame(d)) => d.clone(),
                        _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                    };
                    let var_name = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(
                                self.rt_err("median: second argument must be a variable name")
                            )
                        }
                    };
                    let col = Self::get_col_f64(&df, &var_name)?;
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        col.iter()
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|(&v, _)| v)
                            .collect()
                    } else {
                        col.to_vec()
                    }
                } else if args.len() == 1 {
                    match self.eval_expr(&args[0])? {
                        Value::List(lst) => {
                            lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?
                        }
                        other => {
                            return Err(self
                                .type_err(format!("median() requires numeric list, got {other}")))
                        }
                    }
                } else {
                    return Err(self.rt_err("median() requires at least 1 argument"));
                };
                if nums.is_empty() {
                    return Err(self.rt_err("median(): empty list"));
                }
                let mut sorted = nums.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                let result = if n % 2 == 0 {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                } else {
                    sorted[n / 2]
                };
                Ok(Value::Float(result))
            }

            "variance" => {
                // variance(lista) | variance(df, x) | variance(df, x, if = cond) — amostral (/ n-1)
                let nums: Vec<f64> = if args.len() >= 2 {
                    let df_name = match &args[0] {
                        Expr::Var(n) => n.clone(),
                        _ => {
                            return Err(self.rt_err("variance: first argument must be a DataFrame"))
                        }
                    };
                    let df = match self.env.get(&df_name) {
                        Some(Value::DataFrame(d)) => d.clone(),
                        _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                    };
                    let var_name = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(
                                self.rt_err("variance: second argument must be a variable name")
                            )
                        }
                    };
                    let col = Self::get_col_f64(&df, &var_name)?;
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        col.iter()
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|(&v, _)| v)
                            .collect()
                    } else {
                        col.to_vec()
                    }
                } else if args.len() == 1 {
                    match self.eval_expr(&args[0])? {
                        Value::List(lst) => {
                            lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?
                        }
                        other => {
                            return Err(self.type_err(format!(
                                "variance() requires numeric list, got {other}"
                            )))
                        }
                    }
                } else {
                    return Err(self.rt_err("variance() requires at least 1 argument"));
                };
                let n = nums.len();
                if n < 2 {
                    return Err(self.rt_err("variance(): requires at least 2 observations"));
                }
                let mean = nums.iter().sum::<f64>() / n as f64;
                let v = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                Ok(Value::Float(v))
            }

            "quantile" => {
                // quantile(df, x, p) | quantile(lista, p) | quantile(df, x, p, if = cond) — p ∈ [0,1]
                let (nums, p) = if args.len() >= 3 {
                    let df_name = match &args[0] {
                        Expr::Var(n) => n.clone(),
                        _ => {
                            return Err(self.rt_err("quantile: first argument must be a DataFrame"))
                        }
                    };
                    let df = match self.env.get(&df_name) {
                        Some(Value::DataFrame(d)) => d.clone(),
                        _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                    };
                    let var_name = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(
                                self.rt_err("quantile: second argument must be a variable name")
                            )
                        }
                    };
                    let col = Self::get_col_f64(&df, &var_name)?;
                    let nums = if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        col.iter()
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|(&v, _)| v)
                            .collect()
                    } else {
                        col.to_vec()
                    };
                    let p = match self.eval_expr(&args[2])? {
                        Value::Float(f) => f,
                        Value::Int(i) => i as f64,
                        other => return Err(self.type_mismatch("Float", &other)),
                    };
                    (nums, p)
                } else if args.len() == 2 {
                    let v = self.eval_expr(&args[0])?;
                    let nums = match v {
                        Value::List(lst) => {
                            lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?
                        }
                        other => {
                            return Err(self.type_err(format!(
                                "quantile() requires numeric list, got {other}"
                            )))
                        }
                    };
                    let p = match self.eval_expr(&args[1])? {
                        Value::Float(f) => f,
                        Value::Int(i) => i as f64,
                        other => return Err(self.type_mismatch("Float", &other)),
                    };
                    (nums, p)
                } else {
                    return Err(self.rt_err("quantile(df, x, p) ou quantile(lista, p)"));
                };
                if !(0.0..=1.0).contains(&p) {
                    return Err(self.rt_err("quantile(): p deve estar em [0, 1]"));
                }
                let mut sorted: Vec<f64> = nums.into_iter().filter(|x| x.is_finite()).collect();
                if sorted.is_empty() {
                    return Err(self.rt_err("quantile(): nenhum valor finito"));
                }
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = p * (sorted.len() - 1) as f64;
                let lo = idx.floor() as usize;
                let hi = idx.ceil() as usize;
                let result = if lo == hi {
                    sorted[lo]
                } else {
                    let w = idx - lo as f64;
                    sorted[lo] * (1.0 - w) + sorted[hi] * w
                };
                Ok(Value::Float(result))
            }

            "cov" => {
                // cov(df, x, y) | cov(df, x, y, if = cond) — covariância amostral (/ n-1)
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("cov(df, x, y)".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };
                let x_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(self.rt_err("cov(): second argument must be a variable name")),
                };
                let y_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(self.rt_err("cov(): third argument must be a variable name")),
                };
                let x_col = Self::get_col_f64(&df, &x_name)?;
                let y_col = Self::get_col_f64(&df, &y_name)?;
                let (x_vals, y_vals): (Vec<f64>, Vec<f64>) =
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        x_col
                            .iter()
                            .zip(y_col.iter())
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|((&xi, &yi), _)| (xi, yi))
                            .unzip()
                    } else {
                        (x_col.to_vec(), y_col.to_vec())
                    };
                let n = x_vals.len();
                if n < 2 {
                    return Err(self.rt_err("cov(): requires at least 2 observations"));
                }
                let mx = x_vals.iter().sum::<f64>() / n as f64;
                let my = y_vals.iter().sum::<f64>() / n as f64;
                let c = x_vals
                    .iter()
                    .zip(y_vals.iter())
                    .map(|(&xi, &yi)| (xi - mx) * (yi - my))
                    .sum::<f64>()
                    / (n - 1) as f64;
                Ok(Value::Float(c))
            }

            "corr_pair" => {
                // corr_pair(df, x, y) | corr_pair(df, x, y, if = cond) — Pearson escalar
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("corr_pair(df, x, y)".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };
                let x_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(
                            self.rt_err("corr_pair(): second argument must be a variable name")
                        )
                    }
                };
                let y_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(
                            self.rt_err("corr_pair(): third argument must be a variable name")
                        )
                    }
                };
                let x_col = Self::get_col_f64(&df, &x_name)?;
                let y_col = Self::get_col_f64(&df, &y_name)?;
                let (x_vals, y_vals): (Vec<f64>, Vec<f64>) =
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        x_col
                            .iter()
                            .zip(y_col.iter())
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|((&xi, &yi), _)| (xi, yi))
                            .unzip()
                    } else {
                        (x_col.to_vec(), y_col.to_vec())
                    };
                let n = x_vals.len();
                if n < 2 {
                    return Err(self.rt_err("corr_pair(): requires at least 2 observations"));
                }
                let mx = x_vals.iter().sum::<f64>() / n as f64;
                let my = y_vals.iter().sum::<f64>() / n as f64;
                let mut num = 0.0f64;
                let mut dx2 = 0.0f64;
                let mut dy2 = 0.0f64;
                for (&xi, &yi) in x_vals.iter().zip(y_vals.iter()) {
                    let dx = xi - mx;
                    let dy = yi - my;
                    num += dx * dy;
                    dx2 += dx * dx;
                    dy2 += dy * dy;
                }
                let r = if dx2 > 0.0 && dy2 > 0.0 {
                    num / (dx2.sqrt() * dy2.sqrt())
                } else {
                    0.0
                };
                Ok(Value::Float(r))
            }

            "push" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("push(lista, item)".into()));
                }
                let var_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "push() first argument must be a variable".into(),
                        ))
                    }
                };
                let item = self.eval_expr(&args[1])?;
                let lst = self
                    .env
                    .get(&var_name)
                    .cloned()
                    .ok_or_else(|| self.rt_err(format!("undefined variable '{var_name}'")))?;
                match lst {
                    Value::List(v) => {
                        let mut new_v = (*v).clone();
                        new_v.push(item);
                        self.env.set(&var_name, Value::List(Rc::new(new_v)))?;
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Type("push() requires list".into())),
                }
            }

            "pop" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("pop(lista)".into()));
                }
                let var_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "pop() argument must be a variable".into(),
                        ))
                    }
                };
                let lst = self
                    .env
                    .get(&var_name)
                    .cloned()
                    .ok_or_else(|| self.rt_err(format!("undefined variable '{var_name}'")))?;
                match lst {
                    Value::List(v) => {
                        if v.is_empty() {
                            return Err(HayashiError::Runtime("pop() on empty list".into()));
                        }
                        let mut new_v = (*v).clone();
                        let removed = new_v.pop().unwrap();
                        self.env.set(&var_name, Value::List(Rc::new(new_v)))?;
                        Ok(removed)
                    }
                    _ => Err(HayashiError::Type("pop() requires list".into())),
                }
            }

            "insert" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime("insert(lista, indice, item)".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let idx = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(HayashiError::Type("insert: index must be integer".into())),
                };
                let item = self.eval_expr(&args[2])?;
                match lst {
                    Value::List(v) => {
                        let mut new_v = (*v).clone();
                        if idx > new_v.len() {
                            return Err(HayashiError::Runtime(format!(
                                "insert: index out of range (len={})",
                                new_v.len()
                            )));
                        }
                        new_v.insert(idx, item);
                        Ok(Value::List(Rc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("insert() requires list".into())),
                }
            }

            "remove" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("remove(lista, indice)".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let idx = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(HayashiError::Type("remove: index must be integer".into())),
                };
                match lst {
                    Value::List(v) => {
                        if idx >= v.len() {
                            return Err(HayashiError::Runtime(format!(
                                "remove: index out of range (len={})",
                                v.len()
                            )));
                        }
                        let mut new_v = (*v).clone();
                        new_v.remove(idx);
                        Ok(Value::List(Rc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("remove() requires list".into())),
                }
            }

            "clear" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("clear(lista)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(_) => Ok(Value::List(Rc::new(Vec::new()))),
                    _ => Err(HayashiError::Type("clear() requires list".into())),
                }
            }

            "reverse" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("reverse(lista)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(v) => {
                        let mut new_v = (*v).clone();
                        new_v.reverse();
                        Ok(Value::List(Rc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("reverse() requires list".into())),
                }
            }

            "index" | "indexof" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "index(lista, item) → posição ou -1".into(),
                    ));
                }
                let lst = self.eval_expr(&args[0])?;
                let needle = self.eval_expr(&args[1])?;
                match lst {
                    Value::List(v) => {
                        let pos = v.iter().position(|x| format!("{x}") == format!("{needle}"));
                        Ok(Value::Int(pos.map(|p| p as i64).unwrap_or(-1)))
                    }
                    _ => Err(HayashiError::Type("index() requires list".into())),
                }
            }

            "slice" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime("slice(lista, inicio [, fim])".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let start = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(HayashiError::Type("slice: start must be integer".into())),
                };
                match lst {
                    Value::List(v) => {
                        let end = if args.len() == 3 {
                            match self.eval_expr(&args[2])? {
                                Value::Int(i) => (i as usize).min(v.len()),
                                Value::Float(f) => (f as usize).min(v.len()),
                                _ => {
                                    return Err(HayashiError::Type(
                                        "slice: end must be integer".into(),
                                    ))
                                }
                            }
                        } else {
                            v.len()
                        };
                        let s = start.min(v.len());
                        Ok(Value::List(Rc::new(v[s..end].to_vec())))
                    }
                    _ => Err(HayashiError::Type("slice() requires list".into())),
                }
            }

            "join" => {
                if args.len() < 1 || args.len() > 2 {
                    return Err(HayashiError::Runtime("join(lista [, separador])".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let sep = if args.len() == 2 {
                    match self.eval_expr(&args[1])? {
                        Value::Str(s) => s,
                        _ => {
                            return Err(HayashiError::Type("join: separator must be string".into()))
                        }
                    }
                } else {
                    ", ".to_string()
                };
                match lst {
                    Value::List(v) => {
                        let strs: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                        Ok(Value::Str(strs.join(&sep)))
                    }
                    _ => Err(HayashiError::Type("join() requires list".into())),
                }
            }

            "map" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "map(list, fn) or map(list, |x| expr)".into(),
                    ));
                }
                let lst = match self.eval_expr(&args[0])? {
                    Value::List(v) => v,
                    _ => return Err(HayashiError::Type("map() requires list".into())),
                };
                let fn_val = self.eval_expr(&args[1])?;
                let mut result = Vec::with_capacity(lst.len());
                for item in lst.iter() {
                    let val = self.call_value_fn(&fn_val, &[item.clone()])?;
                    result.push(val);
                }
                Ok(Value::List(Rc::new(result)))
            }

            "unique" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("unique(lista)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(v) => {
                        let mut seen = Vec::new();
                        let mut result = Vec::new();
                        for item in v.iter() {
                            let key = format!("{item}");
                            if !seen.contains(&key) {
                                seen.push(key);
                                result.push(item.clone());
                            }
                        }
                        Ok(Value::List(Rc::new(result)))
                    }
                    _ => Err(HayashiError::Type("unique() requires list".into())),
                }
            }

            "flatten" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("flatten(lista)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(v) => {
                        let mut result = Vec::new();
                        for item in v.iter() {
                            match item {
                                Value::List(inner) => result.extend(inner.iter().cloned()),
                                other => result.push(other.clone()),
                            }
                        }
                        Ok(Value::List(Rc::new(result)))
                    }
                    _ => Err(HayashiError::Type("flatten() requires list".into())),
                }
            }

            "chain" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("chain(seq1, seq2, ...)".into()));
                }
                let mut result = Vec::new();
                for arg in args {
                    match self.eval_expr(arg)? {
                        Value::List(v) => result.extend(v.iter().cloned()),
                        other => return Err(self.type_mismatch("List", &other)),
                    }
                }
                Ok(Value::List(Rc::new(result)))
            }

            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime(
                        "range(start, end [, step]) requires 2 or 3 arguments".into(),
                    ));
                }
                let start = match self.eval_expr(&args[0])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(HayashiError::Type("range: start must be integer".into())),
                };
                let end = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(HayashiError::Type("range: end must be integer".into())),
                };
                let step: i64 = if args.len() == 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Int(i) => i,
                        Value::Float(f) => f as i64,
                        _ => return Err(HayashiError::Type("range: step must be integer".into())),
                    }
                } else if start <= end {
                    1
                } else {
                    -1
                };
                if step == 0 {
                    return Err(HayashiError::Runtime("range: step cannot be zero".into()));
                }
                let mut v = Vec::new();
                let mut cur = start;
                while if step > 0 { cur < end } else { cur > end } {
                    v.push(Value::Int(cur));
                    cur += step;
                }
                Ok(Value::List(Rc::new(v)))
            }

            // ── reg → alias para ols ──────────────────────────────────────────
            "reg" | "regress" => {
                return self.eval_call("ols", args, opts);
            }

            // ── Fama-MacBeth (1973) ──────────────────────────────────────────
            // fmb(formula, df, time=col)
            // Cross-sectional regressions por período, média dos coeficientes
            // SE = σ(β̂_t) / √T  (Fama-MacBeth standard errors)
            "fmb" | "fama_macbeth" | "xtfmb" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("fmb(formula, df, time=col)".into()));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let time_col = match opt_map.get("time") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(_, t)| t.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            HayashiError::Runtime(
                                "fmb requires time=col or xtset(df, id, time)".into(),
                            )
                        })?,
                };
                let nw_lags: usize = match opt_map.get("nw") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    Some(Value::Str(s)) => s.parse().unwrap_or(0),
                    _ => 0,
                };

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::FamaMacBeth::fit(&g_formula, &df, &time_col, nw_lags)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("{result}");
                Ok(Value::Nil)
            }

            // ── portsort: portfolio sorts por quantis ────────────────────────
            // portsort(df, ret, sort_var, n=5)
            // Ordena observações por sort_var, divide em n portfólios,
            // reporta média, SE e t de ret por portfólio + spread H-L.
            "portsort" | "portfolio_sort" | "psort" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "portsort(df, ret_var, sort_var, n=5)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df_raw = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df_raw, opts)?;
                let ret_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be return variable".into(),
                        ))
                    }
                };
                let sort_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "third argument must be sort variable".into(),
                        ))
                    }
                };
                let n_ports: usize = match opt_map.get("n") {
                    Some(Value::Int(v)) => (*v).max(2) as usize,
                    Some(Value::Float(v)) => (*v as usize).max(2),
                    _ => 5,
                };

                let ret_col = Self::get_col_f64(&df, &ret_name)?;
                let sort_col = Self::get_col_f64(&df, &sort_name)?;

                // pares (sort_val, ret_val) — excluir NaN
                let mut pairs: Vec<(f64, f64)> = sort_col
                    .iter()
                    .zip(ret_col.iter())
                    .filter(|(s, r)| s.is_finite() && r.is_finite())
                    .map(|(&s, &r)| (s, r))
                    .collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let n_valid = pairs.len();
                let per_port = n_valid / n_ports;

                if per_port < 1 {
                    return Err(HayashiError::Runtime(format!(
                        "portsort: {n_valid} obs válidas insuficientes para {n_ports} portfólios"
                    )));
                }

                // atribuir portfólios
                struct PortStats {
                    mean: f64,
                    se: f64,
                    n: usize,
                }
                let mut ports: Vec<PortStats> = Vec::new();
                for p in 0..n_ports {
                    let start = p * per_port;
                    let end = if p == n_ports - 1 {
                        n_valid
                    } else {
                        (p + 1) * per_port
                    };
                    let rets: Vec<f64> = pairs[start..end].iter().map(|(_, r)| *r).collect();
                    let n = rets.len();
                    let mean = rets.iter().sum::<f64>() / n as f64;
                    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0).max(1.0);
                    let se = (var / n as f64).sqrt();
                    ports.push(PortStats { mean, se, n });
                }

                // spread H-L
                let hl_mean = ports.last().unwrap().mean - ports[0].mean;
                let hl_se = (ports.last().unwrap().se.powi(2) + ports[0].se.powi(2)).sqrt();
                let hl_t = if hl_se > 1e-15 {
                    hl_mean / hl_se
                } else {
                    f64::NAN
                };
                let hl_p = t_pvalue_two(hl_t, (ports.last().unwrap().n + ports[0].n - 2) as f64);

                let thick = "═".repeat(60);
                let thin = "─".repeat(60);
                println!("\n{thick}");
                println!(
                    "{:^60}",
                    format!(" Portfolio Sort: {ret_name} by {sort_name} ({n_ports} groups) ")
                );
                println!("{thin}");
                println!(
                    "{:<12} {:>8} {:>12} {:>10} {:>10}",
                    "Portfolio", "N", "Mean", "SE", "t"
                );
                println!("{thin}");
                for (i, ps) in ports.iter().enumerate() {
                    let t = if ps.se > 1e-15 {
                        ps.mean / ps.se
                    } else {
                        f64::NAN
                    };
                    let label = if i == 0 {
                        "Low".to_string()
                    } else if i == n_ports - 1 {
                        "High".to_string()
                    } else {
                        format!("P{}", i + 1)
                    };
                    println!(
                        "{:<12} {:>8} {:>12.4} {:>10.4} {:>10.4}",
                        label, ps.n, ps.mean, ps.se, t
                    );
                }
                println!("{thin}");
                let sig = if hl_p < 0.01 {
                    "***"
                } else if hl_p < 0.05 {
                    "**"
                } else if hl_p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!(
                    "{:<12} {:>8} {:>12.4} {:>10.4} {:>10.4} {sig}",
                    "H-L", "", hl_mean, hl_se, hl_t
                );
                println!("{thick}\n");
                Ok(Value::Nil)
            }

            // ── doublesort: portfolio sort bidimensional (Fama-French) ─────
            // doublesort(df, ret, sort1, sort2, n1=5, n2=5)
            "doublesort" | "double_sort" | "bivariate_sort" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "doublesort(df, ret, sort1, sort2, n1=5, n2=5)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df_raw = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df_raw, opts)?;
                let ret_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("ret var".into())),
                };
                let s1_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("sort1 var".into())),
                };
                let s2_name = match &args[3] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("sort2 var".into())),
                };
                let n1: usize = match opt_map.get("n1") {
                    Some(Value::Int(v)) => (*v).max(2) as usize,
                    _ => 5,
                };
                let n2: usize = match opt_map.get("n2") {
                    Some(Value::Int(v)) => (*v).max(2) as usize,
                    _ => 5,
                };

                let ret_col = Self::get_col_f64(&df, &ret_name)?;
                let s1_col = Self::get_col_f64(&df, &s1_name)?;
                let s2_col = Self::get_col_f64(&df, &s2_name)?;

                // atribuir quantis independentes
                let assign_quantile = |vals: &[f64], n_q: usize| -> Vec<usize> {
                    let mut indexed: Vec<(usize, f64)> = vals
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.is_finite())
                        .map(|(i, &v)| (i, v))
                        .collect();
                    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let n = indexed.len();
                    let mut q = vec![usize::MAX; vals.len()];
                    for (rank, &(orig_i, _)) in indexed.iter().enumerate() {
                        q[orig_i] = (rank * n_q / n).min(n_q - 1);
                    }
                    q
                };

                let s1_vec: Vec<f64> = s1_col.to_vec();
                let s2_vec: Vec<f64> = s2_col.to_vec();
                let q1 = assign_quantile(&s1_vec, n1);
                let q2 = assign_quantile(&s2_vec, n2);

                // médias por célula (q1 x q2)
                let mut cell_sum = vec![vec![0.0; n2]; n1];
                let mut cell_n = vec![vec![0usize; n2]; n1];
                for i in 0..ret_col.len() {
                    if q1[i] < n1 && q2[i] < n2 && ret_col[i].is_finite() {
                        cell_sum[q1[i]][q2[i]] += ret_col[i];
                        cell_n[q1[i]][q2[i]] += 1;
                    }
                }

                let thick = "═".repeat(12 + n2 * 10);
                let thin = "─".repeat(12 + n2 * 10);
                println!("\n{thick}");
                println!(" Double Sort: {ret_name} by {s1_name} (rows) × {s2_name} (cols)");
                println!("{thin}");
                print!("{:<12}", format!("{s1_name}\\{s2_name}"));
                for j in 0..n2 {
                    let label = if j == 0 {
                        "Low"
                    } else if j == n2 - 1 {
                        "High"
                    } else {
                        &format!("Q{}", j + 1)
                    };
                    print!("{:>10}", label);
                }
                println!();
                println!("{thin}");
                for i in 0..n1 {
                    let label = if i == 0 {
                        "Low".to_string()
                    } else if i == n1 - 1 {
                        "High".to_string()
                    } else {
                        format!("Q{}", i + 1)
                    };
                    print!("{:<12}", label);
                    for j in 0..n2 {
                        let mean = if cell_n[i][j] > 0 {
                            cell_sum[i][j] / cell_n[i][j] as f64
                        } else {
                            f64::NAN
                        };
                        if mean.is_nan() {
                            print!("{:>10}", ".");
                        } else {
                            print!("{:>10.4}", mean);
                        }
                    }
                    println!();
                }
                println!("{thick}\n");
                Ok(Value::Nil)
            }

            // ── OLS ───────────────────────────────────────────────────────────
            "ols" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ols() requires (formula, dataframe)".into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df_name = match &args[1] {
                    Expr::Var(name) => name.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be a DataFrame variable".into(),
                        ))
                    }
                };
                let df_raw = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df_raw, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let cov = Self::resolve_cov_full(&opt_map, &df)?;

                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = OLS::from_formula(&g_formula, &df, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let fitted = result.x_clean.as_ref().unwrap_or(&x).dot(&result.params);
                let residuals = &y - &fitted;
                let x_used = result.x_clean.clone().unwrap_or(x);

                Ok(Value::OlsResult(OlsModel {
                    result: Rc::new(result),
                    residuals,
                    x: x_used,
                }))
            }

            // ── IV / 2SLS ─────────────────────────────────────────────────────
            "iv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "iv() requires (endog_formula, instrument_formula, dataframe)".into(),
                    ));
                }
                let endog_ast = self.resolve_formula(&args[0])?;
                let instr_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "third argument must be a DataFrame variable".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let cov = Self::resolve_cov_full(&opt_map, &df)?;

                let endog_str = Self::formula_to_string(&endog_ast);
                let instr_str = Self::formula_to_string(&instr_ast);

                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // A fórmula dos instrumentos pode ter LHS vazio (sintaxe ~ z1 + z2).
                // GFormula::parse rejeita LHS vazio; construímos diretamente.
                let g_instr = if instr_ast.lhs.is_empty() {
                    let independents: Vec<String> = instr_ast
                        .rhs
                        .iter()
                        .map(|t| match t {
                            RhsTerm::Var(v) => v.clone(),
                            RhsTerm::Categorical(v) => format!("C({v})"),
                            RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                            RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
                        })
                        .collect();
                    GFormula {
                        dependent: String::new(),
                        independents,
                        intercept: true,
                    }
                } else {
                    GFormula::parse(&instr_str).map_err(|e| HayashiError::Runtime(e.to_string()))?
                };

                let result = IV::from_formula(&g_endog, &g_instr, &df, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::IvResult(Rc::new(result)))
            }

            // ── Teste de instrumentos fracos (Cragg-Donald / Stock-Yogo) ──────
            // weak_iv(endog_formula, instrument_formula, df)
            // Mesma sintaxe do iv(). Calcula F de 1ª etapa (por endog) e
            // estatística de Cragg-Donald. Compara com valores críticos de
            // Stock & Yogo (2005).
            "weak_iv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "weak_iv() requer (formula_estrutural, formula_instrumentos, df)".into(),
                    ));
                }
                let endog_ast = self.resolve_formula(&args[0])?;
                let instr_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "weak_iv(): third argument must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => {
                        return Err(self.rt_err(format!("weak_iv: '{df_name}' is not a DataFrame")))
                    }
                };

                // ── Identifica variáveis ──
                let endog_vars: std::collections::HashSet<String> = endog_ast
                    .rhs
                    .iter()
                    .map(|t| match t {
                        RhsTerm::Var(v) => v.clone(),
                        _ => String::new(),
                    })
                    .filter(|s| !s.is_empty())
                    .collect();
                let instr_vars: std::collections::HashSet<String> = instr_ast
                    .rhs
                    .iter()
                    .map(|t| match t {
                        RhsTerm::Var(v) => v.clone(),
                        _ => String::new(),
                    })
                    .filter(|s| !s.is_empty())
                    .collect();

                // endógenas = em endog mas NÃO em instr
                let x_endog_names: Vec<String> = endog_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .filter(|v| !instr_vars.contains(v))
                    .collect();
                // instrumentos excluídos = em instr mas NÃO em endog
                let z_excl_names: Vec<String> = instr_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .filter(|v| !endog_vars.contains(v))
                    .collect();
                // exógenos incluídos = em ambos
                let x_exog_names: Vec<String> = instr_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .filter(|v| endog_vars.contains(v.as_str()))
                    .collect();

                if x_endog_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "weak_iv: nenhuma variável endógena identificada (vars em endog mas não em instr)".into()
                    ));
                }
                if z_excl_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "weak_iv: nenhum instrumento excluído identificado (vars em instr mas não em endog)".into()
                    ));
                }

                let n = df.n_rows();
                let k_endog = x_endog_names.len();
                let l = z_excl_names.len(); // número de instrumentos excluídos
                let k_exog = x_exog_names.len() + 1; // +1 intercepto

                // ── Monta matrizes ──
                // X_exog: intercepto + exógenos incluídos  (n × k_exog)
                let mut x_exog = Array2::<f64>::ones((n, k_exog));
                for (j, col) in x_exog_names.iter().enumerate() {
                    let v = df
                        .get(col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n {
                        x_exog[[i, j + 1]] = v[i];
                    }
                }

                // Z_excl: instrumentos excluídos  (n × L)
                let mut z_excl = Array2::<f64>::zeros((n, l));
                for (j, col) in z_excl_names.iter().enumerate() {
                    let v = df
                        .get(col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n {
                        z_excl[[i, j]] = v[i];
                    }
                }

                // W = [X_exog | Z_excl]  (n × (k_exog + L))
                let mut w_full = Array2::<f64>::zeros((n, k_exog + l));
                w_full.slice_mut(ndarray::s![.., ..k_exog]).assign(&x_exog);
                w_full.slice_mut(ndarray::s![.., k_exog..]).assign(&z_excl);

                // X_endog: variáveis endógenas  (n × k_endog)
                let mut x_endog_mat = Array2::<f64>::zeros((n, k_endog));
                for (j, col) in x_endog_names.iter().enumerate() {
                    let v = df
                        .get(col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n {
                        x_endog_mat[[i, j]] = v[i];
                    }
                }

                // ── M_exog = I - X_exog (X_exog'X_exog)⁻¹ X_exog' ──
                // para partial out os exógenos incluídos
                let xtx_exog = x_exog.t().dot(&x_exog);
                let xtx_exog_inv = xtx_exog
                    .inv()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // P_exog aplicado a qualquer matriz A: P_exog A = X_exog (X_exog'X_exog)⁻¹ X_exog' A
                let proj_exog = |a: &Array2<f64>| -> Array2<f64> {
                    x_exog.dot(&xtx_exog_inv.dot(&x_exog.t().dot(a)))
                };
                // M_exog Z_excl (partialling out exog de Z_excl)
                let mz = &z_excl - &proj_exog(&z_excl); // n × L
                                                        // M_exog X_endog
                let _mx = &x_endog_mat - &proj_exog(&x_endog_mat); // n × k_endog

                // ── Primeira etapa: regride X_endog em W_full ──
                let wtw = w_full.t().dot(&w_full);
                let wtw_inv = wtw
                    .inv()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let pi_hat = wtw_inv.dot(&w_full.t().dot(&x_endog_mat)); // (k_exog+L) × k_endog
                let x_hat = w_full.dot(&pi_hat); // n × k_endog
                let v_hat = &x_endog_mat - &x_hat; // resíduos 1ª etapa

                // ── Π̂_Z: linhas de pi_hat correspondentes a Z_excl ──
                let pi_z = pi_hat.slice(ndarray::s![k_exog.., ..]).to_owned(); // L × k_endog

                // ── Σ̂_v = v̂'v̂ / (n - k_exog - L) ──
                let df_fs = n - k_exog - l;
                let vtv = v_hat.t().dot(&v_hat); // k_endog × k_endog
                let sigma_v = &vtv / df_fs as f64;

                // ── Matriz de Cragg-Donald: A = Π̂_Z' (Z'M_exog Z) Π̂_Z ──
                let zmz = mz.t().dot(&mz); // L × L  (= Z'M_exog Z)
                let cd_mat = pi_z.t().dot(&zmz.dot(&pi_z)); // k_endog × k_endog

                // ── F de 1ª etapa por variável endógena (partial F em Z_excl) ──
                let mut first_stage_lines = String::new();
                for j in 0..k_endog {
                    // partial F = (Π̂_Zj' Z'M Z Π̂_Zj / L) / Σ̂_vj
                    let pi_zj = pi_z.column(j);
                    let numerator = pi_zj.dot(&zmz.dot(&pi_zj)) / l as f64;
                    let sigma_vj = sigma_v[[j, j]];
                    let f_j = if sigma_vj > 1e-15 {
                        numerator / sigma_vj
                    } else {
                        f64::NAN
                    };
                    let p_j = if f_j.is_finite() {
                        f_pvalue(f_j, l as f64, df_fs as f64)
                    } else {
                        f64::NAN
                    };
                    first_stage_lines.push_str(&format!(
                        "   {:<20} F({},{}) = {:>10.3}   p = {:.4}\n",
                        x_endog_names[j], l, df_fs, f_j, p_j
                    ));
                }

                // ── Cragg-Donald Wald F ──
                let sigma_v_inv = sigma_v
                    .inv()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cd_core = sigma_v_inv.dot(&cd_mat); // k_endog × k_endog

                let cd_stat = if k_endog == 1 {
                    cd_core[[0, 0]] / l as f64
                } else {
                    // λ_min de cd_core / L
                    let (eigenvalues, _) = cd_core
                        .eigh(UPLO::Lower)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    eigenvalues[0] / l as f64 // eigenvalues em ordem crescente
                };

                // ── Valores críticos de Stock & Yogo (2005) (k_endog=1, bias TSLS) ──
                let sy_table: Vec<(usize, [f64; 4])> = vec![
                    (1, [16.38, 8.96, 6.66, 5.53]),
                    (2, [19.93, 11.59, 8.75, 7.25]),
                    (3, [22.30, 12.83, 9.54, 7.80]),
                    (4, [24.58, 13.96, 10.26, 8.31]),
                    (5, [26.87, 15.09, 11.04, 8.84]),
                    (6, [28.55, 16.00, 11.65, 9.23]),
                    (7, [30.10, 16.87, 12.26, 9.63]),
                    (8, [31.49, 17.60, 12.82, 10.00]),
                    (9, [32.84, 18.37, 13.44, 10.37]),
                    (10, [34.16, 19.10, 14.01, 10.73]),
                ];
                let sy_line = if k_endog == 1 {
                    if let Some((_, cvs)) = sy_table.iter().find(|(lv, _)| *lv == l) {
                        format!(
                            "   Stock-Yogo (2005) — valores críticos para viés TSLS máximo (k_endog=1, L={}):\n   10%:{:.2}  15%:{:.2}  20%:{:.2}  25%:{:.2}\n",
                            l, cvs[0], cvs[1], cvs[2], cvs[3]
                        )
                    } else {
                        format!("   Stock-Yogo (2005): tabela disponível para L=1..10 (L={} out of range).\n   Regra de bolso (Staiger & Stock 1997): F > 10.\n", l)
                    }
                } else {
                    format!("   Stock-Yogo (2005): valores críticos para k_endog=1 apenas.\n   Para k_endog={}, consulte tabelas de Andrews, Stock & Sun (2019).\n", k_endog)
                };

                let thick = "═".repeat(70);
                let thin = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Teste de Instrumentos Fracos\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    " n={n}  k_endog={k_endog}  L={l} (instrumentos excluídos)\n"
                ));
                out.push_str("\n── F de 1ª Etapa (partial F em instrumentos excluídos)\n");
                out.push_str(&first_stage_lines);
                out.push_str(&format!("\n── Cragg-Donald Wald F = {:.4}\n", cd_stat));
                out.push_str(&format!("   (λ_min do núcleo de concentração / L)\n"));
                out.push_str(&format!("\n{sy_line}"));
                out.push_str(&format!("{thin}\n"));
                out.push_str(" Regra de bolso: F > 10 (Staiger & Stock 1997)\n");
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Logit ─────────────────────────────────────────────────────────
            "logit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = Logit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let coef_names = Self::coef_names_from_formula(&formula_ast, &df, x.ncols());
                Ok(Value::BinaryResult(BinaryModel {
                    result: Rc::new(result),
                    y,
                    x,
                    kind: "logit".into(),
                    coef_names,
                }))
            }

            // ── Probit ────────────────────────────────────────────────────────
            "probit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = Probit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let coef_names = Self::coef_names_from_formula(&formula_ast, &df, x.ncols());
                Ok(Value::BinaryResult(BinaryModel {
                    result: Rc::new(result),
                    y,
                    x,
                    kind: "probit".into(),
                    coef_names,
                }))
            }

            // ── Heckman Two-Step (Heckit) ─────────────────────────────────────
            // heckman(outcome_formula, select_formula, df)
            // outcome: y ~ x1 + x2       (estimado apenas nos obs selecionados)
            // select:  z ~ w1 + w2 + w3  (probit em todos os obs; z deve ser 0/1)
            "heckman" | "heckit" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "heckman() requer (formula_resultado, formula_seleção, df)".into(),
                    ));
                }
                let out_ast = self.resolve_formula(&args[0])?;
                let sel_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "heckman(): terceiro argumento deve ser nome do DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "heckman: '{df_name}' is not a DataFrame"
                        )))
                    }
                };

                // Equação de resultado
                let out_str = Self::formula_to_string(&out_ast);
                let g_out =
                    GFormula::parse(&out_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec_raw, x_out) = df
                    .to_design_matrix(&g_out)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let out_names = df
                    .formula_var_names(&g_out)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Equação de seleção
                let sel_str = Self::formula_to_string(&sel_ast);
                let g_sel =
                    GFormula::parse(&sel_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (z_vec, x_sel) = df
                    .to_design_matrix(&g_sel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let sel_names = df
                    .formula_var_names(&g_sel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Heckman: y e x_out podem conter NaN para obs não-selecionadas (z=0).
                // Substituir NaN/Inf por 0.0 nessas linhas (valores não são usados na equação de resultado).
                let y_vec = y_vec_raw.mapv(|v| if v.is_finite() { v } else { 0.0 });
                let x_out = x_out.mapv(|v| if v.is_finite() { v } else { 0.0 });

                let result = greeners::Heckman::fit(
                    &y_vec,
                    &x_out,
                    &z_vec,
                    &x_sel,
                    Some(out_names),
                    Some(sel_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::HeckmanResult(Rc::new(result)))
            }

            // ── Tobit — MLE com censura esquerda ──────────────────────────────
            // tobit(formula, df [, ll=0])
            "tobit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ll_limit = match opt_map.get("ll") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    None => 0.0,
                    _ => return Err(HayashiError::Runtime("tobit(): ll must be numeric".into())),
                };
                let result = greeners::Tobit::fit(&y_vec, &x_mat, ll_limit, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::TobitResult(Rc::new(result)))
            }

            // ── Regressão Descontínua — Sharp RD ─────────────────────────────
            // rd(outcome ~ running_var, cutoff, df [, bw=h, poly=1, kernel="triangular"])
            "rd" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "rd() requer (formula, cutoff, df [, bw=..., poly=..., kernel=...])".into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let cutoff = match self.eval_expr(&args[1])? {
                    Value::Float(v) => v,
                    Value::Int(v) => v as f64,
                    _ => {
                        return Err(HayashiError::Type(
                            "rd(): second argument must be o cutoff (número)".into(),
                        ))
                    }
                };
                let df = match self.eval_expr(&args[2])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "rd(): third argument must be DataFrame".into(),
                        ))
                    }
                };

                // Extrair nomes diretamente do AST da fórmula Hayashi
                let outcome_name = formula_ast.lhs.clone();
                let running_name = formula_ast.rhs.first()
                    .and_then(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .ok_or_else(|| HayashiError::Runtime(
                        "rd(): fórmula deve ter exatamente uma variável no lado direito (running var)".into()
                    ))?;

                let y = df
                    .get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let x = df
                    .get(&running_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();

                let bw = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("rd: bw must be numeric".into())),
                };
                let poly = match opt_map.get("poly") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => return Err(HayashiError::Runtime("rd: poly must be integer".into())),
                };
                let kernel =
                    rd_kernel_opt(opt_map.get("kernel")).map_err(|e| HayashiError::Runtime(e))?;

                let result = greeners::RD::fit(
                    &y,
                    &x,
                    cutoff,
                    bw,
                    poly,
                    kernel,
                    Some((outcome_name, running_name)),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RdResult(Rc::new(result)))
            }

            // ── Regressão Descontínua — Fuzzy RD ─────────────────────────────
            // fuzzy_rd(outcome ~ running_var, "treatment_col", cutoff, df [, bw=h, poly=1])
            "fuzzy_rd" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "fuzzy_rd() requer (formula, \"treatment\", cutoff, df [, bw=..., poly=...])".into()
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let treatment_name = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type(
                        "fuzzy_rd(): second argument must be o nome da coluna de tratamento (string)".into()
                    )),
                };
                let cutoff = match self.eval_expr(&args[2])? {
                    Value::Float(v) => v,
                    Value::Int(v) => v as f64,
                    _ => {
                        return Err(HayashiError::Type(
                            "fuzzy_rd(): third argument must be cutoff (número)".into(),
                        ))
                    }
                };
                let df = match self.eval_expr(&args[3])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "fuzzy_rd(): fourth argument must be DataFrame".into(),
                        ))
                    }
                };

                let outcome_name = formula_ast.lhs.clone();
                let running_name = formula_ast.rhs.first()
                    .and_then(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .ok_or_else(|| HayashiError::Runtime(
                        "fuzzy_rd(): fórmula deve ter exatamente uma variável no lado direito (running var)".into()
                    ))?;

                let y = df
                    .get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let d = df
                    .get(&treatment_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let x = df
                    .get(&running_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();

                let bw = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("fuzzy_rd: bw must be numeric".into())),
                };
                let poly = match opt_map.get("poly") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "fuzzy_rd: poly must be integer".into(),
                        ))
                    }
                };
                let kernel =
                    rd_kernel_opt(opt_map.get("kernel")).map_err(|e| HayashiError::Runtime(e))?;

                let result = greeners::RD::fit_fuzzy(
                    &y,
                    &d,
                    &x,
                    cutoff,
                    bw,
                    poly,
                    kernel,
                    Some((outcome_name, running_name, treatment_name)),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RdResult(Rc::new(result)))
            }

            // ── Propensity Score Matching (Rosenbaum & Rubin 1983) ───────────
            // psm(outcome ~ treatment + cov1 + cov2, df [, k=1, caliper=0.2, replace=false, boot=200])
            // O 1º termo RHS é o tratamento; demais são covariáveis para o PS.
            "psm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "psm() requer (formula, df [, k=..., caliper=..., replace=..., boot=...])"
                            .into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "psm(): second argument must be DataFrame".into(),
                        ))
                    }
                };

                let outcome_name = formula_ast.lhs.clone();
                // Primeiro RHS = tratamento; demais = covariáveis
                let mut rhs_names: Vec<String> = formula_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if rhs_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "psm(): fórmula deve ter ao menos 'outcome ~ treatment'".into(),
                    ));
                }
                let treatment_name = rhs_names.remove(0);
                let covariate_names = rhs_names;

                if covariate_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "psm(): forneça ao menos uma covariável: outcome ~ treatment + cov1 + ..."
                            .into(),
                    ));
                }

                let y = df
                    .get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let d = df
                    .get(&treatment_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();

                let x = {
                    let owned_cols: Vec<ndarray::Array1<f64>> = covariate_names
                        .iter()
                        .map(|c| {
                            df.get(c)
                                .map(|a| a.to_owned())
                                .map_err(|e| HayashiError::Runtime(e.to_string()))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let views: Vec<ndarray::ArrayView1<f64>> =
                        owned_cols.iter().map(|a| a.view()).collect();
                    ndarray::stack(ndarray::Axis(1), &views)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                };

                let k = match opt_map.get("k") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => return Err(HayashiError::Runtime("psm: k must be integer".into())),
                };
                let caliper: Option<f64> = match opt_map.get("caliper") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("psm: caliper must be numeric".into())),
                };
                let with_replacement = match opt_map.get("replace") {
                    Some(Value::Bool(b)) => *b,
                    None => false,
                    _ => return Err(HayashiError::Runtime("psm: replace must be boolean".into())),
                };
                let n_boot = match opt_map.get("boot") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 200,
                    _ => return Err(HayashiError::Runtime("psm: boot must be integer".into())),
                };

                let result = greeners::PSM::fit(
                    &y,
                    &d,
                    &x,
                    k,
                    caliper,
                    with_replacement,
                    n_boot,
                    Some((outcome_name, treatment_name, covariate_names)),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PsmResult(Rc::new(result)))
            }

            // ── Controle Sintético (ADH 2010) ────────────────────────────────
            // synth("outcome", "treated_id", t0, df, id="entity", time="year")
            // synth("outcome", "treated_id", t0, df, id="entity", time="year", covs=["x1","x2"])
            "synth" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "synth() requer (outcome, treated_id, t0, df, id=col, time=col [, covs=[...]])".into()
                    ));
                }
                let outcome_col =
                    match self.eval_expr(&args[0])? {
                        Value::Str(s) => s,
                        _ => return Err(HayashiError::Type(
                            "synth(): first argument must be nome da coluna de resultado (string)"
                                .into(),
                        )),
                    };
                let treated_unit = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    Value::Int(v) => v.to_string(),
                    Value::Float(v) => (v as i64).to_string(),
                    _ => {
                        return Err(HayashiError::Type(
                            "synth(): second argument must be o ID da unidade tratada".into(),
                        ))
                    }
                };
                let t0 = match self.eval_expr(&args[2])? {
                    Value::Float(v) => v,
                    Value::Int(v)   => v as f64,
                    _ => return Err(HayashiError::Type(
                        "synth(): third argument must be o período de início do tratamento (número)".into()
                    )),
                };
                let df = match self.eval_expr(&args[3])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "synth(): fourth argument must be DataFrame".into(),
                        ))
                    }
                };

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "synth(): opção id=coluna é obrigatória".into(),
                        ))
                    }
                };
                let time_col = match opt_map.get("time") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "synth(): opção time=coluna é obrigatória".into(),
                        ))
                    }
                };
                let cov_cols: Option<Vec<String>> = match opt_map.get("covs") {
                    Some(Value::List(lst)) => Some(
                        lst.iter()
                            .map(|v| match v {
                                Value::Str(s) => Ok(s.clone()),
                                _ => Err(HayashiError::Type(
                                    "synth(): covs must be a list de strings".into(),
                                )),
                            })
                            .collect::<Result<Vec<_>>>()?,
                    ),
                    None => None,
                    _ => return Err(HayashiError::Runtime("synth(): covs must be a list".into())),
                };

                let result = greeners::SyntheticControl::fit(
                    &outcome_col,
                    &treated_unit,
                    t0,
                    &df,
                    &id_col,
                    &time_col,
                    cov_cols.as_deref(),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::SynthResult(Rc::new(result)))
            }

            // ── Poisson ───────────────────────────────────────────────────────
            "poisson" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let result =
                    greeners::Poisson::fit_with_names(&y_vec, &x_mat, cov, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PoissonResult(Rc::new(result)))
            }

            // ── Negative Binomial (NB2) ───────────────────────────────────────
            "nbreg" | "negbin" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let result = greeners::NegBin::fit_with_names(&y_vec, &x_mat, cov, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::NegBinResult(Rc::new(result)))
            }

            // ── Ordered Logit ─────────────────────────────────────────────────
            "ologit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::OrderedLogit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::OrderedResult(Rc::new(result)))
            }

            // ── Ordered Probit ────────────────────────────────────────────────
            "oprobit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::OrderedProbit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::OrderedResult(Rc::new(result)))
            }

            // ── Multinomial Logit ─────────────────────────────────────────────
            "mlogit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::MNLogit::fit_with_names(&y_vec, &x_mat, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::MNLogitResult(Rc::new(result)))
            }

            // ── Difference-in-Differences (2x2) ──────────────────────────────
            // did(outcome ~ treated_group + post_period, df, cov=HC1)
            "did" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "did(outcome ~ tratado + pos, df) requer fórmula e DataFrame".into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "did(): second argument must be DataFrame".into(),
                        ))
                    }
                };
                // formula: outcome ~ treated_col + post_col
                let rhs_vars: Vec<&str> = formula_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                if rhs_vars.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "did(): fórmula deve ter exatamente 2 variáveis no RHS: treated + post"
                            .into(),
                    ));
                }
                let y = Self::get_col_f64(&df, &formula_ast.lhs)?;
                let treated = Self::get_col_f64(&df, rhs_vars[0])?;
                let post = Self::get_col_f64(&df, rhs_vars[1])?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let result = greeners::DiffInDiff::fit(&y, &treated, &post, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::DidResult(Rc::new(result)))
            }

            // ── Quantile Regression ───────────────────────────────────────────
            // qreg(y ~ x1 + x2, df, tau=0.5, boot=200)
            "qreg" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let tau = match opt_map.get("tau") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    None => 0.5,
                    _ => return Err(HayashiError::Type("tau= must be numeric".into())),
                };
                let n_boot = match opt_map.get("boot") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 200,
                    _ => return Err(HayashiError::Type("boot= must be integer".into())),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::QuantileReg::fit_with_names(
                    &y_vec,
                    &x_mat,
                    tau,
                    n_boot,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::QuantileResult(Rc::new(result)))
            }

            // ── Kaplan-Meier ──────────────────────────────────────────────────
            // km(time_col, event_col, df)
            "km" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "km(time, event, df) requires 3 arguments".into(),
                    ));
                }
                let time_name = match &args[0] {
                    Expr::Var(v) | Expr::Str(v) => v.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "km(): first argument must be nome da coluna de tempo".into(),
                        ))
                    }
                };
                let event_name = match &args[1] {
                    Expr::Var(v) | Expr::Str(v) => v.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "km(): second argument must be nome da coluna de evento".into(),
                        ))
                    }
                };
                let df = match self.eval_expr(&args[2])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "km(): third argument must be DataFrame".into(),
                        ))
                    }
                };
                let times = Self::get_col_f64(&df, &time_name)?;
                let events_f = Self::get_col_f64(&df, &event_name)?;
                let events: ndarray::Array1<u8> = events_f.iter().map(|&v| v as u8).collect();
                let result = greeners::KaplanMeier::fit(&times, &events)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::KMResult(Rc::new(result)))
            }

            // ── Cox Proportional Hazards ──────────────────────────────────────
            // cox(time_col ~ x1 + x2, df, event=event_col)
            "cox" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "cox(time ~ x1 + x2, df, event=col) requer fórmula e DataFrame".into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "cox(): second argument must be DataFrame".into(),
                        ))
                    }
                };
                let event_col = match opt_map.get("event") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "cox() requer opção event=nome_coluna".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("event= must be string".into())),
                };
                let times = Self::get_col_f64(&df, &formula_ast.lhs)?;
                let events_f = Self::get_col_f64(&df, &event_col)?;
                let events: ndarray::Array1<u8> = events_f.iter().map(|&v| v as u8).collect();
                // build covariate matrix from RHS variables
                let rhs_vars: Vec<String> = formula_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if rhs_vars.is_empty() {
                    return Err(HayashiError::Runtime(
                        "cox(): fórmula precisa de ao menos uma covariável no RHS".into(),
                    ));
                }
                let cols: Vec<ndarray::Array1<f64>> = rhs_vars
                    .iter()
                    .map(|v| Self::get_col_f64(&df, v))
                    .collect::<Result<_>>()?;
                let n = times.len();
                let k = cols.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, col) in cols.iter().enumerate() {
                    x_mat.column_mut(j).assign(col);
                }
                let result =
                    greeners::CoxPH::fit_with_names(&times, &events, &x_mat, Some(rhs_vars))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::CoxResult(Rc::new(result)))
            }

            // ── Robust Linear Model (M-estimadores) ───────────────────────────
            // rlm(y ~ x1 + x2, df, norm=huber|tukey|andrews|hampel, cov=HC3)
            // norm padrão: Huber (c=1.345)
            "rlm" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let norm = match opt_map.get("norm") {
                    None => greeners::RobustNorm::Huber(1.345),
                    Some(Value::Str(s)) => match s.as_str() {
                        "huber" => greeners::RobustNorm::Huber(1.345),
                        "tukey" | "bisquare" => greeners::RobustNorm::Tukey(4.685),
                        "andrews" | "wave" => {
                            greeners::RobustNorm::AndrewWave(std::f64::consts::PI)
                        }
                        "hampel" => greeners::RobustNorm::Hampel(2.0, 4.0, 8.0),
                        "ols" | "leastsq" => greeners::RobustNorm::LeastSquares,
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "norm='{other}' unknown — use: huber, tukey, andrews, hampel, ols"
                            )))
                        }
                    },
                    _ => return Err(HayashiError::Type("norm= must be string".into())),
                };
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let result =
                    greeners::RLM::fit_with_names(&y_vec, &x_mat, &norm, cov, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RlmResult(Rc::new(result)))
            }

            // ── GEE (Generalized Estimating Equations) ────────────────────────
            // gee(y ~ x1 + x2, df, id=cluster_col, family=gaussian, corr=exchangeable)
            // family: gaussian (padrão), binomial, poisson
            // corr:   independence (padrão), exchangeable, ar1, unstructured
            "gee" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "gee() requer opção id=coluna_grupo".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("id= must be string".into())),
                };
                let family_str = match opt_map.get("family") {
                    None => "gaussian",
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => "gaussian",
                        "binomial" | "logit" => "binomial",
                        "poisson" => "poisson",
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "family='{other}' unknown — use: gaussian, binomial, poisson"
                            )))
                        }
                    },
                    _ => return Err(HayashiError::Type("family= must be string".into())),
                };
                let corr_str = match opt_map.get("corr") {
                    None => "independence",
                    Some(Value::Str(s)) => s.as_str(),
                    _ => "independence",
                };
                let corr = match corr_str {
                    "independence" | "ind" => greeners::CorrStructure::Independence,
                    "exchangeable" | "exch" => greeners::CorrStructure::Exchangeable,
                    "ar1" | "ar(1)"        => greeners::CorrStructure::AR1,
                    "unstructured" | "uns" => greeners::CorrStructure::Unstructured,
                    other => return Err(HayashiError::Runtime(
                        format!("corr='{other}' unknown — use: independence, exchangeable, ar1, unstructured")
                    )),
                };
                let (family, link) = match family_str {
                    "binomial" => (greeners::Family::Binomial, greeners::Link::Logit),
                    "poisson" => (greeners::Family::Poisson, greeners::Link::Log),
                    _ => (greeners::Family::Gaussian, greeners::Link::Identity),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // converter coluna de id para índices de grupo (usize)
                let id_vals = Self::get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut next_id = 0usize;
                let groups: ndarray::Array1<usize> = id_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *id_map.entry(key).or_insert_with(|| {
                            let id = next_id;
                            next_id += 1;
                            id
                        })
                    })
                    .collect();
                let result = greeners::GEE::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    &family,
                    &link,
                    &corr,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::GeeResult(Rc::new(result)))
            }

            // ── WLS (Weighted Least Squares) ──────────────────────────────────
            // wls(y ~ x1 + x2, df, weights="w_col", cov=HC3)
            "wls" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let w_name = match opt_map.get("weights") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "wls() requer opção weights=\"coluna_pesos\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("weights= must be string".into())),
                };
                let weights = Self::get_col_f64(&df, &w_name)?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::WLS::fit_with_names(&y, &x, &weights, cov, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let fitted = x.dot(&result.params);
                let residuals = &y - &fitted;
                Ok(Value::OlsResult(OlsModel {
                    result: Rc::new(result),
                    residuals,
                    x,
                }))
            }

            // ── ZIP / ZINB (Zero-Inflated Count Models) ───────────────────────
            // zip(y ~ x1 + x2, df)
            // zip(y ~ x1 + x2, df, inflate=["x3", "x4"])
            // zinb(y ~ x1 + x2, df)
            "zip" | "zinb" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_count) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let count_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // inflate= opcional: lista de nomes de colunas para a equação de inflação
                // Se omitido, usa a mesma matriz X do modelo de contagem
                let (x_inflate_opt, inflate_names_opt): (
                    Option<ndarray::Array2<f64>>,
                    Option<Vec<String>>,
                ) = match opt_map.get("inflate") {
                    Some(Value::List(lst)) => {
                        let inames: Vec<String> = lst
                            .iter()
                            .map(|v| match v {
                                Value::Str(s) => Ok(s.clone()),
                                _ => Err(HayashiError::Type(
                                    "inflate= must be a list de strings".into(),
                                )),
                            })
                            .collect::<Result<_>>()?;
                        // intercept + colunas especificadas
                        let n = df.n_rows();
                        let k = inames.len() + 1;
                        let mut xi = ndarray::Array2::<f64>::ones((n, k));
                        for (j, name) in inames.iter().enumerate() {
                            xi.column_mut(j + 1).assign(&Self::get_col_f64(&df, name)?);
                        }
                        let mut full_names = vec!["_cons".to_string()];
                        full_names.extend(inames);
                        (Some(xi), Some(full_names))
                    }
                    None => (None, None),
                    _ => {
                        return Err(HayashiError::Type(
                            "inflate= must be a list de strings".into(),
                        ))
                    }
                };

                let use_negbin = func == "zinb";
                let result = if use_negbin {
                    greeners::ZINB::fit_with_names(
                        &y_vec,
                        &x_count,
                        x_inflate_opt.as_ref(),
                        Some(count_names),
                        inflate_names_opt,
                    )
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else {
                    greeners::ZIP::fit_with_names(
                        &y_vec,
                        &x_count,
                        x_inflate_opt.as_ref(),
                        Some(count_names),
                        inflate_names_opt,
                    )
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                };
                Ok(Value::ZeroInflatedResult(Rc::new(result)))
            }

            // ── MixedLM (Mixed Linear Models — efeitos mistos) ────────────────
            // mixed(y ~ x1 + x2, df, id="group")           # intercept aleatório
            // mixed(y ~ x1 + x2, df, id="group", re=["x1"]) # + slope aleatório
            "mixed" | "mixedlm" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // id= obrigatório: coluna de grupo
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "mixed() requer opção id=\"coluna_grupo\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("id= must be string".into())),
                };

                // re= opcional: lista de variáveis com efeito aleatório de slope
                // Se omitido, modelo de intercept aleatório apenas (re = [1])
                let re_vars: Vec<String> = match opt_map.get("re") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type("re= must be a list de strings".into())),
                        })
                        .collect::<Result<_>>()?,
                    None => vec![],
                    _ => return Err(HayashiError::Type("re= must be a list de strings".into())),
                };

                let (y_vec, x_fixed) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Converter id para índices de grupo
                let id_vals = Self::get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut next_id = 0usize;
                let groups: ndarray::Array1<usize> = id_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *id_map.entry(key).or_insert_with(|| {
                            let id = next_id;
                            next_id += 1;
                            id
                        })
                    })
                    .collect();

                // Construir x_random: intercept + slopes especificados
                let n = df.n_rows();
                let q = re_vars.len() + 1; // +1 para intercept aleatório
                let mut x_random = ndarray::Array2::<f64>::ones((n, q));
                for (j, name) in re_vars.iter().enumerate() {
                    x_random
                        .column_mut(j + 1)
                        .assign(&Self::get_col_f64(&df, name)?);
                }

                let result = greeners::MixedLM::fit_with_names(
                    &y_vec,
                    &x_fixed,
                    &groups,
                    &x_random,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::MixedResult(Rc::new(result)))
            }

            // ── testparm — Wald F-test conjunto (OLS/WLS) ────────────────────
            // testparm(model, ["x1", "x2"])
            // H0: β_x1 = β_x2 = 0 simultaneamente
            "testparm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "testparm(model, [\"x1\", \"x2\"]) requer modelo + lista de variáveis"
                            .into(),
                    ));
                }
                let model_val = self.eval_expr(&args[0])?;
                let tested: Vec<String> = match self.eval_expr(&args[1])? {
                    Value::List(lst) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type(
                                "testparm: lista deve conter strings".into(),
                            )),
                        })
                        .collect::<Result<_>>()?,
                    _ => {
                        return Err(HayashiError::Type(
                            "testparm: second argument must be lista de strings".into(),
                        ))
                    }
                };
                match &model_val {
                    Value::OlsResult(m) => {
                        let vnames = m.result.variable_names.as_deref().unwrap_or(&[]);
                        let indices: Vec<usize> = tested.iter().map(|v| {
                            vnames.iter().position(|n| n == v)
                                .ok_or_else(|| HayashiError::Runtime(
                                    format!("testparm: variável '{v}' not found no modelo")
                                ))
                        }).collect::<Result<_>>()?;
                        let (f_stat, p_val) = m.result.f_test(&indices, &m.x)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        let df1 = indices.len();
                        let df2 = m.result.df_resid;
                        println!("\n{:=^62}", " testparm — Teste F Conjunto ");
                        println!(" H0: {} = 0 (simultâneamente)", tested.join(" = "));
                        println!("{:-^62}", "");
                        println!(" F({df1}, {df2})  =  {f_stat:.4}");
                        println!(" Prob > F      =  {p_val:.4}");
                        if p_val < 0.01       { println!(" Resultado: rejeita H0 a 1%"); }
                        else if p_val < 0.05  { println!(" Resultado: rejeita H0 a 5%"); }
                        else if p_val < 0.10  { println!(" Resultado: rejeita H0 a 10%"); }
                        else                  { println!(" Resultado: não rejeita H0 a 10%"); }
                        println!("{:=^62}", "");
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Runtime(
                        "testparm: suporte atual apenas para OLS/WLS — outros modelos usam chi2; implemente via wald_test()".into()
                    )),
                }
            }

            // ── GLSAR — GLS com erros AR(p) (Cochrane-Orcutt/Prais-Winsten) ─
            // glsar(y ~ x1 + x2, df, ar=1, iter=50)
            "glsar" | "prais" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ar_order = match opt_map.get("ar") {
                    Some(Value::Int(n)) => *n as usize,
                    None => 1,
                    _ => return Err(HayashiError::Type("ar= must be integer".into())),
                };
                let max_iter = match opt_map.get("iter") {
                    Some(Value::Int(n)) => *n as usize,
                    None => 50,
                    _ => return Err(HayashiError::Type("iter= must be integer".into())),
                };
                let result = greeners::GLSAR::fit_with_names(
                    &y_vec,
                    &x_mat,
                    ar_order,
                    max_iter,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::GlsarResult(Rc::new(result)))
            }

            // ── anova — ANOVA one-way ─────────────────────────────────────────
            // anova(df, outcome, by=group_col)
            "anova" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("anova(df, outcome, by=grupo)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "primeiro argumento deve ser DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let outcome_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be outcome variable name".into(),
                        ))
                    }
                };
                let outcome = Self::get_col_f64(&df, &outcome_name)?;
                let by_col = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "anova() requer by=\"coluna_grupo\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("by= must be string".into())),
                };
                let group_vals = Self::get_col_f64(&df, &by_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut next_g = 0usize;
                let groups: ndarray::Array1<usize> = group_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| {
                            let g = next_g;
                            next_g += 1;
                            g
                        })
                    })
                    .collect();
                let result = greeners::Stats::anova_oneway(&outcome, &groups)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("{result}");
                Ok(Value::Nil)
            }

            // ── Beta Regression ───────────────────────────────────────────────
            // betareg(y ~ x1 + x2, df)               # link=logit (padrão)
            // betareg(y ~ x1 + x2, df, link=probit)  # link alternativo
            // betareg(y ~ x1 + x2, df, link=cloglog)
            // Requer y ∈ (0,1) estritamente (proporções, probabilidades)
            "betareg" | "beta" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let link = match opt_map.get("link") {
                    None => greeners::BetaLink::Logit,
                    Some(Value::Str(s)) => match s.as_str() {
                        "logit" => greeners::BetaLink::Logit,
                        "probit" => greeners::BetaLink::Probit,
                        "cloglog" => greeners::BetaLink::CLogLog,
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "betareg: link='{other}' unknown — use: logit, probit, cloglog"
                            )))
                        }
                    },
                    _ => greeners::BetaLink::Logit,
                };
                let result =
                    greeners::BetaModel::fit_with_names(&y_vec, &x_mat, &link, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::BetaResult(Rc::new(result)))
            }

            // glm — Modelos Lineares Generalizados (IRLS via Greeners)
            // glm(y ~ x1 + x2, df, family=poisson, link=log, cov=robust)
            // Famílias: gaussian, binomial, poisson, gamma, inverse_gaussian, negbin, tweedie
            // Links: identity, log, logit, probit, inverse, cloglog
            // Se link omitido usa link canônico da família
            "glm" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;

                let alpha_val = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.0,
                };
                let power_val = match opt_map.get("power") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.5,
                };

                let family = match opt_map.get("family") {
                    None | Some(Value::Str(_)) if opt_map.get("family").is_none() => {
                        greeners::Family::Gaussian
                    }
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => greeners::Family::Gaussian,
                        "binomial" | "logistic" => greeners::Family::Binomial,
                        "poisson"  => greeners::Family::Poisson,
                        "gamma"    => greeners::Family::Gamma,
                        "inverse_gaussian" | "inversegaussian" => greeners::Family::InverseGaussian,
                        "negbin" | "negative_binomial" => greeners::Family::NegativeBinomial(alpha_val),
                        "tweedie" => greeners::Family::Tweedie(power_val),
                        other => return Err(HayashiError::Runtime(
                            format!("glm: family='{other}' unknown — use: gaussian, binomial, poisson, gamma, inverse_gaussian, negbin, tweedie")
                        )),
                    },
                    _ => greeners::Family::Gaussian,
                };

                let result = match opt_map.get("link") {
                    None => {
                        greeners::GLM::fit_with_names(&y_vec, &x_mat, family, cov, Some(var_names))
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    }
                    Some(Value::Str(s)) => {
                        let link = match s.as_str() {
                            "identity"  => greeners::Link::Identity,
                            "log"       => greeners::Link::Log,
                            "logit"     => greeners::Link::Logit,
                            "probit"    => greeners::Link::Probit,
                            "inverse"   => greeners::Link::InversePower,
                            "cloglog"   => greeners::Link::CLogLog,
                            other => return Err(HayashiError::Runtime(
                                format!("glm: link='{other}' unknown — use: identity, log, logit, probit, inverse, cloglog")
                            )),
                        };
                        // fit_with_link não aceita var_names; setar após
                        let mut r = greeners::GLM::fit_with_link(&y_vec, &x_mat, family, link, cov)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        r.variable_names = Some(var_names);
                        r
                    }
                    _ => {
                        greeners::GLM::fit_with_names(&y_vec, &x_mat, family, cov, Some(var_names))
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    }
                };
                Ok(Value::GlmResult(Rc::new(result)))
            }

            // influence — Diagnósticos de influência para OLS
            // influence(model, df)
            // Calcula DFBetas, DFFITS, leverage, resíduos studentizados
            // Imprime sumário e observações influentes
            "influence" => {
                if args.len() < 1 {
                    return Err(HayashiError::Runtime("influence(model, df)".into()));
                }
                let model_val = self.eval_expr(&args[0])?;
                match &model_val {
                    Value::OlsResult(m) => {
                        let mse = m.result.sigma * m.result.sigma;
                        let result = greeners::Influence::compute(&m.residuals, &m.x, mse)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!("{result}");
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Runtime(
                        "influence(): só suportado para modelos OLS/WLS — use: influence(m_ols, df)".into()
                    )),
                }
            }

            // lowess — Suavização não-paramétrica LOWESS
            // lowess(df, y, x, frac=0.67, it=3)
            // frac: fração dos dados usada em cada ajuste local (0 < frac ≤ 1)
            // it: iterações de robustificação (0 = sem robustificação)
            "lowess" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "lowess(df, y_var, x_var, frac=0.67, it=3)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "lowess: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let y_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "lowess: second argument must be nome de coluna y".into(),
                        ))
                    }
                };
                let x_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "lowess: third argument must be nome de coluna x".into(),
                        ))
                    }
                };
                let y_vec = ndarray::Array1::from(Self::get_col_f64(&df, &y_name)?);
                let x_vec = ndarray::Array1::from(Self::get_col_f64(&df, &x_name)?);
                let frac = match opt_map.get("frac") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    None => 0.6667,
                    _ => 0.6667,
                };
                let it = match opt_map.get("it") {
                    Some(Value::Int(v)) => *v as usize,
                    None => 3,
                    _ => 3,
                };
                let result = greeners::Lowess::fit(&y_vec, &x_vec, frac, it)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("{result}");
                Ok(Value::LowessResult(Rc::new(result)))
            }

            // kde — Estimativa de densidade por kernel (univariada)
            // kde(df, var, bw=auto, kernel=gaussian)
            // Imprime: n, bandwidth, suporte [min, max]
            "kde" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "kde(df, var, bw=auto, kernel=gaussian)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kde: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kde: second argument must be nome de coluna".into(),
                        ))
                    }
                };
                let data = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let bw_opt = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let kernel = match opt_map.get("kernel") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => greeners::Kernel::Gaussian,
                        "epanechnikov" => greeners::Kernel::Epanechnikov,
                        "triangular"   => greeners::Kernel::Triangular,
                        "uniform"      => greeners::Kernel::Uniform,
                        other => return Err(HayashiError::Runtime(
                            format!("kde: kernel='{other}' unknown — use: gaussian, epanechnikov, triangular, uniform")
                        )),
                    },
                    _ => greeners::Kernel::Gaussian,
                };
                let result = greeners::KDEUnivariate::fit(&data, bw_opt, kernel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let support_min = result.support.iter().cloned().fold(f64::INFINITY, f64::min);
                let support_max = result
                    .support
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let peak_idx = result
                    .density
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let peak_x = result.support[peak_idx];
                let peak_d = result.density[peak_idx];
                println!("\n{:=^50}", " KDE ");
                println!("{:<20} {:>10}", "Variável:", var_name);
                println!("{:<20} {:>10}", "Observações:", result.n_obs);
                println!("{:<20} {:>10.6}", "Bandwidth:", result.bandwidth);
                println!("{:<20} {:>10.4}", "Suporte min:", support_min);
                println!("{:<20} {:>10.4}", "Suporte max:", support_max);
                println!(
                    "{:<20} {:>10.4} @ x = {:.4}",
                    "Pico (densidade):", peak_d, peak_x
                );
                println!("{:=^50}", "");
                Ok(Value::Nil)
            }

            // pca — Análise de Componentes Principais
            // pca(df, x1, x2, x3, n=2)
            // n=: número de componentes (padrão: min(vars, obs-1))
            // Baseado na decomposição de autovalores da matriz de correlação
            // Variáveis são padronizadas automaticamente (equivalente a cor PCA)
            "pca" | "princomp" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "pca(df, x1, x2, x3, ..., n=k)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "pca: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let var_names = self.resolve_var_list(&args[1..], &df)?;
                let n = df.n_rows();
                let k = var_names.len();
                let n_components = match opt_map.get("n") {
                    Some(Value::Int(v)) => (*v as usize).min(k).min(n - 1),
                    Some(Value::Float(v)) => (*v as usize).min(k).min(n - 1),
                    _ => k.min(n - 1),
                };
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }
                let result = greeners::PCA::fit(&data, n_components)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PcaResult(PcaModel {
                    result: Rc::new(result),
                    var_names,
                }))
            }

            // factor — Análise Fatorial (eixo principal)
            // factor(df, x1, x2, x3, n=2, rotation=varimax)
            // rotation=: none (padrão), varimax
            // Diferença de PCA: PCA maximiza variância explicada;
            //   FA estima fatores latentes com estrutura de covariância específica
            "factor" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "factor(df, x1, x2, x3, ..., n=k, rotation=none|varimax)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "factor: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let var_names = self.resolve_var_list(&args[1..], &df)?;
                let n = df.n_rows();
                let k = var_names.len();
                let n_factors = match opt_map.get("n") {
                    Some(Value::Int(v)) => (*v as usize).min(k),
                    Some(Value::Float(v)) => (*v as usize).min(k),
                    _ => k.min(2),
                };
                let rotation = match opt_map.get("rotation") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "varimax" => greeners::Rotation::Varimax,
                        "none" => greeners::Rotation::None,
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "factor: rotation='{other}' unknown — use: none, varimax"
                            )))
                        }
                    },
                    _ => greeners::Rotation::None,
                };
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }
                let result = greeners::FactorAnalysis::fit(&data, n_factors, rotation)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::FactorResult(FactorModel {
                    result: Rc::new(result),
                    var_names,
                }))
            }

            // manova — Análise de Variância Multivariada (one-way)
            // manova(df, y1, y2, ..., by="group")
            // Testa H0: vetores de médias iguais entre grupos
            // Estatísticas: Wilks' Λ, Pillai's trace, Hotelling-Lawley, Roy's root
            "manova" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "manova(df, y1, y2, ..., by=\"group_col\")".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "manova: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let group_col = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "manova requer by=\"coluna_grupo\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("manova: by= must be string".into())),
                };
                let outcome_names = self.resolve_var_list(&args[1..], &df)?;
                let n = df.n_rows();
                let q = outcome_names.len();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, q));
                for (j, vname) in outcome_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        y_mat[[i, j]] = v;
                    }
                }
                let group_vals = Self::get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: ndarray::Array1<usize> = ndarray::Array1::from(
                    group_vals
                        .iter()
                        .map(|&v| {
                            let key = v as i64;
                            *gmap.entry(key).or_insert_with(|| {
                                let g = gnext;
                                gnext += 1;
                                g
                            })
                        })
                        .collect::<Vec<_>>(),
                );
                let result = greeners::MANOVA::fit(&y_mat, &groups)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("{result}");
                Ok(Value::Nil)
            }

            // bootse — Bootstrap standard errors para modelos OLS
            // bootse(model, n=1000)
            // Reamostral pares (y, X) com reposição para estimar distribuição amostral
            // Compara SE originais com bootstrap SE e IC percentil 95%
            // ── bootstrap genérico ────────────────────────────────────────────
            // bootstrap(estimator, formula, df, n=1000, alpha=0.05)
            // Reamostra linhas do DataFrame com reposição e re-estima.
            // Funciona com qualquer estimador: ols, logit, probit, iv, poisson, etc.
            // bootse(model, n=1000) mantido como alias para OLS pairs bootstrap.
            "bootstrap" | "boot" => {
                // ── Forma 1: bootstrap(estimator, formula, df, n=...) — genérico
                // ── Forma 2: bootse(model, n=...) — OLS pairs (legado, args[0] é Value)
                let n_boot = match opt_map.get("n") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => match opt_map.get("reps") {
                        Some(Value::Int(v)) => *v as usize,
                        Some(Value::Float(v)) => *v as usize,
                        _ => 1000,
                    },
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    _ => 0.05,
                };

                if args.len() >= 3 {
                    // ── Forma genérica: bootstrap(estimator, formula, df, ...)
                    let estimator_name = match &args[0] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => return Err(HayashiError::Type(
                            "bootstrap: first argument must be nome do estimador (ols, logit, ...)"
                                .into(),
                        )),
                    };
                    let formula_expr = args[1].clone();
                    let df_name = match &args[2] {
                        Expr::Var(n) => n.clone(),
                        _ => {
                            return Err(HayashiError::Type(
                                "bootstrap: third argument must be nome do DataFrame".into(),
                            ))
                        }
                    };
                    let df = match self.env.get(&df_name) {
                        Some(Value::DataFrame(d)) => d.clone(),
                        _ => {
                            return Err(HayashiError::Runtime(format!(
                                "'{df_name}' is not a DataFrame"
                            )))
                        }
                    };

                    // estimar no sample completo para referência
                    let extra_opts: Vec<Opt> = opts
                        .iter()
                        .filter(|o| !matches!(o.name.as_str(), "n" | "reps" | "alpha"))
                        .cloned()
                        .collect();
                    let full_result = self.eval_call(
                        &estimator_name,
                        &[formula_expr.clone(), Expr::Var(df_name.clone())],
                        &extra_opts,
                    )?;
                    let full_params = Self::extract_params(&full_result).ok_or_else(|| {
                        HayashiError::Runtime(
                            "bootstrap: modelo not supportado (sem params extraíveis)".into(),
                        )
                    })?;
                    let full_se = Self::extract_se(&full_result).unwrap_or_default();
                    let var_names = Self::extract_var_names(&full_result);
                    let k = full_params.len();

                    // bootstrap loop
                    use rand::seq::SliceRandom;
                    let mut rng = self.get_rng();
                    let n = df.n_rows();
                    let indices: Vec<usize> = (0..n).collect();
                    let mut boot_coefs = ndarray::Array2::<f64>::zeros((n_boot, k));
                    let mut n_ok = 0usize;

                    for b in 0..n_boot {
                        let boot_idx: Vec<usize> =
                            (0..n).map(|_| *indices.choose(&mut rng).unwrap()).collect();
                        let boot_df = match df.iloc(Some(&boot_idx), None) {
                            Ok(d) => d,
                            Err(_) => continue,
                        };
                        self.env
                            .set("__boot_df__", Value::DataFrame(Rc::new(boot_df)))?;
                        match self.eval_call(
                            &estimator_name,
                            &[formula_expr.clone(), Expr::Var("__boot_df__".into())],
                            &extra_opts,
                        ) {
                            Ok(ref result) => {
                                if let Some(params) = Self::extract_params(result) {
                                    for j in 0..k.min(params.len()) {
                                        boot_coefs[[b, j]] = params[j];
                                    }
                                    n_ok += 1;
                                }
                            }
                            Err(_) => {}
                        }
                    }
                    self.env.remove("__boot_df__");

                    if n_ok < 10 {
                        return Err(HayashiError::Runtime(format!(
                            "bootstrap: apenas {n_ok}/{n_boot} replicações convergiram"
                        )));
                    }

                    // truncar para replicações bem-sucedidas
                    let boot_se = greeners::Bootstrap::bootstrap_se(&boot_coefs);
                    let (ci_lo, ci_hi) = greeners::Bootstrap::percentile_ci(&boot_coefs, alpha);

                    let thick = "═".repeat(76);
                    let thin = "─".repeat(76);
                    println!("\n{thick}");
                    println!(
                        "{:^76}",
                        format!(" Bootstrap SE — {} (n={n_ok}/{n_boot}) ", estimator_name)
                    );
                    println!("{thin}");
                    println!(
                        "{:<18} {:>10} {:>10} {:>10} {:>12} {:>12}",
                        "Variável", "β̂", "SE orig.", "SE boot", "IC inf", "IC sup"
                    );
                    println!("{thin}");
                    for i in 0..k {
                        let vname = var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
                        let orig_se = if i < full_se.len() {
                            full_se[i]
                        } else {
                            f64::NAN
                        };
                        println!(
                            "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>12.4}",
                            vname, full_params[i], orig_se, boot_se[i], ci_lo[i], ci_hi[i]
                        );
                    }
                    println!("{thick}");
                    return Ok(Value::Nil);
                }

                // ── Forma legado: bootse(model, n=...) — OLS pairs ──────────
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "bootstrap(estimator, formula, df, n=1000) ou bootse(model, n=1000)".into(),
                    ));
                }
                let model_val = self.eval_expr(&args[0])?;
                match &model_val {
                    Value::OlsResult(m) => {
                        let y_hat = m.x.dot(&m.result.params);
                        let y_vec = &y_hat + &m.residuals;
                        let boot_coefs = greeners::Bootstrap::pairs_bootstrap(&y_vec, &m.x, n_boot)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        let boot_se = greeners::Bootstrap::bootstrap_se(&boot_coefs);
                        let (ci_lo, ci_hi) = greeners::Bootstrap::percentile_ci(&boot_coefs, alpha);
                        let vnames = m.result.variable_names.as_deref().unwrap_or(&[]);
                        let k = m.result.params.len();
                        let thick = "═".repeat(76);
                        let thin  = "─".repeat(76);
                        println!("\n{thick}");
                        println!("{:^76}", format!(" Bootstrap SE (n={n_boot}, pairs) "));
                        println!("{thin}");
                        println!("{:<18} {:>10} {:>10} {:>10} {:>12} {:>12}",
                                 "Variável", "β̂", "SE orig.", "SE boot", "IC inf 95%", "IC sup 95%");
                        println!("{thin}");
                        for i in 0..k {
                            let vname = vnames.get(i).map(|s| s.as_str()).unwrap_or("?");
                            println!("{:<18} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>12.4}",
                                     vname,
                                     m.result.params[i],
                                     m.result.std_errors[i],
                                     boot_se[i],
                                     ci_lo[i],
                                     ci_hi[i]);
                        }
                        println!("{thick}");
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Runtime(
                        "bootse(model) suporta OLS. Para outros: bootstrap(estimator, formula, df, n=1000)".into()
                    )),
                }
            }

            "bootse" => {
                return self.eval_call("bootstrap", args, opts);
            }

            // markov — Markov-Switching AR (Hamilton 1989)
            // markov(df, y, k=2, p=1)
            // k=: número de regimes (padrão: 2)
            // p=: ordem AR dentro de cada regime (padrão: 1)
            // Algoritmo: EM via filtro de Hamilton (forward-backward)
            // Parâmetros por regime: intercept + AR coefficients + variance
            "markov" | "msar" | "markovswitching" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("markov(df, y_var, k=2, p=1)".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "markov: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let y_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "markov: second argument must be variable name".into(),
                        ))
                    }
                };
                let y_vec = ndarray::Array1::from(Self::get_col_f64(&df, &y_name)?);
                let k = match opt_map.get("k") {
                    Some(Value::Int(v)) => (*v as usize).max(2),
                    Some(Value::Float(v)) => (*v as usize).max(2),
                    _ => 2,
                };
                let p = match opt_map.get("p") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let result = greeners::MarkovSwitching::fit(&y_vec, k, p)
                    .map_err(|e| self.rt_err(format!("markov: {e}")))?;
                Ok(Value::MarkovResult(Rc::new(result)))
            }

            // clogit — Conditional Logit (Chamberlain 1980, FE logit)
            // clogit(y ~ x1 + x2, df, group="id_col")
            // Condiciona na soma de y por grupo → elimina efeitos fixos individuais
            // Grupos sem variação em y são automaticamente excluídos
            // Sem intercepto — absorvido pelo FE
            "clogit" | "xtlogit_fe" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let group_col = match opt_map.get("group") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "clogit requer group=\"coluna_id\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("clogit: group= must be string".into())),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let group_vals = Self::get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: Vec<usize> = group_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| {
                            let g = gnext;
                            gnext += 1;
                            g
                        })
                    })
                    .collect();
                let result = greeners::ConditionalLogit::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    Some(var_names),
                )
                .map_err(|e| self.rt_err(format!("clogit: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // cpoisson — Conditional Poisson (FE Poisson)
            // cpoisson(y ~ x1 + x2, df, group="id_col")
            // Equivalente a FE Poisson; consistente sob heterogeidade não observada
            // Só requer que E[y|x,c] = exp(c + xβ) — não requer y ~ Poisson (PPML)
            "cpoisson" | "xtpoisson_fe" | "ppml" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let group_col = match opt_map.get("group") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "cpoisson requer group=\"coluna_id\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("cpoisson: group= must be string".into())),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let group_vals = Self::get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: Vec<usize> = group_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| {
                            let g = gnext;
                            gnext += 1;
                            g
                        })
                    })
                    .collect();
                let result = greeners::ConditionalPoisson::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    Some(var_names),
                )
                .map_err(|e| self.rt_err(format!("cpoisson: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // cmnlogit — Conditional Multinomial Logit
            // cmnlogit(y ~ x1 + x2, df, group="id_col", alts=3)
            "cmnlogit" | "cmlogit" | "conditional_mlogit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let group_col = match opt_map.get("group") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "cmnlogit requires group=\"id_col\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("cmnlogit: group= must be string".into())),
                };
                let n_alts = match opt_map.get("alts") {
                    Some(Value::Int(n)) => *n as usize,
                    Some(Value::Float(f)) => *f as usize,
                    None => {
                        return Err(HayashiError::Runtime(
                            "cmnlogit requires alts=N (number of alternatives)".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("cmnlogit: alts= must be integer".into())),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let group_vals = Self::get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: Vec<usize> = group_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| {
                            let g = gnext;
                            gnext += 1;
                            g
                        })
                    })
                    .collect();
                let result = greeners::ConditionalMNLogit::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    n_alts,
                    Some(var_names),
                )
                .map_err(|e| self.rt_err(format!("cmnlogit: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // gqtest — Goldfeld-Quandt test (heteroskedasticidade)
            // gqtest(model, split=0.2)
            // H0: homocedasticidade
            // Divide os resíduos em dois grupos (descartando `split` do meio)
            // e testa se as variâncias diferem via F
            // split=: fração do meio a descartar (padrão: 0.2)
            // Mais potente que White quando heterocedasticidade é monotônica
            "gqtest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("gqtest(model, split=0.2)".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "gqtest(): only supports OLS models".into(),
                        ))
                    }
                };
                let split = match opt_map.get("split") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.2,
                };
                let (f, p, df1, df2) =
                    greeners::SpecificationTests::goldfeld_quandt_test(&ols.residuals, split)
                        .map_err(|e| self.rt_err(format!("gqtest: {e}")))?;
                let sig = |p: f64| {
                    if p < 0.01 {
                        "***"
                    } else if p < 0.05 {
                        "**"
                    } else if p < 0.10 {
                        "*"
                    } else {
                        ""
                    }
                };
                let sep = "─".repeat(56);
                println!("\nGoldfeld-Quandt Test  —  split = {split:.2}");
                println!("{sep}");
                println!("H₀: homocedasticidade (σ²₁ = σ²₂)");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10.4} {:>10.4} {:>4}",
                    format!("F ~ F({df1},{df2})"),
                    f,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // bphet — Breusch-Pagan test (heteroskedasticidade, OLS)
            // bphet(model)
            // H0: homocedasticidade — LM = n·R² da regressão auxiliar de u² em X
            // Diferente de bptest() que é o LM de efeitos aleatórios (painel)
            "bphet" | "hettest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("bphet(model)".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "bphet(): only supports OLS models".into(),
                        ))
                    }
                };
                let (lm, p) = greeners::Diagnostics::breusch_pagan(&ols.residuals, &ols.x)
                    .map_err(|e| self.rt_err(format!("bphet: {e}")))?;
                let k = ols.x.ncols().saturating_sub(1);
                let sig = |p: f64| {
                    if p < 0.01 {
                        "***"
                    } else if p < 0.05 {
                        "**"
                    } else if p < 0.10 {
                        "*"
                    } else {
                        ""
                    }
                };
                let sep = "─".repeat(56);
                println!("\nBreusch-Pagan Heteroskedasticity Test");
                println!("{sep}");
                println!("H₀: homocedasticidade (variância constante)");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10.4} {:>10.4} {:>4}",
                    format!("LM ~ χ²({k})"),
                    lm,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // ── Testes de diagnóstico para dados em painel ────────────────────

            // bptest — Breusch-Pagan LM test (H0: pooled OLS adequado, σ²_u = 0)
            // bptest(df, y ~ x1 + x2, id="entity_col")
            "bptest" | "xttest0" | "xtbp" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "bptest(df, y ~ x1+x2, id=\"entity_col\")".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let formula_ast = self.resolve_formula(&args[1])?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(id, _)| id.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            self.rt_err(format!("bptest requer id= ou xtset({df_name}, id, time)"))
                        })?,
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // OLS pooled para obter resíduos
                let ols_pooled = OLS::from_formula(&g_formula, &df, CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let resids = &y_vec - &x_mat.dot(&ols_pooled.params);
                // Converter id para usize
                let id_vals = Self::get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut next_id = 0usize;
                let entity_ids: Vec<usize> = id_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *id_map.entry(key).or_insert_with(|| {
                            let id = next_id;
                            next_id += 1;
                            id
                        })
                    })
                    .collect();
                let (lm, p) = greeners::PanelDiagnostics::breusch_pagan_lm(&resids, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(e))?;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!("\n{:=^62}", " Breusch-Pagan LM Test (RE) ");
                println!(" H0: σ²_u = 0 — pooled OLS adequado");
                println!("{:-^62}", "");
                println!(" LM = {lm:.4}    p-valor = {p:.4}  {sig}");
                if p < 0.05 {
                    println!(" Conclusão: rejeita H0 → usar RE ou FE");
                } else {
                    println!(" Conclusão: não rejeita H0 → pooled OLS adequado");
                }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // wooldridge — Teste de Wooldridge para correlação serial em painel
            // H0: sem correlação serial de 1ª ordem nos erros idiossincráticos
            // wooldridge(df, y ~ x1+x2, id="entity", time="time")
            "wooldridge" | "xtserial" | "wooldridge_serial" | "xtwooldridge" => {
                if args.len() < 2 {
                    return Err(self.rt_err("wooldridge(df, y~x, id=\"entity\", time=\"time\")"));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let formula_ast = self.resolve_formula(&args[1])?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(id, _)| id.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            self.rt_err(format!(
                                "wooldridge requer id= ou xtset({df_name}, id, time)"
                            ))
                        })?,
                };
                let time_col = match opt_map.get("time") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(_, t)| t.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            self.rt_err(format!(
                                "wooldridge requer time= ou xtset({df_name}, id, time)"
                            ))
                        })?,
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let id_vals: Vec<i64> = Self::get_col_f64(&df, &id_col)?
                    .iter()
                    .map(|&v| v as i64)
                    .collect();
                let time_vals: Vec<f64> = Self::get_col_f64(&df, &time_col)?.to_vec();
                let (rho, t_stat, p, n_pairs) = greeners::PanelDiagnostics::wooldridge_serial(
                    &y_vec, &x_mat, &id_vals, &time_vals,
                )
                .map_err(|e| HayashiError::Runtime(e))?;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!(
                    "\n{:=^62}",
                    " Wooldridge Test — Correlação Serial em Painel "
                );
                println!(" H0: ρ = -0.5 (sem correlação serial)");
                println!("{:-^62}", "");
                println!(" ρ̂ = {rho:.4}    t = {t_stat:.4}    p = {p:.4}  {sig}");
                println!(" Pares de resíduos: {n_pairs}");
                if p < 0.05 {
                    println!(
                        " Conclusão: rejeita H0 → correlação serial presente → usar SE robustos"
                    );
                } else {
                    println!(" Conclusão: não rejeita H0 → sem evidência de correlação serial");
                }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // pesaran — Pesaran CD test (cross-sectional dependence)
            // H0: sem dependência cross-sectional
            // pesaran(df, y ~ x1+x2, id="entity", time="time")
            "pesaran" | "xtcd" => {
                if args.len() < 2 {
                    return Err(self.rt_err("pesaran(df, y~x, id=\"entity\", time=\"time\")"));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let formula_ast = self.resolve_formula(&args[1])?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(id, _)| id.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            self.rt_err(format!("pesaran requer id= ou xtset({df_name}, id, time)"))
                        })?,
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols_pooled = OLS::from_formula(&g_formula, &df, CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let resids = &y_vec - &x_mat.dot(&ols_pooled.params);
                let id_vals = Self::get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut next_id = 0usize;
                let entity_ids: Vec<usize> = id_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *id_map.entry(key).or_insert_with(|| {
                            let id = next_id;
                            next_id += 1;
                            id
                        })
                    })
                    .collect();
                let (cd, p) = greeners::PanelDiagnostics::pesaran_cd(&resids, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(e))?;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!(
                    "\n{:=^62}",
                    " Pesaran CD Test — Dependência Cross-Sectional "
                );
                println!(" H0: sem dependência cross-sectional");
                println!("{:-^62}", "");
                println!(" CD = {cd:.4}    p-valor = {p:.4}  {sig}");
                if p < 0.05 {
                    println!(" Conclusão: rejeita H0 → dependência CS presente → usar SE robustos por cluster");
                } else {
                    println!(" Conclusão: não rejeita H0 → sem dependência CS detectada");
                }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // mundlak — Teste de Mundlak (adequação de RE vs FE)
            // H0: médias de grupo não correlacionadas com regressores (RE ok)
            // mundlak(df, y ~ x1+x2, id="entity")
            "mundlak" => {
                if args.len() < 2 {
                    return Err(self.rt_err("mundlak(df, y~x, id=\"entity\")"));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let formula_ast = self.resolve_formula(&args[1])?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(id, _)| id.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            self.rt_err(format!("mundlak requer id= ou xtset({df_name}, id, time)"))
                        })?,
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let id_vals: Vec<i64> = Self::get_col_f64(&df, &id_col)?
                    .iter()
                    .map(|&v| v as i64)
                    .collect();
                let (f_stat, p, k, gamma, gamma_se) =
                    greeners::PanelDiagnostics::mundlak(&y_vec, &x_mat, &id_vals)
                        .map_err(|e| HayashiError::Runtime(e))?;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!(
                    "\n{:=^62}",
                    " Mundlak Test — RE vs FE (correlação das médias) "
                );
                println!(" H0: γ = 0 (médias de grupo não correlacionadas com X → RE ok)");
                println!("{:-^62}", "");
                println!(" F({k}, .) = {f_stat:.4}    p = {p:.4}  {sig}");
                println!("{:-^62}", "");
                // Nomes das variáveis variantes no tempo (não-constantes)
                let slope_names: Vec<&str> = var_names
                    .iter()
                    .filter(|n| n.as_str() != "_cons" && n.as_str() != "const")
                    .map(|s| s.as_str())
                    .collect();
                println!(" {:<20} {:>10}  {:>10}", "Variável (γ̂)", "Coef", "Std Err");
                for i in 0..k.min(gamma.len()) {
                    let nm = slope_names.get(i).copied().unwrap_or("?");
                    println!(
                        " {:<20} {:>10.4}  {:>10.4}",
                        nm,
                        gamma[i],
                        gamma_se.get(i).copied().unwrap_or(f64::NAN)
                    );
                }
                if p < 0.05 {
                    println!("\n Conclusão: rejeita H0 → RE é inconsistente → usar FE ou Hausman");
                } else {
                    println!("\n Conclusão: não rejeita H0 → RE adequado");
                }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // abtest — Arellano-Bond m1/m2 test (validação de instrumentos GMM)
            // abtest(df, y ~ x1+x2, id="entity", time="time")
            // m1 deve rejeitar H0 (FD induz AR(1) por construção)
            // m2 NÃO deve rejeitar H0 (valida instrumentos y_{i,t-2})
            "abtest" | "abar" | "abond" | "xtabond_test" | "arellano_bond" => {
                if args.len() < 2 {
                    return Err(self.rt_err("abtest(df, y~x, id=\"entity\", time=\"time\")"));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let formula_ast = self.resolve_formula(&args[1])?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(id, _)| id.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            self.rt_err(format!("abtest requer id= ou xtset({df_name}, id, time)"))
                        })?,
                };
                let time_col = match opt_map.get("time") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(_, t)| t.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            self.rt_err(format!(
                                "abtest requer time= ou xtset({df_name}, id, time)"
                            ))
                        })?,
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let id_vals: Vec<i64> = Self::get_col_f64(&df, &id_col)?
                    .iter()
                    .map(|&v| v as i64)
                    .collect();
                let time_vals: Vec<f64> = Self::get_col_f64(&df, &time_col)?.to_vec();
                let (m1, p1, m2, p2) = greeners::PanelDiagnostics::arellano_bond_test(
                    &y_vec, &x_mat, &id_vals, &time_vals,
                )
                .map_err(|e| HayashiError::Runtime(e))?;
                let sig = |p: f64| {
                    if p < 0.01 {
                        "***"
                    } else if p < 0.05 {
                        "**"
                    } else if p < 0.10 {
                        "*"
                    } else {
                        ""
                    }
                };
                println!(
                    "\n{:=^62}",
                    " Arellano-Bond Test — Autocorrelação em 1ª Diferença "
                );
                println!(" m1 DEVE rejeitar H0 (AR(1) induzido por FD)");
                println!(" m2 NÃO deve rejeitar H0 (valida instrumentos y_{{t-2}})");
                println!("{:-^62}", "");
                println!(" m1 = {m1:.4}    p(m1) = {p1:.4}  {}", sig(p1));
                println!(" m2 = {m2:.4}    p(m2) = {p2:.4}  {}", sig(p2));
                println!("{:-^62}", "");
                if p1 >= 0.05 {
                    println!(" [!] m1 não rejeita H0 — modelo pode estar mal especificado");
                }
                if p2 < 0.05 {
                    println!(" [!] m2 rejeita H0 — instrumentos y_{{t-2}} podem ser inválidos");
                }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // ── SUR (Seemingly Unrelated Regressions) ─────────────────────────
            // sur(df, y1 ~ x1 + x2, y2 ~ x3 + x4, ...)
            // Estimador de Zellner (FGLS entre equações)
            // Cada equação pode ter regressores diferentes
            "sur" | "sureg" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "sur(df, y1~x1+x2, y2~x3+x4, ...) requer df + ao menos 2 fórmulas".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "primeiro argumento deve ser DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let mut equations: Vec<greeners::SurEquation> = Vec::new();
                let mut eq_var_names: Vec<Vec<String>> = Vec::new();

                for arg in &args[1..] {
                    let formula_ast = self.resolve_formula(arg)?;
                    let formula_str = Self::formula_to_string(&formula_ast);
                    let g_formula = GFormula::parse(&formula_str)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let (y, x) = df
                        .to_design_matrix(&g_formula)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let var_names = df
                        .formula_var_names(&g_formula)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    eq_var_names.push(var_names);
                    equations.push(greeners::SurEquation {
                        y,
                        x,
                        name: formula_ast.lhs.clone(),
                    });
                }
                let result = greeners::SUR::fit(&equations)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::SurResult(SurModel {
                    result: Rc::new(result),
                    eq_var_names,
                }))
            }

            // ── Rolling OLS (janela deslizante) ───────────────────────────────
            // rolling(y ~ x1 + x2, df, window=30)
            // Estima OLS para cada janela de tamanho `window`
            // Útil para: coeficientes time-varying, testes de estabilidade
            "rolling" | "rols" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let window = match opt_map.get("window") {
                    Some(Value::Int(n)) => *n as usize,
                    None => {
                        return Err(HayashiError::Runtime(
                            "rolling() requer window=N (ex: window=30)".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("window= must be integer".into())),
                };
                let result = greeners::RollingOLS::fit(&y_vec, &x_mat, window)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RollingResult(Rc::new(result)))
            }

            // ── Recursive OLS (Kalman, acumula observações) ───────────────────
            // recursive(y ~ x1 + x2, df)
            // Expande a janela de 1 em 1 — base para CUSUM e estabilidade
            "recursive" | "recols" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::RecursiveLS::fit(&y_vec, &x_mat)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RecursiveLSResult(Rc::new(result)))
            }

            // ── ic — tabela de critérios de informação (AIC/BIC) ──────────────
            // ic(m1, m2, m3, ...)
            // Compara modelos pelo AIC e BIC; ordena do menor (melhor) para maior
            "ic" | "fitstat" | "estat" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "ic() requer ao menos um modelo".into(),
                    ));
                }
                struct IcRow {
                    label: String,
                    ll: f64,
                    k: usize,
                    n: usize,
                    aic: f64,
                    bic: f64,
                }
                let mut rows: Vec<IcRow> = Vec::new();
                for arg in args {
                    let label = match arg {
                        Expr::Var(name) => name.clone(),
                        _ => "model".to_string(),
                    };
                    let val = self.eval_expr(arg)?;
                    let (ll, k, n) = match &val {
                        Value::OlsResult(m)      => (m.result.log_likelihood, m.result.params.len(), m.result.n_obs),
                        Value::BinaryResult(b)   => (b.result.log_likelihood, b.result.params.len(), b.x.nrows()),
                        Value::PoissonResult(r)  => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::NegBinResult(r)   => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::OrderedResult(r)  => (r.log_likelihood, r.params.len() + r.thresholds.len(), r.n_obs),
                        Value::TobitResult(r)    => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::MixedResult(r)    => (r.log_likelihood, r.fixed_effects.len(), r.n_obs),
                        Value::ZeroInflatedResult(r) => (r.log_likelihood, r.count_params.len() + r.inflate_params.len(), r.n_obs),
                        Value::RollingResult(_) | Value::RecursiveLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                format!("ic(): '{label}' não tem log-verossimilhança — use print() para diagnósticos")
                            ));
                        }
                        _ => return Err(HayashiError::Runtime(
                            format!("ic(): modelo '{label}' não tem log-verossimilhança disponível para ic() — use print()")
                        )),
                    };
                    let aic = -2.0 * ll + 2.0 * k as f64;
                    let bic = -2.0 * ll + (k as f64) * (n as f64).ln();
                    rows.push(IcRow {
                        label,
                        ll,
                        k,
                        n,
                        aic,
                        bic,
                    });
                }
                // Ordenar por AIC
                rows.sort_by(|a, b| {
                    a.aic
                        .partial_cmp(&b.aic)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let min_aic = rows.first().map(|r| r.aic).unwrap_or(0.0);
                let _min_bic = rows
                    .iter()
                    .map(|r| r.bic)
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                println!("\n{:=^80}", " Critérios de Informação ");
                println!(
                    "{:<20} {:>6} {:>6} {:>12} {:>12} {:>8} {:>8}",
                    "Modelo", "N", "k", "Log-Lik", "AIC", "ΔAIC", "BIC"
                );
                println!("{:-^80}", "");
                for row in &rows {
                    println!(
                        "{:<20} {:>6} {:>6} {:>12.4} {:>12.4} {:>8.4} {:>12.4}",
                        row.label,
                        row.n,
                        row.k,
                        row.ll,
                        row.aic,
                        row.aic - min_aic,
                        row.bic
                    );
                }
                if rows.len() > 1 {
                    println!("{:-^80}", "");
                    println!(
                        " Melhor AIC: {}   Melhor BIC: {}",
                        rows.iter()
                            .min_by(|a, b| a.aic.partial_cmp(&b.aic).unwrap())
                            .unwrap()
                            .label,
                        rows.iter()
                            .min_by(|a, b| a.bic.partial_cmp(&b.bic).unwrap())
                            .unwrap()
                            .label
                    );
                    // Pesos de Akaike
                    let delta_aics: Vec<f64> = rows.iter().map(|r| r.aic - min_aic).collect();
                    let rel: Vec<f64> = delta_aics.iter().map(|d| (-d / 2.0).exp()).collect();
                    let sum_rel: f64 = rel.iter().sum();
                    println!(
                        " Pesos Akaike: {}",
                        rows.iter()
                            .zip(rel.iter())
                            .map(|(r, w)| format!("{}={:.3}", r.label, w / sum_rel))
                            .collect::<Vec<_>>()
                            .join("  ")
                    );
                }
                println!("{:=^80}", "");
                Ok(Value::Nil)
            }

            // ── Fixed Effects ─────────────────────────────────────────────────
            "fe" => {
                let (formula_ast, df, _df_name, id_col) =
                    self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                // FE elimina o intercepto via within-transform; forçamos - 1
                // para evitar coluna de zeros pós-demeaning (singular matrix)
                let formula_no_const = if formula_str.contains("- 1") {
                    formula_str
                } else {
                    format!("{} - 1", formula_str)
                };
                let g_formula = GFormula::parse(&formula_no_const)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // tenta int; cai para float→int; cai para string
                let result = if let Ok(ids) = df.get_int(&id_col) {
                    let ids_vec: Vec<i64> = ids.to_vec();
                    FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else if let Ok(floats) = df.get(&id_col) {
                    let ids_vec: Vec<i64> = floats.iter().map(|&v| v as i64).collect();
                    FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else if let Ok(ids) = df.get_string(&id_col) {
                    let ids_vec: Vec<String> = ids.to_vec();
                    FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "column '{id_col}' not found or not usable as entity ID"
                    )));
                };

                Ok(Value::PanelResult(Rc::new(result)))
            }

            // ── Random Effects ────────────────────────────────────────────────
            "re" => {
                let (formula_ast, df, _df_name, id_col) =
                    self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // aceita coluna float de valores inteiros (ex: idcode lido como f64)
                let ids_owned: ndarray::Array1<i64>;
                let ids = match df.get_int(&id_col) {
                    Ok(arr) => arr,
                    Err(_) => {
                        let floats = df.get(id_col.as_str()).map_err(|_| {
                            HayashiError::Runtime(format!(
                                "column '{id_col}' must be integer for re()"
                            ))
                        })?;
                        ids_owned = floats.mapv(|v| v as i64);
                        &ids_owned
                    }
                };

                let result = RandomEffects::from_formula(&g_formula, &df, ids)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::ReResult(Rc::new(result)))
            }

            // ── F-test para Efeitos Fixos (FE vs pooled OLS) ─────────────────
            "ftest_fe" => {
                // ftest_fe(formula, df, id=col)
                // H₀: todos os efeitos individuais são zero (pooled OLS adequado)
                // H₁: efeitos individuais existem (use FE)
                let (formula_ast, df, _df_name, id_col) =
                    self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);

                // FE (within)
                let formula_no_const = if formula_str.contains("- 1") {
                    formula_str.clone()
                } else {
                    format!("{} - 1", formula_str)
                };
                let g_formula_fe = GFormula::parse(&formula_no_const)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids_fe: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ftest_fe: column '{id_col}' not found"
                    )));
                };

                let fe = FixedEffects::from_formula(&g_formula_fe, &df, &entity_ids_fe)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Pooled OLS (com intercepto)
                let g_formula_ols = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_pool, x_pool) = df
                    .to_design_matrix(&g_formula_ols)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols = greeners::OLS::fit(&y_pool, &x_pool, greeners::CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let ssr_pooled = ols.sigma.powi(2) * ols.df_resid as f64;
                let ssr_fe = fe.sigma.powi(2) * fe.df_resid as f64;
                let n = fe.n_obs;
                let n_entities = fe.n_entities;
                let k = fe.params.len();

                let (f_stat, p) = greeners::PanelDiagnostics::f_test_fixed_effects(
                    ssr_pooled, ssr_fe, n, n_entities, k,
                )
                .map_err(|e| HayashiError::Runtime(e))?;

                let df_num = n_entities - 1;
                let df_denom = n - n_entities - k;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos fixos individuais são significativos (use FE)"
                } else {
                    "Não rejeita H₀ → pooled OLS adequado (efeitos individuais não significativos)"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" F-test: Efeitos Fixos vs Pooled OLS\n");
                out.push_str(" H₀: todos os efeitos individuais são zero\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Soma dos Quadrados dos Resíduos\n");
                out.push_str(&format!("   SSR pooled = {:.6}\n", ssr_pooled));
                out.push_str(&format!("   SSR FE     = {:.6}\n", ssr_fe));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
                    df_num, df_denom, f_stat, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Pesaran CD: dependência cross-seccional ───────────────────────
            "pesaran_cd" | "cd_test" => {
                // pesaran_cd(formula, df, id=col)
                // H₀: resíduos independentes entre entidades (sem dependência cross-seccional)
                // H₁: dependência cross-seccional presente
                let (formula_ast, df, _df_name, id_col) =
                    self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // OLS pooled para resíduos
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols = greeners::OLS::fit(&y_vec, &x_mat, greeners::CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let residuals = ols.residuals(&y_vec, &x_mat);

                // IDs de entidade
                let entity_ids: Vec<usize> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.iter().map(|&v| v as usize).collect()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as usize).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "pesaran_cd: column '{id_col}' not found"
                    )));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        s.insert(id);
                    }
                    s.len()
                };
                let t_bar = residuals.len() as f64 / n_entities as f64;

                let (cd, p) = greeners::PanelDiagnostics::pesaran_cd(&residuals, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(e))?;

                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → dependência cross-seccional presente"
                } else {
                    "Não rejeita H₀ → sem evidência de dependência cross-seccional"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Pesaran CD Test (dependência cross-seccional)\n");
                out.push_str(" H₀: ρ_ij = 0 para todo i≠j  (resíduos independentes)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: N={} entidades   T̄≈{:.1}\n",
                    n_entities, t_bar
                ));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   CD ~ N(0,1) = {:.4}   p = {:.4}  {}\n",
                    cd, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Breusch-Pagan LM test (efeitos individuais em painel) ────────
            "bplm" => {
                // bplm(formula, df, id=col)
                // H₀: sem efeitos individuais (σ²_u = 0) — pooled OLS adequado
                // H₁: efeitos individuais existem — use FE ou RE
                let (formula_ast, df, _df_name, id_col) =
                    self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // OLS pooled para obter resíduos
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols = greeners::OLS::fit(&y_vec, &x_mat, greeners::CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let residuals = ols.residuals(&y_vec, &x_mat);

                // IDs de entidade → usize
                let entity_ids: Vec<usize> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.iter().map(|&v| v as usize).collect()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as usize).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "bplm: column '{id_col}' not found ou não usável como ID"
                    )));
                };

                let n = residuals.len();
                let n_entities = {
                    let mut ids_set = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        ids_set.insert(id);
                    }
                    ids_set.len()
                };
                let t_bar = n as f64 / n_entities as f64;

                let (lm, p) = greeners::PanelDiagnostics::breusch_pagan_lm(&residuals, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(e))?;

                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais presentes (use FE ou RE)"
                } else {
                    "Não rejeita H₀ → pooled OLS adequado (sem efeitos individuais)"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Breusch-Pagan LM Test (efeitos individuais)\n");
                out.push_str(" H₀: σ²_u = 0  (sem efeitos individuais)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Dados do Painel\n");
                out.push_str(&format!(
                    "   n = {}   N = {}   T̄ ≈ {:.1}\n",
                    n, n_entities, t_bar
                ));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   LM ~ χ²(1) = {:.4}   p = {:.4}  {}\n",
                    lm, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Chamberlain: correlação period-específica com efeitos individuais
            "chamberlain" => {
                // chamberlain(formula, df, id=col, time=col)
                // H₀: Π_s = 0 para todo s (RE consistente)
                // H₁: pelo menos um Π_s ≠ 0 (efeitos correlacionados com X — use FE)
                // Generalização do Mundlak: usa valores em TODOS os períodos, não só a média
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let time_col = self.get_time_col(&df_name, &opt_map)?;

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "chamberlain: coluna id '{id_col}' not found"
                    )));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "chamberlain: coluna time '{time_col}' not found"
                    )));
                };

                let (f_stat, p, k_active, df_denom, n_entities, t_count) =
                    greeners::PanelDiagnostics::chamberlain(
                        &y_vec,
                        &x_mat,
                        &entity_ids,
                        &time_vals,
                    )
                    .map_err(|e| HayashiError::Runtime(e))?;

                let n_obs = y_vec.len();
                let df1 = k_active;

                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais correlacionados com X (prefira FE)"
                } else {
                    "Não rejeita H₀ → RE consistente (sem correlação period-específica)"
                };

                let thick = "═".repeat(70);
                let thin = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(
                    " Chamberlain Test (correlação period-específica com efeitos individuais)\n",
                );
                out.push_str(" H₀: Π_s = 0 ∀s  (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: n={} obs   N={} entidades   T={} períodos\n",
                    n_obs, n_entities, t_count
                ));
                out.push_str(&format!("   Colunas de augmentação: {} de Chamberlain (k×T, após remover zero-variância)\n", k_active));
                if t_count > 6 {
                    out.push_str(&format!(
                        "   ⚠ T={} — com T grande o teste tem baixo poder em amostras finitas\n",
                        t_count
                    ));
                }
                out.push_str("\n── Teste conjunto H₀: todos os Π_s = 0\n");
                out.push_str(&format!(
                    "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
                    df1, df_denom, f_stat, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(
                    "   Teste mais geral que Mundlak — inclui valores em todos os T períodos\n",
                );
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Arellano-Bond Diff-GMM (OLD mundlak removed — use new mundlak above) ─
            "mundlak_OLD_REMOVED" => {
                let (formula_ast, df, _df_name, id_col) =
                    self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "mundlak: column '{id_col}' not found"
                    )));
                };

                // Nomes dos regressores variantes no tempo (excluindo "const")
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let non_const_names: Vec<&str> = var_names
                    .iter()
                    .filter(|n| n.as_str() != "const")
                    .map(|s| s.as_str())
                    .collect();

                let n = y_vec.len();
                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        s.insert(id);
                    }
                    s.len()
                };

                let (f_stat, p, k, gamma_hat, gamma_se) =
                    greeners::PanelDiagnostics::mundlak(&y_vec, &x_mat, &entity_ids)
                        .map_err(|e| HayashiError::Runtime(e))?;

                let df1 = k;
                let df2_exact = if n > 2 * k + 1 { n - 2 * k - 1 } else { 1 };

                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais correlacionados com X (prefira FE)"
                } else {
                    "Não rejeita H₀ → RE consistente (sem evidência de correlação com efeitos)"
                };

                let thick = "═".repeat(70);
                let thin = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(
                    " Mundlak Test (correlação entre regressores e efeitos individuais)\n",
                );
                out.push_str(" H₀: γ = 0  (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: n={} obs   N={} entidades   k={} regressores variantes\n",
                    n, n_entities, k
                ));
                out.push_str("\n── Coeficientes sobre médias individuais (X̄_i)\n");
                out.push_str(&format!(
                    "   {:<18} {:>10}  {:>10}  {:>8}\n",
                    "Variável (X̄)", "γ̂", "SE", "t"
                ));
                out.push_str(&format!("   {}\n", "─".repeat(52)));
                for i in 0..k {
                    let t_i = if gamma_se[i] > 1e-15 {
                        gamma_hat[i] / gamma_se[i]
                    } else {
                        f64::NAN
                    };
                    let name = non_const_names.get(i).copied().unwrap_or("?");
                    out.push_str(&format!(
                        "   {:<18} {:>10.4}  {:>10.4}  {:>8.3}\n",
                        format!("{}̄", name),
                        gamma_hat[i],
                        gamma_se[i],
                        t_i
                    ));
                }
                out.push_str("\n── Teste conjunto H₀: γ = 0\n");
                out.push_str(&format!(
                    "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
                    df1, df2_exact, f_stat, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Arellano-Bond Diff-GMM ────────────────────────────────────────
            // ab(formula, df, id=col, time=col [, lags=2 [, step=1]])
            // Estima y_it = ρ y_{i,t-1} + X_it'β + α_i + ε_it via Diff-GMM.
            // Instrumenta Δy_{i,t-1} com y_{i,t-2},...,y_{i,t-lags-1} (collapsed).
            "ab" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let time_col = self.get_time_col(&df_name, &opt_map)?;

                let max_lags: usize = match opt_map.get("lags") {
                    Some(Value::Int(v)) => (*v).max(1) as usize,
                    Some(Value::Float(v)) => (*v as i64).max(1) as usize,
                    None => 2,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "ab(): lags must be integer positivo".into(),
                        ))
                    }
                };

                let two_step: bool = match opt_map.get("step") {
                    Some(Value::Int(2)) => true,
                    Some(Value::Float(v)) if *v as i64 == 2 => true,
                    Some(Value::Int(_)) | Some(Value::Float(_)) | None => false,
                    _ => return Err(HayashiError::Runtime("ab(): step deve ser 1 ou 2".into())),
                };

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ab: coluna id '{id_col}' not found"
                    )));
                };

                let time_ids: Vec<i64> = if let Ok(ids) = df.get_int(&time_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&time_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ab: coluna time '{time_col}' not found"
                    )));
                };

                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::ArellanoBond::fit(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_ids,
                    max_lags,
                    two_step,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::AbResult(Rc::new(result)))
            }

            // ── GMM genérico (Two-Step Efficient) ────────────────────────────
            // gmm(endog_formula, instrument_formula, df)
            "gmm" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "gmm(endog_formula, instrument_formula, dataframe)".into(),
                    ));
                }
                let endog_ast = self.resolve_formula(&args[0])?;
                let instr_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "third argument must be a DataFrame variable".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };

                let endog_str = Self::formula_to_string(&endog_ast);
                let instr_str = Self::formula_to_string(&instr_ast);

                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let g_instr = if instr_ast.lhs.is_empty() {
                    let independents: Vec<String> = instr_ast
                        .rhs
                        .iter()
                        .map(|t| match t {
                            RhsTerm::Var(v) => v.clone(),
                            RhsTerm::Categorical(v) => format!("C({v})"),
                            RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                            RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
                        })
                        .collect();
                    GFormula {
                        dependent: String::new(),
                        independents,
                        intercept: true,
                    }
                } else {
                    GFormula::parse(&instr_str).map_err(|e| HayashiError::Runtime(e.to_string()))?
                };

                let (y, x) = df
                    .to_design_matrix(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let z = {
                    let n_rows = df.n_rows();
                    let n_cols = g_instr.independents.len() + if g_instr.intercept { 1 } else { 0 };
                    let mut z_mat = ndarray::Array2::<f64>::zeros((n_rows, n_cols));
                    let mut col_idx = 0;
                    if g_instr.intercept {
                        for i in 0..n_rows {
                            z_mat[[i, 0]] = 1.0;
                        }
                        col_idx = 1;
                    }
                    for (j, var_name) in g_instr.independents.iter().enumerate() {
                        let col_data = df
                            .get(var_name)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        for i in 0..n_rows {
                            z_mat[[i, col_idx + j]] = col_data[i];
                        }
                    }
                    z_mat
                };

                let result =
                    greeners::GMM::fit(&y, &x, &z).map_err(|e| self.rt_err(format!("gmm: {e}")))?;
                Ok(Value::GmmResult(Rc::new(result)))
            }

            // ── System GMM (Blundell-Bond 1998) ──────────────────────────────
            // sysgmm(formula, df, id=col, time=col [, lags=2 [, step=1]])
            // Empilha eq. em 1ª diferença (instrumentadas com níveis defasados)
            // + eq. em níveis (instrumentadas com Δy_{t-1} e ΔX_{t-1}).
            "sysgmm" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let time_col = self.get_time_col(&df_name, &opt_map)?;

                let max_lags: usize = match opt_map.get("lags") {
                    Some(Value::Int(v)) => (*v).max(1) as usize,
                    Some(Value::Float(v)) => (*v as i64).max(1) as usize,
                    None => 2,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "sysgmm(): lags must be integer positivo".into(),
                        ))
                    }
                };

                let two_step: bool = match opt_map.get("step") {
                    Some(Value::Int(2)) => true,
                    Some(Value::Float(v)) if *v as i64 == 2 => true,
                    Some(Value::Int(_)) | Some(Value::Float(_)) | None => false,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "sysgmm(): step deve ser 1 ou 2".into(),
                        ))
                    }
                };

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "sysgmm: coluna id '{id_col}' not found"
                    )));
                };

                let time_ids: Vec<i64> = if let Ok(ids) = df.get_int(&time_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&time_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "sysgmm: coluna time '{time_col}' not found"
                    )));
                };

                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::SystemGmm::fit(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_ids,
                    max_lags,
                    two_step,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::SysGmmResult(Rc::new(result)))
            }

            // ── FE-2SLS (xtivreg, fe) — Hausman (1978) ───────────────────────
            // feiv(endog_formula, instrument_formula, df, id=col [, cov=...])
            // endog_formula: y ~ x1 + x2   (x2 é endógena)
            // instrument_formula: ~ x1 + z1 + z2  (exógenos incluídos + excluídos)
            "feiv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "feiv() requer (formula_estrutural, formula_instrumentos, df, id=col)"
                            .into(),
                    ));
                }

                let endog_ast = self.resolve_formula(&args[0])?;
                let instr_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "feiv(): terceiro argumento deve ser nome do DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "feiv: '{df_name}' is not a DataFrame"
                        )))
                    }
                };

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "feiv(): opção id=col é obrigatória".into(),
                        ))
                    }
                };

                // fórmula estrutural → y e X (sem constante, FE a absorve)
                let endog_str = Self::formula_to_string(&endog_ast);
                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // fórmula de instrumentos → Z (sem constante)
                let instr_vars: Vec<String> = instr_ast
                    .rhs
                    .iter()
                    .map(|t| match t {
                        RhsTerm::Var(v) => v.clone(),
                        RhsTerm::Categorical(v) => format!("C({v})"),
                        RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                        RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
                    })
                    .collect();

                let n = y_vec.len();
                let l = instr_vars.len();
                if l == 0 {
                    return Err(HayashiError::Runtime(
                        "feiv(): formula de instrumentos deve ter ao menos um instrumento".into(),
                    ));
                }
                let mut z_mat = ndarray::Array2::<f64>::zeros((n, l));
                for (j, col_name) in instr_vars.iter().enumerate() {
                    let col = df.get(col_name).map_err(|_| {
                        HayashiError::Runtime(format!(
                            "feiv: instrumento '{col_name}' not found no DataFrame"
                        ))
                    })?;
                    for (i, &v) in col.iter().enumerate() {
                        z_mat[[i, j]] = v;
                    }
                }

                // entity IDs
                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "feiv: coluna id '{id_col}' not found"
                    )));
                };

                let var_names = df
                    .formula_var_names(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result =
                    greeners::FE2SLS::fit(&y_vec, &x_mat, &z_mat, &entity_ids, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::FE2SLSResult(Rc::new(result)))
            }

            // ── PCSE — Panel-Corrected Standard Errors (Beck & Katz 1995) ─────
            // pcse(formula, df, id=col, time=col)
            "pcse" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let time_col = self.get_time_col(&df_name, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let entity_ids = Self::col_as_i64(&df, &id_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let time_ids = Self::col_as_i64(&df, &time_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result =
                    greeners::PCSE::fit(&y_vec, &x_mat, &entity_ids, &time_ids, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PcseResult(Rc::new(result)))
            }

            // ── Panel GLS — Parks (1967) / Stata xtgls ───────────────────────
            // xtgls(formula, df, id=col, time=col [, panels="hetero"|"corr"])
            "xtgls" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let time_col = self.get_time_col(&df_name, &opt_map)?;
                let panels_opt = match opt_map.get("panels") {
                    Some(Value::Str(s)) if s == "corr" => greeners::GlsPanels::Correlated,
                    Some(Value::Str(s)) if s == "hetero" || s == "heteroscedastic" => {
                        greeners::GlsPanels::Hetero
                    }
                    None => greeners::GlsPanels::Hetero,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "xtgls(): panels deve ser \"hetero\" ou \"corr\"".into(),
                        ))
                    }
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let entity_ids = Self::col_as_i64(&df, &id_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let time_ids = Self::col_as_i64(&df, &time_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::PanelGLS::fit(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_ids,
                    panels_opt,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PanelGlsResult(Rc::new(result)))
            }

            // ── Arellano-Bond: teste m1/m2 para autocorrelação serial ─────────
            "ab_test" => {
                // ab_test(formula, df, id=col, time=col)
                // Testa autocorrelação serial nos resíduos da equação em 1ª diferença.
                // m1: DEVE rejeitar H₀ (FD induz AR(1) por construção)
                // m2: NÃO deve rejeitar H₀ (valida instrumentos y_{i,t-2} do GMM)
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let time_col = self.get_time_col(&df_name, &opt_map)?;

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ab_test: coluna id '{id_col}' not found"
                    )));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ab_test: coluna time '{time_col}' not found"
                    )));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        s.insert(id);
                    }
                    s.len()
                };

                let (m1, p1, m2, p2) = greeners::PanelDiagnostics::arellano_bond_test(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_vals,
                )
                .map_err(|e| HayashiError::Runtime(e))?;

                let sig = |p: f64| {
                    if p < 0.01 {
                        "***"
                    } else if p < 0.05 {
                        "**"
                    } else if p < 0.10 {
                        "*"
                    } else {
                        ""
                    }
                };
                let n_obs = y_vec.len();

                let thick = "═".repeat(66);
                let thin = "─".repeat(66);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(
                    " Arellano-Bond Test (autocorrelação serial — resíduos em 1ª diferença)\n",
                );
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: n={} obs   N={} entidades\n",
                    n_obs, n_entities
                ));
                out.push_str("\n── Estatísticas  z ~ N(0,1)   H₀: sem autocorrelação de ordem p\n");
                out.push_str(&format!("   {:-^52}\n", ""));
                out.push_str(&format!(
                    "   {:>4}  {:>10}  {:>10}  {:>6}  {}\n",
                    "p", "z", "p-valor", "sig", "Interpretação"
                ));
                out.push_str(&format!("   {:-^52}\n", ""));
                let interp1 = if p1 < 0.05 {
                    "OK — FD induz AR(1) (esperado)"
                } else {
                    "Inesperado — verificar modelo"
                };
                let interp2 = if p2 >= 0.05 {
                    "OK — instrumentos válidos"
                } else {
                    "Atenção — AR(2) detectado"
                };
                out.push_str(&format!(
                    "   {:>4}  {:>10.4}  {:>10.4}  {:>6}  {}\n",
                    1,
                    m1,
                    p1,
                    sig(p1),
                    interp1
                ));
                out.push_str(&format!(
                    "   {:>4}  {:>10.4}  {:>10.4}  {:>6}  {}\n",
                    2,
                    m2,
                    p2,
                    sig(p2),
                    interp2
                ));
                out.push_str(&format!("   {:-^52}\n", ""));
                out.push_str("\n── Conclusão\n");
                if p1 < 0.05 && p2 >= 0.05 {
                    out.push_str(
                        "   m1 rejeita e m2 não rejeita → estrutura consistente com GMM válido\n",
                    );
                } else if p1 >= 0.05 {
                    out.push_str(
                        "   m1 não rejeita H₀ → checar especificação (AR(1) esperado em FD)\n",
                    );
                } else {
                    out.push_str("   m2 rejeita H₀ → AR(2) nos resíduos; instrumentos y_{t-2} podem ser inválidos\n");
                    out.push_str(
                        "   Considere usar lags mais distantes (y_{t-3}, ...) como instrumentos\n",
                    );
                }
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(
                    "   Variância estimada via sandwich (Σ_i dos produtos cruzados por entidade)\n",
                );
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── wooldridge_OLD_REMOVED (substituído pelo novo acima) ──────────
            "wooldridge_OLD_REMOVED" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let time_col = self.get_time_col(&df_name, &opt_map)?;

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "wooldridge: coluna id '{id_col}' not found"
                    )));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "wooldridge: coluna time '{time_col}' not found"
                    )));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        s.insert(id);
                    }
                    s.len()
                };

                let (rho, t_stat, p, n_pairs) = greeners::PanelDiagnostics::wooldridge_serial(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_vals,
                )
                .map_err(|e| HayashiError::Runtime(e))?;

                let df_t = n_entities - 1;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → autocorrelação serial de 1ª ordem presente"
                } else {
                    "Não rejeita H₀ → sem evidência de autocorrelação serial"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Wooldridge Test (autocorrelação serial em painel)\n");
                out.push_str(" H₀: ρ = -0.5  (sem autocorrelação nos erros idiossincráticos)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: N={} entidades   pares usados={}   df={}\n",
                    n_entities, n_pairs, df_t
                ));
                out.push_str("\n── Estimativa\n");
                out.push_str(&format!("   ρ̂ = {:.4}   (H₀: ρ = -0.500)\n", rho));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   t({}) = {:.4}   p = {:.4}  {}\n",
                    df_t, t_stat, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(
                    "   (SE padrão OLS — use SE robustos clusterizados para inferência formal)\n",
                );
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Hausman FE vs RE ──────────────────────────────────────────────
            "hausman" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("hausman(fe_model, re_model)".into()));
                }

                let fe = match self.eval_expr(&args[0])? {
                    Value::PanelResult(r) => r,
                    _ => {
                        return Err(HayashiError::Type(
                            "hausman(): primeiro argumento deve ser um modelo FE".into(),
                        ))
                    }
                };
                let re = match self.eval_expr(&args[1])? {
                    Value::ReResult(r) => r,
                    _ => {
                        return Err(HayashiError::Type(
                            "hausman(): second argument must be um modelo RE".into(),
                        ))
                    }
                };

                // Variáveis comuns: FE não tem intercepto; RE tem.
                // Alinha por nome quando disponível; senão assume mesma ordem.
                let fe_names: Vec<String> =
                    fe.variable_names.as_ref().cloned().unwrap_or_else(|| {
                        (0..fe.params.len()).map(|i| format!("x{}", i)).collect()
                    });

                let re_names: Vec<String> =
                    re.variable_names.as_ref().cloned().unwrap_or_else(|| {
                        (0..re.params.len()).map(|i| format!("x{}", i)).collect()
                    });

                // Pares (β_FE, σ²_FE, β_RE, σ²_RE) para variáveis em comum (exclui intercepto)
                let mut pairs: Vec<(String, f64, f64, f64, f64)> = Vec::new();
                for (i, fe_name) in fe_names.iter().enumerate() {
                    if fe_name == "const" {
                        continue;
                    }
                    if let Some(j) = re_names.iter().position(|n| n == fe_name) {
                        pairs.push((
                            fe_name.clone(),
                            fe.params[i],
                            fe.std_errors[i].powi(2),
                            re.params[j],
                            re.std_errors[j].powi(2),
                        ));
                    }
                }

                if pairs.is_empty() {
                    return Err(HayashiError::Runtime(
                        "hausman: nenhuma variável comum entre FE e RE (verifique variable_names)"
                            .into(),
                    ));
                }

                // H = Σ (β_FE - β_RE)² / (σ²_FE - σ²_RE)  para pares onde σ²_FE > σ²_RE
                let mut chi2 = 0.0;
                let mut df = 0usize;
                let mut skipped = 0usize;

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();

                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Hausman Test: FE vs RE\n");
                out.push_str(" H₀: efeitos individuais não correlacionados com regressores (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Coeficientes Comuns\n");
                out.push_str(&format!(
                    "   {:<20} {:>10} {:>10} {:>10}\n",
                    "Variável", "β_FE", "β_RE", "Δβ"
                ));
                out.push_str(&format!("   {thin}\n"));

                for (name, bfe, vfe, bre, vre) in &pairs {
                    let diff = bfe - bre;
                    let dvar = vfe - vre;
                    out.push_str(&format!(
                        "   {:<20} {:>10.4} {:>10.4} {:>10.4}\n",
                        name, bfe, bre, diff
                    ));
                    if dvar > 1e-15 {
                        chi2 += diff.powi(2) / dvar;
                        df += 1;
                    } else {
                        skipped += 1;
                    }
                }

                if df == 0 {
                    out.push_str("\n   [!] Var(β_FE) ≤ Var(β_RE) em todos os coeficientes.\n");
                    out.push_str(
                        "       Estatística indefinida — verifique especificação dos modelos.\n",
                    );
                    out.push_str(&format!("\n{thick}\n"));
                    return Ok(Self::diag(out));
                }

                let p = greeners::chi2_pvalue(chi2, df as f64);

                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → use EFEITOS FIXOS (RE pode ser inconsistente)"
                } else {
                    "Não rejeita H₀ → EFEITOS ALEATÓRIOS é consistente e eficiente"
                };

                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   χ²({}) = {:.4}   p = {:.4}  {}\n",
                    df, chi2, p, sig
                ));
                if skipped > 0 {
                    out.push_str(&format!(
                        "   ({} coeficiente(s) excluídos: Var(β_FE) ≤ Var(β_RE))\n",
                        skipped
                    ));
                }
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Diagnósticos ──────────────────────────────────────────────────
            "test" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "test(model, name) requires 2 arguments".into(),
                    ));
                }
                let model = self.eval_expr(&args[0])?;

                let ols = match &model {
                    Value::OlsResult(m) => m.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "test() currently supports OLS models only".into(),
                        ))
                    }
                };

                let test_name = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    other => {
                        return Err(HayashiError::Type(format!(
                            "test name must be a string (e.g. \"white\"), got {other}"
                        )))
                    }
                };

                match test_name.as_str() {
                    // ── Specification tests ──────────────────────────────
                    "white" => match SpecificationTests::white_test(&ols.residuals, &ols.x) {
                        Ok((stat, p, df)) => {
                            println!("White Test for Heteroskedasticity");
                            println!("  LM statistic : {:.4}", stat);
                            println!("  p-value      : {:.4}", p);
                            println!("  df           : {}", df);
                            let verdict = if p < 0.05 {
                                "Reject H0 — evidence of heteroskedasticity"
                            } else {
                                "Fail to reject H0 — no evidence of heteroskedasticity"
                            };
                            println!("  Conclusion   : {}", verdict);
                        }
                        Err(e) => eprintln!("White test error: {e}"),
                    },
                    "bp" => match Diagnostics::breusch_pagan(&ols.residuals, &ols.x) {
                        Ok((stat, p)) => {
                            println!("Breusch-Pagan Test for Heteroskedasticity");
                            println!("  LM statistic : {:.4}", stat);
                            println!("  p-value      : {:.4}", p);
                            let verdict = if p < 0.05 {
                                "Reject H0 — evidence of heteroskedasticity"
                            } else {
                                "Fail to reject H0 — no evidence of heteroskedasticity"
                            };
                            println!("  Conclusion   : {}", verdict);
                        }
                        Err(e) => eprintln!("Breusch-Pagan test error: {e}"),
                    },
                    "dw" => {
                        let stat = Diagnostics::durbin_watson(&ols.residuals);
                        println!("Durbin-Watson Test for Autocorrelation");
                        println!("  DW statistic : {:.4}", stat);
                        let verdict = if stat < 1.5 {
                            "Positive autocorrelation suspected"
                        } else if stat > 2.5 {
                            "Negative autocorrelation suspected"
                        } else {
                            "No strong evidence of autocorrelation"
                        };
                        println!("  Conclusion   : {}", verdict);
                    }

                    // ── Wald / F-test sobre coeficientes ─────────────────
                    other => {
                        let names = ols.result.variable_names.as_ref().ok_or_else(|| {
                            HayashiError::Runtime("model has no variable names".into())
                        })?;
                        let k = ols.result.params.len();
                        let find_idx = |name: &str| -> Result<usize> {
                            let n = name.trim();
                            names
                                .iter()
                                .position(|v| v == n)
                                .or_else(|| {
                                    if n == "_cons" || n == "const" {
                                        Some(k - 1)
                                    } else {
                                        None
                                    }
                                })
                                .ok_or_else(|| {
                                    HayashiError::Runtime(format!(
                                        "variable '{n}' not found in model"
                                    ))
                                })
                        };

                        // "X1 = X2" ou "X1 = 0.5"
                        if let Some((lhs_s, rhs_s)) = other.split_once('=') {
                            let lhs_name = lhs_s.trim();
                            let rhs_trimmed = rhs_s.trim();
                            if let Ok(val) = rhs_trimmed.parse::<f64>() {
                                let idx = find_idx(lhs_name)?;
                                let mut r = ndarray::Array1::<f64>::zeros(k);
                                r[idx] = 1.0;
                                let (t, p) = ols
                                    .result
                                    .t_test(&r, val, &ols.x)
                                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                                println!("\n{:=^60}", " test ");
                                println!("  H₀: {lhs_name} = {val}");
                                println!("  t = {t:.4}   p = {p:.4}");
                                let sig = if p < 0.01 {
                                    "***"
                                } else if p < 0.05 {
                                    "**"
                                } else if p < 0.10 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!("  {sig}");
                            } else {
                                let idx1 = find_idx(lhs_name)?;
                                let idx2 = find_idx(rhs_trimmed)?;
                                let mut r = ndarray::Array1::<f64>::zeros(k);
                                r[idx1] = 1.0;
                                r[idx2] = -1.0;
                                let (t, p) = ols
                                    .result
                                    .t_test(&r, 0.0, &ols.x)
                                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                                println!("\n{:=^60}", " test ");
                                println!("  H₀: {lhs_name} = {rhs_trimmed}");
                                println!("  t = {t:.4}   p = {p:.4}");
                                let sig = if p < 0.01 {
                                    "***"
                                } else if p < 0.05 {
                                    "**"
                                } else if p < 0.10 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!("  {sig}");
                            }
                        } else {
                            let mut extra_names: Vec<String> = Vec::new();
                            for arg in &args[2..] {
                                let name = match self.eval_expr(arg)? {
                                    Value::Str(s) => s,
                                    other => {
                                        return Err(HayashiError::Type(format!(
                                            "test() variable names must be strings, got {other}"
                                        )))
                                    }
                                };
                                extra_names.push(name);
                            }
                            let mut indices = vec![find_idx(other)?];
                            for name in &extra_names {
                                indices.push(find_idx(name)?);
                            }
                            let (f, p) = ols
                                .result
                                .f_test(&indices, &ols.x)
                                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                            let var_list: Vec<&str> =
                                indices.iter().map(|&i| names[i].as_str()).collect();
                            let q = indices.len();
                            println!("\n{:=^60}", " test ");
                            if q == 1 {
                                println!("  H₀: {} = 0", var_list[0]);
                            } else {
                                println!("  H₀: {} = 0", var_list.join(" = "));
                            }
                            println!("  F({q}, {}) = {f:.4}   p = {p:.4}", ols.result.df_resid);
                            let sig = if p < 0.01 {
                                "***"
                            } else if p < 0.05 {
                                "**"
                            } else if p < 0.10 {
                                "*"
                            } else {
                                ""
                            };
                            println!("  {sig}");
                        }
                    }
                }

                Ok(Value::Nil)
            }

            // ── set_seed: reprodutibilidade ────────────────────────────────
            "set_seed" | "seed" | "setseed" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "set_seed(N) — define semente do RNG".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Int(v) => v as u64,
                    Value::Float(v) => v as u64,
                    _ => return Err(HayashiError::Type("seed must be integer".into())),
                };
                self.rng_seed = Some(s);
                use rand::SeedableRng;
                self.rng = rand::rngs::StdRng::seed_from_u64(s);
                println!("set seed {s}");
                Ok(Value::Nil)
            }

            // ── timer: mede tempo de execução ─────────────────────────────
            "timer" | "time" | "bench" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "timer(expr) — mede tempo de avaliação".into(),
                    ));
                }
                let start = std::time::Instant::now();
                let result = self.eval_expr(&args[0])?;
                let elapsed = start.elapsed();
                println!("  elapsed: {:.4}s", elapsed.as_secs_f64());
                Ok(result)
            }

            // ── quietly: avalia expressão, suprime saída ──────────────────
            "quietly" | "quiet" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "quietly(expr) — avalia sem imprimir".into(),
                    ));
                }
                self.eval_expr(&args[0])?;
                Ok(Value::Nil)
            }

            // ── capture: avalia expressão, ignora erros ───────────────────
            "capture" | "cap" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "capture(expr) — avalia ignorando erros".into(),
                    ));
                }
                match self.eval_expr(&args[0]) {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        eprintln!("(captured: {e})");
                        Ok(Value::Nil)
                    }
                }
            }

            // ── assert: erro se condição é falsa ──────────────────────────
            "assert" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "assert(cond [, msg]) — erro se condição falsa".into(),
                    ));
                }
                let val = self.eval_expr(&args[0])?;
                if !Self::value_as_bool(&val) {
                    let msg = if args.len() >= 2 {
                        match self.eval_expr(&args[1])? {
                            Value::Str(s) => s,
                            _ => "assertion failed".into(),
                        }
                    } else {
                        "assertion failed".into()
                    };
                    return Err(HayashiError::Runtime(msg));
                }
                Ok(Value::Nil)
            }

            // ── preserve/restore: salvar e restaurar estado de variáveis ───
            "preserve" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "preserve(df) — salva cópia do DataFrame".into(),
                    ));
                }
                let name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "preserve() requires a variable name".into(),
                        ))
                    }
                };
                let val = self
                    .env
                    .get(&name)
                    .ok_or_else(|| self.rt_err(format!("'{name}' not found")))?
                    .clone();
                self.preserved.insert(name.clone(), val);
                println!("preserve {name}");
                Ok(Value::Nil)
            }

            "restore" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "restore(df) — restaura DataFrame salvo".into(),
                    ));
                }
                let name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "restore() requires a variable name".into(),
                        ))
                    }
                };
                let val = self
                    .preserved
                    .remove(&name)
                    .ok_or_else(|| self.rt_err(format!("'{name}' was not preserved")))?;
                self.env.set(&name, val)?;
                println!("restore {name}");
                Ok(Value::Nil)
            }

            // ── source/do: executa script .hay no ambiente atual ─────────────
            "source" | "do" | "run" | "include" => {
                if args.is_empty() {
                    return Err(self.rt_err("source(\"script.hay\")"));
                }
                let path = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("source() requires a string path".into())),
                };
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| self.rt_err(format!("cannot read '{path}': {e}")))?;
                println!("source {path}");
                crate::lang::run_source(&src, self)?;
                Ok(Value::Nil)
            }

            "import" | "require" => {
                if args.is_empty() {
                    return Err(self.rt_err("import(\"module_or_url\")"));
                }
                let module = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("import() requires a string".into())),
                };

                if self.imported.contains(&module) {
                    return Ok(Value::Nil);
                }

                let resolved = if crate::io::fetch::is_url(&module) {
                    let tmp = crate::io::fetch::download_to_temp(&module)?;
                    tmp.to_string_lossy().to_string()
                } else {
                    self.resolve_import(&module)?
                };

                let alias = match opt_map.get("as") {
                    Some(Value::Str(s)) => Some(s.clone()),
                    _ => None,
                };
                let only: Option<Vec<String>> = match opt_map.get("only") {
                    Some(Value::List(lst)) => Some(
                        lst.iter()
                            .filter_map(|v| match v {
                                Value::Str(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect(),
                    ),
                    _ => None,
                };

                let ns = alias.clone().unwrap_or_else(|| {
                    let base = module
                        .trim_end_matches(".hay")
                        .trim_end_matches(".wasm")
                        .trim_end_matches(".so")
                        .trim_end_matches(".dll")
                        .trim_end_matches(".dylib");
                    base.rsplit('/').next().unwrap_or(&module).to_string()
                });

                let is_wasm = resolved.ends_with(".wasm");
                let is_native = resolved.ends_with(".so")
                    || resolved.ends_with(".dll")
                    || resolved.ends_with(".dylib");

                if is_wasm {
                    use super::plugin::WasmPlugin;
                    let plugin = WasmPlugin::new(&resolved, &ns).map_err(|e| {
                        self.rt_err(format!("import: failed to load WASM plugin: {e}"))
                    })?;
                    self.plugins.insert(ns.clone(), Box::new(plugin));
                    self.imported.insert(module.clone());
                    return Ok(Value::Nil);
                } else if is_native {
                    use super::plugin::RustNativePlugin;
                    let plugin = RustNativePlugin::new(&resolved, &ns).map_err(|e| {
                        self.rt_err(format!("import: failed to load native plugin: {e}"))
                    })?;
                    self.plugins.insert(ns.clone(), Box::new(plugin));
                    self.imported.insert(module.clone());
                    return Ok(Value::Nil);
                }

                // Default script plugin (.hay) loading
                let src = std::fs::read_to_string(&resolved)
                    .map_err(|e| self.rt_err(format!("import: cannot read '{resolved}': {e}")))?;

                self.imported.insert(module.clone());

                let before: std::collections::HashSet<String> =
                    self.env.var_names().into_iter().collect();

                crate::lang::run_source(&src, self)?;

                let new_names: Vec<String> = self
                    .env
                    .var_names()
                    .into_iter()
                    .filter(|n| !before.contains(n))
                    .collect();

                if let Some(ref allowed) = only {
                    for name in &new_names {
                        if !allowed.contains(name) {
                            self.env.remove(name);
                        }
                    }
                } else {
                    for name in &new_names {
                        if let Some(val) = self.env.get(name).cloned() {
                            let qualified = format!("{ns}::{name}");
                            self.env.declare(&qualified, val).ok();
                            self.env.remove(name);
                        }
                    }
                }

                Ok(Value::Nil)
            }

            "plugin_path" => {
                if args.is_empty() {
                    if self.plugin_paths.is_empty() {
                        println!("plugin_path: (none)");
                    } else {
                        for p in &self.plugin_paths {
                            println!("  {p}");
                        }
                    }
                    return Ok(Value::Nil);
                }
                for arg in args {
                    let path = match self.eval_expr(arg)? {
                        Value::Str(s) => s,
                        other => {
                            return Err(
                                self.type_err(format!("plugin_path: expected string, got {other}"))
                            )
                        }
                    };
                    if !self.plugin_paths.contains(&path) {
                        self.plugin_paths.push(path);
                    }
                }
                Ok(Value::Nil)
            }

            // ── help: sistema de ajuda inline ──────────────────────────────
            "help" => {
                let topic = if args.is_empty() {
                    String::new()
                } else {
                    match &args[0] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => String::new(),
                    }
                };
                if topic == "about" {
                    println!("{}", crate::lang::help::help_about());
                } else if topic == "license" {
                    println!("{}", crate::lang::help::help_license());
                } else {
                    match crate::lang::help::help_text(&topic) {
                        Some(h) => println!("{h}"),
                        None => println!(
                            "help: '{}' not documented. Type help() for full list.",
                            topic
                        ),
                    }
                }
                Ok(Value::Nil)
            }

            // ── describe ─────────────────────────────────────────────────────
            "describe" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("describe() takes 1 argument".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => Some(n.clone()),
                    _ => None,
                };
                match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => {
                        println!("{}", df);
                        // mostrar labels se existirem
                        if let Some(ref name) = df_name {
                            if let Some(var_labels) = self.labels.get(name) {
                                if !var_labels.is_empty() {
                                    println!("\n  Labels:");
                                    let mut sorted: Vec<_> = var_labels.iter().collect();
                                    sorted.sort_by_key(|(k, _)| (*k).clone());
                                    for (var, lbl) in sorted {
                                        println!("    {:<20} {}", var, lbl);
                                    }
                                }
                            }
                        }
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Type("describe() requires a DataFrame".into())),
                }
            }

            // ── codebook ─────────────────────────────────────────────────────
            "codebook" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "codebook(df [, var1, var2, ...])".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };

                let requested: Vec<String> = if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    let mut names = df.column_names();
                    names.sort();
                    names
                };

                let sep = "─".repeat(76);
                println!("\n{:═^76}", " Codebook ");

                for name in &requested {
                    use greeners::Column;
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                    println!("\n{sep}");
                    match col {
                        Column::Float(arr) => {
                            let total = arr.len();
                            let vals: Vec<f64> =
                                arr.iter().copied().filter(|x| x.is_finite()).collect();
                            let missing = total - vals.len();
                            let n = vals.len();
                            println!(
                                "  {:<20} type: float    obs: {}    missing: {}",
                                name, total, missing
                            );
                            if n > 0 {
                                let mean = vals.iter().sum::<f64>() / n as f64;
                                let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                    / (n as f64 - 1.0).max(1.0);
                                let sd = var.sqrt();
                                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let mut sorted = vals.clone();
                                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                let pctile = |p: f64| -> f64 {
                                    let idx = (p * (n - 1) as f64).round() as usize;
                                    sorted[idx.min(n - 1)]
                                };
                                let mut unique = sorted.clone();
                                unique.dedup();
                                println!(
                                    "  unique: {}    mean: {:.4}    sd: {:.4}",
                                    unique.len(),
                                    mean,
                                    sd
                                );
                                println!(
                                    "  min: {:.4}    p25: {:.4}    p50: {:.4}    p75: {:.4}    max: {:.4}",
                                    min,
                                    pctile(0.25),
                                    pctile(0.50),
                                    pctile(0.75),
                                    max
                                );
                            }
                        }
                        Column::Int(arr) => {
                            let total = arr.len();
                            let vals: Vec<f64> = arr.iter().map(|&x| x as f64).collect();
                            let n = vals.len();
                            println!("  {:<20} type: int      obs: {}    missing: 0", name, total);
                            if n > 0 {
                                let mean = vals.iter().sum::<f64>() / n as f64;
                                let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                    / (n as f64 - 1.0).max(1.0);
                                let sd = var.sqrt();
                                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let mut sorted = vals.clone();
                                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                let mut unique = sorted.clone();
                                unique.dedup();
                                println!(
                                    "  unique: {}    mean: {:.4}    sd: {:.4}",
                                    unique.len(),
                                    mean,
                                    sd
                                );
                                println!("  min: {:.0}    max: {:.0}", min, max);
                            }
                        }
                        Column::String(arr) => {
                            let total = arr.len();
                            let non_empty = arr.iter().filter(|s: &&String| !s.is_empty()).count();
                            let missing = total - non_empty;
                            let mut unique: Vec<&str> =
                                arr.iter().map(|s: &String| s.as_str()).collect();
                            unique.sort();
                            unique.dedup();
                            println!(
                                "  {:<20} type: string   obs: {}    missing: {}",
                                name, total, missing
                            );
                            println!("  unique: {}", unique.len());
                            if unique.len() <= 10 {
                                let examples: Vec<&str> = unique.iter().take(10).copied().collect();
                                println!("  values: {}", examples.join(", "));
                            } else {
                                let first5: Vec<&str> = unique.iter().take(5).copied().collect();
                                println!(
                                    "  values: {}, ... ({} more)",
                                    first5.join(", "),
                                    unique.len() - 5
                                );
                            }
                        }
                        Column::Bool(arr) => {
                            let total = arr.len();
                            let trues = arr.iter().filter(|&&b| b).count();
                            let falses = total - trues;
                            println!("  {:<20} type: bool     obs: {}    missing: 0", name, total);
                            println!("  true: {}    false: {}", trues, falses);
                        }
                        _ => {
                            println!("  {:<20} type: other", name);
                        }
                    }
                }
                println!("\n{sep}");
                println!();
                Ok(Value::Nil)
            }

            // ── format: formata valor numérico ──────────────────────────────
            "format" | "fmt" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "format(value, fmt_str) — Ex: format(3.14, \"%.2f\")".into(),
                    ));
                }
                let val = match self.eval_expr(&args[0])? {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    other => {
                        return Err(HayashiError::Type(format!(
                            "format(): primeiro argumento must be numeric, não {other}"
                        )))
                    }
                };
                let fmt_s = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => {
                        return Err(HayashiError::Type(
                            "format(): second argument must be string (ex: \"%.2f\")".into(),
                        ))
                    }
                };
                // parse "%.Nf" → N decimal places
                let decimals: usize = if fmt_s.starts_with("%.") && fmt_s.ends_with('f') {
                    fmt_s[2..fmt_s.len() - 1].parse().unwrap_or(4)
                } else if fmt_s.starts_with('%') && fmt_s.ends_with('f') {
                    // "%f" sem especificar decimais
                    6
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "format(): string de formato '{fmt_s}' não reconhecida (use \"%.Nf\")"
                    )));
                };
                Ok(Value::Str(format!("{:.prec$}", val, prec = decimals)))
            }

            // ── duplicates: reportar/dropar/marcar duplicatas ────────────────
            "duplicates" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "duplicates(df, var [, action=report|drop|tag])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "duplicates(): primeiro argumento deve ser variable name".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "'{df_name}' is not a DataFrame"
                        )))
                    }
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "duplicates(): second argument must be nome de coluna".into(),
                        ))
                    }
                };
                let action = match opt_map.get("action") {
                    Some(Value::Str(s)) => s.clone(),
                    None => "report".into(),
                    _ => "report".into(),
                };

                let col = Self::get_col_f64(&df, &var_name)?;
                let n = col.len();

                // contar ocorrências de cada valor
                let mut counts: HashMap<i64, usize> = HashMap::new();
                for &v in col.iter() {
                    let key = v.to_bits() as i64;
                    *counts.entry(key).or_insert(0) += 1;
                }

                let n_dup: usize = counts.values().filter(|&&c| c > 1).map(|c| c - 1).sum();
                let n_unique = counts.len();

                match action.as_str() {
                    "report" => {
                        println!("duplicates report: {var_name}");
                        println!("  observações:    {n}");
                        println!("  valores únicos: {n_unique}");
                        println!("  duplicatas:     {n_dup}");
                        Ok(Value::Int(n_dup as i64))
                    }
                    "drop" => {
                        let mut seen: std::collections::HashSet<i64> =
                            std::collections::HashSet::new();
                        let keep: Vec<usize> = (0..n)
                            .filter(|&i| {
                                let key = col[i].to_bits() as i64;
                                seen.insert(key)
                            })
                            .collect();
                        let new_df = df
                            .iloc(Some(&keep), None)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!(
                            "duplicates drop: {n_dup} obs removidas, {} restantes",
                            new_df.n_rows()
                        );
                        self.env.set(&df_name, Value::DataFrame(Rc::new(new_df)))?;
                        Ok(Value::Nil)
                    }
                    "tag" => {
                        let dup_col: Vec<f64> = (0..n)
                            .map(|i| {
                                let key = col[i].to_bits() as i64;
                                *counts.get(&key).unwrap_or(&1) as f64
                            })
                            .collect();
                        let mut df_mut = df.clone();
                        let arr = ndarray::Array1::from(dup_col);
                        Rc::make_mut(&mut df_mut)
                            .insert("_dup".to_string(), arr)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!("duplicates tag: coluna _dup gerada ({n_dup} duplicatas)");
                        self.env.set(&df_name, Value::DataFrame(df_mut))?;
                        Ok(Value::Nil)
                    }
                    other => Err(HayashiError::Runtime(format!(
                        "duplicates(): action '{other}' desconhecida (report|drop|tag)"
                    ))),
                }
            }

            // ── label: armazena rótulos de variáveis ─────────────────────────
            "label" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "label(df, var, \"descrição\")".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "label(): primeiro argumento deve ser nome do DataFrame".into(),
                        ))
                    }
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "label(): second argument must be variable name".into(),
                        ))
                    }
                };
                let description = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    _ => {
                        return Err(HayashiError::Type(
                            "label(): terceiro argumento must be string".into(),
                        ))
                    }
                };
                self.labels
                    .entry(df_name.clone())
                    .or_insert_with(HashMap::new)
                    .insert(var_name.clone(), description.clone());
                println!("label {df_name}.{var_name} = \"{description}\"");
                Ok(Value::Nil)
            }

            // ── correlate ────────────────────────────────────────────────────
            "correlate" | "corr" | "pwcorr" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "correlate() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };

                // variáveis pedidas ou todas as numéricas
                let names: Vec<String> = if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    use greeners::Column;
                    let mut ns: Vec<String> = df
                        .column_names()
                        .into_iter()
                        .filter(|n| {
                            matches!(df.get_column(n), Ok(Column::Float(_)) | Ok(Column::Int(_)))
                        })
                        .collect();
                    ns.sort();
                    ns
                };

                if names.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "correlate() needs at least 2 numeric variables".into(),
                    ));
                }

                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                let sub = df
                    .select(&refs)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let mat = sub
                    .corr()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // corr() ordena colunas alfabeticamente — sincronizar com a matriz
                let mut sorted_names = names.clone();
                sorted_names.sort();

                let col_w = 10usize;
                let row_label_w = 16usize;
                let trunc = |s: &str, w: usize| {
                    if s.len() > w {
                        s[..w].to_string()
                    } else {
                        s.to_string()
                    }
                };

                // cabeçalho
                print!("{:>width$} |", "", width = row_label_w);
                for name in &sorted_names {
                    print!(" {:>width$}", trunc(name, col_w), width = col_w);
                }
                println!();
                println!(
                    "{}-+{}",
                    "-".repeat(row_label_w),
                    "-".repeat((col_w + 1) * sorted_names.len())
                );

                // p-value: t = r*sqrt(n-2)/sqrt(1-r²), df=n-2
                let show_stars =
                    func == "pwcorr" || matches!(opt_map.get("star"), Some(Value::Bool(true)));
                let n_obs = df.n_rows() as f64;
                let corr_pval = |r: f64| -> f64 {
                    if n_obs <= 2.0 || (1.0 - r * r) <= 0.0 {
                        return 1.0;
                    }
                    let t = r * (n_obs - 2.0).sqrt() / (1.0 - r * r).sqrt();
                    t_pvalue_two(t, n_obs - 2.0)
                };
                let star = |p: f64| -> &str {
                    if p < 0.01 {
                        "***"
                    } else if p < 0.05 {
                        "**"
                    } else if p < 0.10 {
                        "*"
                    } else {
                        ""
                    }
                };

                for (i, row_name) in sorted_names.iter().enumerate() {
                    print!(
                        "{:>width$} |",
                        trunc(row_name, row_label_w),
                        width = row_label_w
                    );
                    for j in 0..=i {
                        let r = mat[[i, j]];
                        if show_stars && i != j {
                            let s = star(corr_pval(r));
                            print!(" {:>7.4}{:<3}", r, s);
                        } else {
                            print!(" {:>10.4}", r);
                        }
                    }
                    println!();
                }
                if show_stars {
                    println!("* p<0.10  ** p<0.05  *** p<0.01");
                }
                println!();
                Ok(Value::Nil)
            }

            // ── summarize ────────────────────────────────────────────────────
            "summarize" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "summarize() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };

                let requested: Vec<String> = if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    let mut names = df.column_names();
                    names.sort();
                    names
                };

                let detail = matches!(opt_map.get("detail"), Some(Value::Bool(true)))
                    || matches!(opt_map.get("d"), Some(Value::Bool(true)));
                let quiet = self.capturing;

                if !quiet {
                    println!(
                        "\n{:<16} {:>9}  {:>7}  {:>12} {:>12} {:>12} {:>12}",
                        "Variable", "Obs", "Missing", "Mean", "Std. Dev.", "Min", "Max"
                    );
                    println!("{}", "-".repeat(91));
                }

                let mut result_dicts: Vec<(String, HashMap<String, Value>)> = Vec::new();

                for name in &requested {
                    use greeners::Column;
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                    let (n_total, n_missing, vals): (usize, usize, Vec<f64>) = match col {
                        Column::Float(arr) => {
                            let total = arr.len();
                            let vals: Vec<f64> =
                                arr.iter().copied().filter(|x| x.is_finite()).collect();
                            let missing = total - vals.len();
                            (total, missing, vals)
                        }
                        Column::Int(arr) => {
                            let vals: Vec<f64> = arr.iter().map(|&x| x as f64).collect();
                            (vals.len(), 0, vals)
                        }
                        _ => {
                            if !quiet {
                                println!("{:<16} {:>9}  {:>7}", name, "(non-numeric)", "");
                            }
                            continue;
                        }
                    };

                    let n = vals.len();
                    if n == 0 {
                        if !quiet {
                            println!("{:<16} {:>9}  {:>7}  (all missing)", name, 0, n_total);
                        }
                        continue;
                    }

                    let mean = vals.iter().sum::<f64>() / n as f64;
                    let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0).max(1.0);
                    let sd = variance.sqrt();
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    if !quiet {
                        let miss_str = if n_missing > 0 {
                            format!("{}", n_missing)
                        } else {
                            String::new()
                        };
                        println!(
                            "{:<16} {:>9}  {:>7}  {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                            name, n, miss_str, mean, sd, min, max
                        );
                    }

                    let mut d = HashMap::new();
                    d.insert("N".into(), Value::Int(n as i64));
                    d.insert("missing".into(), Value::Int(n_missing as i64));
                    d.insert("mean".into(), Value::Float(mean));
                    d.insert("sd".into(), Value::Float(sd));
                    d.insert("min".into(), Value::Float(min));
                    d.insert("max".into(), Value::Float(max));
                    d.insert("variance".into(), Value::Float(variance));

                    if detail {
                        let mut sorted = vals.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let pctile = |p: f64| -> f64 {
                            let idx = (p * (n - 1) as f64).round() as usize;
                            sorted[idx.min(n - 1)]
                        };
                        let p1 = pctile(0.01);
                        let p5 = pctile(0.05);
                        let p10 = pctile(0.10);
                        let p25 = pctile(0.25);
                        let p50 = pctile(0.50);
                        let p75 = pctile(0.75);
                        let p90 = pctile(0.90);
                        let p95 = pctile(0.95);
                        let p99 = pctile(0.99);
                        let skew = if n > 2 {
                            let m3 = vals.iter().map(|x| ((x - mean) / sd).powi(3)).sum::<f64>();
                            m3 * n as f64 / ((n - 1) as f64 * (n - 2) as f64)
                        } else {
                            f64::NAN
                        };
                        let kurt = if n > 3 {
                            let m4 = vals.iter().map(|x| ((x - mean) / sd).powi(4)).sum::<f64>()
                                / n as f64;
                            m4
                        } else {
                            f64::NAN
                        };
                        if !quiet {
                            println!("         Percentiles:");
                            println!("          1%  {:>10.4}       Skewness  {:>10.4}", p1, skew);
                            println!("          5%  {:>10.4}       Kurtosis  {:>10.4}", p5, kurt);
                            println!("         10%  {:>10.4}", p10);
                            println!(
                                "         25%  {:>10.4}       Variance  {:>10.4}",
                                p25, variance
                            );
                            println!("         50%  {:>10.4}", p50);
                            println!("         75%  {:>10.4}", p75);
                            println!("         90%  {:>10.4}", p90);
                            println!("         95%  {:>10.4}", p95);
                            println!("         99%  {:>10.4}", p99);
                        }
                        d.insert("p1".into(), Value::Float(p1));
                        d.insert("p5".into(), Value::Float(p5));
                        d.insert("p10".into(), Value::Float(p10));
                        d.insert("p25".into(), Value::Float(p25));
                        d.insert("p50".into(), Value::Float(p50));
                        d.insert("p75".into(), Value::Float(p75));
                        d.insert("p90".into(), Value::Float(p90));
                        d.insert("p95".into(), Value::Float(p95));
                        d.insert("p99".into(), Value::Float(p99));
                        d.insert("skewness".into(), Value::Float(skew));
                        d.insert("kurtosis".into(), Value::Float(kurt));
                    }
                    result_dicts.push((name.clone(), d));
                }
                if !quiet {
                    println!();
                }

                if quiet {
                    if result_dicts.len() == 1 {
                        let (_, d) = result_dicts.into_iter().next().unwrap();
                        Ok(Value::Dict(Rc::new(d)))
                    } else {
                        let mut outer = HashMap::new();
                        for (name, d) in result_dicts {
                            outer.insert(name, Value::Dict(Rc::new(d)));
                        }
                        Ok(Value::Dict(Rc::new(outer)))
                    }
                } else {
                    Ok(Value::Nil)
                }
            }

            // ── esttab ───────────────────────────────────────────────────────
            // ── eststo: acumula modelo para esttab posterior ──────────────
            "eststo" | "est_store" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("eststo(model)".into()));
                }
                let val = self.eval_expr(&args[0])?;
                let n = self.stored_models.len() + 1;
                self.stored_models.push(val);
                println!(
                    "eststo: modelo {n} armazenado ({} total)",
                    self.stored_models.len()
                );
                Ok(Value::Nil)
            }

            "estclear" => {
                let n = self.stored_models.len();
                self.stored_models.clear();
                println!("estclear: {n} modelos removidos");
                Ok(Value::Nil)
            }

            "esttab" => {
                // sem args → usa modelos acumulados via eststo
                let use_stored = args.is_empty();
                if use_stored && self.stored_models.is_empty() {
                    return Err(HayashiError::Runtime(
                        "esttab() requires models — pass as args or use eststo() first".into(),
                    ));
                }

                let fmt = match opt_map.get("fmt") {
                    Some(Value::Str(s)) => s.clone(),
                    None => "txt".to_string(),
                    _ => return Err(HayashiError::Type("fmt= must be a string".into())),
                };
                let out_path = match opt_map.get("path") {
                    Some(Value::Str(s)) => Some(s.clone()),
                    None => None,
                    _ => return Err(HayashiError::Type("path= must be a string".into())),
                };

                // (nome_variável, coef, se_opt, pval_opt)
                type CoefRow = (String, f64, Option<f64>, Option<f64>);
                // (label, coefs, n_obs, fit_stats)
                struct ModelInfo {
                    label: String,
                    coefs: Vec<CoefRow>,
                    n: usize,
                    r2: Option<f64>,
                    adj_r2: Option<f64>,
                    #[allow(dead_code)]
                    ll: Option<f64>,
                }

                // parseia CSV do OlsResult: variable,coef,se,t,p
                let parse_csv = |csv: &str| -> Vec<CoefRow> {
                    let mut rows = Vec::new();
                    let mut first = true;
                    for line in csv.lines() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        if first {
                            first = false;
                            continue;
                        } // cabeçalho
                        let f: Vec<&str> = line.splitn(6, ',').collect();
                        if f.len() >= 5 {
                            let raw = f[0].trim().trim_matches('"');
                            let name = if raw == "const" {
                                "_cons".to_string()
                            } else {
                                raw.to_string()
                            };
                            let coef = f[1].trim().parse::<f64>().unwrap_or(f64::NAN);
                            let se = f[2].trim().parse::<f64>().unwrap_or(f64::NAN);
                            let p = f[4].trim().parse::<f64>().unwrap_or(1.0);
                            rows.push((name, coef, Some(se), Some(p)));
                        }
                    }
                    rows
                };

                let stars = |p: Option<f64>| match p {
                    Some(p) if p < 0.01 => "***",
                    Some(p) if p < 0.05 => "**",
                    Some(p) if p < 0.10 => "*",
                    _ => "",
                };

                let extract_std = |label: &str,
                                   vnames: &Option<Vec<String>>,
                                   params: &ndarray::Array1<f64>,
                                   se: &ndarray::Array1<f64>,
                                   pv: &ndarray::Array1<f64>,
                                   n: usize|
                 -> ModelInfo {
                    let k = params.len();
                    let fb: Vec<String> = (0..k).map(|i| format!("x{i}")).collect();
                    let nm = vnames.as_ref().unwrap_or(&fb);
                    let coefs: Vec<CoefRow> = nm
                        .iter()
                        .zip(params.iter())
                        .zip(se.iter())
                        .zip(pv.iter())
                        .map(|(((n, &c), &s), &p)| (n.clone(), c, Some(s), Some(p)))
                        .collect();
                    ModelInfo {
                        label: label.to_string(),
                        coefs,
                        n,
                        r2: None,
                        adj_r2: None,
                        ll: None,
                    }
                };

                let mut models: Vec<ModelInfo> = Vec::new();
                let model_vals: Vec<Value> = if use_stored {
                    self.stored_models.clone()
                } else {
                    let mut vals = Vec::new();
                    for a in args {
                        let v = self.eval_expr(a)?;
                        if let Value::List(items) = v {
                            vals.extend(items.iter().cloned());
                        } else {
                            vals.push(v);
                        }
                    }
                    vals
                };
                for val in model_vals {
                    match val {
                        Value::OlsResult(m) => {
                            use greeners::ExportableResult;
                            let coefs = parse_csv(&m.result.to_csv());
                            let n = m.residuals.len();
                            let cov_label = match &m.result.cov_type {
                                CovarianceType::NonRobust => "",
                                CovarianceType::HC1 => " (robust)",
                                CovarianceType::HC2 => " (HC2)",
                                CovarianceType::HC3 => " (HC3)",
                                CovarianceType::HC4 => " (HC4)",
                                CovarianceType::NeweyWest(l) => {
                                    let _ = l;
                                    " (NW)"
                                }
                                CovarianceType::Clustered(_) => " (cluster)",
                                CovarianceType::ClusteredTwoWay(_, _) => " (2w-cluster)",
                            };
                            models.push(ModelInfo {
                                label: format!("OLS{cov_label}"),
                                coefs,
                                n,
                                r2: Some(m.result.r_squared),
                                adj_r2: Some(m.result.adj_r_squared),
                                ll: Some(m.result.log_likelihood),
                            });
                        }
                        Value::BinaryResult(bm) => {
                            let label = if bm.kind == "logit" {
                                "Logit"
                            } else {
                                "Probit"
                            }
                            .to_string();
                            let n = bm.x.nrows();
                            models.push(extract_std(
                                &label,
                                &bm.result.variable_names,
                                &bm.result.params,
                                &bm.result.std_errors,
                                &bm.result.p_values,
                                n,
                            ));
                        }
                        Value::IvResult(r) => {
                            models.push(extract_std(
                                "IV/2SLS",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::PoissonResult(r) => {
                            models.push(extract_std(
                                "Poisson",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::NegBinResult(r) => {
                            models.push(extract_std(
                                "NegBin",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::OrderedResult(r) => {
                            let mut info = extract_std(
                                &r.model_name,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            );
                            for (i, (&thr, &thr_se)) in r
                                .thresholds
                                .iter()
                                .zip(r.threshold_std_errors.iter())
                                .enumerate()
                            {
                                info.coefs.push((
                                    format!("_cut{}", i + 1),
                                    thr,
                                    Some(thr_se),
                                    None,
                                ));
                            }
                            models.push(info);
                        }
                        Value::TobitResult(r) => {
                            let mut info = extract_std(
                                "Tobit",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            );
                            info.coefs.push(("_sigma".into(), r.sigma, None, None));
                            models.push(info);
                        }
                        Value::HeckmanResult(r) => {
                            let mut info = extract_std(
                                "Heckman",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            );
                            let dz = if r.delta_se > 0.0 {
                                r.delta / r.delta_se
                            } else {
                                f64::NAN
                            };
                            let dp = if dz.is_finite() {
                                t_pvalue_two(dz, r.n_selected as f64)
                            } else {
                                f64::NAN
                            };
                            info.coefs.push((
                                "_lambda".into(),
                                r.delta,
                                Some(r.delta_se),
                                Some(dp),
                            ));
                            models.push(info);
                        }
                        Value::PanelResult(r) => {
                            models.push(extract_std(
                                "FE",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::ReResult(r) => {
                            models.push(extract_std(
                                "RE",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                0,
                            ));
                        }
                        Value::AbResult(r) => {
                            models.push(extract_std(
                                "AB-GMM",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::GmmResult(r) => {
                            let names: Option<Vec<String>> =
                                Some((0..r.params.len()).map(|i| format!("x{i}")).collect());
                            models.push(extract_std(
                                "GMM",
                                &names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::SysGmmResult(r) => {
                            models.push(extract_std(
                                "SysGMM",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs_fd,
                            ));
                        }
                        Value::PcseResult(r) => {
                            models.push(extract_std(
                                "PCSE",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::PanelGlsResult(r) => {
                            let label = match r.panels {
                                greeners::panel::GlsPanels::Hetero => "XTGLS-H",
                                greeners::panel::GlsPanels::Correlated => "XTGLS-C",
                            };
                            models.push(extract_std(
                                label,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::FE2SLSResult(r) => {
                            models.push(extract_std(
                                "FE-IV",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::QuantileResult(r) => {
                            let label = format!("QReg(τ={:.2})", r.tau);
                            models.push(extract_std(
                                &label,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                0,
                            ));
                        }
                        Value::CoxResult(r) => {
                            models.push(extract_std(
                                "CoxPH",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::RlmResult(r) => {
                            models.push(extract_std(
                                "RLM",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::GeeResult(r) => {
                            // GEE usa SE robusto (sandwich) por padrão
                            models.push(extract_std(
                                "GEE",
                                &r.variable_names,
                                &r.params,
                                &r.robust_se,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::BetaResult(r) => {
                            models.push(extract_std(
                                "BetaReg",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::GlmResult(r) => {
                            let family_name = format!("GLM({:?})", r.family);
                            models.push(extract_std(
                                &family_name,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::LowessResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta lowess — use predict para extrair valores suavizados".into()
                            ));
                        }
                        Value::PcaResult(_) | Value::FactorResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta PCA/Factor — use print() para ver cargas e variância explicada".into()
                            ));
                        }
                        Value::ConditionalResult(r) => {
                            models.push(extract_std(
                                &r.model_name,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::MarkovResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta Markov Switching — use print() para ver parâmetros por regime".into()
                            ));
                        }
                        Value::GlsarResult(r) => {
                            models.push(extract_std(
                                "GLSAR",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::MixedResult(r) => {
                            // esttab exibe apenas efeitos fixos do MixedLM
                            models.push(extract_std(
                                "MixedLM",
                                &r.variable_names,
                                &r.fixed_effects,
                                &r.fixed_se,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::ZeroInflatedResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta zip/zinb (duas equações) — use print()"
                                    .into(),
                            ));
                        }
                        Value::SurResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta sur (múltiplas equações) — use print()"
                                    .into(),
                            ));
                        }
                        Value::RollingResult(_) | Value::RecursiveLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta rolling/recursive — coeficientes variam ao longo do tempo; use print()".into()
                            ));
                        }
                        Value::MNLogitResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta mlogit (múltiplas equações) — use print()"
                                    .into(),
                            ));
                        }
                        Value::DidResult(_) | Value::KMResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta did/km — resultado tem formato próprio; use print()".into()
                            ));
                        }
                        Value::RdResult(_) | Value::SynthResult(_) | Value::PsmResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta estimadores causais (rd, psm, synth) — use print()".into()
                            ));
                        }
                        Value::VarmaResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta VARMA (coeficientes matriciais) — use print()".into()
                            ));
                        }
                        Value::DecompResult(_) | Value::MstlResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta decomposição sazonal — use print()".into(),
                            ));
                        }
                        Value::UCResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta UCM (parâmetros de variância, não β) — use print()".into()
                            ));
                        }
                        Value::GamResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta GAM (termos smooth não têm tabela β padrão) — use print()".into()
                            ));
                        }
                        Value::MiceResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta MICE (múltiplos datasets) — estime modelo em cada dataset e use Rubin's rules".into()
                            ));
                        }
                        Value::MSARResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta Markov-AR (parâmetros por regime) — use print()".into()
                            ));
                        }
                        Value::SVarResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta SVAR (matrizes A/B estruturais) — use print()".into()
                            ));
                        }
                        Value::ThreeSLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta 3SLS (múltiplas equações) — use print()"
                                    .into(),
                            ));
                        }
                        Value::DFMResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta DFM (fatores latentes) — use print()".into(),
                            ));
                        }
                        Value::EtsResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta ETS (parâmetros de suavização) — use print()".into()
                            ));
                        }
                        Value::ThresholdResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta panel threshold (dois regimes) — use print()".into()
                            ));
                        }
                        _ => {
                            return Err(HayashiError::Type(
                                "esttab(): tipo de modelo not supportado — use print()".into(),
                            ))
                        }
                    }
                }

                // união dos nomes de variáveis na ordem de primeira ocorrência
                let mut all_vars: Vec<String> = Vec::new();
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                for mi in &models {
                    let coefs = &mi.coefs;
                    for (nm, _, _, _) in coefs {
                        if seen.insert(nm.clone()) {
                            all_vars.push(nm.clone());
                        }
                    }
                }

                let n_models = models.len();
                let col_w = 16usize;
                let label_w = all_vars.iter().map(|s| s.len()).max().unwrap_or(8).max(12) + 2;
                let total_w = label_w + n_models * (col_w + 1);

                // monta conteúdo (txt ou latex)
                let mut buf = String::new();

                if fmt == "latex" || fmt == "tex" {
                    buf.push_str("\\begin{tabular}{l");
                    for _ in 0..n_models {
                        buf.push_str("r");
                    }
                    buf.push_str("}\n\\hline\\hline\n");
                    // cabeçalho
                    buf.push_str(" &");
                    for (i, mi) in models.iter().enumerate() {
                        let label = &mi.label;
                        buf.push_str(&format!(" ({}) {}", i + 1, label));
                        if i + 1 < n_models {
                            buf.push('&');
                        }
                    }
                    buf.push_str(" \\\\\n\\hline\n");

                    for var in &all_vars {
                        if var == "_cons" {
                            continue;
                        } // _cons vai no final
                        buf.push_str(&format!("{var}"));
                        for mi in &models {
                            let coefs = &mi.coefs;
                            let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                            match row {
                                Some((_, c, _, p)) => {
                                    buf.push_str(&format!(" & {:.4}{}", c, stars(*p)))
                                }
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                        // SE linha
                        let has_se = models.iter().any(|mi| {
                            mi.coefs
                                .iter()
                                .find(|(nm, _, _, _)| nm == var)
                                .and_then(|(_, _, se, _)| *se)
                                .is_some()
                        });
                        if has_se {
                            buf.push_str(" ");
                            for mi in &models {
                                let coefs = &mi.coefs;
                                let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                                match row.and_then(|(_, _, se, _)| *se) {
                                    Some(se) => buf.push_str(&format!(" & ({:.4})", se)),
                                    None => buf.push_str(" &"),
                                }
                            }
                            buf.push_str(" \\\\\n");
                        }
                    }
                    // _cons no final
                    if all_vars.iter().any(|v| v == "_cons") {
                        buf.push_str("Constant");
                        for mi in &models {
                            let coefs = &mi.coefs;
                            let row = coefs.iter().find(|(nm, _, _, _)| nm == "_cons");
                            match row {
                                Some((_, c, _, p)) => {
                                    buf.push_str(&format!(" & {:.4}{}", c, stars(*p)))
                                }
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                        let has_se = models.iter().any(|mi| {
                            mi.coefs
                                .iter()
                                .find(|(nm, _, _, _)| nm == "_cons")
                                .and_then(|(_, _, se, _)| *se)
                                .is_some()
                        });
                        if has_se {
                            buf.push_str(" ");
                            for mi in &models {
                                let coefs = &mi.coefs;
                                let row = coefs.iter().find(|(nm, _, _, _)| nm == "_cons");
                                match row.and_then(|(_, _, se, _)| *se) {
                                    Some(se) => buf.push_str(&format!(" & ({:.4})", se)),
                                    None => buf.push_str(" &"),
                                }
                            }
                            buf.push_str(" \\\\\n");
                        }
                    }
                    buf.push_str("\\hline\nN");
                    for mi in &models {
                        buf.push_str(&format!(" & {}", mi.n));
                    }
                    buf.push_str(" \\\\\n");
                    if models.iter().any(|mi| mi.r2.is_some()) {
                        buf.push_str("$R^2$");
                        for mi in &models {
                            match mi.r2 {
                                Some(v) => buf.push_str(&format!(" & {:.4}", v)),
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                    }
                    if models.iter().any(|mi| mi.adj_r2.is_some()) {
                        buf.push_str("Adj. $R^2$");
                        for mi in &models {
                            match mi.adj_r2 {
                                Some(v) => buf.push_str(&format!(" & {:.4}", v)),
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                    }
                    buf.push_str("\\hline\\hline\n\\end{tabular}\n");
                    buf.push_str("\\footnotesize{* p$<$0.10, ** p$<$0.05, *** p$<$0.01}\n");
                } else {
                    // ── ASCII txt ─────────────────────────────────────────────
                    let sep = "─".repeat(total_w);

                    // cabeçalho: numeração
                    let mut line = format!("{:<lw$}", "", lw = label_w);
                    for i in 0..n_models {
                        line.push_str(&format!(" {:>cw$}", format!("({})", i + 1), cw = col_w));
                    }
                    buf.push_str(&format!("{line}\n"));

                    // cabeçalho: labels
                    let mut line = format!("{:<lw$}", "", lw = label_w);
                    for mi in &models {
                        line.push_str(&format!(" {:>cw$}", mi.label, cw = col_w));
                    }
                    buf.push_str(&format!("{line}\n"));
                    buf.push_str(&format!("{sep}\n"));

                    let print_var = |var: &str, buf: &mut String| {
                        // linha de coeficientes
                        let display_name = if var == "_cons" { "Constant" } else { var };
                        let mut line = format!("{:<lw$}", display_name, lw = label_w);
                        for mi in &models {
                            let coefs = &mi.coefs;
                            let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                            match row {
                                Some((_, c, _, p)) => {
                                    let s = stars(*p);
                                    let cell = format!("{:.4}{}", c, s);
                                    line.push_str(&format!(" {:>cw$}", cell, cw = col_w));
                                }
                                None => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                            }
                        }
                        buf.push_str(&format!("{line}\n"));

                        // linha de erros padrão
                        let has_se = models.iter().any(|mi| {
                            mi.coefs
                                .iter()
                                .find(|(nm, _, _, _)| nm == var)
                                .and_then(|(_, _, se, _)| *se)
                                .is_some()
                        });
                        if has_se {
                            let mut line = format!("{:<lw$}", "", lw = label_w);
                            for mi in &models {
                                let coefs = &mi.coefs;
                                let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                                match row.and_then(|(_, _, se, _)| *se) {
                                    Some(se) => line.push_str(&format!(
                                        " {:>cw$}",
                                        format!("({:.4})", se),
                                        cw = col_w
                                    )),
                                    None => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                                }
                            }
                            buf.push_str(&format!("{line}\n"));
                        }
                    };

                    for var in &all_vars {
                        if var == "_cons" {
                            continue;
                        }
                        print_var(var, &mut buf);
                    }
                    if all_vars.iter().any(|v| v == "_cons") {
                        print_var("_cons", &mut buf);
                    }

                    buf.push_str(&format!("{sep}\n"));
                    let mut line = format!("{:<lw$}", "N", lw = label_w);
                    for mi in &models {
                        line.push_str(&format!(" {:>cw$}", mi.n, cw = col_w));
                    }
                    buf.push_str(&format!("{line}\n"));
                    if models.iter().any(|mi| mi.r2.is_some()) {
                        let mut line = format!("{:<lw$}", "R²", lw = label_w);
                        for mi in &models {
                            match mi.r2 {
                                Some(v) => line.push_str(&format!(" {:>cw$.4}", v, cw = col_w)),
                                None => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                            }
                        }
                        buf.push_str(&format!("{line}\n"));
                    }
                    if models.iter().any(|mi| mi.adj_r2.is_some()) {
                        let mut line = format!("{:<lw$}", "Adj. R²", lw = label_w);
                        for mi in &models {
                            match mi.adj_r2 {
                                Some(v) => line.push_str(&format!(" {:>cw$.4}", v, cw = col_w)),
                                None => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                            }
                        }
                        buf.push_str(&format!("{line}\n"));
                    }
                    buf.push_str(&format!("{sep}\n"));
                    buf.push_str("* p<0.10  ** p<0.05  *** p<0.01\n");
                }

                if let Some(path) = out_path {
                    std::fs::write(&path, &buf).map_err(|e| HayashiError::Io(e.to_string()))?;
                    println!("Exported table → '{path}'");
                } else {
                    print!("\n{buf}");
                }

                Ok(Value::Nil)
            }

            // ── Função definida pelo usuário ──────────────────────────────────
            other => {
                // scalar math: sqrt(4), ln(2.7), abs(-3), etc.
                if args.len() == 1 {
                    if let Ok(v) = self.eval_expr(&args[0]) {
                        let x = match &v {
                            Value::Float(f) => Some(*f),
                            Value::Int(i) => Some(*i as f64),
                            _ => None,
                        };
                        if let Some(x) = x {
                            if let Ok(res) = greeners::Transforms::apply(&[x], other) {
                                return Ok(Value::Float(res[0]));
                            }
                        }
                    }
                } else if args.len() == 2 {
                    if let (Ok(va), Ok(vb)) = (self.eval_expr(&args[0]), self.eval_expr(&args[1])) {
                        let xa = match &va {
                            Value::Float(f) => Some(*f),
                            Value::Int(i) => Some(*i as f64),
                            _ => None,
                        };
                        let xb = match &vb {
                            Value::Float(f) => Some(*f),
                            Value::Int(i) => Some(*i as f64),
                            _ => None,
                        };
                        if let (Some(a), Some(b)) = (xa, xb) {
                            if let Ok(res) = greeners::Transforms::apply2(&[a], &[b], other) {
                                return Ok(Value::Float(res[0]));
                            }
                        }
                    }
                }

                let user_fn = match self.env.get(other).cloned() {
                    Some(Value::UserFn(f)) => f,
                    _ => {
                        let mut known = self.env.all_names();
                        known.extend(BUILTIN_NAMES.iter().map(|s| s.to_string()));
                        let hint = Self::suggest(other, &known)
                            .map(|s| format!(" — did you mean '{s}'?"))
                            .unwrap_or_default();
                        return Err(self.rt_err(format!("undefined function '{other}'{hint}")));
                    }
                };

                if args.len() != user_fn.params.len() {
                    return Err(HayashiError::Runtime(format!(
                        "fn '{other}': esperado {} argumento(s), recebido {}",
                        user_fn.params.len(),
                        args.len()
                    )));
                }

                // Avalia argumentos antes de modificar o env
                let arg_vals: Vec<Value> = args
                    .iter()
                    .map(|e| self.eval_expr(e))
                    .collect::<Result<_>>()?;

                self.call_stack.push((other.to_string(), self.current_line));
                self.env.push_scope();
                for (param, val) in user_fn.params.iter().zip(arg_vals) {
                    self.env.declare_const(param, val);
                }

                let body = user_fn.body.clone();
                let mut exec_err: Option<HayashiError> = None;
                for s in &body {
                    match self.exec(s) {
                        Ok(()) => {}
                        Err(HayashiError::Return) => break,
                        Err(HayashiError::Break | HayashiError::Continue) => {
                            exec_err = Some(HayashiError::Runtime(
                                "break/continue outside of a loop".into(),
                            ));
                            break;
                        }
                        Err(e) => {
                            exec_err = Some(e);
                            break;
                        }
                    }
                }

                self.env.pop_scope();
                self.call_stack.pop();

                if let Some(e) = exec_err {
                    let frame = format!("  in {other}() at line {}", self.current_line);
                    let msg = format!("{e}");
                    let annotated = if msg.contains("Stack trace:") {
                        format!("{msg}\n{frame}")
                    } else {
                        format!("{msg}\nStack trace:\n{frame}")
                    };
                    return Err(HayashiError::Runtime(annotated));
                }

                Ok(self.return_value.take().unwrap_or(Value::Nil))
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn diag(rendered: String) -> Value {
        Value::DiagResult(Rc::new(DiagResult { rendered }))
    }

    // ── Helpers para aritmética / lógica escalar ──────────────────────────────

    fn value_as_bool(v: &Value) -> bool {
        match v {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0 && !f.is_nan(),
            Value::Nil => false,
            _ => true,
        }
    }

    fn extract_params(v: &Value) -> Option<Vec<f64>> {
        match v {
            Value::OlsResult(m) => Some(m.result.params.to_vec()),
            Value::BinaryResult(m) => Some(m.result.params.to_vec()),
            Value::PoissonResult(r) => Some(r.params.to_vec()),
            Value::NegBinResult(r) => Some(r.params.to_vec()),
            Value::QuantileResult(r) => Some(r.params.to_vec()),
            Value::PanelResult(r) => Some(r.params.to_vec()),
            Value::TobitResult(r) => Some(r.params.to_vec()),
            _ => None,
        }
    }

    fn extract_se(v: &Value) -> Option<Vec<f64>> {
        match v {
            Value::OlsResult(m) => Some(m.result.std_errors.to_vec()),
            Value::BinaryResult(m) => Some(m.result.std_errors.to_vec()),
            Value::PoissonResult(r) => Some(r.std_errors.to_vec()),
            Value::NegBinResult(r) => Some(r.std_errors.to_vec()),
            Value::QuantileResult(r) => Some(r.std_errors.to_vec()),
            Value::PanelResult(r) => Some(r.std_errors.to_vec()),
            Value::TobitResult(r) => Some(r.std_errors.to_vec()),
            _ => None,
        }
    }

    fn extract_var_names(v: &Value) -> Vec<String> {
        match v {
            Value::OlsResult(m) => m.result.variable_names.clone().unwrap_or_default(),
            Value::BinaryResult(m) => m.coef_names.clone(),
            Value::PoissonResult(r) => r.variable_names.clone().unwrap_or_default(),
            Value::NegBinResult(r) => r.variable_names.clone().unwrap_or_default(),
            Value::QuantileResult(r) => r.variable_names.clone().unwrap_or_default(),
            Value::PanelResult(r) => r.variable_names.clone().unwrap_or_default(),
            Value::TobitResult(r) => r.variable_names.clone().unwrap_or_default(),
            _ => vec![],
        }
    }

    fn value_as_f64(v: &Value) -> Result<f64> {
        match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            _ => Err(HayashiError::Type("expected numeric value".into())),
        }
    }

    fn eval_scalar_binop(op: &BinOp, l: Value, r: Value) -> Result<Value> {
        // Comparações (funciona com qualquer tipo comparável)
        match op {
            BinOp::Eq => {
                let eq = match (&l, &r) {
                    (Value::Nil, Value::Nil) => true,
                    (Value::Nil, _) | (_, Value::Nil) => false,
                    (Value::Str(a), Value::Str(b)) => a == b,
                    (Value::Bool(a), Value::Bool(b)) => a == b,
                    _ => {
                        let a = Self::value_as_f64(&l)?;
                        let b = Self::value_as_f64(&r)?;
                        (a - b).abs() < f64::EPSILON
                    }
                };
                return Ok(Value::Bool(eq));
            }
            BinOp::Ne => {
                let ne = match (&l, &r) {
                    (Value::Nil, Value::Nil) => false,
                    (Value::Nil, _) | (_, Value::Nil) => true,
                    (Value::Str(a), Value::Str(b)) => a != b,
                    (Value::Bool(a), Value::Bool(b)) => a != b,
                    _ => {
                        let a = Self::value_as_f64(&l)?;
                        let b = Self::value_as_f64(&r)?;
                        (a - b).abs() >= f64::EPSILON
                    }
                };
                return Ok(Value::Bool(ne));
            }
            _ => {}
        }

        // Aritmética e comparações numéricas
        match (&l, &r) {
            // Int × Int → Int (para Add/Sub/Mul); Div/Pow → Float
            (Value::Int(a), Value::Int(b)) => match op {
                BinOp::Add => Ok(Value::Int(a + b)),
                BinOp::Sub => Ok(Value::Int(a - b)),
                BinOp::Mul => Ok(Value::Int(a * b)),
                BinOp::Div => Ok(Value::Float(*a as f64 / *b as f64)),
                BinOp::Mod => Ok(Value::Int(a % b)),
                BinOp::Pow => Ok(Value::Float((*a as f64).powf(*b as f64))),
                BinOp::Gt => Ok(Value::Bool(a > b)),
                BinOp::Lt => Ok(Value::Bool(a < b)),
                BinOp::GtEq => Ok(Value::Bool(a >= b)),
                BinOp::LtEq => Ok(Value::Bool(a <= b)),
                BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Ne | BinOp::In => unreachable!(),
            },
            // Qualquer Float → Float
            _ => {
                // Concatenação de strings
                if let (BinOp::Add, Value::Str(a), Value::Str(b)) = (op, &l, &r) {
                    return Ok(Value::Str(format!("{a}{b}")));
                }
                let a = Self::value_as_f64(&l)?;
                let b = Self::value_as_f64(&r)?;
                match op {
                    BinOp::Add => Ok(Value::Float(a + b)),
                    BinOp::Sub => Ok(Value::Float(a - b)),
                    BinOp::Mul => Ok(Value::Float(a * b)),
                    BinOp::Div => Ok(Value::Float(a / b)),
                    BinOp::Mod => Ok(Value::Float(a % b)),
                    BinOp::Pow => Ok(Value::Float(a.powf(b))),
                    BinOp::Gt => Ok(Value::Bool(a > b)),
                    BinOp::Lt => Ok(Value::Bool(a < b)),
                    BinOp::GtEq => Ok(Value::Bool(a >= b)),
                    BinOp::LtEq => Ok(Value::Bool(a <= b)),
                    BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Ne | BinOp::In => unreachable!(),
                }
            }
        }
    }

    fn extract_panel_args(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<(Formula, Rc<DataFrame>, String, String)> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "panel estimator requires (formula, dataframe [, id=col])".into(),
            ));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
        let df_name = match &args[1] {
            Expr::Var(name) => name.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "second argument must be a DataFrame variable".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(id, _)| id.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    HayashiError::Runtime(format!(
                        "panel estimator requires id=col or xtset({df_name}, id, time) first"
                    ))
                })?,
        };
        Ok((formula_ast, df, df_name, id_col))
    }

    fn get_time_col(&self, df_name: &str, opt_map: &HashMap<String, Value>) -> Result<String> {
        match opt_map.get("time") {
            Some(Value::Str(s)) => Ok(s.clone()),
            _ => self
                .panel_info
                .get(df_name)
                .map(|(_, t)| t.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    HayashiError::Runtime(format!(
                        "panel estimator requires time=col or xtset({df_name}, id, time) first"
                    ))
                }),
        }
    }

    /// Extrai uma coluna como Vec<i64> — aceita colunas Int ou Float.
    fn col_as_i64(
        df: &DataFrame,
        col: &str,
    ) -> std::result::Result<Vec<i64>, greeners::GreenersError> {
        if let Ok(ids) = df.get_int(col) {
            Ok(ids.to_vec())
        } else if let Ok(floats) = df.get(col) {
            Ok(floats.iter().map(|&v| v as i64).collect())
        } else {
            Err(greeners::GreenersError::VariableNotFound(col.to_string()))
        }
    }

    fn finite_numeric_string(value: &str) -> Option<f64> {
        value.parse::<f64>().ok().filter(|v| v.is_finite())
    }

    fn sort_maybe_numeric_strings(values: &mut [String]) {
        if values
            .iter()
            .all(|value| Self::finite_numeric_string(value).is_some())
        {
            values.sort_by(|a, b| {
                match (
                    Self::finite_numeric_string(a),
                    Self::finite_numeric_string(b),
                ) {
                    (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                    _ => a.cmp(b),
                }
            });
        } else {
            values.sort();
        }
    }

    // ── Nomes dos coeficientes a partir da fórmula ────────────────────────────

    // Ordena um DataFrame por uma única coluna (ascendente).
    // Usado por tsset para garantir ordem temporal.
    fn sort_df_by(df: &DataFrame, col: &str) -> Result<DataFrame> {
        use greeners::Column;
        let n = df.n_rows();

        // índice de ordenação pela coluna t_var
        let mut idx: Vec<usize> = (0..n).collect();
        match df.get_column(col) {
            Ok(Column::Float(arr)) => {
                let v = arr.to_vec();
                idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
            }
            Ok(Column::Int(arr)) => {
                let v: Vec<f64> = arr.iter().map(|&x| x as f64).collect();
                idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
            }
            _ => {
                if let Ok(arr) = df.get_string(col) {
                    let v = arr.to_vec();
                    idx.sort_by(|&a, &b| v[a].cmp(&v[b]));
                } else {
                    return Err(HayashiError::Runtime(format!("column '{col}' not found")));
                }
            }
        }

        let mut builder = DataFrame::builder();
        for name in &df.column_names() {
            match df.get_column(name) {
                Ok(Column::Float(arr)) => {
                    builder =
                        builder.add_column(name, idx.iter().map(|&i| arr[i]).collect::<Vec<_>>());
                }
                Ok(Column::Int(arr)) => {
                    builder = builder
                        .add_column(name, idx.iter().map(|&i| arr[i] as f64).collect::<Vec<_>>());
                }
                _ => {
                    if let Ok(arr) = df.get_string(name) {
                        let v = arr.to_vec();
                        builder =
                            builder.add_string(name, idx.iter().map(|&i| v[i].clone()).collect());
                    }
                }
            }
        }
        builder
            .build()
            .map_err(|e| HayashiError::Runtime(e.to_string()))
    }

    fn coef_names_from_formula(
        formula_ast: &Formula,
        df: &DataFrame,
        n_cols: usize,
    ) -> Vec<String> {
        let mut names: Vec<String> = vec!["_cons".into()];
        for term in &formula_ast.rhs {
            match term {
                RhsTerm::Var(v) => names.push(v.clone()),
                RhsTerm::Transform(fn_, v) => names.push(format!("{fn_}({v})")),
                RhsTerm::Interaction(a, b) => names.push(format!("{a}:{b}")),
                RhsTerm::Categorical(v) => {
                    let raw = Self::col_to_strings(df, v).unwrap_or_default();
                    let mut unique: Vec<String> = raw
                        .into_iter()
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    Self::sort_maybe_numeric_strings(&mut unique);
                    for val in unique.into_iter().skip(1) {
                        names.push(format!("{v}={val}"));
                    }
                }
            }
        }
        names.truncate(n_cols);
        while names.len() < n_cols {
            names.push(format!("x{}", names.len() + 1));
        }
        names
    }

    // ── Extrai coluna como Vec<String> (para tabulate) ────────────────────────

    fn col_to_strings(df: &DataFrame, name: &str) -> Result<Vec<String>> {
        use greeners::Column;
        match df.get_column(name) {
            Ok(Column::Int(arr)) => Ok(arr.iter().map(|x| x.to_string()).collect()),
            Ok(Column::Float(arr)) => Ok(arr
                .iter()
                .map(|x| {
                    if x.is_nan() {
                        ".".to_string()
                    } else if x.fract() == 0.0 && x.abs() < 1e14 {
                        format!("{}", *x as i64)
                    } else {
                        format!("{:.4}", x)
                    }
                })
                .collect()),
            Ok(Column::String(arr)) => Ok(arr.to_vec()),
            Ok(Column::Categorical(cat)) => Ok((0..df.n_rows())
                .map(|row| cat.get_string(row).unwrap_or("").to_string())
                .collect()),
            _ => df.get_string(name).map(|arr| arr.to_vec()).map_err(|_| {
                HayashiError::Runtime(format!(
                    "column '{name}' not found or has unsupported type for tabulate"
                ))
            }),
        }
    }

    // ── Tabela de frequências (uni-variada) ───────────────────────────────────

    fn tabulate_one(df: &DataFrame, var: &str) -> Result<()> {
        let vals = Self::col_to_strings(df, var)?;
        let n = vals.len();

        let mut counts: HashMap<String, usize> = HashMap::new();
        for v in &vals {
            *counts.entry(v.clone()).or_insert(0) += 1;
        }

        let mut unique: Vec<String> = counts.keys().cloned().collect();
        Self::sort_maybe_numeric_strings(&mut unique);

        let label_w = var
            .len()
            .max(12)
            .max(unique.iter().map(|s| s.len()).max().unwrap_or(0))
            + 2;
        let sep = format!("{}-+{}", "-".repeat(label_w), "-".repeat(36));

        println!(
            "\n{:>lw$} | {:>10}  {:>10}  {:>10}",
            var,
            "Freq.",
            "Percent",
            "Cum.",
            lw = label_w
        );
        println!("{sep}");

        let mut cum = 0.0_f64;
        for key in &unique {
            let freq = counts[key];
            let pct = freq as f64 / n as f64 * 100.0;
            cum += pct;
            println!(
                "{:>lw$} | {:>10}  {:>10.2}  {:>10.2}",
                key,
                freq,
                pct,
                cum,
                lw = label_w
            );
        }
        println!("{sep}");
        println!(
            "{:>lw$} | {:>10}  {:>10.2}",
            "Total",
            n,
            100.0_f64,
            lw = label_w
        );
        println!();
        Ok(())
    }

    // ── Tabela cruzada (bi-variada, opcional chi2) ────────────────────────────

    fn tabulate_two(df: &DataFrame, row_var: &str, col_var: &str, do_chi2: bool) -> Result<()> {
        let rows = Self::col_to_strings(df, row_var)?;
        let cols = Self::col_to_strings(df, col_var)?;

        if rows.len() != cols.len() {
            return Err(HayashiError::Runtime(
                "columns have different lengths".into(),
            ));
        }

        // valores únicos, ordenados
        let sort_strs = |mut v: Vec<String>| -> Vec<String> {
            Self::sort_maybe_numeric_strings(&mut v);
            v
        };

        let mut row_set: Vec<String> = rows
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        row_set = sort_strs(row_set);
        let mut col_set: Vec<String> = cols
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        col_set = sort_strs(col_set);

        // contagens
        let mut cell: HashMap<(String, String), usize> = HashMap::new();
        for (r, c) in rows.iter().zip(cols.iter()) {
            *cell.entry((r.clone(), c.clone())).or_insert(0) += 1;
        }

        let n = rows.len();
        let col_totals: Vec<usize> = col_set
            .iter()
            .map(|c| {
                row_set
                    .iter()
                    .map(|r| *cell.get(&(r.clone(), c.clone())).unwrap_or(&0))
                    .sum()
            })
            .collect();
        let row_totals: Vec<usize> = row_set
            .iter()
            .map(|r| {
                col_set
                    .iter()
                    .map(|c| *cell.get(&(r.clone(), c.clone())).unwrap_or(&0))
                    .sum()
            })
            .collect();

        // larguras de coluna
        let cell_w = 10usize;
        let row_lw = row_var
            .len()
            .max(12)
            .max(row_set.iter().map(|s| s.len()).max().unwrap_or(0))
            + 2;
        let col_head_w = col_set.len() * (cell_w + 1) + 1;
        let total_w = cell_w + 2;

        // linha de cabeçalho do col_var
        println!(
            "\n{:>rw$} | {:^chw$}| {:>tw$}",
            "",
            col_var,
            "Total",
            rw = row_lw,
            chw = col_head_w,
            tw = total_w
        );

        // linha com os valores das colunas
        print!("{:>rw$} |", row_var, rw = row_lw);
        for cv in &col_set {
            print!(" {:>cw$}", cv, cw = cell_w);
        }
        println!(" | {:>cw$}", "Total", cw = cell_w);

        let sep = format!(
            "{}-+{}-+{}",
            "-".repeat(row_lw),
            "-".repeat(col_head_w),
            "-".repeat(total_w)
        );
        println!("{sep}");

        for (i, rv) in row_set.iter().enumerate() {
            print!("{:>rw$} |", rv, rw = row_lw);
            for cv in &col_set {
                let cnt = *cell.get(&(rv.clone(), cv.clone())).unwrap_or(&0);
                print!(" {:>cw$}", cnt, cw = cell_w);
            }
            println!(" | {:>cw$}", row_totals[i], cw = cell_w);
        }

        println!("{sep}");
        print!("{:>rw$} |", "Total", rw = row_lw);
        for ct in &col_totals {
            print!(" {:>cw$}", ct, cw = cell_w);
        }
        println!(" | {:>cw$}", n, cw = cell_w);
        println!();

        if do_chi2 {
            let mut stat = 0.0_f64;
            for (i, rv) in row_set.iter().enumerate() {
                for (j, cv) in col_set.iter().enumerate() {
                    let obs = *cell.get(&(rv.clone(), cv.clone())).unwrap_or(&0) as f64;
                    let exp = row_totals[i] as f64 * col_totals[j] as f64 / n as f64;
                    if exp > 0.0 {
                        stat += (obs - exp).powi(2) / exp;
                    }
                }
            }
            let df = (row_set.len() - 1) * (col_set.len() - 1);
            let p = chi2_pvalue(stat, df as f64);
            println!("  Pearson chi2({df}) = {stat:.4}   Pr = {p:.4}");
            println!();
        }

        Ok(())
    }

    // ── Helpers de visualização ASCII ────────────────────────────────────────

    fn ascii_histogram(data: &[f64], bins: usize, title: &str, var: &str, width: usize) {
        if data.is_empty() {
            println!("  (sem dados)");
            return;
        }
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if (max - min).abs() < 1e-15 {
            println!("  (variância zero)");
            return;
        }
        let step = (max - min) / bins as f64;
        let mut counts = vec![0usize; bins];
        for &v in data {
            let idx = ((v - min) / step).floor() as usize;
            let idx = idx.min(bins - 1);
            counts[idx] += 1;
        }
        let max_count = *counts.iter().max().unwrap_or(&1);
        let bar_w = width.max(10);
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;
        let sd = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        println!();
        println!("{:=^width$}", format!(" {title} "), width = bar_w + 34);
        println!("  Variável: {var}   n={n}   μ={mean:.4}   σ={sd:.4}   [{min:.4}, {max:.4}]");
        println!("{:-^width$}", "", width = bar_w + 34);
        for (i, &cnt) in counts.iter().enumerate() {
            let lo = min + i as f64 * step;
            let hi = lo + step;
            let bar_len = if max_count > 0 {
                cnt * bar_w / max_count
            } else {
                0
            };
            let bar: String = "█".repeat(bar_len);
            println!(
                "  [{:>10.4},{:>10.4})  {:>5}  {:<width$}",
                lo,
                hi,
                cnt,
                bar,
                width = bar_w
            );
        }
        println!("{:-^width$}", "", width = bar_w + 34);
        println!();
    }

    fn ascii_scatter(
        xs: &[f64],
        ys: &[f64],
        title: &str,
        xlab: &str,
        ylab: &str,
        w: usize,
        h: usize,
    ) {
        if xs.is_empty() {
            println!("  (sem dados)");
            return;
        }
        let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let xrng = (xmax - xmin).max(1e-15);
        let yrng = (ymax - ymin).max(1e-15);
        let mut grid = vec![vec![' '; w]; h];
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            if x.is_nan() || y.is_nan() {
                continue;
            }
            let col = ((x - xmin) / xrng * (w - 1) as f64).round() as usize;
            let row = h - 1 - ((y - ymin) / yrng * (h - 1) as f64).round() as usize;
            let col = col.min(w - 1);
            let row = row.min(h - 1);
            grid[row][col] = '·';
        }
        println!();
        println!("{:=^width$}", format!(" {title} "), width = w + 18);
        println!("  {:<10}  {:>10.4} ┐", ylab, ymax);
        for (i, row) in grid.iter().enumerate() {
            let y_val = ymax - (i as f64 / (h - 1) as f64) * yrng;
            let prefix = if i == 0 || i == h / 2 || i == h - 1 {
                format!("  {:>10.4} │", y_val)
            } else {
                format!("             │")
            };
            let line: String = row.iter().collect();
            println!("{prefix}{line}");
        }
        println!("             └{}", "─".repeat(w));
        let mid_x = xmin + xrng / 2.0;
        println!(
            "              {:<10.4}{:^width$.4}{:>10.4}",
            xmin,
            mid_x,
            xmax,
            width = w - 20
        );
        println!("              {:^width$}", xlab, width = w);
        println!("  n={}", xs.len());
        println!();
    }

    fn ascii_lineplot(
        xs: &[f64],
        ys: &[f64],
        title: &str,
        xlab: &str,
        ylab: &str,
        w: usize,
        h: usize,
    ) {
        if xs.is_empty() {
            println!("  (sem dados)");
            return;
        }
        let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let xrng = (xmax - xmin).max(1e-15);
        let yrng = (ymax - ymin).max(1e-15);
        // Sort by x
        let mut pairs: Vec<(f64, f64)> = xs
            .iter()
            .zip(ys.iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut grid = vec![vec![' '; w]; h];
        let mut prev_col: Option<(usize, usize)> = None;
        for &(x, y) in &pairs {
            let col = ((x - xmin) / xrng * (w - 1) as f64).round() as usize;
            let row = h - 1 - ((y - ymin) / yrng * (h - 1) as f64).round() as usize;
            let col = col.min(w - 1);
            let row = row.min(h - 1);
            if let Some((pr, pc)) = prev_col {
                // Fill between previous and current column
                if pc < col {
                    for c in pc..=col {
                        let t = (c - pc) as f64 / (col - pc).max(1) as f64;
                        let r = (pr as f64 + t * (row as f64 - pr as f64)).round() as usize;
                        let r = r.min(h - 1);
                        if grid[r][c] == ' ' {
                            grid[r][c] = '─';
                        }
                    }
                }
            }
            grid[row][col] = '●';
            prev_col = Some((row, col));
        }
        println!();
        println!("{:=^width$}", format!(" {title} "), width = w + 18);
        println!("  {:<10}  {:>10.4} ┐", ylab, ymax);
        for (i, row) in grid.iter().enumerate() {
            let y_val = ymax - (i as f64 / (h - 1) as f64) * yrng;
            let prefix = if i == 0 || i == h / 2 || i == h - 1 {
                format!("  {:>10.4} │", y_val)
            } else {
                format!("             │")
            };
            let line: String = row.iter().collect();
            println!("{prefix}{line}");
        }
        println!("             └{}", "─".repeat(w));
        let mid_x = xmin + xrng / 2.0;
        println!(
            "              {:<10.4}{:^width$.4}{:>10.4}",
            xmin,
            mid_x,
            xmax,
            width = w - 20
        );
        println!("              {:^width$}", xlab, width = w);
        println!("  n={}", pairs.len());
        println!();
    }

    fn ascii_boxplot(data: &[f64], title: &str, var: &str, w: usize) {
        if data.is_empty() {
            println!("  (sem dados)");
            return;
        }
        let mut sorted = data.to_vec();
        sorted.retain(|v| !v.is_nan());
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n < 4 {
            println!("  (poucos dados para boxplot)");
            return;
        }
        let q = |p: f64| -> f64 {
            let idx = p * (n - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil().min((n - 1) as f64) as usize;
            sorted[lo] + (idx - lo as f64) * (sorted[hi] - sorted[lo])
        };
        let mn = sorted[0];
        let q1 = q(0.25);
        let med = q(0.50);
        let q3 = q(0.75);
        let mx = sorted[n - 1];
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let iqr = q3 - q1;
        let fence_lo = q1 - 1.5 * iqr;
        let fence_hi = q3 + 1.5 * iqr;
        let whisker_lo = sorted
            .iter()
            .cloned()
            .filter(|&v| v >= fence_lo)
            .fold(f64::INFINITY, f64::min);
        let whisker_hi = sorted
            .iter()
            .cloned()
            .filter(|&v| v <= fence_hi)
            .fold(f64::NEG_INFINITY, f64::max);
        let outliers: Vec<f64> = sorted
            .iter()
            .cloned()
            .filter(|&v| v < fence_lo || v > fence_hi)
            .collect();

        let rng = (mx - mn).max(1e-15);
        let to_col =
            |v: f64| -> usize { (((v - mn) / rng * (w - 1) as f64).round() as usize).min(w - 1) };
        let c_wlo = to_col(whisker_lo);
        let c_q1 = to_col(q1);
        let c_med = to_col(med);
        let c_q3 = to_col(q3);
        let c_whi = to_col(whisker_hi);

        // Build boxplot line
        let mut line = vec![' '; w];
        for c in c_wlo..=c_whi {
            line[c] = '─';
        }
        for c in c_q1..=c_q3 {
            line[c] = '█';
        }
        line[c_wlo] = '├';
        line[c_whi] = '┤';
        line[c_q1] = '▐';
        line[c_q3] = '▌';
        line[c_med] = '|';
        for &v in &outliers {
            let c = to_col(v);
            line[c] = '○';
        }

        println!();
        println!("{:=^width$}", format!(" {title} "), width = w + 18);
        println!("  Variável: {var}   n={n}");
        println!();
        println!("             {}", line.iter().collect::<String>());
        println!();
        println!(
            "  Min:    {:>12.4}   Q1:  {:>12.4}   Mediana: {:>12.4}",
            whisker_lo, q1, med
        );
        println!(
            "  Média:  {:>12.4}   Q3:  {:>12.4}   Max:     {:>12.4}",
            mean, q3, whisker_hi
        );
        println!("  IQR:    {:>12.4}   Outliers: {}", iqr, outliers.len());
        if !outliers.is_empty() && outliers.len() <= 10 {
            let out_str: Vec<String> = outliers.iter().map(|v| format!("{:.3}", v)).collect();
            println!("  Valores: [{}]", out_str.join(", "));
        }
        println!();
    }

    // ── Φ(x) normal CDF — Abramowitz & Stegun 26.2.17 (erro < 7.5e-8) ───────
    fn norm_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        let phi = 1.0 - greeners::norm_pdf(x) * poly;
        if x >= 0.0 {
            phi
        } else {
            1.0 - phi
        }
    }

    // ── ACF / PACF como barras ASCII ─────────────────────────────────────────
    fn ascii_acf(data: &[f64], max_lag: usize, title: &str, width: usize, partial: bool) {
        let n = data.len();
        if n < 4 {
            println!("(dados insuficientes para ACF)");
            return;
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        if var < 1e-15 {
            println!("(variância zero)");
            return;
        }

        // Calcula autocorrelações completas
        let max_lag = max_lag.min(n / 2);
        let acf: Vec<f64> = (0..=max_lag)
            .map(|k| {
                let s: f64 = (0..n - k)
                    .map(|i| (data[i] - mean) * (data[i + k] - mean))
                    .sum();
                s / (n as f64 * var)
            })
            .collect();

        // PACF via algoritmo de Yule-Walker (Durbin-Levinson)
        let values: Vec<f64> = if partial {
            let mut pacf = vec![0.0f64; max_lag + 1];
            pacf[0] = 1.0;
            if max_lag >= 1 {
                pacf[1] = acf[1];
            }
            let mut phi: Vec<Vec<f64>> = vec![vec![0.0; max_lag + 1]; max_lag + 1];
            phi[1][1] = acf[1];
            for k in 2..=max_lag {
                let num: f64 = acf[k] - (1..k).map(|j| phi[k - 1][j] * acf[k - j]).sum::<f64>();
                let den: f64 = 1.0 - (1..k).map(|j| phi[k - 1][j] * acf[j]).sum::<f64>();
                let phi_kk = if den.abs() < 1e-15 { 0.0 } else { num / den };
                phi[k][k] = phi_kk;
                for j in 1..k {
                    phi[k][j] = phi[k - 1][j] - phi_kk * phi[k - 1][k - j];
                }
                pacf[k] = phi_kk;
            }
            pacf
        } else {
            acf.clone()
        };

        let ci = 1.96 / (n as f64).sqrt(); // banda de confiança a 95%
        println!("\n{:=<width$}", "");
        println!(" {title}");
        println!("{:=<width$}", "");
        let half = width / 2;
        for lag in 1..=max_lag {
            let v = values[lag];
            let bar_len = ((v.abs() * half as f64).round() as usize).min(half);
            let in_ci = v.abs() <= ci;
            let bar_char = if in_ci { '─' } else { '█' };
            let bar: String = std::iter::repeat(bar_char).take(bar_len).collect();
            let (left, right) = if v >= 0.0 {
                (format!("{:<half$}", " "), format!("{}", bar))
            } else {
                let pad = half - bar_len;
                (format!("{:>half$}", bar), " ".repeat(pad))
            };
            println!("{:3} |{}|{} {:6.3}", lag, left, right, v);
        }
        println!("{:=<width$}", "");
        println!("  CI ±{:.3} (95%)  │ ── dentro  █ fora", ci);
        println!();
    }

    // ── QQ-plot normal ────────────────────────────────────────────────────────
    fn ascii_qqplot(data: &[f64], title: &str, var: &str, w: usize, h: usize) {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n < 4 {
            println!("(dados insuficientes para QQ-plot)");
            return;
        }
        // Quantis teóricos normais por aproximação de Blom: p_i = (i - 3/8) / (n + 1/4)
        let theoretical: Vec<f64> = (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.375) / (n as f64 + 0.25);
                // Aproximação de Peter Acklam para invnorm (erro < 3.5e-4)
                let q = p - 0.5;
                let r = if q.abs() <= 0.425 {
                    let a = [
                        3.3871328_f64,
                        133.14166789,
                        1971.5909503,
                        13731.693765,
                        45921.953931,
                        67265.770927,
                        33430.575583,
                        2509.0809287,
                    ];
                    let b = [
                        1.0_f64,
                        42.313330701,
                        687.18700749,
                        5394.1960214,
                        21213.794301,
                        39307.895800,
                        28729.085735,
                        5226.4952788,
                    ];
                    let q2 = q * q;
                    let num = a
                        .iter()
                        .enumerate()
                        .fold(0.0, |s, (i, &c)| s + c * q2.powi(i as i32));
                    let den = b
                        .iter()
                        .enumerate()
                        .fold(0.0, |s, (i, &c)| s + c * q2.powi(i as i32));
                    q * num / den
                } else {
                    let pp = if q < 0.0 { p } else { 1.0 - p };
                    let r = (-pp.ln()).sqrt();
                    let c = if r <= 5.0 {
                        [
                            1.42343711_f64,
                            4.63033784,
                            5.76082150,
                            1.42343711,
                            1.63155402,
                            0.07027109,
                        ]
                    } else {
                        [
                            6.65790464_f64,
                            5.46378491,
                            1.78482653,
                            0.05697114,
                            0.18127138,
                            0.00778070,
                        ]
                    };
                    let num = c[0] + r * (c[1] + r * c[2]);
                    let den = 1.0 + r * (c[3] + r * (c[4] + r * c[5]));
                    if q < 0.0 {
                        -(num / den)
                    } else {
                        num / den
                    }
                };
                r
            })
            .collect();
        let mean_s = sorted.iter().sum::<f64>() / n as f64;
        let std_s = (sorted.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-15);
        // Standarizar os quantis empíricos
        let empirical: Vec<f64> = sorted.iter().map(|x| (x - mean_s) / std_s).collect();
        println!("\n{:=<w$}", "");
        println!(" {title}  (normalizado)");
        println!("{:=<w$}", "");
        Self::ascii_scatter(
            &theoretical,
            &empirical,
            title,
            "quantil teórico",
            var,
            w,
            h,
        );
        // Linha de referência (y = x): já visível no scatter se os dados são normais
        println!("  (linha ideal: pontos ao longo da diagonal)");
    }

    // ── Matriz de correlação como heatmap de texto ────────────────────────────
    fn ascii_corrplot(cols: &[Vec<f64>], names: &[String]) {
        let k = cols.len();
        let n = cols[0].len();
        let means: Vec<f64> = cols
            .iter()
            .map(|c| c.iter().sum::<f64>() / n as f64)
            .collect();
        // Calcula correlações
        let mut corr = vec![vec![0.0f64; k]; k];
        for i in 0..k {
            for j in 0..k {
                let xi: Vec<f64> = cols[i].iter().map(|x| x - means[i]).collect();
                let xj: Vec<f64> = cols[j].iter().map(|x| x - means[j]).collect();
                let num: f64 = xi.iter().zip(&xj).map(|(a, b)| a * b).sum();
                let di: f64 = xi.iter().map(|a| a * a).sum::<f64>().sqrt();
                let dj: f64 = xj.iter().map(|b| b * b).sum::<f64>().sqrt();
                corr[i][j] = if di * dj < 1e-15 {
                    0.0
                } else {
                    num / (di * dj)
                };
            }
        }
        // Largura do nome
        let nw = names.iter().map(|n| n.len()).max().unwrap_or(4).max(4);
        // Cabeçalho
        println!("\n{:=<80}", "");
        println!(" Matriz de Correlação");
        println!("{:=<80}", "");
        print!("{:>nw$}", "");
        for n in names {
            print!(" {:>7}", &n[..n.len().min(7)]);
        }
        println!();
        // Linhas
        for i in 0..k {
            print!("{:>nw$}", &names[i][..names[i].len().min(nw)]);
            for j in 0..k {
                let v = corr[i][j];
                // Representação por blocos: ████ para |r|=1, ░░░░ para r≈0
                let shade = if v.abs() >= 0.9 {
                    "████"
                } else if v.abs() >= 0.7 {
                    "▓▓▓▓"
                } else if v.abs() >= 0.5 {
                    "▒▒▒▒"
                } else if v.abs() >= 0.3 {
                    "░░░░"
                } else {
                    "    "
                };
                let sign = if v < 0.0 { "-" } else { "+" };
                print!(" {sign}{shade}",);
            }
            print!("   ");
            for j in 0..k {
                print!(" {:>6.3}", corr[i][j]);
            }
            println!();
        }
        println!("{:=<80}", "");
        println!("  Escala: ████ |r|≥0.9  ▓▓▓▓ ≥0.7  ▒▒▒▒ ≥0.5  ░░░░ ≥0.3  (+neg=-)");
        println!();
    }

    fn resolve_formula(&mut self, expr: &Expr) -> Result<Formula> {
        match expr {
            Expr::Formula(f) => Ok(f.clone()),
            other => {
                let val = self.eval_expr(other)?;
                match val {
                    Value::Str(s) => {
                        let parts: Vec<&str> = s.splitn(2, '~').collect();
                        if parts.len() != 2 {
                            return Err(self.type_err(format!(
                                "string '{s}' is not a valid formula (needs ~)"
                            )));
                        }
                        let lhs = parts[0].trim().to_string();
                        let rhs_str = parts[1].trim();
                        let rhs: Vec<RhsTerm> = rhs_str
                            .split('+')
                            .map(|t| RhsTerm::Var(t.trim().to_string()))
                            .collect();
                        Ok(Formula {
                            lhs,
                            rhs,
                            fe: vec![],
                        })
                    }
                    _ => Err(HayashiError::Type(
                        "first argument must be a formula or string".into(),
                    )),
                }
            }
        }
    }

    fn extract_binary_args_filtered(
        &mut self,
        args: &[Expr],
        opts: &[Opt],
    ) -> Result<(Formula, Rc<DataFrame>)> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "estimator requires (formula, dataframe)".into(),
            ));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
        let df_name = match &args[1] {
            Expr::Var(name) => name.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "second argument must be a DataFrame variable".into(),
                ))
            }
        };
        let df_raw = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let df = self.maybe_filter_df(&df_raw, opts)?;
        Ok((formula_ast, df))
    }

    // ── Métodos de objetos ────────────────────────────────────────────────────

    fn eval_field(
        &mut self,
        obj: &Expr,
        field: &str,
        _args: &[Expr],
        _opts: &[Opt],
    ) -> Result<Value> {
        let val = self.eval_expr(obj)?;
        match (&val, field) {
            (Value::OlsResult(m), "summary") => {
                println!("{}", m.result);
                Ok(Value::Nil)
            }
            (Value::IvResult(r), "summary") => {
                println!("{r}");
                Ok(Value::Nil)
            }
            (Value::BinaryResult(m), "summary") => {
                println!("{m}");
                Ok(Value::Nil)
            }
            (Value::PanelResult(r), "summary") => {
                println!("{r}");
                Ok(Value::Nil)
            }
            (Value::ReResult(r), "summary") => {
                println!("{r}");
                Ok(Value::Nil)
            }
            (_, f) => Err(self.rt_err(format!("unknown method '{f}'"))),
        }
    }

    // ── Avalia expressão elemento-a-elemento sobre colunas de um DataFrame ───

    fn eval_col_expr(&mut self, expr: &Expr, df: &DataFrame) -> Result<Vec<f64>> {
        match expr {
            Expr::Float(v) => {
                let n = df.n_rows();
                Ok(vec![*v; n])
            }
            Expr::Int(v) => {
                let n = df.n_rows();
                Ok(vec![*v as f64; n])
            }
            Expr::Bool(v) => {
                let n = df.n_rows();
                Ok(vec![if *v { 1.0 } else { 0.0 }; n])
            }
            Expr::Str(s) => {
                Err(HayashiError::Type(format!(
                    "string literal \"{s}\" cannot be used as numeric — se comparando com coluna string, use: col == \"{s}\""
                )))
            }
            Expr::Nil => {
                let n = df.n_rows();
                Ok(vec![f64::NAN; n])
            }
            Expr::Var(name) => {
                // _n = row number (1-based), _N = total rows
                if name == "_n" {
                    return Ok((1..=df.n_rows()).map(|i| i as f64).collect());
                }
                if name == "_N" {
                    return Ok(vec![df.n_rows() as f64; df.n_rows()]);
                }
                match df.get_column(name) {
                    Ok(col) => Ok(col.to_float().to_vec()),
                    Err(_) => match self.env.get(name) {
                        Some(Value::Float(f)) => Ok(vec![*f; df.n_rows()]),
                        Some(Value::Int(i)) => Ok(vec![*i as f64; df.n_rows()]),
                        Some(Value::Bool(b)) => Ok(vec![if *b { 1.0 } else { 0.0 }; df.n_rows()]),
                        Some(Value::List(lst)) => {
                            if lst.len() != df.n_rows() {
                                return Err(HayashiError::Runtime(format!(
                                    "list variable '{name}' has length {}, expected {}",
                                    lst.len(), df.n_rows()
                                )));
                            }
                            let mut data = Vec::with_capacity(lst.len());
                            for v in lst.iter() {
                                match v {
                                    Value::Float(f) => data.push(*f),
                                    Value::Int(i_val) => data.push(*i_val as f64),
                                    Value::Bool(b) => data.push(if *b { 1.0 } else { 0.0 }),
                                    other => return Err(HayashiError::Type(format!(
                                        "element in list variable '{name}' is not numeric: {other}"
                                    ))),
                                }
                            }
                            Ok(data)
                        }
                        _ => Err(HayashiError::Runtime(format!(
                            "'{name}' not found as column or scalar variable"
                        ))),
                    },
                }
            }
            Expr::Neg(inner) => {
                let vals = self.eval_col_expr(inner, df)?;
                Ok(vals.into_iter().map(|x| -x).collect())
            }
            Expr::Not(inner) => {
                let vals = self.eval_col_expr(inner, df)?;
                Ok(vals.into_iter().map(|x| if x == 0.0 { 1.0 } else { 0.0 }).collect())
            }
            Expr::BinOp { op, lhs, rhs } => {
                // String column equality/inequality: col == "literal" or "literal" == col
                if matches!(op, BinOp::Eq | BinOp::Ne) {
                    let str_pair = match (lhs.as_ref(), rhs.as_ref()) {
                        (Expr::Var(c), Expr::Str(t)) => Some((c.as_str(), t.as_str())),
                        (Expr::Str(t), Expr::Var(c)) => Some((c.as_str(), t.as_str())),
                        _ => None,
                    };
                    if let Some((col_name, target)) = str_pair {
                        let is_eq = matches!(op, BinOp::Eq);
                        if let Ok(col) = df.get_column(col_name) {
                            use greeners::Column;
                            let maybe: Option<Vec<f64>> = match col {
                                Column::String(arr) => Some(arr.iter().map(|s| {
                                    if (s.as_str() == target) == is_eq { 1.0 } else { 0.0 }
                                }).collect()),
                                Column::Categorical(cat) => Some(cat.to_strings().iter().map(|s| {
                                    if (s.as_str() == target) == is_eq { 1.0 } else { 0.0 }
                                }).collect()),
                                _ => None,
                            };
                            if let Some(v) = maybe { return Ok(v); }
                        }
                    }
                }
                let l = self.eval_col_expr(lhs, df)?;
                let r = self.eval_col_expr(rhs, df)?;
                if l.len() != r.len() {
                    return Err(HayashiError::Runtime("mismatched column lengths".into()));
                }
                Ok(l.into_iter().zip(r).map(|(a, b)| match op {
                    BinOp::Add  => a + b,
                    BinOp::Sub  => a - b,
                    BinOp::Mul  => a * b,
                    BinOp::Div  => a / b,
                    BinOp::Mod  => a % b,
                    BinOp::Pow  => a.powf(b),
                    BinOp::Gt   => if a > b { 1.0 } else { 0.0 },
                    BinOp::Lt   => if a < b { 1.0 } else { 0.0 },
                    BinOp::GtEq => if a >= b { 1.0 } else { 0.0 },
                    BinOp::LtEq => if a <= b { 1.0 } else { 0.0 },
                    BinOp::Eq   => if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 },
                    BinOp::Ne   => if (a - b).abs() >= f64::EPSILON { 1.0 } else { 0.0 },
                    BinOp::And  => if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 },
                    BinOp::Or   => if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 },
                    BinOp::In   => 0.0,
                }).collect())
            }
            Expr::Call { func, args, .. } => {
                // ── regex row-wise sobre colunas string ──
                if func == "regexm" && args.len() >= 2 {
                    if let Expr::Var(col_name) = &args[0] {
                        if let Ok(str_col) = df.get_string(col_name) {
                            let pattern = match &args[1] {
                                Expr::Str(s) => s.clone(),
                                _ => return Err(HayashiError::Type("regexm: pattern must be string literal".into())),
                            };
                            return Ok(greeners::Transforms::regexm_vec(&str_col.to_vec(), &pattern));
                        }
                    }
                }

                // ── geradores aleatórios (tamanho = n_rows do df) ──
                if matches!(func.as_str(), "uniform" | "runiform" | "rnormal" | "rbernoulli") {
                    let n = df.n_rows();
                    use rand::Rng;
                    return Ok(match func.as_str() {
                        "uniform" | "runiform" => {
                            let rng = &mut self.rng;
                            (0..n).map(|_| rng.gen::<f64>()).collect()
                        }
                        "rnormal" => {
                            let rng = &mut self.rng;
                            (0..n).map(|_| standard_normal_draw(rng)).collect()
                        }
                        "rbernoulli" => {
                            let p = if !args.is_empty() {
                                self.eval_col_expr(&args[0], df)?[0]
                            } else { 0.5 };
                            let rng = &mut self.rng;
                            (0..n).map(|_| if rng.gen::<f64>() < p { 1.0 } else { 0.0 }).collect()
                        }
                        _ => unreachable!(),
                    });
                }

                // ── funções multi-coluna (rowmean / rowsum / rowmin / rowmax / rowtotal / rowmiss) ──
                if matches!(func.as_str(), "rowmean" | "rowsum" | "rowmin" | "rowmax" | "rowtotal" | "rowmiss") {
                    if args.is_empty() {
                        return Err(HayashiError::Runtime(
                            format!("{func}() requires at least one column")
                        ));
                    }
                    let cols: Vec<Vec<f64>> = args.iter()
                        .map(|a| self.eval_col_expr(a, df))
                        .collect::<Result<_>>()?;
                    return Ok(match func.as_str() {
                        "rowmean"  => greeners::Transforms::row_mean(&cols),
                        "rowsum"   => greeners::Transforms::row_sum(&cols),
                        "rowmin"   => greeners::Transforms::row_min(&cols),
                        "rowmax"   => greeners::Transforms::row_max(&cols),
                        "rowtotal" => greeners::Transforms::row_total(&cols),
                        "rowmiss"  => greeners::Transforms::row_miss(&cols),
                        _ => unreachable!(),
                    });
                }

                if args.len() == 1 {
                    // ── funções que precisam de toda a coluna ──────────────────
                    match func.as_str() {
                        "rank" => {
                            let vals = self.eval_col_expr(&args[0], df)?;
                            return Ok(greeners::Transforms::rank(&vals));
                        }
                        "cumsum" => {
                            let vals = self.eval_col_expr(&args[0], df)?;
                            return Ok(greeners::Transforms::cumsum(&vals));
                        }
                        "std" | "standardize" | "zscore" => {
                            let vals = self.eval_col_expr(&args[0], df)?;
                            return Ok(greeners::Transforms::standardize(&vals));
                        }
                        "iqr" => {
                            let vals = self.eval_col_expr(&args[0], df)?;
                            let iqr_val = greeners::Transforms::iqr(&vals);
                            return Ok(vec![iqr_val; df.n_rows()]);
                        }
                        "group" => {
                            let col_name = match &args[0] {
                                Expr::Var(name) => name.clone(),
                                _ => return Err(HayashiError::Runtime(
                                    "group() requires a column name".into()
                                )),
                            };
                            let strs = Self::col_to_strings(df, &col_name)?;
                            return Ok(greeners::Transforms::group(&strs));
                        }
                        "year" | "month" | "day" | "hour" | "minute" | "second" | "dow" => {
                            let col_name = match &args[0] {
                                Expr::Var(name) => name.clone(),
                                _ => return Err(HayashiError::Runtime(
                                    format!("{func}() requires a column name"),
                                )),
                            };
                            if let Ok(arr) = df.get_datetime(&col_name) {
                                use chrono::{Datelike, Timelike};
                                let extract = |dt: &chrono::NaiveDateTime| -> f64 {
                                    match func.as_str() {
                                        "year" => dt.year() as f64,
                                        "month" => dt.month() as f64,
                                        "day" => dt.day() as f64,
                                        "hour" => dt.hour() as f64,
                                        "minute" => dt.minute() as f64,
                                        "second" => dt.second() as f64,
                                        "dow" => dt.weekday().num_days_from_monday() as f64,
                                        _ => f64::NAN,
                                    }
                                };
                                return Ok(arr.iter().map(extract).collect());
                            }
                            let vals = self.eval_col_expr(&args[0], df)?;
                            use chrono::DateTime as ChronoDateTime;
                            let result: Vec<f64> = vals.iter().map(|&ts| {
                                let dt = ChronoDateTime::from_timestamp(ts as i64, 0)
                                    .map(|d| d.naive_utc());
                                match dt {
                                    Some(d) => {
                                        use chrono::{Datelike, Timelike};
                                        match func.as_str() {
                                            "year" => d.year() as f64,
                                            "month" => d.month() as f64,
                                            "day" => d.day() as f64,
                                            "hour" => d.hour() as f64,
                                            "minute" => d.minute() as f64,
                                            "second" => d.second() as f64,
                                            "dow" => d.weekday().num_days_from_monday() as f64,
                                            _ => f64::NAN,
                                        }
                                    }
                                    None => f64::NAN,
                                }
                            }).collect();
                            return Ok(result);
                        }
                        _ => {}
                    }

                    // ── funções escalares elemento-a-elemento (1-arg) ─────────
                    let vals = self.eval_col_expr(&args[0], df)?;
                    match greeners::Transforms::apply(&vals, func) {
                        Ok(result) => Ok(result),
                        Err(_) => {
                            if let Some(Value::UserFn(uf)) = self.env.get(func).cloned() {
                                let mut result = Vec::with_capacity(vals.len());
                                for &v in &vals {
                                    self.env.push_scope();
                                    if let Some(p) = uf.params.first() {
                                        self.env.declare_const(p, Value::Float(v));
                                    }
                                    let body = uf.body.clone();
                                    let mut exec_err = None;
                                    for s in &body {
                                        match self.exec(s) {
                                            Ok(()) => {}
                                            Err(HayashiError::Return) => break,
                                            Err(e) => { exec_err = Some(e); break; }
                                        }
                                    }
                                    self.env.pop_scope();
                                    if let Some(e) = exec_err {
                                        return Err(e);
                                    }
                                    match self.return_value.take().unwrap_or(Value::Float(f64::NAN)) {
                                        Value::Float(f) => result.push(f),
                                        Value::Int(i) => result.push(i as f64),
                                        _ => result.push(f64::NAN),
                                    }
                                }
                                Ok(result)
                            } else {
                                Err(HayashiError::Runtime(
                                    format!("função de coluna desconhecida '{func}'")
                                ))
                            }
                        }
                    }
                } else if args.len() == 2 {
                    let a = self.eval_col_expr(&args[0], df)?;
                    let b = self.eval_col_expr(&args[1], df)?;
                    match greeners::Transforms::apply2(&a, &b, func) {
                        Ok(result) => Ok(result),
                        Err(_) => Err(HayashiError::Runtime(
                            format!("função '{func}' not supportada em generate")
                        )),
                    }
                } else if args.len() == 3 {
                    let a = self.eval_col_expr(&args[0], df)?;
                    let b = self.eval_col_expr(&args[1], df)?;
                    let c = self.eval_col_expr(&args[2], df)?;
                    match greeners::Transforms::apply3(&a, &b, &c, func) {
                        Ok(result) => Ok(result),
                        Err(_) => Err(HayashiError::Runtime(
                            format!("função '{func}' not supportada em generate")
                        )),
                    }
                } else {
                    Err(HayashiError::Runtime(format!(
                        "função '{func}' not supportada em generate"
                    )))
                }
            }
            // ── operadores de série temporal ─────────────────────────────────
            // Requerem que o df já esteja ordenado por tsset.
            // L.x = x[i-n], F.x = x[i+n], D.x = x[i] - x[i-n]
            Expr::TsOp { op, var, n } => {
                use greeners::Column;
                let col = df.get_column(var)
                    .map_err(|_| HayashiError::Runtime(format!("column '{var}' not found")))?;
                let vals: Vec<f64> = match col {
                    Column::Float(arr) => arr.to_vec(),
                    Column::Int(arr)   => arr.iter().map(|&x| x as f64).collect(),
                    _ => return Err(HayashiError::Type(format!("column '{var}' is not numeric"))),
                };
                let len = vals.len();
                let n = *n;
                Ok(match op {
                    TsOpKind::Lag  => (0..len)
                        .map(|i| if i >= n { vals[i - n] } else { f64::NAN })
                        .collect(),
                    TsOpKind::Lead => (0..len)
                        .map(|i| if i + n < len { vals[i + n] } else { f64::NAN })
                        .collect(),
                    TsOpKind::Diff => (0..len)
                        .map(|i| if i >= n { vals[i] - vals[i - n] } else { f64::NAN })
                        .collect(),
                })
            }

            Expr::Apply { func, args } => {
                let closure_val = self.eval_expr(func)?;
                let uf = match closure_val {
                    Value::UserFn(f) => f,
                    _ => return Err(HayashiError::Runtime(
                        "generate: pipe target must be a function or closure".into(),
                    )),
                };
                let vals = self.eval_col_expr(&args[0], df)?;
                let mut result = Vec::with_capacity(vals.len());
                for &v in &vals {
                    self.env.push_scope();
                    if let Some(p) = uf.params.first() {
                        self.env.declare_const(p, Value::Float(v));
                    }
                    let body = uf.body.clone();
                    let mut exec_err = None;
                    for s in &body {
                        match self.exec(s) {
                            Ok(()) => {}
                            Err(HayashiError::Return) => break,
                            Err(e) => { exec_err = Some(e); break; }
                        }
                    }
                    self.env.pop_scope();
                    if let Some(e) = exec_err {
                        return Err(e);
                    }
                    match self.return_value.take().unwrap_or(Value::Float(f64::NAN)) {
                        Value::Float(f) => result.push(f),
                        Value::Int(i) => result.push(i as f64),
                        _ => result.push(f64::NAN),
                    }
                }
                Ok(result)
            }

            _ => Err(HayashiError::Runtime(
                "expression type not supported in generate".into()
            )),
        }
    }

    // ── Executa statement ─────────────────────────────────────────────────────

    pub fn exec(&mut self, spanned: &Spanned) -> Result<()> {
        let (stmt, line) = spanned;
        self.current_line = *line;
        match stmt {
            Stmt::Let { name, value } => {
                self.capturing = true;
                let val = self.eval_expr(value)?;
                self.capturing = false;
                self.env.declare(name, val)?;
            }

            Stmt::Const { name, value } => {
                self.capturing = true;
                let val = self.eval_expr(value)?;
                self.capturing = false;
                self.env.declare_const(name, val);
            }

            Stmt::Assign { name, value } => {
                self.capturing = true;
                let val = self.eval_expr(value)?;
                self.capturing = false;
                self.env.set(name, val)?;
            }

            // ── input df \n headers \n rows \n end ───────────────────────────
            Stmt::Input {
                alias,
                headers,
                rows,
            } => {
                if headers.is_empty() {
                    return Err(self.rt_err("input: no variables in header"));
                }
                if rows.is_empty() {
                    return Err(self.rt_err("input: no data rows"));
                }
                let k = headers.len();
                // Verifica que todas as linhas têm o mesmo número de colunas
                for (i, row) in rows.iter().enumerate() {
                    if row.len() != k {
                        return Err(HayashiError::Runtime(format!(
                            "input: linha {} tem {} valores, esperado {} ({})",
                            i + 1,
                            row.len(),
                            k,
                            headers.join(", ")
                        )));
                    }
                }
                let n = rows.len();
                // Transpõe: rows → columns
                let mut col_map: std::collections::HashMap<String, ndarray::Array1<f64>> =
                    std::collections::HashMap::new();
                for (j, name) in headers.iter().enumerate() {
                    let col: ndarray::Array1<f64> =
                        ndarray::Array1::from(rows.iter().map(|r| r[j]).collect::<Vec<_>>());
                    col_map.insert(name.clone(), col);
                }
                let df =
                    greeners::DataFrame::new(col_map).map_err(|e| self.rt_err(e.to_string()))?;
                println!(
                    "input → {alias} ({n} obs, {} vars: {})",
                    k,
                    headers.join(", ")
                );
                self.env.set(&alias, Value::DataFrame(Rc::new(df)))?;
            }

            // ── display expr ─────────────────────────────────────────────────
            Stmt::Display(expr) => {
                let val = self.eval_expr(expr)?;
                match &val {
                    Value::Float(v) => println!("{v}"),
                    Value::Int(v) => println!("{v}"),
                    Value::Bool(v) => println!("{v}"),
                    Value::Str(v) => println!("\"{v}\""),
                    Value::Nil => println!("(nil)"),
                    Value::List(lst) => {
                        for v in lst.iter() {
                            print!("  {v}");
                        }
                        println!();
                    }
                    _ => println!("{val}"),
                }
            }

            Stmt::Load { path, alias, opts } => {
                let path_str = match self.eval_expr(path)? {
                    Value::Str(s) => s,
                    _ => return Err(self.type_err("load requires a string path")),
                };

                let mut opt_sheet: Option<String> = None;
                let mut opt_table: Option<String> = None;
                let mut opt_query: Option<String> = None;
                let mut opt_sep: Option<String> = None;
                for o in opts {
                    let val = match self.eval_expr(&o.value)? {
                        Value::Str(s) => s,
                        Value::Float(f) => format!("{f}"),
                        Value::Int(i) => format!("{i}"),
                        other => format!("{other}"),
                    };
                    match o.name.as_str() {
                        "sheet" => opt_sheet = Some(val),
                        "table" => opt_table = Some(val),
                        "query" => opt_query = Some(val),
                        "sep" | "delimiter" => opt_sep = Some(val),
                        k => {
                            return Err(HayashiError::Runtime(format!(
                                "load: unknown option '{k}' — use: sheet, table, query, sep"
                            )))
                        }
                    }
                }

                // ── ODBC ────────────────────────────────────────────────
                if path_str.starts_with("odbc://") {
                    #[cfg(feature = "odbc")]
                    {
                        let conn_str = &path_str["odbc://".len()..];
                        let sql = if let Some(q) = &opt_query {
                            q.clone()
                        } else if let Some(t) = &opt_table {
                            format!("SELECT * FROM \"{t}\"")
                        } else {
                            return Err(HayashiError::Runtime(
                                "load odbc: requires query= or table= option".into(),
                            ));
                        };
                        let (df, n_rows) = crate::io::odbc::load_odbc(conn_str, &sql)?;
                        println!("Loaded ODBC → {alias} ({n_rows} rows)");
                        self.env.set(alias, Value::DataFrame(Rc::new(df)))?;
                    }
                    #[cfg(not(feature = "odbc"))]
                    {
                        return Err(HayashiError::Runtime(
                            "ODBC support not enabled. Rebuild with: cargo build --features odbc\n\
                             Requires: unixodbc (pacman -S unixodbc)"
                                .into(),
                        ));
                    }
                } else {
                    // ── Arquivo / URL ───────────────────────────────────────
                    let _tmp;
                    let local_path: &str = if crate::io::fetch::is_url(&path_str) {
                        println!("Downloading '{}'…", path_str);
                        _tmp = crate::io::fetch::download_to_temp(&path_str)?;
                        _tmp.to_str()
                            .ok_or_else(|| self.rt_err("temp path is not UTF-8"))?
                    } else {
                        &path_str
                    };

                    let ext = local_path.rsplit('.').next().unwrap_or("").to_lowercase();

                    let (df, n_rows) = match ext.as_str() {
                        "dta" => crate::io::dta::load_dta(local_path)?,
                        "xlsx" | "xls" | "ods" => {
                            crate::io::excel::load_excel(local_path, opt_sheet.as_deref())?
                        }
                        "sqlite" | "sqlite3" | "db" => crate::io::sqlite::load_sqlite(
                            local_path,
                            opt_table.as_deref(),
                            opt_query.as_deref(),
                        )?,
                        "json" => {
                            let df = DataFrame::from_json(local_path)
                                .map_err(|e| self.rt_err(e.to_string()))?;
                            let n = df.n_rows();
                            (df, n)
                        }
                        "tsv" | "tab" => crate::io::dsv::load_dsv(local_path, b'\t')?,
                        "parquet" | "pq" => crate::io::parquet::load_parquet(local_path)?,
                        _ => {
                            let delim = match opt_sep.as_deref() {
                                Some("\\t") | Some("tab") => b'\t',
                                Some(s) if s.len() == 1 => s.as_bytes()[0],
                                Some(s) => {
                                    return Err(HayashiError::Runtime(format!(
                                        "load: sep must be a single character, got '{s}'"
                                    )))
                                }
                                None => b',',
                            };
                            if delim == b',' {
                                let df = DataFrame::from_csv(local_path)
                                    .map_err(|e| self.rt_err(e.to_string()))?;
                                let n = df.n_rows();
                                (df, n)
                            } else {
                                crate::io::dsv::load_dsv(local_path, delim)?
                            }
                        }
                    };
                    println!("Loaded '{}' → {alias} ({} rows)", path_str, n_rows);
                    self.env.set(alias, Value::DataFrame(Rc::new(df)))?;
                } // else (não-ODBC)
            }

            Stmt::Predict {
                df,
                varname,
                model,
                kind,
            } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df}' is not a DataFrame"))),
                };
                let model_val = self.eval_expr(model)?;
                let kind_str = match self.eval_expr(kind)? {
                    Value::Str(s) => s,
                    other => {
                        return Err(HayashiError::Type(format!(
                            "predict kind must be a string, got {other}"
                        )))
                    }
                };

                let vals: Vec<f64> = match (&model_val, kind_str.as_str()) {
                    // ── OLS ──────────────────────────────────────────────────
                    (Value::OlsResult(m), "xb" | "fitted") => {
                        m.x.dot(&m.result.params).to_vec()
                    }
                    (Value::OlsResult(m), "residuals" | "resid" | "e") => {
                        m.residuals.to_vec()
                    }
                    (Value::OlsResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict OLS: kind '{k}' unknown — use: xb, residuals")
                    )),

                    // ── Logit / Probit ────────────────────────────────────────
                    (Value::BinaryResult(m), "pr" | "xb" | "fitted") => {
                        m.result.predict_proba(&m.x).to_vec()
                    }
                    (Value::BinaryResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict logit/probit: kind '{k}' unknown — use: pr")
                    )),

                    // ── Poisson / NegBin ──────────────────────────────────────
                    (Value::PoissonResult(r), "count" | "mu" | "fitted") => {
                        r.fitted_values().to_vec()
                    }
                    (Value::PoissonResult(r), "xb") => {
                        r.x_data().dot(&r.params).to_vec()
                    }
                    (Value::PoissonResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Poisson: kind '{k}' unknown — use: count, xb")
                    )),
                    (Value::NegBinResult(r), "count" | "mu" | "fitted") => {
                        r.fitted_values().to_vec()
                    }
                    (Value::NegBinResult(r), "xb") => {
                        r.x_data().dot(&r.params).to_vec()
                    }
                    (Value::NegBinResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict NegBin: kind '{k}' unknown — use: count, xb")
                    )),

                    // ── Ordered Logit / Probit ────────────────────────────────
                    // "pr"   → P(Y = J) — probabilidade da categoria mais alta
                    // "xb"   → preditor linear Xβ
                    // "yhat" → categoria predita (argmax)
                    // "prN"  → P(Y = N) para categoria específica N (1-indexed)
                    (Value::OrderedResult(r), kind_s) => {
                        let x = Self::build_x_from_varnames(&df_val,
                            r.variable_names.as_deref().unwrap_or(&[]))?;
                        match kind_s {
                            "xb" => x.dot(&r.params).to_vec(),
                            "yhat" => {
                                let probs = r.predict_proba(&x);
                                (0..probs.nrows()).map(|i| {
                                    let row = probs.row(i);
                                    let (cat, _) = row.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                        .unwrap_or((0, &0.0));
                                    (cat + 1) as f64
                                }).collect()
                            }
                            s if s.starts_with("pr") && s.len() > 2 => {
                                let cat: usize = s[2..].parse::<usize>()
                                    .map_err(|_| HayashiError::Runtime(
                                        format!("predict Ordered: '{s}' — use prN onde N é a categoria (1-indexed)")
                                    ))?;
                                if cat == 0 || cat > r.n_categories {
                                    return Err(HayashiError::Runtime(
                                        format!("predict Ordered: categoria {cat} out of range 1..{}", r.n_categories)
                                    ));
                                }
                                let probs = r.predict_proba(&x);
                                (0..probs.nrows()).map(|i| probs[[i, cat - 1]]).collect()
                            }
                            "pr" => {
                                // P(Y = última categoria)
                                let probs = r.predict_proba(&x);
                                let last = r.n_categories - 1;
                                (0..probs.nrows()).map(|i| probs[[i, last]]).collect()
                            }
                            k => return Err(HayashiError::Runtime(
                                format!("predict Ordered: kind '{k}' unknown — use: pr, prN, yhat, xb")
                            )),
                        }
                    }

                    // ── IV / 2SLS ─────────────────────────────────────────────
                    (Value::IvResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::IvResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict IV: kind '{k}' unknown — use: xb")
                    )),

                    // ── Panel FE / RE ─────────────────────────────────────────
                    (Value::PanelResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::PanelResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict FE: kind '{k}' unknown — use: xb")
                    )),
                    (Value::ReResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::ReResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RE: kind '{k}' unknown — use: xb")
                    )),

                    // ── Tobit ─────────────────────────────────────────────────
                    (Value::TobitResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::TobitResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Tobit: kind '{k}' unknown — use: xb")
                    )),

                    // ── Heckman ───────────────────────────────────────────────
                    (Value::HeckmanResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::HeckmanResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Heckman: kind '{k}' unknown — use: xb")
                    )),

                    // ── Cox PH ────────────────────────────────────────────────
                    (Value::CoxResult(r), "loghr" | "xb") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        r.predict_log_hazard(&x).to_vec()
                    }
                    (Value::CoxResult(r), "hr" | "hazard") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        r.predict_hazard_ratio(&x).to_vec()
                    }
                    (Value::CoxResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Cox: kind '{k}' unknown — use: loghr, hr")
                    )),

                    // ── Quantile Regression ───────────────────────────────────
                    (Value::QuantileResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::QuantileResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict QReg: kind '{k}' unknown — use: xb")
                    )),

                    // ── RLM ──────────────────────────────────────────────────
                    (Value::RlmResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::RlmResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RLM: kind '{k}' unknown — use: xb")
                    )),

                    // ── GEE ──────────────────────────────────────────────────
                    (Value::GeeResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::GeeResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict GEE: kind '{k}' unknown — use: xb")
                    )),

                    // ── Beta Regression ───────────────────────────────────────
                    (Value::BetaResult(r), "pr" | "mu" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        r.predict(&x, &greeners::BetaLink::Logit).to_vec()
                    }
                    (Value::BetaResult(r), "xb") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::BetaResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict BetaReg: kind '{k}' unknown — use: pr, xb")
                    )),

                    // ── GLSAR ────────────────────────────────────────────────
                    (Value::GlsarResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        r.fitted_values(&x).to_vec()
                    }
                    (Value::GlsarResult(r), "residuals" | "resid" | "e") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        let y = Self::get_col_f64(&df_val, &varname)?;
                        r.residuals(&y, &x).to_vec()
                    }
                    (Value::GlsarResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict GLSAR: kind '{k}' unknown — use: xb, residuals")
                    )),

                    // ── MixedLM ───────────────────────────────────────────────
                    (Value::MixedResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.fixed_effects).to_vec()
                    }
                    (Value::MixedResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict MixedLM: kind '{k}' unknown — use: xb")
                    )),

                    // ── ZIP / ZINB ────────────────────────────────────────────
                    (Value::ZeroInflatedResult(r), "count" | "mu" | "fitted") => {
                        // E[y|x, w>0] × P(w=0): media incondicional da contagem
                        let names = r.count_var_names.as_deref().unwrap_or(&[]);
                        let x_c = Self::build_x_from_varnames(&df_val, names)?;
                        let inflate_names = r.inflate_var_names.as_deref().unwrap_or(names);
                        let x_i = Self::build_x_from_varnames(&df_val, inflate_names)?;
                        r.predict_count(&x_c, &x_i).to_vec()
                    }
                    (Value::ZeroInflatedResult(r), "pr0") => {
                        // P(y=0 | x) — probabilidade de zero
                        let names = r.count_var_names.as_deref().unwrap_or(&[]);
                        let x_c = Self::build_x_from_varnames(&df_val, names)?;
                        let inflate_names = r.inflate_var_names.as_deref().unwrap_or(names);
                        let x_i = Self::build_x_from_varnames(&df_val, inflate_names)?;
                        r.predict_proba_zero(&x_c, &x_i).to_vec()
                    }
                    (Value::ZeroInflatedResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict ZIP/ZINB: kind '{k}' unknown — use: count, pr0")
                    )),

                    // ── Rolling OLS ───────────────────────────────────────────
                    (Value::RollingResult(r), "residuals" | "resid" | "e") => {
                        r.residuals.to_vec()
                    }
                    (Value::RollingResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RollingOLS: kind '{k}' unknown — use: residuals")
                    )),

                    // ── Recursive LS ──────────────────────────────────────────
                    (Value::RecursiveLSResult(r), "residuals" | "resid" | "e") => {
                        r.residuals.to_vec()
                    }
                    (Value::RecursiveLSResult(r), "cusum") => {
                        r.cusum.to_vec()
                    }
                    (Value::RecursiveLSResult(r), "cusum_sq") => {
                        r.cusum_squares.to_vec()
                    }
                    (Value::RecursiveLSResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RecursiveLS: kind '{k}' unknown — use: residuals, cusum, cusum_sq")
                    )),

                    // ── GLM ──────────────────────────────────────────────────────
                    // pr/mu/fitted → μ̂ = g⁻¹(Xβ) — resposta média predita
                    // xb → Xβ — preditor linear (escala do link)
                    // residuals → resíduos de desvio (deviance residuals)
                    // pearson → resíduos de Pearson (y-μ)/√V(μ)
                    // working → resíduos de trabalho do IRLS
                    (Value::GlmResult(r), "pr" | "mu" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        r.predict_mean(&x).to_vec()
                    }
                    (Value::GlmResult(r), "xb") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        r.predict(&x).to_vec()
                    }
                    (Value::GlmResult(r), "residuals" | "resid" | "e" | "deviance") => {
                        r.residuals().to_vec()
                    }
                    (Value::GlmResult(r), "pearson") => {
                        r.pearson_residuals().to_vec()
                    }
                    (Value::GlmResult(r), "working") => {
                        r.working_residuals().to_vec()
                    }
                    (Value::GlmResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict GLM: kind '{k}' unknown — use: pr, xb, residuals, pearson, working")
                    )),

                    // ── LOWESS ───────────────────────────────────────────────────
                    // smoothed/yhat → valores suavizados ŷ_i
                    // residuals → resíduos y_i - ŷ_i
                    (Value::LowessResult(r), "smoothed" | "yhat" | "fitted") => {
                        r.smoothed.to_vec()
                    }
                    (Value::LowessResult(r), "residuals" | "resid" | "e") => {
                        r.residuals.to_vec()
                    }
                    (Value::LowessResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict LOWESS: kind '{k}' unknown — use: smoothed, residuals")
                    )),

                    // ── PCA ──────────────────────────────────────────────────────
                    // pc1, pc2, ..., pcN → escores do N-ésimo componente principal
                    // Os escores são calculados durante o ajuste (dados de treino)
                    (Value::PcaResult(m), kind_s) => {
                        if kind_s.starts_with("pc") && kind_s.len() > 2 {
                            let comp: usize = kind_s[2..].parse::<usize>()
                                .map_err(|_| HayashiError::Runtime(
                                    format!("predict PCA: '{kind_s}' inválido — use pcN onde N=1..{}", m.result.n_components)
                                ))?;
                            if comp == 0 || comp > m.result.n_components {
                                return Err(HayashiError::Runtime(
                                    format!("predict PCA: componente {comp} out of range 1..{}", m.result.n_components)
                                ));
                            }
                            m.result.scores.column(comp - 1).to_vec()
                        } else {
                            return Err(HayashiError::Runtime(
                                format!("predict PCA: kind '{kind_s}' unknown — use: pc1, pc2, ..., pc{}", m.result.n_components)
                            ));
                        }
                    }

                    // ── Factor Analysis ───────────────────────────────────────────
                    // Factor Analysis não produz escores diretamente (não há método de predict)
                    // Use pca() para escores; factor() é apenas para análise das cargas/estrutura
                    (Value::FactorResult(_), _) => return Err(HayashiError::Runtime(
                        "predict Factor Analysis: escores não disponíveis via FA — use pca() para escores; FA é para análise de cargas".into()
                    )),

                    // ── Markov Switching ──────────────────────────────────────────
                    // smoothed → probabilidades suavizadas do regime mais provável (argmax)
                    // regime1, regime2, ..., regimeN → prob suavizada do regime N
                    (Value::MarkovResult(r), "smoothed" | "regime" | "state") => {
                        // regime mais provável em cada ponto (1-indexed)
                        (0..r.smoothed_probs.nrows()).map(|t| {
                            let row = r.smoothed_probs.row(t);
                            let (best, _) = row.iter().enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                                .unwrap_or((0, &0.0));
                            (best + 1) as f64
                        }).collect()
                    }
                    (Value::MarkovResult(r), kind_s) if kind_s.starts_with("regime") && kind_s.len() > 6 => {
                        let idx: usize = kind_s[6..].parse::<usize>()
                            .map_err(|_| HayashiError::Runtime(
                                format!("predict MarkovSwitching: '{kind_s}' inválido — use regimeN onde N=1..{}", r.n_regimes)
                            ))?;
                        if idx == 0 || idx > r.n_regimes {
                            return Err(HayashiError::Runtime(
                                format!("predict MarkovSwitching: regime {idx} out of range 1..{}", r.n_regimes)
                            ));
                        }
                        r.smoothed_probs.column(idx - 1).to_vec()
                    }
                    (Value::MarkovResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict MarkovSwitching: kind '{k}' unknown — use: regime, regime1, regime2, ...")
                    )),

                    // ── Conditional Logit / Poisson ───────────────────────────────
                    // FE é diferenciado; predição incondicional não disponível
                    (Value::ConditionalResult(_), _) => return Err(HayashiError::Runtime(
                        "predict clogit/cpoisson: efeitos fixos absorvidos — predição incondicional não disponível; use os coeficientes β̂ para odds ratios ou efeitos marginais".into()
                    )),

                    // ── VARMA ─────────────────────────────────────────────────────
                    (Value::VarmaResult(_), _) => return Err(HayashiError::Runtime(
                        "predict varma: predição multivariada not supportada como coluna — use print() para diagnóstico".into()
                    )),

                    // ── UCM ───────────────────────────────────────────────────────
                    (Value::UCResult(r), "level")                     => r.level.to_vec(),
                    (Value::UCResult(r), "trend")                     => r.trend.as_ref()
                        .map(|t| t.to_vec())
                        .unwrap_or_else(|| vec![f64::NAN; r.n_obs]),
                    (Value::UCResult(r), "seasonal")                  => r.seasonal.as_ref()
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|| vec![f64::NAN; r.n_obs]),
                    (Value::UCResult(r), "residuals" | "resid" | "e") => r.residuals.to_vec(),
                    (Value::UCResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict ucm: kind '{k}' unknown — use: level, trend, seasonal, residuals")
                    )),

                    // ── GAM ───────────────────────────────────────────────────────
                    (Value::GamResult(_), _) => return Err(HayashiError::Runtime(
                        "predict gam: valores ajustados não estão armazenados — use gam() com df=dataset e calcule Xβ̂ manualmente".into()
                    )),

                    // ── MICE ──────────────────────────────────────────────────────
                    (Value::MiceResult(_), _) => return Err(HayashiError::Runtime(
                        "predict mice: MICE retorna múltiplos datasets; acesse via pooling de modelos".into()
                    )),

                    // ── SVAR ─────────────────────────────────────────────────────
                    (Value::SVarResult(_), _) => return Err(HayashiError::Runtime(
                        "predict svar: sem valores ajustados — use sirf() e sfevd() para análise de impulso-resposta".into()
                    )),

                    // ── 3SLS ─────────────────────────────────────────────────────
                    (Value::ThreeSLSResult(_), _) => return Err(HayashiError::Runtime(
                        "predict 3sls: múltiplas equações — use print() para ver coeficientes por equação".into()
                    )),

                    // ── DFM ───────────────────────────────────────────────────────
                    (Value::DFMResult(m), kind_s) if kind_s.starts_with('f') => {
                        let idx = kind_s[1..].parse::<usize>()
                            .map(|n| n.saturating_sub(1))
                            .unwrap_or(0);
                        if idx >= m.result.n_factors {
                            return Err(HayashiError::Runtime(format!(
                                "predict dfm: fator f{} não existe — modelo tem {} fatores",
                                idx + 1, m.result.n_factors
                            )));
                        }
                        m.result.factors.column(idx).to_vec()
                    }
                    (Value::DFMResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict dfm: kind '{k}' unknown — use: f1, f2, ... (índice 1-based do fator latente)")
                    )),

                    // ── MarkovAutoregression ───────────────────────────────────────
                    (Value::MSARResult(r), "regime" | "state") => {
                        r.predict_regime().iter().map(|&s| (s + 1) as f64).collect()
                    }
                    (Value::MSARResult(r), kind_s) if kind_s.starts_with("regime") && kind_s.len() > 6 => {
                        let idx = kind_s["regime".len()..].parse::<usize>()
                            .map(|n| n.saturating_sub(1))
                            .unwrap_or(0);
                        if idx >= r.k_regimes {
                            return Err(HayashiError::Runtime(format!(
                                "predict msauto: regime{} out of range 1..{}",
                                idx + 1, r.k_regimes
                            )));
                        }
                        r.smoothed_probs.column(idx).to_vec()
                    }
                    (Value::MSARResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict msauto: kind '{k}' unknown — use: regime, regime1, regime2, ...")
                    )),

                    // ── Decomposição sazonal ──────────────────────────────────────
                    (Value::DecompResult(r), "trend")    => r.trend.to_vec(),
                    (Value::DecompResult(r), "seasonal") => r.seasonal.to_vec(),
                    (Value::DecompResult(r), "residual" | "resid" | "e") => r.residual.to_vec(),
                    (Value::DecompResult(r), "observed" | "fitted") => r.observed.to_vec(),
                    (Value::DecompResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict decompose: kind '{k}' unknown — use: trend, seasonal, residual, observed")
                    )),

                    // ── MSTL ─────────────────────────────────────────────────────
                    (Value::MstlResult(r), "trend") => r.trend.to_vec(),
                    (Value::MstlResult(r), "resid" | "residual" | "e") => r.resid.to_vec(),
                    (Value::MstlResult(r), kind_s) if kind_s.starts_with("seasonal") => {
                        // "seasonal" → primeira componente; "seasonal1" → índice 1-based
                        let idx = if kind_s == "seasonal" {
                            0usize
                        } else {
                            kind_s["seasonal".len()..].parse::<usize>()
                                .map(|n| n.saturating_sub(1))
                                .unwrap_or(0)
                        };
                        if idx >= r.seasonal.len() {
                            return Err(HayashiError::Runtime(format!(
                                "predict mstl: componente seasonal{} não existe — modelo tem {} períodos",
                                idx + 1, r.seasonal.len()
                            )));
                        }
                        r.seasonal[idx].to_vec()
                    }
                    (Value::MstlResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict mstl: kind '{k}' unknown — use: trend, resid, seasonal, seasonal1, seasonal2, ...")
                    )),

                    // ── ETS (suavização exponencial) ──────────────────────────
                    (Value::EtsResult(r), "fitted" | "yhat" | "xb") => r.fitted_values.to_vec(),
                    (Value::EtsResult(r), "residuals" | "resid" | "e") => r.residuals.to_vec(),
                    (Value::EtsResult(r), "level")    => r.level.to_vec(),
                    (Value::EtsResult(r), "trend")    => r.trend.to_vec(),
                    (Value::EtsResult(r), "seasonal") => r.seasonal.to_vec(),
                    (Value::EtsResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict ets: kind '{k}' unknown — use: fitted, residuals, level, trend, seasonal")
                    )),

                    // ── PanelThreshold ────────────────────────────────────────
                    (Value::ThresholdResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict pthresh: kind '{k}' — use print() para ver limiares e coeficientes")
                    )),

                    _ => return Err(HayashiError::Type(
                        "predict: tipo de modelo not supportado".into()
                    )),
                };

                let arr = ndarray::Array1::from(vals);
                Rc::make_mut(&mut df_val)
                    .insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                println!(
                    "({} obs)  {df}.{varname} ({kind_str}) predicted",
                    df_val.n_rows()
                );
                self.env.set(df, Value::DataFrame(df_val))?;
            }

            Stmt::Count { df, cond } => {
                let df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df}' is not a DataFrame"))),
                };
                let n = if let Some(cond_expr) = cond {
                    let mask = self.eval_col_expr(cond_expr, &df_val)?;
                    mask.iter().filter(|&&v| v != 0.0).count()
                } else {
                    df_val.n_rows()
                };
                println!("{n}");
            }

            Stmt::Replace {
                df,
                varname,
                expr,
                cond,
            } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df}' is not a DataFrame"))),
                };
                let new_vals = self.eval_col_expr(expr, &df_val)?;

                let final_vals: Vec<f64> = if let Some(cond_expr) = cond {
                    let mask = self.eval_col_expr(cond_expr, &df_val)?;
                    // lê coluna original para preservar onde mask == 0
                    use greeners::Column;
                    let old_vals: Vec<f64> = match df_val.get_column(varname) {
                        Ok(Column::Float(arr)) => arr.to_vec(),
                        Ok(Column::Int(arr)) => arr.iter().map(|&v| v as f64).collect(),
                        _ => vec![f64::NAN; new_vals.len()],
                    };
                    let n_replaced = mask.iter().filter(|&&m| m != 0.0).count();
                    println!("({n_replaced} real changes made)");
                    mask.into_iter()
                        .zip(old_vals)
                        .zip(new_vals)
                        .map(|((m, old), new)| if m != 0.0 { new } else { old })
                        .collect()
                } else {
                    let n = new_vals.len();
                    println!("({n} real changes made)");
                    new_vals
                };

                let arr = ndarray::Array1::from(final_vals);
                Rc::make_mut(&mut df_val)
                    .insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                self.env.set(df, Value::DataFrame(df_val))?;
            }

            Stmt::Generate { df, varname, expr } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df}' is not a DataFrame"))),
                };
                let vals = self.eval_col_expr(expr, &df_val)?;
                let arr = ndarray::Array1::from(vals);
                Rc::make_mut(&mut df_val)
                    .insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                println!("({} obs)  {df}.{varname} generated", df_val.n_rows());
                self.env.set(df, Value::DataFrame(df_val))?;
            }

            Stmt::Print(exprs, opts) => {
                let opt_map: HashMap<String, Value> = opts
                    .iter()
                    .map(|o| Ok((o.name.clone(), self.eval_expr(&o.value)?)))
                    .collect::<Result<_>>()?;
                let sep = match opt_map.get("sep") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => " ".to_string(),
                };
                let end = match opt_map.get("end") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "\n".to_string(),
                };
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        print!("{sep}");
                    }
                    let val = self.eval_expr(expr)?;
                    print!("{val}");
                }
                print!("{end}");
            }

            Stmt::Export { value, fmt, path } => {
                let val = self.eval_expr(value)?;
                let fmt_str = match self.eval_expr(fmt)? {
                    Value::Str(s) => s,
                    other => {
                        return Err(HayashiError::Type(format!(
                            "export format must be a string, got {other}"
                        )))
                    }
                };
                let path_str = match self.eval_expr(path)? {
                    Value::Str(s) => s,
                    _ => return Err(self.type_err("export path must be a string")),
                };

                use greeners::ExportableResult;

                let ext = path_str.rsplit('.').next().unwrap_or("").to_lowercase();
                let fmt_lower = fmt_str.to_lowercase();
                let effective_fmt = if fmt_lower == "auto" {
                    ext.as_str()
                } else {
                    fmt_lower.as_str()
                };

                match (val, effective_fmt) {
                    // ── DataFrame ─────────────────────────────────────────────
                    (Value::DataFrame(df), "csv" | "delimited") => {
                        df.to_csv(&path_str)
                            .map_err(|e| self.rt_err(e.to_string()))?;
                        println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                    }
                    (Value::DataFrame(df), "json") => {
                        df.to_json(&path_str)
                            .map_err(|e| self.rt_err(e.to_string()))?;
                        println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                    }
                    (Value::DataFrame(df), "tsv" | "tab") => {
                        crate::io::dsv::write_dsv(&df, &path_str, b'\t')?;
                        println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                    }
                    (Value::DataFrame(df), "xlsx" | "xls") => {
                        crate::io::excel::write_excel(&df, &path_str)?;
                        println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                    }
                    (Value::DataFrame(df), "sqlite" | "sqlite3" | "db") => {
                        crate::io::sqlite::write_sqlite(&df, &path_str, "data")?;
                        println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                    }
                    (Value::DataFrame(df), "parquet" | "pq") => {
                        crate::io::parquet::write_parquet(&df, &path_str)?;
                        println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                    }

                    // ── OLS → CSV / LaTeX / HTML ──────────────────────────────
                    (Value::OlsResult(m), "csv") => {
                        let content = m.result.to_csv();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e.to_string()))?;
                        println!("Exported OLS → '{path_str}'");
                    }
                    (Value::OlsResult(m), "latex" | "tex") => {
                        let content = m.result.to_latex();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e.to_string()))?;
                        println!("Exported OLS → '{path_str}'");
                    }
                    (Value::OlsResult(m), "html" | "htm") => {
                        let content = m.result.to_html();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e.to_string()))?;
                        println!("Exported OLS → '{path_str}'");
                    }

                    // ── Qualquer modelo → txt ─────────────────────────────────
                    (Value::IvResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e.to_string()))?;
                        println!("Exported IV results → '{path_str}'");
                    }
                    (Value::BinaryResult(m), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{m}"))
                            .map_err(|e| HayashiError::Io(e.to_string()))?;
                        println!("Exported logit/probit results → '{path_str}'");
                    }
                    (Value::PanelResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e.to_string()))?;
                        println!("Exported FE results → '{path_str}'");
                    }
                    (Value::ReResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e.to_string()))?;
                        println!("Exported RE results → '{path_str}'");
                    }
                    (
                        val @ (Value::PoissonResult(_)
                        | Value::NegBinResult(_)
                        | Value::TobitResult(_)
                        | Value::HeckmanResult(_)
                        | Value::CoxResult(_)
                        | Value::QuantileResult(_)),
                        "txt" | "text",
                    ) => {
                        std::fs::write(&path_str, format!("{val}"))
                            .map_err(|e| HayashiError::Io(e.to_string()))?;
                        println!("Exported results → '{path_str}'");
                    }

                    (_, fmt) => {
                        return Err(HayashiError::Runtime(format!(
                            "unsupported export format '{fmt}' for this value type\n\
                         DataFrame → csv, json, tsv, xlsx, sqlite\n\
                         OLS       → csv, latex, html\n\
                         Models    → txt"
                        )))
                    }
                }
            }

            Stmt::Tsset { df, t_var } => {
                let frame = match self
                    .env
                    .get(df)
                    .ok_or_else(|| self.rt_err(format!("'{df}' not defined")))?
                {
                    Value::DataFrame(d) => d.clone(),
                    _ => return Err(self.type_err(format!("'{df}' is not a DataFrame"))),
                };

                // ordena por t_var (sort_df_by reporta erro se coluna não existe)
                let sorted = Self::sort_df_by(&frame, t_var)?;

                // estatísticas da variável de tempo para o sumário
                let t_vals = self.eval_col_expr(&Expr::Var(t_var.clone()), &sorted)?;
                let t_min = t_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let t_max = t_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let n = sorted.n_rows();

                self.ts_info.insert(df.clone(), t_var.clone());
                self.env.set(df, Value::DataFrame(Rc::new(sorted)))?;

                println!("tsset {df}");
                println!("  variável de tempo : {t_var}  ({t_min} a {t_max})");
                println!("  n = {n}");
                println!();
            }

            // ── if / else ────────────────────────────────────────────────────
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond_val = self.eval_expr(cond)?;
                if Self::value_as_bool(&cond_val) {
                    self.env.push_scope();
                    for s in then_body {
                        self.exec(s)?;
                    }
                    self.env.pop_scope();
                } else if let Some(else_stmts) = else_body {
                    self.env.push_scope();
                    for s in else_stmts {
                        self.exec(s)?;
                    }
                    self.env.pop_scope();
                }
            }

            Stmt::TryCatch {
                try_body,
                error_var,
                catch_body,
            } => {
                self.env.push_scope();
                let mut caught = None;
                for s in try_body {
                    match self.exec(s) {
                        Ok(()) => {}
                        Err(
                            HayashiError::Return | HayashiError::Break | HayashiError::Continue,
                        ) => {
                            self.env.pop_scope();
                            return Err(HayashiError::Return);
                        }
                        Err(e) => {
                            caught = Some(format!("{e}"));
                            break;
                        }
                    }
                }
                self.env.pop_scope();
                if let Some(err_msg) = caught {
                    self.env.push_scope();
                    self.env.declare(error_var, Value::Str(err_msg))?;
                    for s in catch_body {
                        self.exec(s)?;
                    }
                    self.env.pop_scope();
                }
            }

            // ── for var in iter { ... } ───────────────────────────────────────
            Stmt::For { var, iter, body } => {
                macro_rules! run_body {
                    () => {{
                        let mut do_break = false;
                        self.env.push_scope();
                        for s in body {
                            match self.exec(s) {
                                Ok(()) => {}
                                Err(HayashiError::Continue) => break,
                                Err(HayashiError::Break) => {
                                    do_break = true;
                                    break;
                                }
                                Err(e) => {
                                    self.env.pop_scope();
                                    return Err(e);
                                }
                            }
                        }
                        self.env.pop_scope();
                        do_break
                    }};
                }
                match iter {
                    ForIter::Range(start_expr, end_expr) => {
                        let start = match self.eval_expr(start_expr)? {
                            Value::Int(i) => i,
                            Value::Float(f) => f as i64,
                            v => {
                                return Err(HayashiError::Type(format!(
                                    "for: início do range must be integer, não {v}"
                                )))
                            }
                        };
                        let end = match self.eval_expr(end_expr)? {
                            Value::Int(i) => i,
                            Value::Float(f) => f as i64,
                            v => {
                                return Err(HayashiError::Type(format!(
                                    "for: fim do range must be integer, não {v}"
                                )))
                            }
                        };
                        let step: i64 = if start <= end { 1 } else { -1 };
                        let mut cur = start;
                        while if step > 0 { cur < end } else { cur > end } {
                            self.env.set(var, Value::Int(cur))?;
                            if run_body!() {
                                break;
                            }
                            cur += step;
                        }
                    }
                    ForIter::RangeInclusive(start_expr, end_expr) => {
                        let start = match self.eval_expr(start_expr)? {
                            Value::Int(i) => i,
                            Value::Float(f) => f as i64,
                            v => {
                                return Err(HayashiError::Type(format!(
                                    "for: início do range must be integer, não {v}"
                                )))
                            }
                        };
                        let end = match self.eval_expr(end_expr)? {
                            Value::Int(i) => i,
                            Value::Float(f) => f as i64,
                            v => {
                                return Err(HayashiError::Type(format!(
                                    "for: fim do range must be integer, não {v}"
                                )))
                            }
                        };
                        let step: i64 = if start <= end { 1 } else { -1 };
                        let mut cur = start;
                        while if step > 0 { cur <= end } else { cur >= end } {
                            self.env.set(var, Value::Int(cur))?;
                            if run_body!() {
                                break;
                            }
                            cur += step;
                        }
                    }
                    ForIter::Items(iter_expr) => {
                        let items = match self.eval_expr(iter_expr)? {
                            Value::List(v) => (*v).clone(),
                            other => {
                                return Err(HayashiError::Type(format!(
                                    "for: iterador must be a list, não {other}"
                                )))
                            }
                        };
                        for item in items {
                            self.env.set(var, item)?;
                            if run_body!() {
                                break;
                            }
                        }
                    }
                }
            }

            // ── fn nome(params) { corpo } ────────────────────────────────────
            Stmt::Fn { name, params, body } => {
                self.env.set(
                    name,
                    Value::UserFn(Rc::new(UserFn {
                        params: params.clone(),
                        body: body.clone(),
                    })),
                )?;
            }

            // ── return [expr] ─────────────────────────────────────────────────
            Stmt::Return(expr) => {
                let val = match expr {
                    Some(e) => self.eval_expr(e)?,
                    None => Value::Nil,
                };
                self.return_value = Some(val);
                return Err(HayashiError::Return);
            }

            Stmt::Break => return Err(HayashiError::Break),
            Stmt::Continue => return Err(HayashiError::Continue),

            // ── while cond { ... } ───────────────────────────────────────────
            Stmt::While { cond, body } => 'outer: loop {
                let cond_val = self.eval_expr(cond)?;
                if !Self::value_as_bool(&cond_val) {
                    break;
                }
                self.env.push_scope();
                for s in body {
                    match self.exec(s) {
                        Ok(()) => {}
                        Err(HayashiError::Break) => {
                            self.env.pop_scope();
                            break 'outer;
                        }
                        Err(HayashiError::Continue) => break,
                        Err(e) => {
                            self.env.pop_scope();
                            return Err(e);
                        }
                    }
                }
                self.env.pop_scope();
            },

            Stmt::Expr(expr) => {
                if let Expr::Pipe {
                    source,
                    expr: inner,
                } = expr
                {
                    let val = self.eval_expr(inner)?;
                    if let Expr::Var(name) = source.as_ref() {
                        self.env.set(name, val)?;
                    }
                } else {
                    let val = self.eval_expr(expr)?;
                    if !matches!(val, Value::Nil) {
                        match &val {
                            Value::Str(v) => println!("\"{v}\""),
                            _ => println!("{val}"),
                        }
                    }
                }
            }

            Stmt::Block(stmts) => {
                self.env.push_scope();
                for s in stmts {
                    match self.exec(s) {
                        Ok(()) => {}
                        Err(e) => {
                            self.env.pop_scope();
                            return Err(e);
                        }
                    }
                }
                self.env.pop_scope();
            }
        }
        Ok(())
    }
}
