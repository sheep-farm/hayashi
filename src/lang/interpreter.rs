use std::collections::HashMap;
use std::rc::Rc;
use ndarray::{Array1, Array2};
use greeners::{DataFrame, Formula as GFormula, OLS, CovarianceType, IV, Logit, Probit, FixedEffects, RandomEffects};
use greeners::{t_pvalue_two, t_quantile, chi2_pvalue, f_pvalue, norm_pdf, logistic};
use greeners::diagnostics::Diagnostics;
use greeners::specification_tests::SpecificationTests;
use greeners::linalg::{LinalgInverse as _, LinalgEigh as _};
use greeners::linalg::UPLO;
use crate::lang::ast::*;
use crate::lang::error::{HayashiError, Result};

fn t_critical_95(df: f64) -> f64 { t_quantile(0.975, df) }

fn rd_kernel_opt(opt: Option<&Value>) -> std::result::Result<greeners::RdKernel, String> {
    match opt {
        None => Ok(greeners::RdKernel::Triangular),
        Some(Value::Str(s)) => match s.as_str() {
            "triangular"   | "tri" => Ok(greeners::RdKernel::Triangular),
            "uniform"      | "uni" => Ok(greeners::RdKernel::Uniform),
            "epanechnikov" | "epa" => Ok(greeners::RdKernel::Epanechnikov),
            other => Err(format!("kernel '{other}' desconhecido (triangular|uniform|epanechnikov)")),
        },
        _ => Err("kernel deve ser string".into()),
    }
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
        let thin  = "─".repeat(78);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^78}", " Seemingly Unrelated Regressions (SUR) ")?;
        writeln!(f, "{:^78}", "Zellner's Efficient Estimator")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " Cross-Equation Error Correlation (Σ):")?;
        for row in r.sigma_cross.rows() {
            write!(f, "  [")?;
            for v in row { write!(f, " {:>8.4}", v)?; }
            writeln!(f, " ]")?;
        }
        for (eq, vnames) in r.equations.iter().zip(self.eq_var_names.iter()) {
            writeln!(f, "\n{:-^78}", format!(" Equation: {} ", eq.name))?;
            writeln!(f, "{:<20} {:>10} {:>10} {:>8} {:>8}",
                     "Variable", "Coef", "Std Err", "t", "P>|t|")?;
            writeln!(f, "{thin}")?;
            for i in 0..eq.params.len() {
                let vname: &str = vnames.get(i).map(|s| s.as_str()).unwrap_or("?");
                writeln!(f, "{:<20} {:>10.4} {:>10.4} {:>8.3} {:>8.3}",
                         vname, eq.params[i], eq.std_errors[i],
                         eq.t_values[i], eq.p_values[i])?;
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
        let thin  = "─".repeat(62);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^62}", " Principal Component Analysis ")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " {:>20}  {:>10}", "Observações:", r.n_obs)?;
        writeln!(f, " {:>20}  {:>10}", "Componentes:", r.n_components)?;
        writeln!(f, " {:>20}  {:>10}", "Variáveis:", self.var_names.len())?;
        writeln!(f, "\n{:^12} {:>12} {:>12} {:>10}", "Componente", "Var Expl.", "% Acum.", "Eigenvalue")?;
        writeln!(f, "{thin}")?;
        let mut cum = 0.0;
        for i in 0..r.n_components {
            cum += r.explained_variance_ratio[i];
            writeln!(f, " PC{:<9} {:>12.4} {:>12.4} {:>10.4}",
                     i + 1,
                     r.explained_variance_ratio[i],
                     cum,
                     r.explained_variance[i])?;
        }
        writeln!(f, "\n{:^62}", " Loadings ")?;
        writeln!(f, "{thin}")?;
        let hdr: String = (0..r.n_components).map(|i| format!(" {:>8}", format!("PC{}", i+1))).collect();
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
        let thin  = "─".repeat(62);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^62}", " Factor Analysis (Principal Axis) ")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " {:>20}  {:>10}", "Observações:", r.n_obs)?;
        writeln!(f, " {:>20}  {:>10}", "Fatores:", r.n_factors)?;
        writeln!(f, "\n{:^62}", " Cargas Fatoriais (Loadings) ")?;
        writeln!(f, "{thin}")?;
        let hdr: String = (0..r.n_factors).map(|i| format!(" {:>8}", format!("F{}", i+1))).collect();
        writeln!(f, "{:<18}{hdr}  {:>10}", "Variável", "Comunalit.")?;
        for (j, vname) in self.var_names.iter().enumerate() {
            let row: String = (0..r.n_factors)
                .map(|i| format!(" {:>8.4}", r.loadings[[j, i]]))
                .collect();
            writeln!(f, "{:<18}{row}  {:>10.4}", vname, r.communalities[j])?;
        }
        writeln!(f, "\n{:<12} {:>10}", "Eigenvalues:", "")?;
        for (i, &ev) in r.eigenvalues.iter().enumerate() {
            writeln!(f, "  F{:<10} {:>10.4}", i+1, ev)?;
        }
        writeln!(f, "{thick}")
    }
}

// ── Função definida pelo usuário ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UserFn {
    pub params: Vec<String>,
    pub body:   Vec<Stmt>,
}

// ── Resultado de testes de diagnóstico (print-on-demand) ─────────────────────

#[derive(Debug, Clone)]
pub struct DiagResult {
    pub rendered: String,  // output pré-renderizado pelo teste
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
    DataFrame(DataFrame),
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
    List(Rc<Vec<Value>>),
    UserFn(Rc<UserFn>),
    Nil,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Float(v)        => write!(f, "{v}"),
            Value::Int(v)          => write!(f, "{v}"),
            Value::Bool(v)         => write!(f, "{v}"),
            Value::Str(v)          => write!(f, "{v}"),
            Value::DataFrame(df)   => write!(f, "{df}"),
            Value::OlsResult(m)    => write!(f, "{m}"),
            Value::IvResult(r)     => write!(f, "{r}"),
            Value::BinaryResult(m) => write!(f, "{m}"),
            Value::PanelResult(r)  => write!(f, "{r}"),
            Value::ReResult(r)     => write!(f, "{r}"),
            Value::ArimaResult(r)  => write!(f, "{r}"),
            Value::VarResult(r)    => write!(f, "{r}"),
            Value::VecmResult(r)   => write!(f, "{r}"),
            Value::GarchResult(r)  => write!(f, "{r}"),
            Value::DiagResult(r)   => write!(f, "{r}"),
            Value::AbResult(r)     => write!(f, "{r}"),
            Value::SysGmmResult(r) => write!(f, "{r}"),
            Value::FE2SLSResult(r)    => write!(f, "{r}"),
            Value::PcseResult(r)      => write!(f, "{r}"),
            Value::PanelGlsResult(r)  => write!(f, "{r}"),
            Value::TobitResult(r)     => write!(f, "{r}"),
            Value::HeckmanResult(r)   => write!(f, "{r}"),
            Value::RdResult(r)        => write!(f, "{r}"),
            Value::SynthResult(r)     => write!(f, "{r}"),
            Value::PsmResult(r)       => write!(f, "{r}"),
            Value::PoissonResult(r)   => write!(f, "{r}"),
            Value::NegBinResult(r)   => write!(f, "{r}"),
            Value::OrderedResult(r)  => write!(f, "{r}"),
            Value::MNLogitResult(r)  => write!(f, "{r}"),
            Value::DidResult(r)      => write!(f, "{r}"),
            Value::QuantileResult(r) => write!(f, "{r}"),
            Value::KMResult(r)       => write!(f, "{r}"),
            Value::CoxResult(r)      => write!(f, "{r}"),
            Value::RlmResult(r)           => write!(f, "{r}"),
            Value::GeeResult(r)           => write!(f, "{r}"),
            Value::ZeroInflatedResult(r)  => write!(f, "{r}"),
            Value::MixedResult(r)         => write!(f, "{r}"),
            Value::BetaResult(r)          => write!(f, "{r}"),
            Value::GlsarResult(r)         => write!(f, "{r}"),
            Value::SurResult(m)           => write!(f, "{m}"),
            Value::RollingResult(r)       => write!(f, "{r}"),
            Value::RecursiveLSResult(r)   => write!(f, "{r}"),
            Value::GlmResult(r)           => write!(f, "{r}"),
            Value::LowessResult(r)        => write!(f, "{r}"),
            Value::PcaResult(m)           => write!(f, "{m}"),
            Value::FactorResult(m)        => write!(f, "{m}"),
            Value::MarkovResult(r)        => write!(f, "{r}"),
            Value::ConditionalResult(r)   => write!(f, "{r}"),
            Value::VarmaResult(r)         => write!(f, "{r}"),
            Value::DecompResult(r)        => write!(f, "{r}"),
            Value::MstlResult(r)          => write!(f, "{r}"),
            Value::UCResult(r)            => write!(f, "{r}"),
            Value::GamResult(r)           => write!(f, "{r}"),
            Value::MiceResult(r)          => write!(f, "{r}"),
            Value::MSARResult(r)          => write!(f, "{r}"),
            Value::SVarResult(r)          => write!(f, "{r}"),
            Value::ThreeSLSResult(m)      => write!(f, "{m}"),
            Value::DFMResult(m)           => write!(f, "{m}"),
            Value::EtsResult(r)           => write!(f, "{r}"),
            Value::ThresholdResult(r)     => write!(f, "{r}"),
            Value::List(v) => {
                write!(f, "[")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Value::UserFn(f_) => write!(f, "<fn({})>", f_.params.join(", ")),
            Value::Nil             => write!(f, "nil"),
        }
    }
}

// ── Ambiente de variáveis ─────────────────────────────────────────────────────

pub struct Env {
    vars: HashMap<String, Value>,
}

impl Env {
    pub fn new() -> Self {
        Self { vars: HashMap::new() }
    }

    pub fn set(&mut self, name: &str, val: Value) {
        self.vars.insert(name.to_string(), val);
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }
}

// ── Interpetador ──────────────────────────────────────────────────────────────

pub struct Interpreter {
    pub env: Env,
    ts_info: HashMap<String, String>,
    // estrutura de painel declarada via xtset(df, id, time)
    panel_info: HashMap<String, (String, String)>,
    // valor capturado pelo Stmt::Return — consumido em eval_call
    return_value: Option<Value>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            env: Env::new(),
            ts_info: HashMap::new(),
            panel_info: HashMap::new(),
            return_value: None,
        }
    }

    // ── Avalia expressão ──────────────────────────────────────────────────────

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Float(v) => Ok(Value::Float(*v)),
            Expr::Int(v)   => Ok(Value::Int(*v)),
            Expr::Bool(v)  => Ok(Value::Bool(*v)),
            Expr::Str(v)   => Ok(Value::Str(v.clone())),

            Expr::Var(name) => {
                self.env.get(name)
                    .cloned()
                    .ok_or_else(|| HayashiError::Runtime(format!("undefined variable '{name}'")))
            }

            Expr::Formula(_f) => {
                Err(HayashiError::Runtime("formula must be used inside an estimator call".into()))
            }

            // ── Aritmética / lógica escalar ───────────────────────────────────
            Expr::BinOp { op, lhs, rhs } => {
                // Short-circuit para And/Or
                match op {
                    BinOp::And => {
                        let l = self.eval_expr(lhs)?;
                        if !Self::value_as_bool(&l) { return Ok(Value::Bool(false)); }
                        let r = self.eval_expr(rhs)?;
                        return Ok(Value::Bool(Self::value_as_bool(&r)));
                    }
                    BinOp::Or => {
                        let l = self.eval_expr(lhs)?;
                        if Self::value_as_bool(&l) { return Ok(Value::Bool(true)); }
                        let r = self.eval_expr(rhs)?;
                        return Ok(Value::Bool(Self::value_as_bool(&r)));
                    }
                    _ => {}
                }
                let l = self.eval_expr(lhs)?;
                let r = self.eval_expr(rhs)?;
                Self::eval_scalar_binop(op, l, r)
            }

            Expr::Neg(inner) => {
                match self.eval_expr(inner)? {
                    Value::Int(v)   => Ok(Value::Int(-v)),
                    Value::Float(v) => Ok(Value::Float(-v)),
                    _ => Err(HayashiError::Type("negação unária requer número".into())),
                }
            }

            Expr::Not(inner) => {
                let v = self.eval_expr(inner)?;
                Ok(Value::Bool(!Self::value_as_bool(&v)))
            }

            // ── Lista literal ─────────────────────────────────────────────────
            Expr::List(items) => {
                let vals: Vec<Value> = items.iter()
                    .map(|e| self.eval_expr(e))
                    .collect::<Result<_>>()?;
                Ok(Value::List(Rc::new(vals)))
            }

            // ── Indexação: lista[idx] ─────────────────────────────────────────
            Expr::Index { obj, idx } => {
                let obj_val = self.eval_expr(obj)?;
                let idx_val = self.eval_expr(idx)?;
                let i = match idx_val {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(HayashiError::Type("índice deve ser inteiro".into())),
                };
                match obj_val {
                    Value::List(v) => {
                        let len = v.len() as i64;
                        let real = if i < 0 { len + i } else { i };
                        if real < 0 || real >= len {
                            return Err(HayashiError::Runtime(
                                format!("índice {i} fora do intervalo (len={len})")
                            ));
                        }
                        Ok(v[real as usize].clone())
                    }
                    _ => Err(HayashiError::Type("indexação requer uma lista".into())),
                }
            }

            Expr::Call { func, args, opts } => {
                self.eval_call(func, args, opts)
            }

            Expr::Field { obj, field, args, opts } => {
                self.eval_field(obj, field, args, opts)
            }

            Expr::TsOp { .. } => Err(HayashiError::Runtime(
                "operadores L./F./D. só são válidos dentro de generate".into()
            )),
        }
    }

    // ── Converte fórmula AST → string Greeners ────────────────────────────────

    fn formula_to_string(f: &Formula) -> String {
        let rhs_parts: Vec<String> = f.rhs.iter().map(|t| match t {
            RhsTerm::Var(v)              => v.clone(),
            RhsTerm::Categorical(v)      => format!("C({v})"),
            RhsTerm::Transform(fn_, v)   => format!("{fn_}({v})"),
            RhsTerm::Interaction(a, b)   => format!("{a}:{b}"),
        }).collect();

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

    /// Extrai coluna como Array1<f64>; aceita Float e Int.
    fn get_col_f64(df: &DataFrame, name: &str) -> Result<ndarray::Array1<f64>> {
        use greeners::Column;
        let col = df.get_column(name)
            .map_err(|_| HayashiError::Runtime(format!("coluna '{name}' não encontrada")))?;
        match col {
            Column::Float(arr) => Ok(arr.clone()),
            Column::Int(arr)   => Ok(arr.mapv(|v| v as f64)),
            _ => Err(HayashiError::Type(format!("coluna '{name}' não é numérica"))),
        }
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
                    let col = Self::get_col_f64(df, other)
                        .map_err(|_| HayashiError::Runtime(
                            format!("predict: coluna '{other}' não encontrada no DataFrame")
                        ))?;
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
                "nonrobust" | "ols"  => Ok(CovarianceType::NonRobust),
                "robust"             => Ok(CovarianceType::HC1),
                "HC1"                => Ok(CovarianceType::HC1),
                "HC2"                => Ok(CovarianceType::HC2),
                "HC3"                => Ok(CovarianceType::HC3),
                "HC4"                => Ok(CovarianceType::HC4),
                other => Err(HayashiError::Type(format!("unknown covariance type '{other}'"))),
            },
            _ => Err(HayashiError::Type("cov= must be a string".into())),
        }
    }

    fn col_to_cluster_ids(df: &DataFrame, col: &str) -> Result<Vec<usize>> {
        let mut map: HashMap<i64, usize> = HashMap::new();
        let mut next = 0usize;
        if let Ok(arr) = df.get_int(col) {
            Ok(arr.iter().map(|&v| *map.entry(v).or_insert_with(|| { let id = next; next += 1; id })).collect())
        } else if let Ok(arr) = df.get(col) {
            Ok(arr.iter().map(|&v| { let key = v as i64; *map.entry(key).or_insert_with(|| { let id = next; next += 1; id }) }).collect())
        } else if let Ok(arr) = df.get_string(col) {
            let mut smap: HashMap<String, usize> = HashMap::new();
            Ok(arr.iter().map(|v| *smap.entry(v.clone()).or_insert_with(|| { let id = next; next += 1; id })).collect())
        } else {
            Err(HayashiError::Runtime(format!("cluster column '{col}' not found")))
        }
    }

    fn resolve_cov_full(opt_map: &HashMap<String, Value>, df: &DataFrame) -> Result<CovarianceType> {
        if let Some(Value::Str(cluster_col)) = opt_map.get("cluster") {
            let ids = Self::col_to_cluster_ids(df, cluster_col)?;
            if let Some(Value::Str(cluster2_col)) = opt_map.get("cluster2") {
                let ids2 = Self::col_to_cluster_ids(df, cluster2_col)?;
                Ok(CovarianceType::ClusteredTwoWay(ids, ids2))
            } else {
                Ok(CovarianceType::Clustered(ids))
            }
        } else if let Some(Value::Str(nw)) = opt_map.get("nw") {
            let lags: usize = nw.parse().unwrap_or_else(|_| (df.n_rows() as f64).powf(0.25) as usize);
            Ok(CovarianceType::NeweyWest(lags))
        } else if let Some(Value::Int(nw)) = opt_map.get("nw") {
            Ok(CovarianceType::NeweyWest(*nw as usize))
        } else {
            Self::resolve_cov(opt_map.get("cov"))
        }
    }

    // ── Funções built-in ──────────────────────────────────────────────────────

    fn eval_call(&mut self, func: &str, args: &[Expr], opts: &[Opt]) -> Result<Value> {
        // avalia opts primeiro (exceto "if"/"vars" — avaliados lazy pelos builtins)
        let opt_map: HashMap<String, Value> = opts.iter()
            .filter(|o| o.name != "if" && o.name != "vars")
            .map(|o| Ok((o.name.clone(), self.eval_expr(&o.value)?)))
            .collect::<Result<_>>()?;

        match func {
            // ── Builtins de lista ─────────────────────────────────────────────
            "len" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("len() requer exatamente 1 argumento".into()));
                }
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::List(lst) => Ok(Value::Int(lst.len() as i64)),
                    Value::Str(s)    => Ok(Value::Int(s.chars().count() as i64)),
                    _ => Err(HayashiError::Type("len() requer lista ou string".into())),
                }
            }

            // ── Funções de string ─────────────────────────────────────────────
            "upper" | "lower" | "trim" => {
                let s = match self.eval_expr(args.first().ok_or_else(||
                    HayashiError::Runtime(format!("{func}() requer 1 argumento")))?)? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(
                        format!("{func}() requer string, recebeu {v}")
                    )),
                };
                Ok(Value::Str(match func {
                    "upper" => s.to_uppercase(),
                    "lower" => s.to_lowercase(),
                    "trim"  => s.trim().to_string(),
                    _ => unreachable!(),
                }))
            }

            "contains" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("contains(s, padrão) requer 2 argumentos".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("contains: esperado string, recebeu {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("contains: padrão deve ser string, recebeu {v}"))),
                };
                Ok(Value::Bool(s.contains(pat.as_str())))
            }

            "starts_with" | "ends_with" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(format!("{func}(s, padrão) requer 2 argumentos")));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("{func}: esperado string, recebeu {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("{func}: padrão deve ser string, recebeu {v}"))),
                };
                Ok(Value::Bool(match func {
                    "starts_with" => s.starts_with(pat.as_str()),
                    "ends_with"   => s.ends_with(pat.as_str()),
                    _ => unreachable!(),
                }))
            }

            // substr(s, início [, comprimento]) — índice 0-based em chars
            "substr" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime(
                        "substr(s, início [, comprimento]) requer 2 ou 3 argumentos".into()
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("substr: esperado string, recebeu {v}"))),
                };
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len() as i64;
                let start = match self.eval_expr(&args[1])? {
                    Value::Int(i)   => i,
                    Value::Float(f) => f as i64,
                    v => return Err(HayashiError::Type(format!("substr: início deve ser inteiro, recebeu {v}"))),
                };
                let real_start = (if start < 0 { len + start } else { start }).clamp(0, len) as usize;
                let count = if args.len() == 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Int(i)   => i.max(0) as usize,
                        Value::Float(f) => f.max(0.0) as usize,
                        v => return Err(HayashiError::Type(format!("substr: comprimento deve ser inteiro, recebeu {v}"))),
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
                    return Err(HayashiError::Runtime("split(s, delimitador) requer 2 argumentos".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("split: esperado string, recebeu {v}"))),
                };
                let delim = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("split: delimitador deve ser string, recebeu {v}"))),
                };
                let parts: Vec<Value> = s.split(delim.as_str())
                    .map(|p| Value::Str(p.to_string()))
                    .collect();
                Ok(Value::List(Rc::new(parts)))
            }

            // str_replace(s, de, para) — "replace" é palavra-chave
            "str_replace" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime("str_replace(s, de, para) requer 3 argumentos".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("str_replace: esperado string, recebeu {v}"))),
                };
                let from = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("str_replace: 'de' deve ser string, recebeu {v}"))),
                };
                let to = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    v => return Err(HayashiError::Type(format!("str_replace: 'para' deve ser string, recebeu {v}"))),
                };
                Ok(Value::Str(s.replace(from.as_str(), to.as_str())))
            }

            // ── Conversões de tipo ────────────────────────────────────────────
            "str" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("str() requer 1 argumento".into()));
                }
                Ok(Value::Str(format!("{}", self.eval_expr(&args[0])?)))
            }

            "int" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("int() requer 1 argumento".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::Int(i)   => Ok(Value::Int(i)),
                    Value::Float(f) => Ok(Value::Int(f as i64)),
                    Value::Bool(b)  => Ok(Value::Int(if b { 1 } else { 0 })),
                    Value::Str(s)   => s.trim().parse::<i64>().map(Value::Int)
                        .map_err(|_| HayashiError::Runtime(
                            format!("int(): não foi possível converter \"{s}\" para inteiro")
                        )),
                    v => Err(HayashiError::Type(format!("int(): não converte {v}"))),
                }
            }

            "float" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("float() requer 1 argumento".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::Float(f) => Ok(Value::Float(f)),
                    Value::Int(i)   => Ok(Value::Float(i as f64)),
                    Value::Bool(b)  => Ok(Value::Float(if b { 1.0 } else { 0.0 })),
                    Value::Str(s)   => s.trim().parse::<f64>().map(Value::Float)
                        .map_err(|_| HayashiError::Runtime(
                            format!("float(): não foi possível converter \"{s}\" para número")
                        )),
                    v => Err(HayashiError::Type(format!("float(): não converte {v}"))),
                }
            }

            // ── Agregações sobre List ─────────────────────────────────────────
            // "sum" fica para summarize(df) — Stata-style
            // "total" é a soma de uma lista numérica
            "sum" | "mean" | "std" | "min" | "max" | "total" => {
                // Forma 1: mean(list)
                // Forma 2: mean(df, var)  ou  mean(df, var, if=cond)
                let nums: Vec<f64> = if args.len() >= 2 {
                    // forma DataFrame
                    let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type(format!("{func}: primeiro arg deve ser DataFrame"))) };
                    let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                    let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type(format!("{func}: segundo arg deve ser nome de variável"))) };
                    let col = Self::get_col_f64(&df, &var_name)?;
                    // filtro opcional: if=cond
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = Self::eval_col_expr(&cond_opt.value, &df)?;
                        col.iter().zip(mask.iter()).filter(|(_, &m)| m != 0.0).map(|(&v, _)| v).collect()
                    } else {
                        col.to_vec()
                    }
                } else if args.len() == 1 {
                    let v = self.eval_expr(&args[0])?;
                    match v {
                        Value::List(lst) => lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?,
                        other => return Err(HayashiError::Type(format!("{func}() requer lista numérica, recebeu {other}"))),
                    }
                } else {
                    return Err(HayashiError::Runtime(format!("{func}() requer pelo menos 1 argumento")));
                };
                if nums.is_empty() {
                    return Err(HayashiError::Runtime(format!("{func}(): nenhum valor (lista vazia ou filtro excluiu tudo)")));
                }
                let result = match func {
                    "sum" | "total" => nums.iter().sum::<f64>(),
                    "mean"  => nums.iter().sum::<f64>() / nums.len() as f64,
                    "min"   => nums.iter().cloned().fold(f64::INFINITY, f64::min),
                    "max"   => nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    "std"   => {
                        let n = nums.len() as f64;
                        let m = nums.iter().sum::<f64>() / n;
                        (nums.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
                    }
                    _ => unreachable!(),
                };
                Ok(Value::Float(result))
            }

            "push" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("push() requer (lista, item)".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let item = self.eval_expr(&args[1])?;
                match lst {
                    Value::List(v) => {
                        let mut new_v = (*v).clone();
                        new_v.push(item);
                        Ok(Value::List(Rc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("push() requer lista como primeiro argumento".into())),
                }
            }

            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime(
                        "range(start, end [, step]) requer 2 ou 3 argumentos".into()
                    ));
                }
                let start = match self.eval_expr(&args[0])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(HayashiError::Type("range: start deve ser inteiro".into())),
                };
                let end = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(HayashiError::Type("range: end deve ser inteiro".into())),
                };
                let step: i64 = if args.len() == 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Int(i) => i,
                        Value::Float(f) => f as i64,
                        _ => return Err(HayashiError::Type("range: step deve ser inteiro".into())),
                    }
                } else if start <= end { 1 } else { -1 };
                if step == 0 {
                    return Err(HayashiError::Runtime("range: step não pode ser zero".into()));
                }
                let mut v = Vec::new();
                let mut cur = start;
                while if step > 0 { cur < end } else { cur > end } {
                    v.push(Value::Int(cur));
                    cur += step;
                }
                Ok(Value::List(Rc::new(v)))
            }

            // ── OLS ───────────────────────────────────────────────────────────
            "ols" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("ols() requires (formula, dataframe)".into()));
                }
                let formula_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("first argument must be a formula".into())),
                };
                let df_name = match &args[1] {
                    Expr::Var(name) => name.clone(),
                    _ => return Err(HayashiError::Type("second argument must be a DataFrame variable".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' is not a DataFrame"))),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let cov = Self::resolve_cov_full(&opt_map, &df)?;

                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y, x) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = OLS::from_formula(&g_formula, &df, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let fitted = x.dot(&result.params);
                let residuals = &y - &fitted;

                Ok(Value::OlsResult(OlsModel {
                    result: Rc::new(result),
                    residuals,
                    x,
                }))
            }

            // ── IV / 2SLS ─────────────────────────────────────────────────────
            "iv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "iv() requires (endog_formula, instrument_formula, dataframe)".into(),
                    ));
                }
                let endog_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("first argument must be a formula".into())),
                };
                let instr_ast = match &args[1] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("second argument must be a formula".into())),
                };
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => return Err(HayashiError::Type("third argument must be a DataFrame variable".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' is not a DataFrame"))),
                };
                let cov = Self::resolve_cov_full(&opt_map, &df)?;

                let endog_str = Self::formula_to_string(&endog_ast);
                let instr_str = Self::formula_to_string(&instr_ast);

                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // A fórmula dos instrumentos pode ter LHS vazio (sintaxe ~ z1 + z2).
                // GFormula::parse rejeita LHS vazio; construímos diretamente.
                let g_instr = if instr_ast.lhs.is_empty() {
                    let independents: Vec<String> = instr_ast.rhs.iter().map(|t| match t {
                        RhsTerm::Var(v) => v.clone(),
                        RhsTerm::Categorical(v) => format!("C({v})"),
                        RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                        RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
                    }).collect();
                    GFormula { dependent: String::new(), independents, intercept: true }
                } else {
                    GFormula::parse(&instr_str)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
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
                        "weak_iv() requer (formula_estrutural, formula_instrumentos, df)".into()
                    ));
                }
                let endog_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("weak_iv(): primeiro arg deve ser fórmula".into())),
                };
                let instr_ast = match &args[1] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("weak_iv(): segundo arg deve ser fórmula de instrumentos".into())),
                };
                let df_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("weak_iv(): terceiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(HayashiError::Runtime(format!("weak_iv: '{df_name}' não é um DataFrame"))),
                };

                // ── Identifica variáveis ──
                let endog_vars: std::collections::HashSet<String> =
                    endog_ast.rhs.iter().map(|t| match t {
                        RhsTerm::Var(v) => v.clone(), _ => String::new()
                    }).filter(|s| !s.is_empty()).collect();
                let instr_vars: std::collections::HashSet<String> =
                    instr_ast.rhs.iter().map(|t| match t {
                        RhsTerm::Var(v) => v.clone(), _ => String::new()
                    }).filter(|s| !s.is_empty()).collect();

                // endógenas = em endog mas NÃO em instr
                let x_endog_names: Vec<String> = endog_ast.rhs.iter()
                    .filter_map(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .filter(|v| !instr_vars.contains(v))
                    .collect();
                // instrumentos excluídos = em instr mas NÃO em endog
                let z_excl_names: Vec<String> = instr_ast.rhs.iter()
                    .filter_map(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .filter(|v| !endog_vars.contains(v))
                    .collect();
                // exógenos incluídos = em ambos
                let x_exog_names: Vec<String> = instr_ast.rhs.iter()
                    .filter_map(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
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
                    let v = df.get(col).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n { x_exog[[i, j + 1]] = v[i]; }
                }

                // Z_excl: instrumentos excluídos  (n × L)
                let mut z_excl = Array2::<f64>::zeros((n, l));
                for (j, col) in z_excl_names.iter().enumerate() {
                    let v = df.get(col).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n { z_excl[[i, j]] = v[i]; }
                }

                // W = [X_exog | Z_excl]  (n × (k_exog + L))
                let mut w_full = Array2::<f64>::zeros((n, k_exog + l));
                w_full.slice_mut(ndarray::s![.., ..k_exog]).assign(&x_exog);
                w_full.slice_mut(ndarray::s![.., k_exog..]).assign(&z_excl);

                // X_endog: variáveis endógenas  (n × k_endog)
                let mut x_endog_mat = Array2::<f64>::zeros((n, k_endog));
                for (j, col) in x_endog_names.iter().enumerate() {
                    let v = df.get(col).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n { x_endog_mat[[i, j]] = v[i]; }
                }

                // ── M_exog = I - X_exog (X_exog'X_exog)⁻¹ X_exog' ──
                // para partial out os exógenos incluídos
                let xtx_exog = x_exog.t().dot(&x_exog);
                let xtx_exog_inv = xtx_exog.inv()
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
                let wtw_inv = wtw.inv()
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
                    let f_j = if sigma_vj > 1e-15 { numerator / sigma_vj } else { f64::NAN };
                    let p_j = if f_j.is_finite() {
                        f_pvalue(f_j, l as f64, df_fs as f64)
                    } else { f64::NAN };
                    first_stage_lines.push_str(&format!(
                        "   {:<20} F({},{}) = {:>10.3}   p = {:.4}\n",
                        x_endog_names[j], l, df_fs, f_j, p_j
                    ));
                }

                // ── Cragg-Donald Wald F ──
                let sigma_v_inv = sigma_v.inv()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cd_core = sigma_v_inv.dot(&cd_mat); // k_endog × k_endog

                let cd_stat = if k_endog == 1 {
                    cd_core[[0, 0]] / l as f64
                } else {
                    // λ_min de cd_core / L
                    let (eigenvalues, _) = cd_core.eigh(UPLO::Lower)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    eigenvalues[0] / l as f64 // eigenvalues em ordem crescente
                };

                // ── Valores críticos de Stock & Yogo (2005) (k_endog=1, bias TSLS) ──
                let sy_table: Vec<(usize, [f64; 4])> = vec![
                    (1,  [16.38, 8.96, 6.66, 5.53]),
                    (2,  [19.93, 11.59, 8.75, 7.25]),
                    (3,  [22.30, 12.83, 9.54, 7.80]),
                    (4,  [24.58, 13.96, 10.26, 8.31]),
                    (5,  [26.87, 15.09, 11.04, 8.84]),
                    (6,  [28.55, 16.00, 11.65, 9.23]),
                    (7,  [30.10, 16.87, 12.26, 9.63]),
                    (8,  [31.49, 17.60, 12.82, 10.00]),
                    (9,  [32.84, 18.37, 13.44, 10.37]),
                    (10, [34.16, 19.10, 14.01, 10.73]),
                ];
                let sy_line = if k_endog == 1 {
                    if let Some((_, cvs)) = sy_table.iter().find(|(lv, _)| *lv == l) {
                        format!(
                            "   Stock-Yogo (2005) — valores críticos para viés TSLS máximo (k_endog=1, L={}):\n   10%:{:.2}  15%:{:.2}  20%:{:.2}  25%:{:.2}\n",
                            l, cvs[0], cvs[1], cvs[2], cvs[3]
                        )
                    } else {
                        format!("   Stock-Yogo (2005): tabela disponível para L=1..10 (L={} fora do intervalo).\n   Regra de bolso (Staiger & Stock 1997): F > 10.\n", l)
                    }
                } else {
                    format!("   Stock-Yogo (2005): valores críticos para k_endog=1 apenas.\n   Para k_endog={}, consulte tabelas de Andrews, Stock & Sun (2019).\n", k_endog)
                };

                let thick = "═".repeat(70);
                let thin  = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Teste de Instrumentos Fracos\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(" n={n}  k_endog={k_endog}  L={l} (instrumentos excluídos)\n"));
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
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (_, x) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = Logit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let coef_names = Self::coef_names_from_formula(&formula_ast, &df, x.ncols());
                Ok(Value::BinaryResult(BinaryModel { result: Rc::new(result), x, kind: "logit".into(), coef_names }))
            }

            // ── Probit ────────────────────────────────────────────────────────
            "probit" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (_, x) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = Probit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let coef_names = Self::coef_names_from_formula(&formula_ast, &df, x.ncols());
                Ok(Value::BinaryResult(BinaryModel { result: Rc::new(result), x, kind: "probit".into(), coef_names }))
            }

            // ── Heckman Two-Step (Heckit) ─────────────────────────────────────
            // heckman(outcome_formula, select_formula, df)
            // outcome: y ~ x1 + x2       (estimado apenas nos obs selecionados)
            // select:  z ~ w1 + w2 + w3  (probit em todos os obs; z deve ser 0/1)
            "heckman" | "heckit" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "heckman() requer (formula_resultado, formula_seleção, df)".into()
                    ));
                }
                let out_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type(
                        "heckman(): primeiro argumento deve ser fórmula de resultado (y ~ x1+x2)".into()
                    )),
                };
                let sel_ast = match &args[1] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type(
                        "heckman(): segundo argumento deve ser fórmula de seleção (z ~ w1+w2)".into()
                    )),
                };
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => return Err(HayashiError::Type(
                        "heckman(): terceiro argumento deve ser nome do DataFrame".into()
                    )),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(HayashiError::Runtime(
                        format!("heckman: '{df_name}' não é um DataFrame")
                    )),
                };

                // Equação de resultado
                let out_str = Self::formula_to_string(&out_ast);
                let g_out = GFormula::parse(&out_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_out) = df.to_design_matrix(&g_out)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let out_names = df.formula_var_names(&g_out)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Equação de seleção
                let sel_str = Self::formula_to_string(&sel_ast);
                let g_sel = GFormula::parse(&sel_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (z_vec, x_sel) = df.to_design_matrix(&g_sel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let sel_names = df.formula_var_names(&g_sel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::Heckman::fit(
                    &y_vec, &x_out, &z_vec, &x_sel,
                    Some(out_names), Some(sel_names),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::HeckmanResult(Rc::new(result)))
            }

            // ── Tobit — MLE com censura esquerda ──────────────────────────────
            // tobit(formula, df [, ll=0])
            "tobit" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ll_limit = match opt_map.get("ll") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    None => 0.0,
                    _ => return Err(HayashiError::Runtime(
                        "tobit(): ll deve ser numérico".into()
                    )),
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
                        "rd() requer (formula, cutoff, df [, bw=..., poly=..., kernel=...])".into()
                    ));
                }
                let formula_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("rd(): primeiro arg deve ser fórmula".into())),
                };
                let cutoff = match self.eval_expr(&args[1])? {
                    Value::Float(v) => v,
                    Value::Int(v)   => v as f64,
                    _ => return Err(HayashiError::Type("rd(): segundo arg deve ser o cutoff (número)".into())),
                };
                let df = match self.eval_expr(&args[2])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("rd(): terceiro arg deve ser DataFrame".into())),
                };

                // Extrair nomes diretamente do AST da fórmula Hayashi
                let outcome_name = formula_ast.lhs.clone();
                let running_name = formula_ast.rhs.first()
                    .and_then(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .ok_or_else(|| HayashiError::Runtime(
                        "rd(): fórmula deve ter exatamente uma variável no lado direito (running var)".into()
                    ))?;

                let y = df.get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let x = df.get(&running_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();

                let bw = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v))   => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("rd: bw deve ser numérico".into())),
                };
                let poly = match opt_map.get("poly") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => return Err(HayashiError::Runtime("rd: poly deve ser inteiro".into())),
                };
                let kernel = rd_kernel_opt(opt_map.get("kernel"))
                    .map_err(|e| HayashiError::Runtime(e))?;

                let result = greeners::RD::fit(
                    &y, &x, cutoff, bw, poly, kernel,
                    Some((outcome_name, running_name)),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
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
                let formula_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("fuzzy_rd(): primeiro arg deve ser fórmula".into())),
                };
                let treatment_name = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type(
                        "fuzzy_rd(): segundo arg deve ser o nome da coluna de tratamento (string)".into()
                    )),
                };
                let cutoff = match self.eval_expr(&args[2])? {
                    Value::Float(v) => v,
                    Value::Int(v)   => v as f64,
                    _ => return Err(HayashiError::Type("fuzzy_rd(): terceiro arg deve ser cutoff (número)".into())),
                };
                let df = match self.eval_expr(&args[3])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("fuzzy_rd(): quarto arg deve ser DataFrame".into())),
                };

                let outcome_name = formula_ast.lhs.clone();
                let running_name = formula_ast.rhs.first()
                    .and_then(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .ok_or_else(|| HayashiError::Runtime(
                        "fuzzy_rd(): fórmula deve ter exatamente uma variável no lado direito (running var)".into()
                    ))?;

                let y = df.get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let d = df.get(&treatment_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let x = df.get(&running_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();

                let bw = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v))   => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("fuzzy_rd: bw deve ser numérico".into())),
                };
                let poly = match opt_map.get("poly") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => return Err(HayashiError::Runtime("fuzzy_rd: poly deve ser inteiro".into())),
                };
                let kernel = rd_kernel_opt(opt_map.get("kernel"))
                    .map_err(|e| HayashiError::Runtime(e))?;

                let result = greeners::RD::fit_fuzzy(
                    &y, &d, &x, cutoff, bw, poly, kernel,
                    Some((outcome_name, running_name, treatment_name)),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RdResult(Rc::new(result)))
            }

            // ── Propensity Score Matching (Rosenbaum & Rubin 1983) ───────────
            // psm(outcome ~ treatment + cov1 + cov2, df [, k=1, caliper=0.2, replace=false, boot=200])
            // O 1º termo RHS é o tratamento; demais são covariáveis para o PS.
            "psm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "psm() requer (formula, df [, k=..., caliper=..., replace=..., boot=...])".into()
                    ));
                }
                let formula_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("psm(): primeiro arg deve ser fórmula".into())),
                };
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("psm(): segundo arg deve ser DataFrame".into())),
                };

                let outcome_name = formula_ast.lhs.clone();
                // Primeiro RHS = tratamento; demais = covariáveis
                let mut rhs_names: Vec<String> = formula_ast.rhs.iter()
                    .filter_map(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .collect();
                if rhs_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "psm(): fórmula deve ter ao menos 'outcome ~ treatment'".into()
                    ));
                }
                let treatment_name = rhs_names.remove(0);
                let covariate_names = rhs_names;

                if covariate_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "psm(): forneça ao menos uma covariável: outcome ~ treatment + cov1 + ...".into()
                    ));
                }

                let y = df.get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?.to_owned();
                let d = df.get(&treatment_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?.to_owned();

                let x = {
                    let owned_cols: Vec<ndarray::Array1<f64>> = covariate_names.iter()
                        .map(|c| df.get(c)
                            .map(|a| a.to_owned())
                            .map_err(|e| HayashiError::Runtime(e.to_string())))
                        .collect::<Result<Vec<_>>>()?;
                    let views: Vec<ndarray::ArrayView1<f64>> =
                        owned_cols.iter().map(|a| a.view()).collect();
                    ndarray::stack(ndarray::Axis(1), &views)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                };

                let k = match opt_map.get("k") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => return Err(HayashiError::Runtime("psm: k deve ser inteiro".into())),
                };
                let caliper: Option<f64> = match opt_map.get("caliper") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v))   => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("psm: caliper deve ser numérico".into())),
                };
                let with_replacement = match opt_map.get("replace") {
                    Some(Value::Bool(b)) => *b,
                    None => false,
                    _ => return Err(HayashiError::Runtime("psm: replace deve ser booleano".into())),
                };
                let n_boot = match opt_map.get("boot") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 200,
                    _ => return Err(HayashiError::Runtime("psm: boot deve ser inteiro".into())),
                };

                let result = greeners::PSM::fit(
                    &y, &d, &x,
                    k, caliper, with_replacement, n_boot,
                    Some((outcome_name, treatment_name, covariate_names)),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
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
                let outcome_col = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type(
                        "synth(): primeiro arg deve ser nome da coluna de resultado (string)".into()
                    )),
                };
                let treated_unit = match self.eval_expr(&args[1])? {
                    Value::Str(s)  => s,
                    Value::Int(v)  => v.to_string(),
                    Value::Float(v) => (v as i64).to_string(),
                    _ => return Err(HayashiError::Type(
                        "synth(): segundo arg deve ser o ID da unidade tratada".into()
                    )),
                };
                let t0 = match self.eval_expr(&args[2])? {
                    Value::Float(v) => v,
                    Value::Int(v)   => v as f64,
                    _ => return Err(HayashiError::Type(
                        "synth(): terceiro arg deve ser o período de início do tratamento (número)".into()
                    )),
                };
                let df = match self.eval_expr(&args[3])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type(
                        "synth(): quarto arg deve ser DataFrame".into()
                    )),
                };

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(HayashiError::Runtime(
                        "synth(): opção id=coluna é obrigatória".into()
                    )),
                };
                let time_col = match opt_map.get("time") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(HayashiError::Runtime(
                        "synth(): opção time=coluna é obrigatória".into()
                    )),
                };
                let cov_cols: Option<Vec<String>> = match opt_map.get("covs") {
                    Some(Value::List(lst)) => Some(
                        lst.iter().map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type(
                                "synth(): covs deve ser lista de strings".into()
                            )),
                        }).collect::<Result<Vec<_>>>()?
                    ),
                    None => None,
                    _ => return Err(HayashiError::Runtime("synth(): covs deve ser lista".into())),
                };

                let result = greeners::SyntheticControl::fit(
                    &outcome_col,
                    &treated_unit,
                    t0,
                    &df,
                    &id_col,
                    &time_col,
                    cov_cols.as_deref(),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::SynthResult(Rc::new(result)))
            }

            // ── Poisson ───────────────────────────────────────────────────────
            "poisson" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let result = greeners::Poisson::fit_with_names(&y_vec, &x_mat, cov, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PoissonResult(Rc::new(result)))
            }

            // ── Negative Binomial (NB2) ───────────────────────────────────────
            "nbreg" | "negbin" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let result = greeners::NegBin::fit_with_names(&y_vec, &x_mat, cov, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::NegBinResult(Rc::new(result)))
            }

            // ── Ordered Logit ─────────────────────────────────────────────────
            "ologit" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::OrderedLogit::fit_with_names(&y_vec, &x_mat, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::OrderedResult(Rc::new(result)))
            }

            // ── Ordered Probit ────────────────────────────────────────────────
            "oprobit" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::OrderedProbit::fit_with_names(&y_vec, &x_mat, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::OrderedResult(Rc::new(result)))
            }

            // ── Multinomial Logit ─────────────────────────────────────────────
            "mlogit" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
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
                        "did(outcome ~ tratado + pos, df) requer fórmula e DataFrame".into()
                    ));
                }
                let formula_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("did(): primeiro arg deve ser fórmula".into())),
                };
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("did(): segundo arg deve ser DataFrame".into())),
                };
                // formula: outcome ~ treated_col + post_col
                let rhs_vars: Vec<&str> = formula_ast.rhs.iter().filter_map(|t| {
                    if let RhsTerm::Var(v) = t { Some(v.as_str()) } else { None }
                }).collect();
                if rhs_vars.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "did(): fórmula deve ter exatamente 2 variáveis no RHS: treated + post".into()
                    ));
                }
                let y       = Self::get_col_f64(&df, &formula_ast.lhs)?;
                let treated = Self::get_col_f64(&df, rhs_vars[0])?;
                let post    = Self::get_col_f64(&df, rhs_vars[1])?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let result = greeners::DiffInDiff::fit(&y, &treated, &post, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::DidResult(Rc::new(result)))
            }

            // ── Quantile Regression ───────────────────────────────────────────
            // qreg(y ~ x1 + x2, df, tau=0.5, boot=200)
            "qreg" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let tau = match opt_map.get("tau") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    None                  => 0.5,
                    _ => return Err(HayashiError::Type("tau= deve ser numérico".into())),
                };
                let n_boot = match opt_map.get("boot") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None                  => 200,
                    _ => return Err(HayashiError::Type("boot= deve ser inteiro".into())),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::QuantileReg::fit_with_names(&y_vec, &x_mat, tau, n_boot, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::QuantileResult(Rc::new(result)))
            }

            // ── Kaplan-Meier ──────────────────────────────────────────────────
            // km(time_col, event_col, df)
            "km" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "km(time, event, df) requer 3 argumentos".into()
                    ));
                }
                let time_name = match &args[0] {
                    Expr::Var(v) | Expr::Str(v) => v.clone(),
                    _ => return Err(HayashiError::Type("km(): primeiro arg deve ser nome da coluna de tempo".into())),
                };
                let event_name = match &args[1] {
                    Expr::Var(v) | Expr::Str(v) => v.clone(),
                    _ => return Err(HayashiError::Type("km(): segundo arg deve ser nome da coluna de evento".into())),
                };
                let df = match self.eval_expr(&args[2])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("km(): terceiro arg deve ser DataFrame".into())),
                };
                let times  = Self::get_col_f64(&df, &time_name)?;
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
                        "cox(time ~ x1 + x2, df, event=col) requer fórmula e DataFrame".into()
                    ));
                }
                let formula_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("cox(): primeiro arg deve ser fórmula".into())),
                };
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("cox(): segundo arg deve ser DataFrame".into())),
                };
                let event_col = match opt_map.get("event") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime(
                        "cox() requer opção event=nome_coluna".into()
                    )),
                    _ => return Err(HayashiError::Type("event= deve ser string".into())),
                };
                let times    = Self::get_col_f64(&df, &formula_ast.lhs)?;
                let events_f = Self::get_col_f64(&df, &event_col)?;
                let events: ndarray::Array1<u8> = events_f.iter().map(|&v| v as u8).collect();
                // build covariate matrix from RHS variables
                let rhs_vars: Vec<String> = formula_ast.rhs.iter().filter_map(|t| {
                    if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None }
                }).collect();
                if rhs_vars.is_empty() {
                    return Err(HayashiError::Runtime(
                        "cox(): fórmula precisa de ao menos uma covariável no RHS".into()
                    ));
                }
                let cols: Vec<ndarray::Array1<f64>> = rhs_vars.iter()
                    .map(|v| Self::get_col_f64(&df, v))
                    .collect::<Result<_>>()?;
                let n = times.len();
                let k = cols.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, col) in cols.iter().enumerate() {
                    x_mat.column_mut(j).assign(col);
                }
                let result = greeners::CoxPH::fit_with_names(&times, &events, &x_mat, Some(rhs_vars))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::CoxResult(Rc::new(result)))
            }

            // ── Robust Linear Model (M-estimadores) ───────────────────────────
            // rlm(y ~ x1 + x2, df, norm=huber|tukey|andrews|hampel, cov=HC3)
            // norm padrão: Huber (c=1.345)
            "rlm" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let norm = match opt_map.get("norm") {
                    None => greeners::RobustNorm::Huber(1.345),
                    Some(Value::Str(s)) => match s.as_str() {
                        "huber"               => greeners::RobustNorm::Huber(1.345),
                        "tukey" | "bisquare"  => greeners::RobustNorm::Tukey(4.685),
                        "andrews" | "wave"    => greeners::RobustNorm::AndrewWave(std::f64::consts::PI),
                        "hampel"              => greeners::RobustNorm::Hampel(2.0, 4.0, 8.0),
                        "ols" | "leastsq"     => greeners::RobustNorm::LeastSquares,
                        other => return Err(HayashiError::Runtime(
                            format!("norm='{other}' desconhecido — use: huber, tukey, andrews, hampel, ols")
                        )),
                    },
                    _ => return Err(HayashiError::Type("norm= deve ser string".into())),
                };
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let result = greeners::RLM::fit_with_names(&y_vec, &x_mat, &norm, cov, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RlmResult(Rc::new(result)))
            }

            // ── GEE (Generalized Estimating Equations) ────────────────────────
            // gee(y ~ x1 + x2, df, id=cluster_col, family=gaussian, corr=exchangeable)
            // family: gaussian (padrão), binomial, poisson
            // corr:   independence (padrão), exchangeable, ar1, unstructured
            "gee" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime(
                        "gee() requer opção id=coluna_grupo".into()
                    )),
                    _ => return Err(HayashiError::Type("id= deve ser string".into())),
                };
                let family_str = match opt_map.get("family") {
                    None => "gaussian",
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => "gaussian",
                        "binomial" | "logit"  => "binomial",
                        "poisson"             => "poisson",
                        other => return Err(HayashiError::Runtime(
                            format!("family='{other}' desconhecido — use: gaussian, binomial, poisson")
                        )),
                    },
                    _ => return Err(HayashiError::Type("family= deve ser string".into())),
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
                        format!("corr='{other}' desconhecido — use: independence, exchangeable, ar1, unstructured")
                    )),
                };
                let (family, link) = match family_str {
                    "binomial" => (greeners::Family::Binomial, greeners::Link::Logit),
                    "poisson"  => (greeners::Family::Poisson,  greeners::Link::Log),
                    _          => (greeners::Family::Gaussian, greeners::Link::Identity),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // converter coluna de id para índices de grupo (usize)
                let id_vals = Self::get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                let mut next_id = 0usize;
                let groups: ndarray::Array1<usize> = id_vals.iter().map(|&v| {
                    let key = v as i64;
                    *id_map.entry(key).or_insert_with(|| { let id = next_id; next_id += 1; id })
                }).collect();
                let result = greeners::GEE::fit_with_names(
                    &y_vec, &x_mat, &groups, &family, &link, &corr, Some(var_names)
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::GeeResult(Rc::new(result)))
            }

            // ── WLS (Weighted Least Squares) ──────────────────────────────────
            // wls(y ~ x1 + x2, df, weights="w_col", cov=HC3)
            "wls" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let w_name = match opt_map.get("weights") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime(
                        "wls() requer opção weights=\"coluna_pesos\"".into()
                    )),
                    _ => return Err(HayashiError::Type("weights= deve ser string".into())),
                };
                let weights = Self::get_col_f64(&df, &w_name)?;
                let cov = Self::resolve_cov_full(&opt_map, &df)?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::WLS::fit_with_names(&y, &x, &weights, cov, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let fitted = x.dot(&result.params);
                let residuals = &y - &fitted;
                Ok(Value::OlsResult(OlsModel { result: Rc::new(result), residuals, x }))
            }

            // ── ZIP / ZINB (Zero-Inflated Count Models) ───────────────────────
            // zip(y ~ x1 + x2, df)
            // zip(y ~ x1 + x2, df, inflate=["x3", "x4"])
            // zinb(y ~ x1 + x2, df)
            "zip" | "zinb" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_count) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let count_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // inflate= opcional: lista de nomes de colunas para a equação de inflação
                // Se omitido, usa a mesma matriz X do modelo de contagem
                let (x_inflate_opt, inflate_names_opt): (Option<ndarray::Array2<f64>>, Option<Vec<String>>) =
                    match opt_map.get("inflate") {
                        Some(Value::List(lst)) => {
                            let inames: Vec<String> = lst.iter().map(|v| match v {
                                Value::Str(s) => Ok(s.clone()),
                                _ => Err(HayashiError::Type("inflate= deve ser lista de strings".into())),
                            }).collect::<Result<_>>()?;
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
                        _ => return Err(HayashiError::Type("inflate= deve ser lista de strings".into())),
                    };

                let use_negbin = func == "zinb";
                let result = if use_negbin {
                    greeners::ZINB::fit_with_names(
                        &y_vec, &x_count, x_inflate_opt.as_ref(),
                        Some(count_names), inflate_names_opt,
                    ).map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else {
                    greeners::ZIP::fit_with_names(
                        &y_vec, &x_count, x_inflate_opt.as_ref(),
                        Some(count_names), inflate_names_opt,
                    ).map_err(|e| HayashiError::Runtime(e.to_string()))?
                };
                Ok(Value::ZeroInflatedResult(Rc::new(result)))
            }

            // ── MixedLM (Mixed Linear Models — efeitos mistos) ────────────────
            // mixed(y ~ x1 + x2, df, id="group")           # intercept aleatório
            // mixed(y ~ x1 + x2, df, id="group", re=["x1"]) # + slope aleatório
            "mixed" | "mixedlm" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // id= obrigatório: coluna de grupo
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime(
                        "mixed() requer opção id=\"coluna_grupo\"".into()
                    )),
                    _ => return Err(HayashiError::Type("id= deve ser string".into())),
                };

                // re= opcional: lista de variáveis com efeito aleatório de slope
                // Se omitido, modelo de intercept aleatório apenas (re = [1])
                let re_vars: Vec<String> = match opt_map.get("re") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type("re= deve ser lista de strings".into())),
                    }).collect::<Result<_>>()?,
                    None => vec![],
                    _ => return Err(HayashiError::Type("re= deve ser lista de strings".into())),
                };

                let (y_vec, x_fixed) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Converter id para índices de grupo
                let id_vals = Self::get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                let mut next_id = 0usize;
                let groups: ndarray::Array1<usize> = id_vals.iter().map(|&v| {
                    let key = v as i64;
                    *id_map.entry(key).or_insert_with(|| { let id = next_id; next_id += 1; id })
                }).collect();

                // Construir x_random: intercept + slopes especificados
                let n = df.n_rows();
                let q = re_vars.len() + 1; // +1 para intercept aleatório
                let mut x_random = ndarray::Array2::<f64>::ones((n, q));
                for (j, name) in re_vars.iter().enumerate() {
                    x_random.column_mut(j + 1).assign(&Self::get_col_f64(&df, name)?);
                }

                let result = greeners::MixedLM::fit_with_names(
                    &y_vec, &x_fixed, &groups, &x_random, Some(var_names),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::MixedResult(Rc::new(result)))
            }

            // ── testparm — Wald F-test conjunto (OLS/WLS) ────────────────────
            // testparm(model, ["x1", "x2"])
            // H0: β_x1 = β_x2 = 0 simultaneamente
            "testparm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "testparm(model, [\"x1\", \"x2\"]) requer modelo + lista de variáveis".into()
                    ));
                }
                let model_val = self.eval_expr(&args[0])?;
                let tested: Vec<String> = match self.eval_expr(&args[1])? {
                    Value::List(lst) => lst.iter().map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type("testparm: lista deve conter strings".into())),
                    }).collect::<Result<_>>()?,
                    _ => return Err(HayashiError::Type(
                        "testparm: segundo argumento deve ser lista de strings".into()
                    )),
                };
                match &model_val {
                    Value::OlsResult(m) => {
                        let vnames = m.result.variable_names.as_deref().unwrap_or(&[]);
                        let indices: Vec<usize> = tested.iter().map(|v| {
                            vnames.iter().position(|n| n == v)
                                .ok_or_else(|| HayashiError::Runtime(
                                    format!("testparm: variável '{v}' não encontrada no modelo")
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
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ar_order = match opt_map.get("ar") {
                    Some(Value::Int(n)) => *n as usize,
                    None => 1,
                    _ => return Err(HayashiError::Type("ar= deve ser inteiro".into())),
                };
                let max_iter = match opt_map.get("iter") {
                    Some(Value::Int(n)) => *n as usize,
                    None => 50,
                    _ => return Err(HayashiError::Type("iter= deve ser inteiro".into())),
                };
                let result = greeners::GLSAR::fit_with_names(
                    &y_vec, &x_mat, ar_order, max_iter, Some(var_names)
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::GlsarResult(Rc::new(result)))
            }

            // ── anova — ANOVA one-way ─────────────────────────────────────────
            // anova(df, outcome, by=group_col)
            "anova" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "anova(df, outcome, by=grupo)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro argumento deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let outcome_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("segundo argumento deve ser nome da variável outcome".into())),
                };
                let outcome = Self::get_col_f64(&df, &outcome_name)?;
                let by_col = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime(
                        "anova() requer by=\"coluna_grupo\"".into()
                    )),
                    _ => return Err(HayashiError::Type("by= deve ser string".into())),
                };
                let group_vals = Self::get_col_f64(&df, &by_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                let mut next_g = 0usize;
                let groups: ndarray::Array1<usize> = group_vals.iter().map(|&v| {
                    let key = v as i64;
                    *gmap.entry(key).or_insert_with(|| { let g = next_g; next_g += 1; g })
                }).collect();
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
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let link = match opt_map.get("link") {
                    None => greeners::BetaLink::Logit,
                    Some(Value::Str(s)) => match s.as_str() {
                        "logit"   => greeners::BetaLink::Logit,
                        "probit"  => greeners::BetaLink::Probit,
                        "cloglog" => greeners::BetaLink::CLogLog,
                        other => return Err(HayashiError::Runtime(
                            format!("betareg: link='{other}' desconhecido — use: logit, probit, cloglog")
                        )),
                    },
                    _ => greeners::BetaLink::Logit,
                };
                let result = greeners::BetaModel::fit_with_names(&y_vec, &x_mat, &link, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::BetaResult(Rc::new(result)))
            }

            // glm — Modelos Lineares Generalizados (IRLS via Greeners)
            // glm(y ~ x1 + x2, df, family=poisson, link=log, cov=robust)
            // Famílias: gaussian, binomial, poisson, gamma, inverse_gaussian, negbin, tweedie
            // Links: identity, log, logit, probit, inverse, cloglog
            // Se link omitido usa link canônico da família
            "glm" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
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
                            format!("glm: family='{other}' desconhecido — use: gaussian, binomial, poisson, gamma, inverse_gaussian, negbin, tweedie")
                        )),
                    },
                    _ => greeners::Family::Gaussian,
                };

                let result = match opt_map.get("link") {
                    None => greeners::GLM::fit_with_names(&y_vec, &x_mat, family, cov, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?,
                    Some(Value::Str(s)) => {
                        let link = match s.as_str() {
                            "identity"  => greeners::Link::Identity,
                            "log"       => greeners::Link::Log,
                            "logit"     => greeners::Link::Logit,
                            "probit"    => greeners::Link::Probit,
                            "inverse"   => greeners::Link::InversePower,
                            "cloglog"   => greeners::Link::CLogLog,
                            other => return Err(HayashiError::Runtime(
                                format!("glm: link='{other}' desconhecido — use: identity, log, logit, probit, inverse, cloglog")
                            )),
                        };
                        // fit_with_link não aceita var_names; setar após
                        let mut r = greeners::GLM::fit_with_link(&y_vec, &x_mat, family, link, cov)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        r.variable_names = Some(var_names);
                        r
                    }
                    _ => greeners::GLM::fit_with_names(&y_vec, &x_mat, family, cov, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?,
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
                        "lowess(df, y_var, x_var, frac=0.67, it=3)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("lowess: primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let y_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("lowess: segundo arg deve ser nome de coluna y".into())),
                };
                let x_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("lowess: terceiro arg deve ser nome de coluna x".into())),
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
                        "kde(df, var, bw=auto, kernel=gaussian)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("kde: primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("kde: segundo arg deve ser nome de coluna".into())),
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
                            format!("kde: kernel='{other}' desconhecido — use: gaussian, epanechnikov, triangular, uniform")
                        )),
                    },
                    _ => greeners::Kernel::Gaussian,
                };
                let result = greeners::KDEUnivariate::fit(&data, bw_opt, kernel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let support_min = result.support.iter().cloned().fold(f64::INFINITY, f64::min);
                let support_max = result.support.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let peak_idx = result.density.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let peak_x = result.support[peak_idx];
                let peak_d = result.density[peak_idx];
                println!("\n{:=^50}", " KDE ");
                println!("{:<20} {:>10}",   "Variável:", var_name);
                println!("{:<20} {:>10}",   "Observações:", result.n_obs);
                println!("{:<20} {:>10.6}", "Bandwidth:", result.bandwidth);
                println!("{:<20} {:>10.4}", "Suporte min:", support_min);
                println!("{:<20} {:>10.4}", "Suporte max:", support_max);
                println!("{:<20} {:>10.4} @ x = {:.4}", "Pico (densidade):", peak_d, peak_x);
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
                        "pca(df, x1, x2, x3, ..., n=k)".into()
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("pca: primeiro arg deve ser DataFrame".into())),
                };
                let var_names: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("pca: variáveis devem ser identificadores".into())),
                    })
                    .collect::<Result<_>>()?;
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
                    for (i, &v) in col.iter().enumerate() { data[[i, j]] = v; }
                }
                let result = greeners::PCA::fit(&data, n_components)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PcaResult(PcaModel { result: Rc::new(result), var_names }))
            }

            // factor — Análise Fatorial (eixo principal)
            // factor(df, x1, x2, x3, n=2, rotation=varimax)
            // rotation=: none (padrão), varimax
            // Diferença de PCA: PCA maximiza variância explicada;
            //   FA estima fatores latentes com estrutura de covariância específica
            "factor" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "factor(df, x1, x2, x3, ..., n=k, rotation=none|varimax)".into()
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("factor: primeiro arg deve ser DataFrame".into())),
                };
                let var_names: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("factor: variáveis devem ser identificadores".into())),
                    })
                    .collect::<Result<_>>()?;
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
                        "none"    => greeners::Rotation::None,
                        other => return Err(HayashiError::Runtime(
                            format!("factor: rotation='{other}' desconhecido — use: none, varimax")
                        )),
                    },
                    _ => greeners::Rotation::None,
                };
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() { data[[i, j]] = v; }
                }
                let result = greeners::FactorAnalysis::fit(&data, n_factors, rotation)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::FactorResult(FactorModel { result: Rc::new(result), var_names }))
            }

            // manova — Análise de Variância Multivariada (one-way)
            // manova(df, y1, y2, ..., by="group")
            // Testa H0: vetores de médias iguais entre grupos
            // Estatísticas: Wilks' Λ, Pillai's trace, Hotelling-Lawley, Roy's root
            "manova" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "manova(df, y1, y2, ..., by=\"group_col\")".into()
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("manova: primeiro arg deve ser DataFrame".into())),
                };
                let group_col = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime("manova requer by=\"coluna_grupo\"".into())),
                    _ => return Err(HayashiError::Type("manova: by= deve ser string".into())),
                };
                let outcome_names: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("manova: variáveis outcomes devem ser identificadores".into())),
                    })
                    .collect::<Result<_>>()?;
                let n = df.n_rows();
                let q = outcome_names.len();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, q));
                for (j, vname) in outcome_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() { y_mat[[i, j]] = v; }
                }
                let group_vals = Self::get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: ndarray::Array1<usize> = ndarray::Array1::from(
                    group_vals.iter().map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| { let g = gnext; gnext += 1; g })
                    }).collect::<Vec<_>>()
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
            "bootse" | "bootstrap" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("bootse(model, n=1000)".into()));
                }
                let model_val = self.eval_expr(&args[0])?;
                let n_boot = match opt_map.get("n") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1000,
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    _ => 0.05,
                };
                match &model_val {
                    Value::OlsResult(m) => {
                        // Extrair y do OLS: y = Xβ + ê
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
                        "bootse(): suportado apenas para modelos OLS/WLS — use: bootse(m_ols, n=1000)".into()
                    )),
                }
            }

            // markov — Markov-Switching AR (Hamilton 1989)
            // markov(df, y, k=2, p=1)
            // k=: número de regimes (padrão: 2)
            // p=: ordem AR dentro de cada regime (padrão: 1)
            // Algoritmo: EM via filtro de Hamilton (forward-backward)
            // Parâmetros por regime: intercept + AR coefficients + variance
            "markov" | "msar" | "markovswitching" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "markov(df, y_var, k=2, p=1)".into()
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("markov: primeiro arg deve ser DataFrame".into())),
                };
                let y_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("markov: segundo arg deve ser nome da variável".into())),
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
                    .map_err(|e| HayashiError::Runtime(format!("markov: {e}")))?;
                Ok(Value::MarkovResult(Rc::new(result)))
            }

            // clogit — Conditional Logit (Chamberlain 1980, FE logit)
            // clogit(y ~ x1 + x2, df, group="id_col")
            // Condiciona na soma de y por grupo → elimina efeitos fixos individuais
            // Grupos sem variação em y são automaticamente excluídos
            // Sem intercepto — absorvido pelo FE
            "clogit" | "xtlogit_fe" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let group_col = match opt_map.get("group") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime("clogit requer group=\"coluna_id\"".into())),
                    _ => return Err(HayashiError::Type("clogit: group= deve ser string".into())),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let group_vals = Self::get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: Vec<usize> = group_vals.iter().map(|&v| {
                    let key = v as i64;
                    *gmap.entry(key).or_insert_with(|| { let g = gnext; gnext += 1; g })
                }).collect();
                let result = greeners::ConditionalLogit::fit_with_names(
                    &y_vec, &x_mat, &groups, Some(var_names)
                ).map_err(|e| HayashiError::Runtime(format!("clogit: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // cpoisson — Conditional Poisson (FE Poisson)
            // cpoisson(y ~ x1 + x2, df, group="id_col")
            // Equivalente a FE Poisson; consistente sob heterogeidade não observada
            // Só requer que E[y|x,c] = exp(c + xβ) — não requer y ~ Poisson (PPML)
            "cpoisson" | "xtpoisson_fe" | "ppml" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let group_col = match opt_map.get("group") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime("cpoisson requer group=\"coluna_id\"".into())),
                    _ => return Err(HayashiError::Type("cpoisson: group= deve ser string".into())),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let group_vals = Self::get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: Vec<usize> = group_vals.iter().map(|&v| {
                    let key = v as i64;
                    *gmap.entry(key).or_insert_with(|| { let g = gnext; gnext += 1; g })
                }).collect();
                let result = greeners::ConditionalPoisson::fit_with_names(
                    &y_vec, &x_mat, &groups, Some(var_names)
                ).map_err(|e| HayashiError::Runtime(format!("cpoisson: {e}")))?;
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
                    _ => return Err(HayashiError::Type("gqtest(): suporta apenas modelos OLS".into())),
                };
                let split = match opt_map.get("split") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.2,
                };
                let (f, p, df1, df2) = greeners::SpecificationTests::goldfeld_quandt_test(
                    &ols.residuals, split
                ).map_err(|e| HayashiError::Runtime(format!("gqtest: {e}")))?;
                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(56);
                println!("\nGoldfeld-Quandt Test  —  split = {split:.2}");
                println!("{sep}");
                println!("H₀: homocedasticidade (σ²₁ = σ²₂)");
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}",
                         format!("F ~ F({df1},{df2})"), f, p, sig(p));
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
                    _ => return Err(HayashiError::Type("bphet(): suporta apenas modelos OLS".into())),
                };
                let (lm, p) = greeners::Diagnostics::breusch_pagan(&ols.residuals, &ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("bphet: {e}")))?;
                let k = ols.x.ncols().saturating_sub(1);
                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(56);
                println!("\nBreusch-Pagan Heteroskedasticity Test");
                println!("{sep}");
                println!("H₀: homocedasticidade (variância constante)");
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}",
                         format!("LM ~ χ²({k})"), lm, p, sig(p));
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
                        "bptest(df, y ~ x1+x2, id=\"entity_col\")".into()
                    ));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let formula_ast = match &args[1] { Expr::Formula(f) => f.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser fórmula".into())) };
                let id_col = match opt_map.get("id") { Some(Value::Str(s)) => s.clone(), _ => self.panel_info.get(&df_name).map(|(id,_)| id.clone()).filter(|s| !s.is_empty()).ok_or_else(|| HayashiError::Runtime(format!("bptest requer id= ou xtset({df_name}, id, time)")))? };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // OLS pooled para obter resíduos
                let ols_pooled = OLS::from_formula(&g_formula, &df, CovarianceType::NonRobust).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let resids = &y_vec - &x_mat.dot(&ols_pooled.params);
                // Converter id para usize
                let id_vals = Self::get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                let mut next_id = 0usize;
                let entity_ids: Vec<usize> = id_vals.iter().map(|&v| {
                    let key = v as i64;
                    *id_map.entry(key).or_insert_with(|| { let id = next_id; next_id += 1; id })
                }).collect();
                let (lm, p) = greeners::PanelDiagnostics::breusch_pagan_lm(&resids, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(e))?;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                println!("\n{:=^62}", " Breusch-Pagan LM Test (RE) ");
                println!(" H0: σ²_u = 0 — pooled OLS adequado");
                println!("{:-^62}", "");
                println!(" LM = {lm:.4}    p-valor = {p:.4}  {sig}");
                if p < 0.05 { println!(" Conclusão: rejeita H0 → usar RE ou FE"); }
                else        { println!(" Conclusão: não rejeita H0 → pooled OLS adequado"); }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // wooldridge — Teste de Wooldridge para correlação serial em painel
            // H0: sem correlação serial de 1ª ordem nos erros idiossincráticos
            // wooldridge(df, y ~ x1+x2, id="entity", time="time")
            "wooldridge" | "xtserial" | "wooldridge_serial" | "xtwooldridge" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("wooldridge(df, y~x, id=\"entity\", time=\"time\")".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let formula_ast = match &args[1] { Expr::Formula(f) => f.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser fórmula".into())) };
                let id_col   = match opt_map.get("id")   { Some(Value::Str(s)) => s.clone(), _ => self.panel_info.get(&df_name).map(|(id,_)| id.clone()).filter(|s| !s.is_empty()).ok_or_else(|| HayashiError::Runtime(format!("wooldridge requer id= ou xtset({df_name}, id, time)")))? };
                let time_col = match opt_map.get("time") { Some(Value::Str(s)) => s.clone(), _ => self.panel_info.get(&df_name).map(|(_,t)| t.clone()).filter(|s| !s.is_empty()).ok_or_else(|| HayashiError::Runtime(format!("wooldridge requer time= ou xtset({df_name}, id, time)")))? };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let id_vals: Vec<i64>   = Self::get_col_f64(&df, &id_col)?.iter().map(|&v| v as i64).collect();
                let time_vals: Vec<f64> = Self::get_col_f64(&df, &time_col)?.to_vec();
                let (rho, t_stat, p, n_pairs) = greeners::PanelDiagnostics::wooldridge_serial(&y_vec, &x_mat, &id_vals, &time_vals)
                    .map_err(|e| HayashiError::Runtime(e))?;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                println!("\n{:=^62}", " Wooldridge Test — Correlação Serial em Painel ");
                println!(" H0: ρ = -0.5 (sem correlação serial)");
                println!("{:-^62}", "");
                println!(" ρ̂ = {rho:.4}    t = {t_stat:.4}    p = {p:.4}  {sig}");
                println!(" Pares de resíduos: {n_pairs}");
                if p < 0.05 { println!(" Conclusão: rejeita H0 → correlação serial presente → usar SE robustos"); }
                else        { println!(" Conclusão: não rejeita H0 → sem evidência de correlação serial"); }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // pesaran — Pesaran CD test (cross-sectional dependence)
            // H0: sem dependência cross-sectional
            // pesaran(df, y ~ x1+x2, id="entity", time="time")
            "pesaran" | "xtcd" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("pesaran(df, y~x, id=\"entity\", time=\"time\")".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let formula_ast = match &args[1] { Expr::Formula(f) => f.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser fórmula".into())) };
                let id_col = match opt_map.get("id") { Some(Value::Str(s)) => s.clone(), _ => self.panel_info.get(&df_name).map(|(id,_)| id.clone()).filter(|s| !s.is_empty()).ok_or_else(|| HayashiError::Runtime(format!("pesaran requer id= ou xtset({df_name}, id, time)")))? };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols_pooled = OLS::from_formula(&g_formula, &df, CovarianceType::NonRobust).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let resids = &y_vec - &x_mat.dot(&ols_pooled.params);
                let id_vals = Self::get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                let mut next_id = 0usize;
                let entity_ids: Vec<usize> = id_vals.iter().map(|&v| {
                    let key = v as i64;
                    *id_map.entry(key).or_insert_with(|| { let id = next_id; next_id += 1; id })
                }).collect();
                let (cd, p) = greeners::PanelDiagnostics::pesaran_cd(&resids, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(e))?;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                println!("\n{:=^62}", " Pesaran CD Test — Dependência Cross-Sectional ");
                println!(" H0: sem dependência cross-sectional");
                println!("{:-^62}", "");
                println!(" CD = {cd:.4}    p-valor = {p:.4}  {sig}");
                if p < 0.05 { println!(" Conclusão: rejeita H0 → dependência CS presente → usar SE robustos por cluster"); }
                else        { println!(" Conclusão: não rejeita H0 → sem dependência CS detectada"); }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // mundlak — Teste de Mundlak (adequação de RE vs FE)
            // H0: médias de grupo não correlacionadas com regressores (RE ok)
            // mundlak(df, y ~ x1+x2, id="entity")
            "mundlak" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("mundlak(df, y~x, id=\"entity\")".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let formula_ast = match &args[1] { Expr::Formula(f) => f.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser fórmula".into())) };
                let id_col = match opt_map.get("id") { Some(Value::Str(s)) => s.clone(), _ => self.panel_info.get(&df_name).map(|(id,_)| id.clone()).filter(|s| !s.is_empty()).ok_or_else(|| HayashiError::Runtime(format!("mundlak requer id= ou xtset({df_name}, id, time)")))? };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let id_vals: Vec<i64> = Self::get_col_f64(&df, &id_col)?.iter().map(|&v| v as i64).collect();
                let (f_stat, p, k, gamma, gamma_se) = greeners::PanelDiagnostics::mundlak(&y_vec, &x_mat, &id_vals)
                    .map_err(|e| HayashiError::Runtime(e))?;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                println!("\n{:=^62}", " Mundlak Test — RE vs FE (correlação das médias) ");
                println!(" H0: γ = 0 (médias de grupo não correlacionadas com X → RE ok)");
                println!("{:-^62}", "");
                println!(" F({k}, .) = {f_stat:.4}    p = {p:.4}  {sig}");
                println!("{:-^62}", "");
                // Nomes das variáveis variantes no tempo (não-constantes)
                let slope_names: Vec<&str> = var_names.iter()
                    .filter(|n| n.as_str() != "_cons" && n.as_str() != "const")
                    .map(|s| s.as_str()).collect();
                println!(" {:<20} {:>10}  {:>10}", "Variável (γ̂)", "Coef", "Std Err");
                for i in 0..k.min(gamma.len()) {
                    let nm = slope_names.get(i).copied().unwrap_or("?");
                    println!(" {:<20} {:>10.4}  {:>10.4}", nm, gamma[i], gamma_se.get(i).copied().unwrap_or(f64::NAN));
                }
                if p < 0.05 { println!("\n Conclusão: rejeita H0 → RE é inconsistente → usar FE ou Hausman"); }
                else        { println!("\n Conclusão: não rejeita H0 → RE adequado"); }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // abtest — Arellano-Bond m1/m2 test (validação de instrumentos GMM)
            // abtest(df, y ~ x1+x2, id="entity", time="time")
            // m1 deve rejeitar H0 (FD induz AR(1) por construção)
            // m2 NÃO deve rejeitar H0 (valida instrumentos y_{i,t-2})
            "abtest" | "abar" | "abond" | "xtabond_test" | "arellano_bond" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("abtest(df, y~x, id=\"entity\", time=\"time\")".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let formula_ast = match &args[1] { Expr::Formula(f) => f.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser fórmula".into())) };
                let id_col   = match opt_map.get("id")   { Some(Value::Str(s)) => s.clone(), _ => self.panel_info.get(&df_name).map(|(id,_)| id.clone()).filter(|s| !s.is_empty()).ok_or_else(|| HayashiError::Runtime(format!("abtest requer id= ou xtset({df_name}, id, time)")))? };
                let time_col = match opt_map.get("time") { Some(Value::Str(s)) => s.clone(), _ => self.panel_info.get(&df_name).map(|(_,t)| t.clone()).filter(|s| !s.is_empty()).ok_or_else(|| HayashiError::Runtime(format!("abtest requer time= ou xtset({df_name}, id, time)")))? };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let id_vals: Vec<i64>   = Self::get_col_f64(&df, &id_col)?.iter().map(|&v| v as i64).collect();
                let time_vals: Vec<f64> = Self::get_col_f64(&df, &time_col)?.to_vec();
                let (m1, p1, m2, p2) = greeners::PanelDiagnostics::arellano_bond_test(&y_vec, &x_mat, &id_vals, &time_vals)
                    .map_err(|e| HayashiError::Runtime(e))?;
                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                println!("\n{:=^62}", " Arellano-Bond Test — Autocorrelação em 1ª Diferença ");
                println!(" m1 DEVE rejeitar H0 (AR(1) induzido por FD)");
                println!(" m2 NÃO deve rejeitar H0 (valida instrumentos y_{{t-2}})");
                println!("{:-^62}", "");
                println!(" m1 = {m1:.4}    p(m1) = {p1:.4}  {}", sig(p1));
                println!(" m2 = {m2:.4}    p(m2) = {p2:.4}  {}", sig(p2));
                println!("{:-^62}", "");
                if p1 >= 0.05 { println!(" [!] m1 não rejeita H0 — modelo pode estar mal especificado"); }
                if p2 <  0.05 { println!(" [!] m2 rejeita H0 — instrumentos y_{{t-2}} podem ser inválidos"); }
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
                        "sur(df, y1~x1+x2, y2~x3+x4, ...) requer df + ao menos 2 fórmulas".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro argumento deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let mut equations: Vec<greeners::SurEquation> = Vec::new();
                let mut eq_var_names: Vec<Vec<String>> = Vec::new();

                for arg in &args[1..] {
                    let formula_ast = match arg {
                        Expr::Formula(f) => f.clone(),
                        _ => return Err(HayashiError::Type(
                            "sur: cada equação deve ser uma fórmula (y ~ x1 + x2)".into()
                        )),
                    };
                    let formula_str = Self::formula_to_string(&formula_ast);
                    let g_formula = GFormula::parse(&formula_str)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let (y, x) = df.to_design_matrix(&g_formula)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let var_names = df.formula_var_names(&g_formula)
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
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let window = match opt_map.get("window") {
                    Some(Value::Int(n)) => *n as usize,
                    None => return Err(HayashiError::Runtime(
                        "rolling() requer window=N (ex: window=30)".into()
                    )),
                    _ => return Err(HayashiError::Type("window= deve ser inteiro".into())),
                };
                let result = greeners::RollingOLS::fit(&y_vec, &x_mat, window)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RollingResult(Rc::new(result)))
            }

            // ── Recursive OLS (Kalman, acumula observações) ───────────────────
            // recursive(y ~ x1 + x2, df)
            // Expande a janela de 1 em 1 — base para CUSUM e estabilidade
            "recursive" | "recols" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::RecursiveLS::fit(&y_vec, &x_mat)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RecursiveLSResult(Rc::new(result)))
            }

            // ── ic — tabela de critérios de informação (AIC/BIC) ──────────────
            // ic(m1, m2, m3, ...)
            // Compara modelos pelo AIC e BIC; ordena do menor (melhor) para maior
            "ic" | "fitstat" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("ic() requer ao menos um modelo".into()));
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
                    rows.push(IcRow { label, ll, k, n, aic, bic });
                }
                // Ordenar por AIC
                rows.sort_by(|a, b| a.aic.partial_cmp(&b.aic).unwrap_or(std::cmp::Ordering::Equal));
                let min_aic = rows.first().map(|r| r.aic).unwrap_or(0.0);
                let _min_bic = rows.iter().map(|r| r.bic).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
                println!("\n{:=^80}", " Critérios de Informação ");
                println!("{:<20} {:>6} {:>6} {:>12} {:>12} {:>8} {:>8}",
                         "Modelo", "N", "k", "Log-Lik", "AIC", "ΔAIC", "BIC");
                println!("{:-^80}", "");
                for row in &rows {
                    println!("{:<20} {:>6} {:>6} {:>12.4} {:>12.4} {:>8.4} {:>12.4}",
                             row.label, row.n, row.k, row.ll,
                             row.aic, row.aic - min_aic, row.bic);
                }
                if rows.len() > 1 {
                    println!("{:-^80}", "");
                    println!(" Melhor AIC: {}   Melhor BIC: {}",
                             rows.iter().min_by(|a,b| a.aic.partial_cmp(&b.aic).unwrap()).unwrap().label,
                             rows.iter().min_by(|a,b| a.bic.partial_cmp(&b.bic).unwrap()).unwrap().label);
                    // Pesos de Akaike
                    let delta_aics: Vec<f64> = rows.iter().map(|r| r.aic - min_aic).collect();
                    let rel: Vec<f64> = delta_aics.iter().map(|d| (-d / 2.0).exp()).collect();
                    let sum_rel: f64 = rel.iter().sum();
                    println!(" Pesos Akaike: {}",
                             rows.iter().zip(rel.iter()).map(|(r, w)| format!("{}={:.3}", r.label, w / sum_rel))
                                 .collect::<Vec<_>>().join("  "));
                }
                println!("{:=^80}", "");
                Ok(Value::Nil)
            }

            // ── Fixed Effects ─────────────────────────────────────────────────
            "fe" => {
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
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
                    return Err(HayashiError::Runtime(
                        format!("column '{id_col}' not found or not usable as entity ID"),
                    ));
                };

                Ok(Value::PanelResult(Rc::new(result)))
            }

            // ── Random Effects ────────────────────────────────────────────────
            "re" => {
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // aceita coluna float de valores inteiros (ex: idcode lido como f64)
                let ids_owned: ndarray::Array1<i64>;
                let ids = match df.get_int(&id_col) {
                    Ok(arr) => arr,
                    Err(_) => {
                        let floats = df.get(id_col.as_str())
                            .map_err(|_| HayashiError::Runtime(
                                format!("column '{id_col}' must be integer for re()"),
                            ))?;
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
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
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
                    return Err(HayashiError::Runtime(
                        format!("ftest_fe: coluna '{id_col}' não encontrada")
                    ));
                };

                let fe = FixedEffects::from_formula(&g_formula_fe, &df, &entity_ids_fe)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Pooled OLS (com intercepto)
                let g_formula_ols = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_pool, x_pool) = df.to_design_matrix(&g_formula_ols)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols = greeners::OLS::fit(&y_pool, &x_pool, greeners::CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let ssr_pooled = ols.sigma.powi(2) * ols.df_resid as f64;
                let ssr_fe     = fe.sigma.powi(2) * fe.df_resid as f64;
                let n          = fe.n_obs;
                let n_entities = fe.n_entities;
                let k          = fe.params.len();

                let (f_stat, p) = greeners::PanelDiagnostics::f_test_fixed_effects(
                    ssr_pooled, ssr_fe, n, n_entities, k
                ).map_err(|e| HayashiError::Runtime(e))?;

                let df_num   = n_entities - 1;
                let df_denom = n - n_entities - k;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos fixos individuais são significativos (use FE)"
                } else {
                    "Não rejeita H₀ → pooled OLS adequado (efeitos individuais não significativos)"
                };

                let thick = "═".repeat(62);
                let thin  = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" F-test: Efeitos Fixos vs Pooled OLS\n");
                out.push_str(" H₀: todos os efeitos individuais são zero\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Soma dos Quadrados dos Resíduos\n");
                out.push_str(&format!("   SSR pooled = {:.6}\n", ssr_pooled));
                out.push_str(&format!("   SSR FE     = {:.6}\n", ssr_fe));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!("   F({}, {}) = {:.4}   p = {:.4}  {}\n", df_num, df_denom, f_stat, p, sig));
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
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // OLS pooled para resíduos
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
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
                    return Err(HayashiError::Runtime(
                        format!("pesaran_cd: coluna '{id_col}' não encontrada")
                    ));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids { s.insert(id); }
                    s.len()
                };
                let t_bar = residuals.len() as f64 / n_entities as f64;

                let (cd, p) = greeners::PanelDiagnostics::pesaran_cd(&residuals, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(e))?;

                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → dependência cross-seccional presente"
                } else {
                    "Não rejeita H₀ → sem evidência de dependência cross-seccional"
                };

                let thick = "═".repeat(62);
                let thin  = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Pesaran CD Test (dependência cross-seccional)\n");
                out.push_str(" H₀: ρ_ij = 0 para todo i≠j  (resíduos independentes)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!("\n── Painel: N={} entidades   T̄≈{:.1}\n", n_entities, t_bar));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!("   CD ~ N(0,1) = {:.4}   p = {:.4}  {}\n", cd, p, sig));
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
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // OLS pooled para obter resíduos
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols = greeners::OLS::fit(
                    &y_vec,
                    &x_mat,
                    greeners::CovarianceType::NonRobust,
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let residuals = ols.residuals(&y_vec, &x_mat);

                // IDs de entidade → usize
                let entity_ids: Vec<usize> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.iter().map(|&v| v as usize).collect()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as usize).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("bplm: coluna '{id_col}' não encontrada ou não usável como ID")
                    ));
                };

                let n = residuals.len();
                let n_entities = {
                    let mut ids_set = std::collections::HashSet::new();
                    for &id in &entity_ids { ids_set.insert(id); }
                    ids_set.len()
                };
                let t_bar = n as f64 / n_entities as f64;

                let (lm, p) = greeners::PanelDiagnostics::breusch_pagan_lm(&residuals, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(e))?;

                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais presentes (use FE ou RE)"
                } else {
                    "Não rejeita H₀ → pooled OLS adequado (sem efeitos individuais)"
                };

                let thick = "═".repeat(62);
                let thin  = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Breusch-Pagan LM Test (efeitos individuais)\n");
                out.push_str(" H₀: σ²_u = 0  (sem efeitos individuais)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Dados do Painel\n");
                out.push_str(&format!("   n = {}   N = {}   T̄ ≈ {:.1}\n", n, n_entities, t_bar));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!("   LM ~ χ²(1) = {:.4}   p = {:.4}  {}\n", lm, p, sig));
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

                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("chamberlain: coluna id '{id_col}' não encontrada")
                    ));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("chamberlain: coluna time '{time_col}' não encontrada")
                    ));
                };

                let (f_stat, p, k_active, df_denom, n_entities, t_count) =
                    greeners::PanelDiagnostics::chamberlain(&y_vec, &x_mat, &entity_ids, &time_vals)
                        .map_err(|e| HayashiError::Runtime(e))?;

                let n_obs = y_vec.len();
                let df1 = k_active;

                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais correlacionados com X (prefira FE)"
                } else {
                    "Não rejeita H₀ → RE consistente (sem correlação period-específica)"
                };

                let thick = "═".repeat(70);
                let thin  = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Chamberlain Test (correlação period-específica com efeitos individuais)\n");
                out.push_str(" H₀: Π_s = 0 ∀s  (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!("\n── Painel: n={} obs   N={} entidades   T={} períodos\n", n_obs, n_entities, t_count));
                out.push_str(&format!("   Colunas de augmentação: {} de Chamberlain (k×T, após remover zero-variância)\n", k_active));
                if t_count > 6 {
                    out.push_str(&format!("   ⚠ T={} — com T grande o teste tem baixo poder em amostras finitas\n", t_count));
                }
                out.push_str("\n── Teste conjunto H₀: todos os Π_s = 0\n");
                out.push_str(&format!("   F({}, {}) = {:.4}   p = {:.4}  {}\n", df1, df_denom, f_stat, p, sig));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str("   Teste mais geral que Mundlak — inclui valores em todos os T períodos\n");
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Arellano-Bond Diff-GMM (OLD mundlak removed — use new mundlak above) ─
            "mundlak_OLD_REMOVED" => {
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("mundlak: coluna '{id_col}' não encontrada")
                    ));
                };

                // Nomes dos regressores variantes no tempo (excluindo "const")
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let non_const_names: Vec<&str> = var_names.iter()
                    .filter(|n| n.as_str() != "const")
                    .map(|s| s.as_str())
                    .collect();

                let n = y_vec.len();
                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids { s.insert(id); }
                    s.len()
                };

                let (f_stat, p, k, gamma_hat, gamma_se) =
                    greeners::PanelDiagnostics::mundlak(&y_vec, &x_mat, &entity_ids)
                        .map_err(|e| HayashiError::Runtime(e))?;

                let df1 = k;
                let df2_exact = if n > 2 * k + 1 { n - 2 * k - 1 } else { 1 };

                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais correlacionados com X (prefira FE)"
                } else {
                    "Não rejeita H₀ → RE consistente (sem evidência de correlação com efeitos)"
                };

                let thick = "═".repeat(70);
                let thin  = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Mundlak Test (correlação entre regressores e efeitos individuais)\n");
                out.push_str(" H₀: γ = 0  (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!("\n── Painel: n={} obs   N={} entidades   k={} regressores variantes\n", n, n_entities, k));
                out.push_str("\n── Coeficientes sobre médias individuais (X̄_i)\n");
                out.push_str(&format!("   {:<18} {:>10}  {:>10}  {:>8}\n", "Variável (X̄)", "γ̂", "SE", "t"));
                out.push_str(&format!("   {}\n", "─".repeat(52)));
                for i in 0..k {
                    let t_i = if gamma_se[i] > 1e-15 { gamma_hat[i] / gamma_se[i] } else { f64::NAN };
                    let name = non_const_names.get(i).copied().unwrap_or("?");
                    out.push_str(&format!("   {:<18} {:>10.4}  {:>10.4}  {:>8.3}\n",
                        format!("{}̄", name), gamma_hat[i], gamma_se[i], t_i));
                }
                out.push_str("\n── Teste conjunto H₀: γ = 0\n");
                out.push_str(&format!("   F({}, {}) = {:.4}   p = {:.4}  {}\n", df1, df2_exact, f_stat, p, sig));
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
                    _ => return Err(HayashiError::Runtime(
                        "ab(): lags deve ser inteiro positivo".into()
                    )),
                };

                let two_step: bool = match opt_map.get("step") {
                    Some(Value::Int(2)) => true,
                    Some(Value::Float(v)) if *v as i64 == 2 => true,
                    Some(Value::Int(_)) | Some(Value::Float(_)) | None => false,
                    _ => return Err(HayashiError::Runtime(
                        "ab(): step deve ser 1 ou 2".into()
                    )),
                };

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("ab: coluna id '{id_col}' não encontrada")
                    ));
                };

                let time_ids: Vec<i64> = if let Ok(ids) = df.get_int(&time_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&time_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("ab: coluna time '{time_col}' não encontrada")
                    ));
                };

                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::ArellanoBond::fit(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_ids,
                    max_lags,
                    two_step,
                    Some(var_names),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::AbResult(Rc::new(result)))
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
                    _ => return Err(HayashiError::Runtime(
                        "sysgmm(): lags deve ser inteiro positivo".into()
                    )),
                };

                let two_step: bool = match opt_map.get("step") {
                    Some(Value::Int(2)) => true,
                    Some(Value::Float(v)) if *v as i64 == 2 => true,
                    Some(Value::Int(_)) | Some(Value::Float(_)) | None => false,
                    _ => return Err(HayashiError::Runtime(
                        "sysgmm(): step deve ser 1 ou 2".into()
                    )),
                };

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("sysgmm: coluna id '{id_col}' não encontrada")
                    ));
                };

                let time_ids: Vec<i64> = if let Ok(ids) = df.get_int(&time_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&time_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("sysgmm: coluna time '{time_col}' não encontrada")
                    ));
                };

                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::SystemGmm::fit(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_ids,
                    max_lags,
                    two_step,
                    Some(var_names),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::SysGmmResult(Rc::new(result)))
            }

            // ── FE-2SLS (xtivreg, fe) — Hausman (1978) ───────────────────────
            // feiv(endog_formula, instrument_formula, df, id=col [, cov=...])
            // endog_formula: y ~ x1 + x2   (x2 é endógena)
            // instrument_formula: ~ x1 + z1 + z2  (exógenos incluídos + excluídos)
            "feiv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "feiv() requer (formula_estrutural, formula_instrumentos, df, id=col)".into()
                    ));
                }

                let endog_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type(
                        "feiv(): primeiro argumento deve ser fórmula estrutural (y ~ x1 + x2)".into()
                    )),
                };
                let instr_ast = match &args[1] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type(
                        "feiv(): segundo argumento deve ser fórmula dos instrumentos (~ x1 + z1 + z2)".into()
                    )),
                };
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => return Err(HayashiError::Type(
                        "feiv(): terceiro argumento deve ser nome do DataFrame".into()
                    )),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(HayashiError::Runtime(
                        format!("feiv: '{df_name}' não é um DataFrame")
                    )),
                };

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(HayashiError::Runtime(
                        "feiv(): opção id=col é obrigatória".into()
                    )),
                };

                // fórmula estrutural → y e X (sem constante, FE a absorve)
                let endog_str = Self::formula_to_string(&endog_ast);
                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // fórmula de instrumentos → Z (sem constante)
                let instr_vars: Vec<String> = instr_ast.rhs.iter().map(|t| match t {
                    RhsTerm::Var(v) => v.clone(),
                    RhsTerm::Categorical(v) => format!("C({v})"),
                    RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                    RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
                }).collect();

                let n = y_vec.len();
                let l = instr_vars.len();
                if l == 0 {
                    return Err(HayashiError::Runtime(
                        "feiv(): formula de instrumentos deve ter ao menos um instrumento".into()
                    ));
                }
                let mut z_mat = ndarray::Array2::<f64>::zeros((n, l));
                for (j, col_name) in instr_vars.iter().enumerate() {
                    let col = df.get(col_name)
                        .map_err(|_| HayashiError::Runtime(
                            format!("feiv: instrumento '{col_name}' não encontrado no DataFrame")
                        ))?;
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
                    return Err(HayashiError::Runtime(
                        format!("feiv: coluna id '{id_col}' não encontrada")
                    ));
                };

                let var_names = df.formula_var_names(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::FE2SLS::fit(
                    &y_vec,
                    &x_mat,
                    &z_mat,
                    &entity_ids,
                    Some(var_names),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;

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
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let entity_ids = Self::col_as_i64(&df, &id_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let time_ids = Self::col_as_i64(&df, &time_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::PCSE::fit(
                    &y_vec, &x_mat, &entity_ids, &time_ids, Some(var_names),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PcseResult(Rc::new(result)))
            }

            // ── Panel GLS — Parks (1967) / Stata xtgls ───────────────────────
            // xtgls(formula, df, id=col, time=col [, panels="hetero"|"corr"])
            "xtgls" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, &opt_map)?;
                let time_col = self.get_time_col(&df_name, &opt_map)?;
                let panels_opt = match opt_map.get("panels") {
                    Some(Value::Str(s)) if s == "corr" => greeners::GlsPanels::Correlated,
                    Some(Value::Str(s)) if s == "hetero" || s == "heteroscedastic" =>
                        greeners::GlsPanels::Hetero,
                    None => greeners::GlsPanels::Hetero,
                    _ => return Err(HayashiError::Runtime(
                        "xtgls(): panels deve ser \"hetero\" ou \"corr\"".into()
                    )),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let entity_ids = Self::col_as_i64(&df, &id_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let time_ids = Self::col_as_i64(&df, &time_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::PanelGLS::fit(
                    &y_vec, &x_mat, &entity_ids, &time_ids, panels_opt, Some(var_names),
                ).map_err(|e| HayashiError::Runtime(e.to_string()))?;
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

                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("ab_test: coluna id '{id_col}' não encontrada")
                    ));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("ab_test: coluna time '{time_col}' não encontrada")
                    ));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids { s.insert(id); }
                    s.len()
                };

                let (m1, p1, m2, p2) = greeners::PanelDiagnostics::arellano_bond_test(
                    &y_vec, &x_mat, &entity_ids, &time_vals
                ).map_err(|e| HayashiError::Runtime(e))?;

                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let n_obs = y_vec.len();

                let thick = "═".repeat(66);
                let thin  = "─".repeat(66);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Arellano-Bond Test (autocorrelação serial — resíduos em 1ª diferença)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!("\n── Painel: n={} obs   N={} entidades\n", n_obs, n_entities));
                out.push_str("\n── Estatísticas  z ~ N(0,1)   H₀: sem autocorrelação de ordem p\n");
                out.push_str(&format!("   {:-^52}\n", ""));
                out.push_str(&format!("   {:>4}  {:>10}  {:>10}  {:>6}  {}\n", "p", "z", "p-valor", "sig", "Interpretação"));
                out.push_str(&format!("   {:-^52}\n", ""));
                let interp1 = if p1 < 0.05 { "OK — FD induz AR(1) (esperado)" } else { "Inesperado — verificar modelo" };
                let interp2 = if p2 >= 0.05 { "OK — instrumentos válidos" } else { "Atenção — AR(2) detectado" };
                out.push_str(&format!("   {:>4}  {:>10.4}  {:>10.4}  {:>6}  {}\n", 1, m1, p1, sig(p1), interp1));
                out.push_str(&format!("   {:>4}  {:>10.4}  {:>10.4}  {:>6}  {}\n", 2, m2, p2, sig(p2), interp2));
                out.push_str(&format!("   {:-^52}\n", ""));
                out.push_str("\n── Conclusão\n");
                if p1 < 0.05 && p2 >= 0.05 {
                    out.push_str("   m1 rejeita e m2 não rejeita → estrutura consistente com GMM válido\n");
                } else if p1 >= 0.05 {
                    out.push_str("   m1 não rejeita H₀ → checar especificação (AR(1) esperado em FD)\n");
                } else {
                    out.push_str("   m2 rejeita H₀ → AR(2) nos resíduos; instrumentos y_{t-2} podem ser inválidos\n");
                    out.push_str("   Considere usar lags mais distantes (y_{t-3}, ...) como instrumentos\n");
                }
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str("   Variância estimada via sandwich (Σ_i dos produtos cruzados por entidade)\n");
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

                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("wooldridge: coluna id '{id_col}' não encontrada")
                    ));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(
                        format!("wooldridge: coluna time '{time_col}' não encontrada")
                    ));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids { s.insert(id); }
                    s.len()
                };

                let (rho, t_stat, p, n_pairs) = greeners::PanelDiagnostics::wooldridge_serial(
                    &y_vec, &x_mat, &entity_ids, &time_vals
                ).map_err(|e| HayashiError::Runtime(e))?;

                let df_t = n_entities - 1;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → autocorrelação serial de 1ª ordem presente"
                } else {
                    "Não rejeita H₀ → sem evidência de autocorrelação serial"
                };

                let thick = "═".repeat(62);
                let thin  = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Wooldridge Test (autocorrelação serial em painel)\n");
                out.push_str(" H₀: ρ = -0.5  (sem autocorrelação nos erros idiossincráticos)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!("\n── Painel: N={} entidades   pares usados={}   df={}\n", n_entities, n_pairs, df_t));
                out.push_str("\n── Estimativa\n");
                out.push_str(&format!("   ρ̂ = {:.4}   (H₀: ρ = -0.500)\n", rho));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!("   t({}) = {:.4}   p = {:.4}  {}\n", df_t, t_stat, p, sig));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str("   (SE padrão OLS — use SE robustos clusterizados para inferência formal)\n");
                out.push_str(&format!("{thick}\n"));
                Ok(Self::diag(out))
            }

            // ── Hausman FE vs RE ──────────────────────────────────────────────
            "hausman" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "hausman(fe_model, re_model)".into()
                    ));
                }

                let fe = match self.eval_expr(&args[0])? {
                    Value::PanelResult(r) => r,
                    _ => return Err(HayashiError::Type(
                        "hausman(): primeiro argumento deve ser um modelo FE".into()
                    )),
                };
                let re = match self.eval_expr(&args[1])? {
                    Value::ReResult(r) => r,
                    _ => return Err(HayashiError::Type(
                        "hausman(): segundo argumento deve ser um modelo RE".into()
                    )),
                };

                // Variáveis comuns: FE não tem intercepto; RE tem.
                // Alinha por nome quando disponível; senão assume mesma ordem.
                let fe_names: Vec<String> = fe.variable_names
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| (0..fe.params.len()).map(|i| format!("x{}", i)).collect());

                let re_names: Vec<String> = re.variable_names
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| (0..re.params.len()).map(|i| format!("x{}", i)).collect());

                // Pares (β_FE, σ²_FE, β_RE, σ²_RE) para variáveis em comum (exclui intercepto)
                let mut pairs: Vec<(String, f64, f64, f64, f64)> = Vec::new();
                for (i, fe_name) in fe_names.iter().enumerate() {
                    if fe_name == "const" { continue; }
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
                        "hausman: nenhuma variável comum entre FE e RE (verifique variable_names)".into()
                    ));
                }

                // H = Σ (β_FE - β_RE)² / (σ²_FE - σ²_RE)  para pares onde σ²_FE > σ²_RE
                let mut chi2 = 0.0;
                let mut df = 0usize;
                let mut skipped = 0usize;

                let thick = "═".repeat(62);
                let thin  = "─".repeat(62);
                let mut out = String::new();

                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Hausman Test: FE vs RE\n");
                out.push_str(" H₀: efeitos individuais não correlacionados com regressores (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Coeficientes Comuns\n");
                out.push_str(&format!("   {:<20} {:>10} {:>10} {:>10}\n", "Variável", "β_FE", "β_RE", "Δβ"));
                out.push_str(&format!("   {thin}\n"));

                for (name, bfe, vfe, bre, vre) in &pairs {
                    let diff = bfe - bre;
                    let dvar = vfe - vre;
                    out.push_str(&format!("   {:<20} {:>10.4} {:>10.4} {:>10.4}\n", name, bfe, bre, diff));
                    if dvar > 1e-15 {
                        chi2 += diff.powi(2) / dvar;
                        df += 1;
                    } else {
                        skipped += 1;
                    }
                }

                if df == 0 {
                    out.push_str("\n   [!] Var(β_FE) ≤ Var(β_RE) em todos os coeficientes.\n");
                    out.push_str("       Estatística indefinida — verifique especificação dos modelos.\n");
                    out.push_str(&format!("\n{thick}\n"));
                    return Ok(Self::diag(out));
                }

                let p = greeners::chi2_pvalue(chi2, df as f64);

                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → use EFEITOS FIXOS (RE pode ser inconsistente)"
                } else {
                    "Não rejeita H₀ → EFEITOS ALEATÓRIOS é consistente e eficiente"
                };

                out.push_str("\n── Estatística\n");
                out.push_str(&format!("   χ²({}) = {:.4}   p = {:.4}  {}\n", df, chi2, p, sig));
                if skipped > 0 {
                    out.push_str(&format!("   ({} coeficiente(s) excluídos: Var(β_FE) ≤ Var(β_RE))\n", skipped));
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
                    return Err(HayashiError::Runtime("test(model, name) requires 2 arguments".into()));
                }
                let model = self.eval_expr(&args[0])?;

                let ols = match &model {
                    Value::OlsResult(m) => m.clone(),
                    _ => return Err(HayashiError::Type(
                        "test() currently supports OLS models only".into(),
                    )),
                };

                let test_name = match &args[1] {
                    Expr::Var(s) => s.clone(),
                    Expr::Str(s) => s.clone(),
                    _ => return Err(HayashiError::Type("second argument must be a name or string".into())),
                };

                match test_name.as_str() {
                    // ── Specification tests ──────────────────────────────
                    "white" => {
                        match SpecificationTests::white_test(&ols.residuals, &ols.x) {
                            Ok((stat, p, df)) => {
                                println!("White Test for Heteroskedasticity");
                                println!("  LM statistic : {:.4}", stat);
                                println!("  p-value      : {:.4}", p);
                                println!("  df           : {}", df);
                                let verdict = if p < 0.05 { "Reject H0 — evidence of heteroskedasticity" }
                                              else { "Fail to reject H0 — no evidence of heteroskedasticity" };
                                println!("  Conclusion   : {}", verdict);
                            }
                            Err(e) => eprintln!("White test error: {e}"),
                        }
                    }
                    "bp" => {
                        match Diagnostics::breusch_pagan(&ols.residuals, &ols.x) {
                            Ok((stat, p)) => {
                                println!("Breusch-Pagan Test for Heteroskedasticity");
                                println!("  LM statistic : {:.4}", stat);
                                println!("  p-value      : {:.4}", p);
                                let verdict = if p < 0.05 { "Reject H0 — evidence of heteroskedasticity" }
                                              else { "Fail to reject H0 — no evidence of heteroskedasticity" };
                                println!("  Conclusion   : {}", verdict);
                            }
                            Err(e) => eprintln!("Breusch-Pagan test error: {e}"),
                        }
                    }
                    "dw" => {
                        let stat = Diagnostics::durbin_watson(&ols.residuals);
                        println!("Durbin-Watson Test for Autocorrelation");
                        println!("  DW statistic : {:.4}", stat);
                        let verdict = if stat < 1.5 { "Positive autocorrelation suspected" }
                                      else if stat > 2.5 { "Negative autocorrelation suspected" }
                                      else { "No strong evidence of autocorrelation" };
                        println!("  Conclusion   : {}", verdict);
                    }

                    // ── Wald / F-test sobre coeficientes ─────────────────
                    other => {
                        let names = ols.result.variable_names.as_ref()
                            .ok_or_else(|| HayashiError::Runtime("model has no variable names".into()))?;
                        let k = ols.result.params.len();
                        let find_idx = |name: &str| -> Result<usize> {
                            let n = name.trim();
                            names.iter().position(|v| v == n)
                                .or_else(|| if n == "_cons" || n == "const" { Some(k - 1) } else { None })
                                .ok_or_else(|| HayashiError::Runtime(format!("variable '{n}' not found in model")))
                        };

                        // "X1 = X2" ou "X1 = 0.5"
                        if let Some((lhs_s, rhs_s)) = other.split_once('=') {
                            let lhs_name = lhs_s.trim();
                            let rhs_trimmed = rhs_s.trim();
                            if let Ok(val) = rhs_trimmed.parse::<f64>() {
                                let idx = find_idx(lhs_name)?;
                                let mut r = ndarray::Array1::<f64>::zeros(k);
                                r[idx] = 1.0;
                                let (t, p) = ols.result.t_test(&r, val, &ols.x)
                                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                                println!("\n{:=^60}", " test ");
                                println!("  H₀: {lhs_name} = {val}");
                                println!("  t = {t:.4}   p = {p:.4}");
                                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                                println!("  {sig}");
                            } else {
                                let idx1 = find_idx(lhs_name)?;
                                let idx2 = find_idx(rhs_trimmed)?;
                                let mut r = ndarray::Array1::<f64>::zeros(k);
                                r[idx1] = 1.0;
                                r[idx2] = -1.0;
                                let (t, p) = ols.result.t_test(&r, 0.0, &ols.x)
                                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                                println!("\n{:=^60}", " test ");
                                println!("  H₀: {lhs_name} = {rhs_trimmed}");
                                println!("  t = {t:.4}   p = {p:.4}");
                                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                                println!("  {sig}");
                            }
                        } else {
                            // joint F-test: test(model, var1, var2, ...) — first var already parsed as `other`
                            let mut indices = vec![find_idx(other)?];
                            for arg in &args[2..] {
                                let name = match arg {
                                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                                    _ => match self.eval_expr(arg)? {
                                        Value::Str(s) => s,
                                        _ => return Err(HayashiError::Type("test() arguments must be variable names".into())),
                                    },
                                };
                                indices.push(find_idx(&name)?);
                            }
                            let (f, p) = ols.result.f_test(&indices, &ols.x)
                                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                            let var_list: Vec<&str> = indices.iter().map(|&i| names[i].as_str()).collect();
                            let q = indices.len();
                            println!("\n{:=^60}", " test ");
                            if q == 1 {
                                println!("  H₀: {} = 0", var_list[0]);
                            } else {
                                println!("  H₀: {} = 0", var_list.join(" = "));
                            }
                            println!("  F({q}, {}) = {f:.4}   p = {p:.4}", ols.result.df_resid);
                            let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                            println!("  {sig}");
                        }
                    }
                }

                Ok(Value::Nil)
            }

            // ── describe ─────────────────────────────────────────────────────
            "describe" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("describe() takes 1 argument".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => {
                        println!("{}", df);
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Type("describe() requires a DataFrame".into())),
                }
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
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                // variáveis pedidas ou todas as numéricas
                let names: Vec<String> = if args.len() > 1 {
                    args[1..].iter().map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable list must be identifiers".into())),
                    }).collect::<Result<_>>()?
                } else {
                    use greeners::Column;
                    let mut ns: Vec<String> = df.column_names().into_iter().filter(|n| {
                        matches!(df.get_column(n), Ok(Column::Float(_)) | Ok(Column::Int(_)))
                    }).collect();
                    ns.sort();
                    ns
                };

                if names.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "correlate() needs at least 2 numeric variables".into(),
                    ));
                }

                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                let sub = df.select(&refs)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let mat = sub.corr()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // corr() ordena colunas alfabeticamente — sincronizar com a matriz
                let mut sorted_names = names.clone();
                sorted_names.sort();

                let col_w = 10usize;
                let row_label_w = 16usize;
                let trunc = |s: &str, w: usize| {
                    if s.len() > w { s[..w].to_string() } else { s.to_string() }
                };

                // cabeçalho
                print!("{:>width$} |", "", width = row_label_w);
                for name in &sorted_names {
                    print!(" {:>width$}", trunc(name, col_w), width = col_w);
                }
                println!();
                println!("{}-+{}", "-".repeat(row_label_w), "-".repeat((col_w + 1) * sorted_names.len()));

                // triângulo inferior (estilo Stata)
                for (i, row_name) in sorted_names.iter().enumerate() {
                    print!("{:>width$} |", trunc(row_name, row_label_w), width = row_label_w);
                    for j in 0..=i {
                        print!(" {:>10.4}", mat[[i, j]]);
                    }
                    println!();
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
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                // variáveis pedidas (args[1..]) ou todas as colunas numéricas
                let requested: Vec<String> = if args.len() > 1 {
                    args[1..].iter().map(|a| match a {
                        Expr::Var(name) => Ok(name.clone()),
                        Expr::Str(s)    => Ok(s.clone()),
                        _ => Err(HayashiError::Type(
                            "summarize() variable list must be identifiers".into(),
                        )),
                    }).collect::<Result<_>>()?
                } else {
                    let mut names = df.column_names();
                    names.sort();
                    names
                };

                // cabeçalho
                println!(
                    "\n{:<16} {:>9}  {:>7}  {:>12} {:>12} {:>12} {:>12}",
                    "Variable", "Obs", "Missing", "Mean", "Std. Dev.", "Min", "Max"
                );
                println!("{}", "-".repeat(91));

                for name in &requested {
                    use greeners::Column;
                    let col = df.get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                    let (n_total, n_missing, vals): (usize, usize, Vec<f64>) = match col {
                        Column::Float(arr) => {
                            let total = arr.len();
                            let vals: Vec<f64> = arr.iter().copied().filter(|x| x.is_finite()).collect();
                            let missing = total - vals.len();
                            (total, missing, vals)
                        }
                        Column::Int(arr) => {
                            let vals: Vec<f64> = arr.iter().map(|&x| x as f64).collect();
                            (vals.len(), 0, vals)
                        }
                        _ => {
                            println!("{:<16} {:>9}  {:>7}", name, "(non-numeric)", "");
                            continue;
                        }
                    };

                    let n = vals.len();
                    if n == 0 {
                        println!("{:<16} {:>9}  {:>7}  (all missing)", name, 0, n_total);
                        continue;
                    }

                    let mean = vals.iter().sum::<f64>() / n as f64;
                    let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0).max(1.0);
                    let sd  = variance.sqrt();
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    let miss_str = if n_missing > 0 { format!("{}", n_missing) } else { String::new() };
                    println!(
                        "{:<16} {:>9}  {:>7}  {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                        name, n, miss_str, mean, sd, min, max
                    );
                }
                println!();
                Ok(Value::Nil)
            }

            // ── esttab ───────────────────────────────────────────────────────
            "esttab" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("esttab() requires at least one model".into()));
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
                // (label, coefs, n_obs)
                type ModelInfo = (String, Vec<CoefRow>, usize);

                // parseia CSV do OlsResult: variable,coef,se,t,p
                let parse_csv = |csv: &str| -> Vec<CoefRow> {
                    let mut rows = Vec::new();
                    let mut first = true;
                    for line in csv.lines() {
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        if first { first = false; continue; } // cabeçalho
                        let f: Vec<&str> = line.splitn(6, ',').collect();
                        if f.len() >= 5 {
                            let raw  = f[0].trim().trim_matches('"');
                            let name = if raw == "const" { "_cons".to_string() } else { raw.to_string() };
                            let coef = f[1].trim().parse::<f64>().unwrap_or(f64::NAN);
                            let se   = f[2].trim().parse::<f64>().unwrap_or(f64::NAN);
                            let p    = f[4].trim().parse::<f64>().unwrap_or(1.0);
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

                // extrai (label, coefs, n_obs) de tipos com campos padronizados
                let extract_std = |label: &str,
                                   vnames: &Option<Vec<String>>,
                                   params: &ndarray::Array1<f64>,
                                   se:     &ndarray::Array1<f64>,
                                   pv:     &ndarray::Array1<f64>,
                                   n: usize| -> ModelInfo {
                    let k = params.len();
                    let fb: Vec<String> = (0..k).map(|i| format!("x{i}")).collect();
                    let nm = vnames.as_ref().unwrap_or(&fb);
                    let coefs: Vec<CoefRow> = nm.iter()
                        .zip(params.iter())
                        .zip(se.iter())
                        .zip(pv.iter())
                        .map(|(((n, &c), &s), &p)| (n.clone(), c, Some(s), Some(p)))
                        .collect();
                    (label.to_string(), coefs, n)
                };

                let mut models: Vec<ModelInfo> = Vec::new();
                for arg in args {
                    let val = self.eval_expr(arg)?;
                    match val {
                        Value::OlsResult(m) => {
                            use greeners::ExportableResult;
                            let coefs = parse_csv(&m.result.to_csv());
                            let n = m.residuals.len();
                            models.push(("OLS".into(), coefs, n));
                        }
                        Value::BinaryResult(bm) => {
                            let label = if bm.kind == "logit" { "Logit" } else { "Probit" }.to_string();
                            let n = bm.x.nrows();
                            models.push(extract_std(&label, &bm.result.variable_names,
                                &bm.result.params, &bm.result.std_errors, &bm.result.p_values, n));
                        }
                        Value::IvResult(r) => {
                            models.push(extract_std("IV/2SLS", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::PoissonResult(r) => {
                            models.push(extract_std("Poisson", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::NegBinResult(r) => {
                            models.push(extract_std("NegBin", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::OrderedResult(r) => {
                            let mut info = extract_std(&r.model_name, &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs);
                            for (i, (&thr, &thr_se)) in r.thresholds.iter()
                                .zip(r.threshold_std_errors.iter()).enumerate()
                            {
                                info.1.push((format!("_cut{}", i + 1), thr, Some(thr_se), None));
                            }
                            models.push(info);
                        }
                        Value::TobitResult(r) => {
                            let mut info = extract_std("Tobit", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs);
                            info.1.push(("_sigma".into(), r.sigma, None, None));
                            models.push(info);
                        }
                        Value::HeckmanResult(r) => {
                            let mut info = extract_std("Heckman", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs);
                            let dz = if r.delta_se > 0.0 { r.delta / r.delta_se } else { f64::NAN };
                            let dp = if dz.is_finite() {
                                t_pvalue_two(dz, r.n_selected as f64)
                            } else { f64::NAN };
                            info.1.push(("_lambda".into(), r.delta, Some(r.delta_se), Some(dp)));
                            models.push(info);
                        }
                        Value::PanelResult(r) => {
                            models.push(extract_std("FE", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::ReResult(r) => {
                            models.push(extract_std("RE", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, 0));
                        }
                        Value::AbResult(r) => {
                            models.push(extract_std("AB-GMM", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::SysGmmResult(r) => {
                            models.push(extract_std("SysGMM", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs_fd));
                        }
                        Value::PcseResult(r) => {
                            models.push(extract_std("PCSE", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::PanelGlsResult(r) => {
                            let label = match r.panels {
                                greeners::panel::GlsPanels::Hetero      => "XTGLS-H",
                                greeners::panel::GlsPanels::Correlated  => "XTGLS-C",
                            };
                            models.push(extract_std(label, &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::FE2SLSResult(r) => {
                            models.push(extract_std("FE-IV", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::QuantileResult(r) => {
                            let label = format!("QReg(τ={:.2})", r.tau);
                            models.push(extract_std(&label, &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, 0));
                        }
                        Value::CoxResult(r) => {
                            models.push(extract_std("CoxPH", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::RlmResult(r) => {
                            models.push(extract_std("RLM", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::GeeResult(r) => {
                            // GEE usa SE robusto (sandwich) por padrão
                            models.push(extract_std("GEE", &r.variable_names,
                                &r.params, &r.robust_se, &r.p_values, r.n_obs));
                        }
                        Value::BetaResult(r) => {
                            models.push(extract_std("BetaReg", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::GlmResult(r) => {
                            let family_name = format!("GLM({:?})", r.family);
                            models.push(extract_std(&family_name, &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::LowessResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta lowess — use predict para extrair valores suavizados".into()
                            ));
                        }
                        Value::PcaResult(_) | Value::FactorResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta PCA/Factor — use print() para ver cargas e variância explicada".into()
                            ));
                        }
                        Value::ConditionalResult(r) => {
                            models.push(extract_std(&r.model_name, &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::MarkovResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta Markov Switching — use print() para ver parâmetros por regime".into()
                            ));
                        }
                        Value::GlsarResult(r) => {
                            models.push(extract_std("GLSAR", &r.variable_names,
                                &r.params, &r.std_errors, &r.p_values, r.n_obs));
                        }
                        Value::MixedResult(r) => {
                            // esttab exibe apenas efeitos fixos do MixedLM
                            models.push(extract_std("MixedLM", &r.variable_names,
                                &r.fixed_effects, &r.fixed_se, &r.p_values, r.n_obs));
                        }
                        Value::ZeroInflatedResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta zip/zinb (duas equações) — use print()".into()
                            ));
                        }
                        Value::SurResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta sur (múltiplas equações) — use print()".into()
                            ));
                        }
                        Value::RollingResult(_) | Value::RecursiveLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta rolling/recursive — coeficientes variam ao longo do tempo; use print()".into()
                            ));
                        }
                        Value::MNLogitResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta mlogit (múltiplas equações) — use print()".into()
                            ));
                        }
                        Value::DidResult(_) | Value::KMResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta did/km — resultado tem formato próprio; use print()".into()
                            ));
                        }
                        Value::RdResult(_) | Value::SynthResult(_) | Value::PsmResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta estimadores causais (rd, psm, synth) — use print()".into()
                            ));
                        }
                        Value::VarmaResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta VARMA (coeficientes matriciais) — use print()".into()
                            ));
                        }
                        Value::DecompResult(_) | Value::MstlResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta decomposição sazonal — use print()".into()
                            ));
                        }
                        Value::UCResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta UCM (parâmetros de variância, não β) — use print()".into()
                            ));
                        }
                        Value::GamResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta GAM (termos smooth não têm tabela β padrão) — use print()".into()
                            ));
                        }
                        Value::MiceResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta MICE (múltiplos datasets) — estime modelo em cada dataset e use Rubin's rules".into()
                            ));
                        }
                        Value::MSARResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta Markov-AR (parâmetros por regime) — use print()".into()
                            ));
                        }
                        Value::SVarResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta SVAR (matrizes A/B estruturais) — use print()".into()
                            ));
                        }
                        Value::ThreeSLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta 3SLS (múltiplas equações) — use print()".into()
                            ));
                        }
                        Value::DFMResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta DFM (fatores latentes) — use print()".into()
                            ));
                        }
                        Value::EtsResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta ETS (parâmetros de suavização) — use print()".into()
                            ));
                        }
                        Value::ThresholdResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() não suporta panel threshold (dois regimes) — use print()".into()
                            ));
                        }
                        _ => return Err(HayashiError::Type(
                            "esttab(): tipo de modelo não suportado — use print()".into()
                        )),
                    }
                }

                // união dos nomes de variáveis na ordem de primeira ocorrência
                let mut all_vars: Vec<String> = Vec::new();
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                for (_, coefs, _) in &models {
                    for (nm, _, _, _) in coefs {
                        if seen.insert(nm.clone()) { all_vars.push(nm.clone()); }
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
                    for _ in 0..n_models { buf.push_str("r"); }
                    buf.push_str("}\n\\hline\\hline\n");
                    // cabeçalho
                    buf.push_str(" &");
                    for (i, (label, _, _)) in models.iter().enumerate() {
                        buf.push_str(&format!(" ({}) {}", i + 1, label));
                        if i + 1 < n_models { buf.push('&'); }
                    }
                    buf.push_str(" \\\\\n\\hline\n");

                    for var in &all_vars {
                        if var == "_cons" { continue; } // _cons vai no final
                        buf.push_str(&format!("{var}"));
                        for (_, coefs, _) in &models {
                            let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                            match row {
                                Some((_, c, _, p)) => buf.push_str(&format!(" & {:.4}{}", c, stars(*p))),
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                        // SE linha
                        let has_se = models.iter().any(|(_, coefs, _)|
                            coefs.iter().find(|(nm, _, _, _)| nm == var)
                                 .and_then(|(_, _, se, _)| *se).is_some());
                        if has_se {
                            buf.push_str(" ");
                            for (_, coefs, _) in &models {
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
                        for (_, coefs, _) in &models {
                            let row = coefs.iter().find(|(nm, _, _, _)| nm == "_cons");
                            match row {
                                Some((_, c, _, p)) => buf.push_str(&format!(" & {:.4}{}", c, stars(*p))),
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                        let has_se = models.iter().any(|(_, coefs, _)|
                            coefs.iter().find(|(nm, _, _, _)| nm == "_cons")
                                 .and_then(|(_, _, se, _)| *se).is_some());
                        if has_se {
                            buf.push_str(" ");
                            for (_, coefs, _) in &models {
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
                    for (_, _, n) in &models { buf.push_str(&format!(" & {n}")); }
                    buf.push_str(" \\\\\n\\hline\\hline\n\\end{tabular}\n");
                    buf.push_str("\\footnotesize{* p$<$0.10, ** p$<$0.05, *** p$<$0.01}\n");

                } else {
                    // ── ASCII txt ─────────────────────────────────────────────
                    let sep = "─".repeat(total_w);

                    // cabeçalho: numeração
                    let mut line = format!("{:<lw$}", "", lw = label_w);
                    for i in 0..n_models { line.push_str(&format!(" {:>cw$}", format!("({})", i + 1), cw = col_w)); }
                    buf.push_str(&format!("{line}\n"));

                    // cabeçalho: labels
                    let mut line = format!("{:<lw$}", "", lw = label_w);
                    for (label, _, _) in &models { line.push_str(&format!(" {:>cw$}", label, cw = col_w)); }
                    buf.push_str(&format!("{line}\n"));
                    buf.push_str(&format!("{sep}\n"));

                    let print_var = |var: &str, buf: &mut String| {
                        // linha de coeficientes
                        let display_name = if var == "_cons" { "Constant" } else { var };
                        let mut line = format!("{:<lw$}", display_name, lw = label_w);
                        for (_, coefs, _) in &models {
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
                        let has_se = models.iter().any(|(_, coefs, _)|
                            coefs.iter().find(|(nm, _, _, _)| nm == var)
                                 .and_then(|(_, _, se, _)| *se).is_some());
                        if has_se {
                            let mut line = format!("{:<lw$}", "", lw = label_w);
                            for (_, coefs, _) in &models {
                                let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                                match row.and_then(|(_, _, se, _)| *se) {
                                    Some(se) => line.push_str(&format!(" {:>cw$}", format!("({:.4})", se), cw = col_w)),
                                    None      => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                                }
                            }
                            buf.push_str(&format!("{line}\n"));
                        }
                    };

                    for var in &all_vars {
                        if var == "_cons" { continue; }
                        print_var(var, &mut buf);
                    }
                    if all_vars.iter().any(|v| v == "_cons") {
                        print_var("_cons", &mut buf);
                    }

                    buf.push_str(&format!("{sep}\n"));
                    let mut line = format!("{:<lw$}", "N", lw = label_w);
                    for (_, _, n) in &models { line.push_str(&format!(" {:>cw$}", n, cw = col_w)); }
                    buf.push_str(&format!("{line}\n"));
                    buf.push_str(&format!("{sep}\n"));
                    buf.push_str("* p<0.10  ** p<0.05  *** p<0.01\n");
                }

                if let Some(path) = out_path {
                    std::fs::write(&path, &buf).map_err(|e| HayashiError::Io(e))?;
                    println!("Exported table → '{path}'");
                } else {
                    print!("\n{buf}");
                }

                Ok(Value::Nil)
            }

            // ── margins ──────────────────────────────────────────────────────
            "margins" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "margins() requer um modelo estimado como argumento".into(),
                    ));
                }
                let model = self.eval_expr(&args[0])?;

                let sep  = "─".repeat(60);
                let sep2 = "═".repeat(60);

                match model {
                    // ── Logit / Probit ────────────────────────────────────────
                    Value::BinaryResult(bm) => {
                        let n    = bm.x.nrows();
                        let k    = bm.x.ncols();
                        let beta = &bm.result.params;
                        let eta: Vec<f64> = (0..n).map(|i| bm.x.row(i).dot(beta)).collect();
                        let deriv: Vec<f64> = eta.iter().map(|&e| match bm.kind.as_str() {
                            "logit"  => { let p = logistic(e); p * (1.0 - p) }
                            "probit" => norm_pdf(e),
                            _        => 0.0,
                        }).collect();
                        println!("\n{sep2}");
                        println!(" Average Marginal Effects — {}", bm.kind.to_uppercase());
                        println!("{sep2}");
                        println!("{:<22} {:>14}  {:>10}", "Variable", "dy/dx", "");
                        println!("{sep}");
                        for k_idx in 0..k {
                            let name = bm.coef_names.get(k_idx).map(String::as_str).unwrap_or("?");
                            if name == "_cons" { continue; }
                            let ame: f64 = deriv.iter().map(|&d| d * beta[k_idx]).sum::<f64>() / n as f64;
                            println!("{:<22} {:>14.6}", name, ame);
                        }
                        println!("{sep}");
                        println!("n = {n}");
                        println!("{sep2}\n");
                    }

                    // ── Poisson / NegBin ──────────────────────────────────────
                    // AME_k = β_k * (1/n) Σ_i exp(X_iβ)  [derivada de E[y]=exp(Xβ)]
                    Value::PoissonResult(r) => {
                        let x    = r.x_data();
                        let n    = x.nrows();
                        let beta = &r.params;
                        let mu_bar: f64 = (0..n).map(|i| x.row(i).dot(beta).exp()).sum::<f64>() / n as f64;
                        let fb: Vec<String> = (0..beta.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        println!("\n{sep2}");
                        println!(" Average Marginal Effects — POISSON  (dy/dx = β·μ̄)");
                        println!("{sep2}");
                        println!("{:<22} {:>14}  {:>10}", "Variable", "dy/dx", "");
                        println!("{sep}");
                        for (k_idx, name) in names.iter().enumerate() {
                            if name == "_cons" || name == "const" { continue; }
                            if k_idx >= beta.len() { break; }
                            let ame = beta[k_idx] * mu_bar;
                            println!("{:<22} {:>14.6}", name, ame);
                        }
                        println!("{sep}");
                        println!("n = {n}   μ̄ = {mu_bar:.4}");
                        println!("{sep2}\n");
                    }
                    Value::NegBinResult(r) => {
                        let x    = r.x_data();
                        let n    = x.nrows();
                        let beta = &r.params;
                        let mu_bar: f64 = (0..n).map(|i| x.row(i).dot(beta).exp()).sum::<f64>() / n as f64;
                        let fb: Vec<String> = (0..beta.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        println!("\n{sep2}");
                        println!(" Average Marginal Effects — NEG. BINOMIAL  (dy/dx = β·μ̄)");
                        println!("{sep2}");
                        println!("{:<22} {:>14}  {:>10}", "Variable", "dy/dx", "");
                        println!("{sep}");
                        for (k_idx, name) in names.iter().enumerate() {
                            if name == "_cons" || name == "const" { continue; }
                            if k_idx >= beta.len() { break; }
                            let ame = beta[k_idx] * mu_bar;
                            println!("{:<22} {:>14.6}", name, ame);
                        }
                        println!("{sep}");
                        println!("n = {n}   μ̄ = {mu_bar:.4}  α = {:.4}", r.alpha);
                        println!("{sep2}\n");
                    }

                    // ── Ordered Logit / Probit ────────────────────────────────
                    // AME_k(Y=j) = (1/n) Σ_i [f(κ_{j-1} - X_iβ) - f(κ_j - X_iβ)] * β_k
                    // (com κ_0 = -∞ → f(κ_0 - ·) = 0;  κ_J = +∞ → f(κ_J - ·) = 0)
                    Value::OrderedResult(r) => {
                        let x     = r.x_data();
                        let n     = x.nrows();
                        let beta  = &r.params;
                        let cuts  = &r.thresholds;
                        let j     = r.n_categories;
                        let is_logit = r.model_name.to_lowercase().contains("logit");
                        let link_pdf = |u: f64| -> f64 {
                            if is_logit { let p = logistic(u); p * (1.0 - p) } else { norm_pdf(u) }
                        };
                        let fb: Vec<String> = (0..beta.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        // Xβ para cada observação
                        let xb: Vec<f64> = (0..n).map(|i| x.row(i).dot(beta)).collect();
                        // AME[var_k, cat_j]
                        let k = beta.len();
                        println!("\n{sep2}");
                        println!(" Average Marginal Effects — {}", r.model_name.to_uppercase());
                        println!(" dP(Y=j)/dx — um painel por categoria");
                        println!("{sep2}");
                        // header
                        print!("{:<22}", "Variable");
                        for cat_j in 0..j { print!("  {:>10}", format!("P(Y={})", cat_j + 1)); }
                        println!();
                        println!("{sep}");
                        for k_idx in 0..k {
                            let name = names.get(k_idx).map(String::as_str).unwrap_or("?");
                            if name == "_cons" || name == "const" { continue; }
                            print!("{:<22}", name);
                            for cat_j in 0..j {
                                // f(κ_{j-1} - Xβ) — zero para cat_j=0 (sem threshold inferior)
                                let f_lo: f64 = if cat_j == 0 {
                                    0.0
                                } else {
                                    (0..n).map(|i| link_pdf(cuts[cat_j - 1] - xb[i])).sum::<f64>() / n as f64
                                };
                                // f(κ_j - Xβ) — zero para cat_j=J-1 (sem threshold superior)
                                let f_hi: f64 = if cat_j == j - 1 {
                                    0.0
                                } else {
                                    (0..n).map(|i| link_pdf(cuts[cat_j] - xb[i])).sum::<f64>() / n as f64
                                };
                                let ame = (f_lo - f_hi) * beta[k_idx];
                                print!("  {:>10.5}", ame);
                            }
                            println!();
                        }
                        println!("{sep}");
                        println!("n = {n}   Categorias: {j}   Modelo: {}", r.model_name);
                        println!("{sep2}\n");
                    }

                    _ => return Err(HayashiError::Type(
                        "margins() suporta: logit, probit, poisson, negbin, ologit, oprobit".into()
                    )),
                }
                Ok(Value::Nil)
            }

            // ── vecm ─────────────────────────────────────────────────────────
            // vecm(df, y1, y2, ..., lags=2, rank=1)
            // rank = número de relações de cointegração (1 ≤ rank < k)
            "vecm" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "vecm() requer (dataframe, var1, var2, ..., lags=p, rank=r)".into()
                    ));
                }

                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("primeiro argumento deve ser um DataFrame".into())),
                };

                let var_names: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variáveis do VECM devem ser identificadores".into())),
                    })
                    .collect::<Result<_>>()?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let rank = match opt_map.get("rank") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                // monta matriz T×k
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::eval_col_expr(&Expr::Var(vname.clone()), &df)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }

                let result = greeners::VECM::fit(&data, lags, rank)
                    .map_err(|e| HayashiError::Runtime(format!("VECM: {e}")))?;

                Ok(Value::VecmResult(Rc::new(result)))
            }

            // ── var ──────────────────────────────────────────────────────────
            // var(df, y1, y2, ..., lags=2)
            "var" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "var() requer (dataframe, var1, var2, ..., lags=p)".into()
                    ));
                }

                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("primeiro argumento deve ser um DataFrame".into())),
                };

                // args[1..]: nomes das variáveis
                let var_names: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variáveis do VAR devem ser identificadores".into())),
                    })
                    .collect::<Result<_>>()?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                // monta matriz T×k
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::eval_col_expr(&Expr::Var(vname.clone()), &df)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }

                let result = greeners::VAR::fit(&data, lags, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(format!("VAR: {e}")))?;

                Ok(Value::VarResult(Rc::new(result)))
            }

            // ── irf ──────────────────────────────────────────────────────────
            // irf(model, steps=10)
            "irf" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("irf() requer um modelo VAR".into()));
                }
                let model = match self.eval_expr(&args[0])? {
                    Value::VarResult(m) => m,
                    _ => return Err(HayashiError::Type("irf() requer um modelo VAR".into())),
                };

                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let tensor = model.irf(steps)
                    .map_err(|e| HayashiError::Runtime(format!("IRF: {e}")))?;

                let k = model.n_vars;
                let names = &model.var_names;
                let sep = "─".repeat(14 + k * 12);

                println!("\nIRF — VAR({}) — {} passos", model.lags, steps);

                for j in 0..k {
                    println!("\n  Impulso: {}", names[j]);
                    println!("  {sep}");
                    let header: String = names.iter()
                        .map(|n| format!("{:>12}", n))
                        .collect::<Vec<_>>().join("");
                    println!("  {:>6}{header}", "h");
                    println!("  {sep}");
                    for h in 0..steps {
                        let row: String = (0..k)
                            .map(|i| format!("{:>12.4}", tensor[[h, i, j]]))
                            .collect::<Vec<_>>().join("");
                        println!("  {:>6}{row}", h + 1);
                    }
                    println!("  {sep}");
                }
                println!();

                Ok(Value::Nil)
            }

            // ── fevd ─────────────────────────────────────────────────────────
            // fevd(model, steps=10)
            "fevd" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("fevd() requer um modelo VAR".into()));
                }
                let model = match self.eval_expr(&args[0])? {
                    Value::VarResult(m) => m,
                    _ => return Err(HayashiError::Type("fevd() requer um modelo VAR".into())),
                };

                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let tensor = model.fevd(steps)
                    .map_err(|e| HayashiError::Runtime(format!("FEVD: {e}")))?;

                let k = model.n_vars;
                let names = &model.var_names;
                let col_w = names.iter().map(|n| n.len()).max().unwrap_or(8).max(8) + 2;
                let sep = "─".repeat(8 + k * col_w);

                println!("\nFEVD — VAR({}) — {} passos  (% da variância do erro de previsão)", model.lags, steps);

                for i in 0..k {
                    println!("\n  Variável: {}", names[i]);
                    println!("  {sep}");
                    let header: String = names.iter()
                        .map(|n| format!("{:>col_w$}", n))
                        .collect::<Vec<_>>().join("");
                    println!("  {:>6}{header}", "h");
                    println!("  {sep}");
                    for h in 0..steps {
                        let row: String = (0..k)
                            .map(|j| format!("{:>col_w$.1}%", tensor[[h, i, j]] * 100.0))
                            .collect::<Vec<_>>().join("");
                        println!("  {:>6}{row}", h + 1);
                    }
                    println!("  {sep}");
                }
                println!();

                Ok(Value::Nil)
            }

            // ── arima / sarima ───────────────────────────────────────────────
            // arima(df, varname, p=1, d=1, q=1)
            // sarima(df, varname, p=1, d=1, q=1, P=1, D=0, Q=1, s=12)
            "arima" | "sarima" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "arima() requer (dataframe, variável, p=, d=, q=)".into()
                    ));
                }

                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("primeiro argumento deve ser um DataFrame".into())),
                };

                let col_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("segundo argumento deve ser o nome da variável".into())),
                };

                // extrai série como Array1<f64>
                let y = Self::eval_col_expr(&Expr::Var(col_name.clone()), &df)?;
                let y = ndarray::Array1::from(y);

                // opts: p, d, q (ARIMA); P, D, Q, s (SARIMA)
                let get_usize = |key: &str, default: usize| -> usize {
                    match opt_map.get(key) {
                        Some(Value::Int(v))   => *v as usize,
                        Some(Value::Float(v)) => *v as usize,
                        _ => default,
                    }
                };

                let p = get_usize("p", 1);
                let d = get_usize("d", 1);
                let q = get_usize("q", 1);

                let result = if func == "sarima" {
                    let sp = get_usize("P", 0);
                    let sd = get_usize("D", 0);
                    let sq = get_usize("Q", 0);
                    let s  = get_usize("s", 12);
                    greeners::ARIMA::fit_sarimax(&y, (p, d, q), (sp, sd, sq, s), None)
                        .map_err(|e| HayashiError::Runtime(format!("SARIMA: {e}")))?
                } else {
                    greeners::ARIMA::fit(&y, (p, d, q))
                        .map_err(|e| HayashiError::Runtime(format!("ARIMA: {e}")))?
                };

                Ok(Value::ArimaResult(Rc::new(result)))
            }

            // ── forecast ─────────────────────────────────────────────────────
            // forecast(model, steps=8)
            // forecast(model, steps=8, alpha=0.05)
            "forecast" | "fcast" | "predict_h" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("forecast() requer um modelo ARIMA".into()));
                }

                let model = match self.eval_expr(&args[0])? {
                    Value::ArimaResult(m) => m,
                    _ => return Err(HayashiError::Type("forecast() requer um modelo ARIMA".into())),
                };

                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 8,
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    _ => 0.05,
                };

                let (fc, lo, hi) = model
                    .predict_with_ci(steps, None, alpha)
                    .map_err(|e| HayashiError::Runtime(format!("forecast: {e}")))?;

                let sep = "─".repeat(52);
                println!("\nForecast — {} passos à frente  (IC {}%)",
                         steps, ((1.0 - alpha) * 100.0) as usize);
                println!("{sep}");
                println!("{:<6} {:>12} {:>12} {:>12}", "h", "forecast", "lower", "upper");
                println!("{sep}");
                for h in 0..steps {
                    println!("{:<6} {:>12.4} {:>12.4} {:>12.4}", h + 1, fc[h], lo[h], hi[h]);
                }
                println!("{sep}");
                println!();

                Ok(Value::Nil)
            }

            // ── lincom ───────────────────────────────────────────────────────
            // lincom(model, var1=mult1, var2=mult2, ...)
            // Delega álgebra ao Greeners via OlsResult::t_test(r, q, x)
            "lincom" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("lincom() requires an OLS model".into()));
                }

                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type(
                        "lincom() suporta apenas modelos OLS".into()
                    )),
                };

                // nomes dos coeficientes via API do Greeners (sem parse de CSV)
                let var_names: Vec<String> = ols.result.variable_names
                    .clone()
                    .ok_or_else(|| HayashiError::Runtime(
                        "modelo sem variable_names — use from_formula".into()
                    ))?;

                let k = var_names.len();

                // monta vetor de contraste c alinhado com var_names
                // aceita "const" (Greeners) e "_cons" (Stata-compat) como aliases
                let mut c = Array1::<f64>::zeros(k);
                let mut found = false;
                for (idx, greeners_name) in var_names.iter().enumerate() {
                    let lookup = if greeners_name == "const" { "_cons" } else { greeners_name.as_str() };
                    let val = opt_map.get(lookup).or_else(|| opt_map.get(greeners_name.as_str()));
                    if let Some(v) = val {
                        let mult = match v {
                            Value::Float(f) => *f,
                            Value::Int(i)   => *i as f64,
                            _ => return Err(HayashiError::Type(
                                format!("{greeners_name}= deve ser numérico")
                            )),
                        };
                        c[idx] = mult;
                        found = true;
                    }
                }

                if !found {
                    let available: Vec<&str> = var_names.iter()
                        .map(|n| if n == "const" { "_cons" } else { n.as_str() })
                        .collect();
                    return Err(HayashiError::Runtime(
                        format!("nenhum coeficiente encontrado — disponíveis: {}", available.join(", "))
                    ));
                }

                // estimativa pontual c'β
                let estimate = c.dot(&ols.result.params);

                // inferência delegada ao Greeners: t_test usa (X'X)⁻¹σ² internamente
                let (t, p) = ols.result.t_test(&c, 0.0, &ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("lincom: {e}")))?;

                let se = if t.abs() > 1e-15 { estimate / t } else { 0.0 };
                let df_t = ols.result.df_resid as f64;
                let tc   = t_critical_95(df_t);

                // rótulo legível da combinação
                let display_name = |n: &str| if n == "const" { "_cons".to_string() } else { n.to_string() };
                let expr_label: String = var_names.iter().zip(c.iter())
                    .filter(|(_, &m)| m != 0.0)
                    .enumerate()
                    .map(|(i, (name, &mult))| {
                        let dname = display_name(name);
                        let term = if mult == 1.0 {
                            dname
                        } else if mult == -1.0 {
                            format!("-{dname}")
                        } else {
                            format!("{mult}*{dname}")
                        };
                        if i == 0 { term } else if mult < 0.0 { format!(" - {}", &term[1..]) } else { format!(" + {term}") }
                    })
                    .collect();

                let sep = "─".repeat(64);
                println!("\nlincom: {expr_label}");
                println!("{sep}");
                println!("{:<12} {:>10} {:>10} {:>8} {:>10}", "Estimate", "Std.Err.", "t", "df", "p");
                println!("{sep}");
                println!("{:<12.6} {:>10.6} {:>10.4} {:>8.1} {:>10.4}",
                         estimate, se, t, df_t, p);
                println!("{sep}");
                println!("95% CI: [{:.6},  {:.6}]",
                         estimate - tc * se, estimate + tc * se);
                println!();

                Ok(Value::Nil)
            }

            // ── ttest ────────────────────────────────────────────────────────
            "ttest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("ttest() requires a DataFrame".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                let get_col_vals = |df: &DataFrame, col: &str| -> Result<Vec<f64>> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => {
                            if a.iter().any(|v| !v.is_finite()) {
                                return Err(HayashiError::Runtime(
                                    format!("ttest: column '{col}' contains NaN or Inf. Use dropna() first.")
                                ));
                            }
                            Ok(a.iter().copied().collect())
                        }
                        Ok(Column::Int(a)) => Ok(a.iter().map(|&x| x as f64).collect()),
                        _ => Err(HayashiError::Type(format!("'{col}' is not numeric"))),
                    }
                };

                let stats = |v: &[f64]| -> (f64, f64, f64) { // (mean, sd, n)
                    let n = v.len() as f64;
                    let m = v.iter().sum::<f64>() / n;
                    let s = if n > 1.0 { (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { f64::NAN };
                    (m, s, n)
                };

                // ── um argumento variável → uni-amostral ou por grupo ─────────
                if args.len() >= 2 {
                    let var1 = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => return Err(HayashiError::Type("variable name must be an identifier".into())),
                    };

                    // ── PAREADO: ttest(df, v1, v2, paired=true) ──────────────
                    if args.len() >= 3 && matches!(opt_map.get("paired"), Some(Value::Bool(true))) {
                        let var2 = match &args[2] {
                            Expr::Var(n) | Expr::Str(n) => n.clone(),
                            _ => return Err(HayashiError::Type("variable name must be an identifier".into())),
                        };
                        let v1 = get_col_vals(&df, &var1)?;
                        let v2 = get_col_vals(&df, &var2)?;
                        if v1.len() != v2.len() {
                            return Err(HayashiError::Runtime("paired ttest requires equal-length columns".into()));
                        }
                        let diff: Vec<f64> = v1.iter().zip(&v2).map(|(a, b)| a - b).collect();
                        let (m, s, n) = stats(&diff);
                        let se = s / n.sqrt();
                        let t  = m / se;
                        let df_t = n - 1.0;
                        let p  = t_pvalue_two(t, df_t);
                        let tc = t_critical_95(df_t);
                        println!("\nPaired t-test: {var1} - {var2}");
                        println!("{}", "─".repeat(62));
                        println!("{:<14} {:>6}  {:>10}  {:>10}  {:>10}", "Variable", "Obs", "Mean", "Std. Err.", "[95% CI]");
                        println!("{}", "─".repeat(62));
                        println!("{:<14} {:>6.0}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                                 format!("{var1}-{var2}"), n, m, se, m - tc * se, m + tc * se);
                        println!("{}", "─".repeat(62));
                        println!("H0: mean(diff) = 0   t = {t:.4}   df = {df_t:.0}   p = {p:.4}");
                        println!();

                    // ── DOIS GRUPOS: ttest(df, var, by=group) ────────────────
                    } else if let Some(Value::Str(by_col)) = opt_map.get("by") {
                        let by_col = by_col.clone();
                        let vals   = get_col_vals(&df, &var1)?;
                        let groups = Self::col_to_strings(&df, &by_col)?;

                        let mut group_data: HashMap<String, Vec<f64>> = HashMap::new();
                        for (i, g) in groups.iter().enumerate() {
                            group_data.entry(g.clone()).or_default().push(vals[i]);
                        }
                        let mut gkeys: Vec<String> = group_data.keys().cloned().collect();
                        if gkeys.len() != 2 {
                            return Err(HayashiError::Runtime(
                                format!("two-sample ttest requires exactly 2 groups, got {}", gkeys.len())
                            ));
                        }
                        let all_num = gkeys.iter().all(|s| s.parse::<f64>().is_ok());
                        if all_num { gkeys.sort_by(|a, b| a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap()); }
                        else       { gkeys.sort(); }

                        let v1 = &group_data[&gkeys[0]];
                        let v2 = &group_data[&gkeys[1]];
                        let (m1, s1, n1) = stats(v1);
                        let (m2, s2, n2) = stats(v2);

                        // Welch's t
                        let se1sq = s1 * s1 / n1;
                        let se2sq = s2 * s2 / n2;
                        let se    = (se1sq + se2sq).sqrt();
                        let t     = (m1 - m2) / se;
                        let df_t  = (se1sq + se2sq).powi(2) /
                                    (se1sq.powi(2) / (n1 - 1.0) + se2sq.powi(2) / (n2 - 1.0));
                        let p     = t_pvalue_two(t, df_t);
                        let tc    = t_critical_95(df_t);

                        println!("\nTwo-sample t-test (Welch): {var1} by {by_col}");
                        println!("{}", "─".repeat(68));
                        println!("{:<10} {:>6}  {:>10}  {:>10}  {:>10}  {:>10}", "Group", "Obs", "Mean", "Std. Err.", "Std. Dev.", "[95% CI]");
                        println!("{}", "─".repeat(68));
                        for (g, m, s, n, se_g) in [
                            (&gkeys[0], m1, s1, n1, (s1*s1/n1).sqrt()),
                            (&gkeys[1], m2, s2, n2, (s2*s2/n2).sqrt()),
                        ] {
                            println!("{:<10} {:>6.0}  {:>10.4}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                                     g, n, m, se_g, s, m - tc * se_g, m + tc * se_g);
                        }
                        println!("{}", "─".repeat(68));
                        println!("diff = mean({}) - mean({})", gkeys[0], gkeys[1]);
                        println!("H0: diff = 0   Welch's t = {t:.4}   df = {df_t:.2}   p = {p:.4}");
                        println!();

                    // ── UNI-AMOSTRAL: ttest(df, var, mu=0) ───────────────────
                    } else {
                        let mu = match opt_map.get("mu") {
                            Some(Value::Float(f)) => *f,
                            Some(Value::Int(i))   => *i as f64,
                            None => 0.0,
                            _ => return Err(HayashiError::Type("mu= must be numeric".into())),
                        };
                        let v    = get_col_vals(&df, &var1)?;
                        let (m, s, n) = stats(&v);
                        let se   = s / n.sqrt();
                        let t    = (m - mu) / se;
                        let df_t = n - 1.0;
                        let p    = t_pvalue_two(t, df_t);
                        let tc   = t_critical_95(df_t);

                        println!("\nOne-sample t-test: {var1}   H0: mean = {mu}");
                        println!("{}", "─".repeat(62));
                        println!("{:<14} {:>6}  {:>10}  {:>10}  {:>10}", "Variable", "Obs", "Mean", "Std. Err.", "[95% CI]");
                        println!("{}", "─".repeat(62));
                        println!("{:<14} {:>6.0}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                                 var1, n, m, se, m - tc * se, m + tc * se);
                        println!("{}", "─".repeat(62));
                        println!("t = {t:.4}   df = {df_t:.0}   p = {p:.4}");
                        println!();
                    }
                } else {
                    return Err(HayashiError::Runtime(
                        "ttest() requires a variable name as second argument".into(),
                    ));
                }

                Ok(Value::Nil)
            }

            // ── collapse ─────────────────────────────────────────────────────
            "collapse" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "collapse() requires (df, func, [vars...], by=col)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let func_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type(
                        "second argument must be a function name (mean, sum, min, max, count, sd, median)".into(),
                    )),
                };
                let by_col = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(HayashiError::Runtime("collapse() requires by=colname".into())),
                };

                // validar função antes de qualquer cálculo
                match func_name.as_str() {
                    "mean" | "sum" | "min" | "max" | "count" | "sd" | "median" => {}
                    other => return Err(HayashiError::Runtime(format!(
                        "unknown aggregation '{other}' — use: mean, sum, min, max, count, sd, median"
                    ))),
                }

                // variáveis a agregar: args[2..] ou todas as numéricas exceto by
                let agg_vars: Vec<String> = if args.len() > 2 {
                    args[2..].iter().map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable names must be identifiers".into())),
                    }).collect::<Result<_>>()?
                } else {
                    use greeners::Column;
                    df.column_names().into_iter()
                        .filter(|n| n != &by_col && matches!(df.get_column(n), Ok(Column::Float(_)) | Ok(Column::Int(_))))
                        .collect()
                };

                // dados das colunas numéricas a agregar
                let col_data: Vec<Vec<f64>> = agg_vars.iter().map(|col| {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => Ok(a.to_vec()),
                        Ok(Column::Int(a))   => Ok(a.iter().map(|&x| x as f64).collect()),
                        _ => Err(HayashiError::Type(format!("'{col}' is not numeric"))),
                    }
                }).collect::<Result<_>>()?;

                // agrupa índices de linha por valor de by
                let by_strs = Self::col_to_strings(&df, &by_col)?;
                let n_obs = df.n_rows();
                let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
                for (i, v) in by_strs.iter().enumerate() {
                    groups.entry(v.clone()).or_default().push(i);
                }

                // ordena chaves de grupo
                let mut keys: Vec<String> = groups.keys().cloned().collect();
                let keys_numeric = keys.iter().all(|s| s.parse::<f64>().is_ok());
                if keys_numeric {
                    keys.sort_by(|a, b| a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap());
                } else {
                    keys.sort();
                }

                // função de agregação: NaN nos dados propaga NaN no resultado (IEEE 754)
                let agg = |vals: &[f64]| -> f64 {
                    let n = vals.len();
                    if n == 0 { return f64::NAN; }
                    match func_name.as_str() {
                        "count" => n as f64,
                        "sum"    => vals.iter().sum::<f64>(),
                        "mean"   => vals.iter().sum::<f64>() / n as f64,
                        "min"    => vals.iter().cloned().fold(f64::INFINITY,     f64::min),
                        "max"    => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                        "sd"     => {
                            if n < 2 { return f64::NAN; }
                            let m = vals.iter().sum::<f64>() / n as f64;
                            (vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
                        }
                        "median" => {
                            if vals.iter().any(|v| !v.is_finite()) { return f64::NAN; }
                            let mut s = vals.to_vec();
                            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            if n % 2 == 0 { (s[n/2 - 1] + s[n/2]) / 2.0 } else { s[n/2] }
                        }
                        _ => f64::NAN,
                    }
                };

                // constrói o DataFrame resultado
                let mut builder = DataFrame::builder();

                // coluna by (numérica ou string)
                use greeners::Column;
                if matches!(df.get_column(&by_col), Ok(Column::Float(_)) | Ok(Column::Int(_))) {
                    let vals: Vec<f64> = keys.iter()
                        .map(|k| k.parse::<f64>().unwrap_or(f64::NAN))
                        .collect();
                    builder = builder.add_column(&by_col, vals);
                } else {
                    builder = builder.add_string(&by_col, keys.clone());
                }

                // colunas agregadas
                for (ci, col_name) in agg_vars.iter().enumerate() {
                    let vals: Vec<f64> = keys.iter().map(|key| {
                        let subset: Vec<f64> = groups[key].iter().map(|&i| col_data[ci][i]).collect();
                        agg(&subset)
                    }).collect();
                    builder = builder.add_column(col_name, vals);
                }

                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({} groups from {} observations)", keys.len(), n_obs);
                Ok(Value::DataFrame(new_df))
            }

            // ── append ───────────────────────────────────────────────────────
            "append" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("append() requires (df1, df2)".into()));
                }
                let df1 = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let df2 = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("second argument must be a DataFrame".into())),
                };

                let names1 = df1.column_names();
                let names2 = df2.column_names();
                let set1: std::collections::HashSet<&str> = names1.iter().map(String::as_str).collect();
                let n1 = df1.n_rows();
                let n2 = df2.n_rows();

                // união de colunas: ordem de df1 primeiro, depois novas de df2
                let mut all_names = names1.clone();
                for n in &names2 { if !set1.contains(n.as_str()) { all_names.push(n.clone()); } }

                let get_num = |df: &DataFrame, col: &str, n: usize| -> Vec<f64> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => a.to_vec(),
                        Ok(Column::Int(a))   => a.iter().map(|&x| x as f64).collect(),
                        _                    => vec![f64::NAN; n],
                    }
                };
                let get_str = |df: &DataFrame, col: &str, n: usize| -> Vec<String> {
                    df.get_string(col).map(|a| a.to_vec()).unwrap_or_else(|_| vec![String::new(); n])
                };

                let mut builder = DataFrame::builder();
                for col in &all_names {
                    use greeners::Column;
                    let in1 = names1.contains(col);
                    let in2 = names2.contains(col);
                    let is_num = if in1 { matches!(df1.get_column(col), Ok(Column::Float(_)) | Ok(Column::Int(_))) }
                                 else   { matches!(df2.get_column(col), Ok(Column::Float(_)) | Ok(Column::Int(_))) };
                    if is_num {
                        let p1 = if in1 { get_num(&df1, col, n1) } else { vec![f64::NAN; n1] };
                        let p2 = if in2 { get_num(&df2, col, n2) } else { vec![f64::NAN; n2] };
                        builder = builder.add_column(col, p1.into_iter().chain(p2).collect::<Vec<_>>());
                    } else {
                        let p1 = if in1 { get_str(&df1, col, n1) } else { vec![String::new(); n1] };
                        let p2 = if in2 { get_str(&df2, col, n2) } else { vec![String::new(); n2] };
                        builder = builder.add_string(col, p1.into_iter().chain(p2).collect::<Vec<_>>());
                    }
                }

                let new_df = builder.build().map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("({} + {} = {} observations)", n1, n2, n1 + n2);
                Ok(Value::DataFrame(new_df))
            }

            // ── merge ─────────────────────────────────────────────────────────
            "merge" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "merge() requires (df1, df2, key=varname [, type=left|inner|outer])".into(),
                    ));
                }
                let df1 = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let df2 = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("second argument must be a DataFrame".into())),
                };
                let key_col = match opt_map.get("key") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(HayashiError::Runtime("merge() requires key=colname".into())),
                };
                let join_type = match opt_map.get("type") {
                    Some(Value::Str(s)) => s.clone(),
                    None => "left".to_string(),
                    _ => return Err(HayashiError::Runtime("type= must be a string".into())),
                };

                // índice de busca no df2: key_str → primeiro índice de linha
                let key2_strs = Self::col_to_strings(&df2, &key_col)?;
                let mut lookup: HashMap<String, usize> = HashMap::new();
                for (j, v) in key2_strs.iter().enumerate().rev() {
                    lookup.insert(v.clone(), j); // rev para ficar com o primeiro
                }

                let key1_strs = Self::col_to_strings(&df1, &key_col)?;
                let n1 = df1.n_rows();
                let n2 = df2.n_rows();

                // pares (idx_df1, idx_df2) para cada linha do resultado
                type RowPair = (Option<usize>, Option<usize>);
                let mut result_rows: Vec<RowPair> = (0..n1)
                    .map(|i| (Some(i), lookup.get(&key1_strs[i]).copied()))
                    .collect();

                match join_type.as_str() {
                    "left" => {}
                    "inner" => result_rows.retain(|(_, r2)| r2.is_some()),
                    "outer" | "full" => {
                        let matched: std::collections::HashSet<usize> =
                            result_rows.iter().filter_map(|(_, r2)| *r2).collect();
                        for j in 0..n2 {
                            if !matched.contains(&j) { result_rows.push((None, Some(j))); }
                        }
                    }
                    other => return Err(HayashiError::Runtime(
                        format!("unknown merge type '{other}' — use: left, inner, outer"),
                    )),
                }

                let names1 = df1.column_names();
                let names2 = df2.column_names();
                let set1: std::collections::HashSet<&str> = names1.iter().map(String::as_str).collect();

                // colunas extra de df2 (exclui key; sufixo _2 em colisão)
                let extra: Vec<(String, String)> = names2.iter()
                    .filter(|n| *n != &key_col)
                    .map(|n| (n.clone(), if set1.contains(n.as_str()) { format!("{n}_2") } else { n.clone() }))
                    .collect();

                let get_num = |df: &DataFrame, col: &str| -> Vec<f64> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => a.to_vec(),
                        Ok(Column::Int(a))   => a.iter().map(|&x| x as f64).collect(),
                        _                    => vec![],
                    }
                };
                let get_str_col = |df: &DataFrame, col: &str| -> Vec<String> {
                    df.get_string(col).map(|a| a.to_vec()).unwrap_or_default()
                };

                let mut builder = DataFrame::builder();

                // colunas de df1
                for col in &names1 {
                    use greeners::Column;
                    if matches!(df1.get_column(col), Ok(Column::Float(_)) | Ok(Column::Int(_))) {
                        let src = get_num(&df1, col);
                        builder = builder.add_column(col,
                            result_rows.iter().map(|(r1, _)| r1.map_or(f64::NAN, |i| src[i])).collect::<Vec<_>>());
                    } else {
                        let src = get_str_col(&df1, col);
                        builder = builder.add_string(col,
                            result_rows.iter().map(|(r1, _)| r1.map_or(String::new(), |i| src[i].clone())).collect::<Vec<_>>());
                    }
                }

                // colunas extras de df2
                for (src_col, out_col) in &extra {
                    use greeners::Column;
                    if matches!(df2.get_column(src_col), Ok(Column::Float(_)) | Ok(Column::Int(_))) {
                        let src = get_num(&df2, src_col);
                        builder = builder.add_column(out_col,
                            result_rows.iter().map(|(_, r2)| r2.map_or(f64::NAN, |j| src[j])).collect::<Vec<_>>());
                    } else {
                        let src = get_str_col(&df2, src_col);
                        builder = builder.add_string(out_col,
                            result_rows.iter().map(|(_, r2)| r2.map_or(String::new(), |j| src[j].clone())).collect::<Vec<_>>());
                    }
                }

                // indicador _merge: 3=matched, 1=left only, 2=right only
                builder = builder.add_column("_merge",
                    result_rows.iter().map(|(r1, r2)| match (r1, r2) {
                        (Some(_), Some(_)) => 3.0,
                        (Some(_), None)    => 1.0,
                        _                  => 2.0,
                    }).collect::<Vec<_>>());

                let new_df = builder.build().map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n_matched = result_rows.iter().filter(|(_, r2)| r2.is_some()).count();
                let n_out = result_rows.len();
                println!("({n_matched} matched, {} not matched, {n_out} total)", n_out - n_matched);
                Ok(Value::DataFrame(new_df))
            }

            // ── reshape ──────────────────────────────────────────────────────
            // reshape(df, "long",  stubs=[...], i=id_col,    j=new_j_col)
            // reshape(df, "wide",  values=[...], i=id_col,   j=j_col)
            "reshape" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "reshape(df, \"long\"|\"wide\", ...) requer pelo menos 2 argumentos".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("reshape(): arg 1 deve ser DataFrame".into())),
                };
                let direction = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type(
                        "reshape(): arg 2 deve ser \"long\" ou \"wide\"".into()
                    )),
                };
                let i_col = match opt_map.get("i") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime("reshape() requer opção i=coluna_id".into())),
                    _ => return Err(HayashiError::Type("i= deve ser string".into())),
                };
                let j_col = match opt_map.get("j") {
                    Some(Value::Str(s)) => s.clone(),
                    None => return Err(HayashiError::Runtime("reshape() requer opção j=coluna_tempo".into())),
                    _ => return Err(HayashiError::Type("j= deve ser string".into())),
                };

                match direction.as_str() {
                    // ── wide → long ──────────────────────────────────────────
                    "long" => {
                        let stubs: Vec<String> = match opt_map.get("stubs") {
                            Some(Value::List(lst)) => lst.iter().map(|v| match v {
                                Value::Str(s) => Ok(s.clone()),
                                _ => Err(HayashiError::Type("stubs= deve ser lista de strings".into())),
                            }).collect::<Result<_>>()?,
                            None => return Err(HayashiError::Runtime(
                                "reshape long requer opção stubs=[\"var1\", \"var2\", ...]".into()
                            )),
                            _ => return Err(HayashiError::Type("stubs= deve ser lista".into())),
                        };

                        // Para cada stub, detectar colunas e extrair sufixos
                        let col_names = df.column_names();
                        let mut stub_suffixes: Vec<Vec<String>> = Vec::new();
                        for stub in &stubs {
                            let mut suffs: Vec<String> = col_names.iter()
                                .filter(|c| c.starts_with(stub.as_str()) && *c != stub)
                                .map(|c| c[stub.len()..].to_string())
                                .collect();
                            suffs.sort();
                            if suffs.is_empty() {
                                return Err(HayashiError::Runtime(
                                    format!("reshape long: nenhuma coluna com stub '{stub}' encontrada")
                                ));
                            }
                            stub_suffixes.push(suffs);
                        }
                        // Validar que todos os stubs têm os mesmos sufixos
                        let all_suf = stub_suffixes[0].clone();
                        for (stub, suf) in stubs.iter().zip(stub_suffixes.iter()) {
                            if suf != &all_suf {
                                return Err(HayashiError::Runtime(
                                    format!("reshape long: stub '{stub}' tem sufixos diferentes dos demais")
                                ));
                            }
                        }

                        // Coletar valores da coluna id
                        use greeners::Column;
                        let n_rows = df.n_rows();
                        let id_vals: Vec<String> = match df.get_column(&i_col) {
                            Ok(Column::Float(arr)) => arr.iter().map(|v| v.to_string()).collect(),
                            Ok(Column::Int(arr))   => arr.iter().map(|v| v.to_string()).collect(),
                            _ => if let Ok(arr) = df.get_string(&i_col) {
                                arr.to_vec()
                            } else {
                                return Err(HayashiError::Runtime(format!("reshape: coluna id '{i_col}' não encontrada")));
                            }
                        };

                        let n_suf = all_suf.len();
                        let n_out = n_rows * n_suf;

                        // Determinar colunas que não são stubs nem id (passam direto)
                        let stub_cols: std::collections::HashSet<String> = stubs.iter()
                            .flat_map(|s| all_suf.iter().map(move |sf| format!("{s}{sf}")))
                            .collect();
                        let passthrough: Vec<String> = col_names.iter()
                            .filter(|c| **c != i_col && !stub_cols.contains(*c))
                            .cloned().collect();

                        let mut builder = DataFrame::builder();

                        // coluna id: repete cada valor n_suf vezes
                        let id_out: Vec<String> = id_vals.iter()
                            .flat_map(|v| std::iter::repeat(v.clone()).take(n_suf))
                            .collect();
                        builder = builder.add_string(&i_col, id_out);

                        // coluna j: para cada obs, cicla pelos sufixos
                        let j_out: Vec<String> = (0..n_rows)
                            .flat_map(|_| all_suf.iter().cloned())
                            .collect();
                        builder = builder.add_string(&j_col, j_out);

                        // colunas passthrough
                        for pc in &passthrough {
                            match df.get_column(pc) {
                                Ok(Column::Float(arr)) => {
                                    let vals: Vec<f64> = arr.iter()
                                        .flat_map(|&v| std::iter::repeat(v).take(n_suf))
                                        .collect();
                                    builder = builder.add_column(pc, vals);
                                }
                                Ok(Column::Int(arr)) => {
                                    let vals: Vec<f64> = arr.iter()
                                        .flat_map(|&v| std::iter::repeat(v as f64).take(n_suf))
                                        .collect();
                                    builder = builder.add_column(pc, vals);
                                }
                                _ => {}
                            }
                        }

                        // colunas dos stubs
                        for stub in &stubs {
                            let mut vals: Vec<f64> = Vec::with_capacity(n_out);
                            for row in 0..n_rows {
                                for suf in &all_suf {
                                    let col_name = format!("{stub}{suf}");
                                    let v = match df.get_column(&col_name) {
                                        Ok(Column::Float(arr)) => arr[row],
                                        Ok(Column::Int(arr))   => arr[row] as f64,
                                        _ => f64::NAN,
                                    };
                                    vals.push(v);
                                }
                            }
                            builder = builder.add_column(stub, vals);
                        }

                        let new_df = builder.build()
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!("(reshape long: {} obs × {} variáveis → {} obs × {} variáveis)",
                            n_rows, col_names.len(), n_out, new_df.column_names().len());
                        Ok(Value::DataFrame(new_df))
                    }

                    // ── long → wide ──────────────────────────────────────────
                    "wide" => {
                        let values: Vec<String> = match opt_map.get("values") {
                            Some(Value::List(lst)) => lst.iter().map(|v| match v {
                                Value::Str(s) => Ok(s.clone()),
                                _ => Err(HayashiError::Type("values= deve ser lista de strings".into())),
                            }).collect::<Result<_>>()?,
                            None => return Err(HayashiError::Runtime(
                                "reshape wide requer opção values=[\"var1\", \"var2\", ...]".into()
                            )),
                            _ => return Err(HayashiError::Type("values= deve ser lista".into())),
                        };

                        use greeners::Column;
                        let n_rows = df.n_rows();

                        // Coletar valores únicos de j (em ordem de aparição)
                        let j_vals: Vec<String> = {
                            let mut seen = std::collections::HashSet::new();
                            let mut out  = Vec::new();
                            match df.get_column(&j_col) {
                                Ok(Column::Float(arr)) => {
                                    for &v in arr.iter() {
                                        let s = if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") };
                                        if seen.insert(s.clone()) { out.push(s); }
                                    }
                                }
                                Ok(Column::Int(arr)) => {
                                    for &v in arr.iter() {
                                        let s = v.to_string();
                                        if seen.insert(s.clone()) { out.push(s); }
                                    }
                                }
                                _ => {
                                    if let Ok(arr) = df.get_string(&j_col) {
                                        for v in arr.iter() {
                                            if seen.insert(v.clone()) { out.push(v.clone()); }
                                        }
                                    } else {
                                        return Err(HayashiError::Runtime(
                                            format!("reshape wide: coluna j '{j_col}' não encontrada")
                                        ));
                                    }
                                }
                            }
                            out
                        };

                        // j label por linha
                        let row_j: Vec<String> = match df.get_column(&j_col) {
                            Ok(Column::Float(arr)) => arr.iter().map(|&v| {
                                if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") }
                            }).collect(),
                            Ok(Column::Int(arr))   => arr.iter().map(|v| v.to_string()).collect(),
                            _ => df.get_string(&j_col)
                                .map_err(|_| HayashiError::Runtime("reshape wide: j coluna inválida".into()))?
                                .to_vec(),
                        };

                        // id por linha
                        let row_id: Vec<String> = match df.get_column(&i_col) {
                            Ok(Column::Float(arr)) => arr.iter().map(|v| v.to_string()).collect(),
                            Ok(Column::Int(arr))   => arr.iter().map(|v| v.to_string()).collect(),
                            _ => df.get_string(&i_col)
                                .map_err(|_| HayashiError::Runtime("reshape wide: i coluna inválida".into()))?
                                .to_vec(),
                        };

                        // Ordem única de ids
                        let mut seen_ids = std::collections::HashSet::new();
                        let unique_ids: Vec<String> = row_id.iter()
                            .filter(|id| seen_ids.insert((*id).clone()))
                            .cloned().collect();
                        let n_id = unique_ids.len();

                        // id_idx[row] → índice no unique_ids
                        let id_pos: std::collections::HashMap<&str, usize> = unique_ids.iter()
                            .enumerate().map(|(i, s)| (s.as_str(), i)).collect();
                        let j_pos: std::collections::HashMap<&str, usize> = j_vals.iter()
                            .enumerate().map(|(i, s)| (s.as_str(), i)).collect();

                        // Para cada coluna value, construir matrix (n_id × n_j)
                        let mut value_mats: Vec<Vec<f64>> = values.iter().map(|_| {
                            vec![f64::NAN; n_id * j_vals.len()]
                        }).collect();

                        for row in 0..n_rows {
                            let i_idx = id_pos[row_id[row].as_str()];
                            let j_idx = j_pos[row_j[row].as_str()];
                            for (vi, val_col) in values.iter().enumerate() {
                                let v = match df.get_column(val_col) {
                                    Ok(Column::Float(arr)) => arr[row],
                                    Ok(Column::Int(arr))   => arr[row] as f64,
                                    _ => f64::NAN,
                                };
                                value_mats[vi][i_idx * j_vals.len() + j_idx] = v;
                            }
                        }

                        let col_names = df.column_names();
                        let skip: std::collections::HashSet<&str> = values.iter()
                            .chain(std::iter::once(&j_col))
                            .map(String::as_str).collect();
                        let passthrough: Vec<String> = col_names.iter()
                            .filter(|c| **c != i_col && !skip.contains(c.as_str()))
                            .cloned().collect();

                        // Pegar primeiro valor de passthrough por id
                        let mut builder = DataFrame::builder();
                        // id column
                        builder = builder.add_string(&i_col, unique_ids.clone());
                        // passthrough: valor da primeira linha com esse id
                        for pc in &passthrough {
                            let mut vals = vec![f64::NAN; n_id];
                            for row in 0..n_rows {
                                let ii = id_pos[row_id[row].as_str()];
                                if vals[ii].is_nan() {
                                    if let Ok(Column::Float(arr)) = df.get_column(pc) {
                                        vals[ii] = arr[row];
                                    } else if let Ok(Column::Int(arr)) = df.get_column(pc) {
                                        vals[ii] = arr[row] as f64;
                                    }
                                }
                            }
                            builder = builder.add_column(pc, vals);
                        }
                        // value columns
                        for (vi, stub) in values.iter().enumerate() {
                            for (ji, jv) in j_vals.iter().enumerate() {
                                let col_name = format!("{stub}{jv}");
                                let col_vals: Vec<f64> = (0..n_id)
                                    .map(|ii| value_mats[vi][ii * j_vals.len() + ji])
                                    .collect();
                                builder = builder.add_column(&col_name, col_vals);
                            }
                        }

                        let new_df = builder.build()
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!("(reshape wide: {} obs → {} obs × {} variáveis)",
                            n_rows, n_id, new_df.column_names().len());
                        Ok(Value::DataFrame(new_df))
                    }

                    other => Err(HayashiError::Runtime(
                        format!("reshape: direção '{other}' desconhecida — use \"long\" ou \"wide\"")
                    )),
                }
            }

            // ── sort ─────────────────────────────────────────────────────────
            "sort" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "sort() requires (dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let sort_vars: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("sort keys must be identifiers".into())),
                    })
                    .collect::<Result<_>>()?;
                let desc = matches!(opt_map.get("desc"), Some(Value::Bool(true)));

                // extrai chaves de ordenação
                enum SortKey { Num(Vec<f64>), Str(Vec<String>) }
                let keys: Vec<SortKey> = sort_vars.iter().map(|v| {
                    use greeners::Column;
                    match df.get_column(v) {
                        Ok(Column::Float(arr)) => Ok(SortKey::Num(arr.to_vec())),
                        Ok(Column::Int(arr))   => Ok(SortKey::Num(arr.iter().map(|&x| x as f64).collect())),
                        _ => df.get_string(v)
                                .map(|arr| SortKey::Str(arr.to_vec()))
                                .map_err(|_| HayashiError::Runtime(format!("column '{v}' not found"))),
                    }
                }).collect::<Result<_>>()?;

                let n = df.n_rows();
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| {
                    use std::cmp::Ordering;
                    for key in &keys {
                        let ord = match key {
                            SortKey::Num(v) => match (v[a].is_nan(), v[b].is_nan()) {
                                (true,  true)  => Ordering::Equal,
                                (true,  false) => Ordering::Greater,
                                (false, true)  => Ordering::Less,
                                (false, false) => v[a].partial_cmp(&v[b]).unwrap(),
                            },
                            SortKey::Str(v) => v[a].cmp(&v[b]),
                        };
                        if ord != Ordering::Equal {
                            return if desc { ord.reverse() } else { ord };
                        }
                    }
                    Ordering::Equal
                });

                let all_names = df.column_names();
                let mut builder = DataFrame::builder();
                for col_name in &all_names {
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            builder = builder.add_column(col_name, idx.iter().map(|&i| arr[i]).collect::<Vec<_>>());
                        }
                        Ok(Column::Int(arr)) => {
                            builder = builder.add_column(col_name, idx.iter().map(|&i| arr[i] as f64).collect::<Vec<_>>());
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                let v = arr.to_vec();
                                builder = builder.add_string(col_name, idx.iter().map(|&i| v[i].clone()).collect::<Vec<_>>());
                            }
                        }
                    }
                }

                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("({n} observations sorted)");
                Ok(Value::DataFrame(new_df))
            }

            // ── list ──────────────────────────────────────────────────────────
            "list" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("list() requires a DataFrame".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                // args[1..]: Int → nrows; Ident/Str → coluna
                let mut n_explicit: Option<usize> = None;
                let mut col_names: Vec<String> = Vec::new();

                for arg in &args[1..] {
                    match arg {
                        Expr::Int(n)                => n_explicit = Some((*n).max(0) as usize),
                        Expr::Var(n) | Expr::Str(n) => col_names.push(n.clone()),
                        _ => return Err(HayashiError::Type("list() arguments must be identifiers or row count".into())),
                    }
                }

                // vars=[A, B, C] — opção nomeada (somente se nenhuma coluna foi dada positionally)
                if col_names.is_empty() {
                    if let Some(vars_opt) = opts.iter().find(|o| o.name == "vars") {
                        match &vars_opt.value {
                            Expr::List(items) => {
                                for e in items {
                                    match e {
                                        Expr::Var(n) | Expr::Str(n) => col_names.push(n.clone()),
                                        _ => {}
                                    }
                                }
                            }
                            Expr::Var(n) | Expr::Str(n) => col_names.push(n.clone()),
                            _ => {}
                        }
                    }
                }

                // n= opção (sobrepõe default 10; arg positional Int tem prioridade)
                let n_show = if let Some(n) = n_explicit {
                    n
                } else {
                    match opt_map.get("n") {
                        Some(Value::Int(v))   => (*v).max(0) as usize,
                        Some(Value::Float(v)) => (*v as i64).max(0) as usize,
                        _                     => 10usize,
                    }
                };

                if col_names.is_empty() {
                    col_names = df.column_names();
                }

                let n_rows = n_show.min(df.n_rows());

                // extrai dados das colunas
                let cols_data: Vec<(String, Vec<String>)> = col_names.iter().map(|name| {
                    use greeners::Column;
                    let vals: Vec<String> = match df.get_column(name) {
                        Ok(Column::Float(arr)) => arr.iter().take(n_rows).map(|x| {
                            if x.is_nan() { ".".into() }
                            else if x.fract() == 0.0 && x.abs() < 1e14 { format!("{}", *x as i64) }
                            else { format!("{:.4}", x) }
                        }).collect(),
                        Ok(Column::Int(arr)) => arr.iter().take(n_rows).map(|x| x.to_string()).collect(),
                        _ => df.get_string(name).map(|a| a.to_vec().into_iter().take(n_rows).collect())
                                .unwrap_or_else(|_| vec!["?".into(); n_rows]),
                    };
                    (name.clone(), vals)
                }).collect();

                // larguras de coluna
                let row_num_w = n_rows.to_string().len().max(1);
                let widths: Vec<usize> = cols_data.iter().map(|(name, vals)| {
                    vals.iter().map(|v| v.len()).max().unwrap_or(0).max(name.len()) + 1
                }).collect();

                // cabeçalho
                print!("{:>rw$} |", "", rw = row_num_w);
                for (i, (name, _)) in cols_data.iter().enumerate() {
                    print!(" {:>w$}", name, w = widths[i]);
                }
                println!();
                println!("{}-+{}", "-".repeat(row_num_w), "-".repeat(widths.iter().sum::<usize>() + widths.len()));

                // linhas
                for r in 0..n_rows {
                    print!("{:>rw$} |", r + 1, rw = row_num_w);
                    for (i, (_, vals)) in cols_data.iter().enumerate() {
                        print!(" {:>w$}", vals[r], w = widths[i]);
                    }
                    println!();
                }
                if df.n_rows() > n_rows {
                    println!("  ({} more observations not shown)", df.n_rows() - n_rows);
                }
                println!();
                Ok(Value::Nil)
            }

            // ── dropna ───────────────────────────────────────────────────────
            "dropna" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "dropna() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                // colunas a verificar: as listadas ou todas as float
                let check: Vec<String> = if args.len() > 1 {
                    args[1..].iter().map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable names must be identifiers".into())),
                    }).collect::<Result<_>>()?
                } else {
                    use greeners::Column;
                    df.column_names().into_iter()
                        .filter(|n| matches!(df.get_column(n), Ok(Column::Float(_))))
                        .collect()
                };

                let n = df.n_rows();
                let mut keep = vec![true; n];

                for col_name in &check {
                    use greeners::Column;
                    if let Ok(Column::Float(arr)) = df.get_column(col_name) {
                        for (i, &v) in arr.iter().enumerate() {
                            if v.is_nan() { keep[i] = false; }
                        }
                    }
                }

                let n_drop = keep.iter().filter(|&&k| !k).count();
                let n_kept = n - n_drop;

                // reconstrói o DataFrame filtrando as linhas
                let all_names = df.column_names();
                let mut builder = DataFrame::builder();

                for col_name in &all_names {
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            let vals: Vec<f64> = arr.iter().enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        Ok(Column::Int(arr)) => {
                            let vals: Vec<f64> = arr.iter().enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v as f64)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                let vals: Vec<String> = arr.to_vec().into_iter().enumerate()
                                    .filter(|(i, _)| keep[*i])
                                    .map(|(_, v)| v)
                                    .collect();
                                builder = builder.add_string(col_name, vals);
                            }
                        }
                    }
                }

                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({n_drop} observations dropped, {n_kept} remaining)");
                Ok(Value::DataFrame(new_df))
            }

            // ── filter ───────────────────────────────────────────────────────
            // filter(df, condition_expr) → DataFrame com linhas onde cond ≠ 0
            "filter" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "filter(df, cond) requer um DataFrame e uma expressão de condição".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type(
                        "filter(): primeiro argumento deve ser um DataFrame".into(),
                    )),
                };
                let mask = Self::eval_col_expr(&args[1], &df)?;
                let keep: Vec<bool> = mask.iter().map(|&v| v != 0.0 && !v.is_nan()).collect();
                let n       = keep.len();
                let n_kept  = keep.iter().filter(|&&k| k).count();
                let n_drop  = n - n_kept;

                let all_names = df.column_names();
                let mut builder = DataFrame::builder();
                for col_name in &all_names {
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            let vals: Vec<f64> = arr.iter().enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        Ok(Column::Int(arr)) => {
                            let vals: Vec<f64> = arr.iter().enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v as f64)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                let vals: Vec<String> = arr.iter().enumerate()
                                    .filter(|(i, _)| keep[*i])
                                    .map(|(_, v)| v.clone())
                                    .collect();
                                builder = builder.add_string(col_name, vals);
                            }
                        }
                    }
                }
                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("({n_drop} observations removed, {n_kept} remaining)");
                Ok(Value::DataFrame(new_df))
            }

            // ── rename ───────────────────────────────────────────────────────
            "rename" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "rename() requires (dataframe, oldname, newname)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let old = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("oldname must be an identifier".into())),
                };
                let new = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("newname must be an identifier".into())),
                };

                let all_names = df.column_names();
                if !all_names.contains(&old) {
                    return Err(HayashiError::Runtime(
                        format!("column '{old}' not found in DataFrame"),
                    ));
                }

                let mut builder = DataFrame::builder();
                for col_name in &all_names {
                    let out_name = if col_name == &old { &new } else { col_name };
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            builder = builder.add_column(out_name, arr.to_vec());
                        }
                        Ok(Column::Int(arr)) => {
                            let vals: Vec<f64> = arr.iter().map(|&v| v as f64).collect();
                            builder = builder.add_column(out_name, vals);
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                builder = builder.add_string(out_name, arr.to_vec());
                            }
                        }
                    }
                }

                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({old} → {new})");
                Ok(Value::DataFrame(new_df))
            }

            // ── drop ─────────────────────────────────────────────────────────
            "drop" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "drop() requires (dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let drop_names: std::collections::HashSet<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable names must be identifiers".into())),
                    })
                    .collect::<Result<_>>()?;

                let all = df.column_names();
                let keep: Vec<&str> = all.iter()
                    .filter(|n| !drop_names.contains(*n))
                    .map(String::as_str)
                    .collect();

                let new_df = df.select(&keep)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({} variables dropped, {} remaining)", drop_names.len(), keep.len());
                Ok(Value::DataFrame(new_df))
            }

            // ── drop_collinear ────────────────────────────────────────────────
            // drop_collinear(df [, vars=[x1, x2, ...]])
            // Detecta colunas perfeitamente colineares via QR e retorna novo df
            // sem elas. O usuário vê exatamente o que foi removido antes de
            // passar os dados para qualquer estimador.
            "drop_collinear" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "drop_collinear() requer ao menos um DataFrame".into()
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type(
                        "drop_collinear(): primeiro argumento deve ser um DataFrame".into()
                    )),
                };

                // Colunas a checar: vars=[...] ou todas as numéricas
                let check_cols: Vec<String> = match opt_map.get("vars") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type(
                            "drop_collinear(): vars deve ser lista de nomes de colunas".into()
                        )),
                    }).collect::<Result<_>>()?,
                    None => df.column_names().into_iter()
                        .filter(|name| df.get(name).is_ok())
                        .collect(),
                    _ => return Err(HayashiError::Type(
                        "drop_collinear(): vars deve ser lista de strings".into()
                    )),
                };

                if check_cols.is_empty() {
                    println!("drop_collinear: nenhuma coluna numérica encontrada.");
                    return Ok(Value::DataFrame(df));
                }

                let n = df.n_rows();
                let k = check_cols.len();
                let mut mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, col) in check_cols.iter().enumerate() {
                    let col_data = df.get(col)
                        .map_err(|_| HayashiError::Runtime(
                            format!("drop_collinear: coluna '{col}' não encontrada ou não numérica")
                        ))?;
                    for (i, &v) in col_data.iter().enumerate() {
                        mat[[i, j]] = v;
                    }
                }

                let (_clean, keep_idx, omit_idx) =
                    greeners::OLS::detect_collinearity(&mat, 1e-10);

                if omit_idx.is_empty() {
                    println!("drop_collinear: nenhuma colinearidade detectada entre as {} colunas verificadas.", k);
                    return Ok(Value::DataFrame(df));
                }

                let omit_names: Vec<&str> = omit_idx.iter()
                    .map(|&i| check_cols[i].as_str())
                    .collect();
                let keep_names: Vec<&str> = keep_idx.iter()
                    .map(|&i| check_cols[i].as_str())
                    .collect();

                println!("drop_collinear: {} coluna(s) removida(s) por colinearidade perfeita:",
                    omit_names.len());
                for name in &omit_names {
                    println!("  o.{name}");
                }
                println!("  {} coluna(s) mantida(s): {}", keep_names.len(),
                    keep_names.join(", "));

                let new_df = df.drop(&omit_names)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::DataFrame(new_df))
            }

            // ── keep ──────────────────────────────────────────────────────────
            "keep" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "keep() requires (dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let keep_names: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable names must be identifiers".into())),
                    })
                    .collect::<Result<_>>()?;

                let refs: Vec<&str> = keep_names.iter().map(String::as_str).collect();
                let n_before = df.column_names().len();
                let new_df = df.select(&refs)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({} variables kept, {} dropped)", refs.len(), n_before - refs.len());
                Ok(Value::DataFrame(new_df))
            }

            // ── tabulate ─────────────────────────────────────────────────────
            "tabulate" | "tab" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "tabulate() requires (dataframe, varname) or (dataframe, var1, var2)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let var1 = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("variable name must be an identifier".into())),
                };

                if args.len() == 2 {
                    Self::tabulate_one(&df, &var1)?;
                } else {
                    let var2 = match &args[2] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => return Err(HayashiError::Type("variable name must be an identifier".into())),
                    };
                    let do_chi2 = matches!(opt_map.get("chi2"), Some(Value::Bool(true)));
                    Self::tabulate_two(&df, &var1, &var2, do_chi2)?;
                }

                Ok(Value::Nil)
            }

            // ── garch / egarch / gjrgarch ────────────────────────────────────
            // garch(df, varname, p=1, q=1)
            // garch(df, varname, p=1, q=1, dist=t)    — erros Student-t
            // egarch(df, varname, p=1, q=1)
            // gjrgarch(df, varname, p=1, q=1)
            "garch" | "egarch" | "gjrgarch" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        format!("{func}() requer df e nome da variável")
                    ));
                }

                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type(
                        format!("{func}(): primeiro argumento deve ser um DataFrame")
                    )),
                };

                let col_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type(
                        format!("{func}(): segundo argumento deve ser o nome de uma coluna")
                    )),
                };

                let p = match opt_map.get("p") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let q = match opt_map.get("q") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let use_t_dist = matches!(
                    opt_map.get("dist"),
                    Some(Value::Str(s)) if s == "t"
                );

                let y = Array1::from(Self::eval_col_expr(&Expr::Var(col_name), &df)?);

                let result = match (func, use_t_dist) {
                    ("garch",    false) => greeners::GARCH::fit(&y, p, q),
                    ("garch",    true)  => greeners::GARCH::fit_t(&y, p, q),
                    ("egarch",   false) => greeners::EGARCH::fit(&y, p, q),
                    ("egarch",   true)  => greeners::EGARCH::fit_t(&y, p, q),
                    ("gjrgarch", false) => greeners::GJRGARCH::fit(&y, p, q),
                    ("gjrgarch", true)  => greeners::GJRGARCH::fit_t(&y, p, q),
                    _ => unreachable!(),
                };

                Ok(Value::GarchResult(Rc::new(
                    result.map_err(|e| HayashiError::Runtime(format!("{func}: {e}")))?
                )))
            }

            // ljungbox(df, varname, lags=10)
            // ljungbox(model, lags=10)   — aceita GARCH, ARIMA, OLS
            // H₀: as primeiras `lags` autocorrelações são conjuntamente zero
            "ljungbox" | "ljung_box" | "portmanteau" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "ljungbox() requer uma série ou modelo".into()
                    ));
                }

                let series = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => {
                        let col_name = match args.get(1) {
                            Some(Expr::Var(n)) => n.clone(),
                            _ => return Err(HayashiError::Runtime(
                                "ljungbox(df, varname): segundo argumento deve ser o nome da coluna".into()
                            )),
                        };
                        Array1::from(Self::eval_col_expr(&Expr::Var(col_name), &df)?)
                    }
                    // resíduos padronizados de GARCH
                    Value::GarchResult(m) => m.standardized_residuals.clone(),
                    // resíduos de ARIMA
                    Value::ArimaResult(m) => Array1::from_vec(m.residuals().to_vec()),
                    // resíduos de OLS
                    Value::OlsResult(m) => m.residuals.clone(),
                    _ => return Err(HayashiError::Type(
                        "ljungbox(): argumento deve ser DataFrame, GARCH, ARIMA ou OLS".into()
                    )),
                };

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let res = greeners::Diagnostics::ljung_box(&series, lags)
                    .map_err(|e| HayashiError::Runtime(format!("ljungbox: {e}")))?;

                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(62);
                println!("\nLjung-Box Test  —  lags = {}  n = {}", res.lags, res.n_obs);
                println!("{sep}");
                println!("H₀: sem autocorrelação até lag {}", res.lags);
                println!("{sep}");
                println!("{:<6} {:>10} {:>10} {:>8}", "lag", "ACF", "Q", "p-value");
                println!("{sep}");
                let mut q_accum = 0.0_f64;
                let nf = res.n_obs as f64;
                for (i, &rho) in res.acf.iter().enumerate() {
                    let k = i + 1;
                    q_accum += nf * (nf + 2.0) * rho * rho / (nf - k as f64);
                    // p-value cumulativo para o Q até lag k
                    let p_k = greeners::chi2_pvalue(q_accum, k as f64);
                    println!("{:<6} {:>10.4} {:>10.4} {:>8.4} {:>3}",
                        k, rho, q_accum, p_k, sig(p_k));
                }
                println!("{sep}");
                println!("Q({lags}) = {:.4}   p = {:.4}  {}   (*** p<0.01  ** p<0.05  * p<0.10)",
                    res.q_stat, res.p_value, sig(res.p_value));
                println!();

                Ok(Value::Nil)
            }

            // leverage(model)
            // leverage(model, threshold=2)   — múltiplo de k/n; padrão 2
            // Diagonal da hat matrix: h_i = x_i'(X'X)⁻¹x_i
            // Observações com h_i > threshold*k/n merecem atenção
            "leverage" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("leverage() requer um modelo OLS".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("leverage() suporta apenas modelos OLS".into())),
                };

                let threshold = match opt_map.get("threshold") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    _ => 2.0,
                };

                let h = greeners::Diagnostics::leverage(&ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("leverage: {e}")))?;

                let n = h.len();
                let k = ols.x.ncols();
                let cutoff = threshold * k as f64 / n as f64;
                let h_mean = k as f64 / n as f64;

                // mostra apenas observações acima do cutoff (ou todas se poucas)
                let flagged: Vec<(usize, f64)> = h.iter().enumerate()
                    .filter(|(_, &hi)| hi > cutoff)
                    .map(|(i, &hi)| (i + 1, hi))
                    .collect();

                let sep = "─".repeat(46);
                println!("\nLeverage  —  h̄ = {:.4}  cutoff = {:.4} ({}×k/n)", h_mean, cutoff, threshold);
                println!("{sep}");
                if flagged.is_empty() {
                    println!("Nenhuma observação acima do cutoff.");
                } else {
                    println!("{:<8} {:>10}  {}", "obs", "h_i", "");
                    println!("{sep}");
                    for (i, hi) in &flagged {
                        println!("{:<8} {:>10.4}  high leverage", i, hi);
                    }
                    println!("{sep}");
                    println!("{} observação(ões) com h_i > {:.4}", flagged.len(), cutoff);
                }
                println!();

                Ok(Value::Nil)
            }

            // cooks(model)
            // cooks(model, threshold=1)   — limiar absoluto padrão; ou usa 4/n
            // D_i = (e_i²·h_i) / (k·MSE·(1−h_i)²)
            "cooks" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("cooks() requer um modelo OLS".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("cooks() suporta apenas modelos OLS".into())),
                };

                let mse = ols.result.sigma * ols.result.sigma;
                let d = greeners::Diagnostics::cooks_distance(&ols.residuals, &ols.x, mse)
                    .map_err(|e| HayashiError::Runtime(format!("cooks: {e}")))?;

                let n = d.len();
                let k = ols.x.ncols();
                // cutoff configurável; padrão 4/n (regra de bolso mais comum)
                let cutoff = match opt_map.get("threshold") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    _ => 4.0 / n as f64,
                };

                let flagged: Vec<(usize, f64)> = d.iter().enumerate()
                    .filter(|(_, &di)| di > cutoff)
                    .map(|(i, &di)| (i + 1, di))
                    .collect();

                let sep = "─".repeat(46);
                println!("\nCook's Distance  —  n={n}  k={k}  cutoff={cutoff:.4} (4/n)");
                println!("{sep}");
                if flagged.is_empty() {
                    println!("Nenhuma observação influente acima do cutoff.");
                } else {
                    println!("{:<8} {:>10}  {}", "obs", "D_i", "");
                    println!("{sep}");
                    for (i, di) in &flagged {
                        let label = if *di > 1.0 { "muito influente" } else { "influente" };
                        println!("{:<8} {:>10.4}  {}", i, di, label);
                    }
                    println!("{sep}");
                    println!("{} observação(ões) com D_i > {:.4}", flagged.len(), cutoff);
                }
                println!();

                Ok(Value::Nil)
            }

            // vif(model)
            // Variance Inflation Factor — detecta multicolinearidade por variável
            // VIF_j = 1/(1−R²_j); VIF>10 indica multicolinearidade grave
            "vif" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("vif() requer um modelo OLS".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("vif() suporta apenas modelos OLS".into())),
                };

                let vifs = greeners::Diagnostics::vif(&ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("vif: {e}")))?;

                let names = ols.result.variable_names.as_deref()
                    .unwrap_or(&[]);

                let sep = "─".repeat(40);
                println!("\nVariance Inflation Factor (VIF)");
                println!("{sep}");
                println!("{:<20} {:>8}  {}", "Variável", "VIF", "Diagnóstico");
                println!("{sep}");
                for (i, &v) in vifs.iter().enumerate() {
                    let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                    let diag = if v.is_nan() {
                        "constante"
                    } else if v.is_infinite() || v > 10.0 {
                        "multicolinearidade grave"
                    } else if v > 5.0 {
                        "moderada"
                    } else {
                        "ok"
                    };
                    if v.is_nan() {
                        println!("{:<20} {:>8}  {}", name, "—", diag);
                    } else if v.is_infinite() {
                        println!("{:<20} {:>8}  {}", name, "∞", diag);
                    } else {
                        println!("{:<20} {:>8.3}  {}", name, v, diag);
                    }
                }
                println!("{sep}");
                println!("Referência: VIF<5 ok  |  5-10 moderado  |  >10 grave");
                println!();

                Ok(Value::Nil)
            }

            // condnum(model)
            // Condition number da matriz X — diagnóstico global de multicolinearidade
            // κ = σ_max/σ_min; κ>30 indica problema sério
            "condnum" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("condnum() requer um modelo OLS".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("condnum() suporta apenas modelos OLS".into())),
                };

                let kappa = greeners::Diagnostics::condition_number(&ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("condnum: {e}")))?;

                let diag = if kappa.is_infinite() || kappa > 100.0 {
                    "multicolinearidade severa"
                } else if kappa > 30.0 {
                    "multicolinearidade moderada"
                } else if kappa > 10.0 {
                    "atenção"
                } else {
                    "ok"
                };

                let sep = "─".repeat(44);
                println!("\nCondition Number (multicolinearidade global)");
                println!("{sep}");
                if kappa.is_infinite() {
                    println!("{:<20} {:>12}  {}", "κ(X)", "∞", diag);
                } else {
                    println!("{:<20} {:>12.2}  {}", "κ(X)", kappa, diag);
                }
                println!("{sep}");
                println!("Referência: κ<10 ok  |  10-30 atenção  |  30-100 moderado  |  >100 severo");
                println!();

                Ok(Value::Nil)
            }

            // durbinwatson(model)
            // Durbin-Watson: detecta autocorrelação de primeira ordem nos resíduos OLS
            // DW ≈ 2 → sem autocorrelação; DW < 2 → positiva; DW > 2 → negativa
            "durbinwatson" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "durbinwatson() requer um modelo OLS".into()
                    ));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type(
                        "durbinwatson() suporta apenas modelos OLS".into()
                    )),
                };

                let dw = greeners::Diagnostics::durbin_watson(&ols.residuals);

                let interpretation = if dw < 1.5 {
                    "autocorrelação positiva provável"
                } else if dw > 2.5 {
                    "autocorrelação negativa provável"
                } else {
                    "sem autocorrelação evidente"
                };

                let sep = "─".repeat(50);
                println!("\nDurbin-Watson Test");
                println!("{sep}");
                println!("H₀: sem autocorrelação de primeira ordem");
                println!("{sep}");
                println!("{:<18} {:>10}", "DW statistic", format!("{dw:.4}"));
                println!("{:<18} {:>10}", "Interpretação", interpretation);
                println!("{sep}");
                println!("Referência: DW ≈ 2 (sem autocorr.) | <1.5 (positiva) | >2.5 (negativa)");
                println!();

                Ok(Value::Nil)
            }

            // white(model)
            // White (1980): H₀: homocedasticidade
            // Requer modelo OLS — regride u² nos regressores e seus quadrados
            "white" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("white() requer um modelo OLS".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("white() suporta apenas modelos OLS".into())),
                };

                let (lm, p, df) = greeners::SpecificationTests::white_test(&ols.residuals, &ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("white: {e}")))?;

                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(54);
                println!("\nWhite Test (heteroscedasticidade)");
                println!("{sep}");
                println!("H₀: homocedasticidade");
                println!("{sep}");
                println!("{:<24} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<24} {:>10.4} {:>10.4} {:>4}",
                    format!("LM ~ χ²({df})"), lm, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();

                Ok(Value::Nil)
            }

            // reset(model)
            // reset(model, power=3)
            // Ramsey RESET: H₀: especificação linear correta
            // Requer modelo OLS — adiciona ŷ², ..., ŷ^power como regressores
            "reset" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("reset() requer um modelo OLS".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("reset() suporta apenas modelos OLS".into())),
                };

                let power = match opt_map.get("power") {
                    Some(Value::Int(v))   => (*v as usize).max(2),
                    Some(Value::Float(v)) => (*v as usize).max(2),
                    _ => 3,
                };

                let fitted = ols.result.fitted_values(&ols.x);
                // y = resíduos + valores ajustados
                let y = &ols.residuals + &fitted;

                let (f, p, df1, df2) = greeners::SpecificationTests::reset_test(
                    &y, &ols.x, &fitted, power
                ).map_err(|e| HayashiError::Runtime(format!("reset: {e}")))?;

                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(54);
                println!("\nRamsey RESET Test  —  power = {power}");
                println!("{sep}");
                println!("H₀: especificação linear correta");
                println!("{sep}");
                println!("{:<24} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<24} {:>10.4} {:>10.4} {:>4}",
                    format!("F ~ F({df1},{df2})"), f, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();

                Ok(Value::Nil)
            }

            // jb(df, varname) | jb(model)
            // Jarque-Bera: H₀: resíduos normalmente distribuídos
            // Aceita série bruta, OLS, ARIMA, GARCH (resíduos padronizados)
            "jb" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("jb() requer uma série ou modelo".into()));
                }

                let series = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => {
                        let col_name = match args.get(1) {
                            Some(Expr::Var(n)) => n.clone(),
                            _ => return Err(HayashiError::Runtime(
                                "jb(df, varname): segundo argumento deve ser o nome da coluna".into()
                            )),
                        };
                        Array1::from(Self::eval_col_expr(&Expr::Var(col_name), &df)?)
                    }
                    Value::OlsResult(m)   => m.residuals.clone(),
                    Value::ArimaResult(m) => Array1::from_vec(m.residuals().to_vec()),
                    Value::GarchResult(m) => m.standardized_residuals.clone(),
                    _ => return Err(HayashiError::Type(
                        "jb(): argumento deve ser DataFrame, OLS, ARIMA ou GARCH".into()
                    )),
                };

                let (jb, p) = greeners::Diagnostics::jarque_bera(&series)
                    .map_err(|e| HayashiError::Runtime(format!("jb: {e}")))?;

                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(50);
                println!("\nJarque-Bera Test  —  n = {}", series.len());
                println!("{sep}");
                println!("H₀: resíduos normalmente distribuídos");
                println!("{sep}");
                println!("{:<18} {:>10} {:>10} {:>4}", "Teste", "JB", "p-value", "");
                println!("{sep}");
                println!("{:<18} {:>10.4} {:>10.4} {:>4}",
                    "Jarque-Bera ~ χ²(2)", jb, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();

                Ok(Value::Nil)
            }

            // bgodfrey(model, lags=4)
            // Breusch-Godfrey: H₀: sem autocorrelação serial nos resíduos OLS
            // Requer modelo OLS (precisa da matriz X para a regressão auxiliar)
            "bgodfrey" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "bgodfrey() requer um modelo OLS".into()
                    ));
                }

                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type(
                        "bgodfrey() suporta apenas modelos OLS".into()
                    )),
                };

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 4,
                };

                let (lm, p, df) = greeners::SpecificationTests::breusch_godfrey_test(
                    &ols.residuals, &ols.x, lags
                ).map_err(|e| HayashiError::Runtime(format!("bgodfrey: {e}")))?;

                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(54);
                println!("\nBreusch-Godfrey LM Test  —  lags = {lags}");
                println!("{sep}");
                println!("H₀: sem autocorrelação serial de ordem {lags}");
                println!("{sep}");
                println!("{:<24} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<24} {:>10.4} {:>10.4} {:>4}",
                    format!("LM ~ χ²({df})"), lm, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();

                Ok(Value::Nil)
            }

            // aliases para bgodfrey
            "bgtest" | "bg" | "breusch_godfrey" => {
                return self.eval_call("bgodfrey", args, opts);
            }

            // archtest(df, varname, lags=5)
            // Engle (1982) LM test — H₀: sem efeitos ARCH de ordem `lags`
            // Também aceita resíduos de modelo GARCH: archtest(model, lags=5)
            "archtest" | "arch_test" | "engle_arch" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "archtest() requer uma série ou modelo GARCH".into()
                    ));
                }

                let series = match self.eval_expr(&args[0])? {
                    // série bruta: archtest(df, varname, lags=5)
                    Value::DataFrame(df) => {
                        let col_name = match args.get(1) {
                            Some(Expr::Var(n)) => n.clone(),
                            _ => return Err(HayashiError::Runtime(
                                "archtest(df, varname): segundo argumento deve ser o nome da coluna".into()
                            )),
                        };
                        Array1::from(Self::eval_col_expr(&Expr::Var(col_name), &df)?)
                    }
                    // resíduos de GARCH: archtest(model, lags=5)
                    // usa resíduos padronizados z_t = ε_t/√h_t — sob H₀ de
                    // especificação correta, z_t² não deve ter autocorrelação
                    Value::GarchResult(m) => m.standardized_residuals.clone(),
                    _ => return Err(HayashiError::Type(
                        "archtest(): primeiro argumento deve ser um DataFrame ou modelo GARCH".into()
                    )),
                };

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 5,
                };

                let res = greeners::Diagnostics::arch_test(&series, lags)
                    .map_err(|e| HayashiError::Runtime(format!("archtest: {e}")))?;

                let sep = "─".repeat(54);
                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                println!("\nARCH LM Test (Engle 1982)  —  lags = {}  n = {}", res.lags, res.n_obs);
                println!("{sep}");
                println!("H₀: sem efeitos ARCH de ordem {}", res.lags);
                println!("{sep}");
                println!("{:<22} {:>10} {:>10} {:>8}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<22} {:>10.4} {:>10.4} {:>8}",
                    format!("LM  ~ χ²({})", res.lags),
                    res.lm_stat, res.lm_pvalue, sig(res.lm_pvalue));
                println!("{:<22} {:>10.4} {:>10.4} {:>8}",
                    format!("F   ~ F({},{})", res.lags, res.n_obs.saturating_sub(res.lags + 1)),
                    res.f_stat, res.f_pvalue, sig(res.f_pvalue));
                println!("{sep}");
                println!("R² aux = {:.4}   (*** p<0.01  ** p<0.05  * p<0.10)", res.r_squared);
                println!();

                Ok(Value::Nil)
            }

            // forecast_vol(model, steps=10)
            "forecast_vol" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "forecast_vol() requer um modelo GARCH".into()
                    ));
                }

                let model = match self.eval_expr(&args[0])? {
                    Value::GarchResult(m) => m,
                    _ => return Err(HayashiError::Type(
                        "forecast_vol() requer um modelo GARCH/EGARCH/GJRGARCH".into()
                    )),
                };

                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let vol = model.forecast_volatility(steps);
                let model_label = match model.model_type {
                    greeners::GarchModelType::GARCH    => "GARCH",
                    greeners::GarchModelType::EGARCH   => "EGARCH",
                    greeners::GarchModelType::GJRGARCH => "GJR-GARCH",
                };
                let dist_label = match model.dist {
                    greeners::GarchDist::Normal   => "Normal",
                    greeners::GarchDist::StudentT => "Student-t",
                };

                let sep = "─".repeat(40);
                println!("\nForecast de Volatilidade — {model_label}({}, {}) [{dist_label}]  {steps} passos",
                         model.p, model.q);
                println!("{sep}");
                println!("{:<6} {:>14} {:>14}", "h", "var. condicional", "volatilidade");
                println!("{sep}");
                for h in 0..steps {
                    println!("{:<6} {:>14.6} {:>14.6}", h + 1, vol[h], vol[h].sqrt());
                }
                println!("{sep}");
                println!();

                Ok(Value::Nil)
            }

            // diagnostics(model)
            // Roda todos os testes aplicáveis ao tipo de modelo e imprime relatório unificado.
            // OLS:  JB, DW, Breusch-Godfrey, White, RESET, VIF, Cook's D
            // GARCH: Ljung-Box, ARCH LM, JB nos resíduos padronizados
            // ARIMA: Ljung-Box, JB nos resíduos
            "diagnostics" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "diagnostics() requer um modelo (OLS, GARCH ou ARIMA)".into()
                    ));
                }

                let sig = |p: f64| -> &'static str {
                    if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" }
                };
                let thick = "═".repeat(62);
                let thin  = "─".repeat(62);

                match self.eval_expr(&args[0])? {
                    Value::OlsResult(ols) => {
                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — OLS  (n={}  k={})", ols.residuals.len(), ols.x.ncols());
                        println!("{thick}");

                        // ── Normalidade
                        println!("\n── Normalidade dos Resíduos (Jarque-Bera)");
                        match greeners::Diagnostics::jarque_bera(&ols.residuals) {
                            Ok((jb, p)) => println!("   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}", jb, p, sig(p)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── Autocorrelação 1ª ordem
                        let dw = greeners::Diagnostics::durbin_watson(&ols.residuals);
                        let dw_label = if dw < 1.5 { "autocorr. positiva" }
                                       else if dw > 2.5 { "autocorr. negativa" }
                                       else { "sem autocorr. evidente" };
                        println!("\n── Autocorrelação 1ª Ordem (Durbin-Watson)");
                        println!("   DW = {:.4}  [{}]", dw, dw_label);

                        // ── Breusch-Godfrey
                        println!("\n── Autocorrelação Serial (Breusch-Godfrey, lags=4)");
                        match greeners::SpecificationTests::breusch_godfrey_test(&ols.residuals, &ols.x, 4) {
                            Ok((lm, p, df)) => println!("   LM ~ χ²({})   = {:>9.4}   p = {:.4}  {}", df, lm, p, sig(p)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── White
                        println!("\n── Heteroscedasticidade (White)");
                        match greeners::SpecificationTests::white_test(&ols.residuals, &ols.x) {
                            Ok((lm, p, df)) => println!("   LM ~ χ²({})   = {:>9.4}   p = {:.4}  {}", df, lm, p, sig(p)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── RESET
                        println!("\n── Especificação Funcional (RESET, power=3)");
                        let fitted = ols.result.fitted_values(&ols.x);
                        let y = &ols.residuals + &fitted;
                        match greeners::SpecificationTests::reset_test(&y, &ols.x, &fitted, 3) {
                            Ok((f, p, df1, df2)) => println!("   F ~ F({},{}) = {:>9.4}   p = {:.4}  {}", df1, df2, f, p, sig(p)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── VIF
                        println!("\n── Multicolinearidade (VIF)");
                        let names = ols.result.variable_names.as_deref().unwrap_or(&[]);
                        match greeners::Diagnostics::vif(&ols.x) {
                            Ok(vifs) => {
                                for (i, &v) in vifs.iter().enumerate() {
                                    if v.is_nan() { continue; }
                                    let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                                    let diag = if v.is_infinite() || v > 10.0 { "grave" }
                                               else if v > 5.0 { "moderado" } else { "ok" };
                                    println!("   {:<20} VIF = {:>7.3}  [{}]", name, v, diag);
                                }
                            }
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── Cook's D
                        let n   = ols.residuals.len();
                        let mse = ols.result.sigma * ols.result.sigma;
                        let cutoff = 4.0 / n as f64;
                        println!("\n── Observações Influentes (Cook's D > {:.4})", cutoff);
                        match greeners::Diagnostics::cooks_distance(&ols.residuals, &ols.x, mse) {
                            Ok(d) => {
                                let flagged: Vec<(usize, f64)> = d.iter().enumerate()
                                    .filter(|(_, &di)| di > cutoff)
                                    .map(|(i, &di)| (i + 1, di))
                                    .collect();
                                if flagged.is_empty() {
                                    println!("   Nenhuma observação influente.");
                                } else {
                                    for (i, di) in &flagged {
                                        let label = if *di > 1.0 { "muito influente" } else { "influente" };
                                        println!("   obs {:>4}  D = {:.4}  [{}]", i, di, label);
                                    }
                                }
                            }
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n{thin}");
                        println!("  *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::GarchResult(m) => {
                        let model_label = match m.model_type {
                            greeners::GarchModelType::GARCH    => "GARCH",
                            greeners::GarchModelType::EGARCH   => "EGARCH",
                            greeners::GarchModelType::GJRGARCH => "GJR-GARCH",
                        };
                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — {model_label}({}, {})  (n={})", m.p, m.q, m.n_obs);
                        println!("{thick}");

                        let std_res = &m.standardized_residuals;

                        println!("\n── Autocorrelação nos Resíduos Padronizados (Ljung-Box, lags=10)");
                        match greeners::Diagnostics::ljung_box(std_res, 10) {
                            Ok(r) => println!("   Q(10) = {:>9.4}   p = {:.4}  {}", r.q_stat, r.p_value, sig(r.p_value)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n── Efeitos ARCH Residuais (Engle LM, lags=5)");
                        match greeners::Diagnostics::arch_test(std_res, 5) {
                            Ok(r) => println!("   LM ~ χ²({}) = {:>9.4}   p = {:.4}  {}", r.lags, r.lm_stat, r.lm_pvalue, sig(r.lm_pvalue)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n── Normalidade dos Resíduos Padronizados (Jarque-Bera)");
                        match greeners::Diagnostics::jarque_bera(std_res) {
                            Ok((jb, p)) => println!("   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}", jb, p, sig(p)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n{thin}");
                        println!("  *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::ArimaResult(m) => {
                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — ARIMA");
                        println!("{thick}");

                        let resid = Array1::from_vec(m.residuals().to_vec());

                        println!("\n── Autocorrelação nos Resíduos (Ljung-Box, lags=10)");
                        match greeners::Diagnostics::ljung_box(&resid, 10) {
                            Ok(r) => println!("   Q(10) = {:>9.4}   p = {:.4}  {}", r.q_stat, r.p_value, sig(r.p_value)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n── Normalidade dos Resíduos (Jarque-Bera)");
                        match greeners::Diagnostics::jarque_bera(&resid) {
                            Ok((jb, p)) => println!("   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}", jb, p, sig(p)),
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n{thin}");
                        println!("  *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::VarResult(m) => {
                        let k = m.n_vars;
                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — VAR({})  (n={}  k={})", m.lags, m.n_obs, k);
                        println!("{thick}");

                        // ── Critérios de informação
                        println!("\n── Critérios de Informação");
                        println!("   AIC = {:.4}   BIC = {:.4}", m.aic, m.bic);

                        // ── Desvio-padrão residual por equação (diagonal de Σ_u)
                        println!("\n── Desvio-Padrão Residual por Equação");
                        for (i, name) in m.var_names.iter().enumerate() {
                            println!("   {:<22} σ = {:.6}", name, m.sigma_u[[i, i]].sqrt());
                        }

                        // ── Matriz de correlação dos resíduos (Σ_u normalizada)
                        if k > 1 {
                            println!("\n── Correlação Contemporânea dos Resíduos");
                            // header
                            let col_w = m.var_names.iter().map(|n| n.len()).max().unwrap_or(8).max(8) + 2;
                            print!("   {:>col_w$}", "");
                            for name in &m.var_names {
                                print!(" {:>col_w$}", name);
                            }
                            println!();
                            for i in 0..k {
                                print!("   {:<col_w$}", m.var_names[i]);
                                for j in 0..k {
                                    let r = m.sigma_u[[i, j]]
                                        / (m.sigma_u[[i, i]] * m.sigma_u[[j, j]]).sqrt();
                                    if i == j {
                                        print!(" {:>col_w$.4}", 1.0_f64);
                                    } else {
                                        print!(" {:>col_w$.4}", r);
                                    }
                                }
                                println!();
                            }
                        }

                        println!("\n── Nota");
                        println!("   Resíduos não são armazenados em VarResult — para LB/JB por equação,");
                        println!("   extraia a série e rode ljungbox/jb diretamente.");
                        println!("\n{thin}");
                        println!("{thick}\n");
                    }

                    Value::VecmResult(m) => {
                        let k   = m.n_vars;
                        let r   = m.rank;
                        let n   = m.n_obs as f64;
                        let eig = &m.eigenvalues; // ordenados decrescente

                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — VECM  (n={}  k={}  rank={})", m.n_obs, k, r);
                        println!("{thick}");

                        // ── Johansen trace test
                        // λ_trace(r₀) = -n Σ_{i=r₀}^{k-1} ln(1 - λ_i)  H₀: rank ≤ r₀
                        // CVs 5%: Osterwald-Lenum (1992) Tabela 1 — constante restrita
                        let cv_5pct: &[f64] = &[9.24, 19.96, 34.91, 53.12, 76.07, 102.56, 131.70];
                        println!("\n── Teste de Johansen (Trace)");
                        println!("   H₀: rank ≤ r   CVs 5%: Osterwald-Lenum (1992) Tabela 1");
                        println!("   {:<6} {:>10} {:>12} {:>10} {:>6}", "H₀:r≤", "λ_max", "λ_trace", "CV 5%", "");
                        println!("   {}", "─".repeat(46));
                        for r0 in 0..k {
                            let lam_max = if r0 < eig.len() {
                                -n * (1.0 - eig[r0]).max(1e-15).ln()
                            } else { 0.0 };
                            let trace_stat: f64 = (r0..eig.len())
                                .map(|i| -n * (1.0 - eig[i]).max(1e-15).ln())
                                .sum();
                            let cv = cv_5pct.get(k - r0 - 1).copied().unwrap_or(f64::NAN);
                            let reject = if trace_stat > cv { "*" } else { " " };
                            println!("   {:<6} {:>10.4} {:>12.4} {:>10.2} {:>6}",
                                r0, lam_max, trace_stat, cv, reject);
                        }
                        println!("   (* rejeita H₀ a 5%)");

                        // ── Velocidades de ajuste (alpha): k×rank
                        println!("\n── Velocidades de Ajuste (Alpha)  [sinal negativo = correção ao equilíbrio]");
                        for ec in 0..r {
                            println!("   Vetor EC{}", ec + 1);
                            for eq in 0..k {
                                println!("     equação {:>2}   α = {:>9.4}", eq + 1, m.alpha[[eq, ec]]);
                            }
                        }

                        // ── Vetores de cointegração (beta): k×rank
                        println!("\n── Vetores de Cointegração (Beta)");
                        for ec in 0..r {
                            println!("   EC{}:", ec + 1);
                            for var in 0..k {
                                println!("     var {:>2}   β = {:>9.4}", var + 1, m.beta[[var, ec]]);
                            }
                        }

                        println!("\n── Nota");
                        println!("   VecmResult não armazena nomes de variáveis nem resíduos.");
                        println!("   Para nomes, veja a ordem passada em vecm().");
                        println!("\n{thin}");
                        println!("{thick}\n");
                    }

                    Value::IvResult(iv) => {
                        let k   = iv.params.len();
                        let n   = iv.n_obs;
                        let df  = iv.df_resid;
                        let mse = iv.sigma * iv.sigma;
                        let names = iv.variable_names.as_deref().unwrap_or(&[]);

                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — IV/2SLS  (n={}  k={}  df={})", n, k, df);
                        println!("{thick}");

                        println!("\n── Ajuste");
                        println!("   R²  = {:.4}   σ = {:.6}   MSE = {:.6}", iv.r_squared, iv.sigma, mse);

                        println!("\n── Significância dos Coeficientes");
                        let sig = |p: f64| -> &'static str {
                            if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" }
                        };
                        println!("   {:<22} {:>8} {:>8}", "Variável", "p-value", "");
                        println!("   {}", "─".repeat(40));
                        for i in 0..k {
                            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                            println!("   {:<22} {:>8.4} {:>4}", name, iv.p_values[i], sig(iv.p_values[i]));
                        }

                        println!("\n── Testes Não Disponíveis");
                        println!("   Resíduos e matriz Z não armazenados em IvResult.");
                        println!("   • Sargan (sobreidentificação): precisa da matriz Z");
                        println!("   • Endogeneidade (Wu-Hausman): compare IV vs OLS manualmente");
                        println!("   • Instrumento fraco: verifique F da 1ª etapa (regra: F > 10)");
                        println!("\n{thin}");
                        println!("   *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::PanelResult(fe) => {
                        let k = fe.params.len();
                        let names = fe.variable_names.as_deref().unwrap_or(&[]);
                        let sig = |p: f64| -> &'static str {
                            if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" }
                        };

                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — Efeitos Fixos  (n={}  N={}  T≈{:.1}  k={})",
                            fe.n_obs, fe.n_entities,
                            fe.n_obs as f64 / fe.n_entities.max(1) as f64, k);
                        println!("{thick}");

                        println!("\n── Ajuste (Within)");
                        println!("   R² within = {:.4}   σ = {:.6}   df = {}", fe.r_squared, fe.sigma, fe.df_resid);

                        println!("\n── Significância dos Coeficientes");
                        println!("   {:<22} {:>10} {:>8} {:>4}", "Variável", "coef", "p-value", "");
                        println!("   {}", "─".repeat(48));
                        for i in 0..k {
                            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                            println!("   {:<22} {:>10.4} {:>8.4} {:>4}",
                                name, fe.params[i], fe.p_values[i], sig(fe.p_values[i]));
                        }

                        println!("\n── Testes Não Disponíveis");
                        println!("   Resíduos não armazenados em PanelResult.");
                        println!("   • Hausman FE vs RE: use hausman(fe_model, re_model)");
                        println!("   • JB / Ljung-Box: rode sobre resíduos extraídos manualmente");
                        println!("\n{thin}");
                        println!("   *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::ReResult(re) => {
                        let k = re.params.len();
                        let sig = |p: f64| -> &'static str {
                            if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" }
                        };

                        // Decomposição de variância
                        let var_e   = re.sigma_e * re.sigma_e;   // variância dos efeitos individuais
                        let var_u   = re.sigma_u * re.sigma_u;   // variância idiossincrática
                        let var_tot = var_e + var_u;
                        let icc     = if var_tot > 1e-15 { var_e / var_tot } else { 0.0 };

                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — Efeitos Aleatórios  (k={})", k);
                        println!("{thick}");

                        println!("\n── Ajuste");
                        println!("   R² geral = {:.4}", re.r_squared_overall);

                        println!("\n── Decomposição de Variância");
                        println!("   σ_e  (efeitos individuais) = {:.6}   σ_e² = {:.6}", re.sigma_e, var_e);
                        println!("   σ_u  (idiossincrático)     = {:.6}   σ_u² = {:.6}", re.sigma_u, var_u);
                        println!("   ICC  = σ_e²/(σ_e²+σ_u²)   = {:.4}   ({:.1}% da variância é entre entidades)",
                            icc, icc * 100.0);
                        println!("   θ    (peso GLS)            = {:.4}   (0→OLS  1→FE)", re.theta);

                        println!("\n── Significância dos Coeficientes");
                        println!("   {:<22} {:>10} {:>8} {:>4}", "Variável", "coef", "p-value", "");
                        println!("   {}", "─".repeat(48));
                        for i in 0..k {
                            let name = re.variable_names.as_ref()
                                .and_then(|v| v.get(i))
                                .map(|s| s.as_str())
                                .unwrap_or("const");
                            println!("   {:<22} {:>10.4} {:>8.4} {:>4}",
                                name, re.params[i], re.p_values[i], sig(re.p_values[i]));
                        }

                        println!("\n── Testes Não Disponíveis");
                        println!("   • Hausman FE vs RE: use hausman(fe_model, re_model)");
                        println!("   • BP LM test (H₀: sem efeitos individuais): σ_e²/σ_u² acima sugere efeitos");
                        println!("\n{thin}");
                        println!("   *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    _ => return Err(HayashiError::Type(
                        "diagnostics() suporta OLS, GARCH, ARIMA, VAR, VECM, IV, FE e RE".into()
                    )),
                }

                Ok(Value::Nil)
            }

            // ── VARMA(p,q) ────────────────────────────────────────────────────
            // varma(df, y1, y2, ..., p=1, q=1)
            "varma" | "varmax" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "varma(df, y1, y2, ..., p=1, q=1)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_names: Vec<String> = args[1..].iter().map(|a| match a {
                    Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                    _ => Err(HayashiError::Type("variáveis de varma() devem ser identificadores".into())),
                }).collect::<Result<_>>()?;
                let p = match opt_map.get("p") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let q = match opt_map.get("q") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }
                let result = greeners::VARMA::fit(&data, p, q)
                    .map_err(|e| HayashiError::Runtime(format!("VARMA: {e}")))?;
                Ok(Value::VarmaResult(Rc::new(result)))
            }

            // ── Decomposição sazonal ──────────────────────────────────────────
            // decompose(df, var, period=12, model=additive)
            "decompose" | "seasonal_decompose" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "decompose(df, var, period=12, model=additive)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())),
                };
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let period = match opt_map.get("period") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };
                let model_str = match opt_map.get("model") {
                    Some(Value::Str(s)) => s.as_str(),
                    _ => "additive",
                };
                let result = greeners::Decomposition::seasonal_decompose(&series, period, model_str)
                    .map_err(|e| HayashiError::Runtime(format!("decompose: {e}")))?;
                Ok(Value::DecompResult(Rc::new(result)))
            }

            // stl(df, var, period=12, sw=7, tw=0)
            "stl" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "stl(df, var, period=12, sw=7, tw=0)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())),
                };
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let period = match opt_map.get("period") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };
                let sw = match opt_map.get("sw") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 7,
                };
                let tw = match opt_map.get("tw") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 0,
                };
                let result = greeners::Decomposition::stl(&series, period, sw, tw)
                    .map_err(|e| HayashiError::Runtime(format!("stl: {e}")))?;
                Ok(Value::DecompResult(Rc::new(result)))
            }

            // ── MSTL ─────────────────────────────────────────────────────────
            // mstl(df, var, periods=[7, 365])
            "mstl" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "mstl(df, var, periods=[7,365])".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())),
                };
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let periods: Vec<usize> = match opt_map.get("periods") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v {
                        Value::Int(i)   => Ok(*i as usize),
                        Value::Float(f) => Ok(*f as usize),
                        _ => Err(HayashiError::Type("periods= deve ser lista de inteiros".into())),
                    }).collect::<Result<_>>()?,
                    Some(Value::Int(i))   => vec![*i as usize],
                    Some(Value::Float(f)) => vec![*f as usize],
                    _ => vec![7, 365],
                };
                let result = greeners::MSTL::fit(&series, &periods)
                    .map_err(|e| HayashiError::Runtime(format!("mstl: {e}")))?;
                Ok(Value::MstlResult(Rc::new(result)))
            }

            // ── Testes de proporção ───────────────────────────────────────────
            // proptest(count, n, mu=0.5)
            "proptest" | "prtest" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "proptest(count, n, mu=0.5)".into()
                    ));
                }
                let count = match self.eval_expr(&args[0])? {
                    Value::Int(v)   => v as usize,
                    Value::Float(v) => v as usize,
                    _ => return Err(HayashiError::Type("count deve ser inteiro".into())),
                };
                let n = match self.eval_expr(&args[1])? {
                    Value::Int(v)   => v as usize,
                    Value::Float(v) => v as usize,
                    _ => return Err(HayashiError::Type("n deve ser inteiro".into())),
                };
                let mu = match opt_map.get("mu") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    _ => 0.5,
                };
                let (z, p) = greeners::ProportionTests::proportions_ztest_1samp(count, n, mu)
                    .map_err(|e| HayashiError::Runtime(format!("proptest: {e}")))?;
                let p_hat = count as f64 / n as f64;
                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(56);
                println!("\nTeste de Proporção (1 amostra)");
                println!("{sep}");
                println!("  H₀: p = {mu:.4}");
                println!("  p̂ = {p_hat:.4}  (count={count}, n={n})");
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "z", z, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // proptest2(count1, n1, count2, n2)
            "proptest2" | "prtest2" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "proptest2(count1, n1, count2, n2)".into()
                    ));
                }
                let to_usize = |v: Value| -> Result<usize> { match v {
                    Value::Int(i)   => Ok(i as usize),
                    Value::Float(f) => Ok(f as usize),
                    _ => Err(HayashiError::Type("argumentos de proptest2() devem ser inteiros".into())),
                }};
                let c1 = to_usize(self.eval_expr(&args[0])?)?;
                let n1 = to_usize(self.eval_expr(&args[1])?)?;
                let c2 = to_usize(self.eval_expr(&args[2])?)?;
                let n2 = to_usize(self.eval_expr(&args[3])?)?;
                let (z, p) = greeners::ProportionTests::proportions_ztest_2samp(c1, n1, c2, n2)
                    .map_err(|e| HayashiError::Runtime(format!("proptest2: {e}")))?;
                let p1 = c1 as f64 / n1 as f64;
                let p2 = c2 as f64 / n2 as f64;
                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(56);
                println!("\nTeste de Proporção (2 amostras)");
                println!("{sep}");
                println!("  H₀: p₁ = p₂");
                println!("  p̂₁ = {p1:.4}  (count={c1}, n={n1})");
                println!("  p̂₂ = {p2:.4}  (count={c2}, n={n2})");
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "z (bicaudal)", z, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // propci(count, n, alpha=0.05)
            "propci" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("propci(count, n, alpha=0.05)".into()));
                }
                let count = match self.eval_expr(&args[0])? {
                    Value::Int(v)   => v as usize,
                    Value::Float(v) => v as usize,
                    _ => return Err(HayashiError::Type("count deve ser inteiro".into())),
                };
                let n = match self.eval_expr(&args[1])? {
                    Value::Int(v)   => v as usize,
                    Value::Float(v) => v as usize,
                    _ => return Err(HayashiError::Type("n deve ser inteiro".into())),
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    _ => 0.05,
                };
                let (lo, hi) = greeners::ProportionTests::proportion_confint(count, n, alpha)
                    .map_err(|e| HayashiError::Runtime(format!("propci: {e}")))?;
                let p_hat = count as f64 / n as f64;
                let pct = (1.0 - alpha) * 100.0;
                let sep = "─".repeat(56);
                println!("\nIC de Proporção — Wilson Score ({pct:.0}%)");
                println!("{sep}");
                println!("  p̂ = {p_hat:.4}  (count={count}, n={n})");
                println!("  IC [{pct:.0}%]: [{lo:.4}, {hi:.4}]");
                println!("{sep}");
                println!();
                Ok(Value::Nil)
            }

            // chisq2x2(a, b, c, d)  — tabela 2×2
            "chisq2x2" | "chi2_2x2" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime("chisq2x2(a, b, c, d)".into()));
                }
                let to_usize = |v: Value| -> Result<usize> { match v {
                    Value::Int(i)   => Ok(i as usize),
                    Value::Float(f) => Ok(f as usize),
                    _ => Err(HayashiError::Type("células da tabela devem ser inteiros".into())),
                }};
                let a = to_usize(self.eval_expr(&args[0])?)?;
                let b = to_usize(self.eval_expr(&args[1])?)?;
                let c = to_usize(self.eval_expr(&args[2])?)?;
                let d = to_usize(self.eval_expr(&args[3])?)?;
                let table = [[a, b], [c, d]];
                let (chi2, p) = greeners::ProportionTests::chi2_contingency(&table)
                    .map_err(|e| HayashiError::Runtime(format!("chisq2x2: {e}")))?;
                let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(56);
                println!("\nTeste Qui-Quadrado — Tabela 2×2");
                println!("{sep}");
                println!("       | Col 0 | Col 1 |  Total");
                println!("  Row 0|  {:>5} |  {:>5} |  {:>5}", a, b, a + b);
                println!("  Row 1|  {:>5} |  {:>5} |  {:>5}", c, d, c + d);
                println!("  Total|  {:>5} |  {:>5} |  {:>5}", a+c, b+d, a+b+c+d);
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "χ²(1)", chi2, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // ── Múltiplos testes ──────────────────────────────────────────────
            // multipletests(pvalues, method=bonferroni, alpha=0.05)
            "multipletests" | "multtest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "multipletests(pvalues, method=bonferroni, alpha=0.05)".into()
                    ));
                }
                let pvals_val = self.eval_expr(&args[0])?;
                let pvals: Vec<f64> = match pvals_val {
                    Value::List(lst) => lst.iter().map(|v| match v {
                        Value::Float(f) => Ok(*f),
                        Value::Int(i)   => Ok(*i as f64),
                        _ => Err(HayashiError::Type("pvalues deve ser lista de floats".into())),
                    }).collect::<Result<_>>()?,
                    _ => return Err(HayashiError::Type("primeiro arg deve ser lista de p-values".into())),
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    _ => 0.05,
                };
                let method = match opt_map.get("method") {
                    Some(Value::Str(s)) => match s.to_lowercase().as_str() {
                        "bonferroni"              => greeners::MultiTestMethod::Bonferroni,
                        "sidak"                   => greeners::MultiTestMethod::Sidak,
                        "holm" | "holm_bonferroni" | "holmbonferroni" => greeners::MultiTestMethod::HolmBonferroni,
                        "bh" | "benjamini_hochberg" | "fdr_bh"        => greeners::MultiTestMethod::BenjaminiHochberg,
                        "by" | "benjamini_yekutieli" | "fdr_by"       => greeners::MultiTestMethod::BenjaminiYekutieli,
                        other => return Err(HayashiError::Runtime(
                            format!("método desconhecido: '{other}' — use bonferroni, sidak, holm, bh, by")
                        )),
                    },
                    _ => greeners::MultiTestMethod::Bonferroni,
                };
                let method_name = format!("{:?}", method);
                let (rejects, pvals_adj) = greeners::MultipleTests::multipletests(&pvals, alpha, method)
                    .map_err(|e| HayashiError::Runtime(format!("multipletests: {e}")))?;
                let sep = "─".repeat(64);
                println!("\nMúltiplos Testes — {method_name}  (α={alpha})");
                println!("{sep}");
                println!("{:>5}  {:>12}  {:>12}  {:>8}", "#", "p original", "p ajustado", "Rejeitar?");
                println!("{sep}");
                for (i, ((p_orig, p_adj), rej)) in pvals.iter().zip(pvals_adj.iter()).zip(rejects.iter()).enumerate() {
                    let mark = if *rej { "  SIM ***" } else { "  não" };
                    println!("{:>5}  {:>12.6}  {:>12.6}  {}", i + 1, p_orig, p_adj, mark);
                }
                println!("{sep}");
                println!();
                Ok(Value::Nil)
            }

            // ── UCM — Unobserved Components Model ─────────────────────────────
            // ucm(df, var, level=local_linear, seasonal=stochastic, period=12)
            "ucm" | "uc" | "structural_ts" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ucm(df, var, level=local_linear, seasonal=stochastic, period=12)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())),
                };
                let y = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);

                let level = match opt_map.get("level") {
                    Some(Value::Str(s)) => match s.to_lowercase().as_str() {
                        "local_level" | "ll"            => greeners::UCLevel::LocalLevel,
                        "local_linear" | "local_linear_trend" | "llt" => greeners::UCLevel::LocalLinearTrend,
                        "smooth_trend" | "st"           => greeners::UCLevel::SmoothTrend,
                        "random_walk" | "rw"            => greeners::UCLevel::RandomWalk,
                        other => return Err(HayashiError::Runtime(format!(
                            "ucm: level='{other}' desconhecido — use: local_level, local_linear, smooth_trend, random_walk"
                        ))),
                    },
                    _ => greeners::UCLevel::LocalLinearTrend,
                };

                let period = match opt_map.get("period") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };

                let seasonal = match opt_map.get("seasonal") {
                    Some(Value::Str(s)) => match s.to_lowercase().as_str() {
                        "none"              => greeners::UCSeasonal::None,
                        "deterministic"     => greeners::UCSeasonal::Deterministic(period),
                        "stochastic"        => greeners::UCSeasonal::Stochastic(period),
                        other => return Err(HayashiError::Runtime(format!(
                            "ucm: seasonal='{other}' desconhecido — use: none, deterministic, stochastic"
                        ))),
                    },
                    _ => greeners::UCSeasonal::None,
                };

                let result = greeners::UnobservedComponents::fit(&y, level, seasonal)
                    .map_err(|e| HayashiError::Runtime(format!("ucm: {e}")))?;
                Ok(Value::UCResult(Rc::new(result)))
            }

            // ── GAM — Generalized Additive Model (P-splines) ─────────────────
            // gam(y ~ x2, df, smooth="x1", spline_df=10, alpha=0.1, family=gaussian, link=log)
            "gam" | "gamfit" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_linear) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let linear_names = df.formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n = y_vec.len();

                // Parse smooth= option
                let smooth_names: Vec<String> = match opt_map.get("smooth") {
                    Some(Value::Str(s))  => vec![s.clone()],
                    Some(Value::List(lst)) => lst.iter().map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type("smooth= deve ser string ou lista de strings".into())),
                    }).collect::<Result<_>>()?,
                    None => vec![],
                    _ => return Err(HayashiError::Type("smooth= deve ser string ou lista de strings".into())),
                };

                if smooth_names.is_empty() && x_linear.ncols() == 0 {
                    return Err(HayashiError::Runtime("gam: especifique termos lineares (fórmula) e/ou smooth=".into()));
                }

                let spline_df = match opt_map.get("spline_df") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };
                let degree = match opt_map.get("degree") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 3,
                };
                let alpha_pen = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v))   => *v as f64,
                    _ => 0.1,
                };

                // Build smooth basis matrix (concatenate across all smooth vars)
                let q_per = spline_df;
                let q_total = q_per * smooth_names.len().max(1);
                let mut x_smooth = ndarray::Array2::<f64>::zeros((n, q_total));
                for (k, sname) in smooth_names.iter().enumerate() {
                    let col = ndarray::Array1::from(Self::get_col_f64(&df, sname)?);
                    let basis = greeners::BSplineBasis::generate(&col, q_per, degree)
                        .map_err(|e| HayashiError::Runtime(format!("gam spline ({sname}): {e}")))?;
                    for i in 0..n {
                        for j in 0..q_per {
                            x_smooth[[i, k * q_per + j]] = basis[[i, j]];
                        }
                    }
                }
                // If no smooth vars, x_smooth must still be n×1 (placeholder)
                let x_smooth_ref = if smooth_names.is_empty() {
                    ndarray::Array2::<f64>::zeros((n, 1))
                } else {
                    x_smooth
                };

                let alpha_pen_used = if smooth_names.is_empty() { 0.0 } else { alpha_pen };

                // Parse family/link (same as GLM)
                let alpha_val = match opt_map.get("alpha") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 1.0 };
                let power_val = match opt_map.get("power") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 1.5 };
                let family = match opt_map.get("family") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal"   => greeners::Family::Gaussian,
                        "binomial" | "logistic" => greeners::Family::Binomial,
                        "poisson"               => greeners::Family::Poisson,
                        "gamma"                 => greeners::Family::Gamma,
                        "inverse_gaussian"      => greeners::Family::InverseGaussian,
                        "negbin"                => greeners::Family::NegativeBinomial(alpha_val),
                        "tweedie"               => greeners::Family::Tweedie(power_val),
                        other => return Err(HayashiError::Runtime(format!(
                            "gam: family='{other}' desconhecido — use: gaussian, binomial, poisson, gamma, negbin"
                        ))),
                    },
                    _ => greeners::Family::Gaussian,
                };
                let link = match opt_map.get("link") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "identity"  => greeners::Link::Identity,
                        "log"       => greeners::Link::Log,
                        "logit"     => greeners::Link::Logit,
                        "probit"    => greeners::Link::Probit,
                        "inverse"   => greeners::Link::InversePower,
                        "cloglog"   => greeners::Link::CLogLog,
                        other => return Err(HayashiError::Runtime(format!(
                            "gam: link='{other}' desconhecido — use: identity, log, logit, probit, inverse, cloglog"
                        ))),
                    },
                    _ => greeners::Link::Identity,
                };

                let result = greeners::GLMGam::fit_with_names(
                    &y_vec, &x_linear, &x_smooth_ref, &family, &link, alpha_pen_used,
                    Some(linear_names),
                ).map_err(|e| HayashiError::Runtime(format!("gam: {e}")))?;
                Ok(Value::GamResult(Rc::new(result)))
            }

            // ── MICE — Multiple Imputation by Chained Equations ───────────────
            // mice(df, vars=["x1","x2"], m=5, iter=10)
            "mice" | "mi" | "multiple_imputation" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "mice(df, vars=[\"x1\",\"x2\"], m=5, iter=10)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_names: Vec<String> = match opt_map.get("vars") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type("vars= deve ser lista de strings".into())),
                    }).collect::<Result<_>>()?,
                    Some(Value::Str(s)) => vec![s.clone()],
                    None => {
                        // All positional args after df are var names
                        if args.len() > 1 {
                            args[1..].iter().map(|a| match a {
                                Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                                _ => Err(HayashiError::Type("variáveis de mice() devem ser identificadores".into())),
                            }).collect::<Result<_>>()?
                        } else {
                            return Err(HayashiError::Runtime(
                                "mice: especifique vars=[\"x1\",\"x2\",...] ou liste variáveis após df".into()
                            ));
                        }
                    }
                    _ => return Err(HayashiError::Type("vars= deve ser lista de strings".into())),
                };
                let m = match opt_map.get("m") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 5,
                };
                let iter = match opt_map.get("iter") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let mut data: std::collections::HashMap<String, ndarray::Array1<f64>> = std::collections::HashMap::new();
                for vname in &var_names {
                    data.insert(vname.clone(), ndarray::Array1::from(Self::get_col_f64(&df, vname)?));
                }

                let result = greeners::MICE::impute(&data, m, iter)
                    .map_err(|e| HayashiError::Runtime(format!("mice: {e}")))?;
                println!("{result}");
                Ok(Value::MiceResult(Rc::new(result)))
            }

            // ── Markov Autoregression (Hamilton 1989 full MS-AR) ──────────────
            // msauto(df, var, k=2, p=1)
            "msauto" | "markov_ar" | "ms_ar" | "hamilton" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "msauto(df, var, k=2, p=1)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())),
                };
                let y = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let k = match opt_map.get("k") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let p = match opt_map.get("p") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let result = greeners::MarkovAutoregression::fit(&y, k, p)
                    .map_err(|e| HayashiError::Runtime(format!("msauto: {e}")))?;
                Ok(Value::MSARResult(Rc::new(result)))
            }

            // ── SVAR — Structural VAR ─────────────────────────────────────────
            // svar(df, y1, y2, ..., lags=1, id=cholesky)
            // id=cholesky  : identificação recursiva (Cholesky)
            // id=longrun   : restrições de longo prazo (Blanchard-Quah)
            "svar" | "svec" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "svar(df, y1, y2, ..., lags=1, id=cholesky)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_names: Vec<String> = args[1..].iter().map(|a| match a {
                    Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                    _ => Err(HayashiError::Type("variáveis de svar() devem ser identificadores".into())),
                }).collect::<Result<_>>()?;
                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() { data[[i, j]] = v; }
                }
                let identification = match opt_map.get("id") {
                    Some(Value::Str(s)) => match s.to_lowercase().as_str() {
                        "cholesky" | "recursive" => greeners::SVarIdentification::Cholesky,
                        "longrun" | "long_run" | "bq" | "blanchard_quah" => {
                            let mask = ndarray::Array2::from_elem((k, k), f64::NAN);
                            greeners::SVarIdentification::LongRun(mask)
                        }
                        other => return Err(HayashiError::Runtime(format!(
                            "svar: id='{other}' desconhecido — use: cholesky, longrun"
                        ))),
                    },
                    _ => greeners::SVarIdentification::Cholesky,
                };
                let result = greeners::SVAR::fit(&data, lags, identification)
                    .map_err(|e| HayashiError::Runtime(format!("svar: {e}")))?;
                Ok(Value::SVarResult(Rc::new(result)))
            }

            // sirf(model, steps=10) — Structural IRF
            "sirf" | "svar_irf" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("sirf(model, steps=10)".into()));
                }
                let model = match self.eval_expr(&args[0])? {
                    Value::SVarResult(m) => m,
                    _ => return Err(HayashiError::Type("sirf() requer um modelo SVAR".into())),
                };
                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };
                let tensor = model.structural_irf(steps)
                    .map_err(|e| HayashiError::Runtime(format!("sirf: {e}")))?;
                let k = model.var_result.n_vars;
                let names = &model.var_result.var_names;
                let sep = "─".repeat(14 + k * 12);
                println!("\nSVAR Structural IRF — {} — id: {} — {} passos",
                    format!("VAR({})", model.var_result.lags), model.identification, steps);
                for j in 0..k {
                    println!("\n  Impulso: {}", names[j]);
                    println!("  {sep}");
                    let header: String = names.iter().map(|n| format!("{:>12}", n)).collect::<Vec<_>>().join("");
                    println!("  {:>6}  {header}", "h");
                    println!("  {sep}");
                    for h in 0..steps {
                        let row: String = (0..k).map(|i| format!("{:>12.4}", tensor[[h, i, j]])).collect::<Vec<_>>().join("");
                        println!("  {:>6}  {row}", h);
                    }
                }
                println!();
                Ok(Value::Nil)
            }

            // sfevd(model, steps=10) — Structural FEVD
            "sfevd" | "svar_fevd" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("sfevd(model, steps=10)".into()));
                }
                let model = match self.eval_expr(&args[0])? {
                    Value::SVarResult(m) => m,
                    _ => return Err(HayashiError::Type("sfevd() requer um modelo SVAR".into())),
                };
                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };
                let tensor = model.structural_fevd(steps)
                    .map_err(|e| HayashiError::Runtime(format!("sfevd: {e}")))?;
                let k = model.var_result.n_vars;
                let names = &model.var_result.var_names;
                let sep = "─".repeat(14 + k * 12);
                println!("\nSVAR Structural FEVD — {} — id: {}", format!("VAR({})", model.var_result.lags), model.identification);
                for i in 0..k {
                    println!("\n  Resposta: {}", names[i]);
                    println!("  {sep}");
                    let header: String = names.iter().map(|n| format!("{:>12}", n)).collect::<Vec<_>>().join("");
                    println!("  {:>6}  {header}", "h");
                    println!("  {sep}");
                    for h in 0..steps {
                        let row: String = (0..k).map(|j| format!("{:>12.4}", tensor[[h, i, j]])).collect::<Vec<_>>().join("");
                        println!("  {:>6}  {row}", h);
                    }
                }
                println!();
                Ok(Value::Nil)
            }

            // ── 3SLS — Three Stage Least Squares ──────────────────────────────
            // threesl(df, y1~x1+z1, y2~x1+z2, instruments=["z1","z2"])
            "threesl" | "three_sls" | "3sls" | "reg3" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "threesl(df, y1~x1+z1, y2~x2+z2, instruments=[\"z1\",\"z2\"])".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };

                // Parse instruments= option
                let instr_names: Vec<String> = match opt_map.get("instruments") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type("instruments= deve ser lista de strings".into())),
                    }).collect::<Result<_>>()?,
                    Some(Value::Str(s)) => vec![s.clone()],
                    None => return Err(HayashiError::Runtime(
                        "threesl requer instruments=[\"z1\",\"z2\",...] — lista de variáveis exógenas".into()
                    )),
                    _ => return Err(HayashiError::Type("instruments= deve ser lista de strings".into())),
                };

                // Build global instrument matrix Z (n × q)
                let n = df.n_rows();
                let mut z_instr = ndarray::Array2::<f64>::zeros((n, instr_names.len()));
                for (j, zname) in instr_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, zname)?;
                    for (i, &v) in col.iter().enumerate() { z_instr[[i, j]] = v; }
                }

                // Build equations from formulas
                let mut equations: Vec<greeners::Equation> = Vec::new();
                let mut eq_var_names: Vec<Vec<String>> = Vec::new();
                for arg in &args[1..] {
                    let formula_ast = match arg {
                        Expr::Formula(f) => f.clone(),
                        _ => return Err(HayashiError::Type(
                            "threesl: cada equação deve ser uma fórmula (y ~ x1 + z1)".into()
                        )),
                    };
                    let formula_str = Self::formula_to_string(&formula_ast);
                    let g_formula = GFormula::parse(&formula_str)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let (y, x) = df.to_design_matrix(&g_formula)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let var_names = df.formula_var_names(&g_formula)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    eq_var_names.push(var_names);
                    equations.push(greeners::Equation {
                        y,
                        x,
                        name: formula_ast.lhs.clone(),
                    });
                }
                let result = greeners::ThreeSLS::fit(&equations, &z_instr)
                    .map_err(|e| HayashiError::Runtime(format!("threesl: {e}")))?;
                Ok(Value::ThreeSLSResult(ThreeSLSModel {
                    result: Rc::new(result),
                    eq_var_names,
                }))
            }

            // ── DFM — Dynamic Factor Model ────────────────────────────────────
            // dfm(df, y1, y2, ..., factors=2, order=1)
            "dfm" | "dynamic_factor" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "dfm(df, y1, y2, ..., factors=2, order=1)".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))),
                };
                let var_names: Vec<String> = args[1..].iter().map(|a| match a {
                    Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                    _ => Err(HayashiError::Type("variáveis de dfm() devem ser identificadores".into())),
                }).collect::<Result<_>>()?;
                let k_factors = match opt_map.get("factors") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let factor_order = match opt_map.get("order") {
                    Some(Value::Int(v))   => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() { data[[i, j]] = v; }
                }
                let result = greeners::DynamicFactor::fit(&data, k_factors, factor_order)
                    .map_err(|e| HayashiError::Runtime(format!("dfm: {e}")))?;
                Ok(Value::DFMResult(DFMModel { result: Rc::new(result), var_names }))
            }

            // ── Diagnósticos menores de normalidade / forma funcional ─────────

            // adtest(df, var) — Anderson-Darling test para normalidade
            "adtest" | "anderson_darling" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("adtest(df, var)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let r = greeners::Diagnostics::anderson_darling(&ndarray::Array1::from(data))
                    .map_err(|e| HayashiError::Runtime(format!("adtest: {e}")))?;
                let sep = "─".repeat(56);
                println!("\nAnderson-Darling Test (normalidade)");
                println!("{sep}");
                println!("  H₀: dados provêm de distribuição normal");
                println!("  A² (ajustado) = {:.4}  (n={})", r.statistic, r.n_obs);
                println!("{sep}");
                println!("{:<12} {:>10}", "α", "A²*_crítico");
                println!("{sep}");
                for (&sig, &cv) in r.significance_levels.iter().zip(r.critical_values.iter()) {
                    let mark = if r.statistic > cv { " ← REJEITA" } else { "" };
                    println!("{:<12.3} {:>10.3}{mark}", sig, cv);
                }
                println!("{sep}");
                println!("(Rejeita H₀ quando A²* > valor crítico)");
                println!();
                Ok(Value::Nil)
            }

            // lilliefors(df, var) — KS com parâmetros estimados
            "lilliefors" | "lillie" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("lilliefors(df, var)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let (stat, p) = greeners::Diagnostics::lilliefors(&ndarray::Array1::from(data))
                    .map_err(|e| HayashiError::Runtime(format!("lilliefors: {e}")))?;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(56);
                println!("\nLilliefors Test (normalidade — KS com parâmetros estimados)");
                println!("{sep}");
                println!("  H₀: dados provêm de distribuição normal");
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "KS (Lilliefors)", stat, p, sig);
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // omnibus(model) — D'Agostino-Pearson nos resíduos
            "omnibus" | "dagostino" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("omnibus(model)".into()));
                }
                let resids = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m.residuals.to_vec(),
                    _ => return Err(HayashiError::Type("omnibus() suporta apenas modelos OLS".into())),
                };
                let (k2, p) = greeners::Diagnostics::omnibus(&ndarray::Array1::from(resids))
                    .map_err(|e| HayashiError::Runtime(format!("omnibus: {e}")))?;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(56);
                println!("\nD'Agostino-Pearson Omnibus Test (normalidade dos resíduos)");
                println!("{sep}");
                println!("  H₀: resíduos são normalmente distribuídos");
                println!("  (combina assimetria e curtose via K² ~ χ²(2))");
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "K² ~ χ²(2)", k2, p, sig);
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // harveycollier(model) — teste de linearidade via resíduos recursivos
            "harveycollier" | "harvey_collier" | "hctest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("harveycollier(model)".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("harveycollier() suporta apenas modelos OLS".into())),
                };
                // reconstruir y = ŷ + resíduos (OlsModel não armazena y diretamente)
                let y_hat = ols.x.dot(&ols.result.params);
                let y_obs = y_hat + &ols.residuals;
                let (t, p) = greeners::Diagnostics::harvey_collier(&y_obs, &ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("harveycollier: {e}")))?;
                let sig = if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
                let sep = "─".repeat(56);
                println!("\nHarvey-Collier Test (linearidade da especificação)");
                println!("{sep}");
                println!("  H₀: especificação funcional está correta (linear)");
                println!("  (testa se média dos resíduos recursivos é zero)");
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Teste", "Estatística", "p-value", "");
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "t (HC)", t, p, sig);
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // ── ETS — Suavização Exponencial (Holt-Winters) ───────────────────

            // ets(df, var, trend=add, seasonal=add, period=12, damped=false)
            // Alias: ses (simple), hwes (Holt-Winters)
            "ets" | "ses" | "hwes" | "holtwinters" | "exponential_smoothing" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("ets(df, var, trend=add, seasonal=add, period=12, damped=false)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let y = Self::get_col_f64(&df, &var_name)?;
                // Regra para aliases:
                //   ses         → trend=none, seasonal=none
                //   hwes        → trend=add,  seasonal=add
                //   ets         → usa opções explícitas (padrão: add, add)
                let (trend_def, seas_def) = match func {
                    "ses"  => ("none", "none"),
                    "hwes" | "holtwinters" => ("add", "add"),
                    _      => ("add", "add"),
                };
                let trend_str = match opt_map.get("trend") { Some(Value::Str(s)) => s.clone(), _ => trend_def.to_string() };
                let seas_str  = match opt_map.get("seasonal") { Some(Value::Str(s)) => s.clone(), _ => seas_def.to_string() };
                let period    = match opt_map.get("period") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 12 };
                let damped    = match opt_map.get("damped") { Some(Value::Bool(b)) => *b, Some(Value::Str(s)) => s == "true" || s == "yes", _ => false };
                let trend_opt: Option<&str>   = if trend_str == "none" { None } else { Some(&trend_str) };
                let seas_opt:  Option<&str>   = if seas_str  == "none" { None } else { Some(&seas_str) };
                let seas_period = if seas_opt.is_some() { period } else { 0 };
                let result = greeners::ExponentialSmoothing::fit(
                    &ndarray::Array1::from(y.to_vec()),
                    trend_opt, seas_opt, seas_period, damped,
                ).map_err(|e| HayashiError::Runtime(format!("ets: {e}")))?;
                Ok(Value::EtsResult(Rc::new(result)))
            }

            // ── Panel Threshold (Hansen 1999) ─────────────────────────────────

            // pthresh(y ~ x1 + x2, df, q=var, id=id, threshold=auto)
            "pthresh" | "xtthresh" | "panel_threshold" | "threshold" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("pthresh(y ~ x1 + x2, df, q=threshold_var, id=entity_id)".into()));
                }
                let formula = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser fórmula y ~ x1 + x2".into())),
                };
                let df_name = match &args[1] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let q_name = match opt_map.get("q") { Some(Value::Str(s)) => s.clone(), _ => return Err(HayashiError::Runtime("pthresh requer q=variavel_threshold".into())) };
                let id_name = match opt_map.get("id") { Some(Value::Str(s)) => s.clone(), _ => return Err(HayashiError::Runtime("pthresh requer id=coluna_entidade".into())) };
                let formula_str = Self::formula_to_string(&formula);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let q_col = Self::get_col_f64(&df, &q_name)?;
                let id_col = Self::get_col_f64(&df, &id_name)?;
                let entity_ids: ndarray::Array1<i64> = ndarray::Array1::from(id_col.iter().map(|&v| v as i64).collect::<Vec<_>>());
                let q_arr = ndarray::Array1::from(q_col.to_vec());
                let result = greeners::PanelThreshold::fit(&y_vec, &x_mat, &q_arr, &entity_ids)
                    .map_err(|e| HayashiError::Runtime(format!("pthresh: {e}")))?;
                Ok(Value::ThresholdResult(Rc::new(result)))
            }

            // ── Canonical Correlation Analysis ────────────────────────────────

            // cancorr(df, xvars=["x1","x2"], yvars=["y1","y2"])
            // ou cancorr(df, x1, x2, ...) com yvars= como lista
            "cancorr" | "canon" | "cancor" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("cancorr(df, xvars=[\"x1\",\"x2\"], yvars=[\"y1\",\"y2\"])".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let x_names: Vec<String> = match opt_map.get("xvars") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v { Value::Str(s) => Ok(s.clone()), _ => Err(HayashiError::Type("xvars deve ser lista de strings".into())) }).collect::<Result<_>>()?,
                    Some(Value::Str(s)) => vec![s.clone()],
                    None => args[1..].iter().map(|a| match a { Expr::Var(n) | Expr::Str(n) => Ok(n.clone()), _ => Err(HayashiError::Type("args devem ser nomes de variáveis".into())) }).collect::<Result<_>>()?,
                    _ => return Err(HayashiError::Type("xvars= deve ser lista de strings".into())),
                };
                let y_names: Vec<String> = match opt_map.get("yvars") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v { Value::Str(s) => Ok(s.clone()), _ => Err(HayashiError::Type("yvars deve ser lista de strings".into())) }).collect::<Result<_>>()?,
                    Some(Value::Str(s)) => vec![s.clone()],
                    _ => return Err(HayashiError::Runtime("cancorr requer yvars=[\"y1\",\"y2\"]".into())),
                };
                if x_names.is_empty() || y_names.is_empty() { return Err(HayashiError::Runtime("cancorr: xvars e yvars não podem ser vazios".into())); }
                let n = df.n_rows();
                let px = x_names.len(); let py = y_names.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, px));
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, py));
                for (j, name) in x_names.iter().enumerate() { let c = Self::get_col_f64(&df, name)?; for (i, &v) in c.iter().enumerate() { x_mat[[i, j]] = v; } }
                for (j, name) in y_names.iter().enumerate() { let c = Self::get_col_f64(&df, name)?; for (i, &v) in c.iter().enumerate() { y_mat[[i, j]] = v; } }
                let result = greeners::CanCorr::fit(&x_mat, &y_mat)
                    .map_err(|e| HayashiError::Runtime(format!("cancorr: {e}")))?;
                println!("{result}");
                println!("  X vars: {}", x_names.join(", "));
                println!("  Y vars: {}", y_names.join(", "));
                Ok(Value::Nil)
            }

            // ── Estatísticas ponderadas ───────────────────────────────────────

            // summarize_w(df, var, weight=wvar, mu0=0, alpha=0.05)
            "summarize_w" | "dstats_w" | "svymean" | "wtsum" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("summarize_w(df, var, weight=wvar, mu0=0, alpha=0.05)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let weights = match opt_map.get("weight").or_else(|| opt_map.get("weights").or_else(|| opt_map.get("w"))) {
                    Some(Value::Str(wname)) => {
                        let wc = Self::get_col_f64(&df, wname)?;
                        Some(ndarray::Array1::from(wc.to_vec()))
                    }
                    _ => None,
                };
                let w_ref = weights.as_ref();
                let ds = greeners::DescrStatsW::new(&ndarray::Array1::from(data.to_vec()), w_ref)
                    .map_err(|e| HayashiError::Runtime(format!("summarize_w: {e}")))?;
                let mu0 = match opt_map.get("mu0") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 0.0 };
                let alpha = match opt_map.get("alpha") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 0.05 };
                let (t_stat, t_p) = ds.ttest_mean(mu0).map_err(|e| HayashiError::Runtime(format!("summarize_w t-test: {e}")))?;
                let (ci_lo, ci_hi) = ds.conf_int_mean(alpha).map_err(|e| HayashiError::Runtime(format!("summarize_w CI: {e}")))?;
                let label = w_ref.map_or("(pesos iguais)".to_string(), |_| format!("(ponderado)"));
                println!("\n{:=^60}", format!(" DescrStats {label} — {var_name} "));
                println!("{:<20} {:>12}   {:<20} {:>12}", "N", ds.nobs as usize, "Σ pesos", format!("{:.2}", ds.sum_weights));
                println!("{:<20} {:>12.6}   {:<20} {:>12.6}", "Média", ds.mean, "Desvio padrão", ds.std);
                println!("{:<20} {:>12.6}   {:<20} {:>12.6}", "Mín", ds.min, "Máx", ds.max);
                println!("{:<20} {:>12.6}   {:<20} {:>12.6}", "P25", ds.q25, "Mediana", ds.median);
                println!("{:<20} {:>12.6}   {:<20} {:>12.6}", "P75", ds.q75, "Variância", ds.var);
                println!("{:<20} {:>12.6}   {:<20} {:>12.6}", "Assimetria", ds.skewness, "Curtose", ds.kurtosis);
                println!("{:-^60}", "");
                println!("  t-test H₀: μ = {:.4}    t = {:.4}   p = {:.4}", mu0, t_stat, t_p);
                println!("  IC {}%: [{:.6}, {:.6}]", ((1.0 - alpha) * 100.0) as usize, ci_lo, ci_hi);
                println!("{:=^60}", "");
                Ok(Value::Nil)
            }

            // ── Tabstat — tabela de estatísticas por grupo ────────────────────

            // tabstat(df, var1, var2, ..., by=grupo, stats=[mean,sd,n,p25,p75,min,max,sum])
            "tabstat" | "tabstats" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("tabstat(df, var1, ..., by=grupo, stats=[mean,sd,n])".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_names: Vec<String> = {
                    let mut v = Vec::new();
                    for a in &args[1..] {
                        match a { Expr::Var(n) | Expr::Str(n) => v.push(n.clone()),
                            _ => return Err(HayashiError::Type("args devem ser nomes de variáveis".into())) }
                    }
                    v
                };
                if var_names.is_empty() { return Err(HayashiError::Runtime("tabstat: forneça ao menos uma variável".into())); }
                // stats= lista de estatísticas a mostrar
                let default_stats = vec!["mean".to_string(), "sd".to_string(), "n".to_string()];
                let stat_list: Vec<String> = match opt_map.get("stats") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v { Value::Str(s) => s.clone(), _ => "mean".into() }).collect(),
                    Some(Value::Str(s)) => vec![s.clone()],
                    _ => default_stats,
                };
                let by_col: Option<Vec<f64>> = match opt_map.get("by") {
                    Some(Value::Str(bname)) => Some(Self::get_col_f64(&df, bname)?.to_vec()),
                    _ => None,
                };
                // Coleta grupos únicos
                let groups: Vec<Option<String>> = if let Some(ref bv) = by_col {
                    let mut uniq: Vec<f64> = bv.clone(); uniq.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); uniq.dedup();
                    uniq.into_iter().map(|g| Some(format!("{}", g as i64))).collect()
                } else { vec![None] };
                // Cabeçalho
                let stat_w = 10usize;
                let var_w  = var_names.iter().map(|n| n.len()).max().unwrap_or(6).max(6);
                println!("\n{:=^70}", " tabstat ");
                print!("{:>var_w$}", "");
                for s in &stat_list { print!("  {:>stat_w$}", s); }
                println!();
                println!("{}", "-".repeat(var_w + stat_list.len() * (stat_w + 2)));
                for grp in &groups {
                    if let Some(ref g) = grp { println!("  grupo = {g}"); }
                    for vname in &var_names {
                        let col = Self::get_col_f64(&df, vname)?;
                        let data: Vec<f64> = if let Some(ref bv) = by_col {
                            let gval: f64 = grp.as_ref().unwrap().parse::<f64>().unwrap_or(f64::NAN);
                            col.iter().zip(bv.iter()).filter(|(_, &b)| (b - gval).abs() < 1e-9).map(|(&c, _)| c).filter(|v| !v.is_nan()).collect()
                        } else { col.iter().cloned().filter(|v| !v.is_nan()).collect() };
                        if data.is_empty() { continue; }
                        let n = data.len() as f64;
                        let mean = data.iter().sum::<f64>() / n;
                        let sd = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
                        let mut sorted = data.clone(); sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
                        let min = sorted[0]; let max = *sorted.last().unwrap();
                        let sum = data.iter().sum::<f64>();
                        let pct = |p: f64| { let idx = (p / 100.0 * (n - 1.0)).round() as usize; sorted[idx.min(sorted.len() - 1)] };
                        print!("{:>var_w$}", vname);
                        for s in &stat_list {
                            let val = match s.as_str() {
                                "mean" | "avg"   => mean,
                                "sd"   | "std"   => sd,
                                "var"            => sd * sd,
                                "n"    | "count" => n,
                                "sum"            => sum,
                                "min"            => min,
                                "max"            => max,
                                "p25"  | "q1"    => pct(25.0),
                                "p50"  | "median"=> pct(50.0),
                                "p75"  | "q3"    => pct(75.0),
                                "p10"            => pct(10.0),
                                "p90"            => pct(90.0),
                                "iqr"            => pct(75.0) - pct(25.0),
                                "range"          => max - min,
                                "cv"             => if mean.abs() > 1e-15 { sd / mean.abs() } else { f64::NAN },
                                _                => f64::NAN,
                            };
                            if s == "n" { print!("  {:>stat_w$}", val as usize); } else { print!("  {:>12.4}", val); }
                        }
                        println!();
                    }
                }
                println!("{:=^70}", "");
                Ok(Value::Nil)
            }

            // ── xtsum — decomposição within/between ───────────────────────────

            // xtsum(df, var, id=entity_col)
            "xtsum" | "xt_summary" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("xtsum(df, var1, var2, ..., id=entity_col)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let id_name = match opt_map.get("id") { Some(Value::Str(s)) => s.clone(), _ => return Err(HayashiError::Runtime("xtsum requer id=coluna_entidade".into())) };
                let var_names: Vec<String> = { let mut v = Vec::new(); for a in &args[1..] { match a { Expr::Var(n) | Expr::Str(n) => v.push(n.clone()), _ => {} } } v };
                if var_names.is_empty() { return Err(HayashiError::Runtime("xtsum: forneça ao menos uma variável".into())); }
                let id_col = Self::get_col_f64(&df, &id_name)?;
                // Identifica entidades únicas
                let mut ids_uniq: Vec<f64> = id_col.to_vec(); ids_uniq.sort_by(|a,b| a.partial_cmp(b).unwrap()); ids_uniq.dedup();
                let n_total = df.n_rows();
                let n_entities = ids_uniq.len();
                println!("\n{:=^78}", " xtsum — decomposição within/between ");
                println!("{:<20} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}",
                    "Variável", "Tipo", "Média", "DP", "Mín", "Máx", "N");
                println!("{}", "-".repeat(78));
                for vname in &var_names {
                    let col = Self::get_col_f64(&df, vname)?;
                    let vals: Vec<f64> = col.iter().cloned().collect();
                    // Overall
                    let n_total_f = n_total as f64;
                    let mean_ov = vals.iter().sum::<f64>() / n_total_f;
                    let var_ov  = vals.iter().map(|x| (x - mean_ov).powi(2)).sum::<f64>() / (n_total_f - 1.0).max(1.0);
                    let sd_ov   = var_ov.sqrt();
                    let min_ov  = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_ov  = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    println!("{:<20} | {:>7} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8}",
                        vname, "overall", mean_ov, sd_ov, min_ov, max_ov, n_total);
                    // Between: média por entidade
                    let group_means: Vec<f64> = ids_uniq.iter().map(|&gid| {
                        let gvals: Vec<f64> = id_col.iter().zip(vals.iter()).filter(|(&id, _)| (id - gid).abs() < 1e-9).map(|(_, &v)| v).collect();
                        if gvals.is_empty() { f64::NAN } else { gvals.iter().sum::<f64>() / gvals.len() as f64 }
                    }).collect();
                    let n_b = n_entities as f64;
                    let mean_b = group_means.iter().filter(|v| !v.is_nan()).sum::<f64>() / n_b;
                    let var_b  = group_means.iter().filter(|v| !v.is_nan()).map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0).max(1.0);
                    let sd_b   = var_b.sqrt();
                    let min_b  = group_means.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_b  = group_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    println!("{:<20} | {:>7} | {:>8} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8}",
                        "", "between", "", sd_b, min_b, max_b, n_entities);
                    // Within: desvio de cada obs. em relação à média do seu grupo
                    let within_vals: Vec<f64> = id_col.iter().zip(vals.iter()).map(|(&id, &v)| {
                        let gm = group_means[ids_uniq.iter().position(|&g| (g - id).abs() < 1e-9).unwrap_or(0)];
                        v - gm + mean_ov  // demeaned + overall mean
                    }).collect();
                    let n_w = n_total as f64;
                    let var_w  = within_vals.iter().map(|x| (x - mean_ov).powi(2)).sum::<f64>() / (n_w - 1.0).max(1.0);
                    let sd_w   = var_w.sqrt();
                    let min_w  = within_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_w  = within_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    println!("{:<20} | {:>7} | {:>8} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8}",
                        "", "within", "", sd_w, min_w, max_w, n_total);
                }
                println!("{:=^78}", "");
                println!("  Entidades: {}   Períodos médios: {:.1}", n_entities, n_total as f64 / n_entities as f64);
                Ok(Value::Nil)
            }

            // ── Testes não-paramétricos ───────────────────────────────────────

            // spearman(df, var1, var2) — correlação de Spearman
            "spearman" | "spearman_rho" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("spearman(df, var1, var2)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let v1 = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg: nome de variável".into())) };
                let v2 = match &args[2] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("terceiro arg: nome de variável".into())) };
                let x = Self::get_col_f64(&df, &v1)?.to_vec();
                let y = Self::get_col_f64(&df, &v2)?.to_vec();
                let n = x.len().min(y.len());
                if n < 3 { return Err(HayashiError::Runtime("spearman: n < 3".into())); }
                // Ranking com ties (média dos ranks)
                let rank = |vals: &[f64]| -> Vec<f64> {
                    let mut idx: Vec<usize> = (0..vals.len()).collect();
                    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));
                    let mut ranks = vec![0.0f64; vals.len()];
                    let mut i = 0;
                    while i < idx.len() {
                        let mut j = i + 1;
                        while j < idx.len() && (vals[idx[j]] - vals[idx[i]]).abs() < 1e-12 { j += 1; }
                        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-indexed
                        for k in i..j { ranks[idx[k]] = avg_rank; }
                        i = j;
                    }
                    ranks
                };
                let rx = rank(&x[..n]);
                let ry = rank(&y[..n]);
                let nf = n as f64;
                let mean_rx = rx.iter().sum::<f64>() / nf;
                let mean_ry = ry.iter().sum::<f64>() / nf;
                let num: f64 = rx.iter().zip(ry.iter()).map(|(a, b)| (a - mean_rx) * (b - mean_ry)).sum();
                let dx: f64  = rx.iter().map(|a| (a - mean_rx).powi(2)).sum::<f64>();
                let dy: f64  = ry.iter().map(|b| (b - mean_ry).powi(2)).sum::<f64>();
                let rho = if dx * dy < 1e-15 { 0.0 } else { num / (dx * dy).sqrt() };
                let t_stat = rho * ((nf - 2.0) / (1.0 - rho * rho).max(1e-15)).sqrt();
                let p_val = greeners::t_pvalue_two(t_stat.abs(), nf - 2.0);
                println!("\n  Spearman ρ({v1}, {v2})");
                println!("  ρ = {rho:.6}   t = {t_stat:.4}   df = {}   p = {p_val:.4}", n - 2);
                println!("  H₀: ρₛ = 0 (não correlacionadas em ranking)");
                Ok(Value::Nil)
            }

            // ranksum(df, var, by=group) — Mann-Whitney U / Wilcoxon rank-sum
            "ranksum" | "mannwhitney" | "wilcoxon_rs" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("ranksum(df, var, by=group_col)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg: nome de variável".into())) };
                let by_name = match opt_map.get("by") { Some(Value::Str(s)) => s.clone(), _ => return Err(HayashiError::Runtime("ranksum requer by=coluna_grupo".into())) };
                let y_col   = Self::get_col_f64(&df, &var_name)?;
                let grp_col = Self::get_col_f64(&df, &by_name)?;
                let n_total = y_col.len();
                // Separar em dois grupos pelo valor único
                let mut gvals: Vec<f64> = grp_col.to_vec(); gvals.sort_by(|a,b| a.partial_cmp(b).unwrap()); gvals.dedup();
                if gvals.len() != 2 { return Err(HayashiError::Runtime(format!("ranksum: by= deve ter exatamente 2 grupos únicos; encontrou {}", gvals.len()))); }
                let g0: Vec<f64> = (0..n_total).filter(|&i| (grp_col[i] - gvals[0]).abs() < 1e-9).map(|i| y_col[i]).collect();
                let g1: Vec<f64> = (0..n_total).filter(|&i| (grp_col[i] - gvals[1]).abs() < 1e-9).map(|i| y_col[i]).collect();
                let n1 = g0.len(); let n2 = g1.len();
                if n1 < 1 || n2 < 1 { return Err(HayashiError::Runtime("ranksum: um dos grupos está vazio".into())); }
                // Rank combinado com ties
                let mut combined: Vec<(f64, usize)> = g0.iter().map(|&v| (v, 0)).chain(g1.iter().map(|&v| (v, 1))).collect();
                combined.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let m = combined.len();
                let mut ranks = vec![0.0f64; m];
                let mut i = 0;
                while i < m {
                    let mut j = i + 1;
                    while j < m && (combined[j].0 - combined[i].0).abs() < 1e-12 { j += 1; }
                    let avg_rank = (i + j + 1) as f64 / 2.0;
                    for k in i..j { ranks[k] = avg_rank; }
                    i = j;
                }
                let w1: f64 = (0..m).filter(|&k| combined[k].1 == 0).map(|k| ranks[k]).sum();
                let u1 = w1 - (n1 * (n1 + 1)) as f64 / 2.0;
                let u2 = (n1 * n2) as f64 - u1;
                let u = u1.min(u2);
                // Normal approximation (large sample)
                let n1f = n1 as f64; let n2f = n2 as f64; let nf = m as f64;
                let mu_u = n1f * n2f / 2.0;
                let var_u = n1f * n2f * (nf + 1.0) / 12.0;
                let z_stat = (u - mu_u) / var_u.sqrt();
                // p-value via normal approximation
                let p_normal = 2.0 * (1.0 - Self::norm_cdf(z_stat.abs()));
                println!("\n  Mann-Whitney U / Wilcoxon Rank-Sum");
                println!("  {}: n₁={n1}  {}: n₂={n2}", var_name, by_name);
                println!("  Grupo {}:  {var_name}", gvals[0] as i64);
                println!("  Grupo {}:  {var_name}", gvals[1] as i64);
                println!("  W (rank-sum grupo 0) = {w1:.1}");
                println!("  U₁ = {u1:.1}   U₂ = {u2:.1}   U = {u:.1}");
                println!("  z = {z_stat:.4}   p = {p_normal:.4}   (aprox. normal)");
                println!("  H₀: distribuição de {var_name} igual nos dois grupos");
                Ok(Value::Nil)
            }

            // kruskal(df, var, by=group) — Kruskal-Wallis (≥ 2 grupos)
            "kruskal" | "kwallis" | "kruskal_wallis" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("kruskal(df, var, by=group_col)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg: nome de variável".into())) };
                let by_name = match opt_map.get("by") { Some(Value::Str(s)) => s.clone(), _ => return Err(HayashiError::Runtime("kruskal requer by=coluna_grupo".into())) };
                let y_col   = Self::get_col_f64(&df, &var_name)?;
                let grp_col = Self::get_col_f64(&df, &by_name)?;
                let n = y_col.len();
                let mut gvals: Vec<f64> = grp_col.to_vec(); gvals.sort_by(|a,b| a.partial_cmp(b).unwrap()); gvals.dedup();
                let k = gvals.len();
                if k < 2 { return Err(HayashiError::Runtime("kruskal: precisa de pelo menos 2 grupos".into())); }
                // Rank global com ties
                let mut indexed: Vec<(f64, usize)> = y_col.iter().cloned().enumerate().map(|(i,v)| (v, i)).collect();
                indexed.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut global_ranks = vec![0.0f64; n];
                let mut i = 0;
                let n_m = indexed.len();
                while i < n_m {
                    let mut j = i + 1;
                    while j < n_m && (indexed[j].0 - indexed[i].0).abs() < 1e-12 { j += 1; }
                    let avg_rank = (i + j + 1) as f64 / 2.0;
                    for k2 in i..j { global_ranks[indexed[k2].1] = avg_rank; }
                    i = j;
                }
                let nf = n as f64;
                let mean_rank = (nf + 1.0) / 2.0;
                // H statistic
                let h_num: f64 = gvals.iter().map(|&gid| {
                    let idxs: Vec<usize> = (0..n).filter(|&i| (grp_col[i] - gid).abs() < 1e-9).collect();
                    let ni = idxs.len() as f64;
                    if ni == 0.0 { return 0.0; }
                    let rbar = idxs.iter().map(|&i| global_ranks[i]).sum::<f64>() / ni;
                    ni * (rbar - mean_rank).powi(2)
                }).sum();
                let h = 12.0 / (nf * (nf + 1.0)) * h_num;
                let df_kw = (k - 1) as f64;
                let p_val = greeners::chi2_pvalue(h, (k - 1) as f64);
                println!("\n  Kruskal-Wallis H");
                println!("  {var_name} por {by_name}  ({k} grupos, N={n})");
                for gid in &gvals {
                    let gdata: Vec<f64> = (0..n).filter(|&i| (grp_col[i] - gid).abs() < 1e-9).map(|i| y_col[i]).collect();
                    let gn = gdata.len();
                    let gm = gdata.iter().sum::<f64>() / gn as f64;
                    let rbar = (0..n).filter(|&i| (grp_col[i] - gid).abs() < 1e-9).map(|i| global_ranks[i]).sum::<f64>() / gn as f64;
                    println!("    grupo {:>4}: n={gn:>4}  média={gm:>8.4}  rank_médio={rbar:>8.2}", *gid as i64);
                }
                println!("  H = {h:.4}   df = {df_kw}   p = {p_val:.4}   χ² approx.");
                println!("  H₀: mesma distribuição em todos os grupos");
                Ok(Value::Nil)
            }

            // signrank(df, var, mu0=0) — Wilcoxon signed-rank (uma amostra ou pares)
            "signrank" | "wilcoxon_sr" | "wilcoxon_signed_rank" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("signrank(df, var, mu0=0)  ou  signrank(df, d)  onde d = x - y".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg: nome de variável".into())) };
                let mu0 = match opt_map.get("mu0") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 0.0 };
                let data = Self::get_col_f64(&df, &var_name)?;
                let diffs: Vec<f64> = data.iter().map(|&v| v - mu0).filter(|v| v.abs() > 1e-15).collect();
                let n = diffs.len();
                if n == 0 { return Err(HayashiError::Runtime("signrank: todos os diffs são zero".into())); }
                // Rank dos |diffs|
                let mut abs_indexed: Vec<(f64, usize, f64)> = diffs.iter().enumerate().map(|(i, &d)| (d.abs(), i, d.signum())).collect();
                abs_indexed.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut ranks = vec![0.0f64; n];
                let mut i = 0;
                while i < n {
                    let mut j = i + 1;
                    while j < n && (abs_indexed[j].0 - abs_indexed[i].0).abs() < 1e-12 { j += 1; }
                    let avg_rank = (i + j + 1) as f64 / 2.0;
                    for k in i..j { ranks[abs_indexed[k].1] = avg_rank; }
                    i = j;
                }
                let w_plus:  f64 = (0..n).filter(|&i| diffs[i] > 0.0).map(|i| ranks[i]).sum();
                let w_minus: f64 = (0..n).filter(|&i| diffs[i] < 0.0).map(|i| ranks[i]).sum();
                let w = w_plus.min(w_minus);
                // Normal approx (n ≥ 10)
                let nf = n as f64;
                let mu_w = nf * (nf + 1.0) / 4.0;
                let var_w = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0;
                let z_stat = (w - mu_w) / var_w.sqrt();
                let p_val = 2.0 * (1.0 - Self::norm_cdf(z_stat.abs()));
                println!("\n  Wilcoxon Signed-Rank Test");
                println!("  H₀: mediana({var_name}) = {mu0}");
                println!("  n = {n}  (excluindo diffs ≈ 0)");
                println!("  W+ = {w_plus:.1}   W- = {w_minus:.1}   W = {w:.1}");
                println!("  z = {z_stat:.4}   p = {p_val:.4}   (aprox. normal)");
                Ok(Value::Nil)
            }

            // bitest(count, n, mu=0.5) — teste binomial (sinal)
            "bitest" | "signtest" | "binom_test" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("bitest(successes, n, mu=0.5)  ou  bitest(df, var, mu=0)".into()));
                }
                // Suporta dois modos:
                // 1) bitest(count, n, mu=0.5)       — contagens diretas
                // 2) bitest(df, var, mu=0, by=group) — positivos vs negativos na coluna
                let first_val = self.eval_expr(&args[0])?;
                match first_val {
                    Value::Int(count) => {
                        // modo 1
                        let n_trials = match self.eval_expr(&args[1])? { Value::Int(v) => v as usize, Value::Float(v) => v as usize, _ => return Err(HayashiError::Type("segundo arg: n (inteiro)".into())) };
                        let mu = match opt_map.get("mu").or_else(|| opt_map.get("p")) { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 0.5 };
                        let k = count as usize;
                        // p-value via normal approx (prop test)
                        let nf = n_trials as f64; let phat = k as f64 / nf;
                        let se = (mu * (1.0 - mu) / nf).sqrt();
                        let z = (phat - mu) / se;
                        let p = 2.0 * (1.0 - Self::norm_cdf(z.abs()));
                        println!("\n  Binomial / Sign Test");
                        println!("  Sucessos: {k}   n: {n_trials}   p̂ = {:.4}   H₀: p = {mu}", phat);
                        println!("  z = {z:.4}   p = {p:.4}");
                    }
                    Value::DataFrame(_) | Value::Nil => {
                        // Tentativa de modo 2: bitest(df, var, mu=0)
                        let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                        let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                        let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg: nome de variável".into())) };
                        let mu0 = match opt_map.get("mu").or_else(|| opt_map.get("mu0")) { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 0.0 };
                        let data = Self::get_col_f64(&df, &var_name)?;
                        let pos = data.iter().filter(|&&v| v > mu0).count();
                        let neg = data.iter().filter(|&&v| v < mu0).count();
                        let ties = data.len() - pos - neg;
                        let n_eff = pos + neg;
                        let phat = pos as f64 / n_eff as f64;
                        let nf = n_eff as f64;
                        let z = (phat - 0.5) * nf.sqrt() / 0.5;
                        let p = 2.0 * (1.0 - Self::norm_cdf(z.abs()));
                        println!("\n  Sign Test  ({var_name} vs {mu0})");
                        println!("  + : {pos}   - : {neg}   empates: {ties}   n efetivo: {n_eff}");
                        println!("  p̂(+) = {phat:.4}   z = {z:.4}   p = {p:.4}");
                        println!("  H₀: P(X > {mu0}) = 0.5");
                    }
                    _ => return Err(HayashiError::Type("bitest: primeiro arg deve ser inteiro (count) ou DataFrame".into())),
                }
                Ok(Value::Nil)
            }

            // ══════════════════════════════════════════════════════════════════
            // ── Testes de raiz unitária ────────────────────────────────────────
            // ══════════════════════════════════════════════════════════════════

            // ══════════════════════════════════════════════════════════════════
            // ── Filtros de ciclo de negócios ──────────────────────────────────
            // ══════════════════════════════════════════════════════════════════

            // hpfilter(df, var, lambda=1600)  →  cria df.var_trend e df.var_cycle
            "hpfilter" | "hp_filter" | "hprescott" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("hpfilter(df, var, lambda=1600)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("hpfilter: primeiro arg deve ser DataFrame".into())) };
                let mut df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("hpfilter: segundo arg deve ser nome de variável".into())) };
                let lambda = match opt_map.get("lambda") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 1600.0 };
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?.to_vec());
                let (trend, cycle) = greeners::TimeSeries::hp_filter(&series, lambda).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let trend_name = format!("{var_name}_trend");
                let cycle_name = format!("{var_name}_cycle");
                df.insert(trend_name.clone(), trend).map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                df.insert(cycle_name.clone(), cycle).map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!("hpfilter: λ={lambda}  →  {trend_name} e {cycle_name} adicionadas a {df_name}");
                self.env.set(&df_name, Value::DataFrame(df));
                Ok(Value::Nil)
            }

            // bkfilter(df, var, low=6, high=32, k=12)  →  cria df.var_cycle
            "bkfilter" | "bk_filter" | "baxter_king" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("bkfilter(df, var, low=6, high=32, k=12)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("bkfilter: primeiro arg deve ser DataFrame".into())) };
                let mut df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("bkfilter: segundo arg deve ser nome de variável".into())) };
                let low  = match opt_map.get("low")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 6 };
                let high = match opt_map.get("high") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 32 };
                let k    = match opt_map.get("k")    { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 12 };
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?.to_vec());
                let cycle = greeners::TimeSeries::bk_filter(&series, low, high, k).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cycle_name = format!("{var_name}_cycle");
                df.insert(cycle_name.clone(), cycle).map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!("bkfilter: períodos [{low},{high}] k={k}  →  {cycle_name} adicionada a {df_name}");
                self.env.set(&df_name, Value::DataFrame(df));
                Ok(Value::Nil)
            }

            // cffilter(df, var, low=6, high=32, drift=false)  →  cria df.var_cycle
            "cffilter" | "cf_filter" | "christiano_fitzgerald" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("cffilter(df, var, low=6, high=32, drift=false)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("cffilter: primeiro arg deve ser DataFrame".into())) };
                let mut df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("cffilter: segundo arg deve ser nome de variável".into())) };
                let low   = match opt_map.get("low")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 6 };
                let high  = match opt_map.get("high") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 32 };
                let drift = matches!(opt_map.get("drift"), Some(Value::Bool(true)));
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?.to_vec());
                let cycle = greeners::TimeSeries::cf_filter(&series, low, high, drift).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cycle_name = format!("{var_name}_cycle");
                df.insert(cycle_name.clone(), cycle).map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!("cffilter: períodos [{low},{high}] drift={drift}  →  {cycle_name} adicionada a {df_name}");
                self.env.set(&df_name, Value::DataFrame(df));
                Ok(Value::Nil)
            }

            // ══════════════════════════════════════════════════════════════════
            // ── Regressão penalizada — Lasso / Ridge / ElasticNet ─────────────
            // ══════════════════════════════════════════════════════════════════

            // ridge(formula, df, alpha=1.0)
            // β_ridge = (X'X + αI)^{-1} X'y  (forma fechada)
            "ridge" | "ridge_reg" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("ridge(formula, df, alpha=1.0)".into())); }
                let formula = match &args[0] { Expr::Formula(f) => f.clone(), _ => return Err(HayashiError::Type("ridge: primeiro arg deve ser fórmula".into())) };
                let df_name = match &args[1] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("ridge: segundo arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let alpha = match opt_map.get("alpha") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 1.0_f64 };
                let formula_str = Self::formula_to_string(&formula);
                let gformula = GFormula::parse(&formula_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df.to_design_matrix(&gformula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n = x.nrows();
                let k = x.ncols();
                // (X'X + αI)^{-1} X'y
                let xtx = x.t().dot(&x);
                let mut reg = xtx.clone();
                for i in 0..k { reg[[i, i]] += alpha; }
                let reg_inv = reg.inv().map_err(|_| HayashiError::Runtime("ridge: matriz singular".into()))?;
                let params = reg_inv.dot(&x.t().dot(&y));
                let y_hat = x.dot(&params);
                let resid: Vec<f64> = y.iter().zip(y_hat.iter()).map(|(a, b)| a - b).collect();
                let sse: f64 = resid.iter().map(|r| r * r).sum();
                let sst: f64 = { let m = y.mean().unwrap_or(0.0); y.iter().map(|v| (v - m).powi(2)).sum() };
                let r2 = 1.0 - sse / sst;
                let var_names: Vec<String> = gformula.independents.clone();
                println!("\n{:=^60}", " Ridge Regression ");
                println!("  Formula: {formula_str}   α = {alpha}");
                println!("  n = {n}   k = {k}   R² = {r2:.4}");
                println!("\n  {:<20} {:>12}", "Variable", "Coeff");
                println!("  {}", "─".repeat(33));
                if gformula.intercept && var_names.len() < params.len() {
                    println!("  {:<20} {:>12.6}", "const", params[0]);
                    for (i, name) in var_names.iter().enumerate() { println!("  {:<20} {:>12.6}", name, params[i + 1]); }
                } else {
                    for (i, name) in var_names.iter().enumerate() { println!("  {:<20} {:>12.6}", name, params[i]); }
                }
                Ok(Value::Nil)
            }

            // lasso(formula, df, alpha=1.0, tol=1e-6, max_iter=10000)
            // Coordinate descent para Lasso (L1), com intercept não penalizado
            "lasso" | "lasso_reg" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("lasso(formula, df, alpha=1.0, tol=1e-6, max_iter=10000)".into())); }
                let formula = match &args[0] { Expr::Formula(f) => f.clone(), _ => return Err(HayashiError::Type("lasso: primeiro arg deve ser fórmula".into())) };
                let df_name = match &args[1] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("lasso: segundo arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let alpha    = match opt_map.get("alpha")    { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 1.0_f64 };
                let tol      = match opt_map.get("tol")      { Some(Value::Float(v)) => *v, _ => 1e-6_f64 };
                let max_iter = match opt_map.get("max_iter") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 10_000usize };
                let formula_str = Self::formula_to_string(&formula);
                let gformula = GFormula::parse(&formula_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df.to_design_matrix(&gformula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n_obs = x.nrows();
                let k = x.ncols();
                let has_intercept = gformula.intercept;
                let start_col = if has_intercept { 1 } else { 0 };
                // Centrar y; padronizar X (colunas covariáveis)
                let y_mean = y.sum() / n_obs as f64;
                let y_c: ndarray::Array1<f64> = y.mapv(|v| v - y_mean);
                let mut x_std = x.clone();
                let mut col_mean: Vec<f64> = vec![0.0; k];
                let mut col_std_v: Vec<f64> = vec![1.0; k];
                for j in start_col..k {
                    let col = x.column(j);
                    let mj = col.sum() / n_obs as f64;
                    let sj = (col.iter().map(|&v| (v - mj).powi(2)).sum::<f64>() / n_obs as f64).sqrt();
                    col_mean[j] = mj;
                    if sj > 1e-12 {
                        col_std_v[j] = sj;
                        for i in 0..n_obs { x_std[[i, j]] = (x[[i, j]] - mj) / sj; }
                    }
                }
                let soft = |z: f64, lam: f64| -> f64 { if z > lam { z - lam } else if z < -lam { z + lam } else { 0.0 } };
                let xx_diag: Vec<f64> = (start_col..k).map(|j| x_std.column(j).dot(&x_std.column(j))).collect();
                // coordinate descent sem intercept (y já centralizado)
                let mut beta = ndarray::Array1::<f64>::zeros(k - start_col);
                for _iter in 0..max_iter {
                    let mut max_delta = 0.0_f64;
                    let xb: ndarray::Array1<f64> = {
                        let mut v = ndarray::Array1::<f64>::zeros(n_obs);
                        for j in 0..beta.len() {
                            let col_j = x_std.column(start_col + j);
                            v = v + col_j.mapv(|x| x * beta[j]);
                        }
                        v
                    };
                    let r = &y_c - &xb;
                    for j in 0..beta.len() {
                        let denom = xx_diag[j];
                        if denom < 1e-12 { continue; }
                        let rho_j = r.dot(&x_std.column(start_col + j)) + denom * beta[j];
                        let new_b = soft(rho_j / denom, alpha * n_obs as f64 / denom);
                        let delta = (new_b - beta[j]).abs();
                        if delta > max_delta { max_delta = delta; }
                        beta[j] = new_b;
                    }
                    if max_delta < tol { break; }
                }
                // destandardizar: β_j_orig = β_j_std / std_j
                // intercept: intercept = y_mean - sum_j(β_j_orig * mean_j)
                let mut params = ndarray::Array1::<f64>::zeros(k);
                for j in 0..beta.len() { params[start_col + j] = beta[j] / col_std_v[start_col + j]; }
                if has_intercept {
                    params[0] = y_mean - (start_col..k).map(|j| params[j] * col_mean[j]).sum::<f64>();
                }
                let y_hat = x.dot(&params);
                let sse: f64 = y.iter().zip(y_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                let sst: f64 = { let m = y.mean().unwrap_or(0.0); y.iter().map(|v| (v - m).powi(2)).sum() };
                let r2 = 1.0 - sse / sst;
                let n_nonzero = params.iter().skip(start_col).filter(|&&v| v.abs() > 1e-10).count();
                let var_names = &gformula.independents;
                println!("\n{:=^60}", " Lasso Regression ");
                println!("  Formula: {formula_str}   α = {alpha}");
                println!("  n = {}   k = {}   R² = {r2:.4}   vars ativas: {n_nonzero}", x.nrows(), k);
                println!("\n  {:<20} {:>12}", "Variable", "Coeff");
                println!("  {}", "─".repeat(33));
                if gformula.intercept {
                    println!("  {:<20} {:>12.6}", "const", params[0]);
                    for (i, name) in var_names.iter().enumerate() { println!("  {:<20} {:>12.6}", name, params[i + 1]); }
                } else {
                    for (i, name) in var_names.iter().enumerate() { println!("  {:<20} {:>12.6}", name, params[i]); }
                }
                Ok(Value::Nil)
            }

            // elasticnet(formula, df, alpha=1.0, l1_ratio=0.5, ...)
            // Combina L1 e L2: penalty = l1_ratio*α*|β| + (1-l1_ratio)*α/2*β²
            "elasticnet" | "elastic_net" | "enet" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("elasticnet(formula, df, alpha=1.0, l1_ratio=0.5)".into())); }
                let formula = match &args[0] { Expr::Formula(f) => f.clone(), _ => return Err(HayashiError::Type("elasticnet: primeiro arg deve ser fórmula".into())) };
                let df_name = match &args[1] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("elasticnet: segundo arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let alpha    = match opt_map.get("alpha")    { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 1.0_f64 };
                let l1_ratio = match opt_map.get("l1_ratio") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 0.5_f64 };
                let tol      = match opt_map.get("tol")      { Some(Value::Float(v)) => *v, _ => 1e-6_f64 };
                let max_iter = match opt_map.get("max_iter") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 10_000usize };
                let formula_str = Self::formula_to_string(&formula);
                let gformula = GFormula::parse(&formula_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df.to_design_matrix(&gformula).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n_obs = x.nrows();
                let k = x.ncols();
                let has_intercept = gformula.intercept;
                let start_col = if has_intercept { 1 } else { 0 };
                let y_mean = y.sum() / n_obs as f64;
                let y_c: ndarray::Array1<f64> = y.mapv(|v| v - y_mean);
                let mut x_std = x.clone();
                let mut col_mean: Vec<f64> = vec![0.0; k];
                let mut col_std_v: Vec<f64> = vec![1.0; k];
                for j in start_col..k {
                    let col = x.column(j);
                    let mj = col.sum() / n_obs as f64;
                    let sj = (col.iter().map(|&v| (v - mj).powi(2)).sum::<f64>() / n_obs as f64).sqrt();
                    col_mean[j] = mj; if sj > 1e-12 { col_std_v[j] = sj; for i in 0..n_obs { x_std[[i, j]] = (x[[i, j]] - mj) / sj; } }
                }
                let l1 = alpha * l1_ratio;
                let l2 = alpha * (1.0 - l1_ratio);
                let soft = |z: f64, lam: f64| -> f64 { if z > lam { z - lam } else if z < -lam { z + lam } else { 0.0 } };
                let xx_diag: Vec<f64> = (start_col..k).map(|j| x_std.column(j).dot(&x_std.column(j))).collect();
                let mut beta = ndarray::Array1::<f64>::zeros(k - start_col);
                for _iter in 0..max_iter {
                    let mut max_delta = 0.0_f64;
                    let xb: ndarray::Array1<f64> = { let mut v = ndarray::Array1::<f64>::zeros(n_obs); for j in 0..beta.len() { let c = x_std.column(start_col + j); v = v + c.mapv(|x| x * beta[j]); } v };
                    let r = &y_c - &xb;
                    for j in 0..beta.len() {
                        let denom = xx_diag[j] + l2 * n_obs as f64;
                        if denom < 1e-12 { continue; }
                        let rho_j = r.dot(&x_std.column(start_col + j)) + xx_diag[j] * beta[j];
                        let new_b = soft(rho_j / denom, l1 * n_obs as f64 / denom);
                        let delta = (new_b - beta[j]).abs();
                        if delta > max_delta { max_delta = delta; }
                        beta[j] = new_b;
                    }
                    if max_delta < tol { break; }
                }
                let mut params = ndarray::Array1::<f64>::zeros(k);
                for j in 0..beta.len() { params[start_col + j] = beta[j] / col_std_v[start_col + j]; }
                if has_intercept { params[0] = y_mean - (start_col..k).map(|j| params[j] * col_mean[j]).sum::<f64>(); }
                let y_hat = x.dot(&params);
                let sse: f64 = y.iter().zip(y_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                let sst: f64 = { let m = y.mean().unwrap_or(0.0); y.iter().map(|v| (v - m).powi(2)).sum() };
                let r2 = 1.0 - sse / sst;
                let n_nonzero = params.iter().skip(start_col).filter(|&&v| v.abs() > 1e-10).count();
                let var_names = &gformula.independents;
                println!("\n{:=^60}", " ElasticNet Regression ");
                println!("  Formula: {formula_str}   α={alpha}   l1_ratio={l1_ratio}");
                println!("  n={}  k={}  R²={r2:.4}  vars ativas: {n_nonzero}", x.nrows(), k);
                println!("\n  {:<20} {:>12}", "Variable", "Coeff");
                println!("  {}", "─".repeat(33));
                if gformula.intercept {
                    println!("  {:<20} {:>12.6}", "const", params[0]);
                    for (i, name) in var_names.iter().enumerate() { println!("  {:<20} {:>12.6}", name, params[i + 1]); }
                } else {
                    for (i, name) in var_names.iter().enumerate() { println!("  {:<20} {:>12.6}", name, params[i]); }
                }
                Ok(Value::Nil)
            }

            // adf(df, var, lags=N)
            "adf" | "dickey_fuller" | "augmented_df" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("adf(df, var, lags=N)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("adf: primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("adf: segundo arg deve ser nome de variável".into())) };
                let series = Self::get_col_f64(&df, &var_name)?;
                let max_lags = match opt_map.get("lags") { Some(Value::Int(v)) => Some(*v as usize), Some(Value::Float(v)) => Some(*v as usize), _ => None };
                let arr = ndarray::Array1::from(series.to_vec());
                let r = greeners::TimeSeries::adf(&arr, max_lags).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Augmented Dickey-Fuller Test ");
                println!("  Variable: {var_name}   Lags used: {}", r.lags_used);
                println!("  H₀: series has a unit root (non-stationary)");
                println!("  Test statistic:  {:>10.4}", r.test_statistic);
                if let Some(p) = r.p_value { println!("  p-value:         {:>10.4}", p); }
                let (cv1, cv5, cv10) = r.critical_values;
                println!("  Critical values:  1%={cv1:.3}  5%={cv5:.3}  10%={cv10:.3}");
                println!("  Conclusion: {}", if r.is_stationary { "REJEITA H₀ — estacionária" } else { "Não rejeita H₀ — raiz unitária presente" });
                Ok(Value::Nil)
            }

            // kpss(df, var, regression=c, lags=N)
            "kpss" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("kpss(df, var, regression=c|ct, lags=N)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("kpss: primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("kpss: segundo arg deve ser nome de variável".into())) };
                let series = Self::get_col_f64(&df, &var_name)?;
                let regression = match opt_map.get("regression") { Some(Value::Str(s)) => s.clone(), _ => "c".to_string() };
                let max_lags = match opt_map.get("lags") { Some(Value::Int(v)) => Some(*v as usize), Some(Value::Float(v)) => Some(*v as usize), _ => None };
                let arr = ndarray::Array1::from(series.to_vec());
                let r = greeners::TimeSeries::kpss(&arr, &regression, max_lags).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " KPSS Test ");
                println!("  Variable: {var_name}   Regression: {}   Lags: {}", r.regression, r.lags_used);
                println!("  H₀: series is stationary");
                println!("  Test statistic:  {:>10.4}", r.test_statistic);
                let (cv10, cv5, cv25, cv1) = r.critical_values;
                println!("  Critical values:  10%={cv10:.3}  5%={cv5:.3}  2.5%={cv25:.3}  1%={cv1:.3}");
                println!("  Conclusion: {}", if r.is_stationary { "Não rejeita H₀ — estacionária" } else { "REJEITA H₀ — não estacionária" });
                Ok(Value::Nil)
            }

            // pp(df, var, lags=N)
            "pp" | "phillips_perron" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("pp(df, var, lags=N)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("pp: primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("pp: segundo arg deve ser nome de variável".into())) };
                let series = Self::get_col_f64(&df, &var_name)?;
                let max_lags = match opt_map.get("lags") { Some(Value::Int(v)) => Some(*v as usize), Some(Value::Float(v)) => Some(*v as usize), _ => None };
                let arr = ndarray::Array1::from(series.to_vec());
                let r = greeners::TimeSeries::phillips_perron(&arr, max_lags).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Phillips-Perron Test ");
                println!("  Variable: {var_name}   Lags used: {}", r.lags_used);
                println!("  H₀: series has a unit root (non-stationary)");
                println!("  Zα statistic:    {:>10.4}", r.z_alpha);
                println!("  Zt statistic:    {:>10.4}", r.z_t);
                let (cv1, cv5, cv10) = r.critical_values;
                println!("  Critical values:  1%={cv1:.3}  5%={cv5:.3}  10%={cv10:.3}");
                println!("  Conclusion: {}", if r.is_stationary { "REJEITA H₀ — estacionária" } else { "Não rejeita H₀ — raiz unitária presente" });
                Ok(Value::Nil)
            }

            // za(df, var, trim=0.15)
            "za" | "zivot_andrews" | "zivot" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("za(df, var, trim=0.15)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("za: primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("za: segundo arg deve ser nome de variável".into())) };
                let series = Self::get_col_f64(&df, &var_name)?;
                let trim = match opt_map.get("trim") { Some(Value::Float(v)) => *v, Some(Value::Int(v)) => *v as f64, _ => 0.15 };
                let arr = ndarray::Array1::from(series.to_vec());
                let r = greeners::TimeSeries::zivot_andrews(&arr, trim).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Zivot-Andrews Test ");
                println!("  Variable: {var_name}");
                println!("  H₀: unit root (sem quebra estrutural)");
                println!("  Test statistic:  {:>10.4}", r.statistic);
                println!("  Break point:     obs {}", r.break_point);
                let (cv1, cv5, cv10) = r.critical_values;
                println!("  Critical values:  1%={cv1:.3}  5%={cv5:.3}  10%={cv10:.3}");
                println!("  Conclusion: {}", if r.is_stationary { "REJEITA H₀ — estacionária com quebra" } else { "Não rejeita H₀ — raiz unitária" });
                Ok(Value::Nil)
            }

            // ── Cointegração ──────────────────────────────────────────────────

            // granger(df, y, x, lags=N)
            "granger" | "granger_causality" => {
                if args.len() < 3 { return Err(HayashiError::Runtime("granger(df, y, x, lags=N)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("granger: primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let y_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("granger: segundo arg deve ser nome de variável".into())) };
                let x_name = match &args[2] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("granger: terceiro arg deve ser nome de variável".into())) };
                let lags = match opt_map.get("lags") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 4 };
                let y_arr = ndarray::Array1::from(Self::get_col_f64(&df, &y_name)?.to_vec());
                let x_arr = ndarray::Array1::from(Self::get_col_f64(&df, &x_name)?.to_vec());
                let r = greeners::TimeSeries::granger_causality(&y_arr, &x_arr, lags).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Granger Causality Test ");
                println!("  H₀: {x_name} não causa Granger {y_name}   (lags={lags})");
                println!("  F({}, {}) = {:.4}   p = {:.4}", r.df_num, r.df_denom, r.f_statistic, r.p_value);
                println!("  Conclusion: {}", if r.p_value < 0.05 { format!("REJEITA H₀ — {x_name} causa Granger {y_name}") } else { "Não rejeita H₀".to_string() });
                Ok(Value::Nil)
            }

            // engle_granger(df, y1, y2)
            "engle_granger" | "coint" | "egtest" => {
                if args.len() < 3 { return Err(HayashiError::Runtime("engle_granger(df, y1, y2)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("engle_granger: primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let y1_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("engle_granger: segundo arg deve ser nome de variável".into())) };
                let y2_name = match &args[2] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("engle_granger: terceiro arg deve ser nome de variável".into())) };
                let y1_arr = ndarray::Array1::from(Self::get_col_f64(&df, &y1_name)?.to_vec());
                let y2_arr = ndarray::Array1::from(Self::get_col_f64(&df, &y2_name)?.to_vec());
                let r = greeners::TimeSeries::engle_granger(&y1_arr, &y2_arr).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Engle-Granger Cointegration Test ");
                println!("  Variables: {y1_name}, {y2_name}");
                println!("  H₀: sem cointegração");
                println!("  ADF statistic: {:>10.4}", r.adf_statistic);
                let (cv1, cv5, cv10) = r.critical_values;
                println!("  Critical values:  1%={cv1:.3}  5%={cv5:.3}  10%={cv10:.3}");
                let coef = &r.cointegrating_vector;
                if coef.len() >= 2 { println!("  Vetor cointegrante: [{:.4}, {:.4}]", coef[0], coef[1]); }
                println!("  Conclusion: {}", if r.is_cointegrated { "REJEITA H₀ — séries cointegradas" } else { "Não rejeita H₀ — sem cointegração" });
                Ok(Value::Nil)
            }

            // johansen(df, [var1, var2, ...], lags=N, det=0)
            "johansen" | "johansen_trace" | "vecrank" => {
                if args.len() < 2 { return Err(HayashiError::Runtime("johansen(df, [var1, var2, ...], lags=N, det=0)".into())); }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("johansen: primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_names: Vec<String> = match self.eval_expr(&args[1])? {
                    Value::List(lst) => lst.iter().map(|v| format!("{v}")).collect(),
                    _ => return Err(HayashiError::Type("johansen: segundo arg deve ser lista de variáveis".into())),
                };
                let lags = match opt_map.get("lags") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 2 };
                let det = match opt_map.get("det") { Some(Value::Int(v)) => *v as i32, Some(Value::Float(v)) => *v as i32, _ => 0i32 };
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, name) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, name)?;
                    for i in 0..n { data[[i, j]] = col[i]; }
                }
                let r = greeners::TimeSeries::johansen(&data, lags, det).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Johansen Cointegration Test ");
                println!("  Variables: {}   Lags: {lags}   Det order: {det}", var_names.join(", "));
                println!("  Cointegrating rank: {}", r.cointegrating_rank);
                println!("\n  Trace Test:");
                println!("  {:>4}  {:>12}  {:>10}  {:>10}  {:>10}", "Rank", "Trace stat", "10%", "5%", "1%");
                for i in 0..r.n_vars {
                    let stat = r.trace_stats[i];
                    let cv = r.trace_critical_values.row(i);
                    println!("  {:>4}  {:>12.4}  {:>10.4}  {:>10.4}  {:>10.4}", i, stat, cv[0], cv[1], cv[2]);
                }
                println!("\n  Max-Eigenvalue Test:");
                println!("  {:>4}  {:>12}  {:>10}  {:>10}  {:>10}", "Rank", "Max-eig", "10%", "5%", "1%");
                for i in 0..r.n_vars {
                    let stat = r.max_eigen_stats[i];
                    let cv = r.max_eigen_critical_values.row(i);
                    println!("  {:>4}  {:>12.4}  {:>10.4}  {:>10.4}  {:>10.4}", i, stat, cv[0], cv[1], cv[2]);
                }
                Ok(Value::Nil)
            }

            // ══════════════════════════════════════════════════════════════════
            // ── xtset: declara estrutura de painel ────────────────────────────
            // xtset(df, id_col, time_col)  — armazena em panel_info
            // Após xtset, fe/re/ab/etc. não precisam de id= e time=
            "xtset" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "xtset(df, id_col, time_col)  ou  xtset(df, time_col) para série temporal".into()
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("primeiro arg deve ser nome do DataFrame".into())),
                };
                match self.env.get(&df_name) {
                    Some(Value::DataFrame(_)) => {}
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é um DataFrame"))),
                };
                let id_col = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("id_col deve ser identificador ou string".into())),
                };
                let time_col = if args.len() >= 3 {
                    match &args[2] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => return Err(HayashiError::Type("time_col deve ser identificador ou string".into())),
                    }
                } else {
                    String::new()
                };
                self.panel_info.insert(df_name.clone(), (id_col.clone(), time_col.clone()));
                if time_col.is_empty() {
                    println!("xtset {df_name}  (série temporal: t={id_col})");
                } else {
                    println!("xtset {df_name}  id={id_col}  time={time_col}");
                }
                Ok(Value::Nil)
            }

            // ── Visualização ASCII — ACF / PACF / QQ-plot / heatmap ──────────

            // acfplot(df, var, lags=20, width=50, title="")
            "acfplot" | "acf_plot" | "corrgram" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("acfplot(df, var, lags=20, width=50)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let max_lag = match opt_map.get("lags") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 20 };
                let width  = match opt_map.get("width")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 50 };
                let title  = match opt_map.get("title")  { Some(Value::Str(s)) => s.clone(), _ => format!("ACF — {var_name}") };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                Self::ascii_acf(&clean, max_lag, &title, width, false);
                Ok(Value::Nil)
            }

            // pacfplot(df, var, lags=20, width=50, title="")
            "pacfplot" | "pacf_plot" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("pacfplot(df, var, lags=20, width=50)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let max_lag = match opt_map.get("lags") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 20 };
                let width  = match opt_map.get("width")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 50 };
                let title  = match opt_map.get("title")  { Some(Value::Str(s)) => s.clone(), _ => format!("PACF — {var_name}") };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                Self::ascii_acf(&clean, max_lag, &title, width, true);
                Ok(Value::Nil)
            }

            // qqplot(df, var, width=50, height=20, dist="normal", title="")
            "qqplot" | "qnorm" | "pnorm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("qqplot(df, var, width=50, height=20)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let w = match opt_map.get("width")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 50 };
                let h = match opt_map.get("height") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 20 };
                let title = match opt_map.get("title") { Some(Value::Str(s)) => s.clone(), _ => format!("QQ-plot normal — {var_name}") };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                Self::ascii_qqplot(&clean, &title, &var_name, w, h);
                Ok(Value::Nil)
            }

            // corrplot(df, var1, var2, ...) — matriz de correlação ASCII
            "corrplot" | "corr_heatmap" | "pwcorr_plot" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("corrplot(df, var1, var2, ...)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_names: Vec<String> = {
                    let mut v = Vec::new();
                    for a in &args[1..] {
                        match a {
                            Expr::Var(n) | Expr::Str(n) => v.push(n.clone()),
                            _ => return Err(HayashiError::Type("args devem ser nomes de variáveis".into())),
                        }
                    }
                    v
                };
                if var_names.len() < 2 { return Err(HayashiError::Runtime("corrplot: forneça ao menos 2 variáveis".into())); }
                let cols: Vec<Vec<f64>> = {
                    let mut v = Vec::new();
                    for n in &var_names {
                        v.push(Self::get_col_f64(&df, n)?.to_vec());
                    }
                    v
                };
                Self::ascii_corrplot(&cols, &var_names);
                Ok(Value::Nil)
            }

            // ── Visualização ASCII ────────────────────────────────────────────

            // histogram(df, var, bins=20, width=50, title="")
            "histogram" | "hist" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("histogram(df, var, bins=20, width=50)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let bins = match opt_map.get("bins") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 20 };
                let width = match opt_map.get("width") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 50 };
                let title = match opt_map.get("title") { Some(Value::Str(s)) => s.clone(), _ => format!("Histograma — {var_name}") };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                Self::ascii_histogram(&clean, bins, &title, &var_name, width);
                Ok(Value::Nil)
            }

            // scatter(df, x, y, width=60, height=20, title="")
            "scatter" | "scatterplot" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("scatter(df, x, y, width=60, height=20)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let xname = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável (x)".into())) };
                let yname = match &args[2] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("terceiro arg deve ser nome de variável (y)".into())) };
                let xs = Self::get_col_f64(&df, &xname)?;
                let ys = Self::get_col_f64(&df, &yname)?;
                let w = match opt_map.get("width")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 60 };
                let h = match opt_map.get("height") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 20 };
                let title = match opt_map.get("title") { Some(Value::Str(s)) => s.clone(), _ => format!("{yname} vs {xname}") };
                Self::ascii_scatter(&xs.to_vec(), &ys.to_vec(), &title, &xname, &yname, w, h);
                Ok(Value::Nil)
            }

            // lineplot(df, x, y, width=60, height=20, title="")
            "lineplot" | "tsplot" | "twoway" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("lineplot(df, x, y, width=60, height=20)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let xname = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável (x/tempo)".into())) };
                let yname = match &args[2] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("terceiro arg deve ser nome de variável (y)".into())) };
                let xs = Self::get_col_f64(&df, &xname)?;
                let ys = Self::get_col_f64(&df, &yname)?;
                let w = match opt_map.get("width")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 60 };
                let h = match opt_map.get("height") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 20 };
                let title = match opt_map.get("title") { Some(Value::Str(s)) => s.clone(), _ => format!("{yname} — série temporal") };
                Self::ascii_lineplot(&xs.to_vec(), &ys.to_vec(), &title, &xname, &yname, w, h);
                Ok(Value::Nil)
            }

            // boxplot(df, var, width=60, title="")
            "boxplot" | "box" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("boxplot(df, var, width=60)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let w = match opt_map.get("width") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 60 };
                let title = match opt_map.get("title") { Some(Value::Str(s)) => s.clone(), _ => format!("Boxplot — {var_name}") };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                Self::ascii_boxplot(&clean, &title, &var_name, w);
                Ok(Value::Nil)
            }

            // kdensity(df, var, width=60, height=20) — KDE via ascii_scatter dos pontos da densidade
            "kdensity" | "density" | "densityplot" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("kdensity(df, var, width=60, height=20)".into()));
                }
                let df_name = match &args[0] { Expr::Var(n) => n.clone(), _ => return Err(HayashiError::Type("primeiro arg deve ser DataFrame".into())) };
                let df = match self.env.get(&df_name) { Some(Value::DataFrame(d)) => d.clone(), _ => return Err(HayashiError::Runtime(format!("'{df_name}' não é DataFrame"))) };
                let var_name = match &args[1] { Expr::Var(n) | Expr::Str(n) => n.clone(), _ => return Err(HayashiError::Type("segundo arg deve ser nome de variável".into())) };
                let data = Self::get_col_f64(&df, &var_name)?;
                let w = match opt_map.get("width")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 60 };
                let h = match opt_map.get("height") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 20 };
                // bw= option ou Silverman
                let bw_opt = match opt_map.get("bw") { Some(Value::Float(v)) => Some(*v), Some(Value::Int(v)) => Some(*v as f64), _ => None };
                let kernel = match opt_map.get("kernel") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "epanechnikov" | "epa" => greeners::Kernel::Epanechnikov,
                        "triangular"           => greeners::Kernel::Triangular,
                        "uniform"              => greeners::Kernel::Uniform,
                        _                      => greeners::Kernel::Gaussian,
                    },
                    _ => greeners::Kernel::Gaussian,
                };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                let n = clean.len();
                if n < 4 { return Err(HayashiError::Runtime("kdensity: poucos dados".into())); }
                // Estimar KDE — usa support/density já calculados no fit (512 pontos)
                let result = greeners::KDEUnivariate::fit(&ndarray::Array1::from(clean.clone()), bw_opt, kernel)
                    .map_err(|e| HayashiError::Runtime(format!("kdensity: {e}")))?;
                let xs: Vec<f64> = result.support.to_vec();
                let ys: Vec<f64> = result.density.to_vec();
                let title = match opt_map.get("title") { Some(Value::Str(s)) => s.clone(), _ => format!("KDE — {var_name}  (bw={:.4})", result.bandwidth) };
                Self::ascii_lineplot(&xs, &ys, &title, &var_name, "densidade", w, h);
                Ok(Value::Nil)
            }

            // residplot(model, width=60, height=20) — resíduos vs ŷ
            "residplot" | "rvfplot" | "resid_plot" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("residplot(model, width=60, height=20)".into()));
                }
                let (fitted, resids, mname) = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => {
                        let yhat = m.x.dot(&m.result.params).to_vec();
                        (yhat, m.residuals.to_vec(), "OLS".to_string())
                    }
                    _ => return Err(HayashiError::Type("residplot() suporta apenas modelos OLS; para GLM use predict + scatter".into())),
                };
                let w = match opt_map.get("width")  { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 60 };
                let h = match opt_map.get("height") { Some(Value::Int(v)) => *v as usize, Some(Value::Float(v)) => *v as usize, _ => 20 };
                let title = format!("Resíduos vs Ŷ — {mname}");
                Self::ascii_scatter(&fitted, &resids, &title, "ŷ (fitted)", "e (resíduo)", w, h);
                Ok(Value::Nil)
            }

            // ── Função definida pelo usuário ──────────────────────────────────
            other => {
                // Recupera a função do env (se existir)
                let user_fn = match self.env.get(other).cloned() {
                    Some(Value::UserFn(f)) => f,
                    _ => return Err(HayashiError::Runtime(
                        format!("função '{other}' não encontrada")
                    )),
                };

                if args.len() != user_fn.params.len() {
                    return Err(HayashiError::Runtime(format!(
                        "fn '{other}': esperado {} argumento(s), recebido {}",
                        user_fn.params.len(), args.len()
                    )));
                }

                // Avalia argumentos antes de modificar o env
                let arg_vals: Vec<Value> = args.iter()
                    .map(|e| self.eval_expr(e))
                    .collect::<Result<_>>()?;

                // Salva env atual e cria escopo local
                let saved_env = self.env.vars.clone();

                for (param, val) in user_fn.params.iter().zip(arg_vals) {
                    self.env.set(param, val);
                }

                // Executa corpo — captura Return
                let body = user_fn.body.clone();
                let mut exec_err: Option<HayashiError> = None;
                for s in &body {
                    match self.exec(s) {
                        Ok(()) => {}
                        Err(HayashiError::Return) => break,
                        Err(HayashiError::Break | HayashiError::Continue) => {
                            exec_err = Some(HayashiError::Runtime(
                                "break/continue fora de um loop".into()
                            ));
                            break;
                        }
                        Err(e) => { exec_err = Some(e); break; }
                    }
                }

                // Restaura env do escopo externo
                self.env.vars = saved_env;

                if let Some(e) = exec_err {
                    return Err(e);
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
            Value::Bool(b)  => *b,
            Value::Int(i)   => *i != 0,
            Value::Float(f) => *f != 0.0 && !f.is_nan(),
            Value::Nil      => false,
            _               => true,
        }
    }

    fn value_as_f64(v: &Value) -> Result<f64> {
        match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i)   => Ok(*i as f64),
            Value::Bool(b)  => Ok(if *b { 1.0 } else { 0.0 }),
            _ => Err(HayashiError::Type("esperado valor numérico".into())),
        }
    }

    fn eval_scalar_binop(op: &BinOp, l: Value, r: Value) -> Result<Value> {
        // Comparações (funciona com qualquer tipo comparável)
        match op {
            BinOp::Eq => {
                let eq = match (&l, &r) {
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
                BinOp::Add  => Ok(Value::Int(a + b)),
                BinOp::Sub  => Ok(Value::Int(a - b)),
                BinOp::Mul  => Ok(Value::Int(a * b)),
                BinOp::Div  => Ok(Value::Float(*a as f64 / *b as f64)),
                BinOp::Pow  => Ok(Value::Float((*a as f64).powf(*b as f64))),
                BinOp::Gt   => Ok(Value::Bool(a > b)),
                BinOp::Lt   => Ok(Value::Bool(a < b)),
                BinOp::GtEq => Ok(Value::Bool(a >= b)),
                BinOp::LtEq => Ok(Value::Bool(a <= b)),
                BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Ne => unreachable!(),
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
                    BinOp::Add  => Ok(Value::Float(a + b)),
                    BinOp::Sub  => Ok(Value::Float(a - b)),
                    BinOp::Mul  => Ok(Value::Float(a * b)),
                    BinOp::Div  => Ok(Value::Float(a / b)),
                    BinOp::Pow  => Ok(Value::Float(a.powf(b))),
                    BinOp::Gt   => Ok(Value::Bool(a > b)),
                    BinOp::Lt   => Ok(Value::Bool(a < b)),
                    BinOp::GtEq => Ok(Value::Bool(a >= b)),
                    BinOp::LtEq => Ok(Value::Bool(a <= b)),
                    BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Ne => unreachable!(),
                }
            }
        }
    }

    fn extract_panel_args(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<(Formula, DataFrame, String, String)> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "panel estimator requires (formula, dataframe [, id=col])".into(),
            ));
        }
        let formula_ast = match &args[0] {
            Expr::Formula(f) => f.clone(),
            _ => return Err(HayashiError::Type("first argument must be a formula".into())),
        };
        let df_name = match &args[1] {
            Expr::Var(name) => name.clone(),
            _ => return Err(HayashiError::Type("second argument must be a DataFrame variable".into())),
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(HayashiError::Runtime(format!("'{df_name}' is not a DataFrame"))),
        };
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => self.panel_info.get(&df_name)
                .map(|(id, _)| id.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| HayashiError::Runtime(
                    format!("panel estimator requires id=col or xtset({df_name}, id, time) first")
                ))?,
        };
        Ok((formula_ast, df, df_name, id_col))
    }

    fn get_time_col(&self, df_name: &str, opt_map: &HashMap<String, Value>) -> Result<String> {
        match opt_map.get("time") {
            Some(Value::Str(s)) => Ok(s.clone()),
            _ => self.panel_info.get(df_name)
                .map(|(_, t)| t.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| HayashiError::Runtime(
                    format!("panel estimator requires time=col or xtset({df_name}, id, time) first")
                )),
        }
    }

    /// Extrai uma coluna como Vec<i64> — aceita colunas Int ou Float.
    fn col_as_i64(df: &DataFrame, col: &str) -> std::result::Result<Vec<i64>, greeners::GreenersError> {
        if let Ok(ids) = df.get_int(col) {
            Ok(ids.to_vec())
        } else if let Ok(floats) = df.get(col) {
            Ok(floats.iter().map(|&v| v as i64).collect())
        } else {
            Err(greeners::GreenersError::VariableNotFound(col.to_string()))
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
                    return Err(HayashiError::Runtime(format!("coluna '{col}' não encontrada")));
                }
            }
        }

        let mut builder = DataFrame::builder();
        for name in &df.column_names() {
            match df.get_column(name) {
                Ok(Column::Float(arr)) => {
                    builder = builder.add_column(name, idx.iter().map(|&i| arr[i]).collect::<Vec<_>>());
                }
                Ok(Column::Int(arr)) => {
                    builder = builder.add_column(name, idx.iter().map(|&i| arr[i] as f64).collect::<Vec<_>>());
                }
                _ => {
                    if let Ok(arr) = df.get_string(name) {
                        let v = arr.to_vec();
                        builder = builder.add_string(name, idx.iter().map(|&i| v[i].clone()).collect());
                    }
                }
            }
        }
        builder.build().map_err(|e| HayashiError::Runtime(e.to_string()))
    }

    fn coef_names_from_formula(formula_ast: &Formula, df: &DataFrame, n_cols: usize) -> Vec<String> {
        let mut names: Vec<String> = vec!["_cons".into()];
        for term in &formula_ast.rhs {
            match term {
                RhsTerm::Var(v)              => names.push(v.clone()),
                RhsTerm::Transform(fn_, v)   => names.push(format!("{fn_}({v})")),
                RhsTerm::Interaction(a, b)   => names.push(format!("{a}:{b}")),
                RhsTerm::Categorical(v) => {
                    let raw = Self::col_to_strings(df, v).unwrap_or_default();
                    let mut unique: Vec<String> = raw.into_iter()
                        .collect::<std::collections::HashSet<_>>().into_iter().collect();
                    if unique.iter().all(|s| s.parse::<f64>().is_ok()) {
                        unique.sort_by(|a, b| a.parse::<f64>().unwrap()
                            .partial_cmp(&b.parse::<f64>().unwrap()).unwrap());
                    } else { unique.sort(); }
                    for val in unique.into_iter().skip(1) {
                        names.push(format!("{v}={val}"));
                    }
                }
            }
        }
        names.truncate(n_cols);
        while names.len() < n_cols { names.push(format!("x{}", names.len() + 1)); }
        names
    }

    // ── Extrai coluna como Vec<String> (para tabulate) ────────────────────────

    fn col_to_strings(df: &DataFrame, name: &str) -> Result<Vec<String>> {
        use greeners::Column;
        match df.get_column(name) {
            Ok(Column::Int(arr)) => Ok(arr.iter().map(|x| x.to_string()).collect()),
            Ok(Column::Float(arr)) => Ok(arr.iter().map(|x| {
                if x.is_nan() { ".".to_string() }
                else if x.fract() == 0.0 && x.abs() < 1e14 { format!("{}", *x as i64) }
                else { format!("{:.4}", x) }
            }).collect()),
            _ => df.get_string(name)
                    .map(|arr| arr.to_vec())
                    .map_err(|_| HayashiError::Runtime(
                        format!("column '{name}' not found or has unsupported type for tabulate")
                    )),
        }
    }

    // ── Tabela de frequências (uni-variada) ───────────────────────────────────

    fn tabulate_one(df: &DataFrame, var: &str) -> Result<()> {
        let vals = Self::col_to_strings(df, var)?;
        let n = vals.len();

        let mut counts: HashMap<String, usize> = HashMap::new();
        for v in &vals { *counts.entry(v.clone()).or_insert(0) += 1; }

        let mut unique: Vec<String> = counts.keys().cloned().collect();
        let all_num = unique.iter().all(|s| s.parse::<f64>().is_ok());
        if all_num {
            unique.sort_by(|a, b| {
                a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap()
            });
        } else {
            unique.sort();
        }

        let label_w = var.len().max(12).max(unique.iter().map(|s| s.len()).max().unwrap_or(0)) + 2;
        let sep = format!("{}-+{}", "-".repeat(label_w), "-".repeat(36));

        println!("\n{:>lw$} | {:>10}  {:>10}  {:>10}", var, "Freq.", "Percent", "Cum.",
                 lw = label_w);
        println!("{sep}");

        let mut cum = 0.0_f64;
        for key in &unique {
            let freq = counts[key];
            let pct  = freq as f64 / n as f64 * 100.0;
            cum += pct;
            println!("{:>lw$} | {:>10}  {:>10.2}  {:>10.2}", key, freq, pct, cum,
                     lw = label_w);
        }
        println!("{sep}");
        println!("{:>lw$} | {:>10}  {:>10.2}", "Total", n, 100.0_f64, lw = label_w);
        println!();
        Ok(())
    }

    // ── Tabela cruzada (bi-variada, opcional chi2) ────────────────────────────

    fn tabulate_two(df: &DataFrame, row_var: &str, col_var: &str, do_chi2: bool) -> Result<()> {
        let rows = Self::col_to_strings(df, row_var)?;
        let cols = Self::col_to_strings(df, col_var)?;

        if rows.len() != cols.len() {
            return Err(HayashiError::Runtime("columns have different lengths".into()));
        }

        // valores únicos, ordenados
        let sort_strs = |mut v: Vec<String>| -> Vec<String> {
            let all_num = v.iter().all(|s| s.parse::<f64>().is_ok());
            if all_num {
                v.sort_by(|a, b| a.parse::<f64>().unwrap()
                    .partial_cmp(&b.parse::<f64>().unwrap()).unwrap());
            } else {
                v.sort();
            }
            v
        };

        let mut row_set: Vec<String> = rows.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
        row_set = sort_strs(row_set);
        let mut col_set: Vec<String> = cols.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
        col_set = sort_strs(col_set);

        // contagens
        let mut cell: HashMap<(String, String), usize> = HashMap::new();
        for (r, c) in rows.iter().zip(cols.iter()) {
            *cell.entry((r.clone(), c.clone())).or_insert(0) += 1;
        }

        let n = rows.len();
        let col_totals: Vec<usize> = col_set.iter()
            .map(|c| row_set.iter().map(|r| *cell.get(&(r.clone(), c.clone())).unwrap_or(&0)).sum())
            .collect();
        let row_totals: Vec<usize> = row_set.iter()
            .map(|r| col_set.iter().map(|c| *cell.get(&(r.clone(), c.clone())).unwrap_or(&0)).sum())
            .collect();

        // larguras de coluna
        let cell_w   = 10usize;
        let row_lw   = row_var.len().max(12).max(row_set.iter().map(|s| s.len()).max().unwrap_or(0)) + 2;
        let col_head_w = col_set.len() * (cell_w + 1) + 1;
        let total_w  = cell_w + 2;

        // linha de cabeçalho do col_var
        println!("\n{:>rw$} | {:^chw$}| {:>tw$}",
                 "", col_var, "Total",
                 rw = row_lw, chw = col_head_w, tw = total_w);

        // linha com os valores das colunas
        print!("{:>rw$} |", row_var, rw = row_lw);
        for cv in &col_set { print!(" {:>cw$}", cv, cw = cell_w); }
        println!(" | {:>cw$}", "Total", cw = cell_w);

        let sep = format!("{}-+{}-+{}", "-".repeat(row_lw), "-".repeat(col_head_w), "-".repeat(total_w));
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
        for ct in &col_totals { print!(" {:>cw$}", ct, cw = cell_w); }
        println!(" | {:>cw$}", n, cw = cell_w);
        println!();

        if do_chi2 {
            let mut stat = 0.0_f64;
            for (i, rv) in row_set.iter().enumerate() {
                for (j, cv) in col_set.iter().enumerate() {
                    let obs = *cell.get(&(rv.clone(), cv.clone())).unwrap_or(&0) as f64;
                    let exp = row_totals[i] as f64 * col_totals[j] as f64 / n as f64;
                    if exp > 0.0 { stat += (obs - exp).powi(2) / exp; }
                }
            }
            let df = (row_set.len() - 1) * (col_set.len() - 1);
            let p   = chi2_pvalue(stat, df as f64);
            println!("  Pearson chi2({df}) = {stat:.4}   Pr = {p:.4}");
            println!();
        }

        Ok(())
    }

    // ── Helpers de visualização ASCII ────────────────────────────────────────

    fn ascii_histogram(data: &[f64], bins: usize, title: &str, var: &str, width: usize) {
        if data.is_empty() { println!("  (sem dados)"); return; }
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if (max - min).abs() < 1e-15 { println!("  (variância zero)"); return; }
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
            let bar_len = if max_count > 0 { cnt * bar_w / max_count } else { 0 };
            let bar: String = "█".repeat(bar_len);
            println!("  [{:>10.4},{:>10.4})  {:>5}  {:<width$}", lo, hi, cnt, bar, width = bar_w);
        }
        println!("{:-^width$}", "", width = bar_w + 34);
        println!();
    }

    fn ascii_scatter(xs: &[f64], ys: &[f64], title: &str, xlab: &str, ylab: &str, w: usize, h: usize) {
        if xs.is_empty() { println!("  (sem dados)"); return; }
        let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let xrng = (xmax - xmin).max(1e-15);
        let yrng = (ymax - ymin).max(1e-15);
        let mut grid = vec![vec![' '; w]; h];
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            if x.is_nan() || y.is_nan() { continue; }
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
        println!("              {:<10.4}{:^width$.4}{:>10.4}", xmin, mid_x, xmax, width = w - 20);
        println!("              {:^width$}", xlab, width = w);
        println!("  n={}", xs.len());
        println!();
    }

    fn ascii_lineplot(xs: &[f64], ys: &[f64], title: &str, xlab: &str, ylab: &str, w: usize, h: usize) {
        if xs.is_empty() { println!("  (sem dados)"); return; }
        let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let xrng = (xmax - xmin).max(1e-15);
        let yrng = (ymax - ymin).max(1e-15);
        // Sort by x
        let mut pairs: Vec<(f64, f64)> = xs.iter().zip(ys.iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y)).collect();
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
                        if grid[r][c] == ' ' { grid[r][c] = '─'; }
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
        println!("              {:<10.4}{:^width$.4}{:>10.4}", xmin, mid_x, xmax, width = w - 20);
        println!("              {:^width$}", xlab, width = w);
        println!("  n={}", pairs.len());
        println!();
    }

    fn ascii_boxplot(data: &[f64], title: &str, var: &str, w: usize) {
        if data.is_empty() { println!("  (sem dados)"); return; }
        let mut sorted = data.to_vec();
        sorted.retain(|v| !v.is_nan());
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n < 4 { println!("  (poucos dados para boxplot)"); return; }
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
        let whisker_lo = sorted.iter().cloned().filter(|&v| v >= fence_lo).fold(f64::INFINITY, f64::min);
        let whisker_hi = sorted.iter().cloned().filter(|&v| v <= fence_hi).fold(f64::NEG_INFINITY, f64::max);
        let outliers: Vec<f64> = sorted.iter().cloned().filter(|&v| v < fence_lo || v > fence_hi).collect();

        let rng = (mx - mn).max(1e-15);
        let to_col = |v: f64| -> usize { (((v - mn) / rng * (w - 1) as f64).round() as usize).min(w - 1) };
        let c_wlo = to_col(whisker_lo);
        let c_q1  = to_col(q1);
        let c_med = to_col(med);
        let c_q3  = to_col(q3);
        let c_whi = to_col(whisker_hi);

        // Build boxplot line
        let mut line = vec![' '; w];
        for c in c_wlo..=c_whi { line[c] = '─'; }
        for c in c_q1..=c_q3 { line[c] = '█'; }
        line[c_wlo] = '├';
        line[c_whi] = '┤';
        line[c_q1]  = '▐';
        line[c_q3]  = '▌';
        line[c_med] = '|';
        for &v in &outliers { let c = to_col(v); line[c] = '○'; }

        println!();
        println!("{:=^width$}", format!(" {title} "), width = w + 18);
        println!("  Variável: {var}   n={n}");
        println!();
        println!("             {}", line.iter().collect::<String>());
        println!();
        println!("  Min:    {:>12.4}   Q1:  {:>12.4}   Mediana: {:>12.4}", whisker_lo, q1, med);
        println!("  Média:  {:>12.4}   Q3:  {:>12.4}   Max:     {:>12.4}", mean, q3, whisker_hi);
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
        let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        let phi = 1.0 - greeners::norm_pdf(x) * poly;
        if x >= 0.0 { phi } else { 1.0 - phi }
    }

    // ── ACF / PACF como barras ASCII ─────────────────────────────────────────
    fn ascii_acf(data: &[f64], max_lag: usize, title: &str, width: usize, partial: bool) {
        let n = data.len();
        if n < 4 { println!("(dados insuficientes para ACF)"); return; }
        let mean = data.iter().sum::<f64>() / n as f64;
        let var  = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        if var < 1e-15 { println!("(variância zero)"); return; }

        // Calcula autocorrelações completas
        let max_lag = max_lag.min(n / 2);
        let acf: Vec<f64> = (0..=max_lag).map(|k| {
            let s: f64 = (0..n - k).map(|i| (data[i] - mean) * (data[i + k] - mean)).sum();
            s / (n as f64 * var)
        }).collect();

        // PACF via algoritmo de Yule-Walker (Durbin-Levinson)
        let values: Vec<f64> = if partial {
            let mut pacf = vec![0.0f64; max_lag + 1];
            pacf[0] = 1.0;
            if max_lag >= 1 { pacf[1] = acf[1]; }
            let mut phi: Vec<Vec<f64>> = vec![vec![0.0; max_lag + 1]; max_lag + 1];
            phi[1][1] = acf[1];
            for k in 2..=max_lag {
                let num: f64 = acf[k] - (1..k).map(|j| phi[k-1][j] * acf[k-j]).sum::<f64>();
                let den: f64 = 1.0  - (1..k).map(|j| phi[k-1][j] * acf[j]).sum::<f64>();
                let phi_kk = if den.abs() < 1e-15 { 0.0 } else { num / den };
                phi[k][k] = phi_kk;
                for j in 1..k { phi[k][j] = phi[k-1][j] - phi_kk * phi[k-1][k-j]; }
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
        if n < 4 { println!("(dados insuficientes para QQ-plot)"); return; }
        // Quantis teóricos normais por aproximação de Blom: p_i = (i - 3/8) / (n + 1/4)
        let theoretical: Vec<f64> = (1..=n).map(|i| {
            let p = (i as f64 - 0.375) / (n as f64 + 0.25);
            // Aproximação de Peter Acklam para invnorm (erro < 3.5e-4)
            let q = p - 0.5;
            let r = if q.abs() <= 0.425 {
                let a = [3.3871328_f64, 133.14166789, 1971.5909503, 13731.693765, 45921.953931,
                         67265.770927, 33430.575583, 2509.0809287];
                let b = [1.0_f64, 42.313330701, 687.18700749, 5394.1960214, 21213.794301,
                         39307.895800, 28729.085735, 5226.4952788];
                let q2 = q * q;
                let num = a.iter().enumerate().fold(0.0, |s,(i,&c)| s + c * q2.powi(i as i32));
                let den = b.iter().enumerate().fold(0.0, |s,(i,&c)| s + c * q2.powi(i as i32));
                q * num / den
            } else {
                let pp = if q < 0.0 { p } else { 1.0 - p };
                let r = (-pp.ln()).sqrt();
                let c = if r <= 5.0 {
                    [1.42343711_f64, 4.63033784, 5.76082150, 1.42343711, 1.63155402, 0.07027109]
                } else {
                    [6.65790464_f64, 5.46378491, 1.78482653, 0.05697114, 0.18127138, 0.00778070]
                };
                let num = c[0] + r * (c[1] + r * c[2]);
                let den = 1.0  + r * (c[3] + r * (c[4] + r * c[5]));
                if q < 0.0 { -(num / den) } else { num / den }
            };
            r
        }).collect();
        let mean_s = sorted.iter().sum::<f64>() / n as f64;
        let std_s  = (sorted.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / n as f64).sqrt().max(1e-15);
        // Standarizar os quantis empíricos
        let empirical: Vec<f64> = sorted.iter().map(|x| (x - mean_s) / std_s).collect();
        println!("\n{:=<w$}", "");
        println!(" {title}  (normalizado)");
        println!("{:=<w$}", "");
        Self::ascii_scatter(&theoretical, &empirical, title, "quantil teórico", var, w, h);
        // Linha de referência (y = x): já visível no scatter se os dados são normais
        println!("  (linha ideal: pontos ao longo da diagonal)");
    }

    // ── Matriz de correlação como heatmap de texto ────────────────────────────
    fn ascii_corrplot(cols: &[Vec<f64>], names: &[String]) {
        let k = cols.len();
        let n = cols[0].len();
        let means: Vec<f64> = cols.iter().map(|c| c.iter().sum::<f64>() / n as f64).collect();
        // Calcula correlações
        let mut corr = vec![vec![0.0f64; k]; k];
        for i in 0..k {
            for j in 0..k {
                let xi: Vec<f64> = cols[i].iter().map(|x| x - means[i]).collect();
                let xj: Vec<f64> = cols[j].iter().map(|x| x - means[j]).collect();
                let num: f64 = xi.iter().zip(&xj).map(|(a, b)| a * b).sum();
                let di: f64  = xi.iter().map(|a| a * a).sum::<f64>().sqrt();
                let dj: f64  = xj.iter().map(|b| b * b).sum::<f64>().sqrt();
                corr[i][j] = if di * dj < 1e-15 { 0.0 } else { num / (di * dj) };
            }
        }
        // Largura do nome
        let nw = names.iter().map(|n| n.len()).max().unwrap_or(4).max(4);
        // Cabeçalho
        println!("\n{:=<80}", "");
        println!(" Matriz de Correlação");
        println!("{:=<80}", "");
        print!("{:>nw$}", "");
        for n in names { print!(" {:>7}", &n[..n.len().min(7)]); }
        println!();
        // Linhas
        for i in 0..k {
            print!("{:>nw$}", &names[i][..names[i].len().min(nw)]);
            for j in 0..k {
                let v = corr[i][j];
                // Representação por blocos: ████ para |r|=1, ░░░░ para r≈0
                let shade = if v.abs() >= 0.9 { "████" }
                    else if v.abs() >= 0.7 { "▓▓▓▓" }
                    else if v.abs() >= 0.5 { "▒▒▒▒" }
                    else if v.abs() >= 0.3 { "░░░░" }
                    else { "    " };
                let sign = if v < 0.0 { "-" } else { "+" };
                print!(" {sign}{shade}", );
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

    fn extract_binary_args(&mut self, args: &[Expr]) -> Result<(Formula, DataFrame)> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime("estimator requires (formula, dataframe)".into()));
        }
        let formula_ast = match &args[0] {
            Expr::Formula(f) => f.clone(),
            _ => return Err(HayashiError::Type("first argument must be a formula".into())),
        };
        let df_name = match &args[1] {
            Expr::Var(name) => name.clone(),
            _ => return Err(HayashiError::Type("second argument must be a DataFrame variable".into())),
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(HayashiError::Runtime(format!("'{df_name}' is not a DataFrame"))),
        };
        Ok((formula_ast, df))
    }

    // ── Métodos de objetos ────────────────────────────────────────────────────

    fn eval_field(&mut self, obj: &Expr, field: &str, _args: &[Expr], _opts: &[Opt]) -> Result<Value> {
        let val = self.eval_expr(obj)?;
        match (&val, field) {
            (Value::OlsResult(m), "summary")    => { println!("{}", m.result); Ok(Value::Nil) }
            (Value::IvResult(r), "summary")     => { println!("{r}"); Ok(Value::Nil) }
            (Value::BinaryResult(m), "summary") => { println!("{m}"); Ok(Value::Nil) }
            (Value::PanelResult(r), "summary")  => { println!("{r}"); Ok(Value::Nil) }
            (Value::ReResult(r), "summary")     => { println!("{r}"); Ok(Value::Nil) }
            (_, f) => Err(HayashiError::Runtime(format!("unknown method '{f}'"))),
        }
    }

    // ── Avalia expressão elemento-a-elemento sobre colunas de um DataFrame ───

    fn eval_col_expr(expr: &Expr, df: &DataFrame) -> Result<Vec<f64>> {
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
            Expr::Var(name) => {
                use greeners::Column;
                let col = df.get_column(name)
                    .map_err(|_| HayashiError::Runtime(format!("column '{name}' not found")))?;
                match col {
                    Column::Float(arr) => Ok(arr.to_vec()),
                    Column::Int(arr)   => Ok(arr.iter().map(|&x| x as f64).collect()),
                    _ => Err(HayashiError::Type(format!("column '{name}' is not numeric"))),
                }
            }
            Expr::Neg(inner) => {
                let vals = Self::eval_col_expr(inner, df)?;
                Ok(vals.into_iter().map(|x| -x).collect())
            }
            Expr::Not(inner) => {
                let vals = Self::eval_col_expr(inner, df)?;
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
                let l = Self::eval_col_expr(lhs, df)?;
                let r = Self::eval_col_expr(rhs, df)?;
                if l.len() != r.len() {
                    return Err(HayashiError::Runtime("mismatched column lengths".into()));
                }
                Ok(l.into_iter().zip(r).map(|(a, b)| match op {
                    BinOp::Add  => a + b,
                    BinOp::Sub  => a - b,
                    BinOp::Mul  => a * b,
                    BinOp::Div  => a / b,
                    BinOp::Pow  => a.powf(b),
                    BinOp::Gt   => if a > b { 1.0 } else { 0.0 },
                    BinOp::Lt   => if a < b { 1.0 } else { 0.0 },
                    BinOp::GtEq => if a >= b { 1.0 } else { 0.0 },
                    BinOp::LtEq => if a <= b { 1.0 } else { 0.0 },
                    BinOp::Eq   => if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 },
                    BinOp::Ne   => if (a - b).abs() >= f64::EPSILON { 1.0 } else { 0.0 },
                    BinOp::And  => if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 },
                    BinOp::Or   => if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 },
                }).collect())
            }
            Expr::Call { func, args, .. } => {
                // funções escalares aplicadas elemento-a-elemento
                // ── funções multi-coluna (rowmean / rowsum / rowmin / rowmax) ──
                if matches!(func.as_str(), "rowmean" | "rowsum" | "rowmin" | "rowmax") {
                    if args.is_empty() {
                        return Err(HayashiError::Runtime(
                            format!("{func}() requer ao menos uma coluna")
                        ));
                    }
                    let cols: Vec<Vec<f64>> = args.iter()
                        .map(|a| Self::eval_col_expr(a, df))
                        .collect::<Result<_>>()?;
                    let n = df.n_rows();
                    return Ok((0..n).map(|i| {
                        let row: Vec<f64> = cols.iter().map(|c| c[i]).collect();
                        if row.iter().any(|v| !v.is_finite()) { return f64::NAN; }
                        match func.as_str() {
                            "rowmean" => row.iter().sum::<f64>() / row.len() as f64,
                            "rowsum"  => row.iter().sum::<f64>(),
                            "rowmin"  => row.iter().cloned().fold(f64::INFINITY,     f64::min),
                            "rowmax"  => row.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                            _ => f64::NAN,
                        }
                    }).collect());
                }

                if args.len() == 1 {
                    // ── funções que precisam de toda a coluna ──────────────────
                    match func.as_str() {
                        // rank com média para empates (ascendente; NaN vão ao fim)
                        "rank" => {
                            let vals = Self::eval_col_expr(&args[0], df)?;
                            let n = vals.len();
                            let mut order: Vec<usize> = (0..n).collect();
                            order.sort_by(|&a, &b| {
                                match (vals[a].is_nan(), vals[b].is_nan()) {
                                    (true, true)   => std::cmp::Ordering::Equal,
                                    (true, false)  => std::cmp::Ordering::Greater,
                                    (false, true)  => std::cmp::Ordering::Less,
                                    (false, false) => vals[a].partial_cmp(&vals[b]).unwrap(),
                                }
                            });
                            let mut ranks = vec![0.0f64; n];
                            let mut i = 0;
                            while i < n {
                                if vals[order[i]].is_nan() {
                                    for k in i..n { ranks[order[k]] = f64::NAN; }
                                    break;
                                }
                                let mut j = i;
                                while j < n
                                    && !vals[order[j]].is_nan()
                                    && (vals[order[j]] - vals[order[i]]).abs() < 1e-10
                                {
                                    j += 1;
                                }
                                let avg = ((i + 1) as f64 + j as f64) / 2.0;
                                for k in i..j { ranks[order[k]] = avg; }
                                i = j;
                            }
                            return Ok(ranks);
                        }
                        // cumsum: soma cumulativa
                        "cumsum" => {
                            let vals = Self::eval_col_expr(&args[0], df)?;
                            let mut s = 0.0f64;
                            return Ok(vals.into_iter().map(|v| { s += v; s }).collect());
                        }
                        // group: ID inteiro (1-based) para cada valor único
                        "group" => {
                            let col_name = match &args[0] {
                                Expr::Var(name) => name.clone(),
                                _ => return Err(HayashiError::Runtime(
                                    "group() requer o nome de uma coluna".into()
                                )),
                            };
                            let strs = Self::col_to_strings(df, &col_name)?;
                            let mut unique: Vec<String> = strs.iter().cloned()
                                .collect::<std::collections::HashSet<_>>()
                                .into_iter()
                                .collect();
                            if unique.iter().all(|s| s.parse::<f64>().is_ok()) {
                                unique.sort_by(|a, b| {
                                    a.parse::<f64>().unwrap()
                                        .partial_cmp(&b.parse::<f64>().unwrap())
                                        .unwrap()
                                });
                            } else {
                                unique.sort();
                            }
                            let lookup: HashMap<String, f64> = unique.into_iter()
                                .enumerate()
                                .map(|(i, v)| (v, (i + 1) as f64))
                                .collect();
                            return Ok(strs.iter()
                                .map(|v| *lookup.get(v).unwrap_or(&f64::NAN))
                                .collect());
                        }
                        _ => {}
                    }

                    // ── funções escalares elemento-a-elemento ──────────────────
                    let vals = Self::eval_col_expr(&args[0], df)?;
                    let f: fn(f64) -> f64 = match func.as_str() {
                        "log"  | "ln"   => f64::ln,
                        "log2"          => f64::log2,
                        "log10"         => f64::log10,
                        "exp"           => f64::exp,
                        "sqrt"          => f64::sqrt,
                        "abs"           => f64::abs,
                        "floor"         => f64::floor,
                        "ceil"          => f64::ceil,
                        "round"         => f64::round,
                        "sin"           => f64::sin,
                        "cos"           => f64::cos,
                        other => return Err(HayashiError::Runtime(
                            format!("função de coluna desconhecida '{other}'")
                        )),
                    };
                    Ok(vals.into_iter().map(f).collect())
                } else if args.len() == 2 && func == "pow" {
                    let base = Self::eval_col_expr(&args[0], df)?;
                    let exp  = Self::eval_col_expr(&args[1], df)?;
                    Ok(base.into_iter().zip(exp).map(|(a, b)| a.powf(b)).collect())
                } else {
                    Err(HayashiError::Runtime(format!(
                        "função '{func}' não suportada em generate"
                    )))
                }
            }
            // ── operadores de série temporal ─────────────────────────────────
            // Requerem que o df já esteja ordenado por tsset.
            // L.x = x[i-n], F.x = x[i+n], D.x = x[i] - x[i-n]
            Expr::TsOp { op, var, n } => {
                use greeners::Column;
                let col = df.get_column(var)
                    .map_err(|_| HayashiError::Runtime(format!("coluna '{var}' não encontrada")))?;
                let vals: Vec<f64> = match col {
                    Column::Float(arr) => arr.to_vec(),
                    Column::Int(arr)   => arr.iter().map(|&x| x as f64).collect(),
                    _ => return Err(HayashiError::Type(format!("coluna '{var}' não é numérica"))),
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

            _ => Err(HayashiError::Runtime(
                "expression type not supported in generate".into()
            )),
        }
    }

    // ── Executa statement ─────────────────────────────────────────────────────

    pub fn exec(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { name, value } => {
                let val = self.eval_expr(value)?;
                self.env.set(name, val);
            }

            // ── input df \n headers \n rows \n end ───────────────────────────
            Stmt::Input { alias, headers, rows } => {
                if headers.is_empty() {
                    return Err(HayashiError::Runtime("input: nenhuma variável no cabeçalho".into()));
                }
                if rows.is_empty() {
                    return Err(HayashiError::Runtime("input: nenhuma linha de dados".into()));
                }
                let k = headers.len();
                // Verifica que todas as linhas têm o mesmo número de colunas
                for (i, row) in rows.iter().enumerate() {
                    if row.len() != k {
                        return Err(HayashiError::Runtime(format!(
                            "input: linha {} tem {} valores, esperado {} ({})",
                            i + 1, row.len(), k, headers.join(", ")
                        )));
                    }
                }
                let n = rows.len();
                // Transpõe: rows → columns
                let mut col_map: std::collections::HashMap<String, ndarray::Array1<f64>> = std::collections::HashMap::new();
                for (j, name) in headers.iter().enumerate() {
                    let col: ndarray::Array1<f64> = ndarray::Array1::from(
                        rows.iter().map(|r| r[j]).collect::<Vec<_>>()
                    );
                    col_map.insert(name.clone(), col);
                }
                let df = greeners::DataFrame::new(col_map)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("input → {alias} ({n} obs, {} vars: {})", k, headers.join(", "));
                self.env.set(&alias, Value::DataFrame(df));
            }

            // ── display expr ─────────────────────────────────────────────────
            Stmt::Display(expr) => {
                let val = self.eval_expr(expr)?;
                match &val {
                    Value::Float(v)  => println!("{v}"),
                    Value::Int(v)    => println!("{v}"),
                    Value::Bool(v)   => println!("{v}"),
                    Value::Str(v)    => println!("{v}"),
                    Value::Nil       => println!("(nil)"),
                    Value::List(lst) => {
                        for v in lst.iter() { print!("  {v}"); }
                        println!();
                    }
                    _ => println!("{val}"),
                }
            }

            Stmt::Load { path, alias } => {
                let path_str = match self.eval_expr(path)? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("load requires a string path".into())),
                };

                // Para URLs: baixa para arquivo temporário e usa o caminho local
                let _tmp; // mantém o TempPath vivo durante a leitura
                let local_path: &str = if crate::io::fetch::is_url(&path_str) {
                    println!("Downloading '{}'…", path_str);
                    _tmp = crate::io::fetch::download_to_temp(&path_str)?;
                    _tmp.to_str().ok_or_else(|| HayashiError::Runtime("temp path is not UTF-8".into()))?
                } else {
                    &path_str
                };

                let (df, n_rows) = if local_path.ends_with(".dta") {
                    crate::io::dta::load_dta(local_path)?
                } else {
                    let df = DataFrame::from_csv(local_path)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let n = df.n_rows();
                    (df, n)
                };
                println!("Loaded '{}' → {alias} ({} rows)", path_str, n_rows);
                self.env.set(alias, Value::DataFrame(df));
            }

            Stmt::Predict { df, varname, model, kind } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df}' is not a DataFrame"))),
                };
                let model_val = self.eval_expr(model)?;

                let vals: Vec<f64> = match (&model_val, kind.as_str()) {
                    // ── OLS ──────────────────────────────────────────────────
                    (Value::OlsResult(m), "xb" | "fitted") => {
                        m.x.dot(&m.result.params).to_vec()
                    }
                    (Value::OlsResult(m), "residuals" | "resid" | "e") => {
                        m.residuals.to_vec()
                    }
                    (Value::OlsResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict OLS: kind '{k}' desconhecido — use: xb, residuals")
                    )),

                    // ── Logit / Probit ────────────────────────────────────────
                    (Value::BinaryResult(m), "pr" | "xb" | "fitted") => {
                        m.result.predict_proba(&m.x).to_vec()
                    }
                    (Value::BinaryResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict logit/probit: kind '{k}' desconhecido — use: pr")
                    )),

                    // ── Poisson / NegBin ──────────────────────────────────────
                    (Value::PoissonResult(r), "count" | "mu" | "fitted") => {
                        r.fitted_values().to_vec()
                    }
                    (Value::PoissonResult(r), "xb") => {
                        r.x_data().dot(&r.params).to_vec()
                    }
                    (Value::PoissonResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Poisson: kind '{k}' desconhecido — use: count, xb")
                    )),
                    (Value::NegBinResult(r), "count" | "mu" | "fitted") => {
                        r.fitted_values().to_vec()
                    }
                    (Value::NegBinResult(r), "xb") => {
                        r.x_data().dot(&r.params).to_vec()
                    }
                    (Value::NegBinResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict NegBin: kind '{k}' desconhecido — use: count, xb")
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
                                        format!("predict Ordered: categoria {cat} fora do intervalo 1..{}", r.n_categories)
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
                                format!("predict Ordered: kind '{k}' desconhecido — use: pr, prN, yhat, xb")
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
                        format!("predict IV: kind '{k}' desconhecido — use: xb")
                    )),

                    // ── Panel FE / RE ─────────────────────────────────────────
                    (Value::PanelResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::PanelResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict FE: kind '{k}' desconhecido — use: xb")
                    )),
                    (Value::ReResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::ReResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RE: kind '{k}' desconhecido — use: xb")
                    )),

                    // ── Tobit ─────────────────────────────────────────────────
                    (Value::TobitResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::TobitResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Tobit: kind '{k}' desconhecido — use: xb")
                    )),

                    // ── Heckman ───────────────────────────────────────────────
                    (Value::HeckmanResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::HeckmanResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Heckman: kind '{k}' desconhecido — use: xb")
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
                        format!("predict Cox: kind '{k}' desconhecido — use: loghr, hr")
                    )),

                    // ── Quantile Regression ───────────────────────────────────
                    (Value::QuantileResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::QuantileResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict QReg: kind '{k}' desconhecido — use: xb")
                    )),

                    // ── RLM ──────────────────────────────────────────────────
                    (Value::RlmResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::RlmResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RLM: kind '{k}' desconhecido — use: xb")
                    )),

                    // ── GEE ──────────────────────────────────────────────────
                    (Value::GeeResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::GeeResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict GEE: kind '{k}' desconhecido — use: xb")
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
                        format!("predict BetaReg: kind '{k}' desconhecido — use: pr, xb")
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
                        format!("predict GLSAR: kind '{k}' desconhecido — use: xb, residuals")
                    )),

                    // ── MixedLM ───────────────────────────────────────────────
                    (Value::MixedResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = Self::build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.fixed_effects).to_vec()
                    }
                    (Value::MixedResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict MixedLM: kind '{k}' desconhecido — use: xb")
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
                        format!("predict ZIP/ZINB: kind '{k}' desconhecido — use: count, pr0")
                    )),

                    // ── Rolling OLS ───────────────────────────────────────────
                    (Value::RollingResult(r), "residuals" | "resid" | "e") => {
                        r.residuals.to_vec()
                    }
                    (Value::RollingResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RollingOLS: kind '{k}' desconhecido — use: residuals")
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
                        format!("predict RecursiveLS: kind '{k}' desconhecido — use: residuals, cusum, cusum_sq")
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
                        format!("predict GLM: kind '{k}' desconhecido — use: pr, xb, residuals, pearson, working")
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
                        format!("predict LOWESS: kind '{k}' desconhecido — use: smoothed, residuals")
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
                                    format!("predict PCA: componente {comp} fora do intervalo 1..{}", m.result.n_components)
                                ));
                            }
                            m.result.scores.column(comp - 1).to_vec()
                        } else {
                            return Err(HayashiError::Runtime(
                                format!("predict PCA: kind '{kind_s}' desconhecido — use: pc1, pc2, ..., pc{}", m.result.n_components)
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
                                format!("predict MarkovSwitching: regime {idx} fora do intervalo 1..{}", r.n_regimes)
                            ));
                        }
                        r.smoothed_probs.column(idx - 1).to_vec()
                    }
                    (Value::MarkovResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict MarkovSwitching: kind '{k}' desconhecido — use: regime, regime1, regime2, ...")
                    )),

                    // ── Conditional Logit / Poisson ───────────────────────────────
                    // FE é diferenciado; predição incondicional não disponível
                    (Value::ConditionalResult(_), _) => return Err(HayashiError::Runtime(
                        "predict clogit/cpoisson: efeitos fixos absorvidos — predição incondicional não disponível; use os coeficientes β̂ para odds ratios ou efeitos marginais".into()
                    )),

                    // ── VARMA ─────────────────────────────────────────────────────
                    (Value::VarmaResult(_), _) => return Err(HayashiError::Runtime(
                        "predict varma: predição multivariada não suportada como coluna — use print() para diagnóstico".into()
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
                        format!("predict ucm: kind '{k}' desconhecido — use: level, trend, seasonal, residuals")
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
                        format!("predict dfm: kind '{k}' desconhecido — use: f1, f2, ... (índice 1-based do fator latente)")
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
                                "predict msauto: regime{} fora do intervalo 1..{}",
                                idx + 1, r.k_regimes
                            )));
                        }
                        r.smoothed_probs.column(idx).to_vec()
                    }
                    (Value::MSARResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict msauto: kind '{k}' desconhecido — use: regime, regime1, regime2, ...")
                    )),

                    // ── Decomposição sazonal ──────────────────────────────────────
                    (Value::DecompResult(r), "trend")    => r.trend.to_vec(),
                    (Value::DecompResult(r), "seasonal") => r.seasonal.to_vec(),
                    (Value::DecompResult(r), "residual" | "resid" | "e") => r.residual.to_vec(),
                    (Value::DecompResult(r), "observed" | "fitted") => r.observed.to_vec(),
                    (Value::DecompResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict decompose: kind '{k}' desconhecido — use: trend, seasonal, residual, observed")
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
                        format!("predict mstl: kind '{k}' desconhecido — use: trend, resid, seasonal, seasonal1, seasonal2, ...")
                    )),

                    // ── ETS (suavização exponencial) ──────────────────────────
                    (Value::EtsResult(r), "fitted" | "yhat" | "xb") => r.fitted_values.to_vec(),
                    (Value::EtsResult(r), "residuals" | "resid" | "e") => r.residuals.to_vec(),
                    (Value::EtsResult(r), "level")    => r.level.to_vec(),
                    (Value::EtsResult(r), "trend")    => r.trend.to_vec(),
                    (Value::EtsResult(r), "seasonal") => r.seasonal.to_vec(),
                    (Value::EtsResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict ets: kind '{k}' desconhecido — use: fitted, residuals, level, trend, seasonal")
                    )),

                    // ── PanelThreshold ────────────────────────────────────────
                    (Value::ThresholdResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict pthresh: kind '{k}' — use print() para ver limiares e coeficientes")
                    )),

                    _ => return Err(HayashiError::Type(
                        "predict: tipo de modelo não suportado".into()
                    )),
                };

                let arr = ndarray::Array1::from(vals);
                df_val.insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!("({} obs)  {df}.{varname} ({kind}) predicted", df_val.n_rows());
                self.env.set(df, Value::DataFrame(df_val));
            }

            Stmt::Count { df, cond } => {
                let df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df}' is not a DataFrame"))),
                };
                let n = if let Some(cond_expr) = cond {
                    let mask = Self::eval_col_expr(cond_expr, &df_val)?;
                    mask.iter().filter(|&&v| v != 0.0).count()
                } else {
                    df_val.n_rows()
                };
                println!("{n}");
            }

            Stmt::Replace { df, varname, expr, cond } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df}' is not a DataFrame"))),
                };
                let new_vals = Self::eval_col_expr(expr, &df_val)?;

                let final_vals: Vec<f64> = if let Some(cond_expr) = cond {
                    let mask = Self::eval_col_expr(cond_expr, &df_val)?;
                    // lê coluna original para preservar onde mask == 0
                    use greeners::Column;
                    let old_vals: Vec<f64> = match df_val.get_column(varname) {
                        Ok(Column::Float(arr)) => arr.to_vec(),
                        Ok(Column::Int(arr))   => arr.iter().map(|&v| v as f64).collect(),
                        _ => vec![f64::NAN; new_vals.len()],
                    };
                    let n_replaced = mask.iter().filter(|&&m| m != 0.0).count();
                    println!("({n_replaced} real changes made)");
                    mask.into_iter().zip(old_vals).zip(new_vals)
                        .map(|((m, old), new)| if m != 0.0 { new } else { old })
                        .collect()
                } else {
                    let n = new_vals.len();
                    println!("({n} real changes made)");
                    new_vals
                };

                let arr = ndarray::Array1::from(final_vals);
                df_val.insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                self.env.set(df, Value::DataFrame(df_val));
            }

            Stmt::Generate { df, varname, expr } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df}' is not a DataFrame"))),
                };
                let vals = Self::eval_col_expr(expr, &df_val)?;
                let arr = ndarray::Array1::from(vals);
                df_val.insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!("({} obs)  {df}.{varname} generated", df_val.n_rows());
                self.env.set(df, Value::DataFrame(df_val));
            }

            Stmt::Print(expr) => {
                let val = self.eval_expr(expr)?;
                println!("{val}");
            }

            Stmt::Export { value, fmt, path } => {
                let val = self.eval_expr(value)?;
                let path_str = match self.eval_expr(path)? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("export path must be a string".into())),
                };

                use greeners::ExportableResult;

                match (val, fmt.as_str()) {
                    // ── DataFrame → CSV ───────────────────────────────────────
                    (Value::DataFrame(df), "csv" | "delimited") => {
                        df.to_csv(&path_str)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                    }

                    // ── OLS → CSV / LaTeX / HTML ──────────────────────────────
                    (Value::OlsResult(m), "csv") => {
                        let content = m.result.to_csv();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported OLS coefficients → '{path_str}'");
                    }
                    (Value::OlsResult(m), "latex" | "tex") => {
                        let content = m.result.to_latex();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported OLS LaTeX table → '{path_str}'");
                    }
                    (Value::OlsResult(m), "html") => {
                        let content = m.result.to_html();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported OLS HTML table → '{path_str}'");
                    }

                    // ── Outros modelos: texto plain ───────────────────────────
                    (Value::IvResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported IV results → '{path_str}'");
                    }
                    (Value::BinaryResult(m), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{m}"))
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported logit/probit results → '{path_str}'");
                    }
                    (Value::PanelResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported FE results → '{path_str}'");
                    }
                    (Value::ReResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported RE results → '{path_str}'");
                    }

                    (_, fmt) => return Err(HayashiError::Runtime(format!(
                        "unsupported export format '{fmt}' for this value type\n\
                         Available: DataFrame→csv  |  OLS→csv,latex,html  |  other models→txt"
                    ))),
                }
            }

            Stmt::Tsset { df, t_var } => {
                let frame = match self.env.get(df)
                    .ok_or_else(|| HayashiError::Runtime(format!("'{df}' não definido")))?
                {
                    Value::DataFrame(d) => d.clone(),
                    _ => return Err(HayashiError::Type(format!("'{df}' não é um DataFrame"))),
                };

                // ordena por t_var (sort_df_by reporta erro se coluna não existe)
                let sorted = Self::sort_df_by(&frame, t_var)?;

                // estatísticas da variável de tempo para o sumário
                let t_vals = Self::eval_col_expr(&Expr::Var(t_var.clone()), &sorted)?;
                let t_min = t_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let t_max = t_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let n = sorted.n_rows();

                self.ts_info.insert(df.clone(), t_var.clone());
                self.env.set(df, Value::DataFrame(sorted));

                println!("tsset {df}");
                println!("  variável de tempo : {t_var}  ({t_min} a {t_max})");
                println!("  n = {n}");
                println!();
            }

            // ── if / else ────────────────────────────────────────────────────
            Stmt::If { cond, then_body, else_body } => {
                let cond_val = self.eval_expr(cond)?;
                if Self::value_as_bool(&cond_val) {
                    for s in then_body { self.exec(s)?; }
                } else if let Some(else_stmts) = else_body {
                    for s in else_stmts { self.exec(s)?; }
                }
            }

            // ── for var in iter { ... } ───────────────────────────────────────
            // Variáveis declaradas no corpo persistem no escopo externo (R-style).
            Stmt::For { var, iter, body } => {
                // Executa um passo do corpo; retorna true=continue, false=break
                macro_rules! run_body {
                    () => {{
                        let mut do_break = false;
                        for s in body {
                            match self.exec(s) {
                                Ok(()) => {}
                                Err(HayashiError::Continue) => break,
                                Err(HayashiError::Break)    => { do_break = true; break; }
                                Err(e) => return Err(e),
                            }
                        }
                        do_break
                    }};
                }
                match iter {
                    ForIter::Range(start_expr, end_expr) => {
                        let start = match self.eval_expr(start_expr)? {
                            Value::Int(i) => i,
                            Value::Float(f) => f as i64,
                            v => return Err(HayashiError::Type(
                                format!("for: início do range deve ser inteiro, não {v}")
                            )),
                        };
                        let end = match self.eval_expr(end_expr)? {
                            Value::Int(i) => i,
                            Value::Float(f) => f as i64,
                            v => return Err(HayashiError::Type(
                                format!("for: fim do range deve ser inteiro, não {v}")
                            )),
                        };
                        let step: i64 = if start <= end { 1 } else { -1 };
                        let mut cur = start;
                        while if step > 0 { cur < end } else { cur > end } {
                            self.env.set(var, Value::Int(cur));
                            if run_body!() { break; }
                            cur += step;
                        }
                    }
                    ForIter::Items(iter_expr) => {
                        let items = match self.eval_expr(iter_expr)? {
                            Value::List(v) => (*v).clone(),
                            other => return Err(HayashiError::Type(
                                format!("for: iterador deve ser lista, não {other}")
                            )),
                        };
                        for item in items {
                            self.env.set(var, item);
                            if run_body!() { break; }
                        }
                    }
                }
            }

            // ── fn nome(params) { corpo } ────────────────────────────────────
            Stmt::Fn { name, params, body } => {
                self.env.set(name, Value::UserFn(Rc::new(UserFn {
                    params: params.clone(),
                    body: body.clone(),
                })));
            }

            // ── return [expr] ─────────────────────────────────────────────────
            Stmt::Return(expr) => {
                let val = match expr {
                    Some(e) => self.eval_expr(e)?,
                    None    => Value::Nil,
                };
                self.return_value = Some(val);
                return Err(HayashiError::Return);
            }

            Stmt::Break    => return Err(HayashiError::Break),
            Stmt::Continue => return Err(HayashiError::Continue),

            // ── while cond { ... } ───────────────────────────────────────────
            Stmt::While { cond, body } => {
                'outer: loop {
                    let cond_val = self.eval_expr(cond)?;
                    if !Self::value_as_bool(&cond_val) { break; }
                    for s in body {
                        match self.exec(s) {
                            Ok(()) => {}
                            Err(HayashiError::Break)    => break 'outer,
                            Err(HayashiError::Continue) => break,
                            Err(e) => return Err(e),
                        }
                    }
                }
            }

            Stmt::Expr(expr) => {
                let val = self.eval_expr(expr)?;
                if !matches!(val, Value::Nil) {
                    println!("{val}");
                }
            }
        }
        Ok(())
    }
}
