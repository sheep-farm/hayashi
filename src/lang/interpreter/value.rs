use crate::lang::ast::{Expr, Spanned};
use crate::lang::error::HayashiError;
use std::collections::HashMap;
use std::rc::Rc;

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
pub enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    DataFrame(Rc<greeners::DataFrame>),
    OlsResult(super::models::OlsModel),
    IvResult(Rc<greeners::iv::IvResult>),
    BinaryResult(super::models::BinaryModel),
    PanelResult(Rc<greeners::panel::PanelResult>),
    ReResult(Rc<greeners::panel::RandomEffectsResult>),
    ArimaResult(Rc<greeners::ArimaResult>),
    VarResult(Rc<greeners::var::VarResult>),
    VecmResult(Rc<greeners::vecm::VecmResult>),
    GarchResult(Rc<greeners::GarchResult>),
    DiagResult(Rc<super::models::DiagResult>),
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
    SurResult(super::models::SurModel),
    RollingResult(Rc<greeners::RollingResult>),
    RecursiveLSResult(Rc<greeners::RecursiveLSResult>),
    GlmResult(Rc<greeners::GlmResult>),
    LowessResult(Rc<greeners::LowessResult>),
    PcaResult(super::models::PcaModel),
    FactorResult(super::models::FactorModel),
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
    ThreeSLSResult(super::models::ThreeSLSModel),
    DFMResult(super::models::DFMModel),
    EtsResult(Rc<greeners::ETSResult>),
    PenalizedResult(super::models::PenalizedModel),
    ThresholdResult(Rc<greeners::threshold::ThresholdResult>),
    AutoRegResult(Rc<greeners::AutoRegResult>),
    ArdlResult(Rc<greeners::ARDLResult>),
    LocalLevelResult(Rc<greeners::LocalLevelResult>),
    List(Rc<Vec<Value>>),
    Dict(Rc<HashMap<String, Value>>),
    Series(Rc<Series>),
    UserFn(Rc<UserFn>),
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
            Value::PenalizedResult(m) => write!(f, "{m}"),
            Value::ThresholdResult(r) => write!(f, "{r}"),
            Value::AutoRegResult(r) => write!(f, "{r}"),
            Value::ArdlResult(r) => write!(f, "{r}"),
            Value::LocalLevelResult(r) => write!(f, "{r}"),
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
            Value::Plot { format, .. } => write!(f, "Plot({format})"),
            Value::Nil => write!(f, "nil"),
        }
    }
}
