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

/// Language automatic output: respects `quiet_mode` and `capturing`.
/// Use for every print that is not explicit user output (print/display).
#[macro_export]
macro_rules! emit {
    ($self:expr, $($arg:tt)*) => {
        if !$self.capturing && !$self.env.quiet_mode() {
            print!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! emitln {
    ($self:expr) => {
        if !$self.capturing && !$self.env.quiet_mode() {
            println!();
        }
    };
    ($self:expr, $($arg:tt)*) => {
        if !$self.capturing && !$self.env.quiet_mode() {
            println!($($arg)*);
        }
    };
}

// ── eval_call() split by category (see src/lang/interpreter/) ───────────────
// Each submodule implements `impl Interpreter { fn eval_call_X(...) -> Result<Option<Value>> }`
// Returns `Ok(None)` when `func` does not belong to the category, so the dispatcher can try the next.
mod builtins;
mod data_manipulation;
mod descriptive_lang;
mod dispatch;
mod env;
mod estimators_micro;
mod estimators_misc;
mod estimators_panel;
mod estimators_timeseries;
mod eval_expr;
mod execution;
mod helpers;
mod models;
mod post_estimation_ts;
mod value;
mod visualization;

use self::helpers::*;

pub use builtins::BUILTIN_NAMES;
pub use env::Env;
pub use models::{
    BinaryModel, DFMModel, DiagResult, OlsModel, PcaModel, PenalizedModel, SurModel, ThreeSLSModel,
};
pub use value::{ErrorValue, Series, UserFn, Value};

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

// ── Interpreter ──────────────────────────────────────────────────────────────

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
                #[cfg(not(target_arch = "wasm32"))]
                let r = rand::rngs::StdRng::from_entropy();
                #[cfg(target_arch = "wasm32")]
                let r = rand::rngs::StdRng::seed_from_u64(0x4841_5951);
                r
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
            Value::Error(_) => "Error",
            Value::OlsResult(_) => "OlsResult",
            Value::IvResult(_) => "IvResult",
            Value::PenalizedResult(_) => "PenalizedResult",
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

    pub(super) fn eval_as_int(&mut self, expr: &Expr, ctx: &str) -> Result<i64> {
        match self.eval_expr(expr)? {
            Value::Int(i) => Ok(i),
            Value::Float(f) => Ok(f as i64),
            v => Err(self.type_err(format!(
                "{ctx} must be integer, got {}",
                Self::type_name(&v)
            ))),
        }
    }

    pub(super) fn resolve_var_list(
        &mut self,
        args: &[Expr],
        df: &greeners::DataFrame,
    ) -> Result<Vec<String>> {
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

    pub(super) fn call_value_fn(&mut self, f: &Value, args: &[Value]) -> Result<Value> {
        match f {
            Value::UserFn(uf) => {
                if args.len() > uf.params.len() {
                    return Err(self.rt_err(format!(
                        "function expects at most {} arguments, got {}",
                        uf.params.len(),
                        args.len()
                    )));
                }
                self.env.push_scope();
                for (param, val) in uf.params.iter().zip(args.iter()) {
                    self.env.declare_const(param, val.clone());
                }
                for i in args.len()..uf.params.len() {
                    let param = &uf.params[i];
                    let val = if let Some(default_expr) = &uf.defaults[i] {
                        self.eval_expr(default_expr)?
                    } else {
                        return Err(self.rt_err(format!("missing required argument '{param}'")));
                    };
                    self.env.declare_const(param, val);
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
        let mut columns: indexmap::IndexMap<String, greeners::Column> = indexmap::IndexMap::new();
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

            // In release builds, restrict native/WASM plugins
            // to be loaded only from ~/.hay/packages/.
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

    // ── Expression evaluation ───────────────────────────────────────────────

    pub(super) fn maybe_filter_df(
        &mut self,
        df: &Rc<DataFrame>,
        opts: &[Opt],
    ) -> Result<Rc<DataFrame>> {
        if let Some(if_opt) = opts.iter().find(|o| o.name == "if") {
            let mask = self.eval_col_expr(&if_opt.value, df)?;
            filter_df_by_mask(df, &mask)
        } else {
            Ok(df.clone())
        }
    }

    // ── Built-in functions ────────────────────────────────────────────────────

    pub(super) fn resolve_formula(&mut self, expr: &Expr) -> Result<Formula> {
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

    // ── Object methods ──────────────────────────────────────────────────────

    // ── Element-wise expression evaluation over DataFrame columns ───────────

    // ── Statement execution ─────────────────────────────────────────────────
}
