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

/// Saída automática da linguagem: respeita `quiet_mode` e `capturing`.
/// Use para todo print que não seja uma saída explícita do usuário (print/display).
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

// ── eval_call() dividido por categoria (ver src/lang/interpreter/) ──────────
// Cada submódulo implementa `impl Interpreter { fn eval_call_X(...) -> Result<Option<Value>> }`
// Retorna `Ok(None)` quando `func` não pertence à categoria, para o dispatcher tentar a próxima.
mod builtins;
mod data_manipulation;
mod descriptive_lang;
mod estimators_micro;
mod estimators_misc;
mod estimators_panel;
mod estimators_timeseries;
mod env;
mod execution;
mod helpers;
mod models;
mod post_estimation_ts;
mod value;
mod visualization;

use self::helpers::*;

pub use env::Env;
pub use models::{
    BinaryModel, DFMModel, DiagResult, OlsModel, PcaModel, PenalizedModel, SurModel,
    ThreeSLSModel,
};
pub use value::{ErrorValue, Series, UserFn, Value};
pub use builtins::BUILTIN_NAMES;

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


// ── Interpetador ──────────────────────────────────────────────────────────────

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
                        return Err(self.rt_err(format!(
                            "missing required argument '{param}'"
                        )));
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
                defaults: vec![None; params.len()],
                doc: None,
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

            Expr::If {
                cond,
                then_expr,
                else_expr,
            } => {
                let cond_val = self.eval_expr(cond)?;
                if value_as_bool(&cond_val) {
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
                        if !value_as_bool(&l) {
                            return Ok(Value::Bool(false));
                        }
                        let r = self.eval_expr(rhs)?;
                        return Ok(Value::Bool(value_as_bool(&r)));
                    }
                    BinOp::Or => {
                        let l = self.eval_expr(lhs)?;
                        if value_as_bool(&l) {
                            return Ok(Value::Bool(true));
                        }
                        let r = self.eval_expr(rhs)?;
                        return Ok(Value::Bool(value_as_bool(&r)));
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
                eval_scalar_binop(op, l, r)
            }

            Expr::Neg(inner) => match self.eval_expr(inner)? {
                Value::Int(v) => Ok(Value::Int(-v)),
                Value::Float(v) => Ok(Value::Float(-v)),
                _ => Err(HayashiError::Type("negação unária requires number".into())),
            },

            Expr::Not(inner) => {
                let v = self.eval_expr(inner)?;
                Ok(Value::Bool(!value_as_bool(&v)))
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
                    (Value::Error(e), Value::Str(key)) => {
                        let v = match key.as_str() {
                            "kind" => Value::Str(e.kind.clone()),
                            "msg" => Value::Str(e.msg.clone()),
                            "message" => Value::Str(e.msg.clone()),
                            "line" => Value::Int(e.line),
                            _ => {
                                return Err(HayashiError::Runtime(format!(
                                    "error field '{key}' not found (available: kind, msg, line)"
                                )))
                            }
                        };
                        Ok(v)
                    }
                    (Value::Error(_), _) => {
                        Err(HayashiError::Type("error index must be a string".into()))
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
                                        .cloned()
                                        .unwrap_or_else(|| "".to_string());
                                    Value::Str(level)
                                })
                                .collect(),
                            Column::DateTime(arr) => {
                                arr.iter().map(|dt| Value::Str(dt.to_string())).collect()
                            }
                        };
                        Ok(Value::Series(Rc::new(Series::new(key.clone(), vals))))
                    }
                    (Value::DataFrame(_), _) => Err(HayashiError::Type(
                        "DataFrame column index must be a string".into(),
                    )),
                    (Value::Series(s), _) => {
                        let i = match idx_val {
                            Value::Int(i) => i,
                            Value::Float(f) => f as i64,
                            _ => {
                                return Err(HayashiError::Type("series index must be integer".into()))
                            }
                        };
                        let len = s.len() as i64;
                        let real = if i < 0 { len + i } else { i };
                        if real < 0 || real >= len {
                            return Err(HayashiError::Runtime(format!(
                                "index out of range (len={len})"
                            )));
                        }
                        Ok(s.values[real as usize].clone())
                    }
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

            Expr::Block(stmts, final_expr) => {
                self.env.push_scope();
                let mut result = Value::Nil;
                for s in stmts {
                    match self.exec(&(s.clone(), 0)) {
                        Ok(()) => {}
                        Err(e) => {
                            self.env.pop_scope();
                            return Err(e);
                        }
                    }
                }
                if let Some(e) = final_expr {
                    result = self.eval_expr(e)?;
                }
                self.env.pop_scope();
                Ok(result)
            }

            Expr::Quietly(inner) => {
                eprintln!(
                    "warning: quietly() functional form is deprecated; use quietly on / quietly off"
                );
                let old = self.env.quiet_mode;
                self.env.quiet_mode = true;
                let result = self.eval_expr(inner);
                self.env.quiet_mode = old;
                result
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



    fn maybe_filter_df(&mut self, df: &Rc<DataFrame>, opts: &[Opt]) -> Result<Rc<DataFrame>> {
        if let Some(if_opt) = opts.iter().find(|o| o.name == "if") {
            let mask = self.eval_col_expr(&if_opt.value, df)?;
            filter_df_by_mask(df, &mask)
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
        try_group!(eval_call_descriptive_lang);
        try_group!(eval_call_estimators_panel);
        try_group!(eval_call_estimators_micro);
        try_group!(eval_call_builtins);

        // ── Função definida pelo usuário ──────────────────────────────────
        let other = func;
        {
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

            if args.len() > user_fn.params.len() {
                return Err(HayashiError::Runtime(format!(
                    "fn '{other}': esperado no máximo {} argumento(s), recebido {}",
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
            for (param, val) in user_fn.params.iter().zip(arg_vals.iter()) {
                self.env.declare_const(param, val.clone());
            }
            for i in arg_vals.len()..user_fn.params.len() {
                let param = &user_fn.params[i];
                let val = if let Some(default_expr) = &user_fn.defaults[i] {
                    self.eval_expr(default_expr)?
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "fn '{other}': missing required argument '{param}'"
                    )));
                };
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

    // ── Helpers de painel (dependem de estado do interpretador) ─────────────

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

    // ── Helpers de visualização ASCII ────────────────────────────────────────









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
                            let strs = col_to_strings(df, &col_name)?;
                            return Ok(greeners::Transforms::group(&strs));
                        }
                        "date" => {
                            let col_name = match &args[0] {
                                Expr::Var(name) => name.clone(),
                                _ => return Err(HayashiError::Runtime(
                                    "date() requires a column name".into()
                                )),
                            };
                            let strs = col_to_strings(df, &col_name)?;
                            let result: Vec<f64> = strs
                                .iter()
                                .map(|s| {
                                    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                                        .ok()
                                        .and_then(|d| d.and_hms_opt(0, 0, 0))
                                        .map(|dt| dt.and_utc().timestamp() as f64)
                                        .unwrap_or(f64::NAN)
                                })
                                .collect();
                            return Ok(result);
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

}
