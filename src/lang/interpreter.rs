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

    // ── Nomes dos coeficientes a partir da fórmula ────────────────────────────



    // ── Extrai coluna como Vec<String> (para tabulate) ────────────────────────


    // ── Tabela cruzada (bi-variada, opcional chi2) ────────────────────────────

    fn __removed_tabulate_two(df: &DataFrame, row_var: &str, col_var: &str, do_chi2: bool) -> Result<()> {
        let rows = col_to_strings(df, row_var)?;
        let cols = col_to_strings(df, col_var)?;

        if rows.len() != cols.len() {
            return Err(HayashiError::Runtime(
                "columns have different lengths".into(),
            ));
        }

        // valores únicos, ordenados
        let sort_strs = |mut v: Vec<String>| -> Vec<String> {
            sort_maybe_numeric_strings(&mut v);
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
                "             │".to_string()
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
                    (pc..=col).for_each(|c| {
                        let t = (c - pc) as f64 / (col - pc).max(1) as f64;
                        let r = ((pr as f64 + t * (row as f64 - pr as f64)).round() as usize)
                            .min(h - 1);
                        if grid[r][c] == ' ' {
                            grid[r][c] = '─';
                        }
                    });
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
                "             │".to_string()
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
        line[c_wlo..=c_whi].fill('─');
        line[c_q1..=c_q3].fill('█');
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
        for (lag, v) in values.iter().enumerate().skip(1) {
            let bar_len = ((v.abs() * half as f64).round() as usize).min(half);
            let in_ci = v.abs() <= ci;
            let bar_char = if in_ci { '─' } else { '█' };
            let bar: String = std::iter::repeat_n(bar_char, bar_len).collect();
            let (left, right) = if *v >= 0.0 {
                (format!("{:<half$}", " "), bar.to_string())
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
        let n = cols[0].len();
        let means: Vec<f64> = cols
            .iter()
            .map(|c| c.iter().sum::<f64>() / n as f64)
            .collect();
        // Calcula correlações
        let corr: Vec<Vec<f64>> = cols
            .iter()
            .enumerate()
            .map(|(i, col_i)| {
                let xi: Vec<f64> = col_i.iter().map(|x| x - means[i]).collect();
                let di = xi.iter().map(|a| a * a).sum::<f64>().sqrt();
                cols.iter()
                    .enumerate()
                    .map(|(j, col_j)| {
                        let xj: Vec<f64> = col_j.iter().map(|x| x - means[j]).collect();
                        let num: f64 = xi.iter().zip(&xj).map(|(a, b)| a * b).sum();
                        let dj = xj.iter().map(|b| b * b).sum::<f64>().sqrt();
                        if di * dj < 1e-15 {
                            0.0
                        } else {
                            num / (di * dj)
                        }
                    })
                    .collect()
            })
            .collect();
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
        for (name, row) in names.iter().zip(&corr) {
            let name_disp = &name[..name.len().min(nw)];
            print!("{:>nw$}", name_disp);
            for v in row {
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
                let sign = if *v < 0.0 { "-" } else { "+" };
                print!(" {sign}{shade}");
            }
            print!("   ");
            for v in row {
                print!(" {:>6.3}", v);
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
                emitln!(
                    self,
                    "input → {alias} ({n} obs, {} vars: {})",
                    k,
                    headers.join(", ")
                );
                self.env.set(alias, Value::DataFrame(Rc::new(df)))?;
            }

            // ── display expr ─────────────────────────────────────────────────
            Stmt::Display(expr) => {
                eprintln!(
                    "warning: 'display' is deprecated; use 'print' instead"
                );
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
                        emitln!(self, "Loaded ODBC → {alias} ({n_rows} rows)");
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
                        emitln!(self, "Downloading '{}'…", path_str);
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
                    emitln!(self, "Loaded '{}' → {alias} ({} rows)", path_str, n_rows);
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
                        let x = build_x_from_varnames(&df_val,
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
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::IvResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict IV: kind '{k}' unknown — use: xb")
                    )),

                    // ── Panel FE / RE ─────────────────────────────────────────
                    (Value::PanelResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::PanelResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict FE: kind '{k}' unknown — use: xb")
                    )),
                    (Value::ReResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::ReResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RE: kind '{k}' unknown — use: xb")
                    )),

                    // ── Tobit ─────────────────────────────────────────────────
                    (Value::TobitResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::TobitResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Tobit: kind '{k}' unknown — use: xb")
                    )),

                    // ── Heckman ───────────────────────────────────────────────
                    (Value::HeckmanResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::HeckmanResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Heckman: kind '{k}' unknown — use: xb")
                    )),

                    // ── Cox PH ────────────────────────────────────────────────
                    (Value::CoxResult(r), "loghr" | "xb") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        r.predict_log_hazard(&x).to_vec()
                    }
                    (Value::CoxResult(r), "hr" | "hazard") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        r.predict_hazard_ratio(&x).to_vec()
                    }
                    (Value::CoxResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict Cox: kind '{k}' unknown — use: loghr, hr")
                    )),

                    // ── Quantile Regression ───────────────────────────────────
                    (Value::QuantileResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::QuantileResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict QReg: kind '{k}' unknown — use: xb")
                    )),

                    // ── RLM ──────────────────────────────────────────────────
                    (Value::RlmResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::RlmResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict RLM: kind '{k}' unknown — use: xb")
                    )),

                    // ── GEE ──────────────────────────────────────────────────
                    (Value::GeeResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::GeeResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict GEE: kind '{k}' unknown — use: xb")
                    )),

                    // ── Beta Regression ───────────────────────────────────────
                    (Value::BetaResult(r), "pr" | "mu" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        r.predict(&x, &greeners::BetaLink::Logit).to_vec()
                    }
                    (Value::BetaResult(r), "xb") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.params).to_vec()
                    }
                    (Value::BetaResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict BetaReg: kind '{k}' unknown — use: pr, xb")
                    )),

                    // ── GLSAR ────────────────────────────────────────────────
                    (Value::GlsarResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        r.fitted_values(&x).to_vec()
                    }
                    (Value::GlsarResult(r), "residuals" | "resid" | "e") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        let y = get_col_f64(&df_val, varname)?;
                        r.residuals(&y, &x).to_vec()
                    }
                    (Value::GlsarResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict GLSAR: kind '{k}' unknown — use: xb, residuals")
                    )),

                    // ── MixedLM ───────────────────────────────────────────────
                    (Value::MixedResult(r), "xb" | "fitted") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
                        x.dot(&r.fixed_effects).to_vec()
                    }
                    (Value::MixedResult(_), k) => return Err(HayashiError::Runtime(
                        format!("predict MixedLM: kind '{k}' unknown — use: xb")
                    )),

                    // ── ZIP / ZINB ────────────────────────────────────────────
                    (Value::ZeroInflatedResult(r), "count" | "mu" | "fitted") => {
                        // E[y|x, w>0] × P(w=0): media incondicional da contagem
                        let names = r.count_var_names.as_deref().unwrap_or(&[]);
                        let x_c = build_x_from_varnames(&df_val, names)?;
                        let inflate_names = r.inflate_var_names.as_deref().unwrap_or(names);
                        let x_i = build_x_from_varnames(&df_val, inflate_names)?;
                        r.predict_count(&x_c, &x_i).to_vec()
                    }
                    (Value::ZeroInflatedResult(r), "pr0") => {
                        // P(y=0 | x) — probabilidade de zero
                        let names = r.count_var_names.as_deref().unwrap_or(&[]);
                        let x_c = build_x_from_varnames(&df_val, names)?;
                        let inflate_names = r.inflate_var_names.as_deref().unwrap_or(names);
                        let x_i = build_x_from_varnames(&df_val, inflate_names)?;
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
                        let x = build_x_from_varnames(&df_val, names)?;
                        r.predict_mean(&x).to_vec()
                    }
                    (Value::GlmResult(r), "xb") => {
                        let names = r.variable_names.as_deref().unwrap_or(&[]);
                        let x = build_x_from_varnames(&df_val, names)?;
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
                emitln!(self, "({} obs)  {df}.{varname} generated", df_val.n_rows());
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
                        | Value::QuantileResult(_)
                        | Value::GmmResult(_)
                        | Value::DidResult(_)
                        | Value::ArimaResult(_)
                        | Value::GarchResult(_)
                        | Value::VarResult(_)
                        | Value::VecmResult(_)
                        | Value::AutoRegResult(_)
                        | Value::ArdlResult(_)
                        | Value::SVarResult(_)
                        | Value::MSARResult(_)
                        | Value::EtsResult(_)
                        | Value::SurResult(_)
                        | Value::ThreeSLSResult(_)
                        | Value::RdResult(_)
                        | Value::PsmResult(_)
                        | Value::SynthResult(_)
                        | Value::RlmResult(_)
                        | Value::GeeResult(_)
                        | Value::ZeroInflatedResult(_)
                        | Value::MixedResult(_)
                        | Value::BetaResult(_)
                        | Value::GlsarResult(_)
                        | Value::GlmResult(_)
                        | Value::LowessResult(_)
                        | Value::PcaResult(_)
                        | Value::FactorResult(_)
                        | Value::MarkovResult(_)
                        | Value::ConditionalResult(_)
                        | Value::VarmaResult(_)
                        | Value::DecompResult(_)
                        | Value::MstlResult(_)
                        | Value::UCResult(_)
                        | Value::GamResult(_)
                        | Value::MiceResult(_)
                        | Value::DFMResult(_)
                        | Value::ThresholdResult(_)
                        | Value::RollingResult(_)
                        | Value::RecursiveLSResult(_)
                        | Value::AbResult(_)
                        | Value::SysGmmResult(_)
                        | Value::DiagResult(_)
                        | Value::PcseResult(_)
                        | Value::PanelGlsResult(_)
                        | Value::OrderedResult(_)
                        | Value::MNLogitResult(_)
                        | Value::PenalizedResult(_)),
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
                let sorted = sort_df_by(&frame, t_var)?;

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
                if value_as_bool(&cond_val) {
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
                            caught = Some(ErrorValue::from_hayashi_error(&e, self.current_line));
                            break;
                        }
                    }
                }
                self.env.pop_scope();
                if let Some(err) = caught {
                    self.env.push_scope();
                    self.env.declare(error_var, Value::Error(Rc::new(err)))?;
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
            Stmt::Fn { name, params, defaults, doc, body } => {
                self.env.set(
                    name,
                    Value::UserFn(Rc::new(UserFn {
                        params: params.clone(),
                        defaults: defaults.clone(),
                        doc: doc.clone(),
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
                if !value_as_bool(&cond_val) {
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
                            Value::Str(v) => emitln!(self, "\"{v}\""),
                            _ => emitln!(self, "{val}"),
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

            Stmt::QuietlyOn => {
                self.env.set_quiet_mode(true);
            }

            Stmt::QuietlyOff => {
                self.env.set_quiet_mode(false);
            }
        }
        Ok(())
    }
}
