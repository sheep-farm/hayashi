use crate::lang::ast::*;
use crate::lang::dap::protocol::*;
use crate::lang::error::{HayashiError, Result};
use greeners::diagnostics::Diagnostics;
use greeners::linalg::UPLO;
use greeners::linalg::{LinalgEigh as _, LinalgInverse as _};
use greeners::specification_tests::SpecificationTests;
use greeners::{chi2_pvalue, f_pvalue, logistic, norm_pdf, t_pvalue_two, t_quantile};
use greeners::{
    BinaryDiagnostics, CovarianceType, DataFrame, FixedEffects, Formula as GFormula, Logit, Probit,
    RandomEffects, IV, OLS,
};
use ndarray::{Array1, Array2, Axis};
use serde_json::{json, Value as JsonValue};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

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
pub mod models;
mod post_estimation_ts;
mod value;
mod visualization;

use self::helpers::*;

pub use builtins::BUILTIN_NAMES;
pub use env::Env;
pub use models::{
    BinaryModel, DFMModel, DiagResult, OlsModel, PcaModel, PenalizedModel, SurModel, ThreeSLSModel,
};
pub use value::{ErrorValue, SendValue, Series, UserFn, Value};

fn t_critical_95(df: f64) -> f64 {
    t_quantile(0.975, df)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugAction {
    Continue,
    StepOver,
    StepIn,
    StepOut,
    Pause,
}

#[derive(Debug)]
pub struct DebugState {
    pub current_file: std::path::PathBuf,
    pub breakpoints: HashSet<(std::path::PathBuf, usize)>,
    pub action: DebugAction,
    pub step_target_depth: Option<usize>,
    pub pending_command: Option<DebugCommand>,
    pub event_tx: std::sync::mpsc::Sender<DebugEvent>,
    pub control_rx: std::sync::mpsc::Receiver<ControlMessage>,
}

#[derive(Debug, Clone)]
pub enum DebugEvent {
    Initialized,
    Stopped {
        reason: String,
        description: Option<String>,
        thread_id: i64,
        preserve_focus_hint: bool,
    },
    Output {
        category: String,
        output: String,
    },
    Terminated,
    Exited(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugCommand {
    Continue,
    StepOver,
    StepIn,
    StepOut,
    Pause,
    Disconnect,
}

#[derive(Debug, Clone)]
pub enum ControlMessage {
    Command(DebugCommand),
    Request(
        crate::lang::dap::protocol::Request,
        std::sync::mpsc::Sender<crate::lang::dap::protocol::Response>,
    ),
}

impl DebugState {
    pub fn new(
        current_file: std::path::PathBuf,
        event_tx: std::sync::mpsc::Sender<DebugEvent>,
        control_rx: std::sync::mpsc::Receiver<ControlMessage>,
    ) -> Self {
        Self {
            current_file,
            breakpoints: HashSet::new(),
            action: DebugAction::Continue,
            step_target_depth: None,
            pending_command: None,
            event_tx,
            control_rx,
        }
    }
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
    pub debug_state: Option<DebugState>,
    pub current_source: std::path::PathBuf,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
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
            debug_state: None,
            current_source: std::path::PathBuf::new(),
        }
    }

    pub fn set_current_source(&mut self, path: impl Into<std::path::PathBuf>) {
        self.current_source = path.into();
    }

    pub fn enable_debug(
        &mut self,
        current_file: std::path::PathBuf,
        event_tx: std::sync::mpsc::Sender<DebugEvent>,
        control_rx: std::sync::mpsc::Receiver<ControlMessage>,
    ) {
        self.debug_state = Some(DebugState::new(current_file, event_tx, control_rx));
    }

    /// Blocks until the DAP client sends `configurationDone`. While waiting,
    /// it answers `initialize`, `launch`, `setBreakpoints` and other requests.
    pub fn debug_wait_for_start(&mut self) -> Result<()> {
        loop {
            let ds = self
                .debug_state
                .as_ref()
                .ok_or_else(|| HayashiError::Runtime("debugger not enabled".into()))?;
            let msg = ds.control_rx.recv();
            match msg {
                Ok(ControlMessage::Request(req, tx)) => {
                    let is_start = req.command == "configurationDone";
                    let resp = self.process_dap_request(&req);
                    let _ = tx.send(resp);
                    if is_start {
                        return Ok(());
                    }
                }
                Ok(ControlMessage::Command(DebugCommand::Disconnect)) => {
                    return Err(HayashiError::Runtime("debugger disconnected".into()));
                }
                Ok(ControlMessage::Command(_)) => {
                    // Ignore stray control commands before the session starts.
                }
                Err(_) => {
                    return Err(HayashiError::Runtime("debugger disconnected".into()));
                }
            }
        }
    }

    pub fn debug_set_breakpoints(&mut self, file: std::path::PathBuf, lines: Vec<usize>) {
        if let Some(ds) = self.debug_state.as_mut() {
            ds.breakpoints.retain(|(f, _)| f != &file);
            for line in lines {
                ds.breakpoints.insert((file.clone(), line));
            }
        }
    }

    pub fn debug_clear_breakpoints(&mut self) {
        if let Some(ds) = self.debug_state.as_mut() {
            ds.breakpoints.clear();
        }
    }

    pub fn call_stack(&self) -> &[(String, usize)] {
        &self.call_stack
    }

    pub fn current_line(&self) -> usize {
        self.current_line
    }

    pub fn current_source(&self) -> &std::path::Path {
        &self.current_source
    }

    /// Called at the start of every statement execution.
    pub(crate) fn debug_check(&mut self, file: &std::path::Path, line: usize) -> Result<()> {
        if self.debug_state.is_none() {
            return Ok(());
        }

        self.current_line = line;
        {
            let ds = self.debug_state.as_mut().unwrap();
            ds.current_file = file.to_path_buf();
        }

        // Non-blocking drain of any pending control messages while running.
        self.debug_drain_pending()?;

        // Apply any command that arrived before we decide whether to stop.
        let pending = {
            let ds = self.debug_state.as_mut().unwrap();
            ds.pending_command.take()
        };
        if let Some(cmd) = pending {
            self.apply_debug_command(cmd)?;
        }

        // Decide whether to stop using a snapshot of the debug action state.
        let call_stack_len = self.call_stack.len();
        let (should_stop, reason) = {
            let ds = self.debug_state.as_ref().unwrap();
            let mut should_stop = false;

            if ds.breakpoints.contains(&(file.to_path_buf(), line))
                && ds.action != DebugAction::StepIn
            {
                should_stop = true;
            }

            match ds.action {
                DebugAction::StepIn => should_stop = true,
                DebugAction::StepOver => {
                    if let Some(target) = ds.step_target_depth {
                        if call_stack_len <= target {
                            should_stop = true;
                        }
                    }
                }
                DebugAction::StepOut => {
                    if let Some(target) = ds.step_target_depth {
                        if call_stack_len <= target {
                            should_stop = true;
                        }
                    }
                }
                DebugAction::Pause => should_stop = true,
                DebugAction::Continue => {}
            }

            let reason = match ds.action {
                DebugAction::StepIn | DebugAction::StepOver | DebugAction::StepOut => "step",
                DebugAction::Pause => "pause",
                _ => "breakpoint",
            }
            .to_string();

            (should_stop, reason)
        };

        if !should_stop {
            return Ok(());
        }

        {
            let ds = self.debug_state.as_mut().unwrap();
            ds.action = DebugAction::Continue;
            ds.step_target_depth = None;
            ds.event_tx
                .send(DebugEvent::Stopped {
                    reason,
                    description: None,
                    thread_id: 1,
                    preserve_focus_hint: false,
                })
                .ok();
        }

        loop {
            let msg = {
                let ds = self.debug_state.as_ref().unwrap();
                ds.control_rx.recv()
            };
            match msg {
                Ok(ControlMessage::Command(cmd)) => {
                    self.apply_debug_command(cmd)?;
                    break;
                }
                Ok(ControlMessage::Request(req, tx)) => {
                    let resp = self.process_dap_request(&req);
                    let _ = tx.send(resp);
                }
                Err(_) => {
                    return Err(HayashiError::Runtime("debugger disconnected".into()));
                }
            }
        }

        Ok(())
    }

    fn apply_debug_command(&mut self, cmd: DebugCommand) -> Result<()> {
        let ds = self.debug_state.as_mut().unwrap();
        match cmd {
            DebugCommand::Continue => {
                ds.action = DebugAction::Continue;
            }
            DebugCommand::StepOver => {
                ds.action = DebugAction::StepOver;
                ds.step_target_depth = Some(self.call_stack.len());
            }
            DebugCommand::StepIn => {
                ds.action = DebugAction::StepIn;
            }
            DebugCommand::StepOut => {
                ds.action = DebugAction::StepOut;
                ds.step_target_depth = self.call_stack.len().checked_sub(1);
            }
            DebugCommand::Pause => {}
            DebugCommand::Disconnect => {
                return Err(HayashiError::Runtime("debugger disconnected".into()));
            }
        }
        Ok(())
    }

    fn debug_drain_pending(&mut self) -> Result<()> {
        if self.debug_state.is_none() {
            return Ok(());
        }
        loop {
            let msg = {
                let ds = self.debug_state.as_ref().unwrap();
                ds.control_rx.try_recv()
            };
            match msg {
                Ok(ControlMessage::Command(DebugCommand::Pause)) => {
                    let ds = self.debug_state.as_mut().unwrap();
                    ds.action = DebugAction::Pause;
                }
                Ok(ControlMessage::Command(DebugCommand::Disconnect)) => {
                    return Err(HayashiError::Runtime("debugger disconnected".into()));
                }
                Ok(ControlMessage::Command(cmd)) => {
                    let ds = self.debug_state.as_mut().unwrap();
                    ds.pending_command = Some(cmd);
                }
                Ok(ControlMessage::Request(req, tx)) => {
                    let resp = self.process_dap_request(&req);
                    let _ = tx.send(resp);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(HayashiError::Runtime("debugger disconnected".into()));
                }
            }
        }
        Ok(())
    }

    fn process_dap_request(&mut self, req: &Request) -> Response {
        match req.command.as_str() {
            "initialize" => {
                Response::ok(0, req.seq, &req.command).with_body(Capabilities::default())
            }
            "setBreakpoints" => {
                match serde_json::from_value::<SetBreakpointsArguments>(
                    req.arguments.clone().unwrap_or(JsonValue::Null),
                ) {
                    Ok(args) => {
                        let path = std::path::PathBuf::from(&args.source.path);
                        let lines: Vec<usize> =
                            args.breakpoints.iter().map(|b| b.line as usize).collect();
                        self.debug_set_breakpoints(path.clone(), lines.clone());
                        let breakpoints: Vec<Breakpoint> = lines
                            .iter()
                            .map(|l| Breakpoint {
                                line: *l as i64,
                                verified: Some(true),
                                message: None,
                            })
                            .collect();
                        Response::ok(0, req.seq, &req.command)
                            .with_body(json!({ "breakpoints": breakpoints }))
                    }
                    Err(e) => {
                        Response::err(0, req.seq, &req.command, format!("invalid arguments: {e}"))
                    }
                }
            }
            "threads" => Response::ok(0, req.seq, &req.command)
                .with_body(json!({ "threads": [Thread { id: 1, name: "main".into() }] })),
            "stackTrace" => {
                let mut frames: Vec<StackFrame> = Vec::new();
                let current_file = self.current_source().to_string_lossy().to_string();
                let source = Some(Source {
                    path: current_file.clone(),
                    name: None,
                });
                frames.push(StackFrame {
                    id: 0,
                    name: "main".into(),
                    source: source.clone(),
                    line: self.current_line() as i64,
                    column: 0,
                });
                for (i, (name, line)) in self.call_stack().iter().enumerate() {
                    frames.push(StackFrame {
                        id: (i + 1) as i64,
                        name: name.clone(),
                        source: source.clone(),
                        line: *line as i64,
                        column: 0,
                    });
                }
                Response::ok(0, req.seq, &req.command)
                    .with_body(json!({ "stackFrames": frames, "totalFrames": frames.len() as i64 }))
            }
            "scopes" => {
                let frame_id = req
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("frameId"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let scopes = vec![
                    Scope {
                        name: "Locals".into(),
                        presentation_hint: Some("locals".into()),
                        variables_reference: frame_id * 2 + 1,
                        expensive: false,
                    },
                    Scope {
                        name: "Globals".into(),
                        presentation_hint: Some("globals".into()),
                        variables_reference: frame_id * 2 + 2,
                        expensive: false,
                    },
                ];
                Response::ok(0, req.seq, &req.command).with_body(json!({ "scopes": scopes }))
            }
            "variables" => {
                let reference = req
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("variablesReference"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let is_global = reference % 2 == 0;
                let names = if is_global {
                    self.env.global_scope_names()
                } else {
                    self.env.current_scope_names()
                };
                let variables: Vec<Variable> = names
                    .iter()
                    .filter_map(|name| {
                        self.env.get(name).map(|v| {
                            let (value, type_name, child_count) = value_to_variable_parts(v);
                            let mut var = Variable {
                                name: name.clone(),
                                value,
                                type_field: Some(type_name),
                                variables_reference: 0,
                                named_variables: None,
                                indexed_variables: None,
                            };
                            if child_count > 0 {
                                var.variables_reference = next_variable_reference();
                            }
                            var
                        })
                    })
                    .collect();
                Response::ok(0, req.seq, &req.command).with_body(json!({ "variables": variables }))
            }
            "evaluate" => {
                match req
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("expression"))
                    .and_then(|v| v.as_str())
                {
                    Some(expr) => Response::ok(0, req.seq, &req.command)
                        .with_body(json!({ "result": expr, "variablesReference": 0 })),
                    None => Response::err(0, req.seq, &req.command, "missing expression"),
                }
            }
            "launch" | "attach" | "configurationDone" => Response::ok(0, req.seq, &req.command),
            _ => Response::ok(0, req.seq, &req.command),
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
            // Accept both List and Series as column values.
            let list: Vec<Value> = match val {
                Value::List(lst) => lst.as_ref().clone(),
                Value::Series(s) => s.values.clone(),
                _ => {
                    return Err(
                        self.type_err(format!("column '{col_name}' must be a list or series"))
                    )
                }
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
        let home = match std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
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
                    let _ = crate::lang::run_source_with_path(&src, self, Some(&path));
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
            // to be loaded only from ~/.hay/packages/ or exe dir.
            let restrict_to_packages = is_native_or_wasm && !cfg!(debug_assertions);

            // 1. Current directory
            if !restrict_to_packages && std::path::Path::new(cand).exists() {
                return Ok(cand.to_string());
            }

            // 2. Directory of the running executable (e.g. portable Windows installs)
            if let Ok(exe) = std::env::current_exe() {
                if let Some(exe_dir) = exe.parent() {
                    let p = exe_dir.join(cand);
                    if p.exists() {
                        return Ok(p.to_string_lossy().to_string());
                    }
                    // Also check exe_dir/plugins/ and exe_dir/.hay/plugins/
                    for sub in &["plugins", ".hay/plugins"] {
                        let p = exe_dir.join(sub).join(cand);
                        if p.exists() {
                            return Ok(p.to_string_lossy().to_string());
                        }
                    }
                }
            }

            // 3. ~/.hay/plugins/ (or %USERPROFILE%\.hay\plugins\ on Windows)
            if !restrict_to_packages {
                if let Some(home) =
                    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
                {
                    let plugin_path = std::path::Path::new(&home)
                        .join(".hay")
                        .join("plugins")
                        .join(cand);
                    if plugin_path.exists() {
                        return Ok(plugin_path.to_string_lossy().to_string());
                    }
                }
            }

            // 4. ~/.hay/packages/ (installed packages)
            if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
            {
                let pkg_path = std::path::Path::new(&home)
                    .join(".hay")
                    .join("packages")
                    .join(cand);
                if pkg_path.exists() {
                    return Ok(pkg_path.to_string_lossy().to_string());
                }
            }

            // 5. User-declared plugin_paths
            if !restrict_to_packages {
                for dir in &self.plugin_paths {
                    let p = std::path::Path::new(dir).join(cand);
                    if p.exists() {
                        return Ok(p.to_string_lossy().to_string());
                    }
                }
            }

            // 6. HAYASHI_PATH env var (colon or semicolon separated)
            if !restrict_to_packages {
                if let Ok(paths) = std::env::var("HAYASHI_PATH") {
                    let sep = if cfg!(windows) { ';' } else { ':' };
                    for dir in paths.split(sep) {
                        let p = std::path::Path::new(dir).join(cand);
                        if p.exists() {
                            return Ok(p.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        Err(HayashiError::Runtime(format!(
            "import: module '{}' not found (searched: ./, exe dir, ~/.hay/plugins/, ~/.hay/packages/, plugin_path, $HAYASHI_PATH)",
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
        df: &Arc<DataFrame>,
        opts: &[Opt],
    ) -> Result<Arc<DataFrame>> {
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
                        let rhs: Vec<RhsTerm> =
                            rhs_str.split('+').map(|t| RhsTerm::var(t.trim())).collect();
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
    ) -> Result<(Formula, Arc<DataFrame>)> {
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

    // ── Formula materialization ─────────────────────────────────────────────

    /// Atalho para o padrão `formula_to_string → GFormula::parse` que existia
    /// em todos os estimadores.  Agora usa `materialize_formula` internamente,
    /// então `log(K):log(L)`, `I(x^2)`, etc. funcionam corretamente.
    ///
    /// Retorna `(df_aumentado, g_formula, display_names)`.
    pub(super) fn prepare_formula(
        &mut self,
        formula: &Formula,
        df: &Arc<greeners::DataFrame>,
    ) -> Result<(Arc<greeners::DataFrame>, GFormula, Vec<String>)> {
        self.materialize_formula(formula, df)
    }

    /// Materializa os termos de uma fórmula em colunas concretas do DataFrame,
    /// retornando um DataFrame aumentado e uma `GFormula` com apenas nomes planos.
    ///
    /// Isso elimina a serialização `formula_to_string → GFormula::parse` que
    /// impedia `log(K):log(L)`, `I(x^2)`, etc. O Greeners recebe colunas já
    /// computadas; nunca precisa parsear strings como `"log(K)"`.
    ///
    /// # Nomes de exibição
    /// Retorna também um `Vec<String>` paralelo com os nomes legíveis
    /// (`"log(K)"`, `"I(experience^2)"`, …) para usar em `summary()` / `tidy()`.
    pub(super) fn materialize_formula(
        &mut self,
        formula: &Formula,
        df: &Arc<greeners::DataFrame>,
    ) -> Result<(Arc<greeners::DataFrame>, GFormula, Vec<String>)> {
        let mut augmented: greeners::DataFrame = df.as_ref().clone();
        let mut col_names: Vec<String> = Vec::new();
        let mut display_names: Vec<String> = Vec::new();
        let mut counter: usize = 0;

        for term in &formula.rhs {
            let (col, display) = self.materialize_term(term, df, &mut augmented, &mut counter)?;
            col_names.push(col);
            display_names.push(display);
        }

        let g_formula = GFormula {
            dependent: formula.lhs.clone(),
            independents: col_names,
            intercept: true,
        };

        Ok((Arc::new(augmented), g_formula, display_names))
    }

    /// Materializa um único `RhsTerm` em uma coluna do `augmented` DataFrame,
    /// retornando `(col_name, display_name)`.
    fn materialize_term(
        &mut self,
        term: &RhsTerm,
        original_df: &Arc<greeners::DataFrame>,
        augmented: &mut greeners::DataFrame,
        counter: &mut usize,
    ) -> Result<(String, String)> {
        match term {
            // Variável simples que já existe no df — sem cópia
            RhsTerm::Expr(e) if matches!(e.as_ref(), Expr::Var(_)) => {
                if let Expr::Var(v) = e.as_ref() {
                    return Ok((v.clone(), v.clone()));
                }
                unreachable!()
            }

            // C(Var(v)) simples — delega para o Greeners (ele já expande dummies)
            RhsTerm::Categorical(e) if matches!(e.as_ref(), Expr::Var(_)) => {
                if let Expr::Var(v) = e.as_ref() {
                    let col_name = format!("C({v})");
                    let display = col_name.clone();
                    return Ok((col_name, display));
                }
                unreachable!()
            }

            // Qualquer outra expressão: avalia element-wise e insere como coluna
            RhsTerm::Expr(e) => {
                let display = crate::lang::ast::expr_display(e);
                let col_name = format!("__term_{counter}");
                *counter += 1;
                let vals = self.eval_col_expr(e, original_df)?;
                augmented
                    .insert(col_name.clone(), ndarray::Array1::from(vals))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok((col_name, display))
            }

            // C(expr) composta: materializa a expr, depois envolve com C(...)
            RhsTerm::Categorical(e) => {
                let display = format!("C({})", crate::lang::ast::expr_display(e));
                let inner_col = format!("__cat_{counter}");
                *counter += 1;
                let vals = self.eval_col_expr(e, original_df)?;
                augmented
                    .insert(inner_col.clone(), ndarray::Array1::from(vals))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let col_name = format!("C({inner_col})");
                Ok((col_name, display))
            }

            // Interação: materializa ambos os lados e multiplica element-wise
            RhsTerm::Interaction(lhs, rhs) => {
                let (lcol, ldisp) = self.materialize_term(lhs, original_df, augmented, counter)?;
                let (rcol, rdisp) = self.materialize_term(rhs, original_df, augmented, counter)?;

                let col_name = format!("__inter_{counter}");
                *counter += 1;
                let display = format!("{ldisp}:{rdisp}");

                // Produto element-wise das duas colunas já materializadas
                // Usa get_column (que funciona tanto para colunas originais quanto
                // para colunas inseridas via insert())
                let lvals = augmented
                    .get_column(&lcol)
                    .map(|c| c.to_float().to_vec())
                    .map_err(|e| {
                        HayashiError::Runtime(format!("interaction: left column '{lcol}': {e}"))
                    })?;
                let rvals = augmented
                    .get_column(&rcol)
                    .map(|c| c.to_float().to_vec())
                    .map_err(|e| {
                        HayashiError::Runtime(format!("interaction: right column '{rcol}': {e}"))
                    })?;

                let prod: Vec<f64> = lvals.iter().zip(rvals.iter()).map(|(a, b)| a * b).collect();
                augmented
                    .insert(col_name.clone(), ndarray::Array1::from(prod))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok((col_name, display))
            }
        }
    }

    // ── Object methods ──────────────────────────────────────────────────────

    // ── Element-wise expression evaluation over DataFrame columns ───────────

    // ── Statement execution ─────────────────────────────────────────────────
}

pub(crate) fn value_to_variable_parts(v: &Value) -> (String, String, usize) {
    match v {
        Value::Float(f) => (format!("{f}"), "Float".into(), 0),
        Value::Int(i) => (format!("{i}"), "Int".into(), 0),
        Value::Bool(b) => (format!("{b}"), "Bool".into(), 0),
        Value::Str(s) => (s.clone(), "String".into(), 0),
        Value::Nil => ("nil".into(), "Nil".into(), 0),
        Value::DataFrame(df) => (
            format!(
                "DataFrame({} rows, {} cols)",
                df.n_rows(),
                df.column_names().len()
            ),
            "DataFrame".into(),
            df.column_names().len(),
        ),
        Value::List(lst) => (
            format!("List({} items)", lst.len()),
            "List".into(),
            lst.len().min(100),
        ),
        Value::Dict(d) => (
            format!("Dict({} entries)", d.len()),
            "Dict".into(),
            d.len().min(100),
        ),
        Value::Series(s) => (
            format!("Series({}: {} values)", s.name, s.len()),
            "Series".into(),
            s.len().min(100),
        ),
        Value::UserFn(f) => (
            format!("<fn({})>", f.params.join(", ")),
            "Function".into(),
            0,
        ),
        _ => (format!("{v}"), "Model".into(), 0),
    }
}

pub(crate) fn next_variable_reference() -> i64 {
    use std::sync::atomic::{AtomicI64, Ordering};
    static COUNTER: AtomicI64 = AtomicI64::new(1000);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}
