use super::*;
use std::sync::Arc;

impl Interpreter {
    pub(super) fn eval_call(&mut self, func: &str, args: &[Expr], opts: &[Opt]) -> Result<Value> {
        if let Some(pos) = func.find("::") {
            let ns = &func[..pos];
            let member = &func[pos + 2..];
            if self.plugins.contains_key(ns) {
                let mut evaluated_args = Vec::new();
                for arg in args {
                    evaluated_args.push(self.eval_expr(arg)?);
                }
                // Injeta __seed__ como argumento extra quando set_seed() foi chamado.
                // Plugins que declaram `seed: Option<Seed>` como último parâmetro
                // receberão a semente; os demais ignoram o argumento extra silenciosamente.
                if let Some(s) = self.rng_seed {
                    evaluated_args.push(Value::Int(s as i64));
                }
                let mut plugin = self
                    .plugins
                    .remove(ns)
                    .expect("plugin namespace verified by contains_key but missing on remove");
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
            .filter(|o| o.name != "if" && o.name != "dydx" && !is_mutate)
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

        // ── User-defined function ───────────────────────────────────────
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
                    "fn '{other}': expected at most {} argument(s), got {}",
                    user_fn.params.len(),
                    args.len()
                )));
            }

            // Evaluate arguments before mutating env
            let arg_vals: Vec<Value> = args
                .iter()
                .map(|e| self.eval_expr(e))
                .collect::<Result<_>>()?;

            let call_scope_depth = self.env.scope_count();
            self.call_stack
                .push((other.to_string(), self.current_line, call_scope_depth));
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

    // ── Panel helpers (depend on interpreter state) ───────────────────────

    pub(super) fn extract_panel_args(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<(Formula, Arc<DataFrame>, String, String)> {
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

    pub(super) fn get_time_col(
        &self,
        df_name: &str,
        opt_map: &HashMap<String, Value>,
    ) -> Result<String> {
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

    /// Extract a column as Vec<i64> — accepts Int or Float columns.
    pub(super) fn col_as_i64(
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

    // ── ASCII visualization helpers ─────────────────────────────────────────
}
