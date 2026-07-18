use super::*;
use std::sync::Arc;

/// Result of evaluating an expression element-wise over a DataFrame.
///
/// Most expressions yield numeric columns, but string-returning functions
/// (`substr`, `upper`, `lower`, `trim`, `str_replace`, `regexr`, `regexra`,
/// `regexs`) and user functions that return `Value::Str` produce `String`
/// columns. `generate`/`mutate` use this to insert typed columns.
pub(super) enum ColResult {
    Float(Vec<f64>),
    String(Vec<String>),
}

impl Interpreter {
    pub(super) fn eval_expr(&mut self, expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Float(v) => Ok(Value::Float(*v)),
            Expr::Int(v) => Ok(Value::Int(*v)),
            Expr::Bool(v) => Ok(Value::Bool(*v)),
            Expr::Str(v) => Ok(Value::Str(v.clone())),
            Expr::Nil => Ok(Value::Nil),

            Expr::FString(parts) => self.eval_fstring(parts),

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

            Expr::Closure { params, body } => Ok(Value::UserFn(Arc::new(UserFn {
                params: params.clone(),
                defaults: vec![None; params.len()],
                doc: None,
                body: vec![(Stmt::Return(Some(*body.clone())), 0)],
            }))),

            Expr::Apply { func, args } => self.eval_apply(func, args),

            Expr::Pipe { expr, .. } => self.eval_expr(expr),

            Expr::Match { expr, arms } => self.eval_match_expr(expr, arms),

            Expr::If {
                cond,
                then_expr,
                else_expr,
            } => self.eval_if_expr(cond, then_expr, else_expr),

            // ── Scalar arithmetic / logic ─────────────────────────────────────
            Expr::BinOp { op, lhs, rhs } => self.eval_binop_expr(op, lhs, rhs),

            Expr::Neg(inner) => match self.eval_expr(inner)? {
                Value::Int(v) => Ok(Value::Int(-v)),
                Value::Float(v) => Ok(Value::Float(-v)),
                _ => Err(HayashiError::Type("unary negation requires number".into())),
            },

            Expr::Not(inner) => {
                let v = self.eval_expr(inner)?;
                Ok(Value::Bool(!value_as_bool(&v)))
            }

            // ── List literal ────────────────────────────────────────────────
            Expr::List(items) => self.eval_list(items),

            // ── Dict literal ─────────────────────────────────────────────────
            Expr::Dict(pairs) => self.eval_dict(pairs),

            // ── Indexing: list[idx] or dict["key"] ───────────────────────────
            Expr::Index { obj, idx } => self.eval_index(obj, idx),

            Expr::Call { func, args, opts } => self.eval_call(func, args, opts),

            Expr::Field {
                obj,
                field,
                args,
                opts,
            } => self.eval_field(obj, field, args, opts),

            Expr::TsOp { .. } => Err(HayashiError::Runtime(
                "L./F./D. operators are only valid inside generate".into(),
            )),

            Expr::Range(start_expr, end_expr) => self.eval_range(start_expr, end_expr, false),

            Expr::RangeInclusive(start_expr, end_expr) => {
                self.eval_range(start_expr, end_expr, true)
            }

            Expr::Block(stmts, final_expr) => self.eval_block(stmts, final_expr),

            Expr::Quietly(inner) => self.eval_quietly(inner),

            Expr::ParallelFor {
                var,
                var2,
                iter,
                body,
                threads,
            } => self.eval_parallel_for_expr(var, var2, iter, body, threads),
        }
    }

    fn eval_fstring(&mut self, parts: &[FStringPart]) -> Result<Value> {
        let mut result = String::new();
        for part in parts {
            match part {
                FStringPart::Lit(s) => result.push_str(s),
                FStringPart::Interp { expr, fmt } => {
                    let val = self.eval_expr(expr)?;
                    match fmt {
                        None => result.push_str(&format!("{val}")),
                        Some(spec) => {
                            let num = match &val {
                                Value::Float(f) => *f,
                                Value::Int(i) => *i as f64,
                                _ => {
                                    result.push_str(&format!("{val}"));
                                    continue;
                                }
                            };
                            let formatted = match spec.as_str() {
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
                    }
                }
            }
        }
        Ok(Value::Str(result))
    }

    fn eval_apply(&mut self, func: &Expr, args: &[Expr]) -> Result<Value> {
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

    fn eval_match_expr(&mut self, expr: &Expr, arms: &[(Expr, Expr)]) -> Result<Value> {
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

    fn eval_if_expr(&mut self, cond: &Expr, then_expr: &Expr, else_expr: &Expr) -> Result<Value> {
        let cond_val = self.eval_expr(cond)?;
        if value_as_bool(&cond_val) {
            self.eval_expr(then_expr)
        } else {
            self.eval_expr(else_expr)
        }
    }

    fn eval_binop_expr(&mut self, op: &BinOp, lhs: &Expr, rhs: &Expr) -> Result<Value> {
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
                        return Err(
                            self.type_err("'in' requires list, dict, or string on right side")
                        )
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

    fn eval_dict(&mut self, pairs: &[(Expr, Expr)]) -> Result<Value> {
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
        Ok(Value::Dict(Arc::new(map)))
    }

    fn eval_index(&mut self, obj: &Expr, idx: &Expr) -> Result<Value> {
        let obj_val = self.eval_expr(obj)?;
        let idx_val = self.eval_expr(idx)?;
        match (&obj_val, &idx_val) {
            (Value::Dict(m), Value::Str(key)) => m
                .get(key)
                .cloned()
                .ok_or_else(|| HayashiError::Runtime(format!("key '{key}' not found in dict"))),
            (Value::Dict(_), _) => Err(HayashiError::Type("dict index must be a string".into())),
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
            (Value::Error(_), _) => Err(HayashiError::Type("error index must be a string".into())),
            (Value::DataFrame(df), Value::Str(key)) => {
                let col = df.get_column(key).map_err(|_| {
                    HayashiError::Runtime(format!("column '{key}' not found in DataFrame"))
                })?;
                use greeners::Column;
                let vals: Vec<Value> = match col {
                    Column::Float(arr) => arr.iter().map(|&x| Value::Float(x)).collect(),
                    Column::Int(arr) => arr.iter().map(|&x| Value::Int(x)).collect(),
                    Column::Bool(arr) => arr.iter().map(|&x| Value::Bool(x)).collect(),
                    Column::String(arr) => arr.iter().map(|s| Value::Str(s.clone())).collect(),
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
                Ok(Value::Series(Arc::new(Series::new(key.clone(), vals))))
            }
            (Value::DataFrame(_), _) => Err(HayashiError::Type(
                "DataFrame column index must be a string".into(),
            )),
            (Value::OlsResult(m), Value::Str(key)) => self.ols_field(m, key),
            (Value::OlsResult(_), _) => Err(HayashiError::Type(
                "OLS result index must be a string".into(),
            )),
            (Value::Series(s), _) => {
                let i = match idx_val {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(HayashiError::Type("series index must be integer".into())),
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
                    _ => return Err(HayashiError::Type("list index must be integer".into())),
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
            _ => Err(HayashiError::Type("indexing requires list or dict".into())),
        }
    }

    fn eval_block(&mut self, stmts: &[Stmt], final_expr: &Option<Box<Expr>>) -> Result<Value> {
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

    fn eval_quietly(&mut self, inner: &Expr) -> Result<Value> {
        eprintln!("warning: quietly() functional form is deprecated; use quietly on / quietly off");
        let old = self.env.quiet_mode;
        self.env.quiet_mode = true;
        let result = self.eval_expr(inner);
        self.env.quiet_mode = old;
        result
    }

    fn eval_range(&mut self, start: &Expr, end: &Expr, inclusive: bool) -> Result<Value> {
        let start = self.eval_as_int(start, "range start")?;
        let end = self.eval_as_int(end, "range end")?;
        let step: i64 = if start <= end { 1 } else { -1 };
        let mut v = Vec::new();
        let mut cur = start;
        if inclusive {
            while if step > 0 { cur <= end } else { cur >= end } {
                v.push(Value::Int(cur));
                cur += step;
            }
        } else {
            while if step > 0 { cur < end } else { cur > end } {
                v.push(Value::Int(cur));
                cur += step;
            }
        }
        Ok(Value::List(Arc::new(v)))
    }

    fn eval_parallel_for_expr(
        &mut self,
        var: &str,
        var2: &Option<String>,
        iter: &Expr,
        body: &[Spanned],
        threads: &Option<Box<Expr>>,
    ) -> Result<Value> {
        let n_threads = match threads {
            Some(e) => match self.eval_expr(e.as_ref())? {
                Value::Int(n) if n > 0 => Some(n as usize),
                Value::Float(f) if f > 0.0 => Some(f as usize),
                _ => None,
            },
            None => None,
        };
        // Reconstruct ForIter from the boxed Expr.
        let for_iter = match iter {
            Expr::Range(s, e) => ForIter::Range((**s).clone(), (**e).clone()),
            Expr::RangeInclusive(s, e) => ForIter::RangeInclusive((**s).clone(), (**e).clone()),
            other => ForIter::Items(other.clone()),
        };
        // Expression form: return the list, do NOT store in var.
        self.exec_parallel_for(var, var2.as_deref(), &for_iter, body, n_threads)
    }

    fn eval_list(&mut self, items: &[Expr]) -> Result<Value> {
        let vals: Vec<Value> = items
            .iter()
            .map(|e| self.eval_expr(e))
            .collect::<Result<_>>()?;
        Ok(Value::List(Arc::new(vals)))
    }

    // ── Convert AST formula → Greeners string ───────────────────────────────

    pub(super) fn formula_to_string(f: &Formula) -> String {
        let rhs_parts: Vec<String> = f.rhs.iter().map(|t| t.display_name()).collect();
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

    pub(super) fn eval_field(
        &mut self,
        obj: &Expr,
        field: &str,
        _args: &[Expr],
        _opts: &[Opt],
    ) -> Result<Value> {
        let val = self.eval_expr(obj)?;
        match (&val, field) {
            (Value::OlsResult(m), field) => self.ols_field(m, field),
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

    /// Returns a `Value` extracted from an OLS model result by field name.
    fn ols_field(&self, m: &super::models::OlsModel, field: &str) -> Result<Value> {
        use greeners::Column;
        use indexmap::IndexMap;

        let r = &m.result;
        let names = r.variable_names.clone().unwrap_or_default();

        let vec_to_series = |v: &[f64], name: &str| {
            let vals: Vec<Value> = v.iter().map(|&x| Value::Float(x)).collect();
            Value::Series(Arc::new(Series::new(name, vals)))
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

        match field {
            "params" | "coef" | "coefficients" => Ok(vec_to_dataframe(&r.params, "coef")),
            "std_errors" | "se" => Ok(vec_to_dataframe(&r.std_errors, "std_err")),
            "t_values" | "t" => Ok(vec_to_dataframe(&r.t_values, "t")),
            "p_values" | "p" => Ok(vec_to_dataframe(&r.p_values, "p_value")),
            "conf_lower" => Ok(vec_to_dataframe(&r.conf_lower, "conf_low")),
            "conf_upper" => Ok(vec_to_dataframe(&r.conf_upper, "conf_high")),
            "residuals" => Ok(vec_to_series(&m.residuals.to_vec(), "residuals")),
            "fitted" | "fitted_values" => {
                let fitted = m.x.dot(&r.params);
                Ok(vec_to_series(&fitted.to_vec(), "fitted"))
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
            "summary" => {
                println!("{}", m.result);
                Ok(Value::Nil)
            }
            _ => Err(HayashiError::Runtime(format!(
                "OLS result has no field '{field}'"
            ))),
        }
    }

    // ── Evaluate expression element-wise over DataFrame columns ────────────

    pub(super) fn eval_col_expr(&mut self, expr: &Expr, df: &DataFrame) -> Result<Vec<f64>> {
        match self.eval_col_expr_typed(expr, df)? {
            ColResult::Float(v) => Ok(v),
            ColResult::String(_) => Err(HayashiError::Type(
                "expression yields a string column; cannot be used as numeric in this context"
                    .to_string(),
            )),
        }
    }

    /// Like `eval_col_expr`, but preserves string-typed results.
    ///
    /// This is what `generate`/`mutate` use so that columns can be created from
    /// string functions such as `substr(date, 0, 7)` or `upper(name)`.
    pub(super) fn eval_col_expr_typed(&mut self, expr: &Expr, df: &DataFrame) -> Result<ColResult> {
        match expr {
            Expr::Float(_) | Expr::Int(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Nil => {
                self.eval_col_literal(expr, df)
            }
            Expr::Var(name) => self.eval_col_var(name, df),
            Expr::Neg(inner) => self.eval_col_neg(inner, df),
            Expr::Not(inner) => self.eval_col_not(inner, df),
            Expr::BinOp { op, lhs, rhs } => self.eval_col_binop(op, lhs, rhs, df),
            Expr::Call { func, args, .. } => self.eval_col_call(func, args, df),
            // ── time-series operators ───────────────────────────────────────
            // Requires the df to already be sorted by tsset.
            // L.x = x[i-n], F.x = x[i+n], D.x = x[i] - x[i-n]
            Expr::TsOp { op, var, n } => self.eval_col_tsop(op, var, *n, df),

            Expr::Apply { func, args } => self.eval_col_apply(func, args, df),

            _ => Err(HayashiError::Runtime(
                "expression type not supported in generate".into(),
            )),
        }
    }

    fn eval_col_literal(&mut self, expr: &Expr, df: &DataFrame) -> Result<ColResult> {
        let n = df.n_rows();
        match expr {
            Expr::Float(v) => Ok(ColResult::Float(vec![*v; n])),
            Expr::Int(v) => Ok(ColResult::Float(vec![*v as f64; n])),
            Expr::Bool(v) => Ok(ColResult::Float(vec![if *v { 1.0 } else { 0.0 }; n])),
            Expr::Str(s) => Ok(ColResult::String(vec![s.clone(); df.n_rows()])),
            Expr::Nil => Ok(ColResult::Float(vec![f64::NAN; n])),
            _ => Err(HayashiError::Runtime(
                "unsupported literal in generate".into(),
            )),
        }
    }

    fn eval_col_neg(&mut self, inner: &Expr, df: &DataFrame) -> Result<ColResult> {
        let vals = self.eval_col_expr(inner, df)?;
        Ok(ColResult::Float(vals.into_iter().map(|x| -x).collect()))
    }

    fn eval_col_not(&mut self, inner: &Expr, df: &DataFrame) -> Result<ColResult> {
        let vals = self.eval_col_expr(inner, df)?;
        Ok(ColResult::Float(
            vals.into_iter()
                .map(|x| if x == 0.0 { 1.0 } else { 0.0 })
                .collect(),
        ))
    }

    fn eval_col_run_userfn(
        &mut self,
        uf: Arc<UserFn>,
        vals: &[f64],
        mixed_fallback: bool,
    ) -> Result<ColResult> {
        let mut float_result: Vec<f64> = Vec::with_capacity(vals.len());
        let mut str_result: Vec<String> = Vec::with_capacity(vals.len());
        let mut is_str = false;
        let mut is_float = false;

        for &v in vals {
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

            match self.return_value.take().unwrap_or(Value::Float(f64::NAN)) {
                Value::Float(f) => {
                    float_result.push(f);
                    is_float = true;
                }
                Value::Int(i) => {
                    float_result.push(i as f64);
                    is_float = true;
                }
                Value::Str(s) => {
                    str_result.push(s);
                    is_str = true;
                }
                Value::Bool(b) => {
                    float_result.push(if b { 1.0 } else { 0.0 });
                    is_float = true;
                }
                _ => {
                    float_result.push(f64::NAN);
                    is_float = true;
                }
            }
        }

        if is_str && !is_float {
            Ok(ColResult::String(str_result))
        } else if is_float && !is_str {
            Ok(ColResult::Float(float_result))
        } else if mixed_fallback {
            let mut combined: Vec<String> = Vec::with_capacity(vals.len());
            for (i, v) in vals.iter().enumerate() {
                if i < float_result.len() && (i >= str_result.len() || str_result[i].is_empty()) {
                    combined.push(float_result[i].to_string());
                } else if i < str_result.len() {
                    combined.push(str_result[i].clone());
                } else {
                    combined.push(v.to_string());
                }
            }
            Ok(ColResult::String(combined))
        } else {
            Ok(ColResult::Float(float_result))
        }
    }

    fn eval_datetime_part(&self, part: &str, dt: chrono::NaiveDateTime) -> f64 {
        use chrono::{Datelike, Timelike};
        match part {
            "year" => dt.year() as f64,
            "month" => dt.month() as f64,
            "day" => dt.day() as f64,
            "hour" => dt.hour() as f64,
            "minute" => dt.minute() as f64,
            "second" => dt.second() as f64,
            "dow" => dt.weekday().num_days_from_monday() as f64,
            _ => f64::NAN,
        }
    }

    fn eval_col_call_string(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<Option<ColResult>> {
        self.eval_str_col_func(func, args, df)
    }

    fn eval_col_call_regex(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<Option<ColResult>> {
        if func == "regexm" && args.len() >= 2 {
            if let Expr::Var(col_name) = &args[0] {
                if let Ok(str_col) = df.get_string(col_name) {
                    let pattern = match &args[1] {
                        Expr::Str(s) => s.clone(),
                        _ => {
                            return Err(HayashiError::Type(
                                "regexm: pattern must be string literal".into(),
                            ))
                        }
                    };
                    return Ok(Some(ColResult::Float(greeners::Transforms::regexm_vec(
                        &str_col.to_vec(),
                        &pattern,
                    ))));
                }
            }
        }
        Ok(None)
    }

    fn eval_col_call_random(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<Option<ColResult>> {
        if !matches!(func, "uniform" | "runiform" | "rnormal" | "rbernoulli") {
            return Ok(None);
        }
        let n = df.n_rows();
        use rand::Rng;
        Ok(Some(ColResult::Float(match func {
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
                } else {
                    0.5
                };
                let rng = &mut self.rng;
                (0..n)
                    .map(|_| if rng.gen::<f64>() < p { 1.0 } else { 0.0 })
                    .collect()
            }
            _ => unreachable!(),
        })))
    }

    fn eval_col_call_rowwise(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<Option<ColResult>> {
        if !matches!(
            func,
            "rowmean" | "rowsum" | "rowmin" | "rowmax" | "rowtotal" | "rowmiss"
        ) {
            return Ok(None);
        }
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!(
                "{func}() requires at least one column"
            )));
        }
        let cols: Vec<Vec<f64>> = args
            .iter()
            .map(|a| self.eval_col_expr(a, df))
            .collect::<Result<_>>()?;
        Ok(Some(ColResult::Float(match func {
            "rowmean" => greeners::Transforms::row_mean(&cols),
            "rowsum" => greeners::Transforms::row_sum(&cols),
            "rowmin" => greeners::Transforms::row_min(&cols),
            "rowmax" => greeners::Transforms::row_max(&cols),
            "rowtotal" => greeners::Transforms::row_total(&cols),
            "rowmiss" => greeners::Transforms::row_miss(&cols),
            _ => unreachable!(),
        })))
    }

    fn eval_col_call_aggregation(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<Option<ColResult>> {
        if args.len() != 1 {
            return Ok(None);
        }
        match func {
            "rank" => {
                let vals = self.eval_col_expr(&args[0], df)?;
                Ok(Some(ColResult::Float(greeners::Transforms::rank(&vals))))
            }
            "cumsum" => {
                let vals = self.eval_col_expr(&args[0], df)?;
                Ok(Some(ColResult::Float(greeners::Transforms::cumsum(&vals))))
            }
            "std" | "standardize" | "zscore" => {
                let vals = self.eval_col_expr(&args[0], df)?;
                Ok(Some(ColResult::Float(greeners::Transforms::standardize(
                    &vals,
                ))))
            }
            "iqr" => {
                let vals = self.eval_col_expr(&args[0], df)?;
                let iqr_val = greeners::Transforms::iqr(&vals);
                Ok(Some(ColResult::Float(vec![iqr_val; df.n_rows()])))
            }
            _ => Ok(None),
        }
    }

    fn eval_col_call_datetime(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<Option<ColResult>> {
        if args.len() != 1 {
            return Ok(None);
        }

        if !matches!(
            func,
            "group" | "date" | "year" | "month" | "day" | "hour" | "minute" | "second" | "dow"
        ) {
            return Ok(None);
        }

        let col_name = match &args[0] {
            Expr::Var(name) => name.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires a column name"
                )))
            }
        };

        match func {
            "group" => {
                let strs = col_to_strings(df, &col_name)?;
                Ok(Some(ColResult::Float(greeners::Transforms::group(&strs))))
            }
            "date" => {
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
                Ok(Some(ColResult::Float(result)))
            }
            "year" | "month" | "day" | "hour" | "minute" | "second" | "dow" => {
                if let Ok(arr) = df.get_datetime(&col_name) {
                    let result: Vec<f64> = arr
                        .iter()
                        .map(|dt| self.eval_datetime_part(func, *dt))
                        .collect();
                    return Ok(Some(ColResult::Float(result)));
                }
                let vals = self.eval_col_expr(&args[0], df)?;
                use chrono::DateTime as ChronoDateTime;
                let result: Vec<f64> = vals
                    .iter()
                    .map(|&ts| {
                        ChronoDateTime::from_timestamp(ts as i64, 0)
                            .map(|d| self.eval_datetime_part(func, d.naive_utc()))
                            .unwrap_or(f64::NAN)
                    })
                    .collect();
                Ok(Some(ColResult::Float(result)))
            }
            _ => Ok(None),
        }
    }

    fn eval_col_call_single_arg(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<Option<ColResult>> {
        if args.len() != 1 {
            return Ok(None);
        }
        if let Some(res) = self.eval_col_call_aggregation(func, args, df)? {
            return Ok(Some(res));
        }
        self.eval_col_call_datetime(func, args, df)
    }

    fn eval_col_call_elementwise_one_arg(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<ColResult> {
        let vals = self.eval_col_expr(&args[0], df)?;
        match greeners::Transforms::apply(&vals, func) {
            Ok(result) => Ok(ColResult::Float(result)),
            Err(_) => {
                if let Some(Value::UserFn(uf)) = self.env.get(func).cloned() {
                    self.eval_col_run_userfn(uf, &vals, true)
                } else {
                    Err(HayashiError::Runtime(format!(
                        "unknown column function '{func}'"
                    )))
                }
            }
        }
    }

    fn eval_col_call_two_arg(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<ColResult> {
        let a = self.eval_col_expr(&args[0], df)?;
        let b = self.eval_col_expr(&args[1], df)?;
        match greeners::Transforms::apply2(&a, &b, func) {
            Ok(result) => Ok(ColResult::Float(result)),
            Err(_) => Err(HayashiError::Runtime(format!(
                "function '{func}' not supported in generate"
            ))),
        }
    }

    fn eval_col_call_three_arg(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<ColResult> {
        let a = self.eval_col_expr(&args[0], df)?;
        let b = self.eval_col_expr(&args[1], df)?;
        let c = self.eval_col_expr(&args[2], df)?;
        match greeners::Transforms::apply3(&a, &b, &c, func) {
            Ok(result) => Ok(ColResult::Float(result)),
            Err(_) => Err(HayashiError::Runtime(format!(
                "function '{func}' not supported in generate"
            ))),
        }
    }

    fn eval_col_call(&mut self, func: &str, args: &[Expr], df: &DataFrame) -> Result<ColResult> {
        if let Some(str_res) = self.eval_col_call_string(func, args, df)? {
            return Ok(str_res);
        }
        if let Some(res) = self.eval_col_call_regex(func, args, df)? {
            return Ok(res);
        }
        if let Some(res) = self.eval_col_call_random(func, args, df)? {
            return Ok(res);
        }
        if let Some(res) = self.eval_col_call_rowwise(func, args, df)? {
            return Ok(res);
        }
        if let Some(res) = self.eval_col_call_single_arg(func, args, df)? {
            return Ok(res);
        }

        match args.len() {
            1 => self.eval_col_call_elementwise_one_arg(func, args, df),
            2 => self.eval_col_call_two_arg(func, args, df),
            3 => self.eval_col_call_three_arg(func, args, df),
            _ => Err(HayashiError::Runtime(format!(
                "function '{func}' not supported in generate"
            ))),
        }
    }

    fn eval_col_binop(
        &mut self,
        op: &BinOp,
        lhs: &Expr,
        rhs: &Expr,
        df: &DataFrame,
    ) -> Result<ColResult> {
        // String column equality/inequality: col == "literal" or "literal" == col
        if matches!(op, BinOp::Eq | BinOp::Ne) {
            let str_pair = match (lhs, rhs) {
                (Expr::Var(c), Expr::Str(t)) => Some((c.as_str(), t.as_str())),
                (Expr::Str(t), Expr::Var(c)) => Some((c.as_str(), t.as_str())),
                _ => None,
            };
            if let Some((col_name, target)) = str_pair {
                let is_eq = matches!(op, BinOp::Eq);
                if let Ok(col) = df.get_column(col_name) {
                    use greeners::Column;
                    let maybe: Option<Vec<f64>> = match col {
                        Column::String(arr) => Some(
                            arr.iter()
                                .map(|s| {
                                    if (s.as_str() == target) == is_eq {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                })
                                .collect(),
                        ),
                        Column::Categorical(cat) => Some(
                            cat.to_strings()
                                .iter()
                                .map(|s| {
                                    if (s.as_str() == target) == is_eq {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                })
                                .collect(),
                        ),
                        _ => None,
                    };
                    if let Some(v) = maybe {
                        return Ok(ColResult::Float(v));
                    }
                }
            }
        }
        let l = self.eval_col_expr(lhs, df)?;
        let r = self.eval_col_expr(rhs, df)?;
        if l.len() != r.len() {
            return Err(HayashiError::Runtime("mismatched column lengths".into()));
        }
        Ok(ColResult::Float(
            l.into_iter()
                .zip(r)
                .map(|(a, b)| match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Mod => a % b,
                    BinOp::Pow => a.powf(b),
                    BinOp::Gt => {
                        if a > b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Lt => {
                        if a < b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::GtEq => {
                        if a >= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::LtEq => {
                        if a <= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Eq => {
                        if (a - b).abs() < f64::EPSILON {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Ne => {
                        if (a - b).abs() >= f64::EPSILON {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::And => {
                        if a != 0.0 && b != 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Or => {
                        if a != 0.0 || b != 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::In => 0.0,
                })
                .collect(),
        ))
    }

    fn eval_col_var(&mut self, name: &str, df: &DataFrame) -> Result<ColResult> {
        // _n = row number (1-based), _N = total rows
        if name == "_n" {
            return Ok(ColResult::Float(
                (1..=df.n_rows()).map(|i| i as f64).collect(),
            ));
        }
        if name == "_N" {
            return Ok(ColResult::Float(vec![df.n_rows() as f64; df.n_rows()]));
        }
        match df.get_column(name) {
            Ok(col) => {
                use greeners::Column;
                match col {
                    Column::String(arr) => Ok(ColResult::String(arr.to_vec())),
                    Column::Categorical(cat) => {
                        let strs: Vec<String> = (0..df.n_rows())
                            .map(|r| cat.get_string(r).unwrap_or("").to_string())
                            .collect();
                        Ok(ColResult::String(strs))
                    }
                    other => Ok(ColResult::Float(other.to_float().to_vec())),
                }
            }
            Err(_) => match self.env.get(name) {
                Some(Value::Float(f)) => Ok(ColResult::Float(vec![*f; df.n_rows()])),
                Some(Value::Int(i)) => Ok(ColResult::Float(vec![*i as f64; df.n_rows()])),
                Some(Value::Bool(b)) => Ok(ColResult::Float(vec![
                    if *b { 1.0 } else { 0.0 };
                    df.n_rows()
                ])),
                Some(Value::Str(s)) => Ok(ColResult::String(vec![s.clone(); df.n_rows()])),
                Some(Value::List(lst)) => {
                    if lst.len() != df.n_rows() {
                        return Err(HayashiError::Runtime(format!(
                            "list variable '{name}' has length {}, expected {}",
                            lst.len(),
                            df.n_rows()
                        )));
                    }
                    // Heuristic: if every element is Str, treat as string column.
                    let mut all_str = true;
                    for v in lst.iter() {
                        if !matches!(v, Value::Str(_)) {
                            all_str = false;
                            break;
                        }
                    }
                    if all_str {
                        let mut data = Vec::with_capacity(lst.len());
                        for v in lst.iter() {
                            if let Value::Str(s) = v {
                                data.push(s.clone());
                            }
                        }
                        Ok(ColResult::String(data))
                    } else {
                        let mut data = Vec::with_capacity(lst.len());
                        for v in lst.iter() {
                            match v {
                                Value::Float(f) => data.push(*f),
                                Value::Int(i_val) => data.push(*i_val as f64),
                                Value::Bool(b) => data.push(if *b { 1.0 } else { 0.0 }),
                                other => {
                                    return Err(HayashiError::Type(format!(
                                        "element in list variable '{name}' is not numeric: {other}"
                                    )))
                                }
                            }
                        }
                        Ok(ColResult::Float(data))
                    }
                }
                _ => Err(HayashiError::Runtime(format!(
                    "'{name}' not found as column or scalar variable"
                ))),
            },
        }
    }

    fn eval_col_tsop(
        &mut self,
        op: &TsOpKind,
        var: &str,
        n: usize,
        df: &DataFrame,
    ) -> Result<ColResult> {
        use greeners::Column;
        let col = df
            .get_column(var)
            .map_err(|_| HayashiError::Runtime(format!("column '{var}' not found")))?;
        let vals: Vec<f64> = match col {
            Column::Float(arr) => arr.to_vec(),
            Column::Int(arr) => arr.iter().map(|&x| x as f64).collect(),
            _ => return Err(HayashiError::Type(format!("column '{var}' is not numeric"))),
        };
        let len = vals.len();
        Ok(ColResult::Float(match op {
            TsOpKind::Lag => (0..len)
                .map(|i| if i >= n { vals[i - n] } else { f64::NAN })
                .collect(),
            TsOpKind::Lead => (0..len)
                .map(|i| if i + n < len { vals[i + n] } else { f64::NAN })
                .collect(),
            TsOpKind::Diff => (0..len)
                .map(|i| {
                    if i >= n {
                        vals[i] - vals[i - n]
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        }))
    }

    fn eval_col_apply(&mut self, func: &Expr, args: &[Expr], df: &DataFrame) -> Result<ColResult> {
        let closure_val = self.eval_expr(func)?;
        let uf = match closure_val {
            Value::UserFn(f) => f,
            _ => {
                return Err(HayashiError::Runtime(
                    "generate: pipe target must be a function or closure".into(),
                ))
            }
        };
        let vals = self.eval_col_expr(&args[0], df)?;
        self.eval_col_run_userfn(uf, &vals, false)
    }

    /// Tries to evaluate a `Call` expression as a string-column function
    /// (`substr`, `upper`, `lower`, `trim`, `str_replace`, `regexr`,
    /// `regexra`, `regexs`). Returns `Ok(None)` when `func` is not one of
    /// these so the caller can fall back to the numeric path.
    fn eval_str_col_func(
        &mut self,
        func: &str,
        args: &[Expr],
        df: &DataFrame,
    ) -> Result<Option<ColResult>> {
        match func {
            "substr" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime(
                        "substr(s, start [, length]) requires 2 or 3 arguments".into(),
                    ));
                }
                let strs = self.eval_str_col(&args[0], df)?;
                let start = self.eval_int_scalar(&args[1], "substr: start")?;
                let count = if args.len() == 3 {
                    self.eval_int_scalar(&args[2], "substr: length")?.max(0) as usize
                } else {
                    usize::MAX
                };
                let out: Vec<String> = strs
                    .iter()
                    .map(|s| {
                        let chars: Vec<char> = s.chars().collect();
                        let len = chars.len() as i64;
                        let real_start =
                            (if start < 0 { len + start } else { start }).clamp(0, len) as usize;
                        let end = (real_start + count).min(chars.len());
                        chars[real_start..end].iter().collect()
                    })
                    .collect();
                Ok(Some(ColResult::String(out)))
            }
            "upper" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("upper(s) requires 1 argument".into()));
                }
                let strs = self.eval_str_col(&args[0], df)?;
                Ok(Some(ColResult::String(
                    strs.iter().map(|s| s.to_uppercase()).collect(),
                )))
            }
            "lower" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("lower(s) requires 1 argument".into()));
                }
                let strs = self.eval_str_col(&args[0], df)?;
                Ok(Some(ColResult::String(
                    strs.iter().map(|s| s.to_lowercase()).collect(),
                )))
            }
            "trim" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("trim(s) requires 1 argument".into()));
                }
                let strs = self.eval_str_col(&args[0], df)?;
                Ok(Some(ColResult::String(
                    strs.iter().map(|s| s.trim().to_string()).collect(),
                )))
            }
            "str_replace" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "str_replace(s, from, to) requires 3 arguments".into(),
                    ));
                }
                let strs = self.eval_str_col(&args[0], df)?;
                let from = self.eval_str_scalar(&args[1], "str_replace: from")?;
                let to = self.eval_str_scalar(&args[2], "str_replace: to")?;
                Ok(Some(ColResult::String(
                    strs.iter().map(|s| s.replace(&from, &to)).collect(),
                )))
            }
            "regexr" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "regexr(s, pattern, replacement) requires 3 arguments".into(),
                    ));
                }
                let strs = self.eval_str_col(&args[0], df)?;
                let pattern = self.eval_str_scalar(&args[1], "regexr: pattern")?;
                let replacement = self.eval_str_scalar(&args[2], "regexr: replacement")?;
                Ok(Some(ColResult::String(greeners::Transforms::regexr_vec(
                    &strs,
                    &pattern,
                    &replacement,
                ))))
            }
            "regexra" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "regexra(s, pattern, replacement) requires 3 arguments".into(),
                    ));
                }
                let strs = self.eval_str_col(&args[0], df)?;
                let pattern = self.eval_str_scalar(&args[1], "regexra: pattern")?;
                let replacement = self.eval_str_scalar(&args[2], "regexra: replacement")?;
                Ok(Some(ColResult::String(
                    strs.iter()
                        .map(|s| greeners::Transforms::regexra(s, &pattern, &replacement))
                        .collect(),
                )))
            }
            "regexs" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "regexs(s, pattern) requires 2 arguments".into(),
                    ));
                }
                let strs = self.eval_str_col(&args[0], df)?;
                let pattern = self.eval_str_scalar(&args[1], "regexs: pattern")?;
                Ok(Some(ColResult::String(greeners::Transforms::regexs_vec(
                    &strs, &pattern,
                ))))
            }
            _ => Ok(None),
        }
    }

    /// Resolves an expression as a column of strings (broadcasting string
    /// literals and scalar string variables).
    fn eval_str_col(&mut self, expr: &Expr, df: &DataFrame) -> Result<Vec<String>> {
        match self.eval_col_expr_typed(expr, df)? {
            ColResult::String(v) => Ok(v),
            ColResult::Float(v) => {
                // Coerce numeric to strings (Stata-style: integers without
                // decimals, NaN as ".").
                Ok(v.iter()
                    .map(|x| {
                        if x.is_nan() {
                            ".".to_string()
                        } else if x.fract() == 0.0 && x.abs() < 1e14 {
                            format!("{}", *x as i64)
                        } else {
                            format!("{:.4}", x)
                        }
                    })
                    .collect())
            }
        }
    }

    /// Resolves an expression as a single string (literal or scalar var).
    fn eval_str_scalar(&mut self, expr: &Expr, ctx: &str) -> Result<String> {
        match self.eval_expr(expr)? {
            Value::Str(s) => Ok(s),
            other => Err(HayashiError::Type(format!(
                "{ctx}: expected string, got {other}"
            ))),
        }
    }

    /// Resolves an expression as a single integer (literal or scalar var).
    fn eval_int_scalar(&mut self, expr: &Expr, ctx: &str) -> Result<i64> {
        match self.eval_expr(expr)? {
            Value::Int(i) => Ok(i),
            Value::Float(f) => Ok(f as i64),
            other => Err(HayashiError::Type(format!(
                "{ctx}: expected integer, got {other}"
            ))),
        }
    }
}
