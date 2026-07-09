use super::*;

impl Interpreter {
    pub(super) fn eval_expr(&mut self, expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Float(v) => Ok(Value::Float(*v)),
            Expr::Int(v) => Ok(Value::Int(*v)),
            Expr::Bool(v) => Ok(Value::Bool(*v)),
            Expr::Str(v) => Ok(Value::Str(v.clone())),
            Expr::Nil => Ok(Value::Nil),

            Expr::FString(parts) => {
                // Parts were parsed at parse-time; no re-lexing or re-parsing here.
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
                                            let prec: usize =
                                                s[1..s.len() - 1].parse().unwrap_or(2);
                                            format!("{num:.prec$}")
                                        }
                                        s if s.starts_with('.') && s.ends_with('e') => {
                                            let prec: usize =
                                                s[1..s.len() - 1].parse().unwrap_or(2);
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

            // ── Scalar arithmetic / logic ─────────────────────────────────────
            Expr::BinOp { op, lhs, rhs } => {
                // Short-circuit for And/Or
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
                _ => Err(HayashiError::Type("unary negation requires number".into())),
            },

            Expr::Not(inner) => {
                let v = self.eval_expr(inner)?;
                Ok(Value::Bool(!value_as_bool(&v)))
            }

            // ── List literal ────────────────────────────────────────────────
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

            // ── Indexing: list[idx] or dict["key"] ───────────────────────────
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
                                return Err(HayashiError::Type(
                                    "series index must be integer".into(),
                                ))
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
                    _ => Err(HayashiError::Type("indexing requires list or dict".into())),
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
                "L./F./D. operators are only valid inside generate".into(),
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

    // ── Evaluate expression element-wise over DataFrame columns ────────────

    pub(super) fn eval_col_expr(&mut self, expr: &Expr, df: &DataFrame) -> Result<Vec<f64>> {
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
                    "string literal \"{s}\" cannot be used as numeric — when comparing to a string column, use: col == \"{s}\""
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
                // ── regex row-wise over string columns ──
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

                // ── random generators (size = df n_rows) ──
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

                // ── multi-column functions (rowmean / rowsum / rowmin / rowmax / rowtotal / rowmiss) ──
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
                    // ── functions that need the whole column ─────────────────
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

                    // ── element-wise scalar functions (1-arg) ───────────────
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
                                    format!("unknown column function '{func}'")
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
                            format!("function '{func}' not supported in generate")
                        )),
                    }
                } else if args.len() == 3 {
                    let a = self.eval_col_expr(&args[0], df)?;
                    let b = self.eval_col_expr(&args[1], df)?;
                    let c = self.eval_col_expr(&args[2], df)?;
                    match greeners::Transforms::apply3(&a, &b, &c, func) {
                        Ok(result) => Ok(result),
                        Err(_) => Err(HayashiError::Runtime(
                            format!("function '{func}' not supported in generate")
                        )),
                    }
                } else {
                    Err(HayashiError::Runtime(format!(
                        "function '{func}' not supported in generate"
                    )))
                }
            }
            // ── time-series operators ───────────────────────────────────────
            // Requires the df to already be sorted by tsset.
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
}
