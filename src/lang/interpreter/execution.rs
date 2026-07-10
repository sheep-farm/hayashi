use super::*;

impl Interpreter {
    pub(crate) fn exec(&mut self, spanned: &Spanned) -> Result<()> {
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

            Stmt::Input {
                alias,
                headers,
                rows,
            } => self.exec_input(alias, headers, rows)?,

            // ── display expr ─────────────────────────────────────────────────
            Stmt::Display(expr) => {
                eprintln!("warning: 'display' is deprecated; use 'print' instead");
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

            Stmt::Load { path, alias, opts } => self.exec_load(path, alias, opts)?,

            Stmt::Predict {
                df,
                varname,
                model,
                kind,
            } => self.exec_predict(df, varname, model, kind)?,

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
            } => self.exec_replace(df, varname, expr, cond.as_ref())?,

            Stmt::Generate { df, varname, expr } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df}' is not a DataFrame"))),
                };
                let col_name = match self.eval_expr(varname)? {
                    Value::Str(s) => s,
                    other => {
                        return Err(self.type_err(format!(
                            "generate: column name must evaluate to a string, got {other}"
                        )))
                    }
                };
                let vals = self.eval_col_expr(expr, &df_val)?;
                let arr = ndarray::Array1::from(vals);
                Rc::make_mut(&mut df_val)
                    .insert(col_name.clone(), arr)
                    .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                emitln!(self, "({} obs)  {df}.{col_name} generated", df_val.n_rows());
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

            Stmt::Export { value, fmt, path } => self.exec_export(value, fmt, path)?,

            Stmt::Tsset { df, t_var } => {
                let frame = match self
                    .env
                    .get(df)
                    .ok_or_else(|| self.rt_err(format!("'{df}' not defined")))?
                {
                    Value::DataFrame(d) => d.clone(),
                    _ => return Err(self.type_err(format!("'{df}' is not a DataFrame"))),
                };

                let sorted = sort_df_by(&frame, t_var)?;
                let t_vals = self.eval_col_expr(&Expr::Var(t_var.clone()), &sorted)?;
                let t_min = t_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let t_max = t_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let n = sorted.n_rows();

                self.ts_info.insert(df.clone(), t_var.clone());
                self.env.set(df, Value::DataFrame(Rc::new(sorted)))?;

                println!("tsset {df}");
                println!("  time variable : {t_var}  ({t_min} to {t_max})");
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
                finally_body,
            } => self.exec_try_catch(try_body, error_var, catch_body, finally_body)?,

            // ── for var in iter { ... } ───────────────────────────────────────
            Stmt::For {
                var,
                var2,
                iter,
                body,
            } => self.exec_for(var, var2.as_deref(), iter, body)?,

            // ── fn name(params) { body } ─────────────────────────────────────
            Stmt::Fn {
                name,
                params,
                defaults,
                doc,
                body,
            } => {
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

    // ── input ────────────────────────────────────────────────────────────────

    fn exec_input(&mut self, alias: &str, headers: &[String], rows: &[Vec<f64>]) -> Result<()> {
        if headers.is_empty() {
            return Err(self.rt_err("input: no variables in header"));
        }
        if rows.is_empty() {
            return Err(self.rt_err("input: no data rows"));
        }
        let k = headers.len();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != k {
                return Err(HayashiError::Runtime(format!(
                    "input: row {} has {} values, expected {} ({})",
                    i + 1,
                    row.len(),
                    k,
                    headers.join(", ")
                )));
            }
        }
        let n = rows.len();
        let mut col_map: indexmap::IndexMap<String, ndarray::Array1<f64>> =
            indexmap::IndexMap::new();
        for (j, name) in headers.iter().enumerate() {
            let col: ndarray::Array1<f64> =
                ndarray::Array1::from(rows.iter().map(|r| r[j]).collect::<Vec<_>>());
            col_map.insert(name.clone(), col);
        }
        let df = greeners::DataFrame::new(col_map).map_err(|e| self.rt_err(e.to_string()))?;
        emitln!(
            self,
            "input → {alias} ({n} obs, {} vars: {})",
            k,
            headers.join(", ")
        );
        self.env.set(alias, Value::DataFrame(Rc::new(df)))?;
        Ok(())
    }

    // ── load ─────────────────────────────────────────────────────────────────

    fn exec_load(&mut self, path: &Expr, alias: &str, opts: &[Opt]) -> Result<()> {
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
            // ── File / URL ───────────────────────────────────────────
            #[cfg(feature = "native")]
            let _tmp;
            #[cfg(feature = "native")]
            let local_path: &str = if crate::io::fetch::is_url(&path_str) {
                emitln!(self, "Downloading '{}'…", path_str);
                _tmp = crate::io::fetch::download_to_temp(&path_str)?;
                _tmp.to_str()
                    .ok_or_else(|| self.rt_err("temp path is not UTF-8"))?
            } else {
                &path_str
            };
            #[cfg(not(feature = "native"))]
            let local_path: &str = &path_str;

            let ext = local_path.rsplit('.').next().unwrap_or("").to_lowercase();

            let (df, n_rows) = match ext.as_str() {
                #[cfg(feature = "native")]
                "dta" => crate::io::dta::load_dta(local_path)?,
                "xlsx" | "xls" | "ods" => {
                    crate::io::excel::load_excel(local_path, opt_sheet.as_deref())?
                }
                #[cfg(feature = "native")]
                "sqlite" | "sqlite3" | "db" => crate::io::sqlite::load_sqlite(
                    local_path,
                    opt_table.as_deref(),
                    opt_query.as_deref(),
                )?,
                "json" => {
                    let df =
                        DataFrame::from_json(local_path).map_err(|e| self.rt_err(e.to_string()))?;
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
        }
        Ok(())
    }

    // ── predict ──────────────────────────────────────────────────────────────

    fn exec_predict(&mut self, df: &str, varname: &str, model: &Expr, kind: &Expr) -> Result<()> {
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
            // "pr"   → P(Y = J) — probability of the highest category
            // "xb"   → linear predictor Xβ
            // "yhat" → predicted category (argmax)
            // "prN"  → P(Y = N) for a specific category N (1-indexed)
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
                                format!("predict Ordered: '{s}' — use prN where N is the category (1-indexed)")
                            ))?;
                        if cat == 0 || cat > r.n_categories {
                            return Err(HayashiError::Runtime(
                                format!("predict Ordered: category {cat} out of range 1..{}", r.n_categories)
                            ));
                        }
                        let probs = r.predict_proba(&x);
                        (0..probs.nrows()).map(|i| probs[[i, cat - 1]]).collect()
                    }
                    "pr" => {
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
                let names = r.count_var_names.as_deref().unwrap_or(&[]);
                let x_c = build_x_from_varnames(&df_val, names)?;
                let inflate_names = r.inflate_var_names.as_deref().unwrap_or(names);
                let x_i = build_x_from_varnames(&df_val, inflate_names)?;
                r.predict_count(&x_c, &x_i).to_vec()
            }
            (Value::ZeroInflatedResult(r), "pr0") => {
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
            // pr/mu/fitted → μ̂ = g⁻¹(Xβ) — predicted mean response
            // xb → Xβ — linear predictor (link scale)
            // residuals → deviance residuals
            // pearson → Pearson residuals (y-μ)/√V(μ)
            // working → IRLS working residuals
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
            // smoothed/yhat → smoothed values ŷ_i
            // residuals → residuals y_i - ŷ_i
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
            // pc1, pc2, ..., pcN → scores of the N-th principal component
            (Value::PcaResult(m), kind_s) => {
                if kind_s.starts_with("pc") && kind_s.len() > 2 {
                    let comp: usize = kind_s[2..].parse::<usize>()
                        .map_err(|_| HayashiError::Runtime(
                            format!("predict PCA: '{kind_s}' invalid — use pcN where N=1..{}", m.result.n_components)
                        ))?;
                    if comp == 0 || comp > m.result.n_components {
                        return Err(HayashiError::Runtime(
                            format!("predict PCA: component {comp} out of range 1..{}", m.result.n_components)
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
            // Use pca() for scores; factor() is only for loadings/structure analysis
            (Value::FactorResult(_), _) => return Err(HayashiError::Runtime(
                "predict Factor Analysis: scores not available via FA — use pca() for scores; FA is for loadings analysis".into()
            )),

            // ── Markov Switching ──────────────────────────────────────────
            // smoothed → most likely regime (1-indexed)
            // regimeN  → smoothed probability of regime N
            (Value::MarkovResult(r), "smoothed" | "regime" | "state") => {
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
                        format!("predict MarkovSwitching: '{kind_s}' invalid — use regimeN where N=1..{}", r.n_regimes)
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
            (Value::ConditionalResult(_), _) => return Err(HayashiError::Runtime(
                "predict clogit/cpoisson: fixed effects absorbed — unconditional prediction not available; use β̂ coefficients for odds ratios or marginal effects".into()
            )),

            // ── VARMA ─────────────────────────────────────────────────────
            (Value::VarmaResult(_), _) => return Err(HayashiError::Runtime(
                "predict varma: multivariate prediction not supported as a column — use print() for diagnostics".into()
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
                "predict gam: fitted values are not stored — use gam() with df=dataset and compute Xβ̂ manually".into()
            )),

            // ── MICE ──────────────────────────────────────────────────────
            (Value::MiceResult(_), _) => return Err(HayashiError::Runtime(
                "predict mice: MICE returns multiple datasets; access via model pooling".into()
            )),

            // ── SVAR ─────────────────────────────────────────────────────
            (Value::SVarResult(_), _) => return Err(HayashiError::Runtime(
                "predict svar: no fitted values — use sirf() and sfevd() for impulse-response analysis".into()
            )),

            // ── 3SLS ─────────────────────────────────────────────────────
            (Value::ThreeSLSResult(_), _) => return Err(HayashiError::Runtime(
                "predict 3sls: multiple equations — use print() to see coefficients per equation".into()
            )),

            // ── DFM ───────────────────────────────────────────────────────
            (Value::DFMResult(m), kind_s) if kind_s.starts_with('f') => {
                let idx = kind_s[1..].parse::<usize>()
                    .map(|n| n.saturating_sub(1))
                    .unwrap_or(0);
                if idx >= m.result.n_factors {
                    return Err(HayashiError::Runtime(format!(
                        "predict dfm: factor f{} does not exist — model has {} factors",
                        idx + 1, m.result.n_factors
                    )));
                }
                m.result.factors.column(idx).to_vec()
            }
            (Value::DFMResult(_), k) => return Err(HayashiError::Runtime(
                format!("predict dfm: kind '{k}' unknown — use: f1, f2, ... (1-based index of latent factor)")
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

            // ── Seasonal decomposition ───────────────────────────────────
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
                let idx = if kind_s == "seasonal" {
                    0usize
                } else {
                    kind_s["seasonal".len()..].parse::<usize>()
                        .map(|n| n.saturating_sub(1))
                        .unwrap_or(0)
                };
                if idx >= r.seasonal.len() {
                    return Err(HayashiError::Runtime(format!(
                        "predict mstl: seasonal{} component does not exist — model has {} periods",
                        idx + 1, r.seasonal.len()
                    )));
                }
                r.seasonal[idx].to_vec()
            }
            (Value::MstlResult(_), k) => return Err(HayashiError::Runtime(
                format!("predict mstl: kind '{k}' unknown — use: trend, resid, seasonal, seasonal1, seasonal2, ...")
            )),

            // ── ETS (exponential smoothing) ───────────────────────────
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
                format!("predict pthresh: kind '{k}' — use print() to see thresholds and coefficients")
            )),

            _ => return Err(HayashiError::Type(
                "predict: model type not supported".into()
            )),
        };

        let arr = ndarray::Array1::from(vals);
        Rc::make_mut(&mut df_val)
            .insert(varname.to_string(), arr)
            .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
        println!(
            "({} obs)  {df}.{varname} ({kind_str}) predicted",
            df_val.n_rows()
        );
        self.env.set(df, Value::DataFrame(df_val))?;
        Ok(())
    }

    // ── replace ──────────────────────────────────────────────────────────────

    fn exec_replace(
        &mut self,
        df: &str,
        varname: &str,
        expr: &Expr,
        cond: Option<&Expr>,
    ) -> Result<()> {
        let mut df_val = match self.env.get(df) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df}' is not a DataFrame"))),
        };
        let new_vals = self.eval_col_expr(expr, &df_val)?;

        let final_vals: Vec<f64> = if let Some(cond_expr) = cond {
            let mask = self.eval_col_expr(cond_expr, &df_val)?;
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
            .insert(varname.to_string(), arr)
            .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
        self.env.set(df, Value::DataFrame(df_val))?;
        Ok(())
    }

    // ── export ───────────────────────────────────────────────────────────────

    fn exec_export(&mut self, value: &Expr, fmt: &Expr, path: &Expr) -> Result<()> {
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
                #[cfg(feature = "native")]
                {
                    crate::io::sqlite::write_sqlite(&df, &path_str, "data")?;
                    println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                }
                #[cfg(not(feature = "native"))]
                return Err(self.rt_err("SQLite export requires 'native' feature"));
            }
            (Value::DataFrame(df), "parquet" | "pq") => {
                crate::io::parquet::write_parquet(&df, &path_str)?;
                println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
            }

            // ── OLS → CSV / LaTeX / HTML ──────────────────────────────
            (Value::OlsResult(m), "csv") => {
                let content = m.result.to_csv();
                std::fs::write(&path_str, &content).map_err(|e| HayashiError::Io(e.to_string()))?;
                println!("Exported OLS → '{path_str}'");
            }
            (Value::OlsResult(m), "latex" | "tex") => {
                let content = m.result.to_latex();
                std::fs::write(&path_str, &content).map_err(|e| HayashiError::Io(e.to_string()))?;
                println!("Exported OLS → '{path_str}'");
            }
            (Value::OlsResult(m), "html" | "htm") => {
                let content = m.result.to_html();
                std::fs::write(&path_str, &content).map_err(|e| HayashiError::Io(e.to_string()))?;
                println!("Exported OLS → '{path_str}'");
            }

            // ── Any model → txt ───────────────────────────────────
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
        Ok(())
    }

    // ── try/catch/finally ────────────────────────────────────────────────────

    fn exec_try_catch(
        &mut self,
        try_body: &[Spanned],
        error_var: &str,
        catch_body: &[Spanned],
        finally_body: &[Spanned],
    ) -> Result<()> {
        let run_block = |this: &mut Self, body: &[Spanned]| -> Result<()> {
            for s in body {
                this.exec(s)?;
            }
            Ok(())
        };

        self.env.push_scope();
        let mut try_result = Ok(());
        let mut caught: Option<ErrorValue> = None;
        for s in try_body {
            match self.exec(s) {
                Ok(()) => {}
                Err(HayashiError::Return | HayashiError::Break | HayashiError::Continue) => {
                    try_result = Err(HayashiError::Return);
                    break;
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
            let catch_result = (|| -> Result<()> {
                self.env.declare(error_var, Value::Error(Rc::new(err)))?;
                run_block(self, catch_body)
            })();
            self.env.pop_scope();
            if catch_result.is_err() {
                try_result = catch_result;
            }
        }

        if !finally_body.is_empty() {
            self.env.push_scope();
            let finally_result = run_block(self, finally_body);
            self.env.pop_scope();
            if finally_result.is_err() {
                try_result = finally_result;
            }
        }

        try_result
    }

    // ── for ──────────────────────────────────────────────────────────────────

    fn exec_for(
        &mut self,
        var: &str,
        var2: Option<&str>,
        iter: &ForIter,
        body: &[Spanned],
    ) -> Result<()> {
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
                            "for: range start must be integer, not {v}"
                        )))
                    }
                };
                let end = match self.eval_expr(end_expr)? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    v => {
                        return Err(HayashiError::Type(format!(
                            "for: range end must be integer, not {v}"
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
                            "for: range start must be integer, not {v}"
                        )))
                    }
                };
                let end = match self.eval_expr(end_expr)? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    v => {
                        return Err(HayashiError::Type(format!(
                            "for: range end must be integer, not {v}"
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
                let value = self.eval_expr(iter_expr)?;
                match value {
                    Value::List(v) => {
                        let items = (*v).clone();
                        if let Some(v2) = var2 {
                            for (i, item) in items.into_iter().enumerate() {
                                self.env.set(var, Value::Int(i as i64))?;
                                self.env.set(v2, item)?;
                                if run_body!() {
                                    break;
                                }
                            }
                        } else {
                            for item in items {
                                self.env.set(var, item)?;
                                if run_body!() {
                                    break;
                                }
                            }
                        }
                    }
                    Value::Dict(d) => {
                        let Some(v2) = var2 else {
                            return Err(HayashiError::Type(
                                "for: dict iteration requires two variables (for k, v in dict)"
                                    .into(),
                            ));
                        };
                        let items = (*d).clone();
                        for (k, v) in items {
                            self.env.set(var, Value::Str(k))?;
                            self.env.set(v2, v)?;
                            if run_body!() {
                                break;
                            }
                        }
                    }
                    other => {
                        return Err(HayashiError::Type(format!(
                            "for: iterator must be a list or dict, not {other}"
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}
