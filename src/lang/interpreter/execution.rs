use super::eval_expr::ColResult;
use super::*;
use std::sync::Arc;

/// Wrapper for thread results in `parallel for` — values created inside a
/// child thread are moved to the parent thread.  This is safe because:
/// - Send-safe values (Arc, primitives) are genuinely Send.
/// - Rc-backed values are created within the child thread, have refcount 1,
///   and are moved (never cloned) to the parent.  No concurrent refcount
///   access occurs.
struct ThreadResult(std::result::Result<Value, HayashiError>);

// SAFETY: see struct-level comment.
unsafe impl Send for ThreadResult {}

struct LoadOptions {
    sheet: Option<String>,
    #[cfg_attr(not(feature = "sqlite"), allow(dead_code))]
    table: Option<String>,
    query: Option<String>,
    sep: Option<String>,
    columns: Option<Vec<String>>,
    predicate: Option<crate::lang::predicate::RowPredicate>,
}

impl Interpreter {
    pub(crate) fn exec(&mut self, spanned: &Spanned) -> Result<()> {
        let (stmt, line) = spanned;
        self.current_line = *line;
        let file = self.current_source.clone();
        self.debug_check(&file, *line)?;
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
                let col_result = self.eval_col_expr_typed(expr, &df_val)?;
                match col_result {
                    ColResult::Float(vals) => {
                        let arr = ndarray::Array1::from(vals);
                        Arc::make_mut(&mut df_val)
                            .insert(col_name.clone(), arr)
                            .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                    }
                    ColResult::String(strs) => {
                        use greeners::Column;
                        let col = Column::String(ndarray::Array1::from(strs));
                        Arc::make_mut(&mut df_val)
                            .insert_column(col_name.clone(), col)
                            .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                    }
                }
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
                let mut output = String::new();
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        output.push_str(&sep);
                    }
                    let val = self.eval_expr(expr)?;
                    output.push_str(&format!("{val}"));
                }
                output.push_str(&end);
                self.print_output(output);
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
                self.env.set(df, Value::DataFrame(Arc::new(sorted)))?;

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

            // ── parallel for var in iter { ... } ──────────────────────────────
            Stmt::ParallelFor {
                var,
                var2,
                iter,
                body,
                threads,
            } => {
                let n_threads = match threads {
                    Some(e) => match self.eval_expr(e)? {
                        Value::Int(n) if n > 0 => Some(n as usize),
                        Value::Float(f) if f > 0.0 => Some(f as usize),
                        _ => None,
                    },
                    None => None,
                };
                let result = self.exec_parallel_for(var, var2.as_deref(), iter, body, n_threads)?;
                self.env.set(var, result)?;
            }

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
                    Value::UserFn(Arc::new(UserFn {
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
                    self.last_expr_value = Some(Value::Nil);
                } else {
                    let val = self.eval_expr(expr)?;
                    self.last_expr_value = Some(val.clone());
                    if self.auto_display && !matches!(val, Value::Nil) {
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
        self.env.set(alias, Value::DataFrame(Arc::new(df)))?;
        Ok(())
    }

    fn parse_load_options(&mut self, opts: &[Opt]) -> Result<LoadOptions> {
        let mut opt_sheet: Option<String> = None;
        let mut opt_table: Option<String> = None;
        let mut opt_query: Option<String> = None;
        let mut opt_sep: Option<String> = None;
        let mut opt_columns: Option<Vec<String>> = None;
        let mut opt_where: Option<String> = None;
        for o in opts {
            if o.name == "columns" {
                opt_columns = Some(extract_column_names(&o.value)?);
                continue;
            }
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
                "where" => opt_where = Some(val),
                k => {
                    return Err(HayashiError::Runtime(format!(
                        "load: unknown option '{k}' — use: sheet, table, query, sep, columns, where"
                    )))
                }
            }
        }
        let predicate = match opt_where.as_deref() {
            Some(s) => Some(crate::lang::predicate::RowPredicate::parse(s)?),
            None => None,
        };
        Ok(LoadOptions {
            sheet: opt_sheet,
            table: opt_table,
            query: opt_query,
            sep: opt_sep,
            columns: opt_columns,
            predicate,
        })
    }

    fn exec_load_odbc(&mut self, path_str: &str, alias: &str, options: &LoadOptions) -> Result<()> {
        #[cfg(feature = "odbc")]
        {
            let conn_str = &path_str["odbc://".len()..];
            let sql = if let Some(t) = &options.table {
                let cols = options
                    .columns
                    .as_deref()
                    .map(|c| c.join(", "))
                    .unwrap_or_else(|| "*".to_string());
                let mut s = format!("SELECT {cols} FROM \"{t}\"");
                if let Some(p) = &options.predicate {
                    s.push_str(&format!(" WHERE {}", p.to_sql()));
                }
                s
            } else {
                return Err(HayashiError::Runtime(
                    "load odbc: requires table= (with optional columns=/where=) or query=".into(),
                ));
            };
            let (df, n_rows) = crate::io::odbc::load_odbc(conn_str, &sql)?;
            emitln!(self, "Loaded ODBC → {alias} ({n_rows} rows)");
            self.env.set(alias, Value::DataFrame(Arc::new(df)))?;
            Ok(())
        }
        #[cfg(not(feature = "odbc"))]
        {
            let _ = (path_str, alias, options);
            Err(HayashiError::Runtime(
                "ODBC support not enabled. Rebuild with: cargo build --features odbc\n\
                 Requires: unixodbc (pacman -S unixodbc)"
                    .into(),
            ))
        }
    }

    fn exec_load_file(&mut self, path_str: &str, alias: &str, options: &LoadOptions) -> Result<()> {
        // ── File / URL ───────────────────────────────────────────
        #[cfg(feature = "network")]
        let _tmp;
        #[cfg(feature = "network")]
        let local_path: &str = if crate::io::fetch::is_url(path_str) {
            emitln!(self, "Downloading '{}'…", path_str);
            _tmp = crate::io::fetch::download_to_temp(path_str)?;
            _tmp.to_str()
                .ok_or_else(|| self.rt_err("temp path is not UTF-8"))?
        } else {
            path_str
        };
        #[cfg(not(feature = "network"))]
        let local_path: &str = path_str;

        let ext = local_path.rsplit('.').next().unwrap_or("").to_lowercase();

        let (df, n_rows) = self.load_from_extension(local_path, &ext, options)?;
        emitln!(self, "Loaded '{}' → {alias} ({} rows)", path_str, n_rows);
        self.env.set(alias, Value::DataFrame(Arc::new(df)))?;
        Ok(())
    }

    fn load_from_extension(
        &mut self,
        local_path: &str,
        ext: &str,
        options: &LoadOptions,
    ) -> Result<(DataFrame, usize)> {
        let opt_columns = options.columns.as_deref();
        let predicate = options.predicate.as_ref();

        if ext == "json" && (opt_columns.is_some() || predicate.is_some()) {
            return Err(HayashiError::Runtime(
                "load: columns=/where= not yet supported for JSON \
                 (use a different format or post-load filter/keep)"
                    .into(),
            ));
        }

        match ext {
            #[cfg(feature = "dta")]
            "dta" => crate::io::dta::load_dta(local_path, opt_columns, predicate),
            "xlsx" | "xls" | "ods" => crate::io::excel::load_excel(
                local_path,
                options.sheet.as_deref(),
                opt_columns,
                predicate,
            ),
            #[cfg(feature = "sqlite")]
            "sqlite" | "sqlite3" | "db" => crate::io::sqlite::load_sqlite(
                local_path,
                options.table.as_deref(),
                options.query.as_deref(),
                opt_columns,
                predicate,
            ),
            "json" => {
                let df =
                    DataFrame::from_json(local_path).map_err(|e| self.rt_err(e.to_string()))?;
                let n = df.n_rows();
                Ok((df, n))
            }
            "tsv" | "tab" => crate::io::dsv::load_dsv(local_path, b'\t', opt_columns, predicate),
            "parquet" | "pq" => {
                crate::io::parquet::load_parquet(local_path, opt_columns, predicate)
            }
            _ => {
                let delim = match options.sep.as_deref() {
                    Some("\\t") | Some("tab") => b'\t',
                    Some(s) if s.len() == 1 => s.as_bytes()[0],
                    Some(s) => {
                        return Err(HayashiError::Runtime(format!(
                            "load: sep must be a single character, got '{s}'"
                        )))
                    }
                    None => b',',
                };
                if delim == b',' && opt_columns.is_none() && predicate.is_none() {
                    // Caminho padrão (greeners): sem columns/where.
                    let df =
                        DataFrame::from_csv(local_path).map_err(|e| self.rt_err(e.to_string()))?;
                    let n = df.n_rows();
                    Ok((df, n))
                } else {
                    // Loader DSV do hayashi — suporta columns=/where=.
                    crate::io::dsv::load_dsv(local_path, delim, opt_columns, predicate)
                }
            }
        }
    }

    // ── load ─────────────────────────────────────────────────────────────────

    fn exec_load(&mut self, path: &Expr, alias: &str, opts: &[Opt]) -> Result<()> {
        let path_str = match self.eval_expr(path)? {
            Value::Str(s) => s,
            _ => return Err(self.type_err("load requires a string path")),
        };

        let options = self.parse_load_options(opts)?;

        if options.query.is_some() && (options.columns.is_some() || options.predicate.is_some()) {
            return Err(HayashiError::Runtime(
                "load: query= cannot be combined with columns= or where=".into(),
            ));
        }

        if let Some(conn_str) = path_str.strip_prefix("odbc://") {
            self.exec_load_odbc(conn_str, alias, &options)
        } else {
            self.exec_load_file(&path_str, alias, &options)
        }
    }

    // ── predict ──────────────────────────────────────────────────────────────

    fn predict_model_values(
        &self,
        model_val: &Value,
        kind: &str,
        df: &DataFrame,
        varname: &str,
    ) -> Result<Vec<f64>> {
        match (model_val, kind) {
            (Value::OlsResult(m), k) => self.predict_ols_vals(m, k),
            (Value::BinaryResult(m), k) => self.predict_binary_vals(m, k),
            (Value::PoissonResult(r), k) => self.predict_poisson_vals(r, k),
            (Value::NegBinResult(r), k) => self.predict_negbin_vals(r, k),
            (Value::OrderedResult(r), k) => self.predict_ordered_vals(r, k, df),
            (Value::IvResult(r), k) => self.predict_iv_vals(r, k, df),
            (Value::PanelResult(r), k) => self.predict_panel_vals(r, k, df),
            (Value::ReResult(r), k) => self.predict_re_vals(r, k, df),
            (Value::TobitResult(r), k) => self.predict_tobit_vals(r, k, df),
            (Value::HeckmanResult(r), k) => self.predict_heckman_vals(r, k, df),
            (Value::CoxResult(r), k) => self.predict_cox_vals(r, k, df),
            (Value::QuantileResult(r), k) => self.predict_quantile_vals(r, k, df),
            (Value::RlmResult(r), k) => self.predict_rlm_vals(r, k, df),
            (Value::GeeResult(r), k) => self.predict_gee_vals(r, k, df),
            (Value::BetaResult(r), k) => self.predict_beta_vals(r, k, df),
            (Value::GlsarResult(r), k) => self.predict_glsar_vals(r, k, df, varname),
            (Value::MixedResult(r), k) => self.predict_mixedlm_vals(r, k, df),
            (Value::ZeroInflatedResult(r), k) => self.predict_zero_inflated_vals(r, k, df),
            (Value::RollingResult(r), k) => self.predict_rolling_vals(r, k),
            (Value::RecursiveLSResult(r), k) => self.predict_recursive_ls_vals(r, k),
            (Value::GlmResult(r), k) => self.predict_glm_vals(r, k, df),
            (Value::LowessResult(r), k) => self.predict_lowess_vals(r, k),
            (Value::PcaResult(m), k) => self.predict_pca_vals(m, k),
            (Value::FactorResult(_), _) => Self::unsupported_predict(
                "Factor Analysis",
                "scores not available via FA — use pca() for scores; FA is for loadings analysis",
            ),
            (Value::MarkovResult(r), k) => self.predict_markov_vals(r, k),
            (Value::ConditionalResult(_), _) => Self::unsupported_predict(
                "clogit/cpoisson",
                "fixed effects absorbed — unconditional prediction not available; use β̂ coefficients for odds ratios or marginal effects",
            ),
            (Value::VarmaResult(_), _) => Self::unsupported_predict(
                "varma",
                "multivariate prediction not supported as a column — use print() for diagnostics",
            ),
            (Value::UCResult(r), k) => self.predict_ucm_vals(r, k),
            (Value::GamResult(_), _) => Self::unsupported_predict(
                "gam",
                "fitted values are not stored — use gam() with df=dataset and compute Xβ̂ manually",
            ),
            (Value::MiceResult(_), _) => Self::unsupported_predict(
                "mice",
                "MICE returns multiple datasets; access via model pooling",
            ),
            (Value::SVarResult(_), _) => Self::unsupported_predict(
                "svar",
                "no fitted values — use sirf() and sfevd() for impulse-response analysis",
            ),
            (Value::ThreeSLSResult(_), _) => Self::unsupported_predict(
                "3sls",
                "multiple equations — use print() to see coefficients per equation",
            ),
            (Value::DFMResult(m), k) => self.predict_dfm_vals(m, k),
            (Value::MSARResult(r), k) => self.predict_msar_vals(r, k),
            (Value::DecompResult(r), k) => self.predict_decomp_vals(r, k),
            (Value::MstlResult(r), k) => self.predict_mstl_vals(r, k),
            (Value::EtsResult(r), k) => self.predict_ets_vals(r, k),
            (Value::ThresholdResult(_), k) => Err(HayashiError::Runtime(format!(
                "predict pthresh: kind '{k}' — use print() to see thresholds and coefficients"
            ))),
            _ => Err(HayashiError::Type("predict: model type not supported".into())),
        }
    }

    fn linear_xb(
        df: &DataFrame,
        names: Option<&[String]>,
        params: &Array1<f64>,
    ) -> Result<Vec<f64>> {
        let x = build_x_from_varnames(df, names.unwrap_or(&[]))?;
        Ok(x.dot(params).to_vec())
    }

    fn predict_ols_vals(&self, m: &OlsModel, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Ok(m.x.dot(&m.result.params).to_vec()),
            "residuals" | "resid" | "e" => Ok(m.residuals.to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict OLS: kind '{k}' unknown — use: xb, residuals"
            ))),
        }
    }

    fn predict_binary_vals(&self, m: &BinaryModel, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "pr" | "xb" | "fitted" => Ok(m.result.predict_proba(&m.x).to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict logit/probit: kind '{k}' unknown — use: pr"
            ))),
        }
    }

    fn predict_poisson_vals(&self, r: &greeners::PoissonResult, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "count" | "mu" | "fitted" => Ok(r.fitted_values().to_vec()),
            "xb" => Ok(r.x_data().dot(&r.params).to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict Poisson: kind '{k}' unknown — use: count, xb"
            ))),
        }
    }

    fn predict_negbin_vals(&self, r: &greeners::NegBinResult, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "count" | "mu" | "fitted" => Ok(r.fitted_values().to_vec()),
            "xb" => Ok(r.x_data().dot(&r.params).to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict NegBin: kind '{k}' unknown — use: count, xb"
            ))),
        }
    }

    fn predict_ordered_vals(
        &self,
        r: &greeners::OrderedResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        let x = build_x_from_varnames(df, r.variable_names.as_deref().unwrap_or(&[]))?;
        match kind {
            "xb" => Ok(x.dot(&r.params).to_vec()),
            "yhat" => {
                let probs = r.predict_proba(&x);
                Ok((0..probs.nrows())
                    .map(|i| {
                        let row = probs.row(i);
                        let (cat, _) = row
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| nan_last_cmp(a, b))
                            .unwrap_or((0, &0.0));
                        (cat + 1) as f64
                    })
                    .collect())
            }
            s if s.starts_with("pr") && s.len() > 2 => {
                let cat: usize = s[2..].parse::<usize>().map_err(|_| {
                    HayashiError::Runtime(format!(
                        "predict Ordered: '{s}' — use prN where N is the category (1-indexed)"
                    ))
                })?;
                if cat == 0 || cat > r.n_categories {
                    return Err(HayashiError::Runtime(format!(
                        "predict Ordered: category {cat} out of range 1..{}",
                        r.n_categories
                    )));
                }
                let probs = r.predict_proba(&x);
                Ok((0..probs.nrows()).map(|i| probs[[i, cat - 1]]).collect())
            }
            "pr" => {
                let probs = r.predict_proba(&x);
                let last = r.n_categories - 1;
                Ok((0..probs.nrows()).map(|i| probs[[i, last]]).collect())
            }
            k => Err(HayashiError::Runtime(format!(
                "predict Ordered: kind '{k}' unknown — use: pr, prN, yhat, xb"
            ))),
        }
    }

    fn predict_iv_vals(
        &self,
        r: &greeners::iv::IvResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict IV: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_panel_vals(
        &self,
        r: &greeners::panel::PanelResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict FE: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_re_vals(
        &self,
        r: &greeners::panel::RandomEffectsResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict RE: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_tobit_vals(
        &self,
        r: &greeners::TobitResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict Tobit: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_heckman_vals(
        &self,
        r: &greeners::HeckmanResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict Heckman: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_cox_vals(
        &self,
        r: &greeners::CoxResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        let names = r.variable_names.as_deref();
        let x = build_x_from_varnames(df, names.unwrap_or(&[]))?;
        match kind {
            "loghr" | "xb" => Ok(r.predict_log_hazard(&x).to_vec()),
            "hr" | "hazard" => Ok(r.predict_hazard_ratio(&x).to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict Cox: kind '{k}' unknown — use: loghr, hr"
            ))),
        }
    }

    fn predict_quantile_vals(
        &self,
        r: &greeners::QuantileResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict QReg: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_rlm_vals(
        &self,
        r: &greeners::RlmResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict RLM: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_gee_vals(
        &self,
        r: &greeners::GeeResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict GEE: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_beta_vals(
        &self,
        r: &greeners::BetaResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        let names = r.variable_names.as_deref();
        match kind {
            "pr" | "mu" | "fitted" => {
                let x = build_x_from_varnames(df, names.unwrap_or(&[]))?;
                Ok(r.predict(&x, &greeners::BetaLink::Logit).to_vec())
            }
            "xb" => Self::linear_xb(df, names, &r.params),
            k => Err(HayashiError::Runtime(format!(
                "predict BetaReg: kind '{k}' unknown — use: pr, xb"
            ))),
        }
    }

    fn predict_glsar_vals(
        &self,
        r: &greeners::GlsarResult,
        kind: &str,
        df: &DataFrame,
        varname: &str,
    ) -> Result<Vec<f64>> {
        let names = r.variable_names.as_deref();
        match kind {
            "xb" | "fitted" => {
                let x = build_x_from_varnames(df, names.unwrap_or(&[]))?;
                Ok(r.fitted_values(&x).to_vec())
            }
            "residuals" | "resid" | "e" => {
                let x = build_x_from_varnames(df, names.unwrap_or(&[]))?;
                let y = get_col_f64(df, varname)?;
                Ok(r.residuals(&y, &x).to_vec())
            }
            k => Err(HayashiError::Runtime(format!(
                "predict GLSAR: kind '{k}' unknown — use: xb, residuals"
            ))),
        }
    }

    fn predict_mixedlm_vals(
        &self,
        r: &greeners::MixedResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        match kind {
            "xb" | "fitted" => Self::linear_xb(df, r.variable_names.as_deref(), &r.fixed_effects),
            k => Err(HayashiError::Runtime(format!(
                "predict MixedLM: kind '{k}' unknown — use: xb"
            ))),
        }
    }

    fn predict_zero_inflated_vals(
        &self,
        r: &greeners::ZeroInflatedResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        let count_names = r.count_var_names.as_deref();
        let inflate_names = r.inflate_var_names.as_deref().or(count_names);
        let x_c = build_x_from_varnames(df, count_names.unwrap_or(&[]))?;
        let x_i = build_x_from_varnames(df, inflate_names.unwrap_or(&[]))?;
        match kind {
            "count" | "mu" | "fitted" => Ok(r.predict_count(&x_c, &x_i).to_vec()),
            "pr0" => Ok(r.predict_proba_zero(&x_c, &x_i).to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict ZIP/ZINB: kind '{k}' unknown — use: count, pr0"
            ))),
        }
    }

    fn predict_rolling_vals(&self, r: &greeners::RollingResult, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "residuals" | "resid" | "e" => Ok(r.residuals.to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict RollingOLS: kind '{k}' unknown — use: residuals"
            ))),
        }
    }

    fn predict_recursive_ls_vals(
        &self,
        r: &greeners::RecursiveLSResult,
        kind: &str,
    ) -> Result<Vec<f64>> {
        match kind {
            "residuals" | "resid" | "e" => Ok(r.residuals.to_vec()),
            "cusum" => Ok(r.cusum.to_vec()),
            "cusum_sq" => Ok(r.cusum_squares.to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict RecursiveLS: kind '{k}' unknown — use: residuals, cusum, cusum_sq"
            ))),
        }
    }

    fn predict_glm_vals(
        &self,
        r: &greeners::GlmResult,
        kind: &str,
        df: &DataFrame,
    ) -> Result<Vec<f64>> {
        let names = r.variable_names.as_deref();
        match kind {
            "pr" | "mu" | "fitted" => {
                let x = build_x_from_varnames(df, names.unwrap_or(&[]))?;
                Ok(r.predict_mean(&x).to_vec())
            }
            "xb" => {
                let x = build_x_from_varnames(df, names.unwrap_or(&[]))?;
                Ok(r.predict(&x).to_vec())
            }
            "residuals" | "resid" | "e" | "deviance" => Ok(r.residuals().to_vec()),
            "pearson" => Ok(r.pearson_residuals().to_vec()),
            "working" => Ok(r.working_residuals().to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict GLM: kind '{k}' unknown — use: pr, xb, residuals, pearson, working"
            ))),
        }
    }

    fn predict_lowess_vals(&self, r: &greeners::LowessResult, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "smoothed" | "yhat" | "fitted" => Ok(r.smoothed.to_vec()),
            "residuals" | "resid" | "e" => Ok(r.residuals.to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict LOWESS: kind '{k}' unknown — use: smoothed, residuals"
            ))),
        }
    }

    fn predict_pca_vals(&self, m: &PcaModel, kind: &str) -> Result<Vec<f64>> {
        if kind.starts_with("pc") && kind.len() > 2 {
            let comp: usize = kind[2..].parse::<usize>().map_err(|_| {
                HayashiError::Runtime(format!(
                    "predict PCA: '{kind}' invalid — use pcN where N=1..{}",
                    m.result.n_components
                ))
            })?;
            if comp == 0 || comp > m.result.n_components {
                return Err(HayashiError::Runtime(format!(
                    "predict PCA: component {comp} out of range 1..{}",
                    m.result.n_components
                )));
            }
            Ok(m.result.scores.column(comp - 1).to_vec())
        } else {
            Err(HayashiError::Runtime(format!(
                "predict PCA: kind '{kind}' unknown — use: pc1, pc2, ..., pc{}",
                m.result.n_components
            )))
        }
    }

    fn predict_markov_vals(
        &self,
        r: &greeners::MarkovSwitchingResult,
        kind: &str,
    ) -> Result<Vec<f64>> {
        match kind {
            "smoothed" | "regime" | "state" => Ok((0..r.smoothed_probs.nrows())
                .map(|t| {
                    let row = r.smoothed_probs.row(t);
                    let (best, _) = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or((0, &0.0));
                    (best + 1) as f64
                })
                .collect()),
            s if s.starts_with("regime") && s.len() > 6 => {
                let idx: usize = s[6..].parse::<usize>().map_err(|_| {
                    HayashiError::Runtime(format!(
                        "predict MarkovSwitching: '{s}' invalid — use regimeN where N=1..{}",
                        r.n_regimes
                    ))
                })?;
                if idx == 0 || idx > r.n_regimes {
                    return Err(HayashiError::Runtime(format!(
                        "predict MarkovSwitching: regime {idx} out of range 1..{}",
                        r.n_regimes
                    )));
                }
                Ok(r.smoothed_probs.column(idx - 1).to_vec())
            }
            k => Err(HayashiError::Runtime(format!(
                "predict MarkovSwitching: kind '{k}' unknown — use: regime, regime1, regime2, ..."
            ))),
        }
    }

    fn predict_ucm_vals(&self, r: &greeners::UCResult, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "level" => Ok(r.level.to_vec()),
            "trend" => Ok(r
                .trend
                .as_ref()
                .map(|t| t.to_vec())
                .unwrap_or_else(|| vec![f64::NAN; r.n_obs])),
            "seasonal" => Ok(r
                .seasonal
                .as_ref()
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![f64::NAN; r.n_obs])),
            "residuals" | "resid" | "e" => Ok(r.residuals.to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict ucm: kind '{k}' unknown — use: level, trend, seasonal, residuals"
            ))),
        }
    }

    fn predict_dfm_vals(&self, m: &DFMModel, kind: &str) -> Result<Vec<f64>> {
        if let Some(rest) = kind.strip_prefix('f') {
            let idx = rest
                .parse::<usize>()
                .map(|n| n.saturating_sub(1))
                .unwrap_or(0);
            if idx >= m.result.n_factors {
                return Err(HayashiError::Runtime(format!(
                    "predict dfm: factor f{} does not exist — model has {} factors",
                    idx + 1,
                    m.result.n_factors
                )));
            }
            Ok(m.result.factors.column(idx).to_vec())
        } else {
            Err(HayashiError::Runtime(format!(
                "predict dfm: kind '{kind}' unknown — use: f1, f2, ... (1-based index of latent factor)"
            )))
        }
    }

    fn predict_msar_vals(&self, r: &greeners::MarkovAutoregResult, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "regime" | "state" => Ok(r.predict_regime().iter().map(|&s| (s + 1) as f64).collect()),
            s if s.starts_with("regime") && s.len() > 6 => {
                let idx = s["regime".len()..]
                    .parse::<usize>()
                    .map(|n| n.saturating_sub(1))
                    .unwrap_or(0);
                if idx >= r.k_regimes {
                    return Err(HayashiError::Runtime(format!(
                        "predict msauto: regime{} out of range 1..{}",
                        idx + 1,
                        r.k_regimes
                    )));
                }
                Ok(r.smoothed_probs.column(idx).to_vec())
            }
            k => Err(HayashiError::Runtime(format!(
                "predict msauto: kind '{k}' unknown — use: regime, regime1, regime2, ..."
            ))),
        }
    }

    fn predict_decomp_vals(
        &self,
        r: &greeners::DecompositionResult,
        kind: &str,
    ) -> Result<Vec<f64>> {
        match kind {
            "trend" => Ok(r.trend.to_vec()),
            "seasonal" => Ok(r.seasonal.to_vec()),
            "residual" | "resid" | "e" => Ok(r.residual.to_vec()),
            "observed" | "fitted" => Ok(r.observed.to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict decompose: kind '{k}' unknown — use: trend, seasonal, residual, observed"
            ))),
        }
    }

    fn predict_mstl_vals(&self, r: &greeners::MSTLResult, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "trend" => Ok(r.trend.to_vec()),
            "resid" | "residual" | "e" => Ok(r.resid.to_vec()),
            s if s.starts_with("seasonal") => {
                let idx = if s == "seasonal" {
                    0usize
                } else {
                    s["seasonal".len()..]
                        .parse::<usize>()
                        .map(|n| n.saturating_sub(1))
                        .unwrap_or(0)
                };
                if idx >= r.seasonal.len() {
                    return Err(HayashiError::Runtime(format!(
                        "predict mstl: seasonal{} component does not exist — model has {} periods",
                        idx + 1,
                        r.seasonal.len()
                    )));
                }
                Ok(r.seasonal[idx].to_vec())
            }
            k => Err(HayashiError::Runtime(format!(
                "predict mstl: kind '{k}' unknown — use: trend, resid, seasonal, seasonal1, seasonal2, ..."
            ))),
        }
    }

    fn predict_ets_vals(&self, r: &greeners::ETSResult, kind: &str) -> Result<Vec<f64>> {
        match kind {
            "fitted" | "yhat" | "xb" => Ok(r.fitted_values.to_vec()),
            "residuals" | "resid" | "e" => Ok(r.residuals.to_vec()),
            "level" => Ok(r.level.to_vec()),
            "trend" => Ok(r.trend.to_vec()),
            "seasonal" => Ok(r.seasonal.to_vec()),
            k => Err(HayashiError::Runtime(format!(
                "predict ets: kind '{k}' unknown — use: fitted, residuals, level, trend, seasonal"
            ))),
        }
    }

    fn unsupported_predict(model: &str, detail: &str) -> Result<Vec<f64>> {
        Err(HayashiError::Runtime(format!("predict {model}: {detail}")))
    }

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

        let vals = self.predict_model_values(&model_val, &kind_str, &df_val, varname)?;

        let arr = ndarray::Array1::from(vals);
        Arc::make_mut(&mut df_val)
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
        Arc::make_mut(&mut df_val)
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
            (Value::DataFrame(_df), "sqlite" | "sqlite3" | "db") => {
                #[cfg(feature = "sqlite")]
                {
                    crate::io::sqlite::write_sqlite(&_df, &path_str, "data")?;
                    println!("Exported DataFrame → '{path_str}' ({} rows)", _df.n_rows());
                }
                #[cfg(not(feature = "sqlite"))]
                return Err(self.rt_err("SQLite export requires 'sqlite' feature"));
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
                | Value::PenalizedResult(_)
                | Value::LocalLevelResult(_)),
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

    // ── parallel for ──────────────────────────────────────────────────────────

    /// Materializes a `ForIter` into a `Vec<Value>` on the calling thread.
    fn materialize_iter(&mut self, iter: &ForIter) -> Result<Vec<Value>> {
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
                let mut items = Vec::new();
                let mut cur = start;
                while if step > 0 { cur < end } else { cur > end } {
                    items.push(Value::Int(cur));
                    cur += step;
                }
                Ok(items)
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
                let mut items = Vec::new();
                let mut cur = start;
                while if step > 0 { cur <= end } else { cur >= end } {
                    items.push(Value::Int(cur));
                    cur += step;
                }
                Ok(items)
            }
            ForIter::Items(iter_expr) => {
                let value = self.eval_expr(iter_expr)?;
                match value {
                    Value::List(v) => Ok((*v).clone()),
                    Value::Dict(d) => {
                        let mut items = Vec::new();
                        for (k, v) in d.iter() {
                            let mut entry = HashMap::new();
                            entry.insert("key".to_string(), Value::Str(k.clone()));
                            entry.insert("value".to_string(), v.clone());
                            items.push(Value::Dict(Arc::new(entry)));
                        }
                        Ok(items)
                    }
                    other => Err(HayashiError::Type(format!(
                        "for: iterator must be a list or dict, not {other}"
                    ))),
                }
            }
        }
    }

    pub(super) fn exec_parallel_for(
        &mut self,
        var: &str,
        var2: Option<&str>,
        iter: &ForIter,
        body: &[Spanned],
        max_threads: Option<usize>,
    ) -> Result<Value> {
        // 1. Materialize the iterator on the main thread.
        let items = self.materialize_iter(iter)?;

        // 2. Snapshot the current environment (only send-safe values).
        let names = self.env.all_names();
        let mut snapshot: Vec<(String, SendValue)> = Vec::with_capacity(names.len());
        for name in &names {
            if let Some(v) = self.env.get(name) {
                let v = v.clone();
                if v.is_send_safe() {
                    snapshot.push((
                        name.clone(),
                        SendValue::new(v).map_err(HayashiError::Runtime)?,
                    ));
                }
            }
        }

        // 3. Determine the number of threads.
        //    Explicit `threads=N` takes precedence; otherwise use available CPUs.
        let n_threads = max_threads
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            })
            .min(items.len().max(1));

        if items.is_empty() || n_threads == 1 {
            // Fallback: sequential execution (same semantics, no threads).
            let mut results = Vec::with_capacity(items.len());
            for (i, item) in items.into_iter().enumerate() {
                self.env.push_scope();
                self.env.set(var, item)?;
                if let Some(v2) = var2 {
                    self.env.set(v2, Value::Int(i as i64))?;
                }
                let val = self.exec_parallel_body(body)?;
                self.env.pop_scope();
                results.push(val);
            }
            return Ok(Value::List(Arc::new(results)));
        }

        // 4. Divide items into chunks (wrapped as SendValue for thread-safety).
        let chunk_size = items.len().div_ceil(n_threads);
        let chunks: Vec<Vec<SendValue>> = items
            .chunks(chunk_size)
            .map(|c| {
                c.iter()
                    .map(|v| SendValue::new(v.clone()).map_err(HayashiError::Runtime))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        // 5. Shared stdout mutex for thread-safe printing.
        let stdout_lock = Arc::new(std::sync::Mutex::new(()));

        // 6. Spawn threads via scoped threads.
        let chunk_results: Vec<Value> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(chunks.len());
            for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                let snapshot = snapshot.clone(); // cheap: Arc clones inside
                let body = body.to_vec();
                let var = var.to_string();
                let var2 = var2.map(|s| s.to_string());
                let stdout_lock = stdout_lock.clone();

                let handle = s.spawn(move || -> Vec<ThreadResult> {
                    let _guard = stdout_lock.lock().unwrap();

                    // Each thread gets a fresh interpreter.
                    let mut interp = Interpreter::new();
                    // Suppress auto-display: the body's last expression is
                    // collected as a result, not printed. Only explicit
                    // print() calls should produce output.
                    interp.env.set_quiet_mode(true);

                    // Load the snapshot into the global scope.
                    for (name, sv) in &snapshot {
                        let _ = interp.env.set(name, sv.0.clone());
                    }

                    drop(_guard); // release stdout during compute

                    let mut results = Vec::with_capacity(chunk.len());
                    for (i, item) in chunk.into_iter().enumerate() {
                        let item = item.0; // unwrap SendValue
                        interp.env.push_scope();
                        let _ = interp.env.set(&var, item);
                        if let Some(ref v2) = var2 {
                            let _ = interp.env.set(v2, Value::Int(i as i64));
                        }

                        let val = interp.exec_parallel_body(&body);
                        interp.env.pop_scope();
                        results.push(ThreadResult(val));
                    }
                    results
                });
                handles.push((chunk_idx, handle));
            }

            // Join in order, collect results.
            let mut all = Vec::new();
            for (_idx, h) in handles {
                let chunk_results = h.join().unwrap_or_default();
                for tr in chunk_results {
                    match tr.0 {
                        Ok(v) => all.push(v),
                        Err(_e) => all.push(Value::Nil),
                    }
                }
            }
            all
        });

        Ok(Value::List(Arc::new(chunk_results)))
    }

    /// Executes the body of a `parallel for` iteration and returns the
    /// iteration's result value.
    ///
    /// The result is determined by:
    /// 1. An explicit `return expr` inside the body (captured via HayashiError::Return)
    /// 2. Otherwise, the value of the last `Stmt::Expr` in the body
    /// 3. Otherwise, `Value::Nil`
    fn exec_parallel_body(&mut self, body: &[Spanned]) -> Result<Value> {
        let mut last_expr_val = Value::Nil;

        for s in body {
            match self.exec(s) {
                Ok(()) => {
                    // Track the value of standalone expressions.
                    if let Stmt::Expr(expr) = &s.0 {
                        last_expr_val = self.eval_expr(expr)?;
                    }
                }
                Err(HayashiError::Return) => {
                    // Explicit return — use the stored return_value.
                    let val = self.return_value.take().unwrap_or(Value::Nil);
                    return Ok(val);
                }
                Err(HayashiError::Break) | Err(HayashiError::Continue) => {
                    // break/continue skip the rest; return last expr seen.
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(last_expr_val)
    }
}

/// Extrai nomes de colunas da AST de `columns=` sem avaliar contra o
/// ambiente. Aceita `Expr::Var` (identificador sem aspas) e `Expr::Str`
/// (string literal), em lista ou único.
fn extract_column_names(expr: &Expr) -> Result<Vec<String>> {
    match expr {
        Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for e in items {
                out.push(col_name_from_expr(e)?);
            }
            Ok(out)
        }
        other => Ok(vec![col_name_from_expr(other)?]),
    }
}

fn col_name_from_expr(e: &Expr) -> Result<String> {
    match e {
        Expr::Var(name) => Ok(name.clone()),
        Expr::Str(s) => Ok(s.clone()),
        other => Err(HayashiError::Type(format!(
            "load: columns= expects column names (identifiers or strings), got {:?}",
            other
        ))),
    }
}
