use super::helpers::*;
use super::*;
use crate::lang::dap::model_expansion;
use indexmap::IndexMap;
use std::sync::Arc;

mod panel_diagnostics;
mod rolling_recursive;

/// Generic bootstrap/bootse, panel diagnostics, SUR, rolling/recursive
/// OLS, information criteria table, Fixed Effects, Random Effects,
/// panel tests (F-test, Pesaran CD, Breusch-Pagan LM, Chamberlain),
/// Arellano-Bond, generic GMM, System GMM, FE-2SLS, PCSE, Panel GLS,
/// m1/m2 test, Hausman, general specification/Wald tests.
/// Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_estimators_panel(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            "bootstrap" | "boot" => self.bootstrap(func, args, opts, opt_map),
            "bootse" => {
                return self.eval_call("bootstrap", args, opts).map(Some);
            }
            "markov" | "msar" | "markovswitching" => self.markov(func, args, opts, opt_map),
            "clogit" | "xtlogit_fe" => self.clogit(func, args, opts, opt_map),
            "cpoisson" | "xtpoisson_fe" | "ppml" => self.cpoisson(func, args, opts, opt_map),
            "cmnlogit" | "cmlogit" | "conditional_mlogit" => {
                self.cmnlogit(func, args, opts, opt_map)
            }
            "gqtest" => self.gqtest(func, args, opts, opt_map),
            "bphet" | "hettest" => self.bphet(func, args, opts, opt_map),
            "bptest" | "xttest0" | "xtbp" => self.bptest(func, args, opts, opt_map),
            "wooldridge" | "xtserial" | "wooldridge_serial" | "xtwooldridge" => {
                self.wooldridge(func, args, opts, opt_map)
            }
            "pesaran" | "xtcd" => self.eval_pesaran(args, opt_map),
            "mundlak" => self.eval_mundlak(args, opt_map),
            "abtest" | "abar" | "abond" | "xtabond_test" | "arellano_bond" => {
                self.abtest(func, args, opts, opt_map)
            }
            "sur" | "sureg" => self.sur(func, args, opts, opt_map),
            "rolling" | "rols" => self.eval_rolling(args, opts, opt_map),
            "recursive" | "recols" => self.eval_recursive(args, opts),
            "ic" | "fitstat" | "estat" => self.ic(func, args, opts, opt_map),
            "akaike_weights" | "aic_weights" => self.akaike_weights(func, args, opts, opt_map),
            "lrtest" | "lr_test" => self.lrtest(func, args, opts, opt_map),
            "fe" => self.fe(func, args, opts, opt_map),
            "re" => self.re(func, args, opts, opt_map),
            "ftest_fe" => self.ftest_fe(func, args, opts, opt_map),
            "pesaran_cd" | "cd_test" => self.pesaran_cd(func, args, opts, opt_map),
            "bplm" => self.bplm(func, args, opts, opt_map),
            "chamberlain" => self.chamberlain(func, args, opts, opt_map),
            "mundlak_OLD_REMOVED" => self.mundlak_OLD_REMOVED(func, args, opts, opt_map),
            "ab" => self.ab(func, args, opts, opt_map),
            "gmm" => self.gmm(func, args, opts, opt_map),
            "sysgmm" => self.sysgmm(func, args, opts, opt_map),
            "feiv" => self.feiv(func, args, opts, opt_map),
            "pcse" => self.pcse(func, args, opts, opt_map),
            "xtgls" => self.xtgls(func, args, opts, opt_map),
            "ab_test" => self.ab_test(func, args, opts, opt_map),
            "wooldridge_OLD_REMOVED" => self.wooldridge_OLD_REMOVED(func, args, opts, opt_map),
            "hausman" => self.hausman(func, args, opts, opt_map),
            "hausman_robust" | "hausman_r" => self.hausman_robust(func, args, opts, opt_map),
            "ftest_robust" | "f_robust" => self.ftest_robust(func, args, opts, opt_map),
            "test" => self.test(func, args, opts, opt_map),
            _ => return Ok(None),
        };
        result.map(Some)
    }

    // ── Bootstrap helpers ─────────────────────────────────────────────────────

    fn bootstrap_reps(opt_map: &HashMap<String, Value>) -> usize {
        match opt_map.get("n") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => match opt_map.get("reps") {
                Some(Value::Int(v)) => *v as usize,
                Some(Value::Float(v)) => *v as usize,
                _ => 1000,
            },
        }
    }

    fn bootstrap_alpha(opt_map: &HashMap<String, Value>) -> f64 {
        match opt_map.get("alpha") {
            Some(Value::Float(v)) => *v,
            _ => 0.05,
        }
    }

    fn panel_sig_stars(p: f64) -> &'static str {
        if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        }
    }

    fn bootstrap_generic(
        &mut self,
        args: &[Expr],
        opts: &[Opt],
        n_boot: usize,
        alpha: f64,
    ) -> Result<Value> {
        let estimator_name = match &args[0] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "bootstrap: first argument must be estimator name (ols, logit, ...)".into(),
                ))
            }
        };
        let formula_expr = args[1].clone();
        let df_name = match &args[2] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "bootstrap: third argument must be DataFrame name".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "'{df_name}' is not a DataFrame"
                )))
            }
        };

        let extra_opts: Vec<Opt> = opts
            .iter()
            .filter(|o| !matches!(o.name.as_str(), "n" | "reps" | "alpha"))
            .cloned()
            .collect();
        let full_result = self.eval_call(
            &estimator_name,
            &[formula_expr.clone(), Expr::Var(df_name.clone())],
            &extra_opts,
        )?;
        let full_params = extract_params(&full_result).ok_or_else(|| {
            HayashiError::Runtime("bootstrap: model not supported (no extractable params)".into())
        })?;
        let full_se = extract_se(&full_result).unwrap_or_default();
        let var_names = extract_var_names(&full_result);
        let k = full_params.len();

        use rand::seq::SliceRandom;
        let mut rng = self.get_rng();
        let n = df.n_rows();
        let indices: Vec<usize> = (0..n).collect();
        let mut boot_coefs = ndarray::Array2::<f64>::zeros((n_boot, k));
        let mut n_ok = 0usize;

        for b in 0..n_boot {
            let boot_idx: Vec<usize> = (0..n).map(|_| *indices.choose(&mut rng).unwrap()).collect();
            let boot_df = match df.iloc(Some(&boot_idx), None) {
                Ok(d) => d,
                Err(_) => continue,
            };
            self.env
                .set("__boot_df__", Value::DataFrame(Arc::new(boot_df)))?;
            if let Ok(ref result) = self.eval_call(
                &estimator_name,
                &[formula_expr.clone(), Expr::Var("__boot_df__".into())],
                &extra_opts,
            ) {
                if let Some(params) = extract_params(result) {
                    for j in 0..k.min(params.len()) {
                        boot_coefs[[b, j]] = params[j];
                    }
                    n_ok += 1;
                }
            }
        }
        self.env.remove("__boot_df__");

        if n_ok < 10 {
            return Err(HayashiError::Runtime(format!(
                "bootstrap: only {n_ok}/{n_boot} replications converged"
            )));
        }

        let boot_se = greeners::Bootstrap::bootstrap_se(&boot_coefs);
        let (ci_lo, ci_hi) = greeners::Bootstrap::percentile_ci(&boot_coefs, alpha);

        let thick = "═".repeat(76);
        let thin = "─".repeat(76);
        let mut display = String::new();
        display.push_str(&format!("\n{thick}\n"));
        display.push_str(&format!(
            "{:^76}\n",
            format!(" Bootstrap SE — {} (n={n_ok}/{n_boot}) ", estimator_name)
        ));
        display.push_str(&format!("{thin}\n"));
        display.push_str(&format!(
            "{:<18} {:>10} {:>10} {:>10} {:>12} {:>12}\n",
            "Variable", "β̂", "Orig. SE", "Boot SE", "CI lower", "CI upper"
        ));
        display.push_str(&format!("{thin}\n"));

        let mut variable_col = Vec::with_capacity(k);
        let mut beta_col = Vec::with_capacity(k);
        let mut orig_se_col = Vec::with_capacity(k);
        let mut boot_se_col = Vec::with_capacity(k);
        let mut ci_low_col = Vec::with_capacity(k);
        let mut ci_high_col = Vec::with_capacity(k);

        for i in 0..k {
            let vname = var_names.get(i).cloned().unwrap_or_else(|| format!("x{i}"));
            let orig_se = if i < full_se.len() {
                full_se[i]
            } else {
                f64::NAN
            };
            display.push_str(&format!(
                "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>12.4}\n",
                vname, full_params[i], orig_se, boot_se[i], ci_lo[i], ci_hi[i]
            ));
            variable_col.push(vname);
            beta_col.push(full_params[i]);
            orig_se_col.push(orig_se);
            boot_se_col.push(boot_se[i]);
            ci_low_col.push(ci_lo[i]);
            ci_high_col.push(ci_hi[i]);
        }
        display.push_str(&format!("{thick}\n"));

        let mut coef_columns = IndexMap::new();
        coef_columns.insert(
            "variable".into(),
            greeners::Column::String(ndarray::Array1::from(variable_col)),
        );
        coef_columns.insert(
            "beta".into(),
            greeners::Column::Float(ndarray::Array1::from(beta_col)),
        );
        coef_columns.insert(
            "orig_se".into(),
            greeners::Column::Float(ndarray::Array1::from(orig_se_col)),
        );
        coef_columns.insert(
            "boot_se".into(),
            greeners::Column::Float(ndarray::Array1::from(boot_se_col)),
        );
        coef_columns.insert(
            "ci_low".into(),
            greeners::Column::Float(ndarray::Array1::from(ci_low_col)),
        );
        coef_columns.insert(
            "ci_high".into(),
            greeners::Column::Float(ndarray::Array1::from(ci_high_col)),
        );
        let coef_table = Value::DataFrame(Arc::new(
            greeners::DataFrame::from_columns(coef_columns)
                .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
        ));

        let mut rep_columns: IndexMap<String, greeners::Column> = IndexMap::new();
        for j in 0..k {
            let col_name = var_names.get(j).cloned().unwrap_or_else(|| format!("x{j}"));
            let col: Vec<f64> = (0..n_boot).map(|i| boot_coefs[[i, j]]).collect();
            rep_columns.insert(
                col_name,
                greeners::Column::Float(ndarray::Array1::from(col)),
            );
        }
        let replicates = Value::DataFrame(Arc::new(
            greeners::DataFrame::from_columns(rep_columns)
                .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
        ));

        let summary = format!("Bootstrap SE for {estimator_name} (n_ok={n_ok}/{n_boot})");
        let fit = model_expansion::fit_dict(&[
            ("estimator", Value::Str(estimator_name)),
            ("n_boot", Value::Int(n_boot as i64)),
            ("n_ok", Value::Int(n_ok as i64)),
            ("alpha", Value::Float(alpha)),
            ("coef_table", coef_table),
            ("replicates", replicates),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "BootstrapResult",
            fields,
        ))
    }

    fn bootstrap_pairs(&mut self, args: &[Expr], n_boot: usize, alpha: f64) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "bootstrap(estimator, formula, df, n=1000) or bootse(model, n=1000)".into(),
            ));
        }
        let model_val = self.eval_expr(&args[0])?;
        match &model_val {
            Value::OlsResult(m) => {
                let y_hat = m.x.dot(&m.result.params);
                let y_vec = &y_hat + &m.residuals;
                let boot_coefs = greeners::Bootstrap::pairs_bootstrap(&y_vec, &m.x, n_boot)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let boot_se = greeners::Bootstrap::bootstrap_se(&boot_coefs);
                let (ci_lo, ci_hi) = greeners::Bootstrap::percentile_ci(&boot_coefs, alpha);
                let thick = "═".repeat(76);
                let thin = "─".repeat(76);
                let mut display = String::new();
                display.push_str(&format!("\n{thick}\n"));
                display.push_str(&format!(
                    "{:^76}\n",
                    format!(" Bootstrap SE (n={n_boot}, pairs) ")
                ));
                display.push_str(&format!("{thin}\n"));
                display.push_str(&format!(
                    "{:<18} {:>10} {:>10} {:>10} {:>12} {:>12}\n",
                    "Variable", "β̂", "Orig. SE", "Boot SE", "CI lower 95%", "CI upper 95%"
                ));
                display.push_str(&format!("{thin}\n"));

                let vnames_owned = m.result.variable_names.clone().unwrap_or_default();
                let k = m.result.params.len();
                let mut variable_col = Vec::with_capacity(k);
                let mut beta_col = Vec::with_capacity(k);
                let mut orig_se_col = Vec::with_capacity(k);
                let mut boot_se_col = Vec::with_capacity(k);
                let mut ci_low_col = Vec::with_capacity(k);
                let mut ci_high_col = Vec::with_capacity(k);

                for i in 0..k {
                    let vname = vnames_owned
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("x{i}"));
                    display.push_str(&format!(
                        "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>12.4}\n",
                        vname,
                        m.result.params[i],
                        m.result.std_errors[i],
                        boot_se[i],
                        ci_lo[i],
                        ci_hi[i]
                    ));
                    variable_col.push(vname);
                    beta_col.push(m.result.params[i]);
                    orig_se_col.push(m.result.std_errors[i]);
                    boot_se_col.push(boot_se[i]);
                    ci_low_col.push(ci_lo[i]);
                    ci_high_col.push(ci_hi[i]);
                }
                display.push_str(&format!("{thick}\n"));

                let mut coef_columns = IndexMap::new();
                coef_columns.insert(
                    "variable".into(),
                    greeners::Column::String(ndarray::Array1::from(variable_col)),
                );
                coef_columns.insert(
                    "beta".into(),
                    greeners::Column::Float(ndarray::Array1::from(beta_col)),
                );
                coef_columns.insert(
                    "orig_se".into(),
                    greeners::Column::Float(ndarray::Array1::from(orig_se_col)),
                );
                coef_columns.insert(
                    "boot_se".into(),
                    greeners::Column::Float(ndarray::Array1::from(boot_se_col)),
                );
                coef_columns.insert(
                    "ci_low".into(),
                    greeners::Column::Float(ndarray::Array1::from(ci_low_col)),
                );
                coef_columns.insert(
                    "ci_high".into(),
                    greeners::Column::Float(ndarray::Array1::from(ci_high_col)),
                );
                let coef_table = Value::DataFrame(Arc::new(
                    greeners::DataFrame::from_columns(coef_columns).unwrap_or_else(|_| {
                        greeners::DataFrame::from_columns(IndexMap::new()).unwrap()
                    }),
                ));

                let mut rep_columns: IndexMap<String, greeners::Column> = IndexMap::new();
                for j in 0..k {
                    let col_name = vnames_owned
                        .get(j)
                        .cloned()
                        .unwrap_or_else(|| format!("x{j}"));
                    let col: Vec<f64> = (0..n_boot).map(|i| boot_coefs[[i, j]]).collect();
                    rep_columns.insert(
                        col_name,
                        greeners::Column::Float(ndarray::Array1::from(col)),
                    );
                }
                let replicates = Value::DataFrame(Arc::new(
                    greeners::DataFrame::from_columns(rep_columns).unwrap_or_else(|_| {
                        greeners::DataFrame::from_columns(IndexMap::new()).unwrap()
                    }),
                ));

                let summary = format!("Bootstrap SE (pairs, n={n_boot})");
                let fit = model_expansion::fit_dict(&[
                    ("estimator", Value::Str("pairs".into())),
                    ("n_boot", Value::Int(n_boot as i64)),
                    ("n_ok", Value::Int(n_boot as i64)),
                    ("alpha", Value::Float(alpha)),
                    ("coef_table", coef_table),
                    ("replicates", replicates),
                ]);
                let fields = vec![("fit".into(), fit)];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "BootstrapResult",
                    fields,
                ))
            }
            _ => Err(HayashiError::Runtime(
                "bootse(model) supports OLS. For others: bootstrap(estimator, formula, df, n=1000)"
                    .into(),
            )),
        }
    }

    pub(super) fn bootstrap(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let n_boot = Self::bootstrap_reps(opt_map);
        let alpha = Self::bootstrap_alpha(opt_map);
        if args.len() >= 3 {
            self.bootstrap_generic(args, opts, n_boot, alpha)
        } else {
            self.bootstrap_pairs(args, n_boot, alpha)
        }
    }

    pub(super) fn markov(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime("markov(df, y_var, k=2, p=1)".into()));
        }
        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "markov: first argument must be a DataFrame".into(),
                ))
            }
        };
        let y_name = match &args[1] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "markov: second argument must be variable name".into(),
                ))
            }
        };
        let y_vec = ndarray::Array1::from(get_col_f64(&df, &y_name)?);
        let k = match opt_map.get("k") {
            Some(Value::Int(v)) => (*v as usize).max(2),
            Some(Value::Float(v)) => (*v as usize).max(2),
            _ => 2,
        };
        let p = match opt_map.get("p") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 1,
        };
        let result = greeners::MarkovSwitching::fit(&y_vec, k, p)
            .map_err(|e| self.rt_err(format!("markov: {e}")))?;
        Ok(Value::MarkovResult(Rc::new(result)))
    }

    pub(super) fn clogit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let group_col = match opt_map.get("group") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "clogit requires group=\"id_col\"".into(),
                ))
            }
            _ => return Err(HayashiError::Type("clogit: group= must be string".into())),
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let group_vals = get_col_f64(&df, &group_col)?;
        let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut gnext = 0usize;
        let groups: Vec<usize> = group_vals
            .iter()
            .map(|&v| {
                let key = v as i64;
                *gmap.entry(key).or_insert_with(|| {
                    let g = gnext;
                    gnext += 1;
                    g
                })
            })
            .collect();
        let result =
            greeners::ConditionalLogit::fit_with_names(&y_vec, &x_mat, &groups, Some(var_names))
                .map_err(|e| self.rt_err(format!("clogit: {e}")))?;
        Ok(Value::ConditionalResult(Rc::new(result)))
    }

    pub(super) fn cpoisson(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let group_col = match opt_map.get("group") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "cpoisson requires group=\"id_col\"".into(),
                ))
            }
            _ => return Err(HayashiError::Type("cpoisson: group= must be string".into())),
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let group_vals = get_col_f64(&df, &group_col)?;
        let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut gnext = 0usize;
        let groups: Vec<usize> = group_vals
            .iter()
            .map(|&v| {
                let key = v as i64;
                *gmap.entry(key).or_insert_with(|| {
                    let g = gnext;
                    gnext += 1;
                    g
                })
            })
            .collect();
        let result =
            greeners::ConditionalPoisson::fit_with_names(&y_vec, &x_mat, &groups, Some(var_names))
                .map_err(|e| self.rt_err(format!("cpoisson: {e}")))?;
        Ok(Value::ConditionalResult(Rc::new(result)))
    }

    pub(super) fn cmnlogit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let group_col = match opt_map.get("group") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "cmnlogit requires group=\"id_col\"".into(),
                ))
            }
            _ => return Err(HayashiError::Type("cmnlogit: group= must be string".into())),
        };
        let n_alts = match opt_map.get("alts") {
            Some(Value::Int(n)) => *n as usize,
            Some(Value::Float(f)) => *f as usize,
            None => {
                return Err(HayashiError::Runtime(
                    "cmnlogit requires alts=N (number of alternatives)".into(),
                ))
            }
            _ => return Err(HayashiError::Type("cmnlogit: alts= must be integer".into())),
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let group_vals = get_col_f64(&df, &group_col)?;
        let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut gnext = 0usize;
        let groups: Vec<usize> = group_vals
            .iter()
            .map(|&v| {
                let key = v as i64;
                *gmap.entry(key).or_insert_with(|| {
                    let g = gnext;
                    gnext += 1;
                    g
                })
            })
            .collect();
        let result = greeners::ConditionalMNLogit::fit_with_names(
            &y_vec,
            &x_mat,
            &groups,
            n_alts,
            Some(var_names),
        )
        .map_err(|e| self.rt_err(format!("cmnlogit: {e}")))?;
        Ok(Value::ConditionalResult(Rc::new(result)))
    }

    pub(super) fn gqtest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime("gqtest(model, split=0.2)".into()));
        }
        let ols = match self.eval_expr(&args[0])? {
            Value::OlsResult(m) => m,
            _ => {
                return Err(HayashiError::Type(
                    "gqtest(): only supports OLS models".into(),
                ))
            }
        };
        let split = match opt_map.get("split") {
            Some(Value::Float(v)) => *v,
            Some(Value::Int(v)) => *v as f64,
            _ => 0.2,
        };
        let (f, p, df1, df2) =
            greeners::SpecificationTests::goldfeld_quandt_test(&ols.residuals, split)
                .map_err(|e| self.rt_err(format!("gqtest: {e}")))?;
        let sig = Self::panel_sig_stars(p);
        let sep = "─".repeat(56);
        let conclusion = if p < 0.05 {
            "Reject H0 — evidence of heteroskedasticity"
        } else {
            "Do not reject H0 — no evidence of heteroskedasticity"
        };
        let mut display = String::new();
        display.push_str(&format!("\nGoldfeld-Quandt Test  —  split = {split:.2}\n"));
        display.push_str(&format!("{sep}\n"));
        display.push_str("H₀: homoskedasticity (σ²₁ = σ²₂)\n");
        display.push_str(&format!("{sep}\n"));
        display.push_str(&format!(
            "{:<26} {:>10} {:>10} {:>4}\n",
            "Test", "Statistic", "p-value", ""
        ));
        display.push_str(&format!("{sep}\n"));
        display.push_str(&format!(
            "{:<26} {:>10.4} {:>10.4} {:>4}\n",
            format!("F ~ F({df1},{df2})"),
            f,
            p,
            sig
        ));
        display.push_str(&format!("{sep}\n"));
        display.push_str("(*** p<0.01  ** p<0.05  * p<0.10)\n\n");

        let summary = format!("Goldfeld-Quandt F({df1},{df2})={:.4}, p={:.4}", f, p);
        let fit = model_expansion::fit_dict(&[
            ("test", Value::Str("Goldfeld-Quandt".into())),
            ("f_stat", Value::Float(f)),
            ("p_value", Value::Float(p)),
            ("df1", Value::Int(df1 as i64)),
            ("df2", Value::Int(df2 as i64)),
            ("split", Value::Float(split)),
            ("conclusion", Value::Str(conclusion.into())),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "GoldfeldQuandtResult",
            fields,
        ))
    }

    pub(super) fn bphet(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime("bphet(model)".into()));
        }
        let ols = match self.eval_expr(&args[0])? {
            Value::OlsResult(m) => m,
            _ => {
                return Err(HayashiError::Type(
                    "bphet(): only supports OLS models".into(),
                ))
            }
        };
        let (lm, p) = greeners::Diagnostics::breusch_pagan(&ols.residuals, &ols.x)
            .map_err(|e| self.rt_err(format!("bphet: {e}")))?;
        let k = ols.x.ncols().saturating_sub(1);
        let sig = Self::panel_sig_stars(p);
        let sep = "─".repeat(56);
        let conclusion = if p < 0.05 {
            "Reject H0 — evidence of heteroskedasticity"
        } else {
            "Do not reject H0 — no evidence of heteroskedasticity"
        };
        let mut display = String::new();
        display.push_str("\nBreusch-Pagan Heteroskedasticity Test\n");
        display.push_str(&format!("{sep}\n"));
        display.push_str("H₀: homoskedasticity (constant variance)\n");
        display.push_str(&format!("{sep}\n"));
        display.push_str(&format!(
            "{:<26} {:>10} {:>10} {:>4}\n",
            "Test", "Statistic", "p-value", ""
        ));
        display.push_str(&format!("{sep}\n"));
        display.push_str(&format!(
            "{:<26} {:>10.4} {:>10.4} {:>4}\n",
            format!("LM ~ χ²({k})"),
            lm,
            p,
            sig
        ));
        display.push_str(&format!("{sep}\n"));
        display.push_str("(*** p<0.01  ** p<0.05  * p<0.10)\n\n");

        let summary = format!("Breusch-Pagan LM={:.4}, p={:.4}, df={}", lm, p, k);
        let fit = model_expansion::fit_dict(&[
            ("test", Value::Str("Breusch-Pagan".into())),
            ("lm_stat", Value::Float(lm)),
            ("p_value", Value::Float(p)),
            ("df", Value::Int(k as i64)),
            ("conclusion", Value::Str(conclusion.into())),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "BreuschPaganResult",
            fields,
        ))
    }

    pub(super) fn bptest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "bptest(df, y ~ x1+x2, id=\"entity_col\")".into(),
            ));
        }
        let df_name = match &args[0] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "first argument must be a DataFrame".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let formula_ast = self.resolve_formula(&args[1])?;
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(id, _)| id.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    self.rt_err(format!("bptest requires id= or xtset({df_name}, id, time)"))
                })?,
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        // OLS pooled to obtain residuals
        let ols_pooled = OLS::from_formula(&g_formula, &df, CovarianceType::NonRobust)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let resids = &y_vec - &x_mat.dot(&ols_pooled.params);
        // Convert id to usize
        let id_vals = get_col_f64(&df, &id_col)?;
        let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut next_id = 0usize;
        let entity_ids: Vec<usize> = id_vals
            .iter()
            .map(|&v| {
                let key = v as i64;
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();
        let (lm, p) = greeners::PanelDiagnostics::breusch_pagan_lm(&resids, &entity_ids)
            .map_err(HayashiError::Runtime)?;
        let n = resids.len();
        let n_entities = id_map.len();
        let t_bar = n as f64 / n_entities as f64;
        let sig = Self::panel_sig_stars(p);
        let conclusion = if p < 0.05 {
            "Reject H0 → individual effects present (use RE or FE)"
        } else {
            "Do not reject H0 → pooled OLS adequate (no individual effects)"
        };

        let mut display = String::new();
        display.push_str(&format!("\n{:=^62}\n", " Breusch-Pagan LM Test (RE) "));
        display.push_str(" H0: σ²_u = 0 — pooled OLS adequate\n");
        display.push_str(&format!("{:-^62}\n", ""));
        display.push_str(&format!(" LM = {lm:.4}    p-value = {p:.4}  {sig}\n"));
        display.push_str(&format!(
            "   n = {}   N = {}   T̄ ≈ {:.1}\n",
            n, n_entities, t_bar
        ));
        display.push_str(&format!(" Conclusion: {conclusion}\n"));
        display.push_str(&format!("{:=^62}\n", ""));

        let summary = format!("Breusch-Pagan LM test: LM={:.4}, p={:.4}", lm, p);
        let fit = model_expansion::fit_dict(&[
            ("test", Value::Str("Breusch-Pagan LM (RE)".into())),
            ("lm_stat", Value::Float(lm)),
            ("p_value", Value::Float(p)),
            ("n", Value::Int(n as i64)),
            ("n_entities", Value::Int(n_entities as i64)),
            ("t_bar", Value::Float(t_bar)),
            ("conclusion", Value::Str(conclusion.into())),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "BreuschPaganResult",
            fields,
        ))
    }

    pub(super) fn wooldridge(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        self.eval_wooldridge(args, opt_map)
    }

    pub(super) fn abtest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        self.eval_abtest(args, opt_map)
    }

    pub(super) fn sur(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "sur(df, y1~x1+x2, y2~x3+x4, ...) requires df + at least 2 formulas".into(),
            ));
        }
        let df_name = match &args[0] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "first argument must be a DataFrame".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let mut equations: Vec<greeners::SurEquation> = Vec::new();
        let mut eq_var_names: Vec<Vec<String>> = Vec::new();

        for arg in &args[1..] {
            let formula_ast = self.resolve_formula(arg)?;
            let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
            let (y, x) = df
                .to_design_matrix(&g_formula)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let var_names = df
                .formula_var_names(&g_formula)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            eq_var_names.push(var_names);
            equations.push(greeners::SurEquation {
                y,
                x,
                name: formula_ast.lhs.clone(),
            });
        }
        let result =
            greeners::SUR::fit(&equations).map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::SurResult(SurModel {
            result: Rc::new(result),
            eq_var_names,
        }))
    }

    pub(super) fn ic(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "ic() requires at least one model".into(),
            ));
        }
        struct IcRow {
            label: String,
            ll: f64,
            k: usize,
            n: usize,
            aic: f64,
            bic: f64,
        }
        let mut rows: Vec<IcRow> = Vec::new();
        for arg in args {
            let label = match arg {
                Expr::Var(name) => name.clone(),
                _ => "model".to_string(),
            };
            let val = self.eval_expr(arg)?;
            let (ll, k, n) = match &val {
                Value::OlsResult(m) => (
                    m.result.log_likelihood,
                    m.result.params.len(),
                    m.result.n_obs,
                ),
                Value::BinaryResult(b) => {
                    (b.result.log_likelihood, b.result.params.len(), b.x.nrows())
                }
                Value::PoissonResult(r) => (r.log_likelihood, r.params.len(), r.n_obs),
                Value::NegBinResult(r) => (r.log_likelihood, r.params.len(), r.n_obs),
                Value::OrderedResult(r) => (
                    r.log_likelihood,
                    r.params.len() + r.thresholds.len(),
                    r.n_obs,
                ),
                Value::TobitResult(r) => (r.log_likelihood, r.params.len(), r.n_obs),
                Value::MixedResult(r) => (r.log_likelihood, r.fixed_effects.len(), r.n_obs),
                Value::ZeroInflatedResult(r) => (
                    r.log_likelihood,
                    r.count_params.len() + r.inflate_params.len(),
                    r.n_obs,
                ),
                Value::RollingResult(_) | Value::RecursiveLSResult(_) => {
                    return Err(HayashiError::Runtime(format!(
                        "ic(): '{label}' has no log-likelihood — use print() for diagnostics"
                    )));
                }
                _ => {
                    return Err(HayashiError::Runtime(format!(
                    "ic(): model '{label}' has no log-likelihood available for ic() — use print()"
                )))
                }
            };
            let aic = -2.0 * ll + 2.0 * k as f64;
            let bic = -2.0 * ll + (k as f64) * (n as f64).ln();
            rows.push(IcRow {
                label,
                ll,
                k,
                n,
                aic,
                bic,
            });
        }
        // Sort by AIC
        rows.sort_by(|a, b| {
            a.aic
                .partial_cmp(&b.aic)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let min_aic = rows.first().map(|r| r.aic).unwrap_or(0.0);
        let delta_aics: Vec<f64> = rows.iter().map(|r| r.aic - min_aic).collect();
        let rel: Vec<f64> = delta_aics.iter().map(|d| (-d / 2.0).exp()).collect();
        let sum_rel: f64 = rel.iter().sum();
        let weights: Vec<f64> = rel
            .iter()
            .map(|w| if sum_rel > 0.0 { w / sum_rel } else { 0.0 })
            .collect();

        let best_aic = rows.first().map(|r| r.label.clone()).unwrap_or_default();
        let best_bic = rows
            .iter()
            .min_by(|a, b| a.bic.total_cmp(&b.bic))
            .map(|r| r.label.clone())
            .unwrap_or_default();

        let mut display = String::new();
        display.push_str(&format!("\n{:=^80}\n", " Information Criteria "));
        display.push_str(&format!(
            "{:<20} {:>6} {:>6} {:>12} {:>12} {:>8} {:>8}\n",
            "Model", "N", "k", "Log-Lik", "AIC", "ΔAIC", "BIC"
        ));
        display.push_str(&format!("{:-^80}\n", ""));
        for row in &rows {
            display.push_str(&format!(
                "{:<20} {:>6} {:>6} {:>12.4} {:>12.4} {:>8.4} {:>12.4}\n",
                row.label,
                row.n,
                row.k,
                row.ll,
                row.aic,
                row.aic - min_aic,
                row.bic
            ));
        }
        if rows.len() > 1 {
            display.push_str(&format!("{:-^80}\n", ""));
            display.push_str(&format!(
                " Best AIC: {}   Best BIC: {}\n",
                best_aic, best_bic
            ));
            display.push_str(&format!(
                " Akaike weights: {}\n",
                rows.iter()
                    .zip(weights.iter())
                    .map(|(r, w)| format!("{}={:.3}", r.label, w))
                    .collect::<Vec<_>>()
                    .join("  ")
            ));
        }
        display.push_str(&format!("{:=^80}\n", ""));

        let model_col: Vec<String> = rows.iter().map(|r| r.label.clone()).collect();
        let n_col: Vec<i64> = rows.iter().map(|r| r.n as i64).collect();
        let k_col: Vec<i64> = rows.iter().map(|r| r.k as i64).collect();
        let ll_col: Vec<f64> = rows.iter().map(|r| r.ll).collect();
        let aic_col: Vec<f64> = rows.iter().map(|r| r.aic).collect();
        let delta_col: Vec<f64> = rows.iter().map(|r| r.aic - min_aic).collect();
        let bic_col: Vec<f64> = rows.iter().map(|r| r.bic).collect();

        let mut columns = IndexMap::new();
        columns.insert(
            "model".into(),
            greeners::Column::String(ndarray::Array1::from(model_col)),
        );
        columns.insert(
            "n".into(),
            greeners::Column::Int(ndarray::Array1::from(n_col)),
        );
        columns.insert(
            "k".into(),
            greeners::Column::Int(ndarray::Array1::from(k_col)),
        );
        columns.insert(
            "log_likelihood".into(),
            greeners::Column::Float(ndarray::Array1::from(ll_col)),
        );
        columns.insert(
            "aic".into(),
            greeners::Column::Float(ndarray::Array1::from(aic_col)),
        );
        columns.insert(
            "delta_aic".into(),
            greeners::Column::Float(ndarray::Array1::from(delta_col)),
        );
        columns.insert(
            "bic".into(),
            greeners::Column::Float(ndarray::Array1::from(bic_col)),
        );
        columns.insert(
            "akaike_weight".into(),
            greeners::Column::Float(ndarray::Array1::from(weights)),
        );
        let comparison = Value::DataFrame(Arc::new(
            greeners::DataFrame::from_columns(columns)
                .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
        ));

        let summary = format!(
            "IC comparison ({} models); best AIC={}, best BIC={}",
            rows.len(),
            best_aic,
            best_bic
        );
        let fit = model_expansion::fit_dict(&[
            ("n_models", Value::Int(rows.len() as i64)),
            ("best_aic", Value::Str(best_aic)),
            ("best_bic", Value::Str(best_bic)),
            ("comparison", comparison),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "ICComparisonResult",
            fields,
        ))
    }

    pub(super) fn akaike_weights(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(self.rt_err("akaike_weights(m1, m2, ...) requires at least one model"));
        }
        let mut labels: Vec<String> = Vec::new();
        let mut aics: Vec<f64> = Vec::new();
        for arg in args {
            let label = match arg {
                Expr::Var(name) => name.clone(),
                _ => "model".to_string(),
            };
            let val = self.eval_expr(arg)?;
            let ll = match &val {
                Value::OlsResult(m) => m.result.log_likelihood,
                Value::BinaryResult(b) => b.result.log_likelihood,
                Value::PoissonResult(r) => r.log_likelihood,
                Value::NegBinResult(r) => r.log_likelihood,
                Value::OrderedResult(r) => r.log_likelihood,
                Value::TobitResult(r) => r.log_likelihood,
                Value::MixedResult(r) => r.log_likelihood,
                Value::ZeroInflatedResult(r) => r.log_likelihood,
                _ => {
                    return Err(
                        self.rt_err(format!("akaike_weights: '{label}' has no log-likelihood"))
                    )
                }
            };
            let k = match &val {
                Value::OlsResult(m) => m.result.params.len(),
                Value::BinaryResult(b) => b.result.params.len(),
                Value::PoissonResult(r) => r.params.len(),
                Value::NegBinResult(r) => r.params.len(),
                Value::OrderedResult(r) => r.params.len() + r.thresholds.len(),
                Value::TobitResult(r) => r.params.len(),
                Value::MixedResult(r) => r.fixed_effects.len(),
                Value::ZeroInflatedResult(r) => r.count_params.len() + r.inflate_params.len(),
                _ => 0,
            };
            labels.push(label);
            aics.push(-2.0 * ll + 2.0 * k as f64);
        }
        let (deltas, weights) = greeners::ModelSelection::akaike_weights(&aics);
        let mut out = std::collections::HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            out.insert(label.clone(), Value::Float(weights[i]));
        }
        // Print summary
        let sep = "─".repeat(50);
        println!("\nAkaike Weights");
        println!("{sep}");
        println!(
            "{:<20} {:>10} {:>10} {:>10}",
            "Model", "AIC", "ΔAIC", "Weight"
        );
        println!("{sep}");
        for (i, label) in labels.iter().enumerate() {
            println!(
                "{:<20} {:>10.2} {:>10.2} {:>10.4}",
                label, aics[i], deltas[i], weights[i]
            );
        }
        println!("{sep}");
        println!();
        Ok(Value::Dict(Arc::new(out)))
    }

    pub(super) fn lrtest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(
                self.rt_err("lrtest(m_restricted, m_unrestricted) requires two nested models")
            );
        }
        let v_r = self.eval_expr(&args[0])?;
        let v_u = self.eval_expr(&args[1])?;
        // Extract (log_likelihood, n_params) from a model Value
        let extract = |v: &Value| -> Result<(f64, usize)> {
            Ok(match v {
            Value::OlsResult(m) => (m.result.log_likelihood, m.result.params.len()),
            Value::BinaryResult(b) => (b.result.log_likelihood, b.result.params.len()),
            Value::PoissonResult(r) => (r.log_likelihood, r.params.len()),
            Value::NegBinResult(r) => (r.log_likelihood, r.params.len()),
            Value::OrderedResult(r) => {
                (r.log_likelihood, r.params.len() + r.thresholds.len())
            }
            Value::TobitResult(r) => (r.log_likelihood, r.params.len()),
            Value::MixedResult(r) => (r.log_likelihood, r.fixed_effects.len()),
            Value::ZeroInflatedResult(r) => (
                r.log_likelihood,
                r.count_params.len() + r.inflate_params.len(),
            ),
            Value::GlmResult(r) => (r.log_likelihood, r.params.len()),
            Value::GarchResult(r) => (r.log_likelihood, r.params.len()),
            Value::ArimaResult(r) => {
                (r.log_likelihood, r.ar_params.len() + r.ma_params.len() + 1)
            }
            _ => {
                return Err(HayashiError::Runtime(
                    "lrtest: model has no log-likelihood — supports OLS, logit/probit, Poisson, NegBin, Tobit, Ordered, Mixed, ZI, GLM, GARCH, ARIMA".into(),
                ))
            }
        })
        };
        let (ll_r, k_r) = extract(&v_r)?;
        let (ll_u, k_u) = extract(&v_u)?;
        let result = greeners::ModelSelection::lr_test(ll_r, ll_u, k_r, k_u)
            .map_err(HayashiError::Runtime)?;
        print!("{result}");
        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("Likelihood-Ratio Test".into()));
        map.insert("lr_stat".into(), Value::Float(result.lr_stat));
        map.insert("df".into(), Value::Int(result.df as i64));
        map.insert("p_value".into(), Value::Float(result.p_value));
        map.insert("ll_restricted".into(), Value::Float(result.ll_restricted));
        map.insert(
            "ll_unrestricted".into(),
            Value::Float(result.ll_unrestricted),
        );
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn fe(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let (df, mut g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        // FE removes the intercept via within-transform; force intercept=false
        g_formula.intercept = false;

        // try int; fall back to float→int; then to string
        let result = if let Ok(ids) = df.get_int(&id_col) {
            let ids_vec: Vec<i64> = ids.to_vec();
            FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?
        } else if let Ok(floats) = df.get(&id_col) {
            let ids_vec: Vec<i64> = floats.iter().map(|&v| v as i64).collect();
            FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?
        } else if let Ok(ids) = df.get_string(&id_col) {
            let ids_vec: Vec<String> = ids.to_vec();
            FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?
        } else {
            return Err(HayashiError::Runtime(format!(
                "column '{id_col}' not found or not usable as entity ID"
            )));
        };

        Ok(Value::PanelResult(Rc::new(result)))
    }

    pub(super) fn re(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        // accepts float column of integer values (e.g. idcode read as f64)
        let ids_owned: ndarray::Array1<i64>;
        let ids = match df.get_int(&id_col) {
            Ok(arr) => arr,
            Err(_) => {
                let floats = df.get(id_col.as_str()).map_err(|_| {
                    HayashiError::Runtime(format!("column '{id_col}' must be integer for re()"))
                })?;
                ids_owned = floats.mapv(|v| v as i64);
                &ids_owned
            }
        };

        let result = RandomEffects::from_formula(&g_formula, &df, ids)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::ReResult(Rc::new(result)))
    }

    pub(super) fn ftest_fe(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        // ftest_fe(formula, df, id=col)
        // H₀: all individual effects are zero (pooled OLS adequate)
        // H₁: individual effects exist (use FE)
        let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let (df, g_formula_base, _display) = self.prepare_formula(&formula_ast, &df)?;

        // FE (within) — sem intercept
        let mut g_formula_fe = g_formula_base.clone();
        g_formula_fe.intercept = false;

        let entity_ids_fe: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "ftest_fe: column '{id_col}' not found"
            )));
        };

        let fe = FixedEffects::from_formula(&g_formula_fe, &df, &entity_ids_fe)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        // Pooled OLS (com intercept)
        let g_formula_ols = g_formula_base;
        let (y_pool, x_pool) = df
            .to_design_matrix(&g_formula_ols)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let ols = greeners::OLS::fit(&y_pool, &x_pool, greeners::CovarianceType::NonRobust)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let ssr_pooled = ols.sigma.powi(2) * ols.df_resid as f64;
        let ssr_fe = fe.sigma.powi(2) * fe.df_resid as f64;
        let n = fe.n_obs;
        let n_entities = fe.n_entities;
        let k = fe.params.len();

        let (f_stat, p) =
            greeners::PanelDiagnostics::f_test_fixed_effects(ssr_pooled, ssr_fe, n, n_entities, k)
                .map_err(HayashiError::Runtime)?;

        let df_num = n_entities - 1;
        let df_denom = n - n_entities - k;
        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        let verdict = if p < 0.05 {
            "Reject H₀ → individual fixed effects are significant (use FE)"
        } else {
            "Do not reject H₀ → pooled OLS adequate (individual effects not significant)"
        };

        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        let mut out = String::new();
        out.push_str(&format!("\n{thick}\n"));
        out.push_str(" F-test: Fixed Effects vs Pooled OLS\n");
        out.push_str(" H₀: all individual effects are zero\n");
        out.push_str(&format!("{thick}\n"));
        out.push_str("\n── Sum of Squared Residuals\n");
        out.push_str(&format!("   SSR pooled = {:.6}\n", ssr_pooled));
        out.push_str(&format!("   SSR FE     = {:.6}\n", ssr_fe));
        out.push_str("\n── Statistic\n");
        out.push_str(&format!(
            "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
            df_num, df_denom, f_stat, p, sig
        ));
        out.push_str("\n── Conclusion\n");
        out.push_str(&format!("   {}\n", verdict));
        out.push_str(&format!("\n{thin}\n"));
        out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
        out.push_str(&format!("{thick}\n"));
        let mut fields = HashMap::new();
        fields.insert(
            "test".into(),
            Value::Str("F-test: Fixed Effects vs Pooled OLS".into()),
        );
        fields.insert("f_stat".into(), Value::Float(f_stat));
        fields.insert("df_num".into(), Value::Int(df_num as i64));
        fields.insert("df_denom".into(), Value::Int(df_denom as i64));
        fields.insert("p_value".into(), Value::Float(p));
        fields.insert("conclusion".into(), Value::Str(verdict.into()));
        fields.insert("ssr_pooled".into(), Value::Float(ssr_pooled));
        fields.insert("ssr_fe".into(), Value::Float(ssr_fe));
        Ok(diag_with(out, fields))
    }

    pub(super) fn pesaran_cd(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        // pesaran_cd(formula, df, id=col)
        // H₀: residuals independent across entities (no cross-sectional dependence)
        // H₁: cross-sectional dependence present
        let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        // OLS pooled for residuals
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let ols = greeners::OLS::fit(&y_vec, &x_mat, greeners::CovarianceType::NonRobust)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let residuals = ols.residuals(&y_vec, &x_mat);

        // Entity IDs
        let entity_ids: Vec<usize> = if let Ok(ids) = df.get_int(&id_col) {
            ids.iter().map(|&v| v as usize).collect()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as usize).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "pesaran_cd: column '{id_col}' not found"
            )));
        };

        let n_entities = {
            let mut s = std::collections::HashSet::new();
            for &id in &entity_ids {
                s.insert(id);
            }
            s.len()
        };
        let t_bar = residuals.len() as f64 / n_entities as f64;

        let (cd, p) = greeners::PanelDiagnostics::pesaran_cd(&residuals, &entity_ids)
            .map_err(HayashiError::Runtime)?;

        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        let verdict = if p < 0.05 {
            "Reject H₀ → cross-sectional dependence present"
        } else {
            "Do not reject H₀ → no evidence of cross-sectional dependence"
        };

        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        let mut out = String::new();
        out.push_str(&format!("\n{thick}\n"));
        out.push_str(" Pesaran CD Test (cross-sectional dependence)\n");
        out.push_str(" H₀: ρ_ij = 0 for all i≠j  (residuals independent)\n");
        out.push_str(&format!("{thick}\n"));
        out.push_str(&format!(
            "\n── Panel: N={} entities   T̄≈{:.1}\n",
            n_entities, t_bar
        ));
        out.push_str("\n── Statistic\n");
        out.push_str(&format!(
            "   CD ~ N(0,1) = {:.4}   p = {:.4}  {}\n",
            cd, p, sig
        ));
        out.push_str("\n── Conclusion\n");
        out.push_str(&format!("   {}\n", verdict));
        out.push_str(&format!("\n{thin}\n"));
        out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
        out.push_str(&format!("{thick}\n"));
        let mut fields = HashMap::new();
        fields.insert("test".into(), Value::Str("Pesaran CD Test".into()));
        fields.insert("cd_stat".into(), Value::Float(cd));
        fields.insert("p_value".into(), Value::Float(p));
        fields.insert("n_entities".into(), Value::Int(n_entities as i64));
        fields.insert("t_bar".into(), Value::Float(t_bar));
        fields.insert("conclusion".into(), Value::Str(verdict.into()));
        Ok(diag_with(out, fields))
    }

    pub(super) fn bplm(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        // bplm(formula, df, id=col)
        // H₀: no individual effects (σ²_u = 0) — pooled OLS adequate
        // H₁: individual effects exist — use FE or RE
        let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        // OLS pooled to obtain residuals
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let ols = greeners::OLS::fit(&y_vec, &x_mat, greeners::CovarianceType::NonRobust)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let residuals = ols.residuals(&y_vec, &x_mat);

        // Entity IDs → usize
        let entity_ids: Vec<usize> = if let Ok(ids) = df.get_int(&id_col) {
            ids.iter().map(|&v| v as usize).collect()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as usize).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "bplm: column '{id_col}' not found or not usable as ID"
            )));
        };

        let n = residuals.len();
        let n_entities = {
            let mut ids_set = std::collections::HashSet::new();
            for &id in &entity_ids {
                ids_set.insert(id);
            }
            ids_set.len()
        };
        let t_bar = n as f64 / n_entities as f64;

        let (lm, p) = greeners::PanelDiagnostics::breusch_pagan_lm(&residuals, &entity_ids)
            .map_err(HayashiError::Runtime)?;

        let sig = Self::panel_sig_stars(p);
        let verdict = if p < 0.05 {
            "Reject H₀ → individual effects present (use FE or RE)"
        } else {
            "Do not reject H₀ → pooled OLS adequate (no individual effects)"
        };

        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        let mut display = String::new();
        display.push_str(&format!("\n{thick}\n"));
        display.push_str(" Breusch-Pagan LM Test (individual effects)\n");
        display.push_str(" H₀: σ²_u = 0  (no individual effects)\n");
        display.push_str(&format!("{thick}\n"));
        display.push_str("\n── Panel Data\n");
        display.push_str(&format!(
            "   n = {}   N = {}   T̄ ≈ {:.1}\n",
            n, n_entities, t_bar
        ));
        display.push_str("\n── Statistic\n");
        display.push_str(&format!(
            "   LM ~ χ²(1) = {:.4}   p = {:.4}  {}\n",
            lm, p, sig
        ));
        display.push_str("\n── Conclusion\n");
        display.push_str(&format!("   {}\n", verdict));
        display.push_str(&format!("\n{thin}\n"));
        display.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
        display.push_str(&format!("{thick}\n"));

        let summary = format!("Breusch-Pagan LM test: LM={:.4}, p={:.4}", lm, p);
        let fit = model_expansion::fit_dict(&[
            ("test", Value::Str("Breusch-Pagan LM Test".into())),
            ("lm_stat", Value::Float(lm)),
            ("p_value", Value::Float(p)),
            ("n", Value::Int(n as i64)),
            ("n_entities", Value::Int(n_entities as i64)),
            ("t_bar", Value::Float(t_bar)),
            ("conclusion", Value::Str(verdict.into())),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "BreuschPaganResult",
            fields,
        ))
    }

    pub(super) fn chamberlain(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        // chamberlain(formula, df, id=col, time=col)
        // H₀: Π_s = 0 for all s (RE consistent)
        // H₁: at least one Π_s ≠ 0 (effects correlated with X — use FE)
        // Generalization of Mundlak: uses values in ALL periods, not just the mean
        let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let time_col = self.get_time_col(&df_name, opt_map)?;

        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "chamberlain: id column '{id_col}' not found"
            )));
        };

        let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
            arr.to_vec()
        } else if let Ok(arr) = df.get_int(&time_col) {
            arr.iter().map(|&v| v as f64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "chamberlain: time column '{time_col}' not found"
            )));
        };

        let (f_stat, p, k_active, df_denom, n_entities, t_count) =
            greeners::PanelDiagnostics::chamberlain(&y_vec, &x_mat, &entity_ids, &time_vals)
                .map_err(HayashiError::Runtime)?;

        let n_obs = y_vec.len();
        let df1 = k_active;

        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        let verdict = if p < 0.05 {
            "Reject H₀ → individual effects correlated with X (prefer FE)"
        } else {
            "Do not reject H₀ → RE consistent (no period-specific correlation)"
        };

        let thick = "═".repeat(70);
        let thin = "─".repeat(70);
        let mut out = String::new();
        out.push_str(&format!("\n{thick}\n"));
        out.push_str(" Chamberlain Test (period-specific correlation with individual effects)\n");
        out.push_str(" H₀: Π_s = 0 ∀s  (RE consistent)\n");
        out.push_str(&format!("{thick}\n"));
        out.push_str(&format!(
            "\n── Panel: n={} obs   N={} entities   T={} periods\n",
            n_obs, n_entities, t_count
        ));
        out.push_str(&format!(
            "   Chamberlain augmentation columns: {} (k×T, after removing zero-variance)\n",
            k_active
        ));
        if t_count > 6 {
            out.push_str(&format!(
                "   ⚠ T={} — with large T the test has low power in finite samples\n",
                t_count
            ));
        }
        out.push_str("\n── Joint test H₀: all Π_s = 0\n");
        out.push_str(&format!(
            "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
            df1, df_denom, f_stat, p, sig
        ));
        out.push_str("\n── Conclusion\n");
        out.push_str(&format!("   {}\n", verdict));
        out.push_str(&format!("\n{thin}\n"));
        out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
        out.push_str("   More general test than Mundlak — includes values in all T periods\n");
        out.push_str(&format!("{thick}\n"));
        let mut fields = HashMap::new();
        fields.insert("test".into(), Value::Str("Chamberlain Test".into()));
        fields.insert("f_stat".into(), Value::Float(f_stat));
        fields.insert("df_num".into(), Value::Int(df1 as i64));
        fields.insert("df_denom".into(), Value::Int(df_denom as i64));
        fields.insert("p_value".into(), Value::Float(p));
        fields.insert("n_obs".into(), Value::Int(n_obs as i64));
        fields.insert("n_entities".into(), Value::Int(n_entities as i64));
        fields.insert("n_periods".into(), Value::Int(t_count as i64));
        fields.insert("k_active".into(), Value::Int(k_active as i64));
        fields.insert("conclusion".into(), Value::Str(verdict.into()));
        Ok(diag_with(out, fields))
    }

    #[allow(non_snake_case)]
    pub(super) fn mundlak_OLD_REMOVED(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "mundlak: column '{id_col}' not found"
            )));
        };

        // Names of time-varying regressors (excluding "const")
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let non_const_names: Vec<&str> = var_names
            .iter()
            .filter(|n| n.as_str() != "const")
            .map(|s| s.as_str())
            .collect();

        let n = y_vec.len();
        let n_entities = {
            let mut s = std::collections::HashSet::new();
            for &id in &entity_ids {
                s.insert(id);
            }
            s.len()
        };

        let (f_stat, p, k, gamma_hat, gamma_se) =
            greeners::PanelDiagnostics::mundlak(&y_vec, &x_mat, &entity_ids)
                .map_err(HayashiError::Runtime)?;

        let df1 = k;
        let df2_exact = if n > 2 * k + 1 { n - 2 * k - 1 } else { 1 };

        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        let verdict = if p < 0.05 {
            "Reject H₀ → individual effects correlated with X (prefer FE)"
        } else {
            "Do not reject H₀ → RE consistent (no evidence of correlation with effects)"
        };

        let thick = "═".repeat(70);
        let thin = "─".repeat(70);
        let mut out = String::new();
        out.push_str(&format!("\n{thick}\n"));
        out.push_str(" Mundlak Test (correlation between regressors and individual effects)\n");
        out.push_str(" H₀: γ = 0  (RE consistent)\n");
        out.push_str(&format!("{thick}\n"));
        out.push_str(&format!(
            "\n── Panel: n={} obs   N={} entities   k={} time-varying regressors\n",
            n, n_entities, k
        ));
        out.push_str("\n── Coefficients on individual means (X̄_i)\n");
        out.push_str(&format!(
            "   {:<18} {:>10}  {:>10}  {:>8}\n",
            "Variable (X̄)", "γ̂", "SE", "t"
        ));
        out.push_str(&format!("   {}\n", "─".repeat(52)));
        for i in 0..k {
            let t_i = if gamma_se[i] > 1e-15 {
                gamma_hat[i] / gamma_se[i]
            } else {
                f64::NAN
            };
            let name = non_const_names.get(i).copied().unwrap_or("?");
            out.push_str(&format!(
                "   {:<18} {:>10.4}  {:>10.4}  {:>8.3}\n",
                format!("{}̄", name),
                gamma_hat[i],
                gamma_se[i],
                t_i
            ));
        }
        out.push_str("\n── Joint test H₀: γ = 0\n");
        out.push_str(&format!(
            "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
            df1, df2_exact, f_stat, p, sig
        ));
        out.push_str("\n── Conclusion\n");
        out.push_str(&format!("   {}\n", verdict));
        out.push_str(&format!("\n{thin}\n"));
        out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
        out.push_str(&format!("{thick}\n"));
        let mut fields = HashMap::new();
        fields.insert("test".into(), Value::Str("Mundlak Test".into()));
        fields.insert("f_stat".into(), Value::Float(f_stat));
        fields.insert("df_num".into(), Value::Int(df1 as i64));
        fields.insert("df_denom".into(), Value::Int(df2_exact as i64));
        fields.insert("p_value".into(), Value::Float(p));
        fields.insert("n_obs".into(), Value::Int(n as i64));
        fields.insert("n_entities".into(), Value::Int(n_entities as i64));
        fields.insert("k".into(), Value::Int(k as i64));
        fields.insert("conclusion".into(), Value::Str(verdict.into()));
        fields.insert(
            "variable".into(),
            Value::List(Arc::new(
                non_const_names
                    .iter()
                    .map(|s| Value::Str(s.to_string()))
                    .collect(),
            )),
        );
        fields.insert(
            "gamma".into(),
            Value::List(Arc::new(
                gamma_hat.iter().map(|&v| Value::Float(v)).collect(),
            )),
        );
        fields.insert(
            "gamma_se".into(),
            Value::List(Arc::new(
                gamma_se.iter().map(|&v| Value::Float(v)).collect(),
            )),
        );
        Ok(diag_with(out, fields))
    }

    pub(super) fn ab(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let time_col = self.get_time_col(&df_name, opt_map)?;

        let max_lags: usize = match opt_map.get("lags") {
            Some(Value::Int(v)) => (*v).max(1) as usize,
            Some(Value::Float(v)) => (*v as i64).max(1) as usize,
            None => 2,
            _ => {
                return Err(HayashiError::Runtime(
                    "ab(): lags must be positive integer".into(),
                ))
            }
        };

        let two_step: bool = match opt_map.get("step") {
            Some(Value::Int(2)) => true,
            Some(Value::Float(v)) if *v as i64 == 2 => true,
            Some(Value::Int(_)) | Some(Value::Float(_)) | None => false,
            _ => return Err(HayashiError::Runtime("ab(): step must be 1 or 2".into())),
        };

        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "ab: id column '{id_col}' not found"
            )));
        };

        let time_ids: Vec<i64> = if let Ok(ids) = df.get_int(&time_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&time_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "ab: time column '{time_col}' not found"
            )));
        };

        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let result = greeners::ArellanoBond::fit(
            &y_vec,
            &x_mat,
            &entity_ids,
            &time_ids,
            max_lags,
            two_step,
            Some(var_names),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::AbResult(Rc::new(result)))
    }

    pub(super) fn gmm(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "gmm(endog_formula, instrument_formula, dataframe)".into(),
            ));
        }
        let endog_ast = self.resolve_formula(&args[0])?;
        let instr_ast = self.resolve_formula(&args[1])?;
        let df_name = match &args[2] {
            Expr::Var(name) => name.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "third argument must be a DataFrame variable".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };

        let (df_endog, g_endog, _) = self.prepare_formula(&endog_ast, &df)?;

        let g_instr = if instr_ast.lhs.is_empty() {
            let (_, g_i, _) = self.prepare_formula(&instr_ast, &df)?;
            GFormula {
                dependent: String::new(),
                independents: g_i.independents,
                intercept: true,
            }
        } else {
            let (_, g_i, _) = self.prepare_formula(&instr_ast, &df)?;
            g_i
        };

        let (y, x) = df_endog
            .to_design_matrix(&g_endog)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let z = {
            let n_rows = df_endog.n_rows();
            let n_cols = g_instr.independents.len() + if g_instr.intercept { 1 } else { 0 };
            let mut z_mat = ndarray::Array2::<f64>::zeros((n_rows, n_cols));
            let mut col_idx = 0;
            if g_instr.intercept {
                for i in 0..n_rows {
                    z_mat[[i, 0]] = 1.0;
                }
                col_idx = 1;
            }
            for (j, var_name) in g_instr.independents.iter().enumerate() {
                let col_data = df_endog
                    .get(var_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                for i in 0..n_rows {
                    z_mat[[i, col_idx + j]] = col_data[i];
                }
            }
            z_mat
        };

        let result =
            greeners::GMM::fit(&y, &x, &z).map_err(|e| self.rt_err(format!("gmm: {e}")))?;
        Ok(Value::GmmResult(Rc::new(result)))
    }

    pub(super) fn sysgmm(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let time_col = self.get_time_col(&df_name, opt_map)?;

        let max_lags: usize = match opt_map.get("lags") {
            Some(Value::Int(v)) => (*v).max(1) as usize,
            Some(Value::Float(v)) => (*v as i64).max(1) as usize,
            None => 2,
            _ => {
                return Err(HayashiError::Runtime(
                    "sysgmm(): lags must be positive integer".into(),
                ))
            }
        };

        let two_step: bool = match opt_map.get("step") {
            Some(Value::Int(2)) => true,
            Some(Value::Float(v)) if *v as i64 == 2 => true,
            Some(Value::Int(_)) | Some(Value::Float(_)) | None => false,
            _ => {
                return Err(HayashiError::Runtime(
                    "sysgmm(): step must be 1 or 2".into(),
                ))
            }
        };

        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "sysgmm: id column '{id_col}' not found"
            )));
        };

        let time_ids: Vec<i64> = if let Ok(ids) = df.get_int(&time_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&time_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "sysgmm: time column '{time_col}' not found"
            )));
        };

        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let result = greeners::SystemGmm::fit(
            &y_vec,
            &x_mat,
            &entity_ids,
            &time_ids,
            max_lags,
            two_step,
            Some(var_names),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::SysGmmResult(Rc::new(result)))
    }

    pub(super) fn feiv(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "feiv() requires (structural_formula, instrument_formula, df, id=col)".into(),
            ));
        }

        let endog_ast = self.resolve_formula(&args[0])?;
        let instr_ast = self.resolve_formula(&args[1])?;
        let df_name = match &args[2] {
            Expr::Var(name) => name.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "feiv(): third argument must be the DataFrame name".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "feiv: '{df_name}' is not a DataFrame"
                )))
            }
        };

        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(
                    "feiv(): id=col option is required".into(),
                ))
            }
        };

        // structural formula → y and X (no constant, FE absorbs it)
        let (df_endog2, g_endog, _) = self.prepare_formula(&endog_ast, &df)?;
        let (y_vec, x_mat) = df_endog2
            .to_design_matrix(&g_endog)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        // instrument formula → Z (no constant); materializamos para suportar exprs
        let (_, g_instr2, _) = self.prepare_formula(&instr_ast, &df)?;
        let instr_vars: Vec<String> = g_instr2.independents;

        let n = y_vec.len();
        let l = instr_vars.len();
        if l == 0 {
            return Err(HayashiError::Runtime(
                "feiv(): instrument formula must have at least one instrument".into(),
            ));
        }
        let mut z_mat = ndarray::Array2::<f64>::zeros((n, l));
        for (j, col_name) in instr_vars.iter().enumerate() {
            let col = df.get(col_name).map_err(|_| {
                HayashiError::Runtime(format!(
                    "feiv: instrument '{col_name}' not found in DataFrame"
                ))
            })?;
            for (i, &v) in col.iter().enumerate() {
                z_mat[[i, j]] = v;
            }
        }

        // entity IDs
        let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "feiv: id column '{id_col}' not found"
            )));
        };

        let var_names = df
            .formula_var_names(&g_endog)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let result = greeners::FE2SLS::fit(&y_vec, &x_mat, &z_mat, &entity_ids, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::FE2SLSResult(Rc::new(result)))
    }

    pub(super) fn pcse(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let time_col = self.get_time_col(&df_name, opt_map)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let entity_ids =
            Self::col_as_i64(&df, &id_col).map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let time_ids =
            Self::col_as_i64(&df, &time_col).map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result = greeners::PCSE::fit(&y_vec, &x_mat, &entity_ids, &time_ids, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::PcseResult(Rc::new(result)))
    }

    pub(super) fn xtgls(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let time_col = self.get_time_col(&df_name, opt_map)?;
        let panels_opt = match opt_map.get("panels") {
            Some(Value::Str(s)) if s == "corr" => greeners::GlsPanels::Correlated,
            Some(Value::Str(s)) if s == "hetero" || s == "heteroscedastic" => {
                greeners::GlsPanels::Hetero
            }
            None => greeners::GlsPanels::Hetero,
            _ => {
                return Err(HayashiError::Runtime(
                    "xtgls(): panels must be \"hetero\" or \"corr\"".into(),
                ))
            }
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let entity_ids =
            Self::col_as_i64(&df, &id_col).map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let time_ids =
            Self::col_as_i64(&df, &time_col).map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result = greeners::PanelGLS::fit(
            &y_vec,
            &x_mat,
            &entity_ids,
            &time_ids,
            panels_opt,
            Some(var_names),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::PanelGlsResult(Rc::new(result)))
    }

    pub(super) fn ab_test(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        // ab_test(formula, df, id=col, time=col)
        // Tests serial autocorrelation in residuals of the first-differenced equation.
        // m1: MUST reject H₀ (FD induces AR(1) by construction)
        // m2: MUST NOT reject H₀ (validates GMM instruments y_{i,t-2})
        let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let time_col = self.get_time_col(&df_name, opt_map)?;

        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "ab_test: id column '{id_col}' not found"
            )));
        };

        let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
            arr.to_vec()
        } else if let Ok(arr) = df.get_int(&time_col) {
            arr.iter().map(|&v| v as f64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "ab_test: time column '{time_col}' not found"
            )));
        };

        let n_entities = {
            let mut s = std::collections::HashSet::new();
            for &id in &entity_ids {
                s.insert(id);
            }
            s.len()
        };

        let (m1, p1, m2, p2) =
            greeners::PanelDiagnostics::arellano_bond_test(&y_vec, &x_mat, &entity_ids, &time_vals)
                .map_err(HayashiError::Runtime)?;

        let sig = |p: f64| {
            if p < 0.01 {
                "***"
            } else if p < 0.05 {
                "**"
            } else if p < 0.10 {
                "*"
            } else {
                ""
            }
        };
        let n_obs = y_vec.len();

        let thick = "═".repeat(66);
        let thin = "─".repeat(66);
        let mut out = String::new();
        out.push_str(&format!("\n{thick}\n"));
        out.push_str(
            " Arellano-Bond Test (serial autocorrelation — first-differenced residuals)\n",
        );
        out.push_str(&format!("{thick}\n"));
        out.push_str(&format!(
            "\n── Panel: n={} obs   N={} entities\n",
            n_obs, n_entities
        ));
        out.push_str("\n── Statistics  z ~ N(0,1)   H₀: no autocorrelation of order p\n");
        out.push_str(&format!("   {:-^52}\n", ""));
        out.push_str(&format!(
            "   {:>4}  {:>10}  {:>10}  {:>6}  {}\n",
            "p", "z", "p-value", "sig", "Interpretation"
        ));
        out.push_str(&format!("   {:-^52}\n", ""));
        let interp1 = if p1 < 0.05 {
            "OK — FD induces AR(1) (expected)"
        } else {
            "Unexpected — check model"
        };
        let interp2 = if p2 >= 0.05 {
            "OK — instruments valid"
        } else {
            "Warning — AR(2) detected"
        };
        out.push_str(&format!(
            "   {:>4}  {:>10.4}  {:>10.4}  {:>6}  {}\n",
            1,
            m1,
            p1,
            sig(p1),
            interp1
        ));
        out.push_str(&format!(
            "   {:>4}  {:>10.4}  {:>10.4}  {:>6}  {}\n",
            2,
            m2,
            p2,
            sig(p2),
            interp2
        ));
        out.push_str(&format!("   {:-^52}\n", ""));
        out.push_str("\n── Conclusion\n");
        if p1 < 0.05 && p2 >= 0.05 {
            out.push_str(
                "   m1 rejects and m2 does not reject → structure consistent with valid GMM\n",
            );
        } else if p1 >= 0.05 {
            out.push_str("   m1 does not reject H₀ → check specification (AR(1) expected in FD)\n");
        } else {
            out.push_str(
                "   m2 rejects H₀ → AR(2) in residuals; y_{t-2} instruments may be invalid\n",
            );
            out.push_str("   Consider using more distant lags (y_{t-3}, ...) as instruments\n");
        }
        out.push_str(&format!("\n{thin}\n"));
        out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
        out.push_str("   Variance estimated via sandwich (Σ_i of cross-products by entity)\n");
        out.push_str(&format!("{thick}\n"));
        let mut fields = HashMap::new();
        fields.insert("test".into(), Value::Str("Arellano-Bond Test".into()));
        fields.insert("m1_z".into(), Value::Float(m1));
        fields.insert("m1_p".into(), Value::Float(p1));
        fields.insert("m2_z".into(), Value::Float(m2));
        fields.insert("m2_p".into(), Value::Float(p2));
        fields.insert("n_obs".into(), Value::Int(n_obs as i64));
        fields.insert("n_entities".into(), Value::Int(n_entities as i64));
        let verdict = if p1 < 0.05 && p2 >= 0.05 {
            "m1 rejects and m2 does not reject -> valid GMM"
        } else if p1 >= 0.05 {
            "m1 does not reject -> check specification"
        } else {
            "m2 rejects -> AR(2) detected; consider more distant lags"
        };
        fields.insert("conclusion".into(), Value::Str(verdict.into()));
        Ok(diag_with(out, fields))
    }

    #[allow(non_snake_case)]
    pub(super) fn wooldridge_OLD_REMOVED(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
        let time_col = self.get_time_col(&df_name, opt_map)?;

        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
            ids.to_vec()
        } else if let Ok(floats) = df.get(&id_col) {
            floats.iter().map(|&v| v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "wooldridge: id column '{id_col}' not found"
            )));
        };

        let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
            arr.to_vec()
        } else if let Ok(arr) = df.get_int(&time_col) {
            arr.iter().map(|&v| v as f64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "wooldridge: time column '{time_col}' not found"
            )));
        };

        let n_entities = {
            let mut s = std::collections::HashSet::new();
            for &id in &entity_ids {
                s.insert(id);
            }
            s.len()
        };

        let (rho, t_stat, p, n_pairs) =
            greeners::PanelDiagnostics::wooldridge_serial(&y_vec, &x_mat, &entity_ids, &time_vals)
                .map_err(HayashiError::Runtime)?;

        let df_t = n_entities - 1;
        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        let verdict = if p < 0.05 {
            "Reject H₀ → first-order serial autocorrelation present"
        } else {
            "Do not reject H₀ → no evidence of serial autocorrelation"
        };

        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        let mut out = String::new();
        out.push_str(&format!("\n{thick}\n"));
        out.push_str(" Wooldridge Test (panel serial autocorrelation)\n");
        out.push_str(" H₀: ρ = -0.5  (no autocorrelation in idiosyncratic errors)\n");
        out.push_str(&format!("{thick}\n"));
        out.push_str(&format!(
            "\n── Panel: N={} entities   used pairs={}   df={}\n",
            n_entities, n_pairs, df_t
        ));
        out.push_str("\n── Estimate\n");
        out.push_str(&format!("   ρ̂ = {:.4}   (H₀: ρ = -0.500)\n", rho));
        out.push_str("\n── Statistic\n");
        out.push_str(&format!(
            "   t({}) = {:.4}   p = {:.4}  {}\n",
            df_t, t_stat, p, sig
        ));
        out.push_str("\n── Conclusion\n");
        out.push_str(&format!("   {}\n", verdict));
        out.push_str(&format!("\n{thin}\n"));
        out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
        out.push_str("   (OLS standard SE — use cluster-robust SE for formal inference)\n");
        out.push_str(&format!("{thick}\n"));
        let mut fields = HashMap::new();
        fields.insert(
            "test".into(),
            Value::Str("Wooldridge Test (panel serial autocorrelation)".into()),
        );
        fields.insert("rho".into(), Value::Float(rho));
        fields.insert("t_stat".into(), Value::Float(t_stat));
        fields.insert("df".into(), Value::Int(df_t as i64));
        fields.insert("p_value".into(), Value::Float(p));
        fields.insert("n_entities".into(), Value::Int(n_entities as i64));
        fields.insert("n_pairs".into(), Value::Int(n_pairs as i64));
        fields.insert("conclusion".into(), Value::Str(verdict.into()));
        Ok(diag_with(out, fields))
    }

    pub(super) fn hausman(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime("hausman(fe_model, re_model)".into()));
        }

        let fe = match self.eval_expr(&args[0])? {
            Value::PanelResult(r) => r,
            _ => {
                return Err(HayashiError::Type(
                    "hausman(): first argument must be an FE model".into(),
                ))
            }
        };
        let re = match self.eval_expr(&args[1])? {
            Value::ReResult(r) => r,
            _ => {
                return Err(HayashiError::Type(
                    "hausman(): second argument must be an RE model".into(),
                ))
            }
        };

        // Common variables: FE has no intercept; RE has.
        // Align by name when available; otherwise assume same order.
        let fe_names: Vec<String> = fe
            .variable_names
            .as_ref()
            .cloned()
            .unwrap_or_else(|| (0..fe.params.len()).map(|i| format!("x{}", i)).collect());

        let re_names: Vec<String> = re
            .variable_names
            .as_ref()
            .cloned()
            .unwrap_or_else(|| (0..re.params.len()).map(|i| format!("x{}", i)).collect());

        // Pairs (β_FE, σ²_FE, β_RE, σ²_RE) for common variables (exclude intercept)
        let mut pairs: Vec<(String, f64, f64, f64, f64)> = Vec::new();
        for (i, fe_name) in fe_names.iter().enumerate() {
            if fe_name == "const" {
                continue;
            }
            if let Some(j) = re_names.iter().position(|n| n == fe_name) {
                pairs.push((
                    fe_name.clone(),
                    fe.params[i],
                    fe.std_errors[i].powi(2),
                    re.params[j],
                    re.std_errors[j].powi(2),
                ));
            }
        }

        if pairs.is_empty() {
            return Err(HayashiError::Runtime(
                "hausman: no common variable between FE and RE (check variable_names)".into(),
            ));
        }

        // H = Σ (β_FE - β_RE)² / (σ²_FE - σ²_RE)  for pairs where σ²_FE > σ²_RE
        let mut chi2 = 0.0;
        let mut df = 0usize;
        let mut skipped = 0usize;

        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        let mut out = String::new();

        out.push_str(&format!("\n{thick}\n"));
        out.push_str(" Hausman Test: FE vs RE\n");
        out.push_str(" H₀: individual effects uncorrelated with regressors (RE consistent)\n");
        out.push_str(&format!("{thick}\n"));
        out.push_str("\n── Common Coefficients\n");
        out.push_str(&format!(
            "   {:<20} {:>10} {:>10} {:>10}\n",
            "Variable", "β_FE", "β_RE", "Δβ"
        ));
        out.push_str(&format!("   {thin}\n"));

        for (name, bfe, vfe, bre, vre) in &pairs {
            let diff = bfe - bre;
            let dvar = vfe - vre;
            out.push_str(&format!(
                "   {:<20} {:>10.4} {:>10.4} {:>10.4}\n",
                name, bfe, bre, diff
            ));
            if dvar > 1e-15 {
                chi2 += diff.powi(2) / dvar;
                df += 1;
            } else {
                skipped += 1;
            }
        }

        if df == 0 {
            out.push_str("\n   [!] Var(β_FE) ≤ Var(β_RE) for all coefficients.\n");
            out.push_str("       Statistic undefined — check model specification.\n");
            out.push_str(&format!("\n{thick}\n"));
            return Ok(diag(out));
        }

        let p = greeners::chi2_pvalue(chi2, df as f64);

        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        let verdict = if p < 0.05 {
            "Reject H₀ → use FIXED EFFECTS (RE may be inconsistent)"
        } else {
            "Do not reject H₀ → RANDOM EFFECTS is consistent and efficient"
        };

        out.push_str("\n── Statistic\n");
        out.push_str(&format!(
            "   χ²({}) = {:.4}   p = {:.4}  {}\n",
            df, chi2, p, sig
        ));
        if skipped > 0 {
            out.push_str(&format!(
                "   ({} coefficient(s) excluded: Var(β_FE) ≤ Var(β_RE))\n",
                skipped
            ));
        }
        out.push_str("\n── Conclusion\n");
        out.push_str(&format!("   {}\n", verdict));
        out.push_str(&format!("\n{thin}\n"));
        out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
        out.push_str(&format!("{thick}\n"));
        let mut fields = HashMap::new();
        fields.insert("test".into(), Value::Str("Hausman FE vs RE".into()));
        fields.insert("chi2".into(), Value::Float(chi2));
        fields.insert("df".into(), Value::Int(df as i64));
        fields.insert("p_value".into(), Value::Float(p));
        fields.insert("conclusion".into(), Value::Str(verdict.into()));
        fields.insert("n_compared".into(), Value::Int(pairs.len() as i64));
        fields.insert("n_skipped".into(), Value::Int(skipped as i64));
        Ok(diag_with(out, fields))
    }

    pub(super) fn hausman_robust(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "hausman_robust(fe_model, re_model)".into(),
            ));
        }

        let fe = match self.eval_expr(&args[0])? {
            Value::PanelResult(r) => r,
            _ => {
                return Err(HayashiError::Type(
                    "hausman_robust(): first argument must be an FE model".into(),
                ))
            }
        };
        let re = match self.eval_expr(&args[1])? {
            Value::ReResult(r) => r,
            _ => {
                return Err(HayashiError::Type(
                    "hausman_robust(): second argument must be an RE model".into(),
                ))
            }
        };

        // Align parameters by variable names (exclude intercept from RE)
        let fe_names: Vec<String> = fe
            .variable_names
            .clone()
            .unwrap_or_else(|| (0..fe.params.len()).map(|i| format!("x{}", i)).collect());
        let re_names: Vec<String> = re
            .variable_names
            .clone()
            .unwrap_or_else(|| (0..re.params.len()).map(|i| format!("x{}", i)).collect());

        // Find common variables (exclude const/intercept)
        let mut common_indices: Vec<(usize, usize)> = Vec::new();
        for (i, fe_name) in fe_names.iter().enumerate() {
            if fe_name == "const" || fe_name == "_cons" {
                continue;
            }
            if let Some(j) = re_names.iter().position(|n| n == fe_name) {
                common_indices.push((i, j));
            }
        }

        if common_indices.is_empty() {
            return Err(HayashiError::Runtime(
                "hausman_robust: no common variables between FE and RE".into(),
            ));
        }

        let k = common_indices.len();
        let mut fe_beta = ndarray::Array1::<f64>::zeros(k);
        let mut re_beta = ndarray::Array1::<f64>::zeros(k);
        let mut fe_vcov = ndarray::Array2::<f64>::zeros((k, k));
        let mut re_vcov = ndarray::Array2::<f64>::zeros((k, k));
        for (idx, (i, j)) in common_indices.iter().enumerate() {
            fe_beta[idx] = fe.params[*i];
            re_beta[idx] = re.params[*j];
            fe_vcov[(idx, idx)] = fe.std_errors[*i].powi(2);
            re_vcov[(idx, idx)] = re.std_errors[*j].powi(2);
        }

        let result =
            greeners::RobustHausman::compare_arrays(&fe_beta, &re_beta, &fe_vcov, &re_vcov, None)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(diag(format!("{result}")))
    }

    pub(super) fn ftest_robust(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "ftest_robust(model [, vars=\"x1,x2\"])".into(),
            ));
        }

        let model = self.eval_expr(&args[0])?;

        // Extract beta and vcov from model
        let (beta, vcov, n, names) = match &model {
            Value::OlsResult(m) => {
                let p = m.result.params.len();
                let mut vcov = ndarray::Array2::<f64>::zeros((p, p));
                for i in 0..p {
                    vcov[(i, i)] = m.result.std_errors[i].powi(2);
                }
                (
                    m.result.params.clone(),
                    vcov,
                    m.x.nrows(),
                    m.result.variable_names.clone().unwrap_or_default(),
                )
            }
            Value::PanelResult(m) => {
                let p = m.params.len();
                let mut vcov = ndarray::Array2::<f64>::zeros((p, p));
                for i in 0..p {
                    vcov[(i, i)] = m.std_errors[i].powi(2);
                }
                (
                    m.params.clone(),
                    vcov,
                    m.n_obs,
                    m.variable_names.clone().unwrap_or_default(),
                )
            }
            Value::ReResult(m) => {
                let p = m.params.len();
                let mut vcov = ndarray::Array2::<f64>::zeros((p, p));
                for i in 0..p {
                    vcov[(i, i)] = m.std_errors[i].powi(2);
                }
                // RE result has no n_obs; use a large default for df_denom
                (
                    m.params.clone(),
                    vcov,
                    100,
                    m.variable_names.clone().unwrap_or_default(),
                )
            }
            _ => {
                return Err(HayashiError::Type(
                    "ftest_robust(): supports OLS, FE, RE models".into(),
                ))
            }
        };

        // Determine which coefficients to test
        let indices: Vec<usize> = if let Some(Value::Str(vars)) = opt_map.get("vars") {
            let var_list: Vec<String> = vars.split(',').map(|s| s.trim().to_string()).collect();
            var_list
                .iter()
                .filter_map(|v: &String| {
                    if v == "all" {
                        None // handled below
                    } else if let Ok(idx) = v.parse::<usize>() {
                        Some(idx)
                    } else {
                        names.iter().position(|n: &String| n == v)
                    }
                })
                .collect()
        } else {
            // Default: all slopes (exclude intercept)
            (1..beta.len()).collect::<Vec<usize>>()
        };

        let indices: Vec<usize> = if indices.is_empty() {
            (1..beta.len()).collect::<Vec<_>>()
        } else {
            indices
        };

        let names_ref: Vec<String> = if names.is_empty() {
            (0..beta.len()).map(|i| format!("x{}", i + 1)).collect()
        } else {
            names
        };

        let result = greeners::RobustFTest::test(&beta, &vcov, &indices, Some(&names_ref), n)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(diag(format!("{result}")))
    }

    pub(super) fn test(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "test(model, name) requires 2 arguments".into(),
            ));
        }
        let model = self.eval_expr(&args[0])?;

        let ols = match &model {
            Value::OlsResult(m) => m.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "test() currently supports OLS models only".into(),
                ))
            }
        };

        let test_name = match self.eval_expr(&args[1])? {
            Value::Str(s) => s,
            other => {
                return Err(HayashiError::Type(format!(
                    "test name must be a string (e.g. \"white\"), got {other}"
                )))
            }
        };

        match test_name.as_str() {
            // ── Specification tests ──────────────────────────────
            "white" => {
                let (stat, p, df) = SpecificationTests::white_test(&ols.residuals, &ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("white test: {e}")))?;
                let verdict = if p < 0.05 {
                    "Reject H0 — evidence of heteroskedasticity"
                } else {
                    "Fail to reject H0 — no evidence of heteroskedasticity"
                };
                let display = format!(
                    "White Test for Heteroskedasticity\n  LM statistic : {:.4}\n  p-value      : {:.4}\n  df           : {}\n  Conclusion   : {}\n",
                    stat, p, df, verdict
                );
                let summary = format!("White test LM={:.4}, p={:.4}, df={}", stat, p, df);
                let fit = model_expansion::fit_dict(&[
                    ("test", Value::Str("White".into())),
                    ("statistic", Value::Float(stat)),
                    ("p_value", Value::Float(p)),
                    ("df", Value::Int(df as i64)),
                    ("conclusion", Value::Str(verdict.into())),
                ]);
                let fields = vec![("fit".into(), fit)];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "WhiteTestResult",
                    fields,
                ))
            }
            "bp" => {
                let (stat, p) = Diagnostics::breusch_pagan(&ols.residuals, &ols.x)
                    .map_err(|e| HayashiError::Runtime(format!("Breusch-Pagan test: {e}")))?;
                let df = ols.x.ncols().saturating_sub(1) as i64;
                let verdict = if p < 0.05 {
                    "Reject H0 — evidence of heteroskedasticity"
                } else {
                    "Fail to reject H0 — no evidence of heteroskedasticity"
                };
                let display = format!(
                    "Breusch-Pagan Test for Heteroskedasticity\n  LM statistic : {:.4}\n  p-value      : {:.4}\n  Conclusion   : {}\n",
                    stat, p, verdict
                );
                let summary = format!("Breusch-Pagan test LM={:.4}, p={:.4}", stat, p);
                let fit = model_expansion::fit_dict(&[
                    ("test", Value::Str("Breusch-Pagan".into())),
                    ("statistic", Value::Float(stat)),
                    ("p_value", Value::Float(p)),
                    ("df", Value::Int(df)),
                    ("conclusion", Value::Str(verdict.into())),
                ]);
                let fields = vec![("fit".into(), fit)];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "BreuschPaganResult",
                    fields,
                ))
            }
            "dw" => {
                let stat = Diagnostics::durbin_watson(&ols.residuals);
                let verdict = if stat < 1.5 {
                    "Positive autocorrelation suspected"
                } else if stat > 2.5 {
                    "Negative autocorrelation suspected"
                } else {
                    "No strong evidence of autocorrelation"
                };
                let display = format!(
                    "Durbin-Watson Test for Autocorrelation\n  DW statistic : {:.4}\n  Conclusion   : {}\n",
                    stat, verdict
                );
                let summary = format!("Durbin-Watson DW={:.4}", stat);
                let fit = model_expansion::fit_dict(&[
                    ("test", Value::Str("Durbin-Watson".into())),
                    ("statistic", Value::Float(stat)),
                    ("conclusion", Value::Str(verdict.into())),
                ]);
                let fields = vec![("fit".into(), fit)];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "DurbinWatsonResult",
                    fields,
                ))
            }

            // ── Wald / F-test on coefficients ────────────────────
            other => {
                let names =
                    ols.result.variable_names.as_ref().ok_or_else(|| {
                        HayashiError::Runtime("model has no variable names".into())
                    })?;
                let k = ols.result.params.len();
                let find_idx = |name: &str| -> Result<usize> {
                    let n = name.trim();
                    names
                        .iter()
                        .position(|v| v == n)
                        .or_else(|| {
                            if n == "_cons" || n == "const" {
                                Some(k - 1)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| {
                            HayashiError::Runtime(format!("variable '{n}' not found in model"))
                        })
                };

                // "X1 = X2" or "X1 = 0.5"
                if let Some((lhs_s, rhs_s)) = other.split_once('=') {
                    let lhs_name = lhs_s.trim();
                    let rhs_trimmed = rhs_s.trim();
                    if let Ok(val) = rhs_trimmed.parse::<f64>() {
                        let idx = find_idx(lhs_name)?;
                        let mut r = ndarray::Array1::<f64>::zeros(k);
                        r[idx] = 1.0;
                        let (t, p) = ols
                            .result
                            .t_test(&r, val, &ols.x)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        let sig = Self::panel_sig_stars(p);
                        let display = format!(
                            "\n{:=^60}\n  H₀: {lhs_name} = {val}\n  t = {t:.4}   p = {p:.4}\n  {sig}\n",
                            " test "
                        );
                        let summary = format!("Wald t-test: t={:.4}, p={:.4}", t, p);
                        let fit = model_expansion::fit_dict(&[
                            ("test", Value::Str("t-test".into())),
                            ("hypothesis", Value::Str(format!("{lhs_name} = {val}"))),
                            ("coefficient", Value::Str(lhs_name.to_string())),
                            ("restriction_value", Value::Float(val)),
                            ("t_stat", Value::Float(t)),
                            ("p_value", Value::Float(p)),
                            ("sig", Value::Str(sig.into())),
                        ]);
                        let fields = vec![("fit".into(), fit)];
                        Ok(model_expansion::model_result(
                            display,
                            summary,
                            "WaldTestResult",
                            fields,
                        ))
                    } else {
                        let idx1 = find_idx(lhs_name)?;
                        let idx2 = find_idx(rhs_trimmed)?;
                        let mut r = ndarray::Array1::<f64>::zeros(k);
                        r[idx1] = 1.0;
                        r[idx2] = -1.0;
                        let (t, p) = ols
                            .result
                            .t_test(&r, 0.0, &ols.x)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        let sig = Self::panel_sig_stars(p);
                        let display = format!(
                            "\n{:=^60}\n  H₀: {lhs_name} = {rhs_trimmed}\n  t = {t:.4}   p = {p:.4}\n  {sig}\n",
                            " test "
                        );
                        let summary = format!("Wald t-test: t={:.4}, p={:.4}", t, p);
                        let fit = model_expansion::fit_dict(&[
                            ("test", Value::Str("t-test".into())),
                            (
                                "hypothesis",
                                Value::Str(format!("{lhs_name} = {rhs_trimmed}")),
                            ),
                            ("coefficient1", Value::Str(lhs_name.to_string())),
                            ("coefficient2", Value::Str(rhs_trimmed.to_string())),
                            ("t_stat", Value::Float(t)),
                            ("p_value", Value::Float(p)),
                            ("sig", Value::Str(sig.into())),
                        ]);
                        let fields = vec![("fit".into(), fit)];
                        Ok(model_expansion::model_result(
                            display,
                            summary,
                            "WaldTestResult",
                            fields,
                        ))
                    }
                } else {
                    let mut extra_names: Vec<String> = Vec::new();
                    for arg in &args[2..] {
                        let name = match self.eval_expr(arg)? {
                            Value::Str(s) => s,
                            other => {
                                return Err(HayashiError::Type(format!(
                                    "test() variable names must be strings, got {other}"
                                )))
                            }
                        };
                        extra_names.push(name);
                    }
                    let mut indices = vec![find_idx(other)?];
                    for name in &extra_names {
                        indices.push(find_idx(name)?);
                    }
                    let (f, p) = ols
                        .result
                        .f_test(&indices, &ols.x)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let var_list: Vec<&str> = indices.iter().map(|&i| names[i].as_str()).collect();
                    let q = indices.len();
                    let hypothesis = if q == 1 {
                        format!("{} = 0", var_list[0])
                    } else {
                        format!("{} = 0", var_list.join(" = "))
                    };
                    let sig = Self::panel_sig_stars(p);
                    let display = format!(
                        "\n{:=^60}\n  H₀: {hypothesis}\n  F({q}, {}) = {f:.4}   p = {p:.4}\n  {sig}\n",
                        " test ",
                        ols.result.df_resid
                    );
                    let summary =
                        format!("Wald F({q}, {})={:.4}, p={:.4}", ols.result.df_resid, f, p);
                    let variables: Vec<Value> = var_list
                        .iter()
                        .map(|&s| Value::Str(s.to_string()))
                        .collect();
                    let fit = model_expansion::fit_dict(&[
                        ("test", Value::Str("F-test".into())),
                        ("hypothesis", Value::Str(hypothesis)),
                        ("f_stat", Value::Float(f)),
                        ("p_value", Value::Float(p)),
                        ("df_num", Value::Int(q as i64)),
                        ("df_denom", Value::Int(ols.result.df_resid as i64)),
                        ("variables", Value::List(Arc::new(variables))),
                        ("sig", Value::Str(sig.into())),
                    ]);
                    let fields = vec![("fit".into(), fit)];
                    Ok(model_expansion::model_result(
                        display,
                        summary,
                        "WaldTestResult",
                        fields,
                    ))
                }
            }
        }
    }
}
