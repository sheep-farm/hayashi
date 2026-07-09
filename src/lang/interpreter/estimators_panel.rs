use super::helpers::*;
use super::*;

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
            // bootse — Bootstrap standard errors for OLS models
            // bootse(model, n=1000)
            // Resample pairs (y, X) with replacement to estimate sampling distribution
            // Compare original SE with bootstrap SE and 95% percentile CI
            // ── generic bootstrap ────────────────────────────────────────────
            // bootstrap(estimator, formula, df, n=1000, alpha=0.05)
            // Resample DataFrame rows with replacement and re-estimate.
            // Works with any estimator: ols, logit, probit, iv, poisson, etc.
            // bootse(model, n=1000) kept as alias for OLS pairs bootstrap.
            "bootstrap" | "boot" => {
                let n_boot = Self::bootstrap_reps(opt_map);
                let alpha = Self::bootstrap_alpha(opt_map);
                if args.len() >= 3 {
                    self.bootstrap_generic(args, opts, n_boot, alpha)
                } else {
                    self.bootstrap_pairs(args, n_boot, alpha)
                }
            }

            "bootse" => {
                return self.eval_call("bootstrap", args, opts).map(Some);
            }

            // markov — Markov-Switching AR (Hamilton 1989)
            // markov(df, y, k=2, p=1)
            // k=: number of regimes (default: 2)
            // p=: AR order within each regime (default: 1)
            // Algorithm: EM via Hamilton filter (forward-backward)
            // Parameters per regime: intercept + AR coefficients + variance
            "markov" | "msar" | "markovswitching" => {
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

            // clogit — Conditional Logit (Chamberlain 1980, FE logit)
            // clogit(y ~ x1 + x2, df, group="id_col")
            // Conditions on the sum of y by group → eliminates individual fixed effects
            // Groups with no variation in y are automatically dropped
            // No intercept — absorbed by FE
            "clogit" | "xtlogit_fe" => {
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
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
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
                let result = greeners::ConditionalLogit::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    Some(var_names),
                )
                .map_err(|e| self.rt_err(format!("clogit: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // cpoisson — Conditional Poisson (FE Poisson)
            // cpoisson(y ~ x1 + x2, df, group="id_col")
            // Equivalent to FE Poisson; consistent under unobserved heterogeneity
            // Only requires E[y|x,c] = exp(c + xβ) — does not require y ~ Poisson (PPML)
            "cpoisson" | "xtpoisson_fe" | "ppml" => {
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
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
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
                let result = greeners::ConditionalPoisson::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    Some(var_names),
                )
                .map_err(|e| self.rt_err(format!("cpoisson: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // cmnlogit — Conditional Multinomial Logit
            // cmnlogit(y ~ x1 + x2, df, group="id_col", alts=3)
            "cmnlogit" | "cmlogit" | "conditional_mlogit" => {
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
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
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

            // gqtest — Goldfeld-Quandt test (heteroskedasticity)
            // gqtest(model, split=0.2)
            // H0: homoskedasticity
            // Splits residuals into two groups (discarding `split` from the middle)
            // and tests if variances differ via F
            // split=: fraction of the middle to discard (default: 0.2)
            // More powerful than White when heteroskedasticity is monotonic
            "gqtest" => {
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
                let sep = "─".repeat(56);
                println!("\nGoldfeld-Quandt Test  —  split = {split:.2}");
                println!("{sep}");
                println!("H₀: homoskedasticity (σ²₁ = σ²₂)");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Test", "Statistic", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10.4} {:>10.4} {:>4}",
                    format!("F ~ F({df1},{df2})"),
                    f,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // bphet — Breusch-Pagan test (heteroskedasticity, OLS)
            // bphet(model)
            // H0: homoskedasticity — LM = n·R² from auxiliary regression of u² on X
            // Different from bptest() which is the random effects LM (panel)
            "bphet" | "hettest" => {
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
                let sep = "─".repeat(56);
                println!("\nBreusch-Pagan Heteroskedasticity Test");
                println!("{sep}");
                println!("H₀: homoskedasticity (constant variance)");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Test", "Statistic", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10.4} {:>10.4} {:>4}",
                    format!("LM ~ χ²({k})"),
                    lm,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // ── Diagnostic tests for panel data ─────────────────────────────

            // bptest — Breusch-Pagan LM test (H0: pooled OLS adequate, σ²_u = 0)
            // bptest(df, y ~ x1 + x2, id="entity_col")
            "bptest" | "xttest0" | "xtbp" => {
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
                            self.rt_err(format!(
                                "bptest requires id= or xtset({df_name}, id, time)"
                            ))
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
                let mut id_map: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
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
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!("\n{:=^62}", " Breusch-Pagan LM Test (RE) ");
                println!(" H0: σ²_u = 0 — pooled OLS adequate");
                println!("{:-^62}", "");
                println!(" LM = {lm:.4}    p-value = {p:.4}  {sig}");
                if p < 0.05 {
                    println!(" Conclusion: reject H0 → use RE or FE");
                } else {
                    println!(" Conclusion: do not reject H0 → pooled OLS adequate");
                }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // wooldridge — Wooldridge test for panel serial correlation
            // H0: no first-order serial correlation in idiosyncratic errors
            "wooldridge" | "xtserial" | "wooldridge_serial" | "xtwooldridge" => {
                self.eval_wooldridge(args, opt_map)
            }

            // pesaran — Pesaran CD test (cross-sectional dependence)
            "pesaran" | "xtcd" => self.eval_pesaran(args, opt_map),

            // mundlak — Mundlak test (RE vs FE adequacy)
            "mundlak" => self.eval_mundlak(args, opt_map),

            // abtest — Arellano-Bond m1/m2 test (GMM instrument validation)
            "abtest" | "abar" | "abond" | "xtabond_test" | "arellano_bond" => {
                self.eval_abtest(args, opt_map)
            }

            // ── SUR (Seemingly Unrelated Regressions) ─────────────────────────
            // sur(df, y1 ~ x1 + x2, y2 ~ x3 + x4, ...)
            // Zellner estimator (FGLS across equations)
            // Each equation may have different regressors
            "sur" | "sureg" => {
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
                let result = greeners::SUR::fit(&equations)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::SurResult(SurModel {
                    result: Rc::new(result),
                    eq_var_names,
                }))
            }

            // ── Rolling OLS (rolling window) ────────────────────────────────
            "rolling" | "rols" => self.eval_rolling(args, opts, opt_map),

            // ── Recursive OLS (Kalman, accumulates observations) ───────────────
            "recursive" | "recols" => self.eval_recursive(args, opts),

            // ── ic — information criteria table (AIC/BIC) ─────────────────────
            // ic(m1, m2, m3, ...)
            // Compares models by AIC and BIC; sorts from smallest (best) to largest
            "ic" | "fitstat" | "estat" => {
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
                        Value::OlsResult(m)      => (m.result.log_likelihood, m.result.params.len(), m.result.n_obs),
                        Value::BinaryResult(b)   => (b.result.log_likelihood, b.result.params.len(), b.x.nrows()),
                        Value::PoissonResult(r)  => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::NegBinResult(r)   => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::OrderedResult(r)  => (r.log_likelihood, r.params.len() + r.thresholds.len(), r.n_obs),
                        Value::TobitResult(r)    => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::MixedResult(r)    => (r.log_likelihood, r.fixed_effects.len(), r.n_obs),
                        Value::ZeroInflatedResult(r) => (r.log_likelihood, r.count_params.len() + r.inflate_params.len(), r.n_obs),
                        Value::RollingResult(_) | Value::RecursiveLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                format!("ic(): '{label}' has no log-likelihood — use print() for diagnostics")
                            ));
                        }
                        _ => return Err(HayashiError::Runtime(
                            format!("ic(): model '{label}' has no log-likelihood available for ic() — use print()")
                        )),
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
                let _min_bic = rows
                    .iter()
                    .map(|r| r.bic)
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                println!("\n{:=^80}", " Information Criteria ");
                println!(
                    "{:<20} {:>6} {:>6} {:>12} {:>12} {:>8} {:>8}",
                    "Model", "N", "k", "Log-Lik", "AIC", "ΔAIC", "BIC"
                );
                println!("{:-^80}", "");
                for row in &rows {
                    println!(
                        "{:<20} {:>6} {:>6} {:>12.4} {:>12.4} {:>8.4} {:>12.4}",
                        row.label,
                        row.n,
                        row.k,
                        row.ll,
                        row.aic,
                        row.aic - min_aic,
                        row.bic
                    );
                }
                if rows.len() > 1 {
                    println!("{:-^80}", "");
                    println!(
                        " Best AIC: {}   Best BIC: {}",
                        rows.iter()
                            .min_by(|a, b| a.aic.total_cmp(&b.aic))
                            .map(|r| r.label.as_str())
                            .unwrap_or("—"),
                        rows.iter()
                            .min_by(|a, b| a.bic.total_cmp(&b.bic))
                            .map(|r| r.label.as_str())
                            .unwrap_or("—")
                    );
                    // Akaike weights
                    let delta_aics: Vec<f64> = rows.iter().map(|r| r.aic - min_aic).collect();
                    let rel: Vec<f64> = delta_aics.iter().map(|d| (-d / 2.0).exp()).collect();
                    let sum_rel: f64 = rel.iter().sum();
                    println!(
                        " Akaike weights: {}",
                        rows.iter()
                            .zip(rel.iter())
                            .map(|(r, w)| format!("{}={:.3}", r.label, w / sum_rel))
                            .collect::<Vec<_>>()
                            .join("  ")
                    );
                }
                println!("{:=^80}", "");
                Ok(Value::Nil)
            }

            // ── Fixed Effects ─────────────────────────────────────────────────
            "fe" => {
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

            // ── Random Effects ────────────────────────────────────────────────
            "re" => {
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                // accepts float column of integer values (e.g. idcode read as f64)
                let ids_owned: ndarray::Array1<i64>;
                let ids = match df.get_int(&id_col) {
                    Ok(arr) => arr,
                    Err(_) => {
                        let floats = df.get(id_col.as_str()).map_err(|_| {
                            HayashiError::Runtime(format!(
                                "column '{id_col}' must be integer for re()"
                            ))
                        })?;
                        ids_owned = floats.mapv(|v| v as i64);
                        &ids_owned
                    }
                };

                let result = RandomEffects::from_formula(&g_formula, &df, ids)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::ReResult(Rc::new(result)))
            }

            // ── F-test for Fixed Effects (FE vs pooled OLS) ──────────────────
            "ftest_fe" => {
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

                let (f_stat, p) = greeners::PanelDiagnostics::f_test_fixed_effects(
                    ssr_pooled, ssr_fe, n, n_entities, k,
                )
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
                Ok(diag(out))
            }

            // ── Pesaran CD: cross-sectional dependence ────────────────────────
            "pesaran_cd" | "cd_test" => {
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
                Ok(diag(out))
            }

            // ── Breusch-Pagan LM test (individual effects in panel) ─────────
            "bplm" => {
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
                    "Reject H₀ → individual effects present (use FE or RE)"
                } else {
                    "Do not reject H₀ → pooled OLS adequate (no individual effects)"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Breusch-Pagan LM Test (individual effects)\n");
                out.push_str(" H₀: σ²_u = 0  (no individual effects)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Panel Data\n");
                out.push_str(&format!(
                    "   n = {}   N = {}   T̄ ≈ {:.1}\n",
                    n, n_entities, t_bar
                ));
                out.push_str("\n── Statistic\n");
                out.push_str(&format!(
                    "   LM ~ χ²(1) = {:.4}   p = {:.4}  {}\n",
                    lm, p, sig
                ));
                out.push_str("\n── Conclusion\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Chamberlain: period-specific correlation with individual effects
            "chamberlain" => {
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
                    greeners::PanelDiagnostics::chamberlain(
                        &y_vec,
                        &x_mat,
                        &entity_ids,
                        &time_vals,
                    )
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
                out.push_str(
                    " Chamberlain Test (period-specific correlation with individual effects)\n",
                );
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
                out.push_str(
                    "   More general test than Mundlak — includes values in all T periods\n",
                );
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Arellano-Bond Diff-GMM (OLD mundlak removed — use new mundlak above) ─
            "mundlak_OLD_REMOVED" => {
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
                out.push_str(
                    " Mundlak Test (correlation between regressors and individual effects)\n",
                );
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
                Ok(diag(out))
            }

            // ── Arellano-Bond Diff-GMM ────────────────────────────────────────
            // ab(formula, df, id=col, time=col [, lags=2 [, step=1]])
            // Estimates y_it = ρ y_{i,t-1} + X_it'β + α_i + ε_it via Diff-GMM.
            // Instruments Δy_{i,t-1} with y_{i,t-2},...,y_{i,t-lags-1} (collapsed).
            "ab" => {
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

            // ── Generic GMM (Two-Step Efficient) ────────────────────────────
            // gmm(endog_formula, instrument_formula, df)
            "gmm" => {
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

            // ── System GMM (Blundell-Bond 1998) ──────────────────────────────
            // sysgmm(formula, df, id=col, time=col [, lags=2 [, step=1]])
            // Stacks equations in 1st differences (instrumented with lagged levels)
            // + equations in levels (instrumented with Δy_{t-1} and ΔX_{t-1}).
            "sysgmm" => {
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

            // ── FE-2SLS (xtivreg, fe) — Hausman (1978) ───────────────────────
            // feiv(endog_formula, instrument_formula, df, id=col [, cov=...])
            // endog_formula: y ~ x1 + x2   (x2 is endogenous)
            // instrument_formula: ~ x1 + z1 + z2  (included exogenous + excluded)
            "feiv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "feiv() requires (structural_formula, instrument_formula, df, id=col)"
                            .into(),
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

                let result =
                    greeners::FE2SLS::fit(&y_vec, &x_mat, &z_mat, &entity_ids, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::FE2SLSResult(Rc::new(result)))
            }

            // ── PCSE — Panel-Corrected Standard Errors (Beck & Katz 1995) ─────
            // pcse(formula, df, id=col, time=col)
            "pcse" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let time_col = self.get_time_col(&df_name, opt_map)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let entity_ids = Self::col_as_i64(&df, &id_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let time_ids = Self::col_as_i64(&df, &time_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result =
                    greeners::PCSE::fit(&y_vec, &x_mat, &entity_ids, &time_ids, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PcseResult(Rc::new(result)))
            }

            // ── Panel GLS — Parks (1967) / Stata xtgls ───────────────────────
            // xtgls(formula, df, id=col, time=col [, panels="hetero"|"corr"])
            "xtgls" => {
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
                let entity_ids = Self::col_as_i64(&df, &id_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let time_ids = Self::col_as_i64(&df, &time_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
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

            // ── Arellano-Bond: m1/m2 test for serial autocorrelation ─────────
            "ab_test" => {
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

                let (m1, p1, m2, p2) = greeners::PanelDiagnostics::arellano_bond_test(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_vals,
                )
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
                    out.push_str(
                        "   m1 does not reject H₀ → check specification (AR(1) expected in FD)\n",
                    );
                } else {
                    out.push_str("   m2 rejects H₀ → AR(2) in residuals; y_{t-2} instruments may be invalid\n");
                    out.push_str(
                        "   Consider using more distant lags (y_{t-3}, ...) as instruments\n",
                    );
                }
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(
                    "   Variance estimated via sandwich (Σ_i of cross-products by entity)\n",
                );
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── wooldridge_OLD_REMOVED (replaced by the new one above) ─────────
            "wooldridge_OLD_REMOVED" => {
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

                let (rho, t_stat, p, n_pairs) = greeners::PanelDiagnostics::wooldridge_serial(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_vals,
                )
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
                Ok(diag(out))
            }

            // ── Hausman FE vs RE ──────────────────────────────────────────────
            "hausman" => {
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
                let fe_names: Vec<String> =
                    fe.variable_names.as_ref().cloned().unwrap_or_else(|| {
                        (0..fe.params.len()).map(|i| format!("x{}", i)).collect()
                    });

                let re_names: Vec<String> =
                    re.variable_names.as_ref().cloned().unwrap_or_else(|| {
                        (0..re.params.len()).map(|i| format!("x{}", i)).collect()
                    });

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
                        "hausman: no common variable between FE and RE (check variable_names)"
                            .into(),
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
                out.push_str(
                    " H₀: individual effects uncorrelated with regressors (RE consistent)\n",
                );
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
                    return Ok(Some(diag(out)));
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
                Ok(diag(out))
            }

            // ── Diagnostics ──────────────────────────────────────────────────
            "test" => {
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
                    "white" => match SpecificationTests::white_test(&ols.residuals, &ols.x) {
                        Ok((stat, p, df)) => {
                            println!("White Test for Heteroskedasticity");
                            println!("  LM statistic : {:.4}", stat);
                            println!("  p-value      : {:.4}", p);
                            println!("  df           : {}", df);
                            let verdict = if p < 0.05 {
                                "Reject H0 — evidence of heteroskedasticity"
                            } else {
                                "Fail to reject H0 — no evidence of heteroskedasticity"
                            };
                            println!("  Conclusion   : {}", verdict);
                        }
                        Err(e) => eprintln!("White test error: {e}"),
                    },
                    "bp" => match Diagnostics::breusch_pagan(&ols.residuals, &ols.x) {
                        Ok((stat, p)) => {
                            println!("Breusch-Pagan Test for Heteroskedasticity");
                            println!("  LM statistic : {:.4}", stat);
                            println!("  p-value      : {:.4}", p);
                            let verdict = if p < 0.05 {
                                "Reject H0 — evidence of heteroskedasticity"
                            } else {
                                "Fail to reject H0 — no evidence of heteroskedasticity"
                            };
                            println!("  Conclusion   : {}", verdict);
                        }
                        Err(e) => eprintln!("Breusch-Pagan test error: {e}"),
                    },
                    "dw" => {
                        let stat = Diagnostics::durbin_watson(&ols.residuals);
                        println!("Durbin-Watson Test for Autocorrelation");
                        println!("  DW statistic : {:.4}", stat);
                        let verdict = if stat < 1.5 {
                            "Positive autocorrelation suspected"
                        } else if stat > 2.5 {
                            "Negative autocorrelation suspected"
                        } else {
                            "No strong evidence of autocorrelation"
                        };
                        println!("  Conclusion   : {}", verdict);
                    }

                    // ── Wald / F-test on coefficients ────────────────────
                    other => {
                        let names = ols.result.variable_names.as_ref().ok_or_else(|| {
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
                                    HayashiError::Runtime(format!(
                                        "variable '{n}' not found in model"
                                    ))
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
                                println!("\n{:=^60}", " test ");
                                println!("  H₀: {lhs_name} = {val}");
                                println!("  t = {t:.4}   p = {p:.4}");
                                let sig = if p < 0.01 {
                                    "***"
                                } else if p < 0.05 {
                                    "**"
                                } else if p < 0.10 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!("  {sig}");
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
                                println!("\n{:=^60}", " test ");
                                println!("  H₀: {lhs_name} = {rhs_trimmed}");
                                println!("  t = {t:.4}   p = {p:.4}");
                                let sig = if p < 0.01 {
                                    "***"
                                } else if p < 0.05 {
                                    "**"
                                } else if p < 0.10 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!("  {sig}");
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
                            let var_list: Vec<&str> =
                                indices.iter().map(|&i| names[i].as_str()).collect();
                            let q = indices.len();
                            println!("\n{:=^60}", " test ");
                            if q == 1 {
                                println!("  H₀: {} = 0", var_list[0]);
                            } else {
                                println!("  H₀: {} = 0", var_list.join(" = "));
                            }
                            println!("  F({q}, {}) = {f:.4}   p = {p:.4}", ols.result.df_resid);
                            let sig = if p < 0.01 {
                                "***"
                            } else if p < 0.05 {
                                "**"
                            } else if p < 0.10 {
                                "*"
                            } else {
                                ""
                            };
                            println!("  {sig}");
                        }
                    }
                }

                Ok(Value::Nil)
            }

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
                .set("__boot_df__", Value::DataFrame(Rc::new(boot_df)))?;
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
        println!("\n{thick}");
        println!(
            "{:^76}",
            format!(" Bootstrap SE — {} (n={n_ok}/{n_boot}) ", estimator_name)
        );
        println!("{thin}");
        println!(
            "{:<18} {:>10} {:>10} {:>10} {:>12} {:>12}",
            "Variable", "β̂", "Orig. SE", "Boot SE", "CI lower", "CI upper"
        );
        println!("{thin}");
        for i in 0..k {
            let vname = var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
            let orig_se = if i < full_se.len() {
                full_se[i]
            } else {
                f64::NAN
            };
            println!(
                "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>12.4}",
                vname, full_params[i], orig_se, boot_se[i], ci_lo[i], ci_hi[i]
            );
        }
        println!("{thick}");
        Ok(Value::Nil)
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
                let vnames = m.result.variable_names.as_deref().unwrap_or(&[]);
                let k = m.result.params.len();
                let thick = "═".repeat(76);
                let thin = "─".repeat(76);
                println!("\n{thick}");
                println!("{:^76}", format!(" Bootstrap SE (n={n_boot}, pairs) "));
                println!("{thin}");
                println!(
                    "{:<18} {:>10} {:>10} {:>10} {:>12} {:>12}",
                    "Variable", "β̂", "Orig. SE", "Boot SE", "CI lower 95%", "CI upper 95%"
                );
                println!("{thin}");
                for i in 0..k {
                    let vname = vnames.get(i).map(|s| s.as_str()).unwrap_or("?");
                    println!(
                        "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>12.4}",
                        vname,
                        m.result.params[i],
                        m.result.std_errors[i],
                        boot_se[i],
                        ci_lo[i],
                        ci_hi[i]
                    );
                }
                println!("{thick}");
                Ok(Value::Nil)
            }
            _ => Err(HayashiError::Runtime(
                "bootse(model) supports OLS. For others: bootstrap(estimator, formula, df, n=1000)"
                    .into(),
            )),
        }
    }
}
