use super::*;
use super::helpers::*;
use super::models::FactorModel;

/// Finance (Fama-MacBeth, portsort, doublesort) and
/// cross-section/microeconometric estimators: OLS/reg, IV/2SLS, weak instrument
/// test, Logit, Probit, Heckman, Tobit, RD sharp/fuzzy, PSM, Synthetic
/// Control, Poisson, NegBin, Ordered Logit/Probit, MNLogit, DID, Quantile
/// Regression, Kaplan-Meier, Cox, RLM, GEE, WLS, ZIP/ZINB, MixedLM, testparm,
/// GLSAR, ANOVA, Beta Regression.
/// Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_estimators_micro(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            "reg" | "regress" => self.eval_call("ols", args, opts),

            // ── Fama-MacBeth (1973) ──────────────────────────────────────────
            // fmb(formula, df, time=col)
            // Cross-sectional regressions by period, average of coefficients
            // SE = σ(β̂_t) / √T  (Fama-MacBeth standard errors)
            "fmb" | "fama_macbeth" | "xtfmb" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("fmb(formula, df, time=col)".into()));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let time_col = match opt_map.get("time") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(_, t)| t.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            HayashiError::Runtime(
                                "fmb requires time=col or xtset(df, id, time)".into(),
                            )
                        })?,
                };
                let nw_lags: usize = match opt_map.get("nw") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    Some(Value::Str(s)) => s.parse().unwrap_or(0),
                    _ => 0,
                };

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::FamaMacBeth::fit(&g_formula, &df, &time_col, nw_lags)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("{result}");
                Ok(Value::Nil)
            }

            // ── portsort: portfolio sorts by quantiles ────────────────────────
            // portsort(df, ret, sort_var, n=5)
            // Sorts observations by sort_var, divides into n portfolios,
            // reports mean, SE and t of ret per portfolio + H-L spread.
            "portsort" | "portfolio_sort" | "psort" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "portsort(df, ret_var, sort_var, n=5)".into(),
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
                let df_raw = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df_raw, opts)?;
                let ret_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be return variable".into(),
                        ))
                    }
                };
                let sort_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "third argument must be sort variable".into(),
                        ))
                    }
                };
                let n_ports: usize = match opt_map.get("n") {
                    Some(Value::Int(v)) => (*v).max(2) as usize,
                    Some(Value::Float(v)) => (*v as usize).max(2),
                    _ => 5,
                };

                let ret_col = get_col_f64(&df, &ret_name)?;
                let sort_col = get_col_f64(&df, &sort_name)?;

                // pares (sort_val, ret_val) — excluir NaN
                let mut pairs: Vec<(f64, f64)> = sort_col
                    .iter()
                    .zip(ret_col.iter())
                    .filter(|(s, r)| s.is_finite() && r.is_finite())
                    .map(|(&s, &r)| (s, r))
                    .collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let n_valid = pairs.len();
                let per_port = n_valid / n_ports;

                if per_port < 1 {
                    return Err(HayashiError::Runtime(format!(
                        "portsort: {n_valid} valid obs insufficient for {n_ports} portfolios"
                    )));
                }

                // assign portfolios
                struct PortStats {
                    mean: f64,
                    se: f64,
                    n: usize,
                }
                let mut ports: Vec<PortStats> = Vec::new();
                for p in 0..n_ports {
                    let start = p * per_port;
                    let end = if p == n_ports - 1 {
                        n_valid
                    } else {
                        (p + 1) * per_port
                    };
                    let rets: Vec<f64> = pairs[start..end].iter().map(|(_, r)| *r).collect();
                    let n = rets.len();
                    let mean = rets.iter().sum::<f64>() / n as f64;
                    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0).max(1.0);
                    let se = (var / n as f64).sqrt();
                    ports.push(PortStats { mean, se, n });
                }

                // spread H-L
                let hl_mean = ports.last().unwrap().mean - ports[0].mean;
                let hl_se = (ports.last().unwrap().se.powi(2) + ports[0].se.powi(2)).sqrt();
                let hl_t = if hl_se > 1e-15 {
                    hl_mean / hl_se
                } else {
                    f64::NAN
                };
                let hl_p = t_pvalue_two(hl_t, (ports.last().unwrap().n + ports[0].n - 2) as f64);

                let thick = "═".repeat(60);
                let thin = "─".repeat(60);
                println!("\n{thick}");
                println!(
                    "{:^60}",
                    format!(" Portfolio Sort: {ret_name} by {sort_name} ({n_ports} groups) ")
                );
                println!("{thin}");
                println!(
                    "{:<12} {:>8} {:>12} {:>10} {:>10}",
                    "Portfolio", "N", "Mean", "SE", "t"
                );
                println!("{thin}");
                for (i, ps) in ports.iter().enumerate() {
                    let t = if ps.se > 1e-15 {
                        ps.mean / ps.se
                    } else {
                        f64::NAN
                    };
                    let label = if i == 0 {
                        "Low".to_string()
                    } else if i == n_ports - 1 {
                        "High".to_string()
                    } else {
                        format!("P{}", i + 1)
                    };
                    println!(
                        "{:<12} {:>8} {:>12.4} {:>10.4} {:>10.4}",
                        label, ps.n, ps.mean, ps.se, t
                    );
                }
                println!("{thin}");
                let sig = if hl_p < 0.01 {
                    "***"
                } else if hl_p < 0.05 {
                    "**"
                } else if hl_p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!(
                    "{:<12} {:>8} {:>12.4} {:>10.4} {:>10.4} {sig}",
                    "H-L", "", hl_mean, hl_se, hl_t
                );
                println!("{thick}\n");
                Ok(Value::Nil)
            }

            // ── doublesort: portfolio sort bidimensional (Fama-French) ─────
            // doublesort(df, ret, sort1, sort2, n1=5, n2=5)
            "doublesort" | "double_sort" | "bivariate_sort" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "doublesort(df, ret, sort1, sort2, n1=5, n2=5)".into(),
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
                let df_raw = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df_raw, opts)?;
                let ret_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("ret var".into())),
                };
                let s1_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("sort1 var".into())),
                };
                let s2_name = match &args[3] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("sort2 var".into())),
                };
                let n1: usize = match opt_map.get("n1") {
                    Some(Value::Int(v)) => (*v).max(2) as usize,
                    _ => 5,
                };
                let n2: usize = match opt_map.get("n2") {
                    Some(Value::Int(v)) => (*v).max(2) as usize,
                    _ => 5,
                };

                let ret_col = get_col_f64(&df, &ret_name)?;
                let s1_col = get_col_f64(&df, &s1_name)?;
                let s2_col = get_col_f64(&df, &s2_name)?;

                // atribuir quantis independentes
                let assign_quantile = |vals: &[f64], n_q: usize| -> Vec<usize> {
                    let mut indexed: Vec<(usize, f64)> = vals
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.is_finite())
                        .map(|(i, &v)| (i, v))
                        .collect();
                    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let n = indexed.len();
                    let mut q = vec![usize::MAX; vals.len()];
                    for (rank, &(orig_i, _)) in indexed.iter().enumerate() {
                        q[orig_i] = (rank * n_q / n).min(n_q - 1);
                    }
                    q
                };

                let s1_vec: Vec<f64> = s1_col.to_vec();
                let s2_vec: Vec<f64> = s2_col.to_vec();
                let q1 = assign_quantile(&s1_vec, n1);
                let q2 = assign_quantile(&s2_vec, n2);

                // means per cell (q1 x q2)
                let mut cell_sum = vec![vec![0.0; n2]; n1];
                let mut cell_n = vec![vec![0usize; n2]; n1];
                for i in 0..ret_col.len() {
                    if q1[i] < n1 && q2[i] < n2 && ret_col[i].is_finite() {
                        cell_sum[q1[i]][q2[i]] += ret_col[i];
                        cell_n[q1[i]][q2[i]] += 1;
                    }
                }

                let thick = "═".repeat(12 + n2 * 10);
                let thin = "─".repeat(12 + n2 * 10);
                println!("\n{thick}");
                println!(" Double Sort: {ret_name} by {s1_name} (rows) × {s2_name} (cols)");
                println!("{thin}");
                print!("{:<12}", format!("{s1_name}\\{s2_name}"));
                for j in 0..n2 {
                    let label = if j == 0 {
                        "Low"
                    } else if j == n2 - 1 {
                        "High"
                    } else {
                        &format!("Q{}", j + 1)
                    };
                    print!("{:>10}", label);
                }
                println!();
                println!("{thin}");
                for i in 0..n1 {
                    let label = if i == 0 {
                        "Low".to_string()
                    } else if i == n1 - 1 {
                        "High".to_string()
                    } else {
                        format!("Q{}", i + 1)
                    };
                    print!("{:<12}", label);
                    for j in 0..n2 {
                        let mean = if cell_n[i][j] > 0 {
                            cell_sum[i][j] / cell_n[i][j] as f64
                        } else {
                            f64::NAN
                        };
                        if mean.is_nan() {
                            print!("{:>10}", ".");
                        } else {
                            print!("{:>10.4}", mean);
                        }
                    }
                    println!();
                }
                println!("{thick}\n");
                Ok(Value::Nil)
            }

            // ── OLS ───────────────────────────────────────────────────────────
            "ols" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ols() requires (formula, dataframe)".into(),
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
                let formula_str = Self::formula_to_string(&formula_ast);
                let cov = resolve_cov_full(opt_map, &df)?;

                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = OLS::from_formula(&g_formula, &df, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let fitted = result.x_clean.as_ref().unwrap_or(&x).dot(&result.params);
                let residuals = &y - &fitted;
                let x_used = result.x_clean.clone().unwrap_or(x);

                Ok(Value::OlsResult(OlsModel {
                    result: Rc::new(result),
                    residuals,
                    x: x_used,
                }))
            }

            // ── IV / 2SLS ─────────────────────────────────────────────────────
            "iv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "iv() requires (endog_formula, instrument_formula, dataframe)".into(),
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
                let cov = resolve_cov_full(opt_map, &df)?;

                let endog_str = Self::formula_to_string(&endog_ast);
                let instr_str = Self::formula_to_string(&instr_ast);

                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // The instrument formula may have empty LHS (syntax ~ z1 + z2).
                // GFormula::parse rejects empty LHS; we build it directly.
                let g_instr = if instr_ast.lhs.is_empty() {
                    let independents: Vec<String> = instr_ast
                        .rhs
                        .iter()
                        .map(|t| match t {
                            RhsTerm::Var(v) => v.clone(),
                            RhsTerm::Categorical(v) => format!("C({v})"),
                            RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                            RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
                        })
                        .collect();
                    GFormula {
                        dependent: String::new(),
                        independents,
                        intercept: true,
                    }
                } else {
                    GFormula::parse(&instr_str).map_err(|e| HayashiError::Runtime(e.to_string()))?
                };

                let result = IV::from_formula(&g_endog, &g_instr, &df, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::IvResult(Rc::new(result)))
            }

            // ── Teste de instrumentos fracos (Cragg-Donald / Stock-Yogo) ──────
            // weak_iv(endog_formula, instrument_formula, df)
            // Same syntax as iv(). Computes 1st stage F (per endog) and
            // Cragg-Donald statistic. Compares with Stock & Yogo (2005)
            // critical values.
            "weak_iv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "weak_iv() requires (structural_formula, instrument_formula, df)".into(),
                    ));
                }
                let endog_ast = self.resolve_formula(&args[0])?;
                let instr_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "weak_iv(): third argument must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => {
                        return Err(self.rt_err(format!("weak_iv: '{df_name}' is not a DataFrame")))
                    }
                };

                // ── Identify variables ──
                let endog_vars: std::collections::HashSet<String> = endog_ast
                    .rhs
                    .iter()
                    .map(|t| match t {
                        RhsTerm::Var(v) => v.clone(),
                        _ => String::new(),
                    })
                    .filter(|s| !s.is_empty())
                    .collect();
                let instr_vars: std::collections::HashSet<String> = instr_ast
                    .rhs
                    .iter()
                    .map(|t| match t {
                        RhsTerm::Var(v) => v.clone(),
                        _ => String::new(),
                    })
                    .filter(|s| !s.is_empty())
                    .collect();

                // endogenous = in endog but NOT in instr
                let x_endog_names: Vec<String> = endog_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .filter(|v| !instr_vars.contains(v))
                    .collect();
                // excluded instruments = in instr but NOT in endog
                let z_excl_names: Vec<String> = instr_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .filter(|v| !endog_vars.contains(v))
                    .collect();
                // included exogenous = in both
                let x_exog_names: Vec<String> = instr_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .filter(|v| endog_vars.contains(v.as_str()))
                    .collect();

                if x_endog_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "weak_iv: no endogenous variable identified (vars in endog but not in instr)".into()
                    ));
                }
                if z_excl_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "weak_iv: no excluded instrument identified (vars in instr but not in endog)".into()
                    ));
                }

                let n = df.n_rows();
                let k_endog = x_endog_names.len();
                let l = z_excl_names.len(); // number of excluded instruments
                let k_exog = x_exog_names.len() + 1; // +1 intercept

                // ── Build matrices ──
                // X_exog: intercept + included exogenous  (n × k_exog)
                let mut x_exog = Array2::<f64>::ones((n, k_exog));
                for (j, col) in x_exog_names.iter().enumerate() {
                    let v = df
                        .get(col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n {
                        x_exog[[i, j + 1]] = v[i];
                    }
                }

                // Z_excl: excluded instruments  (n × L)
                let mut z_excl = Array2::<f64>::zeros((n, l));
                for (j, col) in z_excl_names.iter().enumerate() {
                    let v = df
                        .get(col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n {
                        z_excl[[i, j]] = v[i];
                    }
                }

                // W = [X_exog | Z_excl]  (n × (k_exog + L))
                let mut w_full = Array2::<f64>::zeros((n, k_exog + l));
                w_full.slice_mut(ndarray::s![.., ..k_exog]).assign(&x_exog);
                w_full.slice_mut(ndarray::s![.., k_exog..]).assign(&z_excl);

                // X_endog: endogenous variables  (n × k_endog)
                let mut x_endog_mat = Array2::<f64>::zeros((n, k_endog));
                for (j, col) in x_endog_names.iter().enumerate() {
                    let v = df
                        .get(col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    for i in 0..n {
                        x_endog_mat[[i, j]] = v[i];
                    }
                }

                // ── M_exog = I - X_exog (X_exog'X_exog)⁻¹ X_exog' ──
                // to partial out included exogenous
                let xtx_exog = x_exog.t().dot(&x_exog);
                let xtx_exog_inv = xtx_exog
                    .inv()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // P_exog aplicado a qualquer matriz A: P_exog A = X_exog (X_exog'X_exog)⁻¹ X_exog' A
                let proj_exog = |a: &Array2<f64>| -> Array2<f64> {
                    x_exog.dot(&xtx_exog_inv.dot(&x_exog.t().dot(a)))
                };
                // M_exog Z_excl (partialling out exog from Z_excl)
                let mz = &z_excl - &proj_exog(&z_excl); // n × L
                                                        // M_exog X_endog
                let _mx = &x_endog_mat - &proj_exog(&x_endog_mat); // n × k_endog

                // ── First stage: regress X_endog on W_full ──
                let wtw = w_full.t().dot(&w_full);
                let wtw_inv = wtw
                    .inv()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let pi_hat = wtw_inv.dot(&w_full.t().dot(&x_endog_mat)); // (k_exog+L) × k_endog
                let x_hat = w_full.dot(&pi_hat); // n × k_endog
                let v_hat = &x_endog_mat - &x_hat; // 1st stage residuals

                // ── Π̂_Z: rows of pi_hat corresponding to Z_excl ──
                let pi_z = pi_hat.slice(ndarray::s![k_exog.., ..]).to_owned(); // L × k_endog

                // ── Σ̂_v = v̂'v̂ / (n - k_exog - L) ──
                let df_fs = n - k_exog - l;
                let vtv = v_hat.t().dot(&v_hat); // k_endog × k_endog
                let sigma_v = &vtv / df_fs as f64;

                // ── Matriz de Cragg-Donald: A = Π̂_Z' (Z'M_exog Z) Π̂_Z ──
                let zmz = mz.t().dot(&mz); // L × L  (= Z'M_exog Z)
                let cd_mat = pi_z.t().dot(&zmz.dot(&pi_z)); // k_endog × k_endog

                // ── 1st Stage F by endogenous variable (partial F on Z_excl) ──
                let mut first_stage_lines = String::new();
                for j in 0..k_endog {
                    // partial F = (Π̂_Zj' Z'M Z Π̂_Zj / L) / Σ̂_vj
                    let pi_zj = pi_z.column(j);
                    let numerator = pi_zj.dot(&zmz.dot(&pi_zj)) / l as f64;
                    let sigma_vj = sigma_v[[j, j]];
                    let f_j = if sigma_vj > 1e-15 {
                        numerator / sigma_vj
                    } else {
                        f64::NAN
                    };
                    let p_j = if f_j.is_finite() {
                        f_pvalue(f_j, l as f64, df_fs as f64)
                    } else {
                        f64::NAN
                    };
                    first_stage_lines.push_str(&format!(
                        "   {:<20} F({},{}) = {:>10.3}   p = {:.4}\n",
                        x_endog_names[j], l, df_fs, f_j, p_j
                    ));
                }

                // ── Cragg-Donald Wald F ──
                let sigma_v_inv = sigma_v
                    .inv()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cd_core = sigma_v_inv.dot(&cd_mat); // k_endog × k_endog

                let cd_stat = if k_endog == 1 {
                    cd_core[[0, 0]] / l as f64
                } else {
                    // λ_min of cd_core / L
                    let (eigenvalues, _) = cd_core
                        .eigh(UPLO::Lower)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    eigenvalues[0] / l as f64 // eigenvalues in ascending order
                };

                // ── Stock & Yogo (2005) critical values (k_endog=1, TSLS bias) ──
                let sy_table: Vec<(usize, [f64; 4])> = vec![
                    (1, [16.38, 8.96, 6.66, 5.53]),
                    (2, [19.93, 11.59, 8.75, 7.25]),
                    (3, [22.30, 12.83, 9.54, 7.80]),
                    (4, [24.58, 13.96, 10.26, 8.31]),
                    (5, [26.87, 15.09, 11.04, 8.84]),
                    (6, [28.55, 16.00, 11.65, 9.23]),
                    (7, [30.10, 16.87, 12.26, 9.63]),
                    (8, [31.49, 17.60, 12.82, 10.00]),
                    (9, [32.84, 18.37, 13.44, 10.37]),
                    (10, [34.16, 19.10, 14.01, 10.73]),
                ];
                let sy_line = if k_endog == 1 {
                    if let Some((_, cvs)) = sy_table.iter().find(|(lv, _)| *lv == l) {
                        format!(
                            "   Stock-Yogo (2005) — critical values for maximum TSLS bias (k_endog=1, L={}):\n   10%:{:.2}  15%:{:.2}  20%:{:.2}  25%:{:.2}\n",
                            l, cvs[0], cvs[1], cvs[2], cvs[3]
                        )
                    } else {
                        format!("   Stock-Yogo (2005): table available for L=1..10 (L={} out of range).\n   Rule of thumb (Staiger & Stock 1997): F > 10.\n", l)
                    }
                } else {
                    format!("   Stock-Yogo (2005): critical values for k_endog=1 only.\n   Para k_endog={}, see tables in Andrews, Stock & Sun (2019).\n", k_endog)
                };

                let thick = "═".repeat(70);
                let thin = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Weak Instrument Test\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    " n={n}  k_endog={k_endog}  L={l} (excluded instruments)\n"
                ));
                out.push_str("\n── 1st Stage F (partial F on excluded instruments)\n");
                out.push_str(&first_stage_lines);
                out.push_str(&format!("\n── Cragg-Donald Wald F = {:.4}\n", cd_stat));
                out.push_str("   (λ_min of concentration kernel / L)\n");
                out.push_str(&format!("\n{sy_line}"));
                out.push_str(&format!("{thin}\n"));
                out.push_str(" Rule of thumb: F > 10 (Staiger & Stock 1997)\n");
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Logit ─────────────────────────────────────────────────────────
            "logit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = Logit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let coef_names = coef_names_from_formula(&formula_ast, &df, x.ncols());
                Ok(Value::BinaryResult(BinaryModel {
                    result: Rc::new(result),
                    y,
                    x,
                    kind: "logit".into(),
                    coef_names,
                }))
            }

            // ── Probit ────────────────────────────────────────────────────────
            "probit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = Probit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let coef_names = coef_names_from_formula(&formula_ast, &df, x.ncols());
                Ok(Value::BinaryResult(BinaryModel {
                    result: Rc::new(result),
                    y,
                    x,
                    kind: "probit".into(),
                    coef_names,
                }))
            }

            // ── Heckman Two-Step (Heckit) ─────────────────────────────────────
            // heckman(outcome_formula, select_formula, df)
            // outcome: y ~ x1 + x2       (estimado apenas nos obs selecionados)
            // select:  z ~ w1 + w2 + w3  (probit em todos os obs; z deve ser 0/1)
            "heckman" | "heckit" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "heckman() requires (outcome_formula, selection_formula, df)".into(),
                    ));
                }
                let out_ast = self.resolve_formula(&args[0])?;
                let sel_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "heckman(): third argument must be DataFrame name".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "heckman: '{df_name}' is not a DataFrame"
                        )))
                    }
                };

                // Outcome equation
                let out_str = Self::formula_to_string(&out_ast);
                let g_out =
                    GFormula::parse(&out_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec_raw, x_out) = df
                    .to_design_matrix(&g_out)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let out_names = df
                    .formula_var_names(&g_out)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Selection equation
                let sel_str = Self::formula_to_string(&sel_ast);
                let g_sel =
                    GFormula::parse(&sel_str).map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (z_vec, x_sel) = df
                    .to_design_matrix(&g_sel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let sel_names = df
                    .formula_var_names(&g_sel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Heckman: y and x_out may contain NaN for unselected obs (z=0).
                // Replace NaN/Inf with 0.0 in those rows (values are not used in outcome equation).
                let y_vec = y_vec_raw.mapv(|v| if v.is_finite() { v } else { 0.0 });
                let x_out = x_out.mapv(|v| if v.is_finite() { v } else { 0.0 });

                let result = greeners::Heckman::fit(
                    &y_vec,
                    &x_out,
                    &z_vec,
                    &x_sel,
                    Some(out_names),
                    Some(sel_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::HeckmanResult(Rc::new(result)))
            }

            // ── Tobit — MLE with left censoring ──────────────────────────────
            // tobit(formula, df [, ll=0])
            "tobit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ll_limit = match opt_map.get("ll") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    None => 0.0,
                    _ => return Err(HayashiError::Runtime("tobit(): ll must be numeric".into())),
                };
                let result = greeners::Tobit::fit(&y_vec, &x_mat, ll_limit, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::TobitResult(Rc::new(result)))
            }

            // ── Regression Discontinuity — Sharp RD ─────────────────────────────
            // rd(outcome ~ running_var, cutoff, df [, bw=h, poly=1, kernel="triangular"])
            "rd" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "rd() requer (formula, cutoff, df [, bw=..., poly=..., kernel=...])".into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let cutoff = match self.eval_expr(&args[1])? {
                    Value::Float(v) => v,
                    Value::Int(v) => v as f64,
                    _ => {
                        return Err(HayashiError::Type(
                            "rd(): second argument must be cutoff (number)".into(),
                        ))
                    }
                };
                let df = match self.eval_expr(&args[2])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "rd(): third argument must be DataFrame".into(),
                        ))
                    }
                };

                // Extract names directly from Hayashi formula AST
                let outcome_name = formula_ast.lhs.clone();
                let running_name = formula_ast.rhs.first()
                    .and_then(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .ok_or_else(|| HayashiError::Runtime(
                        "rd(): formula must have exactly one variable on the right side (running var)".into()
                    ))?;

                let y = df
                    .get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let x = df
                    .get(&running_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();

                let bw = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("rd: bw must be numeric".into())),
                };
                let poly = match opt_map.get("poly") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => return Err(HayashiError::Runtime("rd: poly must be integer".into())),
                };
                let kernel = rd_kernel_opt(opt_map.get("kernel")).map_err(HayashiError::Runtime)?;

                let result = greeners::RD::fit(
                    &y,
                    &x,
                    cutoff,
                    bw,
                    poly,
                    kernel,
                    Some((outcome_name, running_name)),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RdResult(Rc::new(result)))
            }

            // ── Regression Discontinuity — Fuzzy RD ─────────────────────────────
            // fuzzy_rd(outcome ~ running_var, "treatment_col", cutoff, df [, bw=h, poly=1])
            "fuzzy_rd" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "fuzzy_rd() requer (formula, \"treatment\", cutoff, df [, bw=..., poly=...])".into()
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let treatment_name = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type(
                        "fuzzy_rd(): second argument must be the treatment column name (string)".into()
                    )),
                };
                let cutoff = match self.eval_expr(&args[2])? {
                    Value::Float(v) => v,
                    Value::Int(v) => v as f64,
                    _ => {
                        return Err(HayashiError::Type(
                            "fuzzy_rd(): third argument must be cutoff (number)".into(),
                        ))
                    }
                };
                let df = match self.eval_expr(&args[3])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "fuzzy_rd(): fourth argument must be DataFrame".into(),
                        ))
                    }
                };

                let outcome_name = formula_ast.lhs.clone();
                let running_name = formula_ast.rhs.first()
                    .and_then(|t| if let RhsTerm::Var(v) = t { Some(v.clone()) } else { None })
                    .ok_or_else(|| HayashiError::Runtime(
                        "fuzzy_rd(): formula must have exactly one variable on the right side (running var)".into()
                    ))?;

                let y = df
                    .get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let d = df
                    .get(&treatment_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let x = df
                    .get(&running_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();

                let bw = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("fuzzy_rd: bw must be numeric".into())),
                };
                let poly = match opt_map.get("poly") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "fuzzy_rd: poly must be integer".into(),
                        ))
                    }
                };
                let kernel = rd_kernel_opt(opt_map.get("kernel")).map_err(HayashiError::Runtime)?;

                let result = greeners::RD::fit_fuzzy(
                    &y,
                    &d,
                    &x,
                    cutoff,
                    bw,
                    poly,
                    kernel,
                    Some((outcome_name, running_name, treatment_name)),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RdResult(Rc::new(result)))
            }

            // ── Propensity Score Matching (Rosenbaum & Rubin 1983) ───────────
            // psm(outcome ~ treatment + cov1 + cov2, df [, k=1, caliper=0.2, replace=false, boot=200])
            // The 1st RHS term is the treatment; remaining are covariates for PS.
            "psm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "psm() requer (formula, df [, k=..., caliper=..., replace=..., boot=...])"
                            .into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "psm(): second argument must be DataFrame".into(),
                        ))
                    }
                };

                let outcome_name = formula_ast.lhs.clone();
                // First RHS = treatment; remaining = covariates
                let mut rhs_names: Vec<String> = formula_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if rhs_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "psm(): formula must have at least 'outcome ~ treatment'".into(),
                    ));
                }
                let treatment_name = rhs_names.remove(0);
                let covariate_names = rhs_names;

                if covariate_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "psm(): provide at least one covariate: outcome ~ treatment + cov1 + ..."
                            .into(),
                    ));
                }

                let y = df
                    .get(&outcome_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();
                let d = df
                    .get(&treatment_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    .to_owned();

                let x = {
                    let owned_cols: Vec<ndarray::Array1<f64>> = covariate_names
                        .iter()
                        .map(|c| {
                            df.get(c)
                                .map(|a| a.to_owned())
                                .map_err(|e| HayashiError::Runtime(e.to_string()))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let views: Vec<ndarray::ArrayView1<f64>> =
                        owned_cols.iter().map(|a| a.view()).collect();
                    ndarray::stack(ndarray::Axis(1), &views)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                };

                let k = match opt_map.get("k") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1,
                    _ => return Err(HayashiError::Runtime("psm: k must be integer".into())),
                };
                let caliper: Option<f64> = match opt_map.get("caliper") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    None => None,
                    _ => return Err(HayashiError::Runtime("psm: caliper must be numeric".into())),
                };
                let with_replacement = match opt_map.get("replace") {
                    Some(Value::Bool(b)) => *b,
                    None => false,
                    _ => return Err(HayashiError::Runtime("psm: replace must be boolean".into())),
                };
                let n_boot = match opt_map.get("boot") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 200,
                    _ => return Err(HayashiError::Runtime("psm: boot must be integer".into())),
                };

                let result = greeners::PSM::fit(
                    &y,
                    &d,
                    &x,
                    k,
                    caliper,
                    with_replacement,
                    n_boot,
                    Some((outcome_name, treatment_name, covariate_names)),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PsmResult(Rc::new(result)))
            }

            // ── Synthetic Control (ADH 2010) ────────────────────────────────
            // synth("outcome", "treated_id", t0, df, id="entity", time="year")
            // synth("outcome", "treated_id", t0, df, id="entity", time="year", covs=["x1","x2"])
            "synth" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "synth() requer (outcome, treated_id, t0, df, id=col, time=col [, covs=[...]])".into()
                    ));
                }
                let outcome_col =
                    match self.eval_expr(&args[0])? {
                        Value::Str(s) => s,
                        _ => return Err(HayashiError::Type(
                            "synth(): first argument must be outcome column name (string)"
                                .into(),
                        )),
                    };
                let treated_unit = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    Value::Int(v) => v.to_string(),
                    Value::Float(v) => (v as i64).to_string(),
                    _ => {
                        return Err(HayashiError::Type(
                            "synth(): second argument must be treated unit ID".into(),
                        ))
                    }
                };
                let t0 = match self.eval_expr(&args[2])? {
                    Value::Float(v) => v,
                    Value::Int(v)   => v as f64,
                    _ => return Err(HayashiError::Type(
                        "synth(): third argument must be treatment start period (number)".into()
                    )),
                };
                let df = match self.eval_expr(&args[3])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "synth(): fourth argument must be DataFrame".into(),
                        ))
                    }
                };

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "synth(): id=coluna option is required".into(),
                        ))
                    }
                };
                let time_col = match opt_map.get("time") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "synth(): time=coluna option is required".into(),
                        ))
                    }
                };
                let cov_cols: Option<Vec<String>> = match opt_map.get("covs") {
                    Some(Value::List(lst)) => Some(
                        lst.iter()
                            .map(|v| match v {
                                Value::Str(s) => Ok(s.clone()),
                                _ => Err(HayashiError::Type(
                                    "synth(): covs must be a list of strings".into(),
                                )),
                            })
                            .collect::<Result<Vec<_>>>()?,
                    ),
                    None => None,
                    _ => return Err(HayashiError::Runtime("synth(): covs must be a list".into())),
                };

                let result = greeners::SyntheticControl::fit(
                    &outcome_col,
                    &treated_unit,
                    t0,
                    &df,
                    &id_col,
                    &time_col,
                    cov_cols.as_deref(),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::SynthResult(Rc::new(result)))
            }

            // ── Poisson ───────────────────────────────────────────────────────
            "poisson" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cov = resolve_cov_full(opt_map, &df)?;
                let result =
                    greeners::Poisson::fit_with_names(&y_vec, &x_mat, cov, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PoissonResult(Rc::new(result)))
            }

            // ── Negative Binomial (NB2) ───────────────────────────────────────
            "nbreg" | "negbin" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cov = resolve_cov_full(opt_map, &df)?;
                let result = greeners::NegBin::fit_with_names(&y_vec, &x_mat, cov, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::NegBinResult(Rc::new(result)))
            }

            // ── Ordered Logit ─────────────────────────────────────────────────
            "ologit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::OrderedLogit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::OrderedResult(Rc::new(result)))
            }

            // ── Ordered Probit ────────────────────────────────────────────────
            "oprobit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::OrderedProbit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::OrderedResult(Rc::new(result)))
            }

            // ── Multinomial Logit ─────────────────────────────────────────────
            "mlogit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::MNLogit::fit_with_names(&y_vec, &x_mat, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::MNLogitResult(Rc::new(result)))
            }

            // ── Difference-in-Differences (2x2) ──────────────────────────────
            // did(outcome ~ treated_group + post_period, df, cov=HC1)
            "did" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "did(outcome ~ treated + post, df) requires formula and DataFrame".into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "did(): second argument must be DataFrame".into(),
                        ))
                    }
                };
                // formula: outcome ~ treated_col + post_col
                let rhs_vars: Vec<&str> = formula_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                if rhs_vars.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "did(): formula must have exactly 2 variables on RHS: treated + post"
                            .into(),
                    ));
                }
                let y = get_col_f64(&df, &formula_ast.lhs)?;
                let treated = get_col_f64(&df, rhs_vars[0])?;
                let post = get_col_f64(&df, rhs_vars[1])?;
                let cov = resolve_cov_full(opt_map, &df)?;
                let result = greeners::DiffInDiff::fit(&y, &treated, &post, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::DidResult(Rc::new(result)))
            }

            // ── Quantile Regression ───────────────────────────────────────────
            // qreg(y ~ x1 + x2, df, tau=0.5, boot=200)
            "qreg" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let tau = match opt_map.get("tau") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    None => 0.5,
                    _ => return Err(HayashiError::Type("tau= must be numeric".into())),
                };
                let n_boot = match opt_map.get("boot") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 200,
                    _ => return Err(HayashiError::Type("boot= must be integer".into())),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::QuantileReg::fit_with_names(
                    &y_vec,
                    &x_mat,
                    tau,
                    n_boot,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::QuantileResult(Rc::new(result)))
            }

            // ── Kaplan-Meier ──────────────────────────────────────────────────
            // km(time_col, event_col, df)
            "km" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "km(time, event, df) requires 3 arguments".into(),
                    ));
                }
                let time_name = match &args[0] {
                    Expr::Var(v) | Expr::Str(v) => v.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "km(): first argument must be nome da coluna de tempo".into(),
                        ))
                    }
                };
                let event_name = match &args[1] {
                    Expr::Var(v) | Expr::Str(v) => v.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "km(): second argument must be nome da coluna de evento".into(),
                        ))
                    }
                };
                let df = match self.eval_expr(&args[2])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "km(): third argument must be DataFrame".into(),
                        ))
                    }
                };
                let times = get_col_f64(&df, &time_name)?;
                let events_f = get_col_f64(&df, &event_name)?;
                let events: ndarray::Array1<u8> = events_f.iter().map(|&v| v as u8).collect();
                let result = greeners::KaplanMeier::fit(&times, &events)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::KMResult(Rc::new(result)))
            }

            // ── Cox Proportional Hazards ──────────────────────────────────────
            // cox(time_col ~ x1 + x2, df, event=event_col)
            "cox" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "cox(time ~ x1 + x2, df, event=col) requires formula and DataFrame".into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "cox(): second argument must be DataFrame".into(),
                        ))
                    }
                };
                let event_col = match opt_map.get("event") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "cox() requires event=coluna option".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("event= must be string".into())),
                };
                let times = get_col_f64(&df, &formula_ast.lhs)?;
                let events_f = get_col_f64(&df, &event_col)?;
                let events: ndarray::Array1<u8> = events_f.iter().map(|&v| v as u8).collect();
                // build covariate matrix from RHS variables
                let rhs_vars: Vec<String> = formula_ast
                    .rhs
                    .iter()
                    .filter_map(|t| {
                        if let RhsTerm::Var(v) = t {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if rhs_vars.is_empty() {
                    return Err(HayashiError::Runtime(
                        "cox(): formula needs at least one covariate on RHS".into(),
                    ));
                }
                let cols: Vec<ndarray::Array1<f64>> = rhs_vars
                    .iter()
                    .map(|v| get_col_f64(&df, v))
                    .collect::<Result<_>>()?;
                let n = times.len();
                let k = cols.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, col) in cols.iter().enumerate() {
                    x_mat.column_mut(j).assign(col);
                }
                let result =
                    greeners::CoxPH::fit_with_names(&times, &events, &x_mat, Some(rhs_vars))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::CoxResult(Rc::new(result)))
            }

            // ── Robust Linear Model (M-estimadores) ───────────────────────────
            // rlm(y ~ x1 + x2, df, norm=huber|tukey|andrews|hampel, cov=HC3)
            // default norm: Huber (c=1.345)
            "rlm" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let norm = match opt_map.get("norm") {
                    None => greeners::RobustNorm::Huber(1.345),
                    Some(Value::Str(s)) => match s.as_str() {
                        "huber" => greeners::RobustNorm::Huber(1.345),
                        "tukey" | "bisquare" => greeners::RobustNorm::Tukey(4.685),
                        "andrews" | "wave" => {
                            greeners::RobustNorm::AndrewWave(std::f64::consts::PI)
                        }
                        "hampel" => greeners::RobustNorm::Hampel(2.0, 4.0, 8.0),
                        "ols" | "leastsq" => greeners::RobustNorm::LeastSquares,
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "norm='{other}' unknown — use: huber, tukey, andrews, hampel, ols"
                            )))
                        }
                    },
                    _ => return Err(HayashiError::Type("norm= must be string".into())),
                };
                let cov = resolve_cov_full(opt_map, &df)?;
                let result =
                    greeners::RLM::fit_with_names(&y_vec, &x_mat, &norm, cov, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::RlmResult(Rc::new(result)))
            }

            // ── GEE (Generalized Estimating Equations) ────────────────────────
            // gee(y ~ x1 + x2, df, id=cluster_col, family=gaussian, corr=exchangeable)
            // family: gaussian (default), binomial, poisson
            // corr:   independence (default), exchangeable, ar1, unstructured
            "gee" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "gee() requires id=group_column option".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("id= must be string".into())),
                };
                let family_str = match opt_map.get("family") {
                    None => "gaussian",
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => "gaussian",
                        "binomial" | "logit" => "binomial",
                        "poisson" => "poisson",
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "family='{other}' unknown — use: gaussian, binomial, poisson"
                            )))
                        }
                    },
                    _ => return Err(HayashiError::Type("family= must be string".into())),
                };
                let corr_str = match opt_map.get("corr") {
                    None => "independence",
                    Some(Value::Str(s)) => s.as_str(),
                    _ => "independence",
                };
                let corr = match corr_str {
                    "independence" | "ind" => greeners::CorrStructure::Independence,
                    "exchangeable" | "exch" => greeners::CorrStructure::Exchangeable,
                    "ar1" | "ar(1)"        => greeners::CorrStructure::AR1,
                    "unstructured" | "uns" => greeners::CorrStructure::Unstructured,
                    other => return Err(HayashiError::Runtime(
                        format!("corr='{other}' unknown — use: independence, exchangeable, ar1, unstructured")
                    )),
                };
                let (family, link) = match family_str {
                    "binomial" => (greeners::Family::Binomial, greeners::Link::Logit),
                    "poisson" => (greeners::Family::Poisson, greeners::Link::Log),
                    _ => (greeners::Family::Gaussian, greeners::Link::Identity),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // convert id column to group indices (usize)
                let id_vals = get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut next_id = 0usize;
                let groups: ndarray::Array1<usize> = id_vals
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
                let result = greeners::GEE::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    &family,
                    &link,
                    &corr,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::GeeResult(Rc::new(result)))
            }

            // ── WLS (Weighted Least Squares) ──────────────────────────────────
            // wls(y ~ x1 + x2, df, weights="w_col", cov=HC3)
            "wls" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let w_name = match opt_map.get("weights") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "wls() requires weights=\"weights_column\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("weights= must be string".into())),
                };
                let weights = get_col_f64(&df, &w_name)?;
                let cov = resolve_cov_full(opt_map, &df)?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::WLS::fit_with_names(&y, &x, &weights, cov, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let fitted = x.dot(&result.params);
                let residuals = &y - &fitted;
                Ok(Value::OlsResult(OlsModel {
                    result: Rc::new(result),
                    residuals,
                    x,
                }))
            }

            // ── ZIP / ZINB (Zero-Inflated Count Models) ───────────────────────
            // zip(y ~ x1 + x2, df)
            // zip(y ~ x1 + x2, df, inflate=["x3", "x4"])
            // zinb(y ~ x1 + x2, df)
            "zip" | "zinb" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_count) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let count_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // inflate= optional: list of column names for inflation equation
                // If omitted, uses the same X matrix as the count model
                let (x_inflate_opt, inflate_names_opt): (
                    Option<ndarray::Array2<f64>>,
                    Option<Vec<String>>,
                ) = match opt_map.get("inflate") {
                    Some(Value::List(lst)) => {
                        let inames: Vec<String> = lst
                            .iter()
                            .map(|v| match v {
                                Value::Str(s) => Ok(s.clone()),
                                _ => Err(HayashiError::Type(
                                    "inflate= must be a list of strings".into(),
                                )),
                            })
                            .collect::<Result<_>>()?;
                        // intercept + colunas especificadas
                        let n = df.n_rows();
                        let k = inames.len() + 1;
                        let mut xi = ndarray::Array2::<f64>::ones((n, k));
                        for (j, name) in inames.iter().enumerate() {
                            xi.column_mut(j + 1).assign(&get_col_f64(&df, name)?);
                        }
                        let mut full_names = vec!["_cons".to_string()];
                        full_names.extend(inames);
                        (Some(xi), Some(full_names))
                    }
                    None => (None, None),
                    _ => {
                        return Err(HayashiError::Type(
                            "inflate= must be a list of strings".into(),
                        ))
                    }
                };

                let use_negbin = func == "zinb";
                let result = if use_negbin {
                    greeners::ZINB::fit_with_names(
                        &y_vec,
                        &x_count,
                        x_inflate_opt.as_ref(),
                        Some(count_names),
                        inflate_names_opt,
                    )
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else {
                    greeners::ZIP::fit_with_names(
                        &y_vec,
                        &x_count,
                        x_inflate_opt.as_ref(),
                        Some(count_names),
                        inflate_names_opt,
                    )
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?
                };
                Ok(Value::ZeroInflatedResult(Rc::new(result)))
            }

            // ── MixedLM (Mixed Linear Models — mixed effects) ────────────────
            // mixed(y ~ x1 + x2, df, id="group")           # random intercept
            // mixed(y ~ x1 + x2, df, id="group", re=["x1"]) # + random slope
            "mixed" | "mixedlm" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // id= required: group column
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "mixed() requires id=\"group_column\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("id= must be string".into())),
                };

                // re= optional: list of variables with random slope effect
                // Se omitido, modelo de random intercept apenas (re = [1])
                let re_vars: Vec<String> = match opt_map.get("re") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type("re= must be a list of strings".into())),
                        })
                        .collect::<Result<_>>()?,
                    None => vec![],
                    _ => return Err(HayashiError::Type("re= must be a list of strings".into())),
                };

                let (y_vec, x_fixed) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Convert id to group indices
                let id_vals = get_col_f64(&df, &id_col)?;
                let mut id_map: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut next_id = 0usize;
                let groups: ndarray::Array1<usize> = id_vals
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

                // Build x_random: intercept + specified slopes
                let n = df.n_rows();
                let q = re_vars.len() + 1; // +1 para random intercept
                let mut x_random = ndarray::Array2::<f64>::ones((n, q));
                for (j, name) in re_vars.iter().enumerate() {
                    x_random
                        .column_mut(j + 1)
                        .assign(&get_col_f64(&df, name)?);
                }

                let result = greeners::MixedLM::fit_with_names(
                    &y_vec,
                    &x_fixed,
                    &groups,
                    &x_random,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::MixedResult(Rc::new(result)))
            }

            // ── testparm — Joint Wald F-test (OLS/WLS) ────────────────────
            // testparm(model, ["x1", "x2"])
            // H0: β_x1 = β_x2 = 0 simultaneously
            "testparm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "testparm(model, [\"x1\", \"x2\"]) requires model + list of variables"
                            .into(),
                    ));
                }
                let model_val = self.eval_expr(&args[0])?;
                let tested: Vec<String> = match self.eval_expr(&args[1])? {
                    Value::List(lst) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type(
                                "testparm: list must contain strings".into(),
                            )),
                        })
                        .collect::<Result<_>>()?,
                    _ => {
                        return Err(HayashiError::Type(
                            "testparm: second argument must be list of strings".into(),
                        ))
                    }
                };
                match &model_val {
                    Value::OlsResult(m) => {
                        let vnames = m.result.variable_names.as_deref().unwrap_or(&[]);
                        let indices: Vec<usize> = tested.iter().map(|v| {
                            vnames.iter().position(|n| n == v)
                                .ok_or_else(|| HayashiError::Runtime(
                                    format!("testparm: variable '{v}' not found in model")
                                ))
                        }).collect::<Result<_>>()?;
                        let (f_stat, p_val) = m.result.f_test(&indices, &m.x)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        let df1 = indices.len();
                        let df2 = m.result.df_resid;
                        println!("\n{:=^62}", " testparm — Joint F Test ");
                        println!(" H0: {} = 0 (simultaneously)", tested.join(" = "));
                        println!("{:-^62}", "");
                        println!(" F({df1}, {df2})  =  {f_stat:.4}");
                        println!(" Prob > F      =  {p_val:.4}");
                        if p_val < 0.01       { println!(" Result: rejects H0 at 1%"); }
                        else if p_val < 0.05  { println!(" Result: rejects H0 at 5%"); }
                        else if p_val < 0.10  { println!(" Result: rejects H0 at 10%"); }
                        else                  { println!(" Result: does not reject H0 at 10%"); }
                        println!("{:=^62}", "");
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Runtime(
                        "testparm: current support only for OLS/WLS — other models use chi2; implement via wald_test()".into()
                    )),
                }
            }

            // ── GLSAR — GLS com erros AR(p) (Cochrane-Orcutt/Prais-Winsten) ─
            // glsar(y ~ x1 + x2, df, ar=1, iter=50)
            "glsar" | "prais" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ar_order = match opt_map.get("ar") {
                    Some(Value::Int(n)) => *n as usize,
                    None => 1,
                    _ => return Err(HayashiError::Type("ar= must be integer".into())),
                };
                let max_iter = match opt_map.get("iter") {
                    Some(Value::Int(n)) => *n as usize,
                    None => 50,
                    _ => return Err(HayashiError::Type("iter= must be integer".into())),
                };
                let result = greeners::GLSAR::fit_with_names(
                    &y_vec,
                    &x_mat,
                    ar_order,
                    max_iter,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::GlsarResult(Rc::new(result)))
            }

            // ── anova — ANOVA one-way ─────────────────────────────────────────
            // anova(df, outcome, by=group_col)
            "anova" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("anova(df, outcome, by=grupo)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "primeiro argumento deve ser DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let outcome_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be outcome variable name".into(),
                        ))
                    }
                };
                let outcome = get_col_f64(&df, &outcome_name)?;
                let by_col = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "anova() requer by=\"coluna_grupo\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("by= must be string".into())),
                };
                let group_vals = get_col_f64(&df, &by_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut next_g = 0usize;
                let groups: ndarray::Array1<usize> = group_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| {
                            let g = next_g;
                            next_g += 1;
                            g
                        })
                    })
                    .collect();
                let result = greeners::Stats::anova_oneway(&outcome, &groups)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("{result}");
                Ok(Value::Nil)
            }

            // ── Beta Regression ───────────────────────────────────────────────
            // betareg(y ~ x1 + x2, df)               # link=logit (default)
            // betareg(y ~ x1 + x2, df, link=probit)  # link alternativo
            // betareg(y ~ x1 + x2, df, link=cloglog)
            // Requires y ∈ (0,1) strictly (proportions, probabilities)
            "betareg" | "beta" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let link = match opt_map.get("link") {
                    None => greeners::BetaLink::Logit,
                    Some(Value::Str(s)) => match s.as_str() {
                        "logit" => greeners::BetaLink::Logit,
                        "probit" => greeners::BetaLink::Probit,
                        "cloglog" => greeners::BetaLink::CLogLog,
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "betareg: link='{other}' unknown — use: logit, probit, cloglog"
                            )))
                        }
                    },
                    _ => greeners::BetaLink::Logit,
                };
                let result =
                    greeners::BetaModel::fit_with_names(&y_vec, &x_mat, &link, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::BetaResult(Rc::new(result)))
            }

            // glm — Modelos Lineares Generalizados (IRLS via Greeners)
            // glm(y ~ x1 + x2, df, family=poisson, link=log, cov=robust)
            // Families: gaussian, binomial, poisson, gamma, inverse_gaussian, negbin, tweedie
            // Links: identity, log, logit, probit, inverse, cloglog
            // If link omitted uses canonical link of family
            "glm" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cov = resolve_cov_full(opt_map, &df)?;

                let alpha_val = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.0,
                };
                let power_val = match opt_map.get("power") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.5,
                };

                let family = match opt_map.get("family") {
                    None | Some(Value::Str(_)) if opt_map.get("family").is_none() => {
                        greeners::Family::Gaussian
                    }
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => greeners::Family::Gaussian,
                        "binomial" | "logistic" => greeners::Family::Binomial,
                        "poisson"  => greeners::Family::Poisson,
                        "gamma"    => greeners::Family::Gamma,
                        "inverse_gaussian" | "inversegaussian" => greeners::Family::InverseGaussian,
                        "negbin" | "negative_binomial" => greeners::Family::NegativeBinomial(alpha_val),
                        "tweedie" => greeners::Family::Tweedie(power_val),
                        other => return Err(HayashiError::Runtime(
                            format!("glm: family='{other}' unknown — use: gaussian, binomial, poisson, gamma, inverse_gaussian, negbin, tweedie")
                        )),
                    },
                    _ => greeners::Family::Gaussian,
                };

                let result = match opt_map.get("link") {
                    None => {
                        greeners::GLM::fit_with_names(&y_vec, &x_mat, family, cov, Some(var_names))
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    }
                    Some(Value::Str(s)) => {
                        let link = match s.as_str() {
                            "identity"  => greeners::Link::Identity,
                            "log"       => greeners::Link::Log,
                            "logit"     => greeners::Link::Logit,
                            "probit"    => greeners::Link::Probit,
                            "inverse"   => greeners::Link::InversePower,
                            "cloglog"   => greeners::Link::CLogLog,
                            other => return Err(HayashiError::Runtime(
                                format!("glm: link='{other}' unknown — use: identity, log, logit, probit, inverse, cloglog")
                            )),
                        };
                        // fit_with_link does not accept var_names; set after
                        let mut r = greeners::GLM::fit_with_link(&y_vec, &x_mat, family, link, cov)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        r.variable_names = Some(var_names);
                        r
                    }
                    _ => {
                        greeners::GLM::fit_with_names(&y_vec, &x_mat, family, cov, Some(var_names))
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?
                    }
                };
                Ok(Value::GlmResult(Rc::new(result)))
            }

            // influence — Influence diagnostics for OLS
            // influence(model, df)
            // Calculates DFBetas, DFFITS, leverage, studentized residuals
            // Prints summary and influential observations
            "influence" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("influence(model, df)".into()));
                }
                let model_val = self.eval_expr(&args[0])?;
                match &model_val {
                    Value::OlsResult(m) => {
                        let mse = m.result.sigma * m.result.sigma;
                        let result = greeners::Influence::compute(&m.residuals, &m.x, mse)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!("{result}");
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Runtime(
                        "influence(): only supported for OLS/WLS models — use: influence(m_ols, df)".into()
                    )),
                }
            }

            // lowess — Non-parametric LOWESS smoothing
            // lowess(df, y, x, frac=0.67, it=3)
            // frac: fraction of data used in each local fit (0 < frac ≤ 1)
            // it: robustification iterations (0 = no robustification)
            "lowess" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "lowess(df, y_var, x_var, frac=0.67, it=3)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "lowess: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let y_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "lowess: second argument must be y column name".into(),
                        ))
                    }
                };
                let x_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "lowess: third argument must be x column name".into(),
                        ))
                    }
                };
                let y_vec = ndarray::Array1::from(get_col_f64(&df, &y_name)?);
                let x_vec = ndarray::Array1::from(get_col_f64(&df, &x_name)?);
                let frac = match opt_map.get("frac") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    None => 0.6667,
                    _ => 0.6667,
                };
                let it = match opt_map.get("it") {
                    Some(Value::Int(v)) => *v as usize,
                    None => 3,
                    _ => 3,
                };
                let result = greeners::Lowess::fit(&y_vec, &x_vec, frac, it)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("{result}");
                Ok(Value::LowessResult(Rc::new(result)))
            }

            // kde — Kernel density estimation (univariate)
            // kde(df, var, bw=auto, kernel=gaussian)
            // Prints: n, bandwidth, support [min, max]
            "kde" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "kde(df, var, bw=auto, kernel=gaussian)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kde: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kde: second argument must be column name".into(),
                        ))
                    }
                };
                let data = ndarray::Array1::from(get_col_f64(&df, &var_name)?);
                let bw_opt = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let kernel = match opt_map.get("kernel") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => greeners::Kernel::Gaussian,
                        "epanechnikov" => greeners::Kernel::Epanechnikov,
                        "triangular"   => greeners::Kernel::Triangular,
                        "uniform"      => greeners::Kernel::Uniform,
                        other => return Err(HayashiError::Runtime(
                            format!("kde: kernel='{other}' unknown — use: gaussian, epanechnikov, triangular, uniform")
                        )),
                    },
                    _ => greeners::Kernel::Gaussian,
                };
                let result = greeners::KDEUnivariate::fit(&data, bw_opt, kernel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let support_min = result.support.iter().cloned().fold(f64::INFINITY, f64::min);
                let support_max = result
                    .support
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let peak_idx = result
                    .density
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let peak_x = result.support[peak_idx];
                let peak_d = result.density[peak_idx];
                println!("\n{:=^50}", " KDE ");
                println!("{:<20} {:>10}", "Variable:", var_name);
                println!("{:<20} {:>10}", "Observations:", result.n_obs);
                println!("{:<20} {:>10.6}", "Bandwidth:", result.bandwidth);
                println!("{:<20} {:>10.4}", "Support min:", support_min);
                println!("{:<20} {:>10.4}", "Support max:", support_max);
                println!(
                    "{:<20} {:>10.4} @ x = {:.4}",
                    "Peak (density):", peak_d, peak_x
                );
                println!("{:=^50}", "");
                Ok(Value::Nil)
            }

            // pca — Principal Component Analysis
            // pca(df, x1, x2, x3, n=2)
            // n=: number of components (default: min(vars, obs-1))
            // Based on eigenvalue decomposition of correlation matrix
            // Variables are standardized automatically (equivalent to PCA cor)
            "pca" | "princomp" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "pca(df, x1, x2, x3, ..., n=k)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "pca: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let var_names = self.resolve_var_list(&args[1..], &df)?;
                let n = df.n_rows();
                let k = var_names.len();
                let n_components = match opt_map.get("n") {
                    Some(Value::Int(v)) => (*v as usize).min(k).min(n - 1),
                    Some(Value::Float(v)) => (*v as usize).min(k).min(n - 1),
                    _ => k.min(n - 1),
                };
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }
                let result = greeners::PCA::fit(&data, n_components)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PcaResult(PcaModel {
                    result: Rc::new(result),
                    var_names,
                }))
            }

            // factor — Factor Analysis (principal axis)
            // factor(df, x1, x2, x3, n=2, rotation=varimax)
            // rotation=: none (default), varimax
            // Difference from PCA: PCA maximizes explained variance;
            //   FA estimates latent factors with specific covariance structure
            "factor" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "factor(df, x1, x2, x3, ..., n=k, rotation=none|varimax)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "factor: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let var_names = self.resolve_var_list(&args[1..], &df)?;
                let n = df.n_rows();
                let k = var_names.len();
                let n_factors = match opt_map.get("n") {
                    Some(Value::Int(v)) => (*v as usize).min(k),
                    Some(Value::Float(v)) => (*v as usize).min(k),
                    _ => k.min(2),
                };
                let rotation = match opt_map.get("rotation") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "varimax" => greeners::Rotation::Varimax,
                        "none" => greeners::Rotation::None,
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "factor: rotation='{other}' unknown — use: none, varimax"
                            )))
                        }
                    },
                    _ => greeners::Rotation::None,
                };
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }
                let result = greeners::FactorAnalysis::fit(&data, n_factors, rotation)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::FactorResult(FactorModel {
                    result: Rc::new(result),
                    var_names,
                }))
            }

            // manova — Multivariate Analysis of Variance (one-way)
            // manova(df, y1, y2, ..., by="group")
            // Tests H0: mean vectors equal across groups
            // Statistics: Wilks' Λ, Pillai's trace, Hotelling-Lawley, Roy's root
            "manova" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "manova(df, y1, y2, ..., by=\"group_col\")".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "manova: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let group_col = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "manova requer by=\"coluna_grupo\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("manova: by= must be string".into())),
                };
                let outcome_names = self.resolve_var_list(&args[1..], &df)?;
                let n = df.n_rows();
                let q = outcome_names.len();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, q));
                for (j, vname) in outcome_names.iter().enumerate() {
                    let col = get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        y_mat[[i, j]] = v;
                    }
                }
                let group_vals = get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: ndarray::Array1<usize> = ndarray::Array1::from(
                    group_vals
                        .iter()
                        .map(|&v| {
                            let key = v as i64;
                            *gmap.entry(key).or_insert_with(|| {
                                let g = gnext;
                                gnext += 1;
                                g
                            })
                        })
                        .collect::<Vec<_>>(),
                );
                let result = greeners::MANOVA::fit(&y_mat, &groups)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("{result}");
                Ok(Value::Nil)
            }

            // ── User-defined function ──────────────────────────────────
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
