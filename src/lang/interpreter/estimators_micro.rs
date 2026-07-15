use super::helpers::*;
use super::models::FactorModel;
use super::*;

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

                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

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
                pairs.sort_by(|a, b| nan_last_cmp(&a.0, &b.0));
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
                    indexed.sort_by(|a, b| nan_last_cmp(&a.1, &b.1));
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
                let df_raw2 = self.maybe_filter_df(&df_raw, opts)?;
                let (df, g_formula, display_names) =
                    self.prepare_formula(&formula_ast, &df_raw2)?;
                let cov = resolve_cov_full(opt_map, &df)?;

                let (y, x) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Usa fit_with_names para preservar nomes legíveis (e.g. "log(K):log(L)")
                let var_names: Vec<String> = std::iter::once("_cons".to_string())
                    .chain(display_names)
                    .collect();
                let result = OLS::fit_with_names(&y, &x, cov, Some(var_names))
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
                let (df_endog, g_endog, _) = self.prepare_formula(&endog_ast, &df)?;
                let cov = resolve_cov_full(opt_map, &df_endog)?;

                // Instrumento pode ter LHS vazio (~ z1 + z2)
                let g_instr = if instr_ast.lhs.is_empty() {
                    let (df_instr, g_i, _) = self.prepare_formula(&instr_ast, &df)?;
                    let _ = df_instr; // df aumentado não é necessário para instrumento
                    GFormula {
                        dependent: String::new(),
                        independents: g_i.independents,
                        intercept: true,
                    }
                } else {
                    let (_, g_i, _) = self.prepare_formula(&instr_ast, &df)?;
                    g_i
                };

                let result = IV::from_formula(&g_endog, &g_instr, &df_endog, cov)
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
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
                    .collect();
                let instr_vars: std::collections::HashSet<String> = instr_ast
                    .rhs
                    .iter()
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
                    .collect();

                // endogenous = in endog but NOT in instr
                let x_endog_names: Vec<String> = endog_ast
                    .rhs
                    .iter()
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
                    .filter(|v| !instr_vars.contains(v))
                    .collect();
                // excluded instruments = in instr but NOT in endog
                let z_excl_names: Vec<String> = instr_ast
                    .rhs
                    .iter()
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
                    .filter(|v| !endog_vars.contains(v))
                    .collect();
                // included exogenous = in both
                let x_exog_names: Vec<String> = instr_ast
                    .rhs
                    .iter()
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
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

            // ── Sargan / Hansen J overidentification test ───────────────────
            // estat_overid(endog_formula, instrument_formula, df)
            "estat_overid" | "sargan" | "overid" | "sargan_test" => {
                if args.len() < 3 {
                    return Err(self.rt_err(
                        "estat_overid(endog_formula, instrument_formula, df) requires 3 arguments",
                    ));
                }
                let endog_ast = self.resolve_formula(&args[0])?;
                let instr_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(self.rt_err("third argument must be a DataFrame variable")),
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
                // Build y, x, z
                let (y, x) = df_endog
                    .to_design_matrix(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let instr_formula = GFormula {
                    dependent: g_endog.dependent.clone(),
                    independents: g_instr.independents.clone(),
                    intercept: g_instr.intercept,
                };
                let (_, z) = df_endog
                    .to_design_matrix(&instr_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // Fit IV to get beta
                let iv_result = IV::fit_with_names(&y, &x, &z, CovarianceType::NonRobust, None)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = IV::sargan_test(&y, &x, &z, &iv_result.params)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Durbin-Wu-Hausman endogeneity test ──────────────────────────
            // estat_endog(endog_formula, instrument_formula, df)
            "estat_endog" | "endog_test" | "dwh" => {
                if args.len() < 3 {
                    return Err(self.rt_err(
                        "estat_endog(endog_formula, instrument_formula, df) requires 3 arguments",
                    ));
                }
                let endog_ast = self.resolve_formula(&args[0])?;
                let instr_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(self.rt_err("third argument must be a DataFrame variable")),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let (df_endog, g_endog, _) = self.prepare_formula(&endog_ast, &df)?;

                // Identify endogenous variables (in endog but NOT in instr)
                let instr_vars: std::collections::HashSet<String> = instr_ast
                    .rhs
                    .iter()
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
                    .collect();
                let endog_var_names: Vec<String> = endog_ast
                    .rhs
                    .iter()
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
                    .filter(|v| !instr_vars.contains(v))
                    .collect();

                if endog_var_names.is_empty() {
                    return Err(self.rt_err(
                        "estat_endog: no endogenous variable found (variables in endog formula not present in instrument formula)",
                    ));
                }

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

                // Build y, x, z
                let (y, x) = df_endog
                    .to_design_matrix(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let instr_formula = GFormula {
                    dependent: g_endog.dependent.clone(),
                    independents: g_instr.independents.clone(),
                    intercept: g_instr.intercept,
                };
                let (_, z) = df_endog
                    .to_design_matrix(&instr_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Find column indices of endogenous variables in X
                // X columns come from g_endog.independents (+ intercept if present)
                let x_names = df_endog
                    .formula_var_names(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let endog_cols: Vec<usize> = x_names
                    .iter()
                    .enumerate()
                    .filter(|(_, name)| endog_var_names.contains(name))
                    .map(|(i, _)| i)
                    .collect();

                let result = IV::endogeneity_test(&y, &x, &z, &endog_cols, endog_var_names)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Binary model diagnostics ────────────────────────────────────
            // estat_classification(model [, threshold=0.5])
            "estat_classification" | "classification" => {
                if args.is_empty() {
                    return Err(
                        self.rt_err("estat_classification(model) requires a logit/probit model")
                    );
                }
                let v = self.eval_expr(&args[0])?;
                let model = match &v {
                    Value::BinaryResult(m) => m.clone(),
                    _ => {
                        return Err(self
                            .rt_err("estat_classification: argument must be a logit/probit model"))
                    }
                };
                let threshold = match opt_map.get("threshold") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.5,
                };
                let probs = model.result.predict_proba(&model.x);
                let y_slice: Vec<f64> = model.y.to_vec();
                let probs_slice: Vec<f64> = probs.to_vec();
                let result = BinaryDiagnostics::classification(&y_slice, &probs_slice, threshold)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                print!("{result}");
                Ok(Value::Nil)
            }

            // lroc(model) / roc(model) — ROC curve and AUC
            "lroc" | "roc" | "estat_roc" => {
                if args.is_empty() {
                    return Err(self.rt_err("lroc(model) requires a logit/probit model"));
                }
                let v = self.eval_expr(&args[0])?;
                let model = match &v {
                    Value::BinaryResult(m) => m.clone(),
                    _ => return Err(self.rt_err("lroc: argument must be a logit/probit model")),
                };
                let probs = model.result.predict_proba(&model.x);
                let y_slice: Vec<f64> = model.y.to_vec();
                let probs_slice: Vec<f64> = probs.to_vec();
                let result = BinaryDiagnostics::roc(&y_slice, &probs_slice)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                print!("{result}");
                Ok(Value::Nil)
            }

            // estat_gof(model [, groups=10]) — Hosmer-Lemeshow
            "estat_gof" | "hosmer_lemeshow" | "hltest" => {
                if args.is_empty() {
                    return Err(self.rt_err("estat_gof(model) requires a logit/probit model"));
                }
                let v = self.eval_expr(&args[0])?;
                let model = match &v {
                    Value::BinaryResult(m) => m.clone(),
                    _ => {
                        return Err(self.rt_err("estat_gof: argument must be a logit/probit model"))
                    }
                };
                let n_groups = match opt_map.get("groups") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };
                let probs = model.result.predict_proba(&model.x);
                let y_slice: Vec<f64> = model.y.to_vec();
                let probs_slice: Vec<f64> = probs.to_vec();
                let result = BinaryDiagnostics::hosmer_lemeshow(&y_slice, &probs_slice, n_groups)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                print!("{result}");
                Ok(Value::Nil)
            }

            // linktest(model) — specification error detection
            "linktest" => {
                if args.is_empty() {
                    return Err(self.rt_err("linktest(model) requires a logit/probit model"));
                }
                let v = self.eval_expr(&args[0])?;
                let model = match &v {
                    Value::BinaryResult(m) => m.clone(),
                    _ => return Err(self.rt_err("linktest: argument must be a logit/probit model")),
                };
                let result = BinaryDiagnostics::linktest(&model.y, &model.x, &model.result.params)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Logit ─────────────────────────────────────────────────────────
            "logit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df_out, g_out, out_display) = self.prepare_formula(&out_ast, &df)?;
                let (y_vec_raw, x_out) = df_out
                    .to_design_matrix(&g_out)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let out_names = {
                    let mut n = vec!["_cons".to_string()];
                    n.extend(out_display);
                    n
                };

                // Selection equation
                let (df_sel, g_sel, sel_display) = self.prepare_formula(&sel_ast, &df)?;
                let (z_vec, x_sel) = df_sel
                    .to_design_matrix(&g_sel)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let sel_names = {
                    let mut n = vec!["_cons".to_string()];
                    n.extend(sel_display);
                    n
                };

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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                    .and_then(|t| t.as_var().map(|s| s.to_string()))
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
                        "fuzzy_rd(): second argument must be the treatment column name (string)"
                            .into(),
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
                    .and_then(|t| t.as_var().map(|s| s.to_string()))
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
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
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
                let outcome_col = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => {
                        return Err(HayashiError::Type(
                            "synth(): first argument must be outcome column name (string)".into(),
                        ))
                    }
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
                    Value::Int(v) => v as f64,
                    _ => {
                        return Err(HayashiError::Type(
                            "synth(): third argument must be treatment start period (number)"
                                .into(),
                        ))
                    }
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let result = greeners::OrderedLogit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::OrderedResult(Rc::new(result)))
            }

            // ── Ordered Probit ────────────────────────────────────────────────
            "oprobit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let result = greeners::OrderedProbit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::OrderedResult(Rc::new(result)))
            }

            // ── Multinomial Logit ─────────────────────────────────────────────
            "mlogit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let rhs_vars: Vec<String> = formula_ast
                    .rhs
                    .iter()
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
                    .collect();
                if rhs_vars.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "did(): formula must have exactly 2 variables on RHS: treated + post"
                            .into(),
                    ));
                }
                let y = get_col_f64(&df, &formula_ast.lhs)?;
                let treated = get_col_f64(&df, &rhs_vars[0])?;
                let post = get_col_f64(&df, &rhs_vars[1])?;
                let cov = resolve_cov_full(opt_map, &df)?;
                let result = greeners::DiffInDiff::fit(&y, &treated, &post, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::DidResult(Rc::new(result)))
            }

            // ── Event Study (dynamic DiD with leads and lags) ───────────────
            // eventstudy(y ~ event_time + x1 + x2, df, ref=-1, min=-5, max=5, cov=HC1)
            "eventstudy" | "event_study" | "es" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "eventstudy(y ~ event_time + controls, df) requires formula and DataFrame"
                            .into(),
                    ));
                }
                let formula_ast = self.resolve_formula(&args[0])?;
                let df = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "eventstudy(): second argument must be DataFrame".into(),
                        ))
                    }
                };
                let rhs_vars: Vec<String> = formula_ast
                    .rhs
                    .iter()
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
                    .collect();
                if rhs_vars.is_empty() {
                    return Err(HayashiError::Runtime(
                        "eventstudy(): formula must have at least event_time on RHS".into(),
                    ));
                }
                let event_col = &rhs_vars[0];
                let y = get_col_f64(&df, &formula_ast.lhs)?;
                let event_vals = get_col_f64(&df, event_col)?;
                let event_time: Vec<i64> = event_vals.iter().map(|&v| v as i64).collect();

                // Controls: remaining RHS vars
                let n = y.len();
                let control_vars = &rhs_vars[1..];
                let x_controls = if control_vars.is_empty() {
                    ndarray::Array2::zeros((n, 0))
                } else {
                    let mut x = ndarray::Array2::zeros((n, control_vars.len()));
                    for (j, v) in control_vars.iter().enumerate() {
                        let col = get_col_f64(&df, v)?;
                        for i in 0..n {
                            x[(i, j)] = col[i];
                        }
                    }
                    x
                };

                let reference = match opt_map.get("ref") {
                    Some(Value::Int(v)) => *v,
                    Some(Value::Float(v)) => *v as i64,
                    None => -1,
                    _ => -1,
                };
                let min_t = match opt_map.get("min") {
                    Some(Value::Int(v)) => *v,
                    Some(Value::Float(v)) => *v as i64,
                    None => -5,
                    _ => -5,
                };
                let max_t = match opt_map.get("max") {
                    Some(Value::Int(v)) => *v,
                    Some(Value::Float(v)) => *v as i64,
                    None => 5,
                    _ => 5,
                };
                let cov = resolve_cov_full(opt_map, &df)?;
                let result = greeners::EventStudy::fit(
                    &y,
                    &event_time,
                    &x_controls,
                    reference,
                    min_t,
                    max_t,
                    cov,
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                print!("{result}");
                Ok(Value::Nil)
            }

            // ── NLS (Nonlinear Least Squares) ────────────────────────────────
            // nls_exp(y ~ x, df, start=[1.0, 0.5])
            // nls_power(y ~ x, df, start=[1.0, 0.5])
            // nls_logistic(y ~ x, df, start=[100.0, 1.0, 5.0])
            // nls_cobb_douglas(y ~ x1 + x2, df, start=[1.0, 0.5, 0.5])
            "nls_exp" | "nls_power" | "nls_logistic" | "nls_cobb_douglas" | "nls_ces" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                // For NLS, extract raw RHS columns WITHOUT intercept
                // (the nonlinear function has its own scale parameter)
                let rhs_vars: Vec<String> = g_formula.independents.clone();
                let n = df.n_rows();
                let n_x = rhs_vars.len();
                let mut x_mat = ndarray::Array2::zeros((n, n_x));
                for (j, v) in rhs_vars.iter().enumerate() {
                    let col = get_col_f64(&df, v)?;
                    for i in 0..n {
                        x_mat[(i, j)] = col[i];
                    }
                }
                let y_vec = get_col_f64(&df, &g_formula.dependent)?;

                // Parse start values from start=[...] option
                let start: Vec<f64> = match opt_map.get("start") {
                    Some(Value::List(items)) => items
                        .iter()
                        .filter_map(|v| match v {
                            Value::Float(f) => Some(*f),
                            Value::Int(i) => Some(*i as f64),
                            _ => None,
                        })
                        .collect(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires start=[v1, v2, ...] option with starting values"
                        )))
                    }
                };

                let max_iter = match opt_map.get("max_iter") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 200,
                    _ => 200,
                };
                let tol = match opt_map.get("tol") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    None => 1e-8,
                    _ => 1e-8,
                };

                #[allow(clippy::type_complexity)]
                let (predict_fn, param_names): (
                    &dyn Fn(&[f64], &[f64]) -> f64,
                    Vec<String>,
                ) = match func {
                    "nls_exp" => (&greeners::predict_exp, vec!["a".into(), "b".into()]),
                    "nls_power" => (&greeners::predict_power, vec!["a".into(), "b".into()]),
                    "nls_logistic" => (
                        &greeners::predict_logistic,
                        vec!["a".into(), "b".into(), "c".into()],
                    ),
                    "nls_cobb_douglas" => {
                        let mut names = vec!["a".into()];
                        for i in 0..n_x {
                            names.push(format!("b{i}"));
                        }
                        (&greeners::predict_cobb_douglas, names)
                    }
                    "nls_ces" => (
                        &greeners::predict_ces,
                        vec!["a".into(), "b1".into(), "b2".into(), "rho".into()],
                    ),
                    _ => unreachable!(),
                };

                if start.len() != param_names.len() {
                    return Err(HayashiError::Runtime(format!(
                        "{func}() requires {} starting values, got {}",
                        param_names.len(),
                        start.len()
                    )));
                }

                let result = greeners::NLS::fit_with_names(
                    &y_vec,
                    &x_mat,
                    predict_fn,
                    &start,
                    param_names,
                    max_iter,
                    tol,
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Double/Debiased ML ───────────────────────────────────────────
            // double_ml(y ~ d + x1 + x2, df [, folds=5, poly=2])
            "double_ml" | "dml" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                // First RHS variable is treatment (d), rest are controls (x)
                let independents = &g_formula.independents;
                if independents.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "double_ml() requires y ~ d + x1 + x2 + ... (treatment + controls)".into(),
                    ));
                }
                let d_var = &independents[0];
                let x_vars = &independents[1..];

                let n = df.n_rows();
                let y_vec = get_col_f64(&df, &g_formula.dependent)?;
                let d_vec = get_col_f64(&df, d_var)?;
                let mut x_mat = ndarray::Array2::zeros((n, x_vars.len()));
                for (j, v) in x_vars.iter().enumerate() {
                    let col = get_col_f64(&df, v)?;
                    for i in 0..n {
                        x_mat[(i, j)] = col[i];
                    }
                }

                let n_folds = match opt_map.get("folds") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 5,
                    _ => 5,
                };
                let poly_degree = match opt_map.get("poly") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 2,
                    _ => 2,
                };

                let result =
                    greeners::DoubleML::fit_plr(&y_vec, &d_vec, &x_mat, n_folds, poly_degree)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Stochastic Frontier ──────────────────────────────────────────
            // sfa_production(y ~ x1 + x2, df)
            // sfa_cost(y ~ x1 + x2, df)
            "sfa_production" | "sfa_cost" | "frontier" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let var_names = g_formula.independents.clone();
                let model_type = if func == "sfa_cost" {
                    "cost"
                } else {
                    "production"
                };

                let result = if model_type == "production" {
                    greeners::StochasticFrontier::fit_production(&y_vec, &x_mat, Some(var_names))
                } else {
                    greeners::StochasticFrontier::fit_cost(&y_vec, &x_mat, Some(var_names))
                }
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Panel Tobit (random effects) ─────────────────────────────────
            // panel_tobit(y ~ x1 + x2, df, id="firm" [, censor=0])
            "panel_tobit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "panel_tobit() requires id=\"column\" option".into(),
                        ))
                    }
                };
                let panel_ids: Vec<i64> = {
                    let col = df
                        .get_column(&id_col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(int_arr) = col.as_int() {
                        int_arr.iter().copied().collect()
                    } else if let Some(float_arr) = col.as_float() {
                        float_arr.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "panel_tobit: id column '{id_col}' must be numeric"
                        )));
                    }
                };

                let censor = match opt_map.get("censor") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    None => 0.0,
                    _ => 0.0,
                };

                let var_names = g_formula.independents.clone();
                let result =
                    greeners::PanelTobit::fit(&y_vec, &x_mat, &panel_ids, censor, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Panel Heckman (selection with random effects) ───────────────
            // panel_heckman(y ~ x1 + x2, df, sel="z ~ w1 + w2", id="firm")
            "panel_heckman" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "panel_heckman() requires id=\"column\" option".into(),
                        ))
                    }
                };
                let sel_formula_str = match opt_map.get("sel") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "panel_heckman() requires sel=\"z ~ w1 + w2\" option".into(),
                        ))
                    }
                };

                // Parse selection formula from string
                let sel_hayashi_formula = {
                    let parts: Vec<&str> = sel_formula_str.splitn(2, '~').collect();
                    if parts.len() != 2 {
                        return Err(HayashiError::Runtime(format!(
                            "panel_heckman: sel formula '{sel_formula_str}' is not valid (needs ~)"
                        )));
                    }
                    let lhs = parts[0].trim().to_string();
                    let rhs_str = parts[1].trim();
                    let rhs: Vec<crate::lang::ast::RhsTerm> = rhs_str
                        .split('+')
                        .map(|t| crate::lang::ast::RhsTerm::var(t.trim()))
                        .collect();
                    crate::lang::ast::Formula {
                        lhs,
                        rhs,
                        fe: vec![],
                    }
                };
                let (df_sel, sel_g_formula, _) = self.prepare_formula(&sel_hayashi_formula, &df)?;
                let z_col = &sel_g_formula.dependent;
                let z_vec: Vec<bool> = {
                    let col = df_sel
                        .get_column(z_col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(b) = col.as_bool() {
                        b.iter().copied().collect()
                    } else if let Some(f) = col.as_float() {
                        f.iter().map(|v| *v > 0.0).collect()
                    } else if let Some(i) = col.as_int() {
                        i.iter().map(|v| *v != 0).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "panel_heckman: selection variable '{z_col}' must be numeric/boolean"
                        )));
                    }
                };
                let (_z_dummy, w_mat) = df_sel
                    .to_design_matrix(&sel_g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let panel_ids: Vec<i64> = {
                    let col = df
                        .get_column(&id_col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(int_arr) = col.as_int() {
                        int_arr.iter().copied().collect()
                    } else if let Some(float_arr) = col.as_float() {
                        float_arr.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "panel_heckman: id column '{id_col}' must be numeric"
                        )));
                    }
                };

                let sel_names = sel_g_formula.independents.clone();
                let out_names = g_formula.independents.clone();
                let result = greeners::PanelHeckman::fit(
                    &z_vec,
                    &y_vec,
                    &w_mat,
                    &x_mat,
                    &panel_ids,
                    Some(sel_names),
                    Some(out_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Spatial panel (SAR/SEM with fixed effects) ──────────────────
            // spatial_panel_sar(y ~ x1 + x2, df, w=W, id="entity")
            // spatial_panel_sem(y ~ x1 + x2, df, w=W, id="entity")
            "spatial_panel_sar" | "spatial_panel_sem" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "spatial_panel requires id=\"column\" option".into(),
                        ))
                    }
                };
                let entity_ids: Vec<i64> = {
                    let col = df
                        .get_column(&id_col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(int_arr) = col.as_int() {
                        int_arr.iter().copied().collect()
                    } else if let Some(float_arr) = col.as_float() {
                        float_arr.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "spatial_panel: id column '{id_col}' must be numeric"
                        )));
                    }
                };

                // Extract W matrix from w= option (list of lists)
                let w_mat = match opt_map.get("w") {
                    Some(Value::List(rows)) => {
                        let n_rows = rows.len();
                        let mut w = ndarray::Array2::<f64>::zeros((n_rows, n_rows));
                        for (i, row) in rows.iter().enumerate() {
                            match row {
                                Value::List(cols) => {
                                    if cols.len() != n_rows {
                                        return Err(HayashiError::Runtime(format!(
                                            "{func}: W must be square, row {i} has {} cols, expected {n_rows}",
                                            cols.len()
                                        )));
                                    }
                                    for (j, val) in cols.iter().enumerate() {
                                        w[(i, j)] = match val {
                                            Value::Float(f) => *f,
                                            Value::Int(v) => *v as f64,
                                            _ => return Err(HayashiError::Runtime(
                                                format!("{func}: W matrix contains non-numeric values")
                                            )),
                                        };
                                    }
                                }
                                _ => return Err(HayashiError::Runtime(
                                    format!("{func}: W must be a list of lists (matrix)")
                                )),
                            }
                        }
                        w
                    }
                    _ => return Err(HayashiError::Runtime(
                        format!("{func}() requires w=W option with a spatial weights matrix (list of lists)")
                    )),
                };

                let var_names = g_formula.independents.clone();
                let result = if func == "spatial_panel_sar" {
                    greeners::SpatialPanel::fit_sar(
                        &y_vec,
                        &x_mat,
                        &w_mat,
                        &entity_ids,
                        Some(var_names),
                    )
                } else {
                    greeners::SpatialPanel::fit_sem(
                        &y_vec,
                        &x_mat,
                        &w_mat,
                        &entity_ids,
                        Some(var_names),
                    )
                }
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Bayesian Stochastic Frontier ─────────────────────────────────
            // bayes_sfa_production(y ~ x1 + x2, df [, burn=1000, draws=2000])
            // bayes_sfa_cost(y ~ x1 + x2, df [, burn=1000, draws=2000])
            "bayes_sfa_production" | "bayes_sfa_cost" | "bayes_frontier" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let n_burn = match opt_map.get("burn") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 1000,
                    _ => 1000,
                };
                let n_draws = match opt_map.get("draws") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    None => 2000,
                    _ => 2000,
                };

                let var_names = g_formula.independents.clone();
                let model_type = if func == "bayes_sfa_cost" {
                    "cost"
                } else {
                    "production"
                };
                let result = if model_type == "production" {
                    greeners::BayesianSFA::fit_production(
                        &y_vec,
                        &x_mat,
                        Some(var_names),
                        n_burn,
                        n_draws,
                    )
                } else {
                    greeners::BayesianSFA::fit_cost(
                        &y_vec,
                        &x_mat,
                        Some(var_names),
                        n_burn,
                        n_draws,
                    )
                }
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── MIDAS ────────────────────────────────────────────────────────
            // midas(y ~ x, df, freq=3, lags=12, poly=2)
            "midas" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let freq = match opt_map.get("freq") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 3,
                };
                let n_lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };
                let poly = match opt_map.get("poly") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };

                if g_formula.independents.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "midas: exactly one high-frequency regressor required".into(),
                    ));
                }

                let y_col = &g_formula.dependent;
                let x_col = &g_formula.independents[0];

                let y_vec: Vec<f64> = {
                    let col = df
                        .get_column(y_col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    col.as_float()
                        .ok_or_else(|| {
                            HayashiError::Runtime(format!(
                                "midas: y column '{y_col}' must be numeric"
                            ))
                        })?
                        .to_vec()
                };
                let x_vec: Vec<f64> = {
                    let col = df
                        .get_column(x_col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    col.as_float()
                        .ok_or_else(|| {
                            HayashiError::Runtime(format!(
                                "midas: x column '{x_col}' must be numeric"
                            ))
                        })?
                        .to_vec()
                };

                let result = greeners::Midas::fit(
                    &ndarray::Array1::from_vec(y_vec),
                    &ndarray::Array1::from_vec(x_vec),
                    freq,
                    n_lags,
                    poly,
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── TVP (Time-Varying Parameters) ────────────────────────────────
            // tvp(y ~ x1 + x2, df)
            "tvp" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result = greeners::TVP::fit(&y_arr, &x_arr, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── SETAR (Threshold AR) ────────────────────────────────────────
            // setar(y, df, order=2, delay=1)
            "setar" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let ar_order = match opt_map.get("order") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let delay = match opt_map.get("delay") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                let y_col = &g_formula.dependent;
                let y_vec: Vec<f64> = {
                    let col = df
                        .get_column(y_col)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    col.as_float()
                        .ok_or_else(|| {
                            HayashiError::Runtime(format!(
                                "setar: y column '{y_col}' must be numeric"
                            ))
                        })?
                        .to_vec()
                };

                let result =
                    greeners::SETAR::fit(&ndarray::Array1::from_vec(y_vec), ar_order, delay)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Panel Quantile ──────────────────────────────────────────────
            // panel_qreg(y ~ x1 + x2, df, id="firm", tau=0.5)
            "panel_qreg" | "panel_quantile" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires id=\"column\" option"
                        )))
                    }
                };
                let tau = match opt_map.get("tau") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.5,
                };

                // Extract y and x manually (no intercept — FE absorb it)
                let y_vec: Vec<f64> = {
                    let col = df
                        .get_column(&g_formula.dependent)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    col.as_float()
                        .ok_or_else(|| {
                            HayashiError::Runtime(format!("{func}: y column must be numeric"))
                        })?
                        .to_vec()
                };
                let n = y_vec.len();
                let k = g_formula.independents.len();
                let mut x_arr = ndarray::Array2::<f64>::zeros((n, k));
                for (j, name) in g_formula.independents.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: x column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        x_arr[(i, j)] = vals[i];
                    }
                }
                let var_names = g_formula.independents.clone();

                let entity_ids: Vec<i64> = {
                    let col = df
                        .get_column(id_col.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(i) = col.as_int() {
                        i.to_vec()
                    } else if let Some(f) = col.as_float() {
                        f.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "{func}: id column '{id_col}' must be numeric"
                        )));
                    }
                };

                let result = greeners::PanelQuantile::fit(
                    &ndarray::Array1::from_vec(y_vec),
                    &x_arr,
                    &entity_ids,
                    tau,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── MS-VAR (Markov-Switching VAR) ────────────────────────────────
            // msvar(y1 + y2, df, regimes=2, lags=1)
            "msvar" | "ms_var" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let n_regimes = match opt_map.get("regimes") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                // Build Y matrix from all variables (dependent + independents)
                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        y_mat[(i, j)] = vals[i];
                    }
                }

                let result = greeners::MSVAR::fit(&y_mat, n_regimes, lags, Some(all_cols))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── FAVAR (Factor-Augmented VAR) ────────────────────────────────
            // favar(y1 + y2 + y3, df, observed="rate", factors=2, lags=1, irf=0)
            "favar" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let n_factors = match opt_map.get("factors") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let irf_steps = match opt_map.get("irf") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 0,
                };
                let observed_col = match opt_map.get("observed") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires observed=\"column\" option"
                        )))
                    }
                };

                // X = all formula variables, observed = separate column
                let mut x_cols: Vec<String> = vec![g_formula.dependent.clone()];
                x_cols.extend(g_formula.independents.iter().cloned());
                let n = df.n_rows();
                let n_x = x_cols.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, n_x));
                for (j, name) in x_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        x_mat[(i, j)] = vals[i];
                    }
                }

                let obs_col = df
                    .get_column(&observed_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let obs_vals = obs_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!(
                        "{func}: observed column '{observed_col}' must be numeric"
                    ))
                })?;
                let obs_mat = ndarray::Array2::from_shape_vec((n, 1), obs_vals.to_vec())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::FAVAR::fit(
                    &x_mat,
                    &obs_mat,
                    n_factors,
                    lags,
                    irf_steps,
                    None,
                    Some(vec![observed_col.clone()]),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Spatial Durbin Model (panel) ────────────────────────────────
            // spatial_durbin(y ~ x1 + x2, df, w=W, id="entity")
            "spatial_durbin" | "sdm" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires id=\"column\" option"
                        )))
                    }
                };

                // Extract W matrix
                let w_mat = match opt_map.get("w") {
                    Some(Value::List(rows)) => {
                        let n_rows = rows.len();
                        let mut w = ndarray::Array2::<f64>::zeros((n_rows, n_rows));
                        for (i, row) in rows.iter().enumerate() {
                            match row {
                                Value::List(cols) => {
                                    if cols.len() != n_rows {
                                        return Err(HayashiError::Runtime(format!(
                                            "{func}: W must be square, row {i} has {} cols, expected {n_rows}",
                                            cols.len()
                                        )));
                                    }
                                    for (j, val) in cols.iter().enumerate() {
                                        w[(i, j)] = match val {
                                            Value::Float(f) => *f,
                                            Value::Int(v) => *v as f64,
                                            _ => return Err(HayashiError::Runtime(
                                                format!("{func}: W matrix contains non-numeric values")
                                            )),
                                        };
                                    }
                                }
                                _ => return Err(HayashiError::Runtime(
                                    format!("{func}: W must be a list of lists (matrix)")
                                )),
                            }
                        }
                        w
                    }
                    _ => return Err(HayashiError::Runtime(
                        format!("{func}() requires w=W option with a spatial weights matrix (list of lists)")
                    )),
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let entity_ids: Vec<i64> = {
                    let col = df
                        .get_column(id_col.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(i) = col.as_int() {
                        i.to_vec()
                    } else if let Some(f) = col.as_float() {
                        f.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "{func}: id column '{id_col}' must be numeric"
                        )));
                    }
                };

                let result = greeners::SpatialDurbin::fit(
                    &y_arr,
                    &x_arr,
                    &w_mat,
                    &entity_ids,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Johansen with structural breaks ─────────────────────────────
            // johansen_break(y1 + y2, df, lags=1, breaks=[50])
            "johansen_break" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                // Build Y matrix from all variables
                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        y_mat[(i, j)] = vals[i];
                    }
                }

                // Parse break points from breaks= option (list of ints)
                let break_points: Vec<usize> = match opt_map.get("breaks") {
                    Some(Value::List(items)) => items
                        .iter()
                        .filter_map(|v| match v {
                            Value::Int(i) => Some(*i as usize),
                            Value::Float(f) => Some(*f as usize),
                            _ => None,
                        })
                        .collect(),
                    _ => vec![],
                };

                let result = greeners::JohansenBreak::fit(&y_mat, lags, &break_points)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── TVP-VAR (Time-Varying Parameter VAR) ─────────────────────────
            // tvp_var(y1 ~ y2, df, lags=1)
            "tvp_var" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        y_mat[(i, j)] = vals[i];
                    }
                }

                let result = greeners::TvpVar::fit(&y_mat, lags, Some(all_cols))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Spatial Durbin Error Model (SDEM) ───────────────────────────
            // spatial_durbin_error(y ~ x1 + x2, df, w=W, id="entity")
            "spatial_durbin_error" | "sdem" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires id=\"column\" option"
                        )))
                    }
                };

                let w_mat = match opt_map.get("w") {
                    Some(Value::List(rows)) => {
                        let n_rows = rows.len();
                        let mut w = ndarray::Array2::<f64>::zeros((n_rows, n_rows));
                        for (i, row) in rows.iter().enumerate() {
                            match row {
                                Value::List(cols) => {
                                    if cols.len() != n_rows {
                                        return Err(HayashiError::Runtime(format!(
                                            "{func}: W must be square, row {i} has {} cols, expected {n_rows}",
                                            cols.len()
                                        )));
                                    }
                                    for (j, val) in cols.iter().enumerate() {
                                        w[(i, j)] = match val {
                                            Value::Float(f) => *f,
                                            Value::Int(v) => *v as f64,
                                            _ => return Err(HayashiError::Runtime(
                                                format!("{func}: W matrix contains non-numeric values")
                                            )),
                                        };
                                    }
                                }
                                _ => return Err(HayashiError::Runtime(
                                    format!("{func}: W must be a list of lists (matrix)")
                                )),
                            }
                        }
                        w
                    }
                    _ => return Err(HayashiError::Runtime(
                        format!("{func}() requires w=W option with a spatial weights matrix (list of lists)")
                    )),
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let entity_ids: Vec<i64> = {
                    let col = df
                        .get_column(id_col.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(i) = col.as_int() {
                        i.to_vec()
                    } else if let Some(f) = col.as_float() {
                        f.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "{func}: id column '{id_col}' must be numeric"
                        )));
                    }
                };

                let result = greeners::SpatialDurbinError::fit(
                    &y_arr,
                    &x_arr,
                    &w_mat,
                    &entity_ids,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── FMOLS (Fully Modified OLS) ─────────────────────────────────
            // fmols(y ~ x1 + x2, df)
            "fmols" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result = greeners::FMOLS::fit(&y_arr, &x_arr, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Quantile VAR ───────────────────────────────────────────────
            // qvar(y1 ~ y2, df, lags=1, tau=0.5, boot=100)
            "qvar" | "quantile_var" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let tau = match opt_map.get("tau") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.5,
                };
                let n_boot = match opt_map.get("boot") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 100,
                };

                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        y_mat[(i, j)] = vals[i];
                    }
                }

                let result = greeners::QuantileVAR::fit(&y_mat, lags, tau, n_boot, Some(all_cols))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Panel Smooth Transition Regression (PSTR) ──────────────────
            // pstr(y ~ x1 + x2, df, q="transition_var", id="entity")
            "pstr" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires id=\"column\" option"
                        )))
                    }
                };
                let q_col = match opt_map.get("q") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires q=\"transition_var\" option"
                        )))
                    }
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                // Extract transition variable
                let q_col_data = df
                    .get_column(q_col.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let q_vals = q_col_data.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!(
                        "{func}: transition variable '{q_col}' must be numeric"
                    ))
                })?;
                let q_arr = ndarray::Array1::from_vec(q_vals.to_vec());

                let entity_ids: Vec<i64> = {
                    let col = df
                        .get_column(id_col.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(i) = col.as_int() {
                        i.to_vec()
                    } else if let Some(f) = col.as_float() {
                        f.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "{func}: id column '{id_col}' must be numeric"
                        )));
                    }
                };

                let result =
                    greeners::PSTR::fit(&y_arr, &x_arr, &q_arr, &entity_ids, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── MODWT Wavelet Decomposition ────────────────────────────────
            // modwt(df, var, scales=4)
            "modwt" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("modwt(df, var, scales=4)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "modwt: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "modwt: second argument must be a variable name".into(),
                        ))
                    }
                };
                let scales = match opt_map.get("scales") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 4,
                };

                let col = df
                    .get_column(var_name.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let vals = col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: variable '{var_name}' must be numeric"))
                })?;
                let x_arr = ndarray::Array1::from_vec(vals.to_vec());

                let result = greeners::MODWT::fit(&x_arr, scales)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Copula dependence modeling ─────────────────────────────────
            // copula(y1 + y2, df, type="gaussian")
            "copula" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let copula_type = match opt_map.get("type") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => greeners::CopulaType::Gaussian,
                        "clayton" => greeners::CopulaType::Clayton,
                        "gumbel" => greeners::CopulaType::Gumbel,
                        "frank" => greeners::CopulaType::Frank,
                        _ => {
                            return Err(HayashiError::Runtime(format!(
                                "{func}: type must be gaussian, clayton, gumbel, or frank"
                            )))
                        }
                    },
                    _ => greeners::CopulaType::Gaussian,
                };

                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        x_mat[(i, j)] = vals[i];
                    }
                }

                let result = greeners::Copula::fit(&x_mat, copula_type, Some(all_cols))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── NARDL (Nonlinear ARDL) ─────────────────────────────────────
            // nardl(y ~ x, df, lags=1)
            "nardl" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                if g_formula.independents.len() != 1 {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: requires exactly one regressor (y ~ x)"
                    )));
                }
                let x_name = &g_formula.independents[0];
                let y_col = df
                    .get_column(&g_formula.dependent)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: dependent must be numeric"))
                })?;
                let x_col = df
                    .get_column(x_name)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let x_vals = x_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: regressor '{x_name}' must be numeric"))
                })?;

                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());
                let x_vec = ndarray::Array1::from_vec(x_vals.to_vec());

                let result = greeners::NARDL::fit(&y_arr, &x_vec, lags)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Panel VAR (PVAR) ───────────────────────────────────────────
            // pvar(y1 ~ y2, df, id="entity", lags=1)
            "pvar" | "panel_var" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires id=\"column\" option"
                        )))
                    }
                };

                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        y_mat[(i, j)] = vals[i];
                    }
                }

                let entity_ids: Vec<i64> = {
                    let col = df
                        .get_column(id_col.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(i) = col.as_int() {
                        i.to_vec()
                    } else if let Some(f) = col.as_float() {
                        f.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "{func}: id column '{id_col}' must be numeric"
                        )));
                    }
                };

                let result = greeners::PanelVAR::fit(&y_mat, &entity_ids, lags, Some(all_cols))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Functional coefficient model ───────────────────────────────
            // fcoef(y ~ x1 + x2, df, z="moderator", points=20)
            "fcoef" | "functional_coef" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let z_col = match opt_map.get("z") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires z=\"moderator\" option"
                        )))
                    }
                };
                let n_points = match opt_map.get("points") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let z_col_data = df
                    .get_column(z_col.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let z_vals = z_col_data.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: moderator '{z_col}' must be numeric"))
                })?;
                let z_arr = ndarray::Array1::from_vec(z_vals.to_vec());

                let result = greeners::FunctionalCoef::fit(
                    &y_arr,
                    &x_arr,
                    &z_arr,
                    None,
                    n_points,
                    Some(var_names),
                    Some(z_col),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── DCC-GARCH ──────────────────────────────────────────────────
            // dcc_garch(y1 ~ y2, df)
            "dcc_garch" | "dcc" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut r_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        r_mat[(i, j)] = vals[i];
                    }
                }

                let result = greeners::DCCGARCH::fit(&r_mat, Some(all_cols))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Threshold VAR (TVAR) ───────────────────────────────────────
            // tvar(y1 ~ y2, df, q="threshold_var", lags=1, delay=1)
            "tvar" | "threshold_var" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let max_delay = match opt_map.get("delay") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let q_col = match opt_map.get("q") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires q=\"threshold_var\" option"
                        )))
                    }
                };

                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        y_mat[(i, j)] = vals[i];
                    }
                }

                let q_col_data = df
                    .get_column(q_col.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let q_vals = q_col_data.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!(
                        "{func}: threshold variable '{q_col}' must be numeric"
                    ))
                })?;
                let q_arr = ndarray::Array1::from_vec(q_vals.to_vec());

                let result = greeners::TVAR::fit(&y_mat, &q_arr, lags, max_delay, Some(all_cols))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Bayesian VAR (BVAR) ────────────────────────────────────────
            // bvar(y1 ~ y2, df, lags=1, lambda1=0.1, lambda2=0.2, lambda3=1.0)
            "bvar" | "bayesian_var" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let lambda1 = match opt_map.get("lambda1") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let lambda2 = match opt_map.get("lambda2") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let lambda3 = match opt_map.get("lambda3") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };

                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        y_mat[(i, j)] = vals[i];
                    }
                }

                let result =
                    greeners::BVAR::fit(&y_mat, lags, lambda1, lambda2, lambda3, Some(all_cols))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Mixed-Frequency VAR (MF-VAR) ───────────────────────────────
            // mfvar(df_low, y_low1, y_low2, df_high, y_high1, agg=3, lags=1)
            "mfvar" | "mixed_freq_var" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "mfvar(df_low, y_low1, ..., df_high, y_high1, ..., agg=3, lags=1)".into(),
                    ));
                }
                let df_low_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "mfvar: first arg must be DataFrame".into(),
                        ))
                    }
                };
                let df_low = match self.env.get(&df_low_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_low_name}' is not a DataFrame"))),
                };

                // Find second DataFrame in args
                let mut df_high_name = String::new();
                let mut df_high_idx = 0;
                for (i, a) in args.iter().enumerate().skip(1) {
                    if let Expr::Var(n) = a {
                        if let Some(Value::DataFrame(_)) = self.env.get(n) {
                            if n != &df_low_name {
                                df_high_name = n.clone();
                                df_high_idx = i;
                                break;
                            }
                        }
                    }
                }
                if df_high_name.is_empty() {
                    return Err(HayashiError::Runtime(
                        "mfvar: need second DataFrame for high-freq".into(),
                    ));
                }
                let df_high = match self.env.get(&df_high_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_high_name}' is not a DataFrame"))),
                };

                // Low-freq variables: args[1..df_high_idx]
                let low_vars: Vec<String> = args[1..df_high_idx]
                    .iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type(
                            "mfvar: variables must be identifiers".into(),
                        )),
                    })
                    .collect::<Result<_>>()?;

                // High-freq variables: args[df_high_idx+1..]
                let high_vars: Vec<String> = args[df_high_idx + 1..]
                    .iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type(
                            "mfvar: variables must be identifiers".into(),
                        )),
                    })
                    .collect::<Result<_>>()?;

                let agg_ratio = match opt_map.get("agg") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 3,
                };
                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                // Build matrices
                let n_low = df_low.n_rows();
                let n_high = df_high.n_rows();
                let k_low = low_vars.len();
                let k_high = high_vars.len();

                let mut y_low = ndarray::Array2::<f64>::zeros((n_low, k_low));
                for (j, vname) in low_vars.iter().enumerate() {
                    let col = df_low
                        .get_column(vname)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: '{vname}' must be numeric"))
                    })?;
                    for i in 0..n_low {
                        y_low[(i, j)] = vals[i];
                    }
                }

                let mut y_high = ndarray::Array2::<f64>::zeros((n_high, k_high));
                for (j, vname) in high_vars.iter().enumerate() {
                    let col = df_high
                        .get_column(vname)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: '{vname}' must be numeric"))
                    })?;
                    for i in 0..n_high {
                        y_high[(i, j)] = vals[i];
                    }
                }

                let result = greeners::MFVAR::fit(
                    &y_low,
                    &y_high,
                    agg_ratio,
                    lags,
                    Some(low_vars),
                    Some(high_vars),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Time-varying copula ────────────────────────────────────────
            // tvcopula(y1 ~ y2, df, type="gaussian")
            "tvcopula" | "tv_copula" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let copula_type = match opt_map.get("type") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal" => greeners::TvCopulaType::Gaussian,
                        "clayton" => greeners::TvCopulaType::Clayton,
                        "gumbel" => greeners::TvCopulaType::Gumbel,
                        _ => {
                            return Err(HayashiError::Runtime(format!(
                                "{func}: type must be gaussian, clayton, or gumbel"
                            )))
                        }
                    },
                    _ => greeners::TvCopulaType::Gaussian,
                };

                let mut all_cols: Vec<String> = vec![g_formula.dependent.clone()];
                all_cols.extend(g_formula.independents.iter().cloned());
                let n_vars = all_cols.len();
                let n = df.n_rows();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, n_vars));
                for (j, name) in all_cols.iter().enumerate() {
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: column '{name}' must be numeric"))
                    })?;
                    for i in 0..n {
                        x_mat[(i, j)] = vals[i];
                    }
                }

                let result = greeners::TvCopula::fit(&x_mat, copula_type, Some(all_cols))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Stochastic Volatility (SV) ─────────────────────────────────
            // sv(df, var, iter=100)
            "sv" | "stochastic_vol" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("sv(df, var, iter=100)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("sv: first arg must be DataFrame".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "sv: second arg must be variable name".into(),
                        ))
                    }
                };
                let n_iter = match opt_map.get("iter") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 100,
                };

                let col = df
                    .get_column(var_name.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let vals = col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{var_name}' must be numeric"))
                })?;
                let y_arr = ndarray::Array1::from_vec(vals.to_vec());

                let result = greeners::SV::fit(&y_arr, n_iter, Some(var_name))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Factor-augmented panel ─────────────────────────────────────
            // fapanel(y ~ x1 + x2, df, aux=aux_df, id="entity", period="period", factors=2)
            "fapanel" | "fa_panel" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let aux_name = match opt_map.get("aux") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires aux=\"aux_df\" option"
                        )))
                    }
                };
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires id=\"column\" option"
                        )))
                    }
                };
                let period_col = match opt_map.get("period") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires period=\"column\" option"
                        )))
                    }
                };
                let n_factors = match opt_map.get("factors") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };

                let aux_df = match self.env.get(&aux_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{aux_name}' is not a DataFrame"))),
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                // Build aux matrix (T x n_aux)
                let n_aux_cols = aux_df.n_cols();
                let n_aux_rows = aux_df.n_rows();
                let mut aux_mat = ndarray::Array2::<f64>::zeros((n_aux_rows, n_aux_cols));
                let aux_col_names: Vec<String> = aux_df.column_names();
                for (j, cname) in aux_col_names.iter().enumerate() {
                    if let Ok(col) = aux_df.get_column(cname) {
                        if let Some(vals) = col.as_float() {
                            for i in 0..n_aux_rows {
                                aux_mat[(i, j)] = vals[i];
                            }
                        }
                    }
                }

                // Extract entity and period IDs
                let entity_ids: Vec<i64> = {
                    let col = df
                        .get_column(id_col.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(i) = col.as_int() {
                        i.to_vec()
                    } else if let Some(f) = col.as_float() {
                        f.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!("{func}: id must be numeric")));
                    }
                };
                let period_ids: Vec<i64> = {
                    let col = df
                        .get_column(period_col.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if let Some(i) = col.as_int() {
                        i.to_vec()
                    } else if let Some(f) = col.as_float() {
                        f.iter().map(|v| *v as i64).collect()
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "{func}: period must be numeric"
                        )));
                    }
                };

                let result = greeners::FAPanel::fit(
                    &y_arr,
                    &x_arr,
                    &aux_mat,
                    &entity_ids,
                    &period_ids,
                    n_factors,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Hawkes process ─────────────────────────────────────────────
            // hawkes(df, time_var, T=100)
            "hawkes" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "hawkes(df, time_var [, T=100])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "hawkes: first arg must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "hawkes: second arg must be variable name".into(),
                        ))
                    }
                };
                let t_window = match opt_map.get("T") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };

                let col = df
                    .get_column(var_name.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let vals = col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{var_name}' must be numeric"))
                })?;
                let event_times: Vec<f64> = vals.to_vec();

                let result = greeners::Hawkes::fit(&event_times, t_window)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Random Forest regression ───────────────────────────────────
            // rf(y ~ x1 + x2, df, trees=100, depth=10)
            "rf" | "random_forest" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let n_trees = match opt_map.get("trees") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 100,
                };
                let max_depth = match opt_map.get("depth") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result = greeners::RandomForest::fit(
                    &y_arr,
                    &x_arr,
                    n_trees,
                    max_depth,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Gradient Boosting ──────────────────────────────────────────
            // gbm(y ~ x1 + x2, df, trees=100, lr=0.1, depth=3)
            "gbm" | "gradient_boosting" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let n_trees = match opt_map.get("trees") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 100,
                };
                let lr = match opt_map.get("lr") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let max_depth = match opt_map.get("depth") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let subsample = match opt_map.get("subsample") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result = greeners::GradientBoosting::fit(
                    &y_arr,
                    &x_arr,
                    n_trees,
                    lr,
                    max_depth,
                    subsample,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Neural Network (MLP) ───────────────────────────────────────
            // mlp(y ~ x1 + x2, df, hidden=10, lr=0.01, epochs=200)
            "mlp" | "neural_net" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let n_hidden = match opt_map.get("hidden") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };
                let lr = match opt_map.get("lr") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let n_epochs = match opt_map.get("epochs") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result =
                    greeners::MLP::fit(&y_arr, &x_arr, n_hidden, lr, n_epochs, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Synthetic DiD ──────────────────────────────────────────────
            // synthdid(y, treated, period, df)
            "synthdid" | "synthetic_did" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "synthdid(df, y_var, treated_var, treatment_period)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "synthdid: first arg must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };

                // Parse: df, y_var, treated_var, treatment_period
                // args[0] = df, args[1] = y_var, args[2] = treated_var, args[3] = treatment_period
                let y_var = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "synthdid: y_var must be identifier".into(),
                        ))
                    }
                };
                let treated_var = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "synthdid: treated_var must be identifier".into(),
                        ))
                    }
                };
                let treatment_period = match &args[3] {
                    Expr::Int(v) => *v as usize,
                    Expr::Float(v) => *v as usize,
                    _ => {
                        return Err(HayashiError::Type(
                            "synthdid: treatment_period must be integer".into(),
                        ))
                    }
                };

                // Build outcome matrix (units x periods)
                // Need unit and period columns
                let unit_col = match opt_map.get("unit") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires unit=\"column\" option"
                        )))
                    }
                };
                let period_col = match opt_map.get("period") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires period=\"column\" option"
                        )))
                    }
                };

                // Extract unique units and periods
                let unit_col_data = df
                    .get_column(unit_col.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let period_col_data = df
                    .get_column(period_col.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let units: Vec<i64> = if let Some(i) = unit_col_data.as_int() {
                    i.to_vec()
                } else if let Some(f) = unit_col_data.as_float() {
                    f.iter().map(|v| *v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: unit must be numeric"
                    )));
                };
                let periods: Vec<i64> = if let Some(i) = period_col_data.as_int() {
                    i.to_vec()
                } else if let Some(f) = period_col_data.as_float() {
                    f.iter().map(|v| *v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: period must be numeric"
                    )));
                };

                let y_col = df
                    .get_column(y_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric"))
                })?;

                let treated_col = df
                    .get_column(treated_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let treated_vals: Vec<bool> = if let Some(b) = treated_col.as_bool() {
                    b.to_vec()
                } else if let Some(i) = treated_col.as_int() {
                    i.iter().map(|&v| v != 0).collect()
                } else if let Some(f) = treated_col.as_float() {
                    f.iter().map(|&v| v != 0.0).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: '{treated_var}' must be boolean or numeric"
                    )));
                };

                // Get unique units and periods
                let mut unique_units: Vec<i64> = units
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                unique_units.sort();
                let mut unique_periods: Vec<i64> = periods
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                unique_periods.sort();

                let n_units = unique_units.len();
                let n_periods = unique_periods.len();

                // Build outcome matrix
                let mut y_mat = ndarray::Array2::<f64>::zeros((n_units, n_periods));
                let mut treated_vec = vec![false; n_units];

                let unit_to_idx: std::collections::HashMap<i64, usize> = unique_units
                    .iter()
                    .enumerate()
                    .map(|(i, &u)| (u, i))
                    .collect();
                let period_to_idx: std::collections::HashMap<i64, usize> = unique_periods
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (p, i))
                    .collect();

                for row in 0..df.n_rows() {
                    let ui = unit_to_idx[&units[row]];
                    let pi = period_to_idx[&periods[row]];
                    y_mat[(ui, pi)] = y_vals[row];
                    if treated_vals[row] {
                        treated_vec[ui] = true;
                    }
                }

                let result = greeners::SyntheticDiD::fit(&y_mat, &treated_vec, treatment_period)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── CUPED ──────────────────────────────────────────────────────
            // cuped(y ~ x, df, treated="treated_var")
            "cuped" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let treated_col = match opt_map.get("treated") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires treated=\"column\" option"
                        )))
                    }
                };

                // y = dependent, x = pre-treatment covariate
                let y_col = df
                    .get_column(g_formula.dependent.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!(
                        "{func}: '{}' must be numeric",
                        g_formula.dependent
                    ))
                })?;

                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

                // Pre-treatment covariate: first independent
                let x_var = g_formula
                    .independents
                    .first()
                    .ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: need at least 1 covariate"))
                    })?
                    .clone();
                let x_col = df
                    .get_column(x_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let x_vals = x_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{x_var}' must be numeric"))
                })?;
                let x_arr = ndarray::Array1::from_vec(x_vals.to_vec());

                // Treatment indicator
                let treated_data = df
                    .get_column(treated_col.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let treated_vec: Vec<bool> = if let Some(b) = treated_data.as_bool() {
                    b.to_vec()
                } else if let Some(i) = treated_data.as_int() {
                    i.iter().map(|&v| v != 0).collect()
                } else if let Some(f) = treated_data.as_float() {
                    f.iter().map(|&v| v != 0.0).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: '{treated_col}' must be boolean or numeric"
                    )));
                };

                let result = greeners::CUPED::fit(&y_arr, &x_arr, &treated_vec)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Quantile Regression Forest ─────────────────────────────────
            // qrf(y ~ x1 + x2, df, quantiles="0.1,0.5,0.9", trees=100, depth=10)
            "qrf" | "quantile_forest" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let quantiles_str = match opt_map.get("quantiles") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "0.1,0.5,0.9".to_string(),
                };
                let quantiles: Vec<f64> = quantiles_str
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .filter(|q| *q > 0.0 && *q < 1.0)
                    .collect();
                if quantiles.is_empty() {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: quantiles must be comma-separated values in (0,1)"
                    )));
                }

                let n_trees = match opt_map.get("trees") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 100,
                };
                let max_depth = match opt_map.get("depth") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result = greeners::QRF::fit(
                    &y_arr,
                    &x_arr,
                    quantiles,
                    n_trees,
                    max_depth,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── XGBoost ───────────────────────────────────────────────────
            // xgboost(y ~ x1 + x2, df, trees=100, lr=0.3, depth=6, lambda=1.0, alpha=0.0, gamma=0.0)
            "xgboost" | "xgb" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let n_trees = match opt_map.get("trees") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 100,
                };
                let lr = match opt_map.get("lr") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let max_depth = match opt_map.get("depth") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let lambda = match opt_map.get("lambda") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let gamma = match opt_map.get("gamma") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let subsample = match opt_map.get("subsample") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let colsample = match opt_map.get("colsample") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result = greeners::XGBoost::fit(
                    &y_arr,
                    &x_arr,
                    n_trees,
                    lr,
                    max_depth,
                    lambda,
                    alpha,
                    gamma,
                    subsample,
                    colsample,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Double ML with cross-fitting ──────────────────────────────
            // dml_crossfit(y ~ d, df, x="x1,x2", folds=5)
            "dml_crossfit" | "dml_cf" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                // y = dependent, d = first independent (treatment)
                let y_var = g_formula.dependent.clone();
                let d_var = g_formula
                    .independents
                    .first()
                    .ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: need treatment variable"))
                    })?
                    .clone();

                // Confounders from x option
                let x_str = match opt_map.get("x") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires x=\"x1,x2\" option (confounders)"
                        )))
                    }
                };
                let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
                let n_folds = match opt_map.get("folds") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };

                let n = df.n_rows();
                let y_col = df
                    .get_column(y_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric"))
                })?;
                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

                let d_col = df
                    .get_column(d_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let d_vals = d_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{d_var}' must be numeric"))
                })?;
                let d_arr = ndarray::Array1::from_vec(d_vals.to_vec());

                let k = x_vars.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, xname) in x_vars.iter().enumerate() {
                    let col = df
                        .get_column(xname.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: '{xname}' must be numeric"))
                    })?;
                    for i in 0..n {
                        x_mat[(i, j)] = vals[i];
                    }
                }

                let result = greeners::DMLCrossfit::fit(&y_arr, &d_arr, &x_mat, n_folds)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Bayesian Synthetic Control ────────────────────────────────
            // bsc(y_treated, y_controls_matrix, treatment_period, prior=1.0)
            // bsc(df, y_var, control_vars_str, treatment_period, prior=1.0)
            "bsc" | "bayesian_sc" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "bsc(df, y_var, control_vars, treatment_period [, prior=1.0])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "bsc: first arg must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let y_var = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("bsc: y_var must be identifier".into())),
                };
                let control_vars_str = match &args[2] {
                    Expr::Str(s) => s.clone(),
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "bsc: control_vars must be string".into(),
                        ))
                    }
                };
                let treatment_period = match &args[3] {
                    Expr::Int(v) => *v as usize,
                    Expr::Float(v) => *v as usize,
                    _ => {
                        return Err(HayashiError::Type(
                            "bsc: treatment_period must be integer".into(),
                        ))
                    }
                };
                let prior = match opt_map.get("prior") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };

                let control_vars: Vec<String> = control_vars_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                let n = df.n_rows();
                let n_controls = control_vars.len();

                let y_col = df
                    .get_column(y_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric"))
                })?;
                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

                let mut y_controls = ndarray::Array2::<f64>::zeros((n, n_controls));
                for (j, cname) in control_vars.iter().enumerate() {
                    let col = df
                        .get_column(cname.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: '{cname}' must be numeric"))
                    })?;
                    for i in 0..n {
                        y_controls[(i, j)] = vals[i];
                    }
                }

                let result =
                    greeners::BayesianSC::fit(&y_arr, &y_controls, treatment_period, prior)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── LSTM Time Series ───────────────────────────────────────────
            // lstm(df, var [, hidden=10, seqlen=10, lr=0.01, epochs=100, forecast=5])
            "lstm" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(format!(
                        "{func}() requires (df, var)"
                    )));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(format!(
                            "{func}: first arg must be DataFrame"
                        )))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let y_var = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(format!(
                            "{func}: var must be identifier"
                        )))
                    }
                };

                let n_hidden = match opt_map.get("hidden") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let seq_len = match opt_map.get("seqlen") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let lr = match opt_map.get("lr") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let n_epochs = match opt_map.get("epochs") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let n_forecast = match opt_map.get("forecast") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };

                let y_col = df
                    .get_column(y_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric"))
                })?;
                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

                let result =
                    greeners::LSTM::fit(&y_arr, n_hidden, seq_len, lr, n_epochs, n_forecast)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Causal Forest ──────────────────────────────────────────────
            // causalforest(y ~ treated, df, x="x1,x2", trees=100, depth=5)
            "causalforest" | "causal_forest" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let y_var = g_formula.dependent.clone();
                let t_var = g_formula
                    .independents
                    .first()
                    .ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: need treatment variable"))
                    })?
                    .clone();

                let x_str = match opt_map.get("x") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires x=\"x1,x2\" option (features)"
                        )))
                    }
                };
                let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
                let n_trees = match opt_map.get("trees") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let max_depth = match opt_map.get("depth") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };

                let n = df.n_rows();
                let y_col = df
                    .get_column(y_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric"))
                })?;
                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

                let t_col = df
                    .get_column(t_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let t_vec: Vec<bool> = if let Some(b) = t_col.as_bool() {
                    b.to_vec()
                } else if let Some(i) = t_col.as_int() {
                    i.iter().map(|&v| v != 0).collect()
                } else if let Some(f) = t_col.as_float() {
                    f.iter().map(|&v| v != 0.0).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: '{t_var}' must be boolean or numeric"
                    )));
                };

                let k = x_vars.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, xname) in x_vars.iter().enumerate() {
                    let col = df
                        .get_column(xname.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: '{xname}' must be numeric"))
                    })?;
                    for i in 0..n {
                        x_mat[(i, j)] = vals[i];
                    }
                }

                let result = greeners::CausalForest::fit(
                    &y_arr,
                    &t_vec,
                    &x_mat,
                    n_trees,
                    max_depth,
                    Some(x_vars),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Generalized Random Forest ──────────────────────────────────
            // grf(y ~ treated, df, x="x1,x2", trees=100, depth=5)
            "grf" | "generalized_rf" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let y_var = g_formula.dependent.clone();
                let t_var = g_formula
                    .independents
                    .first()
                    .ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: need treatment variable"))
                    })?
                    .clone();

                let x_str = match opt_map.get("x") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires x=\"x1,x2\" option (features)"
                        )))
                    }
                };
                let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
                let n_trees = match opt_map.get("trees") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let max_depth = match opt_map.get("depth") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };

                let n = df.n_rows();
                let y_col = df
                    .get_column(y_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric"))
                })?;
                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

                let t_col = df
                    .get_column(t_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let t_vec: Vec<bool> = if let Some(b) = t_col.as_bool() {
                    b.to_vec()
                } else if let Some(i) = t_col.as_int() {
                    i.iter().map(|&v| v != 0).collect()
                } else if let Some(f) = t_col.as_float() {
                    f.iter().map(|&v| v != 0.0).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: '{t_var}' must be boolean or numeric"
                    )));
                };

                let k = x_vars.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, xname) in x_vars.iter().enumerate() {
                    let col = df
                        .get_column(xname.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: '{xname}' must be numeric"))
                    })?;
                    for i in 0..n {
                        x_mat[(i, j)] = vals[i];
                    }
                }

                let result =
                    greeners::GRF::fit(&y_arr, &t_vec, &x_mat, n_trees, max_depth, Some(x_vars))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Conformal Prediction ───────────────────────────────────────
            // conformal(y ~ x1 + x2, df, alpha=0.1, calib=0.3)
            "conformal" | "conformal_pred" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let calib_frac = match opt_map.get("calib") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                // Use x_arr itself as test set (in-sample intervals)
                let result = greeners::ConformalPrediction::fit(
                    &y_arr,
                    &x_arr,
                    &x_arr,
                    alpha,
                    calib_frac,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Transformer Time Series ────────────────────────────────────
            // transformer(df, var [, d_model=8, seqlen=10, lr=0.001, epochs=100, forecast=5])
            "transformer" | "transformer_ts" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(format!(
                        "{func}() requires (df, var)"
                    )));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(format!(
                            "{func}: first arg must be DataFrame"
                        )))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let y_var = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(format!(
                            "{func}: var must be identifier"
                        )))
                    }
                };

                let d_model = match opt_map.get("d_model") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let seq_len = match opt_map.get("seqlen") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let lr = match opt_map.get("lr") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let n_epochs = match opt_map.get("epochs") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let n_forecast = match opt_map.get("forecast") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };

                let y_col = df
                    .get_column(y_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric"))
                })?;
                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

                let result =
                    greeners::Transformer::fit(&y_arr, d_model, seq_len, lr, n_epochs, n_forecast)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── DR-Learner ────────────────────────────────────────────────
            // dr_learner(y ~ treated, df, x="x1,x2", folds=3)
            "dr_learner" | "drlearner" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let y_var = g_formula.dependent.clone();
                let t_var = g_formula
                    .independents
                    .first()
                    .ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: need treatment variable"))
                    })?
                    .clone();

                let x_str = match opt_map.get("x") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "{func}() requires x=\"x1,x2\" option (features)"
                        )))
                    }
                };
                let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
                let n_folds = match opt_map.get("folds") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };

                let n = df.n_rows();
                let y_col = df
                    .get_column(y_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let y_vals = y_col.as_float().ok_or_else(|| {
                    HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric"))
                })?;
                let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

                let t_col = df
                    .get_column(t_var.as_str())
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let t_vec: Vec<bool> = if let Some(b) = t_col.as_bool() {
                    b.to_vec()
                } else if let Some(i) = t_col.as_int() {
                    i.iter().map(|&v| v != 0).collect()
                } else if let Some(f) = t_col.as_float() {
                    f.iter().map(|&v| v != 0.0).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "{func}: '{t_var}' must be boolean or numeric"
                    )));
                };

                let k = x_vars.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, xname) in x_vars.iter().enumerate() {
                    let col = df
                        .get_column(xname.as_str())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let vals = col.as_float().ok_or_else(|| {
                        HayashiError::Runtime(format!("{func}: '{xname}' must be numeric"))
                    })?;
                    for i in 0..n {
                        x_mat[(i, j)] = vals[i];
                    }
                }

                let result =
                    greeners::DRLearner::fit(&y_arr, &t_vec, &x_mat, n_folds, Some(x_vars))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── BART ──────────────────────────────────────────────────────
            // bart(y ~ x1 + x2, df, trees=20, depth=3, iter=100, burnin=20)
            "bart" | "bayesian_trees" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let n_trees = match opt_map.get("trees") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let max_depth = match opt_map.get("depth") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let n_iter = match opt_map.get("iter") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let burn_in = match opt_map.get("burnin") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result = greeners::BART::fit(
                    &y_arr,
                    &x_arr,
                    n_trees,
                    max_depth,
                    n_iter,
                    burn_in,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Gaussian Process Regression ───────────────────────────────
            // gp(y ~ x1 + x2, df)
            "gp" | "gaussian_process" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                let (y_arr, x_arr) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = g_formula.independents.clone();

                let result = greeners::GaussianProcess::fit(&y_arr, &x_arr, Some(var_names))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
            }

            // ── Spatial econometrics ────────────────────────────────────────
            // spatial_sar(y ~ x1 + x2, df, w=W_matrix)
            // spatial_sem(y ~ x1 + x2, df, w=W_matrix)
            "spatial_sar" | "spatial_sem" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                // Extract W matrix from w= option (list of lists)
                let w_matrix = match opt_map.get("w") {
                    Some(Value::List(rows)) => {
                        let n_rows = rows.len();
                        let mut w_mat = ndarray::Array2::<f64>::zeros((n_rows, n_rows));
                        for (i, row) in rows.iter().enumerate() {
                            match row {
                                Value::List(cols) => {
                                    if cols.len() != n_rows {
                                        return Err(HayashiError::Runtime(format!(
                                            "{func}: W must be square, row {i} has {} cols, expected {n_rows}",
                                            cols.len()
                                        )));
                                    }
                                    for (j, val) in cols.iter().enumerate() {
                                        w_mat[(i, j)] = match val {
                                            Value::Float(f) => *f,
                                            Value::Int(v) => *v as f64,
                                            _ => return Err(HayashiError::Runtime(
                                                format!("{func}: W matrix contains non-numeric values")
                                            )),
                                        };
                                    }
                                }
                                _ => return Err(HayashiError::Runtime(
                                    format!("{func}: W must be a list of lists (matrix)")
                                )),
                            }
                        }
                        w_mat
                    }
                    _ => return Err(HayashiError::Runtime(
                        format!("{func}() requires w=W option with a spatial weights matrix (list of lists)")
                    )),
                };

                // Extract raw RHS columns (with intercept)
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let var_names = g_formula.independents.clone();

                let result = if func == "spatial_sar" {
                    greeners::Spatial::fit_sar(&y_vec, &x_mat, &w_matrix, Some(var_names))
                } else {
                    greeners::Spatial::fit_sem(&y_vec, &x_mat, &w_matrix, Some(var_names))
                }
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                print!("{result}");
                Ok(Value::Nil)
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                    .filter_map(|t| t.as_var().map(|s| s.to_string()))
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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

            // ── Panel nonlinear: xtlogit, xtprobit, xtpoisson ──────────────
            // Convenience wrappers over gee() with pre-set family/link
            "xtlogit" | "xtprobit" | "xtpoisson" | "xtgee" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(format!(
                            "{}() requires id=group_column option",
                            func
                        )))
                    }
                    _ => return Err(HayashiError::Type("id= must be string".into())),
                };
                let (family, link) = match func {
                    "xtlogit" => (greeners::Family::Binomial, greeners::Link::Logit),
                    "xtprobit" => (greeners::Family::Binomial, greeners::Link::Probit),
                    "xtpoisson" => (greeners::Family::Poisson, greeners::Link::Log),
                    _ => {
                        // xtgee — use family= option
                        let family_str = match opt_map.get("family") {
                            None => "gaussian",
                            Some(Value::Str(s)) => s.as_str(),
                            _ => "gaussian",
                        };
                        match family_str {
                            "binomial" | "logit" => {
                                (greeners::Family::Binomial, greeners::Link::Logit)
                            }
                            "poisson" => (greeners::Family::Poisson, greeners::Link::Log),
                            _ => (greeners::Family::Gaussian, greeners::Link::Identity),
                        }
                    }
                };
                let corr_str = match opt_map.get("corr") {
                    None => "exchangeable",
                    Some(Value::Str(s)) => s.as_str(),
                    _ => "exchangeable",
                };
                let corr = match corr_str {
                    "independence" | "ind" => greeners::CorrStructure::Independence,
                    "exchangeable" | "exch" => greeners::CorrStructure::Exchangeable,
                    "ar1" | "ar(1)" => greeners::CorrStructure::AR1,
                    "unstructured" | "uns" => greeners::CorrStructure::Unstructured,
                    other => {
                        return Err(HayashiError::Runtime(format!(
                            "corr='{other}' unknown — use: independence, exchangeable, ar1, unstructured"
                        )))
                    }
                };
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

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
                    x_random.column_mut(j + 1).assign(&get_col_f64(&df, name)?);
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
