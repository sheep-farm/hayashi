use super::helpers::*;
use super::*;

mod timeseries_models;

/// margins, VECM/VAR/IRF/FEVD, ARIMA/SARIMA/AutoReg/ARDL/Kalman/forecast,
/// lincom/nlcom. Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_post_estimation_ts(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── margins ──────────────────────────────────────────────────────
            "margins" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "margins() requires an estimated model as an argument".into(),
                    ));
                }
                let model = self.eval_expr(&args[0])?;

                // dydx=[X1, X2] — which variables to show (lazy, column names)
                let dydx_filter: Option<Vec<String>> =
                    opts.iter()
                        .find(|o| o.name == "dydx")
                        .map(|o| match &o.value {
                            Expr::List(items) => items
                                .iter()
                                .filter_map(|e| match e {
                                    Expr::Var(n) | Expr::Str(n) => Some(n.clone()),
                                    _ => None,
                                })
                                .collect(),
                            Expr::Var(n) | Expr::Str(n) => vec![n.clone()],
                            _ => vec![],
                        });
                let show_var = |name: &str| -> bool {
                    match &dydx_filter {
                        None => name != "_cons" && name != "const",
                        Some(list) => list.iter().any(|s| s == name),
                    }
                };

                // at_X=value — fixes variable X at the given value for margins calculation
                let at_vals: HashMap<String, f64> = opt_map
                    .iter()
                    .filter_map(|(k, v)| {
                        let var = k.strip_prefix("at_")?.to_string();
                        match v {
                            Value::Float(f) => Some((var, *f)),
                            Value::Int(i) => Some((var, *i as f64)),
                            _ => None,
                        }
                    })
                    .collect();

                let sep = "─".repeat(60);
                let sep2 = "═".repeat(60);

                match model {
                    // ── Logit / Probit ────────────────────────────────────────
                    Value::BinaryResult(bm) => {
                        let mut x_use = bm.x.clone();
                        for (var, val) in &at_vals {
                            if let Some(idx) = bm.coef_names.iter().position(|n| n == var) {
                                x_use = greeners::Margins::with_at(&x_use, idx, *val);
                            }
                        }
                        let vcov = Self::binary_mle_vcov(&bm.kind, &bm.result.params, &bm.y, &bm.x);
                        let mut ame_result = if bm.kind == "logit" {
                            match &vcov {
                                Some(v) => greeners::Margins::ame_logit_with_vcov(
                                    &bm.result.params,
                                    &x_use,
                                    &bm.coef_names,
                                    v,
                                ),
                                None => greeners::Margins::ame_logit(
                                    &bm.result.params,
                                    &x_use,
                                    &bm.coef_names,
                                ),
                            }
                        } else {
                            match &vcov {
                                Some(v) => greeners::Margins::ame_probit_with_vcov(
                                    &bm.result.params,
                                    &x_use,
                                    &bm.coef_names,
                                    v,
                                ),
                                None => greeners::Margins::ame_probit(
                                    &bm.result.params,
                                    &x_use,
                                    &bm.coef_names,
                                ),
                            }
                        };
                        if let Ok(normal_dist) = Normal::new(0.0, 1.0) {
                            for i in 0..ame_result.effects.len() {
                                let se = ame_result.std_errors[i];
                                if se.is_finite() && se > 1e-15 {
                                    let z = ame_result.effects[i] / se;
                                    ame_result.z_values[i] = z;
                                    ame_result.p_values[i] = 2.0 * (1.0 - normal_dist.cdf(z.abs()));
                                }
                            }
                        }
                        let at_label = if at_vals.is_empty() {
                            String::new()
                        } else {
                            format!(
                                "  at({})",
                                at_vals
                                    .iter()
                                    .map(|(k, v)| format!("{k}={v}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        };
                        let has_se = ame_result.std_errors.iter().any(|s| s.is_finite());
                        println!("\n{sep2}");
                        println!(
                            " Average Marginal Effects — {}{at_label}",
                            bm.kind.to_uppercase()
                        );
                        println!("{sep2}");
                        if has_se {
                            println!(
                                "{:<18} {:>10} {:>10} {:>8} {:>8}",
                                "Variable", "dy/dx", "Std.Err.", "z", "P>|z|"
                            );
                        } else {
                            println!("{:<22} {:>14}", "Variable", "dy/dx");
                        }
                        println!("{sep}");
                        for (i, name) in ame_result.variable_names.iter().enumerate() {
                            if !show_var(name) {
                                continue;
                            }
                            if has_se {
                                let sig = if ame_result.p_values[i] < 0.01 {
                                    "***"
                                } else if ame_result.p_values[i] < 0.05 {
                                    "**"
                                } else if ame_result.p_values[i] < 0.10 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!(
                                    "{:<18} {:>10.6} {:>10.6} {:>8.3} {:>8.4} {sig}",
                                    name,
                                    ame_result.effects[i],
                                    ame_result.std_errors[i],
                                    ame_result.z_values[i],
                                    ame_result.p_values[i]
                                );
                            } else {
                                println!("{:<22} {:>14.6}", name, ame_result.effects[i]);
                            }
                        }
                        println!("{sep}");
                        println!("n = {}", ame_result.n_obs);
                        println!("{sep2}\n");
                    }

                    // ── Poisson / NegBin ──────────────────────────────────────
                    Value::PoissonResult(r) => {
                        let x = r.x_data();
                        let fb: Vec<String> =
                            (0..r.params.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        let mut x_use = x.to_owned();
                        for (var, val) in &at_vals {
                            if let Some(idx) = names.iter().position(|n| n == var) {
                                x_use = greeners::Margins::with_at(&x_use, idx, *val);
                            }
                        }
                        let ame_result =
                            greeners::Margins::ame_exponential(&r.params, &x_use, names);
                        let at_label = if at_vals.is_empty() {
                            String::new()
                        } else {
                            format!(
                                "  at({})",
                                at_vals
                                    .iter()
                                    .map(|(k, v)| format!("{k}={v}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        };
                        println!("\n{sep2}");
                        println!(" Average Marginal Effects — POISSON{at_label}  (dy/dx = β·μ̄)");
                        println!("{sep2}");
                        println!("{:<22} {:>14}", "Variable", "dy/dx");
                        println!("{sep}");
                        for (k_idx, name) in ame_result.variable_names.iter().enumerate() {
                            if !show_var(name) {
                                continue;
                            }
                            let ame = ame_result.effects[k_idx];
                            println!("{:<22} {:>14.6}", name, ame);
                        }
                        println!("{sep}");
                        println!("n = {}", ame_result.n_obs);
                        println!("{sep2}\n");
                    }
                    Value::NegBinResult(r) => {
                        let x = r.x_data();
                        let fb: Vec<String> =
                            (0..r.params.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        let mut x_use = x.to_owned();
                        for (var, val) in &at_vals {
                            if let Some(idx) = names.iter().position(|n| n == var) {
                                x_use = greeners::Margins::with_at(&x_use, idx, *val);
                            }
                        }
                        let ame_result =
                            greeners::Margins::ame_exponential(&r.params, &x_use, names);
                        let at_label = if at_vals.is_empty() {
                            String::new()
                        } else {
                            format!(
                                "  at({})",
                                at_vals
                                    .iter()
                                    .map(|(k, v)| format!("{k}={v}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        };
                        println!("\n{sep2}");
                        println!(
                            " Average Marginal Effects — NEG. BINOMIAL{at_label}  (dy/dx = β·μ̄)"
                        );
                        println!("{sep2}");
                        println!("{:<22} {:>14}", "Variable", "dy/dx");
                        println!("{sep}");
                        for (k_idx, name) in ame_result.variable_names.iter().enumerate() {
                            if !show_var(name) {
                                continue;
                            }
                            let ame = ame_result.effects[k_idx];
                            println!("{:<22} {:>14.6}", name, ame);
                        }
                        println!("{sep}");
                        println!("n = {}   α = {:.4}", ame_result.n_obs, r.alpha);
                        println!("{sep2}\n");
                    }

                    // ── Ordered Logit / Probit ────────────────────────────────
                    // AME_k(Y=j) = (1/n) Σ_i [f(κ_{j-1} - X_iβ) - f(κ_j - X_iβ)] * β_k
                    // (com κ_0 = -∞ → f(κ_0 - ·) = 0;  κ_J = +∞ → f(κ_J - ·) = 0)
                    Value::OrderedResult(r) => {
                        let x = r.x_data();
                        let n = x.nrows();
                        let beta = &r.params;
                        let cuts = &r.thresholds;
                        let j = r.n_categories;
                        let is_logit = r.model_name.to_lowercase().contains("logit");
                        let link_pdf = |u: f64| -> f64 {
                            if is_logit {
                                let p = logistic(u);
                                p * (1.0 - p)
                            } else {
                                norm_pdf(u)
                            }
                        };
                        let fb: Vec<String> = (0..beta.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        // Xβ for each observation
                        let xb: Vec<f64> = (0..n).map(|i| x.row(i).dot(beta)).collect();
                        // AME[var_k, cat_j]
                        let k = beta.len();
                        println!("\n{sep2}");
                        println!(
                            " Average Marginal Effects — {}",
                            r.model_name.to_uppercase()
                        );
                        println!(" dP(Y=j)/dx — um painel por categoria");
                        println!("{sep2}");
                        // header
                        print!("{:<22}", "Variable");
                        for cat_j in 0..j {
                            print!("  {:>10}", format!("P(Y={})", cat_j + 1));
                        }
                        println!();
                        println!("{sep}");
                        for k_idx in 0..k {
                            let name = names.get(k_idx).map(String::as_str).unwrap_or("?");
                            if name == "_cons" || name == "const" {
                                continue;
                            }
                            print!("{:<22}", name);
                            for cat_j in 0..j {
                                // f(κ_{j-1} - Xβ) — zero para cat_j=0 (sem threshold inferior)
                                let f_lo: f64 = if cat_j == 0 {
                                    0.0
                                } else {
                                    (0..n)
                                        .map(|i| link_pdf(cuts[cat_j - 1] - xb[i]))
                                        .sum::<f64>()
                                        / n as f64
                                };
                                // f(κ_j - Xβ) — zero para cat_j=J-1 (sem threshold superior)
                                let f_hi: f64 = if cat_j == j - 1 {
                                    0.0
                                } else {
                                    (0..n).map(|i| link_pdf(cuts[cat_j] - xb[i])).sum::<f64>()
                                        / n as f64
                                };
                                let ame = (f_lo - f_hi) * beta[k_idx];
                                print!("  {:>10.5}", ame);
                            }
                            println!();
                        }
                        println!("{sep}");
                        println!("n = {n}   Categorias: {j}   Modelo: {}", r.model_name);
                        println!("{sep2}\n");
                    }

                    _ => {
                        return Err(HayashiError::Type(
                            "margins() suporta: logit, probit, poisson, negbin, ologit, oprobit"
                                .into(),
                        ))
                    }
                }
                Ok(Value::Nil)
            }

            // ── vecm ─────────────────────────────────────────────────────────
            "vecm" => self.eval_vecm(args, opt_map),

            // ── var ──────────────────────────────────────────────────────────
            "var" => self.eval_var(args, opt_map),

            // ── irf ──────────────────────────────────────────────────────────
            "irf" => self.eval_irf(args, opt_map),

            // ── fevd ─────────────────────────────────────────────────────────
            "fevd" => self.eval_fevd(args, opt_map),

            // ── arima / sarima ───────────────────────────────────────────────
            "arima" | "sarima" => self.eval_arima(func, args, opt_map),

            // ── autoreg ──────────────────────────────────────────────────────
            // autoreg(df, y, lags=p, trend="c")
            "autoreg" | "ar" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "autoreg(df, var, lags=p, trend=\"c\"|\"ct\"|\"t\"|\"n\")".into(),
                    ));
                }

                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "autoreg: primeiro argumento deve ser um DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df, opts)?;

                let col_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "autoreg: second argument must be variable name".into(),
                        ))
                    }
                };

                let y = ndarray::Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?);

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                let trend = match opt_map.get("trend") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "c".to_string(),
                };

                let result = greeners::AutoReg::fit(&y, lags, None, &trend)
                    .map_err(|e| self.rt_err(format!("autoreg: {e}")))?;

                Ok(Value::AutoRegResult(Rc::new(result)))
            }

            // ── ardl ─────────────────────────────────────────────────────────
            // ardl(y ~ x1 + x2, df, lags=p, xlags=q)
            "ardl" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ardl(y ~ x1 + x2, df, lags=p, xlags=q)".into(),
                    ));
                }

                let formula_ast = self.resolve_formula(&args[0])?;

                let df_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "ardl: segundo argumento deve ser um DataFrame".into(),
                        ))
                    }
                };
                let df_raw = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df_raw, opts)?;

                let y_lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                let x_lags = match opt_map.get("xlags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                // to_design_matrix retorna (y, x_com_constante)
                let (y_vec, x_with_const) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // ARDL::fit adds its own constant; remove intercept column
                let x_no_const = if x_with_const.ncols() > 1 {
                    x_with_const.slice(ndarray::s![.., 1..]).to_owned()
                } else {
                    return Err(HayashiError::Runtime(
                        "ardl: formula must have at least one regressor besides intercept".into(),
                    ));
                };

                let y_arr = ndarray::Array1::from_vec(y_vec.to_vec());

                let result = greeners::ARDL::fit(&y_arr, &x_no_const, y_lags, x_lags)
                    .map_err(|e| self.rt_err(format!("ardl: {e}")))?;

                Ok(Value::ArdlResult(Rc::new(result)))
            }

            // ── kalman ───────────────────────────────────────────────────────
            // kalman(df, var, model="ll"|"llt", sigma_obs=s, sigma_state=s)
            //
            // Predefined models (State Space Form):
            //   "ll"  — Local Level:        y_t = mu_t + e_t
            //                               mu_t = mu_{t-1} + eta_t
            //   "llt" — Local Linear Trend: y_t = mu_t + e_t
            //                               mu_t = mu_{t-1} + nu_{t-1} + eta_t
            //                               nu_t = nu_{t-1} + zeta_t
            //
            // Adiciona colunas {var}_filtered e {var}_smoothed ao DataFrame.
            "kalman" | "kfilter" | "ssm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "kalman(df, var, model=\"ll\"|\"llt\", sigma_obs=s, sigma_state=s)".into(),
                    ));
                }

                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kalman: primeiro argumento deve ser um DataFrame".into(),
                        ))
                    }
                };
                let mut df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };

                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kalman: second argument must be variable name".into(),
                        ))
                    }
                };

                let model_kind = match opt_map.get("model") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "ll".to_string(),
                };

                let y_vec: Vec<f64> = get_col_f64(&df, &var_name)?.to_vec();
                let n = y_vec.len();
                if n < 4 {
                    return Err(HayashiError::Runtime(
                        "kalman: series too short (minimum 4 observations)".into(),
                    ));
                }

                // Estimate sigma_obs from diff(y) if not provided
                let diff_var: f64 = {
                    let diffs: Vec<f64> = y_vec.windows(2).map(|w| w[1] - w[0]).collect();
                    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
                    diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64
                };
                let sigma_obs_default = (diff_var / 2.0).sqrt().max(1e-6);

                let sigma_obs = match opt_map.get("sigma_obs") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => sigma_obs_default,
                };
                let sigma_state = match opt_map.get("sigma_state") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => sigma_obs * 0.1,
                };
                let sigma_slope = match opt_map.get("sigma_slope") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => sigma_state * 0.1,
                };

                // Observations as Vec<Array1<f64>> (scalar-wrapped)
                let obs: Vec<ndarray::Array1<f64>> = y_vec
                    .iter()
                    .map(|&v| ndarray::Array1::from_vec(vec![v]))
                    .collect();

                let ss_result = match model_kind.as_str() {
                    "ll" | "local_level" => {
                        // H=[[1]], F=[[1]], R=[[1]], Q=[[sigma_state^2]], R_obs=[[sigma_obs^2]]
                        let model = greeners::StateSpaceModel {
                            h: ndarray::Array2::from_elem((1, 1), 1.0),
                            f: ndarray::Array2::from_elem((1, 1), 1.0),
                            r: ndarray::Array2::from_elem((1, 1), 1.0),
                            q: ndarray::Array2::from_elem((1, 1), sigma_state.powi(2)),
                            r_obs: ndarray::Array2::from_elem((1, 1), sigma_obs.powi(2)),
                            s0: ndarray::Array1::from_vec(vec![y_vec[0]]),
                            p0: ndarray::Array2::from_elem((1, 1), sigma_obs.powi(2) * 10.0),
                        };
                        greeners::state_space_estimate(&model, &obs)
                            .map_err(|e| self.rt_err(format!("kalman (ll): {e}")))?
                    }
                    "llt" | "local_linear_trend" => {
                        // States: [level, slope]
                        // H = [[1, 0]]
                        // F = [[1, 1], [0, 1]]
                        // R = I_2, Q = diag(sigma_state^2, sigma_slope^2)
                        let h = ndarray::array![[1.0_f64, 0.0]];
                        let f = ndarray::array![[1.0_f64, 1.0], [0.0, 1.0]];
                        let r = ndarray::Array2::<f64>::eye(2);
                        let mut q = ndarray::Array2::<f64>::zeros((2, 2));
                        q[[0, 0]] = sigma_state.powi(2);
                        q[[1, 1]] = sigma_slope.powi(2);
                        let r_obs = ndarray::Array2::from_elem((1, 1), sigma_obs.powi(2));
                        let init_slope = if n > 1 { y_vec[1] - y_vec[0] } else { 0.0 };
                        let model = greeners::StateSpaceModel {
                            h,
                            f,
                            r,
                            q,
                            r_obs,
                            s0: ndarray::Array1::from_vec(vec![y_vec[0], init_slope]),
                            p0: {
                                let mut p = ndarray::Array2::<f64>::zeros((2, 2));
                                p[[0, 0]] = sigma_obs.powi(2) * 10.0;
                                p[[1, 1]] = sigma_slope.powi(2) * 10.0;
                                p
                            },
                        };
                        greeners::state_space_estimate(&model, &obs)
                            .map_err(|e| self.rt_err(format!("kalman (llt): {e}")))?
                    }
                    other => {
                        return Err(HayashiError::Runtime(format!(
                            "kalman: modelo '{other}' desconhecido — use \"ll\" ou \"llt\""
                        )))
                    }
                };

                // Extract filtered and smoothed level (state 0 in both models)
                let filtered: ndarray::Array1<f64> = ndarray::Array1::from_vec(
                    ss_result.filtered_states.iter().map(|s| s[0]).collect(),
                );
                let smoothed: ndarray::Array1<f64> = ndarray::Array1::from_vec(
                    ss_result.smoothed_states.iter().map(|s| s[0]).collect(),
                );

                let filt_name = format!("{var_name}_filtered");
                let smooth_name = format!("{var_name}_smoothed");

                Rc::make_mut(&mut df)
                    .insert(filt_name.clone(), filtered)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Rc::make_mut(&mut df)
                    .insert(smooth_name.clone(), smoothed)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // For LLT, also add trend (slope = state 1)
                if matches!(model_kind.as_str(), "llt" | "local_linear_trend") {
                    let slope_filt: ndarray::Array1<f64> = ndarray::Array1::from_vec(
                        ss_result.filtered_states.iter().map(|s| s[1]).collect(),
                    );
                    let slope_smooth: ndarray::Array1<f64> = ndarray::Array1::from_vec(
                        ss_result.smoothed_states.iter().map(|s| s[1]).collect(),
                    );
                    let sf_name = format!("{var_name}_slope_filtered");
                    let ss_name = format!("{var_name}_slope_smoothed");
                    Rc::make_mut(&mut df)
                        .insert(sf_name.clone(), slope_filt)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    Rc::make_mut(&mut df)
                        .insert(ss_name.clone(), slope_smooth)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    println!(
                        "\nKalman ({}):  T={}  loglik={:.4}  σ_obs={:.4}  σ_state={:.4}  σ_slope={:.4}",
                        model_kind, n, ss_result.log_likelihood, sigma_obs, sigma_state, sigma_slope
                    );
                    println!(
                        "  → {filt_name}, {smooth_name}, {sf_name}, {ss_name} adicionadas a {df_name}"
                    );
                } else {
                    println!(
                        "\nKalman ({}):  T={}  loglik={:.4}  σ_obs={:.4}  σ_state={:.4}",
                        model_kind, n, ss_result.log_likelihood, sigma_obs, sigma_state
                    );
                    println!("  → {filt_name}, {smooth_name} adicionadas a {df_name}");
                }

                self.env.set(&df_name, Value::DataFrame(df))?;
                Ok(Value::Nil)
            }

            // ── forecast ─────────────────────────────────────────────────────
            // forecast(model, steps=8)
            // forecast(model, steps=8, alpha=0.05)
            "forecast" | "fcast" | "predict_h" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "forecast() requires an ARIMA model".into(),
                    ));
                }

                let model = match self.eval_expr(&args[0])? {
                    Value::ArimaResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "forecast() requires an ARIMA model".into(),
                        ))
                    }
                };

                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 8,
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.05,
                };

                let (fc, lo, hi) = model
                    .predict_with_ci(steps, None, alpha)
                    .map_err(|e| self.rt_err(format!("forecast: {e}")))?;

                let sep = "─".repeat(52);
                println!(
                    "\nForecast — {} steps ahead  (CI {}%)",
                    steps,
                    ((1.0 - alpha) * 100.0) as usize
                );
                println!("{sep}");
                println!(
                    "{:<6} {:>12} {:>12} {:>12}",
                    "h", "forecast", "lower", "upper"
                );
                println!("{sep}");
                for h in 0..steps {
                    println!(
                        "{:<6} {:>12.4} {:>12.4} {:>12.4}",
                        h + 1,
                        fc[h],
                        lo[h],
                        hi[h]
                    );
                }
                println!("{sep}");
                println!();

                Ok(Value::Nil)
            }

            // ── lincom ───────────────────────────────────────────────────────
            // lincom(model, var1=mult1, var2=mult2, ...)
            // Delegates algebra to Greeners via OlsResult::t_test(r, q, x)
            // ── nlcom: non-linear combination of coefs (delta method) ────────
            // nlcom(model, expr) — expr uses coefficient names as variables
            // Examples: nlcom(m, X1 / X2)   nlcom(m, exp(_cons))   nlcom(m, X1 * X2)
            "nlcom" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("nlcom(model, expression)".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("nlcom() requires an OLS model".into())),
                };
                let names =
                    ols.result.variable_names.as_ref().ok_or_else(|| {
                        HayashiError::Runtime("model has no variable names".into())
                    })?;
                let params = &ols.result.params;
                let k = params.len();
                let expr = &args[1];

                // save existing variables and bind coefficients
                let mut saved: Vec<(String, Option<Value>)> = Vec::new();
                for (i, name) in names.iter().enumerate() {
                    saved.push((name.clone(), self.env.get(name).cloned()));
                    self.env.set(name, Value::Float(params[i]))?;
                }

                // avaliar g(β̂)
                let g = match self.eval_expr(expr)? {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    _ => {
                        for (name, old) in &saved {
                            match old {
                                Some(v) => {
                                    self.env.set(name, v.clone())?;
                                }
                                None => {
                                    self.env.remove(name);
                                }
                            }
                        }
                        return Err(HayashiError::Type(
                            "nlcom: expression must evaluate to a number".into(),
                        ));
                    }
                };

                // numerical gradient (central differences)
                let h = 1e-7;
                let mut grad = ndarray::Array1::<f64>::zeros(k);
                for j in 0..k {
                    let orig = params[j];
                    self.env.set(&names[j], Value::Float(orig + h))?;
                    let g_plus = match self.eval_expr(expr)? {
                        Value::Float(f) => f,
                        Value::Int(i) => i as f64,
                        _ => g,
                    };
                    self.env.set(&names[j], Value::Float(orig - h))?;
                    let g_minus = match self.eval_expr(expr)? {
                        Value::Float(f) => f,
                        Value::Int(i) => i as f64,
                        _ => g,
                    };
                    grad[j] = (g_plus - g_minus) / (2.0 * h);
                    self.env.set(&names[j], Value::Float(orig))?;
                }

                // restore variables
                for (name, old) in &saved {
                    match old {
                        Some(v) => {
                            self.env.set(name, v.clone())?;
                        }
                        None => {
                            self.env.remove(name);
                        }
                    }
                }

                // V = σ²(X'X)⁻¹
                let xt_x = ols.x.t().dot(&ols.x);
                let xt_x_inv = xt_x.inv().map_err(|e| self.rt_err(format!("nlcom: {e}")))?;
                let sigma2 = ols.result.sigma * ols.result.sigma;
                let vcov = &xt_x_inv * sigma2;

                // SE = sqrt(g' V g)
                let se = (grad.dot(&vcov.dot(&grad))).max(0.0).sqrt();
                let t = if se > 1e-15 { g / se } else { f64::NAN };
                let p = t_pvalue_two(t, ols.result.df_resid as f64);

                println!("\n{:=^60}", " nlcom ");
                println!("  g(β̂) = {g:.6}");
                println!("  SE    = {se:.6}   (delta method)");
                println!("  t     = {t:.4}   p = {p:.4}");
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                if !sig.is_empty() {
                    println!("  {sig}");
                }
                println!("{:=^60}\n", "");
                Ok(Value::Float(g))
            }

            "lincom" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "lincom() requires an OLS model".into(),
                    ));
                }

                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "lincom() only supports OLS models".into(),
                        ))
                    }
                };

                // nomes dos coeficientes via API do Greeners (sem parse de CSV)
                let var_names: Vec<String> =
                    ols.result.variable_names.clone().ok_or_else(|| {
                        HayashiError::Runtime("modelo sem variable_names — use from_formula".into())
                    })?;

                let k = var_names.len();

                // monta vetor de contraste c alinhado com var_names
                // aceita "const" (Greeners) e "_cons" (Stata-compat) como aliases
                let mut c = Array1::<f64>::zeros(k);
                let mut found = false;
                for (idx, greeners_name) in var_names.iter().enumerate() {
                    let lookup = if greeners_name == "const" {
                        "_cons"
                    } else {
                        greeners_name.as_str()
                    };
                    let val = opt_map
                        .get(lookup)
                        .or_else(|| opt_map.get(greeners_name.as_str()));
                    if let Some(v) = val {
                        let mult = match v {
                            Value::Float(f) => *f,
                            Value::Int(i) => *i as f64,
                            _ => {
                                return Err(HayashiError::Type(format!(
                                    "{greeners_name}= must be numeric"
                                )))
                            }
                        };
                        c[idx] = mult;
                        found = true;
                    }
                }

                if !found {
                    let available: Vec<&str> = var_names
                        .iter()
                        .map(|n| if n == "const" { "_cons" } else { n.as_str() })
                        .collect();
                    return Err(HayashiError::Runtime(format!(
                        "no coefficients found — available: {}",
                        available.join(", ")
                    )));
                }

                // estimativa pontual c'β
                let estimate = c.dot(&ols.result.params);

                // inference delegated to Greeners: t_test uses (X'X)⁻¹σ² internally
                let (t, p) = ols
                    .result
                    .t_test(&c, 0.0, &ols.x)
                    .map_err(|e| self.rt_err(format!("lincom: {e}")))?;

                let se = if t.abs() > 1e-15 { estimate / t } else { 0.0 };
                let df_t = ols.result.df_resid as f64;
                let tc = t_critical_95(df_t);

                // readable label for the combination
                let display_name = |n: &str| {
                    if n == "const" {
                        "_cons".to_string()
                    } else {
                        n.to_string()
                    }
                };
                let expr_label: String = var_names
                    .iter()
                    .zip(c.iter())
                    .filter(|(_, &m)| m != 0.0)
                    .enumerate()
                    .map(|(i, (name, &mult))| {
                        let dname = display_name(name);
                        let term = if mult == 1.0 {
                            dname
                        } else if mult == -1.0 {
                            format!("-{dname}")
                        } else {
                            format!("{mult}*{dname}")
                        };
                        if i == 0 {
                            term
                        } else if mult < 0.0 {
                            format!(" - {}", &term[1..])
                        } else {
                            format!(" + {term}")
                        }
                    })
                    .collect();

                let sep = "─".repeat(64);
                println!("\nlincom: {expr_label}");
                println!("{sep}");
                println!(
                    "{:<12} {:>10} {:>10} {:>8} {:>10}",
                    "Estimate", "Std.Err.", "t", "df", "p"
                );
                println!("{sep}");
                println!(
                    "{:<12.6} {:>10.6} {:>10.4} {:>8.1} {:>10.4}",
                    estimate, se, t, df_t, p
                );
                println!("{sep}");
                println!(
                    "95% CI: [{:.6},  {:.6}]",
                    estimate - tc * se,
                    estimate + tc * se
                );
                println!();

                Ok(Value::Nil)
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
