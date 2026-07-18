use super::super::helpers::*;
use super::super::*;

impl Interpreter {
    pub(super) fn weak_iv(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            _ => return Err(self.rt_err(format!("weak_iv: '{df_name}' is not a DataFrame"))),
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
                "weak_iv: no endogenous variable identified (vars in endog but not in instr)"
                    .into(),
            ));
        }
        if z_excl_names.is_empty() {
            return Err(HayashiError::Runtime(
                "weak_iv: no excluded instrument identified (vars in instr but not in endog)"
                    .into(),
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
        let proj_exog =
            |a: &Array2<f64>| -> Array2<f64> { x_exog.dot(&xtx_exog_inv.dot(&x_exog.t().dot(a))) };
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
        let mut first_stage_f = Vec::new();
        let mut first_stage_p = Vec::new();
        let mut first_stage_names = Vec::new();
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
            first_stage_f.push(f_j);
            first_stage_p.push(p_j);
            first_stage_names.push(x_endog_names[j].clone());
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
        let mut fields = HashMap::new();
        fields.insert("test".into(), Value::Str("Weak Instrument Test".into()));
        fields.insert("n".into(), Value::Int(n as i64));
        fields.insert("k_endog".into(), Value::Int(k_endog as i64));
        fields.insert("n_instruments".into(), Value::Int(l as i64));
        fields.insert("df_first_stage".into(), Value::Int(df_fs as i64));
        fields.insert("cragg_donald_f".into(), Value::Float(cd_stat));
        fields.insert(
            "first_stage_names".into(),
            Value::List(Arc::new(
                first_stage_names.into_iter().map(Value::Str).collect(),
            )),
        );
        fields.insert(
            "first_stage_f".into(),
            Value::List(Arc::new(
                first_stage_f.into_iter().map(Value::Float).collect(),
            )),
        );
        fields.insert(
            "first_stage_p".into(),
            Value::List(Arc::new(
                first_stage_p.into_iter().map(Value::Float).collect(),
            )),
        );
        Ok(diag_with(out, fields))
    }

    pub(super) fn estat_overid(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let mut map = HashMap::new();
        map.insert(
            "test".into(),
            Value::Str("Sargan / Hansen J Overidentification Test".into()),
        );
        map.insert("j_stat".into(), Value::Float(result.sargan_stat));
        map.insert("df".into(), Value::Int(result.df as i64));
        map.insert("p_value".into(), Value::Float(result.p_value));
        map.insert(
            "n_instruments".into(),
            Value::Int(result.n_instruments as i64),
        );
        map.insert(
            "n_regressors".into(),
            Value::Int(result.n_regressors as i64),
        );
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn estat_endog(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let mut map = HashMap::new();
        map.insert(
            "test".into(),
            Value::Str("Durbin-Wu-Hausman Endogeneity Test".into()),
        );
        map.insert("f_stat".into(), Value::Float(result.f_stat));
        map.insert("df".into(), Value::Int(result.df as i64));
        map.insert("p_value".into(), Value::Float(result.p_value));
        map.insert(
            "endogenous_vars".into(),
            Value::List(Arc::new(
                result.endogenous_vars.into_iter().map(Value::Str).collect(),
            )),
        );
        let conclusion = if result.p_value < 0.05 {
            "reject H0 -> endogeneity present, IV/2SLS preferred"
        } else {
            "do not reject H0 -> OLS consistent"
        };
        map.insert("conclusion".into(), Value::Str(conclusion.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn estat_classification(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(self.rt_err("estat_classification(model) requires a logit/probit model"));
        }
        let v = self.eval_expr(&args[0])?;
        let model = match &v {
            Value::BinaryResult(m) => m.clone(),
            _ => {
                return Err(
                    self.rt_err("estat_classification: argument must be a logit/probit model")
                )
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
        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("Classification Table".into()));
        map.insert("threshold".into(), Value::Float(result.threshold));
        map.insert("tp".into(), Value::Int(result.tp as i64));
        map.insert("tn".into(), Value::Int(result.tn as i64));
        map.insert("fp".into(), Value::Int(result.fp as i64));
        map.insert("fn".into(), Value::Int(result.fn_count as i64));
        map.insert("sensitivity".into(), Value::Float(result.sensitivity));
        map.insert("specificity".into(), Value::Float(result.specificity));
        map.insert("correct_rate".into(), Value::Float(result.correct_rate));
        map.insert("n".into(), Value::Int(result.n as i64));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn lroc(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("ROC / AUC".into()));
        map.insert("auc".into(), Value::Float(result.auc));
        map.insert("gini".into(), Value::Float(result.gini));
        map.insert(
            "n_thresholds".into(),
            Value::Int(result.n_thresholds as i64),
        );
        map.insert(
            "fpr".into(),
            Value::List(Arc::new(result.fpr.into_iter().map(Value::Float).collect())),
        );
        map.insert(
            "tpr".into(),
            Value::List(Arc::new(result.tpr.into_iter().map(Value::Float).collect())),
        );
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn estat_gof(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(self.rt_err("estat_gof(model) requires a logit/probit model"));
        }
        let v = self.eval_expr(&args[0])?;
        let model = match &v {
            Value::BinaryResult(m) => m.clone(),
            _ => return Err(self.rt_err("estat_gof: argument must be a logit/probit model")),
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
        let mut map = HashMap::new();
        map.insert(
            "test".into(),
            Value::Str("Hosmer-Lemeshow Goodness-of-Fit".into()),
        );
        map.insert("hl_stat".into(), Value::Float(result.hl_stat));
        map.insert("p_value".into(), Value::Float(result.p_value));
        map.insert("n_groups".into(), Value::Int(result.n_groups as i64));
        map.insert("df".into(), Value::Int(result.df as i64));
        let conclusion = if result.p_value < 0.05 {
            "reject H0 -> model does not fit adequately"
        } else {
            "do not reject H0 -> model fits adequately"
        };
        map.insert("conclusion".into(), Value::Str(conclusion.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn linktest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let mut map = HashMap::new();
        map.insert(
            "test".into(),
            Value::Str("Linktest (Specification Test)".into()),
        );
        map.insert("n".into(), Value::Int(result.n as i64));
        map.insert("hat_coef".into(), Value::Float(result.hat_coef));
        map.insert("hat_se".into(), Value::Float(result.hat_se));
        map.insert("hat_z".into(), Value::Float(result.hat_z));
        map.insert("hat_p".into(), Value::Float(result.hat_p));
        map.insert("hatsq_coef".into(), Value::Float(result.hatsq_coef));
        map.insert("hatsq_se".into(), Value::Float(result.hatsq_se));
        map.insert("hatsq_z".into(), Value::Float(result.hatsq_z));
        map.insert("hatsq_p".into(), Value::Float(result.hatsq_p));
        let conclusion = if result.hatsq_p < 0.05 {
            "reject H0 -> possible specification error (hatsq significant)"
        } else {
            "do not reject H0 -> model appears correctly specified"
        };
        map.insert("conclusion".into(), Value::Str(conclusion.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn influence(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                "influence(): only supported for OLS/WLS models — use: influence(m_ols, df)".into(),
            )),
        }
    }
}
