use super::super::*;
use crate::lang::dap::model_expansion;

impl Interpreter {
    pub(super) fn panel_tobit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let result = greeners::PanelTobit::fit(&y_vec, &x_mat, &panel_ids, censor, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let names = result.variable_names.clone().unwrap_or_default();
        let summary = format!(
            "PanelTobit(k={}, n={}, panels={}), logLik={:.4}",
            result.beta.len(),
            result.n_obs,
            result.n_panels,
            result.log_likelihood
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "coefficients".into(),
                model_expansion::coef_dataframe(
                    &names,
                    &result.beta,
                    &result.std_errors,
                    &result.t_values,
                    &result.p_values,
                    None,
                    None,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_panels", Value::Int(result.n_panels as i64)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("sigma_alpha", Value::Float(result.sigma_alpha)),
                    ("sigma_epsilon", Value::Float(result.sigma_epsilon)),
                    ("rho", Value::Float(result.rho)),
                    ("censor_left", Value::Float(result.censor_left)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "PanelTobitResult",
            fields,
        ))
    }

    pub(super) fn panel_heckman(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let sel_coef_names: Vec<String> = std::iter::once("const".to_string())
            .chain(result.sel_names.clone().unwrap_or_default())
            .collect();
        let out_coef_names: Vec<String> = std::iter::once("const".to_string())
            .chain(result.out_names.clone().unwrap_or_default())
            .collect();
        let summary = format!(
            "PanelHeckman(k_out={}, k_sel={}, n={}), rho={:.4}",
            result.beta.len(),
            result.gamma.len(),
            result.n_obs,
            result.rho
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "selection_coefficients".into(),
                model_expansion::coef_dataframe(
                    &sel_coef_names,
                    &result.gamma,
                    &result.gamma_se,
                    &result.gamma_t,
                    &result.gamma_p,
                    None,
                    None,
                ),
            ),
            (
                "outcome_coefficients".into(),
                model_expansion::coef_dataframe(
                    &out_coef_names,
                    &result.beta,
                    &result.beta_se,
                    &result.beta_t,
                    &result.beta_p,
                    None,
                    None,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_selected", Value::Int(result.n_selected as i64)),
                    ("n_panels", Value::Int(result.n_panels as i64)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("rho", Value::Float(result.rho)),
                    ("sigma", Value::Float(result.sigma)),
                    ("sigma_alpha", Value::Float(result.sigma_alpha)),
                    ("sigma_nu", Value::Float(result.sigma_nu)),
                    ("imr_mean", Value::Float(result.imr_mean)),
                    ("imr_coef", Value::Float(result.imr_coef)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "PanelHeckmanResult",
            fields,
        ))
    }

    pub(super) fn panel_qreg(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                .ok_or_else(|| HayashiError::Runtime(format!("{func}: y column must be numeric")))?
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

        let names = result.variable_names.clone().unwrap_or_default();
        let summary = format!(
            "PanelQuantile(tau={:.2}, k={}, n={}), pseudoR2={:.4}",
            result.tau,
            result.beta.len(),
            result.n_obs,
            result.pseudo_r2
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "coefficients".into(),
                model_expansion::coef_dataframe(
                    &names,
                    &result.beta,
                    &result.std_errors,
                    &result.t_values,
                    &result.p_values,
                    None,
                    None,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_entities", Value::Int(result.n_entities as i64)),
                    ("tau", Value::Float(result.tau)),
                    ("pseudo_r2", Value::Float(result.pseudo_r2)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "PanelQuantileResult",
            fields,
        ))
    }

    pub(super) fn fmols(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_arr, x_arr) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = g_formula.independents.clone();

        let result = greeners::FMOLS::fit(&y_arr, &x_arr, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "FMOLS(k={}, n={}), R2={:.4}",
            result.n_regressors, result.n_obs, result.r_squared
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "coefficients".into(),
                model_expansion::coef_dataframe(
                    &result.variable_names,
                    &result.beta,
                    &result.beta_se,
                    &result.beta_t,
                    &result.beta_p,
                    None,
                    None,
                ),
            ),
            (
                "omega".into(),
                model_expansion::array2_to_dataframe("omega", &result.omega),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_regressors", Value::Int(result.n_regressors as i64)),
                    ("bandwidth", Value::Int(result.bandwidth as i64)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("alpha", Value::Float(result.alpha)),
                    ("alpha_se", Value::Float(result.alpha_se)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "FmolsResult",
            fields,
        ))
    }

    pub(super) fn pstr(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let result = greeners::PSTR::fit(&y_arr, &x_arr, &q_arr, &entity_ids, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "PSTR(k={}, n={}), gamma={:.4}, c={:.4}",
            result.n_regressors, result.n_obs, result.gamma, result.c
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "regime0".into(),
                model_expansion::coef_dataframe(
                    &result.variable_names,
                    &result.beta0,
                    &result.beta0_se,
                    &result.beta0_t,
                    &result.beta0_p,
                    None,
                    None,
                ),
            ),
            (
                "regime1".into(),
                model_expansion::coef_dataframe(
                    &result.variable_names,
                    &result.beta1,
                    &result.beta1_se,
                    &result.beta1_t,
                    &result.beta1_p,
                    None,
                    None,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_entities", Value::Int(result.n_entities as i64)),
                    ("n_regressors", Value::Int(result.n_regressors as i64)),
                    ("gamma", Value::Float(result.gamma)),
                    ("c", Value::Float(result.c)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("transition_var", Value::Str(result.transition_var.clone())),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "PstrResult",
            fields,
        ))
    }

    pub(super) fn pvar(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let lag_names =
            model_expansion::var_lag_names(result.n_vars, result.lags, &result.var_names, false);
        let summary = format!(
            "PanelVAR(lags={}, k={}, n={}), J={:.4}",
            result.lags, result.n_vars, result.n_obs, result.j_stat
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "coefficients".into(),
                model_expansion::array2_to_dataframe_named(&result.coeffs, &lag_names),
            ),
            (
                "std_errors".into(),
                model_expansion::array2_to_dataframe_named(&result.std_errors, &lag_names),
            ),
            (
                "t_values".into(),
                model_expansion::array2_to_dataframe_named(&result.t_values, &lag_names),
            ),
            (
                "p_values".into(),
                model_expansion::array2_to_dataframe_named(&result.p_values, &lag_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_entities", Value::Int(result.n_entities as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                    ("n_instruments", Value::Int(result.n_instruments as i64)),
                    ("j_stat", Value::Float(result.j_stat)),
                    ("j_p", Value::Float(result.j_p)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "PanelVarResult",
            fields,
        ))
    }

    pub(super) fn fcoef(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let mut coef_col_names = vec!["const".to_string()];
        coef_col_names.extend(result.variable_names.iter().cloned());
        let summary = format!(
            "FunctionalCoef(n_points={}, k={}, n={}), R2={:.4}",
            result.n_points, result.n_regressors, result.n_obs, result.r_squared
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "coefficients".into(),
                model_expansion::array2_to_dataframe_named(&result.coefficients, &coef_col_names),
            ),
            (
                "std_errors".into(),
                model_expansion::array2_to_dataframe_named(&result.std_errors, &coef_col_names),
            ),
            (
                "z_points".into(),
                model_expansion::array1_to_series("z_points", &result.z_points),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_points", Value::Int(result.n_points as i64)),
                    ("n_regressors", Value::Int(result.n_regressors as i64)),
                    ("bandwidth", Value::Float(result.bandwidth)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("kernel", Value::Str(format!("{:?}", result.kernel))),
                    ("moderator_name", Value::Str(result.moderator_name.clone())),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "FunctionalCoefResult",
            fields,
        ))
    }

    pub(super) fn mfvar(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let lag_names =
            model_expansion::var_lag_names(result.n_vars, result.lags, &result.var_names, false);
        let summary = format!(
            "MfVar(k={}, n={}, lags={}), AIC={:.4}",
            result.n_vars, result.n_obs, result.lags, result.aic
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "coefficients".into(),
                model_expansion::array2_to_dataframe_named(&result.coeffs, &lag_names),
            ),
            (
                "std_errors".into(),
                model_expansion::array2_to_dataframe_named(&result.std_errors, &lag_names),
            ),
            (
                "t_values".into(),
                model_expansion::array2_to_dataframe_named(&result.t_values, &lag_names),
            ),
            (
                "p_values".into(),
                model_expansion::array2_to_dataframe_named(&result.p_values, &lag_names),
            ),
            (
                "midas_weights".into(),
                model_expansion::array1_to_series("midas_weights", &result.midas_weights),
            ),
            (
                "midas_theta".into(),
                model_expansion::array1_to_series("midas_theta", &result.midas_theta),
            ),
            (
                "aggregated".into(),
                model_expansion::array2_to_dataframe("aggregated", &result.aggregated),
            ),
            (
                "resid_cov".into(),
                model_expansion::array2_to_dataframe_named(&result.resid_cov, &result.var_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                    ("agg_ratio", Value::Int(result.agg_ratio as i64)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "MfVarResult",
            fields,
        ))
    }

    pub(super) fn fapanel(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let summary = format!(
            "FaPanel(k={}, factors={}, n={}), R2={:.4}",
            result.n_regressors, result.n_factors, result.n_obs, result.r_squared
        );
        let fields: Vec<(String, Value)> = vec![
            (
                "regressor_coefficients".into(),
                model_expansion::coef_dataframe(
                    &result.regressor_names,
                    &result.beta,
                    &result.beta_se,
                    &result.beta_t,
                    &result.beta_p,
                    None,
                    None,
                ),
            ),
            (
                "factor_coefficients".into(),
                model_expansion::coef_dataframe(
                    &result.factor_names,
                    &result.gamma,
                    &result.gamma_se,
                    &result.gamma_t,
                    &result.gamma_p,
                    None,
                    None,
                ),
            ),
            (
                "factors".into(),
                model_expansion::array2_to_dataframe_named(&result.factors, &result.factor_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_entities", Value::Int(result.n_entities as i64)),
                    ("n_regressors", Value::Int(result.n_regressors as i64)),
                    ("n_factors", Value::Int(result.n_factors as i64)),
                    ("r_squared", Value::Float(result.r_squared)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "FaPanelResult",
            fields,
        ))
    }
}
