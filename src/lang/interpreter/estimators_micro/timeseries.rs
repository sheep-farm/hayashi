use super::super::*;
use crate::lang::dap::model_expansion;

impl Interpreter {
    pub(super) fn midas(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                    HayashiError::Runtime(format!("midas: y column '{y_col}' must be numeric"))
                })?
                .to_vec()
        };
        let x_vec: Vec<f64> = {
            let col = df
                .get_column(x_col)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            col.as_float()
                .ok_or_else(|| {
                    HayashiError::Runtime(format!("midas: x column '{x_col}' must be numeric"))
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

        let summary = format!(
            "MIDAS(beta={:.4}, R2={:.4}), n={}, lags={}",
            result.beta, result.r_squared, result.n_obs, result.n_lags
        );
        let fields = vec![
            (
                "weights".into(),
                model_expansion::array1_to_series("weights", &result.weights),
            ),
            (
                "gamma".into(),
                model_expansion::array1_to_series("gamma", &result.gamma),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("alpha", Value::Float(result.alpha)),
                    ("beta", Value::Float(result.beta)),
                    ("alpha_se", Value::Float(result.alpha_se)),
                    ("beta_se", Value::Float(result.beta_se)),
                    ("beta_t", Value::Float(result.beta_t)),
                    ("beta_p", Value::Float(result.beta_p)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("adj_r_squared", Value::Float(result.adj_r_squared)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_lags", Value::Int(result.n_lags as i64)),
                    ("freq_ratio", Value::Int(result.freq_ratio as i64)),
                    ("poly_degree", Value::Int(result.poly_degree as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "MidasResult",
            fields,
        ))
    }

    pub(super) fn tvp(
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

        let result = greeners::TVP::fit(&y_arr, &x_arr, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let names = result.variable_names.as_deref().unwrap_or(&[]);
        let summary = format!(
            "TVP(sigma_eps={:.4}, sigma_eta={:.4}), n={}, k={}",
            result.sigma_epsilon,
            result.sigma_eta,
            result.n_obs,
            result.k()
        );
        let fields = vec![
            (
                "beta_smoothed".into(),
                model_expansion::array2_to_dataframe_named(&result.beta_smoothed, names),
            ),
            (
                "beta_se".into(),
                model_expansion::array2_to_dataframe_named(&result.beta_se, names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("sigma_epsilon", Value::Float(result.sigma_epsilon)),
                    ("sigma_eta", Value::Float(result.sigma_eta)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_regressors", Value::Int(result.k() as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "TvpResult",
            fields,
        ))
    }

    pub(super) fn setar(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                    HayashiError::Runtime(format!("setar: y column '{y_col}' must be numeric"))
                })?
                .to_vec()
        };

        let result = greeners::SETAR::fit(&ndarray::Array1::from_vec(y_vec), ar_order, delay)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let names = model_expansion::ar_coef_names(result.ar_order);
        let summary = format!(
            "SETAR(threshold={:.4}, delay={}), n={}, R2={:.4}",
            result.threshold, result.delay, result.n_obs, result.r_squared
        );
        let fields = vec![
            (
                "regime_low".into(),
                model_expansion::coef_dataframe(
                    &names,
                    &result.beta_low,
                    &result.se_low,
                    &result.t_low,
                    &result.p_low,
                    None,
                    None,
                ),
            ),
            (
                "regime_high".into(),
                model_expansion::coef_dataframe(
                    &names,
                    &result.beta_high,
                    &result.se_high,
                    &result.t_high,
                    &result.p_high,
                    None,
                    None,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("threshold", Value::Float(result.threshold)),
                    ("delay", Value::Int(result.delay as i64)),
                    ("ar_order", Value::Int(result.ar_order as i64)),
                    ("n_low", Value::Int(result.n_low as i64)),
                    ("n_high", Value::Int(result.n_high as i64)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("sigma", Value::Float(result.sigma)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "SetarResult",
            fields,
        ))
    }

    pub(super) fn msvar(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let k = result.n_vars;
        let p = result.lags;
        let var_names = &result.var_names;
        let lag_names = model_expansion::var_lag_names(k, p, var_names, false);
        let regime_names = model_expansion::regime_col_names(result.n_regimes);
        let summary = format!(
            "MSVAR(regimes={}, lags={}), n={}, vars={}",
            result.n_regimes, result.lags, result.n_obs, result.n_vars
        );
        let fields = vec![
            (
                "regime_intercepts".into(),
                model_expansion::array2_to_dataframe_named(&result.regime_intercepts, var_names),
            ),
            (
                "ar_coeffs".into(),
                model_expansion::array2_to_dataframe_named(&result.ar_coeffs, &lag_names),
            ),
            (
                "regime_covariances".into(),
                model_expansion::array3_to_list_of_dataframes(
                    &result.regime_covariances,
                    var_names,
                ),
            ),
            (
                "transition_matrix".into(),
                model_expansion::array2_to_dataframe_named(
                    &result.transition_matrix,
                    &regime_names,
                ),
            ),
            (
                "filtered_probs".into(),
                model_expansion::array2_to_dataframe_named(&result.filtered_probs, &regime_names),
            ),
            (
                "smoothed_probs".into(),
                model_expansion::array2_to_dataframe_named(&result.smoothed_probs, &regime_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("n_regimes", Value::Int(result.n_regimes as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "MsVarResult",
            fields,
        ))
    }

    pub(super) fn favar(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let all_names: Vec<String> = result
            .factor_names
            .iter()
            .chain(result.observed_names.iter())
            .cloned()
            .collect();
        let k = all_names.len();
        let var_names = model_expansion::var_param_names(k, result.lags, &all_names);
        let summary = format!(
            "FAVAR(factors={}, observed={}, lags={}), n={}, R2={:.4}",
            result.n_factors,
            result.n_observed,
            result.lags,
            result.n_obs,
            result.total_variance_explained
        );
        let fields = vec![
            (
                "factors".into(),
                model_expansion::array2_to_dataframe_named(&result.factors, &result.factor_names),
            ),
            (
                "loadings".into(),
                model_expansion::array2_to_dataframe_named(&result.loadings, &result.factor_names),
            ),
            (
                "var_coeffs".into(),
                model_expansion::array2_to_dataframe_named(&result.var_coeffs, &var_names),
            ),
            (
                "var_sigma".into(),
                model_expansion::array2_to_dataframe_named(&result.var_sigma, &all_names),
            ),
            (
                "irf".into(),
                if irf_steps > 0 {
                    model_expansion::array3_to_list_of_dataframes(&result.irf, &all_names)
                } else {
                    Value::List(Arc::new(vec![]))
                },
            ),
            (
                "variance_explained".into(),
                model_expansion::array1_to_series("variance_explained", &result.variance_explained),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    (
                        "total_variance_explained",
                        Value::Float(result.total_variance_explained),
                    ),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_series", Value::Int(result.n_series as i64)),
                    ("n_factors", Value::Int(result.n_factors as i64)),
                    ("n_observed", Value::Int(result.n_observed as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "FavarResult",
            fields,
        ))
    }

    pub(super) fn johansen_break(
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

        let cv_names: Vec<String> = (0..result.cointegration_rank)
            .map(|i| format!("CV{}", i + 1))
            .collect();
        let summary = format!(
            "JohansenBreak(rank={}, breaks={}), n={}, vars={}",
            result.cointegration_rank, result.n_breaks, result.n_obs, result.n_vars
        );
        let fields = vec![
            (
                "trace_stats".into(),
                model_expansion::array1_to_series("trace_stats", &result.trace_stats),
            ),
            (
                "lambda_max_stats".into(),
                model_expansion::array1_to_series("lambda_max_stats", &result.lambda_max_stats),
            ),
            (
                "trace_cv_5".into(),
                model_expansion::array1_to_series("trace_cv_5", &result.trace_cv_5),
            ),
            (
                "lambda_max_cv_5".into(),
                model_expansion::array1_to_series("lambda_max_cv_5", &result.lambda_max_cv_5),
            ),
            (
                "eigenvalues".into(),
                model_expansion::array1_to_series("eigenvalues", &result.eigenvalues),
            ),
            (
                "cointegrating_vectors".into(),
                model_expansion::array2_to_dataframe_named(
                    &result.cointegrating_vectors,
                    &cv_names,
                ),
            ),
            (
                "break_points".into(),
                model_expansion::int_series("break_points", &result.break_points),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    (
                        "cointegration_rank",
                        Value::Int(result.cointegration_rank as i64),
                    ),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                    ("n_breaks", Value::Int(result.n_breaks as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "JohansenBreakResult",
            fields,
        ))
    }

    pub(super) fn tvp_var(
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

        let var_names = &result.var_names;
        let summary = format!(
            "TVP-VAR(lags={}), n={}, vars={}",
            result.lags, result.n_obs, result.n_vars
        );
        let fields = vec![
            (
                "beta_smoothed".into(),
                model_expansion::array3_to_list_of_dataframes(&result.beta_smoothed, var_names),
            ),
            (
                "sigma".into(),
                model_expansion::array2_to_dataframe_named(&result.sigma, var_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("q_scale", Value::Float(result.q_scale)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                    ("n_regressors", Value::Int(result.n_regressors as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "TvpVarResult",
            fields,
        ))
    }

    pub(super) fn qvar(
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

        let var_names = &result.var_names;
        let param_names = model_expansion::var_param_names(result.n_vars, result.lags, var_names);
        let summary = format!(
            "QVAR(tau={:.2}, lags={}), n={}, vars={}",
            result.tau, result.lags, result.n_obs, result.n_vars
        );
        let fields = vec![
            (
                "coeffs".into(),
                model_expansion::array2_to_dataframe_named(&result.coeffs, &param_names),
            ),
            (
                "std_errors".into(),
                model_expansion::array2_to_dataframe_named(&result.std_errors, &param_names),
            ),
            (
                "t_values".into(),
                model_expansion::array2_to_dataframe_named(&result.t_values, &param_names),
            ),
            (
                "p_values".into(),
                model_expansion::array2_to_dataframe_named(&result.p_values, &param_names),
            ),
            (
                "pseudo_r2".into(),
                model_expansion::array1_to_series("pseudo_r2", &result.pseudo_r2),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("tau", Value::Float(result.tau)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "QuantileVarResult",
            fields,
        ))
    }

    pub(super) fn modwt(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let summary = format!(
            "MODWT({}), n={}, scales={}",
            result.wavelet, result.n_obs, result.n_scales
        );
        let fields = vec![
            (
                "wavelet_coeffs".into(),
                model_expansion::vec_array1_to_series_list(&result.wavelet_coeffs, "wavelet"),
            ),
            (
                "details".into(),
                model_expansion::vec_array1_to_series_list(&result.details, "detail"),
            ),
            (
                "scaling_coeffs".into(),
                model_expansion::array1_to_series("scaling_coeffs", &result.scaling_coeffs),
            ),
            (
                "smooth".into(),
                model_expansion::array1_to_series("smooth", &result.smooth),
            ),
            (
                "energy".into(),
                model_expansion::series_from_vec("energy", &result.energy),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("wavelet", Value::Str(result.wavelet.clone())),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_scales", Value::Int(result.n_scales as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "ModwtResult",
            fields,
        ))
    }

    pub(super) fn copula(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let var_names = &result.var_names;
        let summary = format!(
            "Copula({}), theta={:.4}, n={}, vars={}",
            result.copula_type, result.theta, result.n_obs, result.n_vars
        );
        let fields = vec![
            ("theta".into(), Value::Float(result.theta)),
            (
                "corr_matrix".into(),
                model_expansion::array2_to_dataframe_named(&result.corr_matrix, var_names),
            ),
            (
                "kendall_tau".into(),
                model_expansion::array2_to_dataframe_named(&result.kendall_tau, var_names),
            ),
            (
                "spearman_rho".into(),
                model_expansion::array2_to_dataframe_named(&result.spearman_rho, var_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("copula_type", Value::Str(result.copula_type.to_string())),
                    ("theta", Value::Float(result.theta)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "CopulaResult",
            fields,
        ))
    }

    pub(super) fn nardl(
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

        if g_formula.independents.len() != 1 {
            return Err(HayashiError::Runtime(format!(
                "{func}: requires exactly one regressor (y ~ x)"
            )));
        }
        let x_name = &g_formula.independents[0];
        let y_col = df
            .get_column(&g_formula.dependent)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: dependent must be numeric")))?;
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

        let summary = format!(
            "NARDL(beta_pos={:.4}, beta_neg={:.4}), n={}, lags={}",
            result.beta_pos, result.beta_neg, result.n_obs, result.lags
        );
        let fields = vec![
            (
                "coefficients".into(),
                model_expansion::coef_dataframe(
                    &result.coef_names,
                    &result.coefficients,
                    &result.std_errors,
                    &result.t_values,
                    &result.p_values,
                    None,
                    None,
                ),
            ),
            (
                "theta_pos".into(),
                model_expansion::array1_to_series("theta_pos", &result.theta_pos),
            ),
            (
                "theta_neg".into(),
                model_expansion::array1_to_series("theta_neg", &result.theta_neg),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("beta_pos", Value::Float(result.beta_pos)),
                    ("beta_neg", Value::Float(result.beta_neg)),
                    ("rho", Value::Float(result.rho)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("lr_asym_f", Value::Float(result.lr_asym_f)),
                    ("lr_asym_p", Value::Float(result.lr_asym_p)),
                    ("sr_asym_f", Value::Float(result.sr_asym_f)),
                    ("sr_asym_p", Value::Float(result.sr_asym_p)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "NardlResult",
            fields,
        ))
    }

    pub(super) fn dcc_garch(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            let vals = col.to_float();
            for i in 0..n {
                r_mat[(i, j)] = vals[i];
            }
        }

        let result = greeners::DCCGARCH::fit(&r_mat, Some(all_cols))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let var_names = &result.var_names;
        let garch_col_names: Vec<String> = vec!["omega".into(), "alpha".into(), "beta".into()];
        let summary = format!(
            "DCC-GARCH(alpha={:.4}, beta={:.4}), n={}, series={}",
            result.dcc_alpha, result.dcc_beta, result.n_obs, result.n_series
        );
        let fields = vec![
            (
                "garch_params".into(),
                model_expansion::array2_to_dataframe_named(&result.garch_params, &garch_col_names),
            ),
            (
                "conditional_vols".into(),
                model_expansion::array2_to_dataframe_named(&result.conditional_vols, var_names),
            ),
            (
                "dcc_correlations".into(),
                model_expansion::array3_to_list_of_dataframes(&result.dcc_correlations, var_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("dcc_alpha", Value::Float(result.dcc_alpha)),
                    ("dcc_beta", Value::Float(result.dcc_beta)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_series", Value::Int(result.n_series as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "DccGarchResult",
            fields,
        ))
    }

    pub(super) fn tvar(
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

        let var_names = &result.var_names;
        let lag_names =
            model_expansion::var_lag_names(result.n_vars, result.lags, var_names, false);
        let summary = format!(
            "TVAR(threshold={:.4}, delay={}), vars={}",
            result.threshold, result.delay, result.n_vars
        );
        let fields = vec![
            (
                "coeffs_low".into(),
                model_expansion::array2_to_dataframe_named(&result.coeffs_low, &lag_names),
            ),
            (
                "coeffs_high".into(),
                model_expansion::array2_to_dataframe_named(&result.coeffs_high, &lag_names),
            ),
            (
                "se_low".into(),
                model_expansion::array2_to_dataframe_named(&result.se_low, &lag_names),
            ),
            (
                "se_high".into(),
                model_expansion::array2_to_dataframe_named(&result.se_high, &lag_names),
            ),
            (
                "t_low".into(),
                model_expansion::array2_to_dataframe_named(&result.t_low, &lag_names),
            ),
            (
                "t_high".into(),
                model_expansion::array2_to_dataframe_named(&result.t_high, &lag_names),
            ),
            (
                "p_low".into(),
                model_expansion::array2_to_dataframe_named(&result.p_low, &lag_names),
            ),
            (
                "p_high".into(),
                model_expansion::array2_to_dataframe_named(&result.p_high, &lag_names),
            ),
            (
                "cov_low".into(),
                model_expansion::array2_to_dataframe_named(&result.cov_low, var_names),
            ),
            (
                "cov_high".into(),
                model_expansion::array2_to_dataframe_named(&result.cov_high, var_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("threshold", Value::Float(result.threshold)),
                    ("delay", Value::Int(result.delay as i64)),
                    ("n_low", Value::Int(result.n_low as i64)),
                    ("n_high", Value::Int(result.n_high as i64)),
                    ("rss", Value::Float(result.rss)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                    ("threshold_var", Value::Str(result.threshold_var.clone())),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "TvarResult",
            fields,
        ))
    }

    pub(super) fn bvar(
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

        let result = greeners::BVAR::fit(&y_mat, lags, lambda1, lambda2, lambda3, Some(all_cols))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let var_names = &result.var_names;
        let param_names = model_expansion::var_param_names(result.n_vars, result.lags, var_names);
        let summary = format!(
            "BVAR(lags={}), n={}, vars={}",
            result.lags, result.n_obs, result.n_vars
        );
        let fields = vec![
            (
                "coeffs".into(),
                model_expansion::array2_to_dataframe_named(&result.coeffs, &param_names),
            ),
            (
                "std_errors".into(),
                model_expansion::array2_to_dataframe_named(&result.std_errors, &param_names),
            ),
            (
                "t_values".into(),
                model_expansion::array2_to_dataframe_named(&result.t_values, &param_names),
            ),
            (
                "p_values".into(),
                model_expansion::array2_to_dataframe_named(&result.p_values, &param_names),
            ),
            (
                "resid_cov".into(),
                model_expansion::array2_to_dataframe_named(&result.resid_cov, var_names),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("lambda1", Value::Float(result.hyperparams[0])),
                    ("lambda2", Value::Float(result.hyperparams[1])),
                    ("lambda3", Value::Float(result.hyperparams[2])),
                    ("mu", Value::Float(result.hyperparams[3])),
                    ("log_marginal", Value::Float(result.log_marginal)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("lags", Value::Int(result.lags as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "BvarResult",
            fields,
        ))
    }

    pub(super) fn tvcopula(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let summary = format!(
            "TvCopula({:?}), n={}, vars={}",
            result.copula_type, result.n_obs, result.n_vars
        );
        let fields = vec![
            (
                "theta_path".into(),
                model_expansion::array1_to_series("theta_path", &result.theta_path),
            ),
            (
                "kendall_tau_path".into(),
                model_expansion::array1_to_series("kendall_tau_path", &result.kendall_tau_path),
            ),
            (
                "dynamics_params".into(),
                model_expansion::array1_to_series("dynamics_params", &result.dynamics_params),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    (
                        "copula_type",
                        Value::Str(format!("{:?}", result.copula_type)),
                    ),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("mean_theta", Value::Float(result.mean_theta)),
                    ("std_theta", Value::Float(result.std_theta)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "TvCopulaResult",
            fields,
        ))
    }

    pub(super) fn sv(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let summary = format!(
            "SV(mu={:.4}, phi={:.4}, sigma_eta={:.4}), n={}",
            result.mu, result.phi, result.sigma_eta, result.n_obs
        );
        let fields = vec![
            (
                "log_vol".into(),
                model_expansion::array1_to_series("log_vol", &result.log_vol),
            ),
            (
                "cond_vol".into(),
                model_expansion::array1_to_series("cond_vol", &result.cond_vol),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("mu", Value::Float(result.mu)),
                    ("phi", Value::Float(result.phi)),
                    ("sigma_eta", Value::Float(result.sigma_eta)),
                    ("mu_se", Value::Float(result.mu_se)),
                    ("phi_se", Value::Float(result.phi_se)),
                    ("sigma_eta_se", Value::Float(result.sigma_eta_se)),
                    ("phi_t", Value::Float(result.phi_t)),
                    ("phi_p", Value::Float(result.phi_p)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_iter", Value::Int(result.n_iter as i64)),
                    ("var_name", Value::Str(result.var_name.clone())),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "SvResult",
            fields,
        ))
    }

    pub(super) fn hawkes(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let summary = format!(
            "Hawkes(mu={:.4}, alpha={:.4}, beta={:.4}), n={}, eta={:.4}",
            result.mu, result.alpha, result.beta, result.n_events, result.branching_ratio
        );
        let fields = vec![
            (
                "intensity_at_events".into(),
                model_expansion::array1_to_series(
                    "intensity_at_events",
                    &result.intensity_at_events,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("mu", Value::Float(result.mu)),
                    ("alpha", Value::Float(result.alpha)),
                    ("beta", Value::Float(result.beta)),
                    ("branching_ratio", Value::Float(result.branching_ratio)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("aic", Value::Float(result.aic)),
                    ("bic", Value::Float(result.bic)),
                    ("n_events", Value::Int(result.n_events as i64)),
                    ("time_window", Value::Float(result.time_window)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "HawkesResult",
            fields,
        ))
    }
}
