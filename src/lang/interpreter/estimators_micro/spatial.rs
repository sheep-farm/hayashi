use super::super::*;
use crate::lang::dap::model_expansion;

impl Interpreter {
    pub(super) fn spatial_panel_sar(
        &mut self,
        func: &str,
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
                                    _ => {
                                        return Err(HayashiError::Runtime(format!(
                                            "{func}: W matrix contains non-numeric values"
                                        )))
                                    }
                                };
                            }
                        }
                        _ => {
                            return Err(HayashiError::Runtime(format!(
                                "{func}: W must be a list of lists (matrix)"
                            )))
                        }
                    }
                }
                w
            }
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires w=W option with a spatial weights matrix (list of lists)"
                )))
            }
        };

        let var_names = g_formula.independents.clone();
        let result = if func == "spatial_panel_sar" {
            greeners::SpatialPanel::fit_sar(&y_vec, &x_mat, &w_mat, &entity_ids, Some(var_names))
        } else {
            greeners::SpatialPanel::fit_sem(&y_vec, &x_mat, &w_mat, &entity_ids, Some(var_names))
        }
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let result_names = result.variable_names.clone().unwrap_or_default();
        let summary = format!(
            "SpatialPanel{}(k={}, n={}, N={}), R2={:.4}",
            if result.model_type == "sar" {
                "SAR"
            } else {
                "SEM"
            },
            result.beta.len(),
            result.n_obs,
            result.n_entities,
            result.r_squared
        );
        let fields = vec![
            (
                "coefficients".into(),
                model_expansion::coef_dataframe(
                    &result_names,
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
                    ("model_type", Value::Str(result.model_type.clone())),
                    ("spatial_param", Value::Float(result.spatial_param)),
                    ("spatial_se", Value::Float(result.spatial_se)),
                    ("spatial_t", Value::Float(result.spatial_t)),
                    ("spatial_p", Value::Float(result.spatial_p)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("sigma", Value::Float(result.sigma)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_entities", Value::Int(result.n_entities as i64)),
                    ("n_periods", Value::Int(result.n_periods as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "SpatialPanelResult",
            fields,
        ))
    }

    pub(super) fn spatial_durbin(
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
                                    _ => {
                                        return Err(HayashiError::Runtime(format!(
                                            "{func}: W matrix contains non-numeric values"
                                        )))
                                    }
                                };
                            }
                        }
                        _ => {
                            return Err(HayashiError::Runtime(format!(
                                "{func}: W must be a list of lists (matrix)"
                            )))
                        }
                    }
                }
                w
            }
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires w=W option with a spatial weights matrix (list of lists)"
                )))
            }
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

        let result =
            greeners::SpatialDurbin::fit(&y_arr, &x_arr, &w_mat, &entity_ids, Some(var_names))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let result_names = result.variable_names.clone();
        let summary = format!(
            "SpatialDurbin(k={}, n={}, N={}), rho={:.4}",
            result.beta.len(),
            result.n_obs,
            result.n_entities,
            result.rho
        );
        let fields = vec![
            (
                "direct_effects".into(),
                model_expansion::coef_dataframe(
                    &result_names,
                    &result.beta,
                    &result.beta_se,
                    &result.beta_t,
                    &result.beta_p,
                    None,
                    None,
                ),
            ),
            (
                "indirect_effects".into(),
                model_expansion::coef_dataframe(
                    &result_names,
                    &result.theta,
                    &result.theta_se,
                    &result.theta_t,
                    &result.theta_p,
                    None,
                    None,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("rho", Value::Float(result.rho)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_entities", Value::Int(result.n_entities as i64)),
                    ("n_regressors", Value::Int(result.n_regressors as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "SpatialDurbinResult",
            fields,
        ))
    }

    pub(super) fn spatial_durbin_error(
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
                                    _ => {
                                        return Err(HayashiError::Runtime(format!(
                                            "{func}: W matrix contains non-numeric values"
                                        )))
                                    }
                                };
                            }
                        }
                        _ => {
                            return Err(HayashiError::Runtime(format!(
                                "{func}: W must be a list of lists (matrix)"
                            )))
                        }
                    }
                }
                w
            }
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires w=W option with a spatial weights matrix (list of lists)"
                )))
            }
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

        let result =
            greeners::SpatialDurbinError::fit(&y_arr, &x_arr, &w_mat, &entity_ids, Some(var_names))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let result_names = result.variable_names.clone();
        let summary = format!(
            "SpatialDurbinError(k={}, n={}, N={}), lambda={:.4}",
            result.beta.len(),
            result.n_obs,
            result.n_entities,
            result.lambda
        );
        let fields = vec![
            (
                "direct_effects".into(),
                model_expansion::coef_dataframe(
                    &result_names,
                    &result.beta,
                    &result.beta_se,
                    &result.beta_t,
                    &result.beta_p,
                    None,
                    None,
                ),
            ),
            (
                "indirect_effects".into(),
                model_expansion::coef_dataframe(
                    &result_names,
                    &result.theta,
                    &result.theta_se,
                    &result.theta_t,
                    &result.theta_p,
                    None,
                    None,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("lambda", Value::Float(result.lambda)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_entities", Value::Int(result.n_entities as i64)),
                    ("n_regressors", Value::Int(result.n_regressors as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "SpatialDurbinErrorResult",
            fields,
        ))
    }

    pub(super) fn spatial_sar(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                                    _ => {
                                        return Err(HayashiError::Runtime(format!(
                                            "{func}: W matrix contains non-numeric values"
                                        )))
                                    }
                                };
                            }
                        }
                        _ => {
                            return Err(HayashiError::Runtime(format!(
                                "{func}: W must be a list of lists (matrix)"
                            )))
                        }
                    }
                }
                w_mat
            }
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires w=W option with a spatial weights matrix (list of lists)"
                )))
            }
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

        let mut coef_names = vec![if result.model_type == "sar" {
            "rho".into()
        } else {
            "lambda".into()
        }];
        coef_names.extend(result.variable_names.clone().unwrap_or_default());

        let summary = format!(
            "Spatial{}(k={}, n={}), R2={:.4}",
            if result.model_type == "sar" {
                "SAR"
            } else {
                "SEM"
            },
            result.params.len(),
            result.n_obs,
            result.r_squared
        );
        let fields = vec![
            (
                "coefficients".into(),
                model_expansion::coef_dataframe(
                    &coef_names,
                    &result.params,
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
                    ("model_type", Value::Str(result.model_type.clone())),
                    ("spatial_param", Value::Float(result.spatial_param)),
                    ("spatial_se", Value::Float(result.spatial_se)),
                    ("spatial_t", Value::Float(result.spatial_t)),
                    ("spatial_p", Value::Float(result.spatial_p)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("converged", Value::Bool(result.converged)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "SpatialResult",
            fields,
        ))
    }
}
