use super::super::helpers::*;
use super::super::models::FactorModel;
use super::super::*;
use crate::lang::dap::model_expansion;

impl Interpreter {
    pub(super) fn rf(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let result =
            greeners::RandomForest::fit(&y_arr, &x_arr, n_trees, max_depth, Some(var_names))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "RandomForest(trees={}, depth={}), n={}, R2={:.4}",
            result.n_trees, result.max_depth, result.n_obs, result.r_squared
        );
        let fields = vec![
            (
                "fitted".into(),
                model_expansion::array1_to_series("fitted", &result.fitted),
            ),
            (
                "oob_predictions".into(),
                model_expansion::array1_to_series("oob_predictions", &result.oob_predictions),
            ),
            (
                "feature_importance".into(),
                model_expansion::feature_importance_df(
                    &result.variable_names,
                    &result.feature_importance,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("oob_r_squared", Value::Float(result.oob_r_squared)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("mse", Value::Float(result.mse)),
                    ("n_trees", Value::Int(result.n_trees as i64)),
                    ("max_depth", Value::Int(result.max_depth as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "RandomForestResult",
            fields,
        ))
    }

    pub(super) fn gbm(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let summary = format!(
            "GradientBoosting(trees={}, depth={}), n={}, R2={:.4}",
            result.n_trees, result.max_depth, result.n_obs, result.r_squared
        );
        let fields = vec![
            (
                "fitted".into(),
                model_expansion::array1_to_series("fitted", &result.fitted),
            ),
            (
                "feature_importance".into(),
                model_expansion::feature_importance_df(
                    &result.variable_names,
                    &result.feature_importance,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("init_value", Value::Float(result.init_value)),
                    ("learning_rate", Value::Float(result.learning_rate)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("mse", Value::Float(result.mse)),
                    ("n_trees", Value::Int(result.n_trees as i64)),
                    ("max_depth", Value::Int(result.max_depth as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "GradientBoostingResult",
            fields,
        ))
    }

    pub(super) fn mlp(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let result = greeners::MLP::fit(&y_arr, &x_arr, n_hidden, lr, n_epochs, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "MLP(hidden={}, epochs={}), n={}, R2={:.4}",
            result.n_hidden, result.n_epochs, result.n_obs, result.r_squared
        );
        let fields = vec![
            (
                "fitted".into(),
                model_expansion::array1_to_series("fitted", &result.fitted),
            ),
            (
                "w1".into(),
                model_expansion::array2_to_dataframe("w1", &result.w1),
            ),
            (
                "b1".into(),
                model_expansion::array1_to_series("b1", &result.b1),
            ),
            (
                "w2".into(),
                model_expansion::array2_to_dataframe("w2", &result.w2),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("b2", Value::Float(result.b2)),
                    ("n_hidden", Value::Int(result.n_hidden as i64)),
                    ("learning_rate", Value::Float(result.learning_rate)),
                    ("n_epochs", Value::Int(result.n_epochs as i64)),
                    ("final_mse", Value::Float(result.final_mse)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "MlpResult",
            fields,
        ))
    }

    pub(super) fn qrf(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let quantile_names: Vec<String> =
            result.quantiles.iter().map(|&q| format!("q_{q}")).collect();
        let summary = format!(
            "QRF(trees={}, depth={}), n={}, oob_R2={:.4}",
            result.n_trees, result.max_depth, result.n_obs, result.oob_r_squared
        );
        let fields = vec![
            (
                "quantile_predictions".into(),
                model_expansion::array2_to_dataframe_named(
                    &result.quantile_predictions,
                    &quantile_names,
                ),
            ),
            (
                "feature_importance".into(),
                model_expansion::feature_importance_df(
                    &result.variable_names,
                    &result.feature_importance,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    (
                        "quantiles",
                        Value::List(Arc::new(
                            result.quantiles.iter().map(|&q| Value::Float(q)).collect(),
                        )),
                    ),
                    ("oob_r_squared", Value::Float(result.oob_r_squared)),
                    ("n_trees", Value::Int(result.n_trees as i64)),
                    ("max_depth", Value::Int(result.max_depth as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "QrfResult",
            fields,
        ))
    }

    pub(super) fn xgboost(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let summary = format!(
            "XGBoost(trees={}, depth={}), n={}, R2={:.4}",
            result.n_trees, result.max_depth, result.n_obs, result.r_squared
        );
        let fields = vec![
            (
                "fitted".into(),
                model_expansion::array1_to_series("fitted", &result.fitted),
            ),
            (
                "feature_importance".into(),
                model_expansion::feature_importance_df(
                    &result.variable_names,
                    &result.feature_importance,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("init_value", Value::Float(result.init_value)),
                    ("learning_rate", Value::Float(result.learning_rate)),
                    ("lambda", Value::Float(result.lambda)),
                    ("alpha", Value::Float(result.alpha)),
                    ("gamma", Value::Float(result.gamma)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("mse", Value::Float(result.mse)),
                    ("n_trees", Value::Int(result.n_trees as i64)),
                    ("max_depth", Value::Int(result.max_depth as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "XgboostResult",
            fields,
        ))
    }

    pub(super) fn lstm(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
        let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

        let result = greeners::LSTM::fit(&y_arr, n_hidden, seq_len, lr, n_epochs, n_forecast)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "LSTM(hidden={}, seqlen={}, epochs={}), n={}, R2={:.4}",
            result.n_hidden, result.seq_len, result.n_epochs, result.n_obs, result.r_squared
        );
        let fields = vec![
            (
                "fitted".into(),
                model_expansion::array1_to_series("fitted", &result.fitted),
            ),
            (
                "forecast".into(),
                model_expansion::array1_to_series("forecast", &result.forecast),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("final_hidden", Value::Float(result.final_hidden)),
                    ("final_cell", Value::Float(result.final_cell)),
                    ("n_hidden", Value::Int(result.n_hidden as i64)),
                    ("seq_len", Value::Int(result.seq_len as i64)),
                    ("learning_rate", Value::Float(result.learning_rate)),
                    ("n_epochs", Value::Int(result.n_epochs as i64)),
                    ("mse", Value::Float(result.mse)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("n_samples", Value::Int(result.n_samples as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "LstmResult",
            fields,
        ))
    }

    pub(super) fn causalforest(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let y_var = g_formula.dependent.clone();
        let t_var = g_formula
            .independents
            .first()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: need treatment variable")))?
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
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
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
            greeners::CausalForest::fit(&y_arr, &t_vec, &x_mat, n_trees, max_depth, Some(x_vars))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "CausalForest(ate={:.4}, se={:.4}), n={}",
            result.ate, result.ate_se, result.n_obs
        );
        let fields = vec![
            (
                "treatment_effects".into(),
                model_expansion::array1_to_series("treatment_effects", &result.treatment_effects),
            ),
            (
                "feature_importance".into(),
                model_expansion::feature_importance_df(
                    &result.variable_names,
                    &result.feature_importance,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("ate", Value::Float(result.ate)),
                    ("ate_se", Value::Float(result.ate_se)),
                    (
                        "ate_ci",
                        Value::List(Arc::new(vec![
                            Value::Float(result.ate_ci[0]),
                            Value::Float(result.ate_ci[1]),
                        ])),
                    ),
                    ("heterogeneity", Value::Float(result.heterogeneity)),
                    ("n_trees", Value::Int(result.n_trees as i64)),
                    ("max_depth", Value::Int(result.max_depth as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "CausalForestResult",
            fields,
        ))
    }

    pub(super) fn grf(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let y_var = g_formula.dependent.clone();
        let t_var = g_formula
            .independents
            .first()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: need treatment variable")))?
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
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
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

        let result = greeners::GRF::fit(&y_arr, &t_vec, &x_mat, n_trees, max_depth, Some(x_vars))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "GRF(ate={:.4}, se={:.4}), n={}",
            result.ate, result.ate_se, result.n_obs
        );
        let fields = vec![
            (
                "cate".into(),
                model_expansion::array1_to_series("cate", &result.cate),
            ),
            (
                "propensity".into(),
                model_expansion::array1_to_series("propensity", &result.propensity),
            ),
            (
                "outcome_reg".into(),
                model_expansion::array1_to_series("outcome_reg", &result.outcome_reg),
            ),
            (
                "feature_importance".into(),
                model_expansion::feature_importance_df(
                    &result.variable_names,
                    &result.feature_importance,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("ate", Value::Float(result.ate)),
                    ("ate_se", Value::Float(result.ate_se)),
                    (
                        "ate_ci",
                        Value::List(Arc::new(vec![
                            Value::Float(result.ate_ci[0]),
                            Value::Float(result.ate_ci[1]),
                        ])),
                    ),
                    ("heterogeneity", Value::Float(result.heterogeneity)),
                    ("n_trees", Value::Int(result.n_trees as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "GrfResult",
            fields,
        ))
    }

    pub(super) fn conformal(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let summary = format!(
            "Conformal(alpha={:.4}, coverage={:.4}), n={}",
            result.alpha, result.coverage, result.n_test
        );
        let fields = vec![
            (
                "predictions".into(),
                model_expansion::array1_to_series("predictions", &result.predictions),
            ),
            (
                "lower".into(),
                model_expansion::array1_to_series("lower", &result.lower),
            ),
            (
                "upper".into(),
                model_expansion::array1_to_series("upper", &result.upper),
            ),
            (
                "scores".into(),
                model_expansion::series_from_vec("scores", &result.scores),
            ),
            (
                "coefficients".into(),
                model_expansion::coefficients_df(&result.variable_names, &result.coefficients),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("quantile", Value::Float(result.quantile)),
                    ("alpha", Value::Float(result.alpha)),
                    ("coverage", Value::Float(result.coverage)),
                    ("n_train", Value::Int(result.n_train as i64)),
                    ("n_calib", Value::Int(result.n_calib as i64)),
                    ("n_test", Value::Int(result.n_test as i64)),
                    (
                        "empirical_coverage",
                        Value::Float(result.empirical_coverage),
                    ),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "ConformalResult",
            fields,
        ))
    }

    pub(super) fn transformer(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
        let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

        let result = greeners::Transformer::fit(&y_arr, d_model, seq_len, lr, n_epochs, n_forecast)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "Transformer(d_model={}, seqlen={}, epochs={}), n={}, R2={:.4}",
            result.d_model, result.seq_len, result.n_epochs, result.n_obs, result.r_squared
        );
        let fields = vec![
            (
                "fitted".into(),
                model_expansion::array1_to_series("fitted", &result.fitted),
            ),
            (
                "forecast".into(),
                model_expansion::array1_to_series("forecast", &result.forecast),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_heads", Value::Int(result.n_heads as i64)),
                    ("d_model", Value::Int(result.d_model as i64)),
                    ("seq_len", Value::Int(result.seq_len as i64)),
                    ("learning_rate", Value::Float(result.learning_rate)),
                    ("n_epochs", Value::Int(result.n_epochs as i64)),
                    ("mse", Value::Float(result.mse)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("n_samples", Value::Int(result.n_samples as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "TransformerResult",
            fields,
        ))
    }

    pub(super) fn dr_learner(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let y_var = g_formula.dependent.clone();
        let t_var = g_formula
            .independents
            .first()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: need treatment variable")))?
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
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
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

        let result = greeners::DRLearner::fit(&y_arr, &t_vec, &x_mat, n_folds, Some(x_vars))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "DRLearner(ate={:.4}, se={:.4}), n={}",
            result.ate, result.ate_se, result.n_obs
        );
        let fields = vec![
            (
                "cate".into(),
                model_expansion::array1_to_series("cate", &result.cate),
            ),
            (
                "propensity".into(),
                model_expansion::array1_to_series("propensity", &result.propensity),
            ),
            (
                "outcome_reg".into(),
                model_expansion::array1_to_series("outcome_reg", &result.outcome_reg),
            ),
            (
                "cate_coefficients".into(),
                model_expansion::coefficients_df(&result.variable_names, &result.cate_coefficients),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("ate", Value::Float(result.ate)),
                    ("ate_se", Value::Float(result.ate_se)),
                    (
                        "ate_ci",
                        Value::List(Arc::new(vec![
                            Value::Float(result.ate_ci[0]),
                            Value::Float(result.ate_ci[1]),
                        ])),
                    ),
                    ("n_folds", Value::Int(result.n_folds as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "DrLearnerResult",
            fields,
        ))
    }

    pub(super) fn bart(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        Ok(Value::BartResult(Rc::new(result)))
    }

    pub(super) fn gp(
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

        let result = greeners::GaussianProcess::fit(&y_arr, &x_arr, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::GpResult(Rc::new(result)))
    }

    pub(super) fn tmle(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let y_var = g_formula.dependent.clone();
        let t_var = g_formula
            .independents
            .first()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: need treatment variable")))?
            .clone();

        let w_str = match opt_map.get("w") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires w=\"x1,x2\" option (confounders)"
                )))
            }
        };
        let w_vars: Vec<String> = w_str.split(',').map(|s| s.trim().to_string()).collect();

        let n = df.n_rows();
        let y_col = df
            .get_column(y_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
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

        let p = w_vars.len();
        let mut w_mat = ndarray::Array2::<f64>::zeros((n, p));
        for (j, wname) in w_vars.iter().enumerate() {
            let col = df
                .get_column(wname.as_str())
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let vals = col.as_float().ok_or_else(|| {
                HayashiError::Runtime(format!("{func}: '{wname}' must be numeric"))
            })?;
            for i in 0..n {
                w_mat[(i, j)] = vals[i];
            }
        }

        let result = greeners::TMLE::fit(&y_arr, &t_vec, &w_mat)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "TMLE(ate={:.4}, se={:.4}, p={:.4}), n={}",
            result.ate, result.se, result.p_value, result.n_obs
        );
        let fields = vec![
            (
                "propensity".into(),
                model_expansion::array1_to_series("propensity", &result.propensity),
            ),
            (
                "initial_q".into(),
                model_expansion::array1_to_series("initial_q", &result.initial_q),
            ),
            (
                "targeted_q".into(),
                model_expansion::array1_to_series("targeted_q", &result.targeted_q),
            ),
            (
                "clever_covariate".into(),
                model_expansion::array1_to_series("clever_covariate", &result.clever_covariate),
            ),
            (
                "eif".into(),
                model_expansion::array1_to_series("eif", &result.eif),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("ate", Value::Float(result.ate)),
                    ("se", Value::Float(result.se)),
                    ("t_stat", Value::Float(result.t_stat)),
                    ("p_value", Value::Float(result.p_value)),
                    (
                        "ci",
                        Value::List(Arc::new(vec![
                            Value::Float(result.ci[0]),
                            Value::Float(result.ci[1]),
                        ])),
                    ),
                    ("epsilon", Value::Float(result.epsilon)),
                    ("initial_ate", Value::Float(result.initial_ate)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_confounders", Value::Int(result.n_confounders as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "TmleResult",
            fields,
        ))
    }

    pub(super) fn orf(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let y_var = g_formula.dependent.clone();
        let t_var = g_formula
            .independents
            .first()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: need treatment variable")))?
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
        let w_str = match opt_map.get("w") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires w=\"c1,c2\" option (confounders)"
                )))
            }
        };
        let w_vars: Vec<String> = w_str.split(',').map(|s| s.trim().to_string()).collect();
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
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
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

        let p = w_vars.len();
        let mut w_mat = ndarray::Array2::<f64>::zeros((n, p));
        for (j, wname) in w_vars.iter().enumerate() {
            let col = df
                .get_column(wname.as_str())
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let vals = col.as_float().ok_or_else(|| {
                HayashiError::Runtime(format!("{func}: '{wname}' must be numeric"))
            })?;
            for i in 0..n {
                w_mat[(i, j)] = vals[i];
            }
        }

        let result = greeners::OrthogonalForest::fit(
            &y_arr,
            &t_vec,
            &x_mat,
            &w_mat,
            n_trees,
            max_depth,
            Some(x_vars),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "ORF(ate={:.4}, se={:.4}), n={}",
            result.ate, result.ate_se, result.n_obs
        );
        let fields = vec![
            (
                "cate".into(),
                model_expansion::array1_to_series("cate", &result.cate),
            ),
            (
                "feature_importance".into(),
                model_expansion::feature_importance_df(
                    &result.feature_names,
                    &result.feature_importance,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("ate", Value::Float(result.ate)),
                    ("ate_se", Value::Float(result.ate_se)),
                    (
                        "ate_ci",
                        Value::List(Arc::new(vec![
                            Value::Float(result.ate_ci[0]),
                            Value::Float(result.ate_ci[1]),
                        ])),
                    ),
                    ("n_trees", Value::Int(result.n_trees as i64)),
                    ("max_depth", Value::Int(result.max_depth as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "OrfResult",
            fields,
        ))
    }

    pub(super) fn spectral(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (features)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let k = match opt_map.get("k") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires k=N option (number of clusters)"
                )))
            }
        };
        let sigma = match opt_map.get("sigma") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            _ => None,
        };

        let n = df.n_rows();
        let kk = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, kk));
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

        let result = greeners::SpectralClustering::fit(&x_mat, k, sigma, None)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::SpectralResult(Rc::new(result)))
    }

    pub(super) fn isotonic(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let y_var = g_formula.dependent.clone();
        let x_var = g_formula
            .independents
            .first()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: need x variable")))?
            .clone();

        let decreasing = opt_map.get("decreasing").is_some();

        let y_col = df
            .get_column(y_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
        let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

        let x_col = df
            .get_column(x_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let x_vals = x_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{x_var}' must be numeric")))?;
        let x_arr = ndarray::Array1::from_vec(x_vals.to_vec());

        let result = greeners::IsotonicRegression::fit(&x_arr, &y_arr, !decreasing, None)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::IsotonicResult(Rc::new(result)))
    }

    pub(super) fn mice_chained(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let vars_str = match opt_map.get("vars") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires vars=\"x1,x2\" option"
                )))
            }
        };
        let var_names: Vec<String> = vars_str.split(',').map(|s| s.trim().to_string()).collect();
        let m = match opt_map.get("m") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };
        let max_iter = match opt_map.get("iter") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };

        let n = df.n_rows();
        let kk = var_names.len();
        let mut data_mat = ndarray::Array2::<f64>::zeros((n, kk));
        for (j, vname) in var_names.iter().enumerate() {
            let col = df
                .get_column(vname.as_str())
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let vals = col.to_float();
            for i in 0..n {
                data_mat[(i, j)] = vals[i];
            }
        }

        let result = greeners::MiceChained::fit(&data_mat, m, max_iter, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        print!("{result}");
        Ok(Value::Nil)
    }

    pub(super) fn kmeans(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (features)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let k = match opt_map.get("k") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires k=N option (number of clusters)"
                )))
            }
        };

        let n = df.n_rows();
        let kk = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, kk));
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

        let result = greeners::KMeans::fit(&x_mat, k, None, None)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::KmeansResult(Rc::new(result)))
    }

    pub(super) fn dbscan(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (features)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let eps = match opt_map.get("eps") {
            Some(Value::Float(v)) => *v,
            Some(Value::Int(v)) => *v as f64,
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires eps=N option (neighborhood radius)"
                )))
            }
        };
        let min_pts = match opt_map.get("minpts") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires minpts=N option (min points)"
                )))
            }
        };

        let n = df.n_rows();
        let kk = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, kk));
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

        let result = greeners::DBSCAN::fit(&x_mat, eps, min_pts)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::DbscanResult(Rc::new(result)))
    }

    pub(super) fn gmm_clust(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (features)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let k = match opt_map.get("k") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires k=N option (number of clusters)"
                )))
            }
        };
        let max_iter = match opt_map.get("iter") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };
        let tol = match opt_map.get("tol") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            _ => None,
        };

        let n = df.n_rows();
        let kk = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, kk));
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

        let result = greeners::GmmClustering::fit(&x_mat, k, max_iter, tol)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::GmmClusteringResult(Rc::new(result)))
    }

    pub(super) fn reg_path(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_arr, x_arr) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = g_formula.independents.clone();

        let reg_type = match opt_map.get("type") {
            Some(Value::Str(s)) => s.clone(),
            _ => "lasso".to_string(),
        };
        let alpha = match opt_map.get("alpha") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            _ => None,
        };
        let n_lam = match opt_map.get("nlambda") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };

        let result =
            greeners::RegPath::fit(&y_arr, &x_arr, &reg_type, alpha, n_lam, Some(var_names))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        print!("{result}");
        Ok(Value::Nil)
    }

    pub(super) fn qrf_inf(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let (y_arr, x_arr) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = g_formula.independents.clone();

        let q_str = match opt_map.get("q") {
            Some(Value::Str(s)) => s.clone(),
            _ => "0.1,0.5,0.9".to_string(),
        };
        let quantiles: Vec<f64> = q_str
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.5))
            .collect();
        let n_boot = match opt_map.get("boot") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };
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
        let conf = match opt_map.get("conf") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            _ => None,
        };

        let result = greeners::QrfInference::fit(
            &y_arr,
            &x_arr,
            quantiles,
            n_boot,
            n_trees,
            max_depth,
            conf,
            Some(var_names),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let quantile_names: Vec<String> =
            result.quantiles.iter().map(|&q| format!("q_{q}")).collect();
        let summary = format!(
            "QRFInference(n={}, cov={:.4}, conf={:.4}), boot={}",
            result.n_obs, result.coverage, result.confidence, result.n_bootstrap
        );
        let fields = vec![
            (
                "point_estimates".into(),
                model_expansion::array2_to_dataframe_named(
                    &result.point_estimates,
                    &quantile_names,
                ),
            ),
            (
                "lower".into(),
                model_expansion::array2_to_dataframe_named(&result.lower, &quantile_names),
            ),
            (
                "upper".into(),
                model_expansion::array2_to_dataframe_named(&result.upper, &quantile_names),
            ),
            (
                "feature_importance".into(),
                model_expansion::feature_importance_df(
                    &result.variable_names,
                    &result.feature_importance,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    (
                        "quantiles",
                        Value::List(Arc::new(
                            result.quantiles.iter().map(|&q| Value::Float(q)).collect(),
                        )),
                    ),
                    ("coverage", Value::Float(result.coverage)),
                    ("confidence", Value::Float(result.confidence)),
                    ("n_bootstrap", Value::Int(result.n_bootstrap as i64)),
                    ("n_trees", Value::Int(result.n_trees as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_features", Value::Int(result.n_features as i64)),
                    ("oob_r_squared", Value::Float(result.oob_r_squared)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "QrfInferenceResult",
            fields,
        ))
    }

    pub(super) fn hclust(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (features)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let linkage = match opt_map.get("linkage") {
            Some(Value::Str(s)) => match s.as_str() {
                "ward" => greeners::Linkage::Ward,
                "single" => greeners::Linkage::Single,
                "complete" => greeners::Linkage::Complete,
                "average" => greeners::Linkage::Average,
                _ => greeners::Linkage::Ward,
            },
            _ => greeners::Linkage::Ward,
        };
        let cut_height = match opt_map.get("cut") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            _ => None,
        };

        let n = df.n_rows();
        let kk = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, kk));
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

        let result = greeners::HierarchicalClustering::fit(&x_mat, linkage, cut_height)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::HierarchicalResult(Rc::new(result)))
    }

    pub(super) fn tsne(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (features)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let perplexity = match opt_map.get("perplexity") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            _ => None,
        };
        let max_iter = match opt_map.get("iter") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };
        let lr = match opt_map.get("lr") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            _ => None,
        };

        let n = df.n_rows();
        let kk = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, kk));
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

        let result = greeners::TSNE::fit(&x_mat, perplexity, None, max_iter, lr)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        print!("{result}");
        Ok(Value::Nil)
    }

    pub(super) fn umap(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (features)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let n_neighbors = match opt_map.get("neighbors") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };
        let max_iter = match opt_map.get("iter") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };

        let n = df.n_rows();
        let kk = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, kk));
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

        let result = greeners::UMAP::fit(&x_mat, n_neighbors, None, None, max_iter)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        print!("{result}");
        Ok(Value::Nil)
    }

    pub(super) fn biplot(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!("{func}() requires (df)")));
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

        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (features)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let bp_type = match opt_map.get("type") {
            Some(Value::Str(s)) => match s.as_str() {
                "form" => greeners::BiplotType::Form,
                "covariance" => greeners::BiplotType::Covariance,
                _ => greeners::BiplotType::Symmetric,
            },
            _ => greeners::BiplotType::Symmetric,
        };

        let n = df.n_rows();
        let kk = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, kk));
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

        let result = greeners::Biplot::fit(&x_mat, bp_type, Some(x_vars))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        print!("{result}");
        Ok(Value::Nil)
    }

    pub(super) fn lowess(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

    pub(super) fn kde(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                "triangular" => greeners::Kernel::Triangular,
                "uniform" => greeners::Kernel::Uniform,
                other => {
                    return Err(HayashiError::Runtime(format!(
                "kde: kernel='{other}' unknown — use: gaussian, epanechnikov, triangular, uniform"
            )))
                }
            },
            _ => greeners::Kernel::Gaussian,
        };
        let result = greeners::KDEUnivariate::fit(&data, bw_opt, kernel)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::KdeResult(Rc::new(result)))
    }

    pub(super) fn pca(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

    pub(super) fn factor(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
}
