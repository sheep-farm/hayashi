use super::super::*;
use crate::lang::dap::model_expansion;

impl Interpreter {
    pub(super) fn sfa_production(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let names: Vec<String> = std::iter::once("const".into())
            .chain(
                result
                    .variable_names
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .cloned(),
            )
            .collect();
        let summary = format!(
            "SFA({}), n={}, lambda={:.4}, gamma={:.4}",
            result.model_type, result.n_obs, result.lambda, result.gamma
        );
        let fields = vec![
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
                "efficiency".into(),
                model_expansion::array1_to_series("efficiency", &result.efficiency),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("model_type", Value::Str(result.model_type.clone())),
                    ("sigma_v", Value::Float(result.sigma_v)),
                    ("sigma_u", Value::Float(result.sigma_u)),
                    ("sigma", Value::Float(result.sigma)),
                    ("lambda", Value::Float(result.lambda)),
                    ("gamma", Value::Float(result.gamma)),
                    ("log_likelihood", Value::Float(result.log_likelihood)),
                    ("mean_efficiency", Value::Float(result.mean_efficiency)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "SfaResult",
            fields,
        ))
    }

    pub(super) fn bayes_sfa_production(
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
            greeners::BayesianSFA::fit_production(&y_vec, &x_mat, Some(var_names), n_burn, n_draws)
        } else {
            greeners::BayesianSFA::fit_cost(&y_vec, &x_mat, Some(var_names), n_burn, n_draws)
        }
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let names: Vec<String> = std::iter::once("const".into())
            .chain(
                result
                    .variable_names
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .cloned(),
            )
            .collect();
        let summary = format!(
            "BayesSFA({}), n={}, mean_eff={:.4}",
            result.model_type, result.n_obs, result.mean_efficiency
        );
        let fields = vec![
            (
                "coefficients".into(),
                model_expansion::posterior_coef_df(
                    &names,
                    &result.beta,
                    &result.beta_sd,
                    &result.beta_ci_low,
                    &result.beta_ci_high,
                    &result.beta, // no p_positive in BayesianSfaResult, reuse mean as placeholder
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("model_type", Value::Str(result.model_type.clone())),
                    ("sigma_v", Value::Float(result.sigma_v)),
                    ("sigma_u", Value::Float(result.sigma_u)),
                    ("lambda", Value::Float(result.lambda)),
                    ("gamma", Value::Float(result.gamma)),
                    ("mean_efficiency", Value::Float(result.mean_efficiency)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_draws", Value::Int(result.n_draws as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "BayesianSfaResult",
            fields,
        ))
    }

    pub(super) fn bayes_lm(
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

        let result = greeners::BayesianLinear::fit(&y_arr, &x_arr, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let names: Vec<String> = std::iter::once("const".into())
            .chain(result.variable_names.iter().cloned())
            .collect();
        let ci_low = result.beta_ci.column(0).to_owned();
        let ci_high = result.beta_ci.column(1).to_owned();
        let summary = format!(
            "BayesLM(sigma2={:.4}, R2={:.4}), n={}, pred={}",
            result.sigma2, result.r_squared, result.n_obs, result.n_pred
        );
        let fields = vec![
            (
                "coefficients".into(),
                model_expansion::posterior_coef_df(
                    &names,
                    &result.beta,
                    &result.beta_cov.diag().mapv(f64::sqrt),
                    &ci_low,
                    &ci_high,
                    &result.p_positive,
                ),
            ),
            (
                "beta_cov".into(),
                model_expansion::array2_to_dataframe_named(&result.beta_cov, &names),
            ),
            (
                "fitted".into(),
                model_expansion::array1_to_series("fitted", &result.fitted),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("sigma2", Value::Float(result.sigma2)),
                    ("sigma2_shape", Value::Float(result.sigma2_shape)),
                    ("sigma2_scale", Value::Float(result.sigma2_scale)),
                    ("a_prior", Value::Float(result.a_prior)),
                    ("b_prior", Value::Float(result.b_prior)),
                    ("log_marginal", Value::Float(result.log_marginal)),
                    ("r_squared", Value::Float(result.r_squared)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_pred", Value::Int(result.n_pred as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "BayesianLinearResult",
            fields,
        ))
    }
}
