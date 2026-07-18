use super::super::*;

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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
    }
}
