use super::*;

impl Interpreter {
    /// `rolling` / `rols` — OLS with rolling window.
    pub(super) fn eval_rolling(
        &mut self,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let formula_str = Self::formula_to_string(&formula_ast);
        let g_formula = GFormula::parse(&formula_str)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let window = match opt_map.get("window") {
            Some(Value::Int(n)) => *n as usize,
            None => {
                return Err(HayashiError::Runtime(
                    "rolling() requires window=N (e.g. window=30)".into(),
                ))
            }
            _ => return Err(HayashiError::Type("window= must be integer".into())),
        };
        let dates: Option<Vec<String>> = match opt_map.get("date") {
            Some(Value::Str(col)) => {
                let arr = df.get_string(col)
                    .map_err(|e| HayashiError::Runtime(format!("rolling: date column: {e}")))?;
                Some(arr.to_vec())
            }
            None => None,
            _ => return Err(HayashiError::Type("rolling: date= must be string column name".into())),
        };
        let var_names = df.formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(format!("rolling: variable names: {e}")))?;
        let dates_ref = dates.as_deref();
        let result = greeners::RollingOLS::fit(&y_vec, &x_mat, window, dates_ref, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::RollingResult(Rc::new(result)))
    }

    /// `recursive` / `recols` — recursive OLS (Kalman).
    pub(super) fn eval_recursive(
        &mut self,
        args: &[Expr],
        opts: &[Opt],
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let formula_str = Self::formula_to_string(&formula_ast);
        let g_formula = GFormula::parse(&formula_str)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result = greeners::RecursiveLS::fit(&y_vec, &x_mat)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::RecursiveLSResult(Rc::new(result)))
    }
}
