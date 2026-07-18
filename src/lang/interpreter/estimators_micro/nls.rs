use super::super::helpers::*;
use super::super::*;

impl Interpreter {
    pub(super) fn nls_exp(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let (predict_fn, param_names): (&dyn Fn(&[f64], &[f64]) -> f64, Vec<String>) = match func {
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
}
