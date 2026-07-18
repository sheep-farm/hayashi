use super::super::*;

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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
    }
}
