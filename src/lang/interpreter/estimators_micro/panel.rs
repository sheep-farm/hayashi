use super::super::*;

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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
    }
}
