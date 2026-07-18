use super::super::*;

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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
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

        print!("{result}");
        Ok(Value::Nil)
    }
}
