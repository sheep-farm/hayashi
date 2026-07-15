use super::*;
use std::sync::Arc;

impl Interpreter {
    /// `count(df)` / `nrow(df)` — row count as a value.
    pub(super) fn eval_count(&mut self, args: &[Expr]) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "count(df) or count(df, condition)".into(),
            ));
        }
        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            other => return Err(self.type_mismatch("DataFrame", &other)),
        };
        if args.len() >= 2 {
            let mask = self.eval_col_expr(&args[1], &df)?;
            let n = mask.iter().filter(|&&v| v != 0.0 && !v.is_nan()).count();
            return Ok(Value::Int(n as i64));
        }
        Ok(Value::Int(df.n_rows() as i64))
    }

    /// `collapse(df, func, [vars...], by=col)` — aggregation by group.
    pub(super) fn eval_collapse(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "collapse() requires (df, func, [vars...], by=col)".into(),
            ));
        }
        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "first argument must be a DataFrame".into(),
                ))
            }
        };
        let func_name = match &args[1] {
            Expr::Var(n) => n.clone(),
            _ => return Err(HayashiError::Type(
                "second argument must be a function name (mean, sum, min, max, count, sd, median)"
                    .into(),
            )),
        };
        let by_col = match opt_map.get("by") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(
                    "collapse() requires by=colname".into(),
                ))
            }
        };

        // validate function before any computation
        match func_name.as_str() {
            "mean" | "sum" | "min" | "max" | "count" | "sd" | "median" => {}
            other => {
                return Err(HayashiError::Runtime(format!(
                    "unknown aggregation '{other}' — use: mean, sum, min, max, count, sd, median"
                )))
            }
        }

        // variables to aggregate: args[2..] or all numeric except by
        let agg_vars: Vec<String> = if args.len() > 2 {
            self.resolve_var_list(&args[2..], &df)?
        } else {
            use greeners::Column;
            df.column_names()
                .into_iter()
                .filter(|n| {
                    n != &by_col
                        && matches!(df.get_column(n), Ok(Column::Float(_)) | Ok(Column::Int(_)))
                })
                .collect()
        };

        // numeric column data to aggregate
        let col_data: Vec<Vec<f64>> = agg_vars
            .iter()
            .map(|col| {
                use greeners::Column;
                match df.get_column(col) {
                    Ok(Column::Float(a)) => Ok(a.to_vec()),
                    Ok(Column::Int(a)) => Ok(a.iter().map(|&x| x as f64).collect()),
                    _ => Err(self.type_err(format!("'{col}' is not numeric"))),
                }
            })
            .collect::<Result<_>>()?;

        // group row indices by by value
        let by_strs = col_to_strings(&df, &by_col)?;
        let n_obs = df.n_rows();
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, v) in by_strs.iter().enumerate() {
            groups.entry(v.clone()).or_default().push(i);
        }

        // sort group keys
        let mut keys: Vec<String> = groups.keys().cloned().collect();
        sort_maybe_numeric_strings(&mut keys);

        // aggregation function: NaN in input propagates to output (IEEE 754)
        let agg = |vals: &[f64]| -> f64 {
            let n = vals.len();
            if n == 0 {
                return f64::NAN;
            }
            match func_name.as_str() {
                "count" => n as f64,
                "sum" => vals.iter().sum::<f64>(),
                "mean" => vals.iter().sum::<f64>() / n as f64,
                "min" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
                "max" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "sd" => {
                    if n < 2 {
                        return f64::NAN;
                    }
                    let m = vals.iter().sum::<f64>() / n as f64;
                    (vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
                }
                "median" => {
                    if vals.iter().any(|v| !v.is_finite()) {
                        return f64::NAN;
                    }
                    let mut s = vals.to_vec();
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if n.is_multiple_of(2) {
                        (s[n / 2 - 1] + s[n / 2]) / 2.0
                    } else {
                        s[n / 2]
                    }
                }
                _ => f64::NAN,
            }
        };

        // build the result DataFrame
        let mut builder = DataFrame::builder();

        // by column (numeric or string)
        use greeners::Column;
        if matches!(
            df.get_column(&by_col),
            Ok(Column::Float(_)) | Ok(Column::Int(_))
        ) {
            let vals: Vec<f64> = keys
                .iter()
                .map(|k| k.parse::<f64>().unwrap_or(f64::NAN))
                .collect();
            builder = builder.add_column(&by_col, vals);
        } else {
            builder = builder.add_string(&by_col, keys.clone());
        }

        // aggregated columns
        for (ci, col_name) in agg_vars.iter().enumerate() {
            let vals: Vec<f64> = keys
                .iter()
                .map(|key| {
                    let subset: Vec<f64> = groups[key].iter().map(|&i| col_data[ci][i]).collect();
                    agg(&subset)
                })
                .collect();
            builder = builder.add_column(col_name, vals);
        }

        let new_df = builder
            .build()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        println!("({} groups from {} observations)", keys.len(), n_obs);
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    /// `group_by(df, by_col, stat, var1, var2, ...)` — like collapse, but pipe-friendly.
    pub(super) fn eval_group_by(&mut self, args: &[Expr]) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "group_by(df, by_col, stat, var1, var2, ...)".into(),
            ));
        }
        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "first argument must be a DataFrame".into(),
                ))
            }
        };
        let by_col = match &args[1] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            other => match self.eval_expr(other)? {
                Value::Str(s) => s,
                _ => return Err(self.type_err("by column must be a name or string")),
            },
        };
        let func_name =
            match &args[2] {
                Expr::Var(n) => n.clone(),
                _ => return Err(self.type_err(
                    "third argument must be aggregation: mean, sum, min, max, count, sd, median",
                )),
            };
        match func_name.as_str() {
            "mean" | "sum" | "min" | "max" | "count" | "sd" | "median" => {}
            other => {
                return Err(HayashiError::Runtime(format!(
                    "unknown aggregation '{other}' — use: mean, sum, min, max, count, sd, median"
                )))
            }
        }

        let agg_vars: Vec<String> = if args.len() > 3 {
            self.resolve_var_list(&args[3..], &df)?
        } else {
            use greeners::Column;
            df.column_names()
                .into_iter()
                .filter(|n| {
                    n != &by_col
                        && matches!(df.get_column(n), Ok(Column::Float(_)) | Ok(Column::Int(_)))
                })
                .collect()
        };

        let col_data: Vec<Vec<f64>> = agg_vars
            .iter()
            .map(|col| {
                use greeners::Column;
                match df.get_column(col) {
                    Ok(Column::Float(a)) => Ok(a.to_vec()),
                    Ok(Column::Int(a)) => Ok(a.iter().map(|&x| x as f64).collect()),
                    _ => Err(self.type_err(format!("'{col}' is not numeric"))),
                }
            })
            .collect::<Result<_>>()?;

        let by_strs = col_to_strings(&df, &by_col)?;
        let n_obs = df.n_rows();
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, v) in by_strs.iter().enumerate() {
            groups.entry(v.clone()).or_default().push(i);
        }
        let mut keys: Vec<String> = groups.keys().cloned().collect();
        sort_maybe_numeric_strings(&mut keys);

        let agg_fn = |vals: &[f64]| -> f64 {
            let n = vals.len();
            if n == 0 {
                return f64::NAN;
            }
            match func_name.as_str() {
                "count" => n as f64,
                "sum" => vals.iter().sum::<f64>(),
                "mean" => vals.iter().sum::<f64>() / n as f64,
                "min" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
                "max" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "sd" => {
                    if n < 2 {
                        return f64::NAN;
                    }
                    let m = vals.iter().sum::<f64>() / n as f64;
                    (vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
                }
                "median" => {
                    if vals.iter().any(|v| !v.is_finite()) {
                        return f64::NAN;
                    }
                    let mut s = vals.to_vec();
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if n.is_multiple_of(2) {
                        (s[n / 2 - 1] + s[n / 2]) / 2.0
                    } else {
                        s[n / 2]
                    }
                }
                _ => f64::NAN,
            }
        };

        let mut builder = DataFrame::builder();
        use greeners::Column;
        if matches!(
            df.get_column(&by_col),
            Ok(Column::Float(_)) | Ok(Column::Int(_))
        ) {
            let vals: Vec<f64> = keys
                .iter()
                .map(|k| k.parse::<f64>().unwrap_or(f64::NAN))
                .collect();
            builder = builder.add_column(&by_col, vals);
        } else {
            builder = builder.add_string(&by_col, keys.clone());
        }
        for (ci, col_name) in agg_vars.iter().enumerate() {
            let vals: Vec<f64> = keys
                .iter()
                .map(|key| {
                    let subset: Vec<f64> = groups[key].iter().map(|&i| col_data[ci][i]).collect();
                    agg_fn(&subset)
                })
                .collect();
            builder = builder.add_column(col_name, vals);
        }
        let new_df = builder
            .build()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        if !self.capturing {
            println!("({} groups from {} observations)", keys.len(), n_obs);
        }
        Ok(Value::DataFrame(Arc::new(new_df)))
    }
}
