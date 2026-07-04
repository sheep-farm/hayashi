use super::*;

/// ttest, count/nrow, collapse, group_by, pivot_longer/pivot_wider, append,
/// merge, reshape, sort, list, winsor, tabgen, ci, centile, recode, dropna,
/// filter, encode/decode, rename, drop, drop_collinear, mutate/generate(),
/// keep/select, tabulate. Extraído de `eval_call` (ver src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_data_manipulation(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── ttest ────────────────────────────────────────────────────────
            // ── count(df) / nrow(df) — contagem de linhas como valor ─────────
            "count" | "nrow" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "count(df) ou count(df, condição)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };
                if args.len() >= 2 {
                    let mask = self.eval_col_expr(&args[1], &df)?;
                    let n = mask.iter().filter(|&&v| v != 0.0 && !v.is_nan()).count();
                    return Ok(Some(Value::Int(n as i64)));
                }
                Ok(Value::Int(df.n_rows() as i64))
            }

            "ttest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("ttest() requires a DataFrame".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };

                let get_col_vals = |df: &DataFrame, col: &str| -> Result<Vec<f64>> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => {
                            if a.iter().any(|v| !v.is_finite()) {
                                return Err(HayashiError::Runtime(
                                    format!("ttest: column '{col}' contains NaN or Inf. Use dropna() first.")
                                ));
                            }
                            Ok(a.iter().copied().collect())
                        }
                        Ok(Column::Int(a)) => Ok(a.iter().map(|&x| x as f64).collect()),
                        _ => Err(self.type_err(format!("'{col}' is not numeric"))),
                    }
                };

                let _stats = |v: &[f64]| -> (f64, f64, f64) {
                    // (mean, sd, n)
                    let n = v.len() as f64;
                    let m = v.iter().sum::<f64>() / n;
                    let s = if n > 1.0 {
                        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
                    } else {
                        f64::NAN
                    };
                    (m, s, n)
                };

                // ── um argumento variável → uni-amostral ou por grupo ─────────
                if args.len() >= 2 {
                    let var1 = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(HayashiError::Type(
                                "variable name must be an identifier".into(),
                            ))
                        }
                    };

                    use greeners::Stats;
                    use ndarray::Array1;

                    // ── PAREADO: ttest(df, v1, v2, paired=true) ──────────────
                    if args.len() >= 3 && matches!(opt_map.get("paired"), Some(Value::Bool(true))) {
                        let var2 = match &args[2] {
                            Expr::Var(n) | Expr::Str(n) => n.clone(),
                            _ => {
                                return Err(HayashiError::Type(
                                    "variable name must be an identifier".into(),
                                ))
                            }
                        };
                        let v1_vec = get_col_vals(&df, &var1)?;
                        let v2_vec = get_col_vals(&df, &var2)?;
                        let v1 = Array1::from(v1_vec);
                        let v2 = Array1::from(v2_vec);

                        let res = Stats::ttest_paired_full(&v1, &v2)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                        let _tc = t_critical_95(res.df);
                        println!("\nPaired t-test: {var1} - {var2}");
                        println!("{}", "─".repeat(62));
                        println!(
                            "{:<14} {:>6}  {:>10}  {:>10}  {:>10}",
                            "Variable", "Obs", "Mean", "Std. Err.", "[95% CI]"
                        );
                        println!("{}", "─".repeat(62));
                        println!(
                            "{:<14} {:>6.0}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                            format!("{var1}-{var2}"),
                            res.n as f64,
                            res.mean,
                            res.std_err,
                            res.ci_lower,
                            res.ci_upper
                        );
                        println!("{}", "─".repeat(62));
                        println!(
                            "H0: mean(diff) = 0   t = {:.4}   df = {:.0}   p = {:.4}",
                            res.t_statistic, res.df, res.p_value
                        );
                        println!();

                    // ── DOIS GRUPOS: ttest(df, var, by=group) ────────────────
                    } else if let Some(Value::Str(by_col)) = opt_map.get("by") {
                        let by_col = by_col.clone();
                        let vals = get_col_vals(&df, &var1)?;
                        let groups = Self::col_to_strings(&df, &by_col)?;

                        let mut group_data: HashMap<String, Vec<f64>> = HashMap::new();
                        for (i, g) in groups.iter().enumerate() {
                            group_data.entry(g.clone()).or_default().push(vals[i]);
                        }
                        let mut gkeys: Vec<String> = group_data.keys().cloned().collect();
                        if gkeys.len() != 2 {
                            return Err(HayashiError::Runtime(format!(
                                "two-sample ttest requires exactly 2 groups, got {}",
                                gkeys.len()
                            )));
                        }
                        Self::sort_maybe_numeric_strings(&mut gkeys);

                        let equal_var = matches!(opt_map.get("unequal"), Some(Value::Bool(false)));

                        let v1 = Array1::from(group_data[&gkeys[0]].clone());
                        let v2 = Array1::from(group_data[&gkeys[1]].clone());

                        let res = Stats::compare_means(&v1, &v2, equal_var)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                        let tc = t_critical_95(res.df);

                        let title = if equal_var {
                            format!("Two-sample t-test (Equal Variances): {var1} by {by_col}")
                        } else {
                            format!("Two-sample t-test (Welch): {var1} by {by_col}")
                        };
                        println!("\n{}", title);
                        println!("{}", "─".repeat(68));
                        println!(
                            "{:<10} {:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
                            "Group", "Obs", "Mean", "Std. Err.", "Std. Dev.", "[95% CI]"
                        );
                        println!("{}", "─".repeat(68));
                        for (g, m, s, n, se_g) in [
                            (&gkeys[0], res.mean1, res.std_dev1, res.n1, res.std_err1),
                            (&gkeys[1], res.mean2, res.std_dev2, res.n2, res.std_err2),
                        ] {
                            println!(
                                "{:<10} {:>6.0}  {:>10.4}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                                g,
                                n as f64,
                                m,
                                se_g,
                                s,
                                m - tc * se_g,
                                m + tc * se_g
                            );
                        }
                        println!("{}", "─".repeat(68));
                        println!("diff = mean({}) - mean({})", gkeys[0], gkeys[1]);
                        let t_label = if equal_var { "t" } else { "Welch's t" };
                        println!(
                            "H0: diff = 0   {} = {:.4}   df = {:.2}   p = {:.4}",
                            t_label, res.t_statistic, res.df, res.p_value
                        );
                        println!();

                    // ── UNI-AMOSTRAL: ttest(df, var, mu=0) ───────────────────
                    } else {
                        let mu = match opt_map.get("mu") {
                            Some(Value::Float(f)) => *f,
                            Some(Value::Int(i)) => *i as f64,
                            None => 0.0,
                            _ => return Err(HayashiError::Type("mu= must be numeric".into())),
                        };
                        let v_vec = get_col_vals(&df, &var1)?;
                        let v = Array1::from(v_vec);

                        let res = Stats::ttest_1samp_full(&v, mu)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                        let _tc = t_critical_95(res.df);

                        println!("\nOne-sample t-test: {var1}   H0: mean = {mu}");
                        println!("{}", "─".repeat(62));
                        println!(
                            "{:<14} {:>6}  {:>10}  {:>10}  {:>10}",
                            "Variable", "Obs", "Mean", "Std. Err.", "[95% CI]"
                        );
                        println!("{}", "─".repeat(62));
                        println!(
                            "{:<14} {:>6.0}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                            var1, res.n as f64, res.mean, res.std_err, res.ci_lower, res.ci_upper
                        );
                        println!("{}", "─".repeat(62));
                        println!(
                            "t = {:.4}   df = {:.0}   p = {:.4}",
                            res.t_statistic, res.df, res.p_value
                        );
                        println!();
                    }
                } else {
                    return Err(HayashiError::Runtime(
                        "ttest() requires a variable name as second argument".into(),
                    ));
                }

                Ok(Value::Nil)
            }

            // ── collapse ─────────────────────────────────────────────────────
            "collapse" => {
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
                        "second argument must be a function name (mean, sum, min, max, count, sd, median)".into(),
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

                // validar função antes de qualquer cálculo
                match func_name.as_str() {
                    "mean" | "sum" | "min" | "max" | "count" | "sd" | "median" => {}
                    other => return Err(HayashiError::Runtime(format!(
                        "unknown aggregation '{other}' — use: mean, sum, min, max, count, sd, median"
                    ))),
                }

                // variáveis a agregar: args[2..] ou todas as numéricas exceto by
                let agg_vars: Vec<String> = if args.len() > 2 {
                    self.resolve_var_list(&args[2..], &df)?
                } else {
                    use greeners::Column;
                    df.column_names()
                        .into_iter()
                        .filter(|n| {
                            n != &by_col
                                && matches!(
                                    df.get_column(n),
                                    Ok(Column::Float(_)) | Ok(Column::Int(_))
                                )
                        })
                        .collect()
                };

                // dados das colunas numéricas a agregar
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

                // agrupa índices de linha por valor de by
                let by_strs = Self::col_to_strings(&df, &by_col)?;
                let n_obs = df.n_rows();
                let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
                for (i, v) in by_strs.iter().enumerate() {
                    groups.entry(v.clone()).or_default().push(i);
                }

                // ordena chaves de grupo
                let mut keys: Vec<String> = groups.keys().cloned().collect();
                Self::sort_maybe_numeric_strings(&mut keys);

                // função de agregação: NaN nos dados propaga NaN no resultado (IEEE 754)
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
                            (vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64)
                                .sqrt()
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

                // constrói o DataFrame resultado
                let mut builder = DataFrame::builder();

                // coluna by (numérica ou string)
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

                // colunas agregadas
                for (ci, col_name) in agg_vars.iter().enumerate() {
                    let vals: Vec<f64> = keys
                        .iter()
                        .map(|key| {
                            let subset: Vec<f64> =
                                groups[key].iter().map(|&i| col_data[ci][i]).collect();
                            agg(&subset)
                        })
                        .collect();
                    builder = builder.add_column(col_name, vals);
                }

                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({} groups from {} observations)", keys.len(), n_obs);
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── group_by ──────────────────────────────────────────────────────
            // group_by(df, by_col, stat, var1, var2, ...)
            // like collapse but by= is positional, pipe-friendly
            "group_by" | "groupby" => {
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
                let func_name = match &args[2] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(self.type_err(
                        "third argument must be aggregation: mean, sum, min, max, count, sd, median",
                    )),
                };
                match func_name.as_str() {
                    "mean" | "sum" | "min" | "max" | "count" | "sd" | "median" => {}
                    other => return Err(HayashiError::Runtime(format!(
                        "unknown aggregation '{other}' — use: mean, sum, min, max, count, sd, median"
                    ))),
                }

                let agg_vars: Vec<String> = if args.len() > 3 {
                    self.resolve_var_list(&args[3..], &df)?
                } else {
                    use greeners::Column;
                    df.column_names()
                        .into_iter()
                        .filter(|n| {
                            n != &by_col
                                && matches!(
                                    df.get_column(n),
                                    Ok(Column::Float(_)) | Ok(Column::Int(_))
                                )
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

                let by_strs = Self::col_to_strings(&df, &by_col)?;
                let n_obs = df.n_rows();
                let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
                for (i, v) in by_strs.iter().enumerate() {
                    groups.entry(v.clone()).or_default().push(i);
                }
                let mut keys: Vec<String> = groups.keys().cloned().collect();
                Self::sort_maybe_numeric_strings(&mut keys);

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
                            (vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64)
                                .sqrt()
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
                            let subset: Vec<f64> =
                                groups[key].iter().map(|&i| col_data[ci][i]).collect();
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
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── pivot_longer / pivot_wider ───────────────────────────────────
            "pivot_longer" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "pivot_longer(df, stubs=[...], i=id_col, j=time_col)".into(),
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
                let i_col = match opt_map.get("i") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "pivot_longer requires i=id_col".into(),
                        ))
                    }
                };
                let j_col = match opt_map.get("j") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "pivot_longer requires j=time_col".into(),
                        ))
                    }
                };
                let stubs: Vec<String> = match opt_map.get("stubs") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type("stubs must be strings".into())),
                        })
                        .collect::<Result<_>>()?,
                    _ => {
                        if args.len() > 1 {
                            self.resolve_var_list(&args[1..], &df)?
                        } else {
                            return Err(HayashiError::Runtime(
                                "pivot_longer requires stubs".into(),
                            ));
                        }
                    }
                };

                let col_names = df.column_names();
                let mut stub_suffixes: Vec<Vec<String>> = Vec::new();
                for stub in &stubs {
                    let mut suffs: Vec<String> = col_names
                        .iter()
                        .filter(|c| c.starts_with(stub.as_str()) && *c != stub)
                        .map(|c| c[stub.len()..].to_string())
                        .collect();
                    suffs.sort();
                    if suffs.is_empty() {
                        return Err(HayashiError::Runtime(format!(
                            "pivot_longer: no columns with stub '{stub}' found"
                        )));
                    }
                    stub_suffixes.push(suffs);
                }
                let time_vals = &stub_suffixes[0];
                let n_i = df.n_rows();
                let n_t = time_vals.len();
                let n_long = n_i * n_t;

                let mut builder = DataFrame::builder();
                let id_data = Self::get_col_f64(&df, &i_col)?;
                let ids: Vec<f64> = (0..n_long).map(|idx| id_data[idx / n_t]).collect();
                builder = builder.add_column(&i_col, ids);

                let time_numeric = time_vals.iter().all(|s| s.parse::<f64>().is_ok());
                if time_numeric {
                    let ts: Vec<f64> = (0..n_long)
                        .map(|idx| time_vals[idx % n_t].parse::<f64>().unwrap())
                        .collect();
                    builder = builder.add_column(&j_col, ts);
                } else {
                    let ts: Vec<String> = (0..n_long)
                        .map(|idx| time_vals[idx % n_t].clone())
                        .collect();
                    builder = builder.add_string(&j_col, ts);
                }

                for (si, stub) in stubs.iter().enumerate() {
                    let suffs = &stub_suffixes[si];
                    let mut vals = Vec::with_capacity(n_long);
                    for i in 0..n_i {
                        for suf in suffs {
                            let col_name = format!("{stub}{suf}");
                            let col = Self::get_col_f64(&df, &col_name)?;
                            vals.push(col[i]);
                        }
                    }
                    builder = builder.add_column(stub, vals);
                }

                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                if !self.capturing {
                    println!("pivot_longer: {} → {} observations", n_i, n_long);
                }
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            "pivot_wider" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "pivot_wider(df, i=id_col, j=time_col, values=var)".into(),
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
                let i_col = match opt_map.get("i") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "pivot_wider requires i=id_col".into(),
                        ))
                    }
                };
                let j_col = match opt_map.get("j") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "pivot_wider requires j=time_col".into(),
                        ))
                    }
                };
                let val_vars: Vec<String> = if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    match opt_map.get("values") {
                        Some(Value::Str(s)) => vec![s.clone()],
                        Some(Value::List(lst)) => lst
                            .iter()
                            .map(|v| match v {
                                Value::Str(s) => Ok(s.clone()),
                                _ => Err(HayashiError::Type("values must be strings".into())),
                            })
                            .collect::<Result<_>>()?,
                        _ => df
                            .column_names()
                            .into_iter()
                            .filter(|n| n != &i_col && n != &j_col)
                            .collect(),
                    }
                };

                let id_vals = Self::get_col_f64(&df, &i_col)?;
                let j_strs = Self::col_to_strings(&df, &j_col)?;

                let mut unique_ids: Vec<f64> = id_vals.to_vec();
                unique_ids.sort_by(|a, b| a.partial_cmp(b).unwrap());
                unique_ids.dedup();

                let mut unique_j: Vec<String> = j_strs.clone();
                unique_j.sort();
                unique_j.dedup();

                let n_wide = unique_ids.len();
                let mut builder = DataFrame::builder();
                builder = builder.add_column(&i_col, unique_ids.clone());

                for var in &val_vars {
                    let var_data = Self::get_col_f64(&df, var)?;
                    for jv in &unique_j {
                        let col_name = format!("{var}{jv}");
                        let mut vals = vec![f64::NAN; n_wide];
                        for (row, (id, j)) in id_vals.iter().zip(j_strs.iter()).enumerate() {
                            if j == jv {
                                if let Ok(pos) =
                                    unique_ids.binary_search_by(|a| a.partial_cmp(id).unwrap())
                                {
                                    vals[pos] = var_data[row];
                                }
                            }
                        }
                        builder = builder.add_column(&col_name, vals);
                    }
                }

                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                if !self.capturing {
                    println!("pivot_wider: {} → {} observations", df.n_rows(), n_wide);
                }
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── append ───────────────────────────────────────────────────────
            "append" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("append() requires (df1, df2)".into()));
                }
                let df1 = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df2 = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be a DataFrame".into(),
                        ))
                    }
                };

                let names1 = df1.column_names();
                let names2 = df2.column_names();
                let set1: std::collections::HashSet<&str> =
                    names1.iter().map(String::as_str).collect();
                let n1 = df1.n_rows();
                let n2 = df2.n_rows();

                // união de colunas: ordem de df1 primeiro, depois novas de df2
                let mut all_names = names1.clone();
                for n in &names2 {
                    if !set1.contains(n.as_str()) {
                        all_names.push(n.clone());
                    }
                }

                let get_num = |df: &DataFrame, col: &str, n: usize| -> Vec<f64> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => a.to_vec(),
                        Ok(Column::Int(a)) => a.iter().map(|&x| x as f64).collect(),
                        _ => vec![f64::NAN; n],
                    }
                };
                let get_str = |df: &DataFrame, col: &str, n: usize| -> Vec<String> {
                    df.get_string(col)
                        .map(|a| a.to_vec())
                        .unwrap_or_else(|_| vec![String::new(); n])
                };

                let mut builder = DataFrame::builder();
                for col in &all_names {
                    use greeners::Column;
                    let in1 = names1.contains(col);
                    let in2 = names2.contains(col);
                    let is_num = if in1 {
                        matches!(
                            df1.get_column(col),
                            Ok(Column::Float(_)) | Ok(Column::Int(_))
                        )
                    } else {
                        matches!(
                            df2.get_column(col),
                            Ok(Column::Float(_)) | Ok(Column::Int(_))
                        )
                    };
                    if is_num {
                        let p1 = if in1 {
                            get_num(&df1, col, n1)
                        } else {
                            vec![f64::NAN; n1]
                        };
                        let p2 = if in2 {
                            get_num(&df2, col, n2)
                        } else {
                            vec![f64::NAN; n2]
                        };
                        builder =
                            builder.add_column(col, p1.into_iter().chain(p2).collect::<Vec<_>>());
                    } else {
                        let p1 = if in1 {
                            get_str(&df1, col, n1)
                        } else {
                            vec![String::new(); n1]
                        };
                        let p2 = if in2 {
                            get_str(&df2, col, n2)
                        } else {
                            vec![String::new(); n2]
                        };
                        builder =
                            builder.add_string(col, p1.into_iter().chain(p2).collect::<Vec<_>>());
                    }
                }

                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("({} + {} = {} observations)", n1, n2, n1 + n2);
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── merge ─────────────────────────────────────────────────────────
            "merge" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "merge() requires (df1, df2, key=varname [, type=left|inner|outer])".into(),
                    ));
                }
                let df1 = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df2 = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be a DataFrame".into(),
                        ))
                    }
                };
                let key_col = match opt_map.get("key") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(HayashiError::Runtime("merge() requires key=colname".into())),
                };
                let join_type = match opt_map.get("type") {
                    Some(Value::Str(s)) => s.clone(),
                    None => "left".to_string(),
                    _ => return Err(HayashiError::Runtime("type= must be a string".into())),
                };

                // índice de busca no df2: key_str → primeiro índice de linha
                let key2_strs = Self::col_to_strings(&df2, &key_col)?;
                let mut lookup: HashMap<String, usize> = HashMap::new();
                for (j, v) in key2_strs.iter().enumerate().rev() {
                    lookup.insert(v.clone(), j); // rev para ficar com o primeiro
                }

                let key1_strs = Self::col_to_strings(&df1, &key_col)?;
                let n1 = df1.n_rows();
                let n2 = df2.n_rows();

                // pares (idx_df1, idx_df2) para cada linha do resultado
                type RowPair = (Option<usize>, Option<usize>);
                let mut result_rows: Vec<RowPair> = (0..n1)
                    .map(|i| (Some(i), lookup.get(&key1_strs[i]).copied()))
                    .collect();

                match join_type.as_str() {
                    "left" => {}
                    "inner" => result_rows.retain(|(_, r2)| r2.is_some()),
                    "outer" | "full" => {
                        let matched: std::collections::HashSet<usize> =
                            result_rows.iter().filter_map(|(_, r2)| *r2).collect();
                        for j in 0..n2 {
                            if !matched.contains(&j) {
                                result_rows.push((None, Some(j)));
                            }
                        }
                    }
                    other => {
                        return Err(HayashiError::Runtime(format!(
                            "unknown merge type '{other}' — use: left, inner, outer"
                        )))
                    }
                }

                let names1 = df1.column_names();
                let names2 = df2.column_names();
                let set1: std::collections::HashSet<&str> =
                    names1.iter().map(String::as_str).collect();

                // colunas extra de df2 (exclui key; sufixo _2 em colisão)
                let extra: Vec<(String, String)> = names2
                    .iter()
                    .filter(|n| *n != &key_col)
                    .map(|n| {
                        (
                            n.clone(),
                            if set1.contains(n.as_str()) {
                                format!("{n}_2")
                            } else {
                                n.clone()
                            },
                        )
                    })
                    .collect();

                let get_num = |df: &DataFrame, col: &str| -> Vec<f64> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => a.to_vec(),
                        Ok(Column::Int(a)) => a.iter().map(|&x| x as f64).collect(),
                        _ => vec![],
                    }
                };
                let get_str_col = |df: &DataFrame, col: &str| -> Vec<String> {
                    df.get_string(col).map(|a| a.to_vec()).unwrap_or_default()
                };

                let mut builder = DataFrame::builder();

                // colunas de df1
                for col in &names1 {
                    use greeners::Column;
                    if matches!(
                        df1.get_column(col),
                        Ok(Column::Float(_)) | Ok(Column::Int(_))
                    ) {
                        let src = get_num(&df1, col);
                        builder = builder.add_column(
                            col,
                            result_rows
                                .iter()
                                .map(|(r1, _)| r1.map_or(f64::NAN, |i| src[i]))
                                .collect::<Vec<_>>(),
                        );
                    } else {
                        let src = get_str_col(&df1, col);
                        builder = builder.add_string(
                            col,
                            result_rows
                                .iter()
                                .map(|(r1, _)| r1.map_or(String::new(), |i| src[i].clone()))
                                .collect::<Vec<_>>(),
                        );
                    }
                }

                // colunas extras de df2
                for (src_col, out_col) in &extra {
                    use greeners::Column;
                    if matches!(
                        df2.get_column(src_col),
                        Ok(Column::Float(_)) | Ok(Column::Int(_))
                    ) {
                        let src = get_num(&df2, src_col);
                        builder = builder.add_column(
                            out_col,
                            result_rows
                                .iter()
                                .map(|(_, r2)| r2.map_or(f64::NAN, |j| src[j]))
                                .collect::<Vec<_>>(),
                        );
                    } else {
                        let src = get_str_col(&df2, src_col);
                        builder = builder.add_string(
                            out_col,
                            result_rows
                                .iter()
                                .map(|(_, r2)| r2.map_or(String::new(), |j| src[j].clone()))
                                .collect::<Vec<_>>(),
                        );
                    }
                }

                // indicador _merge: 3=matched, 1=left only, 2=right only
                builder = builder.add_column(
                    "_merge",
                    result_rows
                        .iter()
                        .map(|(r1, r2)| match (r1, r2) {
                            (Some(_), Some(_)) => 3.0,
                            (Some(_), None) => 1.0,
                            _ => 2.0,
                        })
                        .collect::<Vec<_>>(),
                );

                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n_matched = result_rows.iter().filter(|(_, r2)| r2.is_some()).count();
                let n_out = result_rows.len();
                emitln!(
                    self,
                    "({n_matched} matched, {} not matched, {n_out} total)",
                    n_out - n_matched
                );
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── reshape ──────────────────────────────────────────────────────
            // reshape(df, "long",  stubs=[...], i=id_col,    j=new_j_col)
            // reshape(df, "wide",  values=[...], i=id_col,   j=j_col)
            "reshape" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "reshape(df, \"long\"|\"wide\", ...) requer pelo menos 2 argumentos".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "reshape(): arg 1 deve ser DataFrame".into(),
                        ))
                    }
                };
                let direction = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => {
                        return Err(HayashiError::Type(
                            "reshape(): arg 2 deve ser \"long\" ou \"wide\"".into(),
                        ))
                    }
                };
                let i_col = match opt_map.get("i") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "reshape() requer opção i=coluna_id".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("i= must be string".into())),
                };
                let j_col = match opt_map.get("j") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "reshape() requer opção j=coluna_tempo".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("j= must be string".into())),
                };

                match direction.as_str() {
                    // ── wide → long ──────────────────────────────────────────
                    "long" => {
                        let stubs: Vec<String> = match opt_map.get("stubs") {
                            Some(Value::List(lst)) => lst
                                .iter()
                                .map(|v| match v {
                                    Value::Str(s) => Ok(s.clone()),
                                    _ => Err(HayashiError::Type(
                                        "stubs= must be a list de strings".into(),
                                    )),
                                })
                                .collect::<Result<_>>()?,
                            None => {
                                return Err(HayashiError::Runtime(
                                    "reshape long requer opção stubs=[\"var1\", \"var2\", ...]"
                                        .into(),
                                ))
                            }
                            _ => return Err(HayashiError::Type("stubs= must be a list".into())),
                        };

                        // Para cada stub, detectar colunas e extrair sufixos
                        let col_names = df.column_names();
                        let mut stub_suffixes: Vec<Vec<String>> = Vec::new();
                        for stub in &stubs {
                            let mut suffs: Vec<String> = col_names
                                .iter()
                                .filter(|c| c.starts_with(stub.as_str()) && *c != stub)
                                .map(|c| c[stub.len()..].to_string())
                                .collect();
                            suffs.sort();
                            if suffs.is_empty() {
                                return Err(HayashiError::Runtime(format!(
                                    "reshape long: nenhuma coluna com stub '{stub}' encontrada"
                                )));
                            }
                            stub_suffixes.push(suffs);
                        }
                        // Validar que todos os stubs têm os mesmos sufixos
                        let all_suf = stub_suffixes[0].clone();
                        for (stub, suf) in stubs.iter().zip(stub_suffixes.iter()) {
                            if suf != &all_suf {
                                return Err(HayashiError::Runtime(format!(
                                    "reshape long: stub '{stub}' tem sufixos diferentes dos demais"
                                )));
                            }
                        }

                        // Coletar valores da coluna id
                        use greeners::Column;
                        let n_rows = df.n_rows();
                        let id_vals: Vec<String> = match df.get_column(&i_col) {
                            Ok(Column::Float(arr)) => arr.iter().map(|v| v.to_string()).collect(),
                            Ok(Column::Int(arr)) => arr.iter().map(|v| v.to_string()).collect(),
                            _ => {
                                if let Ok(arr) = df.get_string(&i_col) {
                                    arr.to_vec()
                                } else {
                                    return Err(self.rt_err(format!(
                                        "reshape: coluna id '{i_col}' not found"
                                    )));
                                }
                            }
                        };

                        let n_suf = all_suf.len();
                        let n_out = n_rows * n_suf;

                        // Determinar colunas que não são stubs nem id (passam direto)
                        let stub_cols: std::collections::HashSet<String> = stubs
                            .iter()
                            .flat_map(|s| all_suf.iter().map(move |sf| format!("{s}{sf}")))
                            .collect();
                        let passthrough: Vec<String> = col_names
                            .iter()
                            .filter(|c| **c != i_col && !stub_cols.contains(*c))
                            .cloned()
                            .collect();

                        let mut builder = DataFrame::builder();

                        // coluna id: repete cada valor n_suf vezes
                        let id_out: Vec<String> = id_vals
                            .iter()
                            .flat_map(|v| std::iter::repeat_n(v.clone(), n_suf))
                            .collect();
                        builder = builder.add_string(&i_col, id_out);

                        // coluna j: para cada obs, cicla pelos sufixos
                        let j_out: Vec<String> =
                            (0..n_rows).flat_map(|_| all_suf.iter().cloned()).collect();
                        builder = builder.add_string(&j_col, j_out);

                        // colunas passthrough
                        for pc in &passthrough {
                            match df.get_column(pc) {
                                Ok(Column::Float(arr)) => {
                                    let vals: Vec<f64> = arr
                                        .iter()
                                        .flat_map(|&v| std::iter::repeat_n(v, n_suf))
                                        .collect();
                                    builder = builder.add_column(pc, vals);
                                }
                                Ok(Column::Int(arr)) => {
                                    let vals: Vec<f64> = arr
                                        .iter()
                                        .flat_map(|&v| std::iter::repeat_n(v as f64, n_suf))
                                        .collect();
                                    builder = builder.add_column(pc, vals);
                                }
                                _ => {}
                            }
                        }

                        // colunas dos stubs
                        for stub in &stubs {
                            let mut vals: Vec<f64> = Vec::with_capacity(n_out);
                            for row in 0..n_rows {
                                for suf in &all_suf {
                                    let col_name = format!("{stub}{suf}");
                                    let v = match df.get_column(&col_name) {
                                        Ok(Column::Float(arr)) => arr[row],
                                        Ok(Column::Int(arr)) => arr[row] as f64,
                                        _ => f64::NAN,
                                    };
                                    vals.push(v);
                                }
                            }
                            builder = builder.add_column(stub, vals);
                        }

                        let new_df = builder
                            .build()
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!(
                            "(reshape long: {} obs × {} variáveis → {} obs × {} variáveis)",
                            n_rows,
                            col_names.len(),
                            n_out,
                            new_df.column_names().len()
                        );
                        Ok(Value::DataFrame(Rc::new(new_df)))
                    }

                    // ── long → wide ──────────────────────────────────────────
                    "wide" => {
                        let values: Vec<String> = match opt_map.get("values") {
                            Some(Value::List(lst)) => lst
                                .iter()
                                .map(|v| match v {
                                    Value::Str(s) => Ok(s.clone()),
                                    _ => Err(HayashiError::Type(
                                        "values= must be a list de strings".into(),
                                    )),
                                })
                                .collect::<Result<_>>()?,
                            None => {
                                return Err(HayashiError::Runtime(
                                    "reshape wide requer opção values=[\"var1\", \"var2\", ...]"
                                        .into(),
                                ))
                            }
                            _ => return Err(HayashiError::Type("values= must be a list".into())),
                        };

                        use greeners::Column;
                        let n_rows = df.n_rows();

                        // Coletar valores únicos de j (em ordem de aparição)
                        let j_vals: Vec<String> = {
                            let mut seen = std::collections::HashSet::new();
                            let mut out = Vec::new();
                            match df.get_column(&j_col) {
                                Ok(Column::Float(arr)) => {
                                    for &v in arr.iter() {
                                        let s = if v.fract() == 0.0 {
                                            format!("{}", v as i64)
                                        } else {
                                            format!("{v}")
                                        };
                                        if seen.insert(s.clone()) {
                                            out.push(s);
                                        }
                                    }
                                }
                                Ok(Column::Int(arr)) => {
                                    for &v in arr.iter() {
                                        let s = v.to_string();
                                        if seen.insert(s.clone()) {
                                            out.push(s);
                                        }
                                    }
                                }
                                _ => {
                                    if let Ok(arr) = df.get_string(&j_col) {
                                        for v in arr.iter() {
                                            if seen.insert(v.clone()) {
                                                out.push(v.clone());
                                            }
                                        }
                                    } else {
                                        return Err(HayashiError::Runtime(format!(
                                            "reshape wide: coluna j '{j_col}' not found"
                                        )));
                                    }
                                }
                            }
                            out
                        };

                        // j label por linha
                        let row_j: Vec<String> = match df.get_column(&j_col) {
                            Ok(Column::Float(arr)) => arr
                                .iter()
                                .map(|&v| {
                                    if v.fract() == 0.0 {
                                        format!("{}", v as i64)
                                    } else {
                                        format!("{v}")
                                    }
                                })
                                .collect(),
                            Ok(Column::Int(arr)) => arr.iter().map(|v| v.to_string()).collect(),
                            _ => df
                                .get_string(&j_col)
                                .map_err(|_| {
                                    HayashiError::Runtime("reshape wide: j coluna inválida".into())
                                })?
                                .to_vec(),
                        };

                        // id por linha
                        let row_id: Vec<String> = match df.get_column(&i_col) {
                            Ok(Column::Float(arr)) => arr.iter().map(|v| v.to_string()).collect(),
                            Ok(Column::Int(arr)) => arr.iter().map(|v| v.to_string()).collect(),
                            _ => df
                                .get_string(&i_col)
                                .map_err(|_| {
                                    HayashiError::Runtime("reshape wide: i coluna inválida".into())
                                })?
                                .to_vec(),
                        };

                        // Ordem única de ids
                        let mut seen_ids = std::collections::HashSet::new();
                        let unique_ids: Vec<String> = row_id
                            .iter()
                            .filter(|id| seen_ids.insert((*id).clone()))
                            .cloned()
                            .collect();
                        let n_id = unique_ids.len();

                        // id_idx[row] → índice no unique_ids
                        let id_pos: std::collections::HashMap<&str, usize> = unique_ids
                            .iter()
                            .enumerate()
                            .map(|(i, s)| (s.as_str(), i))
                            .collect();
                        let j_pos: std::collections::HashMap<&str, usize> = j_vals
                            .iter()
                            .enumerate()
                            .map(|(i, s)| (s.as_str(), i))
                            .collect();

                        // Para cada coluna value, construir matrix (n_id × n_j)
                        let mut value_mats: Vec<Vec<f64>> = values
                            .iter()
                            .map(|_| vec![f64::NAN; n_id * j_vals.len()])
                            .collect();

                        for row in 0..n_rows {
                            let i_idx = id_pos[row_id[row].as_str()];
                            let j_idx = j_pos[row_j[row].as_str()];
                            for (vi, val_col) in values.iter().enumerate() {
                                let v = match df.get_column(val_col) {
                                    Ok(Column::Float(arr)) => arr[row],
                                    Ok(Column::Int(arr)) => arr[row] as f64,
                                    _ => f64::NAN,
                                };
                                value_mats[vi][i_idx * j_vals.len() + j_idx] = v;
                            }
                        }

                        let col_names = df.column_names();
                        let skip: std::collections::HashSet<&str> = values
                            .iter()
                            .chain(std::iter::once(&j_col))
                            .map(String::as_str)
                            .collect();
                        let passthrough: Vec<String> = col_names
                            .iter()
                            .filter(|c| **c != i_col && !skip.contains(c.as_str()))
                            .cloned()
                            .collect();

                        // Pegar primeiro valor de passthrough por id
                        let mut builder = DataFrame::builder();
                        // id column
                        builder = builder.add_string(&i_col, unique_ids.clone());
                        // passthrough: valor da primeira linha com esse id
                        for pc in &passthrough {
                            let mut vals = vec![f64::NAN; n_id];
                            for row in 0..n_rows {
                                let ii = id_pos[row_id[row].as_str()];
                                if vals[ii].is_nan() {
                                    if let Ok(Column::Float(arr)) = df.get_column(pc) {
                                        vals[ii] = arr[row];
                                    } else if let Ok(Column::Int(arr)) = df.get_column(pc) {
                                        vals[ii] = arr[row] as f64;
                                    }
                                }
                            }
                            builder = builder.add_column(pc, vals);
                        }
                        // value columns
                        for (vi, stub) in values.iter().enumerate() {
                            for (ji, jv) in j_vals.iter().enumerate() {
                                let col_name = format!("{stub}{jv}");
                                let col_vals: Vec<f64> = (0..n_id)
                                    .map(|ii| value_mats[vi][ii * j_vals.len() + ji])
                                    .collect();
                                builder = builder.add_column(&col_name, col_vals);
                            }
                        }

                        let new_df = builder
                            .build()
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!(
                            "(reshape wide: {} obs → {} obs × {} variáveis)",
                            n_rows,
                            n_id,
                            new_df.column_names().len()
                        );
                        Ok(Value::DataFrame(Rc::new(new_df)))
                    }

                    other => Err(HayashiError::Runtime(format!(
                        "reshape: direção '{other}' desconhecida — use \"long\" ou \"wide\""
                    ))),
                }
            }

            // ── sort ─────────────────────────────────────────────────────────
            "sort" => {
                if args.len() == 1 {
                    if let Value::List(v) = self.eval_expr(&args[0])? {
                        let mut new_v = (*v).clone();
                        new_v.sort_by(|a, b| {
                            let fa = match a {
                                Value::Float(f) => Some(*f),
                                Value::Int(i) => Some(*i as f64),
                                _ => None,
                            };
                            let fb = match b {
                                Value::Float(f) => Some(*f),
                                Value::Int(i) => Some(*i as f64),
                                _ => None,
                            };
                            match (fa, fb) {
                                (Some(a), Some(b)) => {
                                    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                                }
                                _ => format!("{a}").cmp(&format!("{b}")),
                            }
                        });
                        return Ok(Some(Value::List(Rc::new(new_v))));
                    }
                }
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "sort(list) or sort(dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame or List".into(),
                        ))
                    }
                };
                let sort_vars = self.resolve_var_list(&args[1..], &df)?;
                let desc = matches!(opt_map.get("desc"), Some(Value::Bool(true)));

                // extrai chaves de ordenação
                enum SortKey {
                    Num(Vec<f64>),
                    Str(Vec<String>),
                }
                let keys: Vec<SortKey> = sort_vars
                    .iter()
                    .map(|v| {
                        use greeners::Column;
                        match df.get_column(v) {
                            Ok(Column::Float(arr)) => Ok(SortKey::Num(arr.to_vec())),
                            Ok(Column::Int(arr)) => {
                                Ok(SortKey::Num(arr.iter().map(|&x| x as f64).collect()))
                            }
                            _ => df
                                .get_string(v)
                                .map(|arr| SortKey::Str(arr.to_vec()))
                                .map_err(|_| self.rt_err(format!("column '{v}' not found"))),
                        }
                    })
                    .collect::<Result<_>>()?;

                let n = df.n_rows();
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| {
                    use std::cmp::Ordering;
                    for key in &keys {
                        let ord = match key {
                            SortKey::Num(v) => match (v[a].is_nan(), v[b].is_nan()) {
                                (true, true) => Ordering::Equal,
                                (true, false) => Ordering::Greater,
                                (false, true) => Ordering::Less,
                                (false, false) => v[a].partial_cmp(&v[b]).unwrap(),
                            },
                            SortKey::Str(v) => v[a].cmp(&v[b]),
                        };
                        if ord != Ordering::Equal {
                            return if desc { ord.reverse() } else { ord };
                        }
                    }
                    Ordering::Equal
                });

                let all_names = df.column_names();
                let mut builder = DataFrame::builder();
                for col_name in &all_names {
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            builder = builder.add_column(
                                col_name,
                                idx.iter().map(|&i| arr[i]).collect::<Vec<_>>(),
                            );
                        }
                        Ok(Column::Int(arr)) => {
                            builder = builder.add_column(
                                col_name,
                                idx.iter().map(|&i| arr[i] as f64).collect::<Vec<_>>(),
                            );
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                let v = arr.to_vec();
                                builder = builder.add_string(
                                    col_name,
                                    idx.iter().map(|&i| v[i].clone()).collect::<Vec<_>>(),
                                );
                            }
                        }
                    }
                }

                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("({n} observations sorted)");
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── list ──────────────────────────────────────────────────────────
            "list" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("list() requires a DataFrame".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };

                // args[1..]: Int → nrows; Ident/Str → coluna
                let mut n_explicit: Option<usize> = None;
                let mut col_names: Vec<String> = Vec::new();

                for arg in &args[1..] {
                    match arg {
                        Expr::Int(n) => n_explicit = Some((*n).max(0) as usize),
                        Expr::Var(n) | Expr::Str(n) => col_names.push(n.clone()),
                        _ => {
                            return Err(HayashiError::Type(
                                "list() arguments must be identifiers or row count".into(),
                            ))
                        }
                    }
                }

                // vars=[A, B, C] — opção nomeada (somente se nenhuma coluna foi dada positionally)
                if col_names.is_empty() {
                    if let Some(vars_opt) = opts.iter().find(|o| o.name == "vars") {
                        match &vars_opt.value {
                            Expr::List(items) => {
                                for e in items {
                                    match e {
                                        Expr::Var(n) | Expr::Str(n) => col_names.push(n.clone()),
                                        _ => {}
                                    }
                                }
                            }
                            Expr::Var(n) | Expr::Str(n) => col_names.push(n.clone()),
                            _ => {}
                        }
                    }
                }

                // n= opção (sobrepõe default 10; arg positional Int tem prioridade)
                let n_show = if let Some(n) = n_explicit {
                    n
                } else {
                    match opt_map.get("n") {
                        Some(Value::Int(v)) => (*v).max(0) as usize,
                        Some(Value::Float(v)) => (*v as i64).max(0) as usize,
                        _ => 10usize,
                    }
                };

                if col_names.is_empty() {
                    col_names = df.column_names();
                }

                let n_rows = n_show.min(df.n_rows());

                // extrai dados das colunas
                let cols_data: Vec<(String, Vec<String>)> = col_names
                    .iter()
                    .map(|name| {
                        use greeners::Column;
                        let vals: Vec<String> = match df.get_column(name) {
                            Ok(Column::Float(arr)) => arr
                                .iter()
                                .take(n_rows)
                                .map(|x| {
                                    if x.is_nan() {
                                        ".".into()
                                    } else if x.fract() == 0.0 && x.abs() < 1e14 {
                                        format!("{}", *x as i64)
                                    } else {
                                        format!("{:.4}", x)
                                    }
                                })
                                .collect(),
                            Ok(Column::Int(arr)) => {
                                arr.iter().take(n_rows).map(|x| x.to_string()).collect()
                            }
                            _ => df
                                .get_string(name)
                                .map(|a| a.to_vec().into_iter().take(n_rows).collect())
                                .unwrap_or_else(|_| vec!["?".into(); n_rows]),
                        };
                        (name.clone(), vals)
                    })
                    .collect();

                // larguras de coluna
                let row_num_w = n_rows.to_string().len().max(1);
                let widths: Vec<usize> = cols_data
                    .iter()
                    .map(|(name, vals)| {
                        vals.iter()
                            .map(|v| v.len())
                            .max()
                            .unwrap_or(0)
                            .max(name.len())
                            + 1
                    })
                    .collect();

                // cabeçalho
                print!("{:>rw$} |", "", rw = row_num_w);
                for (i, (name, _)) in cols_data.iter().enumerate() {
                    print!(" {:>w$}", name, w = widths[i]);
                }
                println!();
                println!(
                    "{}-+{}",
                    "-".repeat(row_num_w),
                    "-".repeat(widths.iter().sum::<usize>() + widths.len())
                );

                // linhas
                for r in 0..n_rows {
                    print!("{:>rw$} |", r + 1, rw = row_num_w);
                    for (i, (_, vals)) in cols_data.iter().enumerate() {
                        print!(" {:>w$}", vals[r], w = widths[i]);
                    }
                    println!();
                }
                if df.n_rows() > n_rows {
                    println!("  ({} more observations not shown)", df.n_rows() - n_rows);
                }
                println!();
                Ok(Value::Nil)
            }

            // ── winsor: winsoriza coluna no percentil p e 1-p ──────────────
            // winsor(df, var, p=0.01)       → in-place, corta 1% em cada cauda
            // winsor(df, var, p=0.05, gen=var_w)  → cria nova coluna
            "winsor" | "winsorize" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "winsor(df, var, p=0.01 [, gen=new])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let mut df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be a variable name".into(),
                        ))
                    }
                };
                let p = match opt_map.get("p") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.01,
                };
                let gen_name = match opt_map.get("gen") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => var_name.clone(),
                };

                let winsorized = df
                    .winsorize(&var_name, p)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let orig = Self::get_col_f64(&df, &var_name)?;
                let lo = winsorized.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = winsorized.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let n_clip = orig
                    .iter()
                    .zip(winsorized.iter())
                    .filter(|(a, b)| a != b)
                    .count();

                Rc::make_mut(&mut df)
                    .insert(gen_name.clone(), winsorized)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                self.env.set(&df_name, Value::DataFrame(df))?;
                println!("winsor {var_name} → {gen_name}  (p={p}, range=[{lo:.4}, {hi:.4}], {n_clip} obs clipped)");
                Ok(Value::Nil)
            }

            // ── tabgen: gera dummies a partir de coluna categórica ────────────
            // tabgen(df, var)              → cria var_0, var_1, ...
            // tabgen(df, var, prefix=d)    → cria d_0, d_1, ...
            "tabgen" | "tab_gen" | "xi" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "tabgen(df, var [, prefix=nome])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let mut df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be a variable name".into(),
                        ))
                    }
                };
                let prefix = match opt_map.get("prefix") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => var_name.clone(),
                };

                let dummies = df
                    .generate_dummies(&var_name, &prefix)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n_dummies = dummies.len();
                let dummy_names: Vec<String> = dummies.iter().map(|(n, _)| n.clone()).collect();
                for (col_name, vals) in dummies {
                    Rc::make_mut(&mut df)
                        .insert(col_name, vals)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                }
                self.env.set(&df_name, Value::DataFrame(df))?;
                println!("tabgen {var_name}: {n_dummies} dummies geradas (prefix={prefix}_)");
                for name in &dummy_names {
                    println!("  {name}");
                }
                Ok(Value::Nil)
            }

            // ── ci: intervalo de confiança para a média ─────────────────────
            "ci" | "ci_means" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("ci(df, var [, level=0.95])".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("df".into())),
                };
                let var = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("var".into())),
                };
                let level = match opt_map.get("level") {
                    Some(Value::Float(v)) => *v,
                    _ => 0.95,
                };
                let col = Self::get_col_f64(&df, &var)?;
                let vals: Vec<f64> = col.iter().filter(|v| v.is_finite()).copied().collect();
                let n = vals.len() as f64;
                let mean = vals.iter().sum::<f64>() / n;
                let sd = (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
                let se = sd / n.sqrt();
                let alpha = 1.0 - level;
                let t_crit = greeners::t_quantile(1.0 - alpha / 2.0, n - 1.0);
                let lo = mean - t_crit * se;
                let hi = mean + t_crit * se;
                println!("\n  Variable: {var}   Obs: {}", vals.len());
                println!("  Mean:     {mean:.6}");
                println!("  Std. Err: {se:.6}");
                println!("  [{:.0}% CI] [{lo:.6}, {hi:.6}]\n", level * 100.0);
                Ok(Value::Nil)
            }

            // ── centile: percentis arbitrários ────────────────────────────────
            "centile" | "pctile" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "centile(df, var, centiles=[25, 50, 75])".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("df".into())),
                };
                let var = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("var".into())),
                };
                let col = Self::get_col_f64(&df, &var)?;
                let mut sorted: Vec<f64> = col.iter().filter(|v| v.is_finite()).copied().collect();
                if sorted.is_empty() {
                    return Err(HayashiError::Runtime(format!(
                        "centile: no finite observations in '{var}'"
                    )));
                }
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                let pcts = match opt_map.get("centiles") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .filter_map(|v| match v {
                            Value::Int(i) => Some(*i as f64),
                            Value::Float(f) => Some(*f),
                            _ => None,
                        })
                        .collect::<Vec<f64>>(),
                    _ => vec![1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0],
                };
                println!("\n  Variable: {var}   Obs: {n}");
                for p in &pcts {
                    let idx = (p / 100.0 * (n - 1) as f64).round() as usize;
                    let val = sorted[idx.min(n - 1)];
                    println!("    {:>5.1}%  {:>12.4}", p, val);
                }
                println!();
                Ok(Value::Nil)
            }

            // ── recode: recodifica valores ───────────────────────────────────
            // recode(df, var, rules=[[0, 1], [1, 2], [2, 3]])
            // ou recode(df, var, from=[1,2,3], to=[10,20,30])
            "recode" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "recode(df, var, from=[...], to=[...])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type("df".into())),
                };
                let mut df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("var".into())),
                };
                let from_vals: Vec<f64> = match opt_map.get("from") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .filter_map(|v| match v {
                            Value::Int(i) => Some(*i as f64),
                            Value::Float(f) => Some(*f),
                            _ => None,
                        })
                        .collect(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "recode requer from=[...] e to=[...]".into(),
                        ))
                    }
                };
                let to_vals: Vec<f64> = match opt_map.get("to") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .filter_map(|v| match v {
                            Value::Int(i) => Some(*i as f64),
                            Value::Float(f) => Some(*f),
                            _ => None,
                        })
                        .collect(),
                    _ => return Err(HayashiError::Runtime("recode requer to=[...]".into())),
                };
                let col = Self::get_col_f64(&df, &var)?;
                let recoded: Vec<f64> = col
                    .iter()
                    .map(|&v| {
                        for (i, &fv) in from_vals.iter().enumerate() {
                            if (v - fv).abs() < 0.5 {
                                return to_vals.get(i).copied().unwrap_or(v);
                            }
                        }
                        v
                    })
                    .collect();
                let n_changed = col
                    .iter()
                    .zip(recoded.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                Rc::make_mut(&mut df)
                    .insert(var.clone(), ndarray::Array1::from(recoded))
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                self.env.set(&df_name, Value::DataFrame(df))?;
                println!("recode {var}: {n_changed} changes");
                Ok(Value::Nil)
            }

            // ── dropna ───────────────────────────────────────────────────────
            "dropna" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "dropna() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };

                let check: Vec<String> = if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    use greeners::Column;
                    df.column_names()
                        .into_iter()
                        .filter(|n| matches!(df.get_column(n), Ok(Column::Float(_))))
                        .collect()
                };

                let n = df.n_rows();
                let mut keep = vec![true; n];

                for col_name in &check {
                    use greeners::Column;
                    if let Ok(Column::Float(arr)) = df.get_column(col_name) {
                        for (i, &v) in arr.iter().enumerate() {
                            if v.is_nan() {
                                keep[i] = false;
                            }
                        }
                    }
                }

                let n_drop = keep.iter().filter(|&&k| !k).count();
                let n_kept = n - n_drop;

                // reconstrói o DataFrame filtrando as linhas
                let all_names = df.column_names();
                let mut builder = DataFrame::builder();

                for col_name in &all_names {
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            let vals: Vec<f64> = arr
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        Ok(Column::Int(arr)) => {
                            let vals: Vec<f64> = arr
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v as f64)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                let vals: Vec<String> = arr
                                    .to_vec()
                                    .into_iter()
                                    .enumerate()
                                    .filter(|(i, _)| keep[*i])
                                    .map(|(_, v)| v)
                                    .collect();
                                builder = builder.add_string(col_name, vals);
                            }
                        }
                    }
                }

                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                emitln!(self, "({n_drop} observations dropped, {n_kept} remaining)");
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── filter ───────────────────────────────────────────────────────
            // filter(df, condition_expr) → DataFrame com linhas onde cond ≠ 0
            "filter" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("filter(list|df, fn|cond)".into()));
                }
                if let Value::List(lst) = self.eval_expr(&args[0])? {
                    let fn_val = self.eval_expr(&args[1])?;
                    let mut result = Vec::new();
                    for item in lst.iter() {
                        let pred = self.call_value_fn(&fn_val, std::slice::from_ref(item))?;
                        if Self::value_as_bool(&pred) {
                            result.push(item.clone());
                        }
                    }
                    return Ok(Some(Value::List(Rc::new(result))));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "filter() requires list or DataFrame".into(),
                        ))
                    }
                };
                let mask = self.eval_col_expr(&args[1], &df)?;
                let keep: Vec<bool> = mask.iter().map(|&v| v != 0.0 && !v.is_nan()).collect();
                let n = keep.len();
                let n_kept = keep.iter().filter(|&&k| k).count();
                let n_drop = n - n_kept;

                let all_names = df.column_names();
                let mut builder = DataFrame::builder();
                for col_name in &all_names {
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            let vals: Vec<f64> = arr
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        Ok(Column::Int(arr)) => {
                            let vals: Vec<f64> = arr
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v as f64)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                let vals: Vec<String> = arr
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, _)| keep[*i])
                                    .map(|(_, v)| v.clone())
                                    .collect();
                                builder = builder.add_string(col_name, vals);
                            }
                        }
                    }
                }
                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("({n_drop} observations removed, {n_kept} remaining)");
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── encode: string → numérico ─────────────────────────────────────
            // encode(df, col)           → substitui coluna string por numérica (0, 1, 2...)
            // encode(df, col, gen=new)  → cria nova coluna, mantém original
            "encode" | "destring" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "encode(df, col [, gen=new_name])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let mut df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let col_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be nome de coluna".into(),
                        ))
                    }
                };
                let gen_name = match opt_map.get("gen") {
                    Some(Value::Str(s)) => Some(s.clone()),
                    _ => None,
                };

                let (numeric, label_map) = df
                    .encode(&col_name)
                    .map_err(|e| HayashiError::Runtime(format!("encode '{col_name}': {e}")))?;

                let target_col = gen_name.unwrap_or_else(|| col_name.clone());
                Rc::make_mut(&mut df)
                    .insert(target_col.clone(), numeric)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                self.env.set(&df_name, Value::DataFrame(df))?;

                println!("encode {col_name} → {target_col}");
                for (i, label) in label_map.iter().enumerate() {
                    println!("  {i} = \"{label}\"");
                }
                Ok(Value::Nil)
            }

            // ── decode: numérico → string (oposto de encode) ─────────────────
            // decode(df, col, labels=["a", "b", "c"])
            "decode" | "tostring" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "decode(df, col, labels=[...])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let mut df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let col_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be nome de coluna".into(),
                        ))
                    }
                };
                let labels: Vec<String> = match opt_map.get("labels") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .filter_map(|v| match v {
                            Value::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "decode() requer labels=[\"a\", \"b\", ...]".into(),
                        ))
                    }
                };
                let vals = Self::get_col_f64(&df, &col_name)?;
                let str_vals: Vec<String> = vals
                    .iter()
                    .map(|&v| {
                        let idx = v as usize;
                        labels.get(idx).cloned().unwrap_or_else(|| format!("{v}"))
                    })
                    .collect();
                Rc::make_mut(&mut df)
                    .insert_column(
                        col_name.clone(),
                        greeners::Column::String(ndarray::Array1::from(str_vals)),
                    )
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                self.env.set(&df_name, Value::DataFrame(df))?;
                println!("decode {col_name}: {} labels applied", labels.len());
                Ok(Value::Nil)
            }

            // ── rename ───────────────────────────────────────────────────────
            "rename" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "rename() requires (dataframe, oldname, newname)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let old = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("oldname must be an identifier".into())),
                };
                let new = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("newname must be an identifier".into())),
                };

                let all_names = df.column_names();
                if !all_names.contains(&old) {
                    return Err(HayashiError::Runtime(format!(
                        "column '{old}' not found in DataFrame"
                    )));
                }

                let mut builder = DataFrame::builder();
                for col_name in &all_names {
                    let out_name = if col_name == &old { &new } else { col_name };
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            builder = builder.add_column(out_name, arr.to_vec());
                        }
                        Ok(Column::Int(arr)) => {
                            let vals: Vec<f64> = arr.iter().map(|&v| v as f64).collect();
                            builder = builder.add_column(out_name, vals);
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                builder = builder.add_string(out_name, arr.to_vec());
                            }
                        }
                    }
                }

                let new_df = builder
                    .build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({old} → {new})");
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── drop ─────────────────────────────────────────────────────────
            "drop" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "drop() requires (dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let drop_names: std::collections::HashSet<String> = self
                    .resolve_var_list(&args[1..], &df)?
                    .into_iter()
                    .collect();

                let all = df.column_names();
                let keep: Vec<&str> = all
                    .iter()
                    .filter(|n| !drop_names.contains(*n))
                    .map(String::as_str)
                    .collect();

                let new_df = df
                    .select(&keep)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!(
                    "({} variables dropped, {} remaining)",
                    drop_names.len(),
                    keep.len()
                );
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── drop_collinear ────────────────────────────────────────────────
            // drop_collinear(df [, vars=[x1, x2, ...]])
            // Detecta colunas perfeitamente colineares via QR e retorna novo df
            // sem elas. O usuário vê exatamente o que foi removido antes de
            // passar os dados para qualquer estimador.
            "drop_collinear" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "drop_collinear() requer ao menos um DataFrame".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "drop_collinear(): primeiro argumento deve ser um DataFrame".into(),
                        ))
                    }
                };

                // Colunas a checar: vars=[...] ou todas as numéricas
                let check_cols: Vec<String> = match opt_map.get("vars") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type(
                                "drop_collinear(): vars must be a list de nomes de colunas".into(),
                            )),
                        })
                        .collect::<Result<_>>()?,
                    None => df
                        .column_names()
                        .into_iter()
                        .filter(|name| df.get(name).is_ok())
                        .collect(),
                    _ => {
                        return Err(HayashiError::Type(
                            "drop_collinear(): vars must be a list de strings".into(),
                        ))
                    }
                };

                if check_cols.is_empty() {
                    println!("drop_collinear: nenhuma coluna numérica encontrada.");
                    return Ok(Some(Value::DataFrame(df)));
                }

                let n = df.n_rows();
                let k = check_cols.len();
                let mut mat = ndarray::Array2::<f64>::zeros((n, k));
                for (j, col) in check_cols.iter().enumerate() {
                    let col_data = df.get(col).map_err(|_| {
                        HayashiError::Runtime(format!(
                            "drop_collinear: column '{col}' not found ou não numérica"
                        ))
                    })?;
                    for (i, &v) in col_data.iter().enumerate() {
                        mat[[i, j]] = v;
                    }
                }

                let (_clean, keep_idx, omit_idx) = greeners::OLS::detect_collinearity(&mat, 1e-10);

                if omit_idx.is_empty() {
                    println!("drop_collinear: nenhuma colinearidade detectada entre as {} colunas verificadas.", k);
                    return Ok(Some(Value::DataFrame(df)));
                }

                let omit_names: Vec<&str> =
                    omit_idx.iter().map(|&i| check_cols[i].as_str()).collect();
                let keep_names: Vec<&str> =
                    keep_idx.iter().map(|&i| check_cols[i].as_str()).collect();

                println!(
                    "drop_collinear: {} coluna(s) removida(s) por colinearidade perfeita:",
                    omit_names.len()
                );
                for name in &omit_names {
                    println!("  o.{name}");
                }
                println!(
                    "  {} coluna(s) mantida(s): {}",
                    keep_names.len(),
                    keep_names.join(", ")
                );

                let new_df = DataFrame::drop(&df, &omit_names)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── mutate / generate() ──────────────────────────────────────────
            "mutate" | "generate" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "mutate(df, col1 = expr1, col2 = expr2, ...)".into(),
                    ));
                }
                let mut df_val = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };
                if opts.is_empty() {
                    return Err(HayashiError::Runtime(
                        "mutate: provide at least one column (e.g. mutate(df, z = x^2))".into(),
                    ));
                }
                let mut generated = Vec::new();
                for o in opts {
                    let vals = self.eval_col_expr(&o.value, &df_val)?;
                    let arr = ndarray::Array1::from(vals);
                    Rc::make_mut(&mut df_val)
                        .insert(o.name.clone(), arr)
                        .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                    generated.push(o.name.clone());
                }
                if !self.capturing {
                    emitln!(
                        self,
                        "({} obs)  {} column(s) generated: {}",
                        df_val.n_rows(),
                        generated.len(),
                        generated.join(", ")
                    );
                }
                Ok(Value::DataFrame(df_val))
            }

            // ── keep / select ────────────────────────────────────────────────
            "keep" | "select" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "keep() requires (dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let keep_names = self.resolve_var_list(&args[1..], &df)?;

                let refs: Vec<&str> = keep_names.iter().map(String::as_str).collect();
                let n_before = df.column_names().len();
                let new_df = df
                    .select(&refs)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                emitln!(
                    self,
                    "({} variables kept, {} dropped)",
                    refs.len(),
                    n_before - refs.len()
                );
                Ok(Value::DataFrame(Rc::new(new_df)))
            }

            // ── tabulate ─────────────────────────────────────────────────────
            "tabulate" | "tab" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "tabulate() requires (dataframe, varname) or (dataframe, var1, var2)"
                            .into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let var1 = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "variable name must be an identifier".into(),
                        ))
                    }
                };

                if args.len() == 2 {
                    Self::tabulate_one(&df, &var1)?;
                } else {
                    let var2 = match &args[2] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(HayashiError::Type(
                                "variable name must be an identifier".into(),
                            ))
                        }
                    };
                    let do_chi2 = matches!(opt_map.get("chi2"), Some(Value::Bool(true)));
                    Self::tabulate_two(&df, &var1, &var2, do_chi2)?;
                }

                Ok(Value::Nil)
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
