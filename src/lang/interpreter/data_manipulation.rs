use super::eval_expr::ColResult;
use super::helpers::*;
use super::*;
use greeners::dataframe::DataFrameBuilder;
use std::sync::Arc;

mod aggregation;

fn df_col_f64(df: &DataFrame, col: &str) -> Vec<f64> {
    use greeners::Column;
    match df.get_column(col) {
        Ok(Column::Float(a)) => a.to_vec(),
        Ok(Column::Int(a)) => a.iter().map(|&x| x as f64).collect(),
        _ => Vec::new(),
    }
}

fn df_col_string(df: &DataFrame, col: &str) -> Vec<String> {
    use greeners::Column;
    match df.get_column(col) {
        Ok(Column::String(a)) => a.to_vec(),
        Ok(Column::DateTime(a)) => a
            .iter()
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .collect(),
        Ok(Column::Categorical(cat)) => cat.to_strings(),
        Ok(Column::Bool(a)) => a.iter().map(|v| v.to_string()).collect(),
        _ => df.get_string(col).map(|a| a.to_vec()).unwrap_or_default(),
    }
}

/// ttest, count/nrow, collapse, group_by, pivot_longer/pivot_wider, append,
/// merge, reshape, sort, list, winsor, tabgen, ci, centile, recode, dropna,
/// filter, encode/decode, rename, drop, drop_collinear, mutate/generate(),
/// keep/select, tabulate. Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_data_manipulation(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            "count" | "nrow" => self.eval_count(args),
            "ttest" => self.eval_ttest(args, opt_map),
            "collapse" => self.eval_collapse(args, opt_map),
            "group_by" | "groupby" => self.eval_group_by(args),
            "pivot_longer" => self.pivot_longer(func, args, opts, opt_map),
            "pivot_wider" => self.pivot_wider(func, args, opts, opt_map),
            "append" => self.append(func, args, opts, opt_map),
            "rbind" => self.rbind(func, args, opts, opt_map),
            "merge" => self.merge(func, args, opts, opt_map),
            "reshape" => self.reshape(func, args, opts, opt_map),
            "sort" => self.sort(func, args, opts, opt_map),
            "list" => self.list(func, args, opts, opt_map),
            "winsor" | "winsorize" => self.winsor(func, args, opts, opt_map),
            "tabgen" | "tab_gen" | "xi" => self.tabgen(func, args, opts, opt_map),
            "ci" | "ci_means" => self.ci(func, args, opts, opt_map),
            "centile" | "pctile" => self.centile(func, args, opts, opt_map),
            "recode" => self.recode(func, args, opts, opt_map),
            "dropna" => self.dropna(func, args, opts, opt_map),
            "ffill" => self.ffill(func, args, opts, opt_map),
            "filter" => self.filter(func, args, opts, opt_map),
            "encode" | "destring" => self.encode(func, args, opts, opt_map),
            "decode" | "tostring" => self.decode(func, args, opts, opt_map),
            "rename" => self.rename(func, args, opts, opt_map),
            "drop" => self.drop(func, args, opts, opt_map),
            "drop_collinear" => self.drop_collinear(func, args, opts, opt_map),
            "mutate" | "generate" => self.mutate(func, args, opts, opt_map),
            "keep" | "select" => self.keep(func, args, opts, opt_map),
            "tabulate" | "tab" => self.tabulate(func, args, opts, opt_map),
            _ => return Ok(None),
        };
        result.map(Some)
    }

    // ── t-test helpers ────────────────────────────────────────────────────────

    fn ttest_get_col_vals(&self, df: &DataFrame, col: &str) -> Result<Vec<f64>> {
        use greeners::Column;
        match df.get_column(col) {
            Ok(Column::Float(a)) => {
                if a.iter().any(|v| !v.is_finite()) {
                    return Err(HayashiError::Runtime(format!(
                        "ttest: column '{col}' contains NaN or Inf. Use dropna() first."
                    )));
                }
                Ok(a.iter().copied().collect())
            }
            Ok(Column::Int(a)) => Ok(a.iter().map(|&x| x as f64).collect()),
            _ => Err(self.type_err(format!("'{col}' is not numeric"))),
        }
    }

    fn eval_ttest(&mut self, args: &[Expr], opt_map: &HashMap<String, Value>) -> Result<Value> {
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

        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "ttest() requires a variable name as second argument".into(),
            ));
        }

        let var1 = match &args[1] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "variable name must be an identifier".into(),
                ))
            }
        };

        if args.len() >= 3 && matches!(opt_map.get("paired"), Some(Value::Bool(true))) {
            let var2 = match &args[2] {
                Expr::Var(n) | Expr::Str(n) => n.clone(),
                _ => {
                    return Err(HayashiError::Type(
                        "variable name must be an identifier".into(),
                    ))
                }
            };
            return self.ttest_paired(&df, &var1, &var2);
        }

        if let Some(Value::Str(by_col)) = opt_map.get("by") {
            let equal_var = matches!(opt_map.get("unequal"), Some(Value::Bool(false)));
            return self.ttest_two_sample(&df, &var1, by_col, equal_var);
        }

        let mu = match opt_map.get("mu") {
            Some(Value::Float(f)) => *f,
            Some(Value::Int(i)) => *i as f64,
            None => 0.0,
            _ => return Err(HayashiError::Type("mu= must be numeric".into())),
        };
        self.ttest_one_sample(&df, &var1, mu)
    }

    fn ttest_paired(&self, df: &DataFrame, var1: &str, var2: &str) -> Result<Value> {
        use greeners::Stats;
        use ndarray::Array1;
        let v1_vec = self.ttest_get_col_vals(df, var1)?;
        let v2_vec = self.ttest_get_col_vals(df, var2)?;
        let v1 = Array1::from(v1_vec);
        let v2 = Array1::from(v2_vec);
        let res =
            Stats::ttest_paired_full(&v1, &v2).map_err(|e| HayashiError::Runtime(e.to_string()))?;
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
        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("paired t-test".into()));
        map.insert("var1".into(), Value::Str(var1.into()));
        map.insert("var2".into(), Value::Str(var2.into()));
        map.insert("n".into(), Value::Int(res.n as i64));
        map.insert("mean".into(), Value::Float(res.mean));
        map.insert("std_err".into(), Value::Float(res.std_err));
        map.insert("ci_lower".into(), Value::Float(res.ci_lower));
        map.insert("ci_upper".into(), Value::Float(res.ci_upper));
        map.insert("t_stat".into(), Value::Float(res.t_statistic));
        map.insert("df".into(), Value::Float(res.df));
        map.insert("p_value".into(), Value::Float(res.p_value));
        Ok(Value::Dict(Arc::new(map)))
    }

    fn ttest_two_sample(
        &self,
        df: &DataFrame,
        var1: &str,
        by_col: &str,
        equal_var: bool,
    ) -> Result<Value> {
        use greeners::Stats;
        use ndarray::Array1;
        let vals = self.ttest_get_col_vals(df, var1)?;
        let groups = col_to_strings(df, by_col)?;
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
        sort_maybe_numeric_strings(&mut gkeys);
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
        let mut map = HashMap::new();
        map.insert(
            "test".into(),
            Value::Str(if equal_var {
                "two-sample t-test (equal variances)".into()
            } else {
                "two-sample t-test (Welch)".into()
            }),
        );
        map.insert("variable".into(), Value::Str(var1.into()));
        map.insert("by".into(), Value::Str(by_col.into()));
        map.insert("group1".into(), Value::Str(gkeys[0].clone()));
        map.insert("group2".into(), Value::Str(gkeys[1].clone()));
        map.insert("mean1".into(), Value::Float(res.mean1));
        map.insert("mean2".into(), Value::Float(res.mean2));
        map.insert("diff".into(), Value::Float(res.diff));
        map.insert("n1".into(), Value::Int(res.n1 as i64));
        map.insert("n2".into(), Value::Int(res.n2 as i64));
        map.insert("std_err1".into(), Value::Float(res.std_err1));
        map.insert("std_err2".into(), Value::Float(res.std_err2));
        map.insert("t_stat".into(), Value::Float(res.t_statistic));
        map.insert("df".into(), Value::Float(res.df));
        map.insert("p_value".into(), Value::Float(res.p_value));
        map.insert("ci_lower".into(), Value::Float(res.ci_lower));
        map.insert("ci_upper".into(), Value::Float(res.ci_upper));
        map.insert("equal_var".into(), Value::Bool(equal_var));
        Ok(Value::Dict(Arc::new(map)))
    }

    fn ttest_one_sample(&self, df: &DataFrame, var1: &str, mu: f64) -> Result<Value> {
        use greeners::Stats;
        use ndarray::Array1;
        let v_vec = self.ttest_get_col_vals(df, var1)?;
        let v = Array1::from(v_vec);
        let res =
            Stats::ttest_1samp_full(&v, mu).map_err(|e| HayashiError::Runtime(e.to_string()))?;
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
        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("one-sample t-test".into()));
        map.insert("variable".into(), Value::Str(var1.into()));
        map.insert("mu".into(), Value::Float(mu));
        map.insert("n".into(), Value::Int(res.n as i64));
        map.insert("mean".into(), Value::Float(res.mean));
        map.insert("std_dev".into(), Value::Float(res.std_dev));
        map.insert("std_err".into(), Value::Float(res.std_err));
        map.insert("ci_lower".into(), Value::Float(res.ci_lower));
        map.insert("ci_upper".into(), Value::Float(res.ci_upper));
        map.insert("t_stat".into(), Value::Float(res.t_statistic));
        map.insert("df".into(), Value::Float(res.df));
        map.insert("p_value".into(), Value::Float(res.p_value));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn pivot_longer(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                    return Err(HayashiError::Runtime("pivot_longer requires stubs".into()));
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
        let id_data = get_col_f64(&df, &i_col)?;
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
                    let col = get_col_f64(&df, &col_name)?;
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
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn pivot_wider(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let j_strs = col_to_strings(&df, &j_col)?;

        // Detectar se a coluna i é string: se sim, preservar como
        // string no DataFrame resultante; se não, converter para f64.
        let i_is_string = matches!(
            df.get_column(&i_col),
            Ok(greeners::Column::String(_) | greeners::Column::Categorical(_))
        );

        let mut unique_j: Vec<String> = j_strs.clone();
        unique_j.sort();
        unique_j.dedup();

        let (builder, n_wide) = if i_is_string {
            self.pivot_wider_string_ids(
                DataFrame::builder(),
                &df,
                &i_col,
                &j_strs,
                &unique_j,
                &val_vars,
            )?
        } else {
            self.pivot_wider_numeric_ids(
                DataFrame::builder(),
                &df,
                &i_col,
                &j_strs,
                &unique_j,
                &val_vars,
            )?
        };

        let new_df = builder
            .build()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        if !self.capturing {
            println!("pivot_wider: {} → {} observations", df.n_rows(), n_wide);
        }
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    fn pivot_wider_string_ids(
        &self,
        mut builder: DataFrameBuilder,
        df: &Arc<DataFrame>,
        i_col: &str,
        j_strs: &[String],
        unique_j: &[String],
        val_vars: &[String],
    ) -> Result<(DataFrameBuilder, usize)> {
        let id_strs = col_to_strings(df, i_col)?;
        let mut unique_id_strs: Vec<String> = id_strs.clone();
        unique_id_strs.sort();
        unique_id_strs.dedup();
        let n_wide = unique_id_strs.len();
        builder = builder.add_string(i_col, unique_id_strs.clone());
        for var in val_vars {
            let var_data = get_col_f64(df, var)?;
            for jv in unique_j {
                let col_name = format!("{var}{jv}");
                let mut vals = vec![f64::NAN; n_wide];
                for (row, (id, j)) in id_strs.iter().zip(j_strs.iter()).enumerate() {
                    if j == jv {
                        if let Ok(pos) = unique_id_strs.binary_search(id) {
                            vals[pos] = var_data[row];
                        }
                    }
                }
                builder = builder.add_column(&col_name, vals);
            }
        }
        Ok((builder, n_wide))
    }

    fn pivot_wider_numeric_ids(
        &self,
        mut builder: DataFrameBuilder,
        df: &Arc<DataFrame>,
        i_col: &str,
        j_strs: &[String],
        unique_j: &[String],
        val_vars: &[String],
    ) -> Result<(DataFrameBuilder, usize)> {
        let id_vals = get_col_f64(df, i_col)?;
        let mut unique_ids: Vec<f64> = id_vals.to_vec();
        unique_ids.sort_by(nan_last_cmp);
        unique_ids.dedup();
        let n_wide = unique_ids.len();
        builder = builder.add_column(i_col, unique_ids.clone());
        for var in val_vars {
            let var_data = get_col_f64(df, var)?;
            for jv in unique_j {
                let col_name = format!("{var}{jv}");
                let mut vals = vec![f64::NAN; n_wide];
                for (row, (id, j)) in id_vals.iter().zip(j_strs.iter()).enumerate() {
                    if j == jv {
                        if let Ok(pos) = unique_ids.binary_search_by(|a| nan_last_cmp(a, id)) {
                            vals[pos] = var_data[row];
                        }
                    }
                }
                builder = builder.add_column(&col_name, vals);
            }
        }
        Ok((builder, n_wide))
    }

    pub(super) fn append(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let set1: std::collections::HashSet<&str> = names1.iter().map(String::as_str).collect();
        let n1 = df1.n_rows();
        let n2 = df2.n_rows();

        // union of columns: df1 order first, then new ones from df2
        let mut all_names = names1.clone();
        for n in &names2 {
            if !set1.contains(n.as_str()) {
                all_names.push(n.clone());
            }
        }

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
                    df_col_f64(&df1, col)
                } else {
                    vec![f64::NAN; n1]
                };
                let p2 = if in2 {
                    df_col_f64(&df2, col)
                } else {
                    vec![f64::NAN; n2]
                };
                builder = builder.add_column(col, p1.into_iter().chain(p2).collect::<Vec<_>>());
            } else {
                let p1 = if in1 {
                    df_col_string(&df1, col)
                } else {
                    vec![String::new(); n1]
                };
                let p2 = if in2 {
                    df_col_string(&df2, col)
                } else {
                    vec![String::new(); n2]
                };
                builder = builder.add_string(col, p1.into_iter().chain(p2).collect::<Vec<_>>());
            }
        }

        let new_df = builder
            .build()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        println!("({} + {} = {} observations)", n1, n2, n1 + n2);
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn rbind(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "rbind() requires a list of DataFrames".into(),
            ));
        }
        let dfs: Vec<Arc<DataFrame>> = match self.eval_expr(&args[0])? {
            Value::List(lst) => {
                let mut out = Vec::with_capacity(lst.len());
                for v in lst.iter() {
                    match v {
                        Value::DataFrame(d) => out.push(d.clone()),
                        Value::Nil => {} // skip nils from parallel for
                        _ => {
                            return Err(HayashiError::Type(
                                "rbind: list must contain only DataFrames (or nil)".into(),
                            ))
                        }
                    }
                }
                out
            }
            Value::DataFrame(d) => vec![d],
            _ => {
                return Err(HayashiError::Type(
                    "rbind: expected a list of DataFrames".into(),
                ))
            }
        };

        if dfs.is_empty() {
            let empty = DataFrame::builder()
                .build()
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            return Ok(Value::DataFrame(Arc::new(empty)));
        }

        // Collect all column names (preserving order from first df).
        let mut all_names = dfs[0].column_names();
        let seen: std::collections::HashSet<String> = all_names.iter().cloned().collect();
        let mut new_cols: Vec<String> = Vec::new();
        for df in &dfs[1..] {
            for n in df.column_names() {
                if !seen.contains(n.as_str()) {
                    new_cols.push(n.clone());
                }
            }
        }
        all_names.extend(new_cols);

        let total_rows: usize = dfs.iter().map(|d| d.n_rows()).sum();

        let mut builder = DataFrame::builder();
        for col in &all_names {
            use greeners::Column;
            // Determine column type from first df that has it.
            let is_num = dfs.iter().any(|df| {
                matches!(
                    df.get_column(col),
                    Ok(Column::Float(_)) | Ok(Column::Int(_))
                )
            });

            if is_num {
                let mut buf = Vec::with_capacity(total_rows);
                for df in &dfs {
                    let n = df.n_rows();
                    if df.column_names().contains(col) {
                        buf.extend_from_slice(&df_col_f64(df, col));
                    } else {
                        buf.extend(std::iter::repeat_n(f64::NAN, n));
                    }
                }
                builder = builder.add_column(col, buf);
            } else {
                let mut buf = Vec::with_capacity(total_rows);
                for df in &dfs {
                    let n = df.n_rows();
                    if df.column_names().contains(col) {
                        buf.extend_from_slice(&df_col_string(df, col));
                    } else {
                        buf.extend(std::iter::repeat_n(String::new(), n));
                    }
                }
                builder = builder.add_string(col, buf);
            }
        }

        let new_df = builder
            .build()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn merge(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        // lookup index in df2: key_str → first row index
        let key2_strs = col_to_strings(&df2, &key_col)?;
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (j, v) in key2_strs.iter().enumerate().rev() {
            lookup.insert(v.clone(), j); // rev so we keep the first
        }

        let key1_strs = col_to_strings(&df1, &key_col)?;
        let n1 = df1.n_rows();
        let n2 = df2.n_rows();

        // pairs (idx_df1, idx_df2) for each result row
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
        let set1: std::collections::HashSet<&str> = names1.iter().map(String::as_str).collect();

        // extra columns from df2 (excludes key; suffix _2 on collision)
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

        let mut builder = DataFrame::builder();

        // columns from df1
        for col in &names1 {
            use greeners::Column;
            if matches!(
                df1.get_column(col),
                Ok(Column::Float(_)) | Ok(Column::Int(_))
            ) {
                let src = df_col_f64(&df1, col);
                builder = builder.add_column(
                    col,
                    result_rows
                        .iter()
                        .map(|(r1, _)| r1.map_or(f64::NAN, |i| src[i]))
                        .collect::<Vec<_>>(),
                );
            } else {
                let src = df_col_string(&df1, col);
                builder = builder.add_string(
                    col,
                    result_rows
                        .iter()
                        .map(|(r1, _)| r1.map_or(String::new(), |i| src[i].clone()))
                        .collect::<Vec<_>>(),
                );
            }
        }

        // extra columns from df2
        for (src_col, out_col) in &extra {
            use greeners::Column;
            if matches!(
                df2.get_column(src_col),
                Ok(Column::Float(_)) | Ok(Column::Int(_))
            ) {
                let src = df_col_f64(&df2, src_col);
                builder = builder.add_column(
                    out_col,
                    result_rows
                        .iter()
                        .map(|(_, r2)| r2.map_or(f64::NAN, |j| src[j]))
                        .collect::<Vec<_>>(),
                );
            } else {
                let src = df_col_string(&df2, src_col);
                builder = builder.add_string(
                    out_col,
                    result_rows
                        .iter()
                        .map(|(_, r2)| r2.map_or(String::new(), |j| src[j].clone()))
                        .collect::<Vec<_>>(),
                );
            }
        }

        // _merge indicator: 3=matched, 1=left only, 2=right only
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
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn reshape(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "reshape(df, \"long\"|\"wide\", ...) requires at least 2 arguments".into(),
            ));
        }
        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "reshape(): arg 1 must be a DataFrame".into(),
                ))
            }
        };
        let direction = match self.eval_expr(&args[1])? {
            Value::Str(s) => s,
            _ => {
                return Err(HayashiError::Type(
                    "reshape(): arg 2 must be \"long\" or \"wide\"".into(),
                ))
            }
        };
        let i_col = match opt_map.get("i") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "reshape() requires option i=id_col".into(),
                ))
            }
            _ => return Err(HayashiError::Type("i= must be string".into())),
        };
        let j_col = match opt_map.get("j") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "reshape() requires option j=time_col".into(),
                ))
            }
            _ => return Err(HayashiError::Type("j= must be string".into())),
        };

        match direction.as_str() {
            "long" => self.reshape_long(&df, &i_col, &j_col, opt_map),
            "wide" => self.reshape_wide(&df, &i_col, &j_col, opt_map),
            other => Err(HayashiError::Runtime(format!(
                "reshape: direction '{other}' unknown — use \"long\" or \"wide\""
            ))),
        }
    }

    fn reshape_long(
        &self,
        df: &greeners::DataFrame,
        i_col: &str,
        j_col: &str,
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        use greeners::Column;

        let stubs: Vec<String> = match opt_map.get("stubs") {
            Some(Value::List(lst)) => lst
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(HayashiError::Type(
                        "stubs= must be a list of strings".into(),
                    )),
                })
                .collect::<Result<_>>()?,
            None => {
                return Err(HayashiError::Runtime(
                    "reshape long requires option stubs=[\"var1\", \"var2\", ...]".into(),
                ))
            }
            _ => return Err(HayashiError::Type("stubs= must be a list".into())),
        };

        // For each stub, detect columns and extract suffixes
        let col_names = df.column_names();
        let mut stub_suffixes: Vec<Vec<String>> = Vec::new();
        for stub in &stubs {
            let mut suffs: Vec<String> = col_names
                .iter()
                .filter(|c| c.as_str().starts_with(stub.as_str()) && c.as_str() != stub)
                .map(|c| c[stub.len()..].to_string())
                .collect();
            suffs.sort();
            if suffs.is_empty() {
                return Err(HayashiError::Runtime(format!(
                    "reshape long: no column with stub '{stub}' found"
                )));
            }
            stub_suffixes.push(suffs);
        }
        // Validate that all stubs have the same suffixes
        let all_suf = stub_suffixes[0].clone();
        for (stub, suf) in stubs.iter().zip(stub_suffixes.iter()) {
            if suf != &all_suf {
                return Err(HayashiError::Runtime(format!(
                    "reshape long: stub '{stub}' has different suffixes from the others"
                )));
            }
        }

        // Collect id column values
        let n_rows = df.n_rows();
        let id_vals: Vec<String> = match df.get_column(i_col) {
            Ok(Column::Float(arr)) => arr.iter().map(|v| v.to_string()).collect(),
            Ok(Column::Int(arr)) => arr.iter().map(|v| v.to_string()).collect(),
            _ => {
                if let Ok(arr) = df.get_string(i_col) {
                    arr.to_vec()
                } else {
                    return Err(self.rt_err(format!("reshape: id column '{i_col}' not found")));
                }
            }
        };

        let n_suf = all_suf.len();
        let n_out = n_rows * n_suf;

        // Determine columns that are not stubs nor id (pass through)
        let stub_cols: std::collections::HashSet<String> = stubs
            .iter()
            .flat_map(|s| all_suf.iter().map(move |sf| format!("{s}{sf}")))
            .collect();
        let passthrough: Vec<String> = col_names
            .iter()
            .filter(|c| c.as_str() != i_col && !stub_cols.contains(c.as_str()))
            .cloned()
            .collect();

        let mut builder = DataFrame::builder();

        // id column: repeat each value n_suf times
        let id_out: Vec<String> = id_vals
            .iter()
            .flat_map(|v| std::iter::repeat_n(v.clone(), n_suf))
            .collect();
        builder = builder.add_string(i_col, id_out);

        // j column: for each obs, cycle through suffixes
        let j_out: Vec<String> = (0..n_rows).flat_map(|_| all_suf.iter().cloned()).collect();
        builder = builder.add_string(j_col, j_out);

        // passthrough columns
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

        // stub columns
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
            "(reshape long: {} obs × {} variables → {} obs × {} variables)",
            n_rows,
            col_names.len(),
            n_out,
            new_df.column_names().len()
        );
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    fn reshape_wide(
        &self,
        df: &greeners::DataFrame,
        i_col: &str,
        j_col: &str,
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        use greeners::Column;

        let values: Vec<String> = match opt_map.get("values") {
            Some(Value::List(lst)) => lst
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(HayashiError::Type(
                        "values= must be a list of strings".into(),
                    )),
                })
                .collect::<Result<_>>()?,
            None => {
                return Err(HayashiError::Runtime(
                    "reshape wide requires option values=[\"var1\", \"var2\", ...]".into(),
                ))
            }
            _ => return Err(HayashiError::Type("values= must be a list".into())),
        };

        let n_rows = df.n_rows();

        // Collect unique j values (in order of appearance)
        let j_vals: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::new();
            match df.get_column(j_col) {
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
                    if let Ok(arr) = df.get_string(j_col) {
                        for v in arr.iter() {
                            if seen.insert(v.clone()) {
                                out.push(v.clone());
                            }
                        }
                    } else {
                        return Err(HayashiError::Runtime(format!(
                            "reshape wide: j column '{j_col}' not found"
                        )));
                    }
                }
            }
            out
        };

        // j label per row
        let row_j: Vec<String> = match df.get_column(j_col) {
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
                .get_string(j_col)
                .map_err(|_| HayashiError::Runtime("reshape wide: invalid j column".into()))?
                .to_vec(),
        };

        // id per row
        let row_id: Vec<String> = match df.get_column(i_col) {
            Ok(Column::Float(arr)) => arr.iter().map(|v| v.to_string()).collect(),
            Ok(Column::Int(arr)) => arr.iter().map(|v| v.to_string()).collect(),
            _ => df
                .get_string(i_col)
                .map_err(|_| HayashiError::Runtime("reshape wide: invalid i column".into()))?
                .to_vec(),
        };

        // Unique id order
        let mut seen_ids = std::collections::HashSet::new();
        let unique_ids: Vec<String> = row_id
            .iter()
            .filter(|id| seen_ids.insert((*id).clone()))
            .cloned()
            .collect();
        let n_id = unique_ids.len();

        // id_idx[row] → index in unique_ids
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

        // For each value column, build matrix (n_id × n_j)
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
            .map(|s| s.as_str())
            .chain(std::iter::once(j_col))
            .collect();
        let passthrough: Vec<String> = col_names
            .iter()
            .filter(|c| c.as_str() != i_col && !skip.contains(c.as_str()))
            .cloned()
            .collect();

        // Take first passthrough value per id
        let mut builder = DataFrame::builder();
        // id column
        builder = builder.add_string(i_col, unique_ids.clone());
        // passthrough: value of first row with this id
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
            "(reshape wide: {} obs → {} obs × {} variables)",
            n_rows,
            n_id,
            new_df.column_names().len()
        );
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn sort(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                return Ok(Value::List(Arc::new(new_v)));
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

        // extract sorting keys
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
                    SortKey::Num(v) => nan_last_cmp(&v[a], &v[b]),
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
                    builder = builder
                        .add_column(col_name, idx.iter().map(|&i| arr[i]).collect::<Vec<_>>());
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
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn list(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        // args[1..]: Int → nrows; Ident/Str → column
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

        // vars=[A, B, C] — named option (only if no column was given positionally)
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

        // n= option (overrides default 10; positional Int arg takes priority)
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

        // extract column data
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
                        .map(|a| a.into_iter().take(n_rows).collect())
                        .unwrap_or_else(|_| vec!["?".into(); n_rows]),
                };
                (name.clone(), vals)
            })
            .collect();

        // column widths
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

        // header
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

        // rows
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

    pub(super) fn winsor(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let orig = get_col_f64(&df, &var_name)?;
        let lo = winsorized.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = winsorized.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let n_clip = orig
            .iter()
            .zip(winsorized.iter())
            .filter(|(a, b)| a != b)
            .count();

        Arc::make_mut(&mut df)
            .insert(gen_name.clone(), winsorized)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        self.env.set(&df_name, Value::DataFrame(df))?;
        println!("winsor {var_name} → {gen_name}  (p={p}, range=[{lo:.4}, {hi:.4}], {n_clip} obs clipped)");
        Ok(Value::Nil)
    }

    pub(super) fn tabgen(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "tabgen(df, var [, prefix=name])".into(),
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
            Arc::make_mut(&mut df)
                .insert(col_name, vals)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        }
        self.env.set(&df_name, Value::DataFrame(df))?;
        println!("tabgen {var_name}: {n_dummies} dummies generated (prefix={prefix}_)");
        for name in &dummy_names {
            println!("  {name}");
        }
        Ok(Value::Nil)
    }

    pub(super) fn ci(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let col = get_col_f64(&df, &var)?;
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
        let mut map = HashMap::new();
        map.insert("variable".into(), Value::Str(var));
        map.insert("n".into(), Value::Int(vals.len() as i64));
        map.insert("mean".into(), Value::Float(mean));
        map.insert("sd".into(), Value::Float(sd));
        map.insert("std_err".into(), Value::Float(se));
        map.insert("level".into(), Value::Float(level));
        map.insert("ci_lower".into(), Value::Float(lo));
        map.insert("ci_upper".into(), Value::Float(hi));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn centile(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let col = get_col_f64(&df, &var)?;
        let mut sorted: Vec<f64> = col.iter().filter(|v| v.is_finite()).copied().collect();
        if sorted.is_empty() {
            return Err(HayashiError::Runtime(format!(
                "centile: no finite observations in '{var}'"
            )));
        }
        sorted.sort_by(nan_last_cmp);
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
        let mut centile_vec = Vec::new();
        let mut value_vec = Vec::new();
        for p in &pcts {
            let idx = (p / 100.0 * (n - 1) as f64).round() as usize;
            let val = sorted[idx.min(n - 1)];
            println!("    {:>5.1}%  {:>12.4}", p, val);
            centile_vec.push(Value::Float(*p));
            value_vec.push(Value::Float(val));
        }
        println!();
        let mut columns = HashMap::new();
        columns.insert("centile".into(), Value::List(Arc::new(centile_vec)));
        columns.insert("value".into(), Value::List(Arc::new(value_vec)));
        let df = self.dict_to_dataframe(&columns)?;
        Ok(Value::DataFrame(Arc::new(df)))
    }

    pub(super) fn recode(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                    "recode requires from=[...] and to=[...]".into(),
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
            _ => return Err(HayashiError::Runtime("recode requires to=[...]".into())),
        };
        let col = get_col_f64(&df, &var)?;
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
        Arc::make_mut(&mut df)
            .insert(var.clone(), ndarray::Array1::from(recoded))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        self.env.set(&df_name, Value::DataFrame(df))?;
        println!("recode {var}: {n_changed} changes");
        Ok(Value::Nil)
    }

    pub(super) fn dropna(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        // rebuild the DataFrame filtering rows
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
                            .cloned()
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
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn ffill(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "ffill(df) requires a DataFrame as first argument".into(),
            ));
        }
        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(df) => df,
            _ => {
                return Err(HayashiError::Type(
                    "ffill: first argument must be a DataFrame".into(),
                ))
            }
        };
        let new_df = df
            .fillna_ffill()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn filter(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime("filter(list|df, fn|cond)".into()));
        }
        if let Value::List(lst) = self.eval_expr(&args[0])? {
            let fn_val = self.eval_expr(&args[1])?;
            let mut result = Vec::new();
            for item in lst.iter() {
                let pred = self.call_value_fn(&fn_val, std::slice::from_ref(item))?;
                if value_as_bool(&pred) {
                    result.push(item.clone());
                }
            }
            return Ok(Value::List(Arc::new(result)));
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
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn encode(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                    "second argument must be a column name".into(),
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
        Arc::make_mut(&mut df)
            .insert(target_col.clone(), numeric)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        self.env.set(&df_name, Value::DataFrame(df))?;

        println!("encode {col_name} → {target_col}");
        for (i, label) in label_map.iter().enumerate() {
            println!("  {i} = \"{label}\"");
        }
        Ok(Value::Nil)
    }

    pub(super) fn decode(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                    "second argument must be a column name".into(),
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
                    "decode() requires labels=[\"a\", \"b\", ...]".into(),
                ))
            }
        };
        let vals = get_col_f64(&df, &col_name)?;
        let str_vals: Vec<String> = vals
            .iter()
            .map(|&v| {
                let idx = v as usize;
                labels.get(idx).cloned().unwrap_or_else(|| format!("{v}"))
            })
            .collect();
        Arc::make_mut(&mut df)
            .insert_column(
                col_name.clone(),
                greeners::Column::String(ndarray::Array1::from(str_vals)),
            )
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        self.env.set(&df_name, Value::DataFrame(df))?;
        println!("decode {col_name}: {} labels applied", labels.len());
        Ok(Value::Nil)
    }

    pub(super) fn rename(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn drop(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn drop_collinear(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "drop_collinear() requires at least one DataFrame".into(),
            ));
        }
        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(df) => df,
            _ => {
                return Err(HayashiError::Type(
                    "drop_collinear(): first argument must be a DataFrame".into(),
                ))
            }
        };

        // Columns to check: vars=[...] or all numeric columns
        let check_cols: Vec<String> = match opt_map.get("vars") {
            Some(Value::List(lst)) => lst
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(HayashiError::Type(
                        "drop_collinear(): vars must be a list of column names".into(),
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
                    "drop_collinear(): vars must be a list of strings".into(),
                ))
            }
        };

        if check_cols.is_empty() {
            println!("drop_collinear: no numeric column found.");
            return Ok(Value::DataFrame(df));
        }

        let n = df.n_rows();
        let k = check_cols.len();
        let mut mat = ndarray::Array2::<f64>::zeros((n, k));
        for (j, col) in check_cols.iter().enumerate() {
            let col_data = df.get(col).map_err(|_| {
                HayashiError::Runtime(format!(
                    "drop_collinear: column '{col}' not found or not numeric"
                ))
            })?;
            for (i, &v) in col_data.iter().enumerate() {
                mat[[i, j]] = v;
            }
        }

        let (_clean, keep_idx, omit_idx) = greeners::OLS::detect_collinearity(&mat, 1e-10);

        if omit_idx.is_empty() {
            println!(
                "drop_collinear: no collinearity detected among the {} checked columns.",
                k
            );
            return Ok(Value::DataFrame(df));
        }

        let omit_names: Vec<&str> = omit_idx.iter().map(|&i| check_cols[i].as_str()).collect();
        let keep_names: Vec<&str> = keep_idx.iter().map(|&i| check_cols[i].as_str()).collect();

        println!(
            "drop_collinear: {} column(s) removed due to perfect collinearity:",
            omit_names.len()
        );
        for name in &omit_names {
            println!("  o.{name}");
        }
        println!(
            "  {} column(s) kept: {}",
            keep_names.len(),
            keep_names.join(", ")
        );

        let new_df =
            DataFrame::drop(&df, &omit_names).map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn mutate(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            let col_result = self.eval_col_expr_typed(&o.value, &df_val)?;
            match col_result {
                ColResult::Float(vals) => {
                    let arr = ndarray::Array1::from(vals);
                    Arc::make_mut(&mut df_val)
                        .insert(o.name.clone(), arr)
                        .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                }
                ColResult::String(strs) => {
                    use greeners::Column;
                    let col = Column::String(ndarray::Array1::from(strs));
                    Arc::make_mut(&mut df_val)
                        .insert_column(o.name.clone(), col)
                        .map_err(|e: greeners::GreenersError| self.rt_err(e.to_string()))?;
                }
            }
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

    pub(super) fn keep(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        Ok(Value::DataFrame(Arc::new(new_df)))
    }

    pub(super) fn tabulate(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "tabulate() requires (dataframe, varname) or (dataframe, var1, var2)".into(),
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
            let tb_df = tabulate_one(&df, &var1)?;
            Ok(Value::DataFrame(Arc::new(tb_df)))
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
            let (tb_df, chi2_map) = tabulate_two(&df, &var1, &var2, do_chi2)?;
            if let Some(mut map) = chi2_map {
                map.insert("table".into(), Value::DataFrame(Arc::new(tb_df)));
                Ok(Value::Dict(Arc::new(map)))
            } else {
                Ok(Value::DataFrame(Arc::new(tb_df)))
            }
        }
    }
}
