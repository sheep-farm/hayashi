use super::helpers::*;
use super::*;
use std::sync::Arc;

/// ETS, panel threshold, canonical correlation, weighted statistics, tabstat,
/// xtsum, non-parametric tests, unit-root tests, business-cycle filters,
/// penalized regression (lasso/ridge/elasticnet), cointegration, xtset.
/// Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_estimators_misc(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── ETS — Exponential Smoothing (Holt-Winters) ───────────────────

            // ets(df, var, trend=add, seasonal=add, period=12, damped=false)
            // Alias: ses (simple), hwes (Holt-Winters)
            "ets" | "ses" | "hwes" | "holtwinters" | "exponential_smoothing" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ets(df, var, trend=add, seasonal=add, period=12, damped=false)".into(),
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
                let df = match self.env.get(&df_name) {
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
                let y = get_col_f64(&df, &var_name)?;
                // Rule for aliases:
                //   ses         → trend=none, seasonal=none
                //   hwes        → trend=add,  seasonal=add
                //   ets         → uses explicit options (default: add, add)
                let (trend_def, seas_def) = match func {
                    "ses" => ("none", "none"),
                    "hwes" | "holtwinters" => ("add", "add"),
                    _ => ("add", "add"),
                };
                let trend_str = match opt_map.get("trend") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => trend_def.to_string(),
                };
                let seas_str = match opt_map.get("seasonal") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => seas_def.to_string(),
                };
                let period = match opt_map.get("period") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };
                let damped = match opt_map.get("damped") {
                    Some(Value::Bool(b)) => *b,
                    Some(Value::Str(s)) => s == "true" || s == "yes",
                    _ => false,
                };
                let trend_opt: Option<&str> = if trend_str == "none" {
                    None
                } else {
                    Some(&trend_str)
                };
                let seas_opt: Option<&str> = if seas_str == "none" {
                    None
                } else {
                    Some(&seas_str)
                };
                let seas_period = if seas_opt.is_some() { period } else { 0 };
                let result = greeners::ExponentialSmoothing::fit(
                    &ndarray::Array1::from(y.to_vec()),
                    trend_opt,
                    seas_opt,
                    seas_period,
                    damped,
                )
                .map_err(|e| self.rt_err(format!("ets: {e}")))?;
                Ok(Value::EtsResult(Rc::new(result)))
            }

            // ── Panel Threshold (Hansen 1999) ─────────────────────────────────

            // pthresh(y ~ x1 + x2, df, q=var, id=id, threshold=auto)
            "pthresh" | "xtthresh" | "panel_threshold" | "threshold" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "pthresh(y ~ x1 + x2, df, q=threshold_var, id=entity_id)".into(),
                    ));
                }
                let formula = self.resolve_formula(&args[0])?;
                let df_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let q_name = match opt_map.get("q") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "pthresh requires q=threshold_variable".into(),
                        ))
                    }
                };
                let id_name = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "pthresh requires id=entity_column".into(),
                        ))
                    }
                };
                let (df, g_formula, _display) = self.prepare_formula(&formula, &df)?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let q_col = get_col_f64(&df, &q_name)?;
                let id_col = get_col_f64(&df, &id_name)?;
                let entity_ids: ndarray::Array1<i64> =
                    ndarray::Array1::from(id_col.iter().map(|&v| v as i64).collect::<Vec<_>>());
                let q_arr = ndarray::Array1::from(q_col.to_vec());
                let result = greeners::PanelThreshold::fit(&y_vec, &x_mat, &q_arr, &entity_ids)
                    .map_err(|e| self.rt_err(format!("pthresh: {e}")))?;
                Ok(Value::ThresholdResult(Rc::new(result)))
            }

            // ── Canonical Correlation Analysis ────────────────────────────────

            // cancorr(df, xvars=["x1","x2"], yvars=["y1","y2"])
            // ou cancorr(df, x1, x2, ...) com yvars= como lista
            "cancorr" | "canon" | "cancor" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "cancorr(df, xvars=[\"x1\",\"x2\"], yvars=[\"y1\",\"y2\"])".into(),
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
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let x_names: Vec<String> = match opt_map.get("xvars") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type("xvars must be a list of strings".into())),
                        })
                        .collect::<Result<_>>()?,
                    Some(Value::Str(s)) => vec![s.clone()],
                    None => args[1..]
                        .iter()
                        .map(|a| match a {
                            Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                            _ => Err(HayashiError::Type("args must be variable names".into())),
                        })
                        .collect::<Result<_>>()?,
                    _ => {
                        return Err(HayashiError::Type(
                            "xvars= must be a list de strings".into(),
                        ))
                    }
                };
                let y_names: Vec<String> = match opt_map.get("yvars") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type("yvars must be a list of strings".into())),
                        })
                        .collect::<Result<_>>()?,
                    Some(Value::Str(s)) => vec![s.clone()],
                    _ => {
                        return Err(HayashiError::Runtime(
                            "cancorr requer yvars=[\"y1\",\"y2\"]".into(),
                        ))
                    }
                };
                if x_names.is_empty() || y_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "cancorr: xvars and yvars cannot be empty".into(),
                    ));
                }
                let n = df.n_rows();
                let px = x_names.len();
                let py = y_names.len();
                let mut x_mat = ndarray::Array2::<f64>::zeros((n, px));
                let mut y_mat = ndarray::Array2::<f64>::zeros((n, py));
                for (j, name) in x_names.iter().enumerate() {
                    let c = get_col_f64(&df, name)?;
                    for (i, &v) in c.iter().enumerate() {
                        x_mat[[i, j]] = v;
                    }
                }
                for (j, name) in y_names.iter().enumerate() {
                    let c = get_col_f64(&df, name)?;
                    for (i, &v) in c.iter().enumerate() {
                        y_mat[[i, j]] = v;
                    }
                }
                let result = greeners::CanCorr::fit(&x_mat, &y_mat)
                    .map_err(|e| self.rt_err(format!("cancorr: {e}")))?;
                println!("{result}");
                println!("  X vars: {}", x_names.join(", "));
                println!("  Y vars: {}", y_names.join(", "));
                Ok(Value::Nil)
            }

            // ── Weighted statistics ───────────────────────────────────────

            // summarize_w(df, var, weight=wvar, mu0=0, alpha=0.05)
            "summarize_w" | "dstats_w" | "svymean" | "wtsum" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "summarize_w(df, var, weight=wvar, mu0=0, alpha=0.05)".into(),
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
                let df = match self.env.get(&df_name) {
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
                let data = get_col_f64(&df, &var_name)?;
                let weights = match opt_map
                    .get("weight")
                    .or_else(|| opt_map.get("weights").or_else(|| opt_map.get("w")))
                {
                    Some(Value::Str(wname)) => {
                        let wc = get_col_f64(&df, wname)?;
                        Some(ndarray::Array1::from(wc.to_vec()))
                    }
                    _ => None,
                };
                let w_ref = weights.as_ref();
                let ds = greeners::DescrStatsW::new(&ndarray::Array1::from(data.to_vec()), w_ref)
                    .map_err(|e| self.rt_err(format!("summarize_w: {e}")))?;
                let mu0 = match opt_map.get("mu0") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.0,
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.05,
                };
                let (t_stat, t_p) = ds
                    .ttest_mean(mu0)
                    .map_err(|e| self.rt_err(format!("summarize_w t-test: {e}")))?;
                let (ci_lo, ci_hi) = ds
                    .conf_int_mean(alpha)
                    .map_err(|e| self.rt_err(format!("summarize_w CI: {e}")))?;
                let label =
                    w_ref.map_or("(equal weights)".to_string(), |_| "(weighted)".to_string());
                println!("\n{:=^60}", format!(" DescrStats {label} — {var_name} "));
                println!(
                    "{:<20} {:>12}   {:<20} {:>12}",
                    "N",
                    ds.nobs as usize,
                    "Σ weights",
                    format!("{:.2}", ds.sum_weights)
                );
                println!(
                    "{:<20} {:>12.6}   {:<20} {:>12.6}",
                    "Mean", ds.mean, "Std Dev", ds.std
                );
                println!(
                    "{:<20} {:>12.6}   {:<20} {:>12.6}",
                    "Min", ds.min, "Max", ds.max
                );
                println!(
                    "{:<20} {:>12.6}   {:<20} {:>12.6}",
                    "P25", ds.q25, "Median", ds.median
                );
                println!(
                    "{:<20} {:>12.6}   {:<20} {:>12.6}",
                    "P75", ds.q75, "Variance", ds.var
                );
                println!(
                    "{:<20} {:>12.6}   {:<20} {:>12.6}",
                    "Skewness", ds.skewness, "Kurtosis", ds.kurtosis
                );
                println!("{:-^60}", "");
                println!(
                    "  t-test H₀: μ = {:.4}    t = {:.4}   p = {:.4}",
                    mu0, t_stat, t_p
                );
                println!(
                    "  CI {}%: [{:.6}, {:.6}]",
                    ((1.0 - alpha) * 100.0) as usize,
                    ci_lo,
                    ci_hi
                );
                println!("{:=^60}", "");
                Ok(Value::Nil)
            }

            // ── Tabstat — statistics table by group ────────────────────

            // tabstat(df, var1, var2, ..., by=grupo, stats=[mean,sd,n,p25,p75,min,max,sum])
            "tabstat" | "tabstats" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "tabstat(df, var1, ..., by=group, stats=[mean,sd,n])".into(),
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
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_names = self.resolve_var_list(&args[1..], &df)?;
                if var_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "tabstat: provide at least one variable".into(),
                    ));
                }
                // stats= list of statistics to show
                let default_stats = vec!["mean".to_string(), "sd".to_string(), "n".to_string()];
                let stat_list: Vec<String> = match opt_map.get("stats") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => s.clone(),
                            _ => "mean".into(),
                        })
                        .collect(),
                    Some(Value::Str(s)) => vec![s.clone()],
                    _ => default_stats,
                };
                let by_col: Option<Vec<f64>> = match opt_map.get("by") {
                    Some(Value::Str(bname)) => Some(get_col_f64(&df, bname)?.to_vec()),
                    _ => None,
                };
                // Collect unique groups
                let groups: Vec<Option<String>> = if let Some(ref bv) = by_col {
                    let mut uniq: Vec<f64> = bv.clone();
                    uniq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    uniq.dedup();
                    uniq.into_iter()
                        .map(|g| Some(format!("{}", g as i64)))
                        .collect()
                } else {
                    vec![None]
                };
                // Header
                let stat_w = 10usize;
                let var_w = var_names.iter().map(|n| n.len()).max().unwrap_or(6).max(6);
                println!("\n{:=^70}", " tabstat ");
                print!("{:>var_w$}", "");
                for s in &stat_list {
                    print!("  {:>stat_w$}", s);
                }
                println!();
                println!("{}", "-".repeat(var_w + stat_list.len() * (stat_w + 2)));
                for grp in &groups {
                    if let Some(ref g) = grp {
                        println!("  grupo = {g}");
                    }
                    for vname in &var_names {
                        let col = get_col_f64(&df, vname)?;
                        let data: Vec<f64> = if let Some(ref bv) = by_col {
                            let gval: f64 =
                                grp.as_ref().unwrap().parse::<f64>().unwrap_or(f64::NAN);
                            col.iter()
                                .zip(bv.iter())
                                .filter(|(_, &b)| (b - gval).abs() < 1e-9)
                                .map(|(&c, _)| c)
                                .filter(|v| !v.is_nan())
                                .collect()
                        } else {
                            col.iter().cloned().filter(|v| !v.is_nan()).collect()
                        };
                        if data.is_empty() {
                            continue;
                        }
                        let n = data.len() as f64;
                        let mean = data.iter().sum::<f64>() / n;
                        let sd = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / (n - 1.0).max(1.0))
                        .sqrt();
                        let mut sorted = data.clone();
                        sorted.sort_by(nan_last_cmp);
                        let min = sorted[0];
                        let max = *sorted.last().unwrap();
                        let sum = data.iter().sum::<f64>();
                        let pct = |p: f64| {
                            let idx = (p / 100.0 * (n - 1.0)).round() as usize;
                            sorted[idx.min(sorted.len() - 1)]
                        };
                        print!("{:>var_w$}", vname);
                        for s in &stat_list {
                            let val = match s.as_str() {
                                "mean" | "avg" => mean,
                                "sd" | "std" => sd,
                                "var" => sd * sd,
                                "n" | "count" => n,
                                "sum" => sum,
                                "min" => min,
                                "max" => max,
                                "p25" | "q1" => pct(25.0),
                                "p50" | "median" => pct(50.0),
                                "p75" | "q3" => pct(75.0),
                                "p10" => pct(10.0),
                                "p90" => pct(90.0),
                                "iqr" => pct(75.0) - pct(25.0),
                                "range" => max - min,
                                "cv" => {
                                    if mean.abs() > 1e-15 {
                                        sd / mean.abs()
                                    } else {
                                        f64::NAN
                                    }
                                }
                                _ => f64::NAN,
                            };
                            if s == "n" {
                                print!("  {:>stat_w$}", val as usize);
                            } else {
                                print!("  {:>12.4}", val);
                            }
                        }
                        println!();
                    }
                }
                println!("{:=^70}", "");
                Ok(Value::Nil)
            }

            // ── xtsum — within/between decomposition ───────────────────────────

            // xtsum(df, var, id=entity_col)
            "xtsum" | "xt_summary" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "xtsum(df, var1, var2, ..., id=entity_col)".into(),
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
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let id_name = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "xtsum requer id=coluna_entidade".into(),
                        ))
                    }
                };
                let var_names: Vec<String> = {
                    let mut v = Vec::new();
                    for a in &args[1..] {
                        match a {
                            Expr::Var(n) | Expr::Str(n) => v.push(n.clone()),
                            _ => {}
                        }
                    }
                    v
                };
                if var_names.is_empty() {
                    return Err(HayashiError::Runtime(
                        "xtsum: provide at least one variable".into(),
                    ));
                }
                let id_col = get_col_f64(&df, &id_name)?;
                // Identify unique entities
                let mut ids_uniq: Vec<f64> = id_col.to_vec();
                ids_uniq.sort_by(nan_last_cmp);
                ids_uniq.dedup();
                let n_total = df.n_rows();
                let n_entities = ids_uniq.len();
                println!("\n{:=^78}", " xtsum — within/between decomposition ");
                println!(
                    "{:<20} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}",
                    "Variable", "Type", "Mean", "SD", "Min", "Max", "N"
                );
                println!("{}", "-".repeat(78));
                for vname in &var_names {
                    let col = get_col_f64(&df, vname)?;
                    let vals: Vec<f64> = col.iter().cloned().collect();
                    // Overall
                    let n_total_f = n_total as f64;
                    let mean_ov = vals.iter().sum::<f64>() / n_total_f;
                    let var_ov = vals.iter().map(|x| (x - mean_ov).powi(2)).sum::<f64>()
                        / (n_total_f - 1.0).max(1.0);
                    let sd_ov = var_ov.sqrt();
                    let min_ov = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_ov = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    println!(
                        "{:<20} | {:>7} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8}",
                        vname, "overall", mean_ov, sd_ov, min_ov, max_ov, n_total
                    );
                    // Between: mean by entity
                    let group_means: Vec<f64> = ids_uniq
                        .iter()
                        .map(|&gid| {
                            let gvals: Vec<f64> = id_col
                                .iter()
                                .zip(vals.iter())
                                .filter(|(&id, _)| (id - gid).abs() < 1e-9)
                                .map(|(_, &v)| v)
                                .collect();
                            if gvals.is_empty() {
                                f64::NAN
                            } else {
                                gvals.iter().sum::<f64>() / gvals.len() as f64
                            }
                        })
                        .collect();
                    let n_b = n_entities as f64;
                    let mean_b = group_means.iter().filter(|v| !v.is_nan()).sum::<f64>() / n_b;
                    let var_b = group_means
                        .iter()
                        .filter(|v| !v.is_nan())
                        .map(|x| (x - mean_b).powi(2))
                        .sum::<f64>()
                        / (n_b - 1.0).max(1.0);
                    let sd_b = var_b.sqrt();
                    let min_b = group_means.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_b = group_means
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    println!(
                        "{:<20} | {:>7} | {:>8} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8}",
                        "", "between", "", sd_b, min_b, max_b, n_entities
                    );
                    // Within: deviation of each obs from its group mean
                    let within_vals: Vec<f64> = id_col
                        .iter()
                        .zip(vals.iter())
                        .map(|(&id, &v)| {
                            let gm = group_means[ids_uniq
                                .iter()
                                .position(|&g| (g - id).abs() < 1e-9)
                                .unwrap_or(0)];
                            v - gm + mean_ov // demeaned + overall mean
                        })
                        .collect();
                    let n_w = n_total as f64;
                    let var_w = within_vals
                        .iter()
                        .map(|x| (x - mean_ov).powi(2))
                        .sum::<f64>()
                        / (n_w - 1.0).max(1.0);
                    let sd_w = var_w.sqrt();
                    let min_w = within_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_w = within_vals
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    println!(
                        "{:<20} | {:>7} | {:>8} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8}",
                        "", "within", "", sd_w, min_w, max_w, n_total
                    );
                }
                println!("{:=^78}", "");
                println!(
                    "  Entities: {}   Avg periods: {:.1}",
                    n_entities,
                    n_total as f64 / n_entities as f64
                );
                Ok(Value::Nil)
            }

            // ── Non-parametric tests ───────────────────────────────────────

            // spearman(df, var1, var2) — Spearman correlation
            "spearman" | "spearman_rho" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("spearman(df, var1, var2)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let v1 = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be a variable name".into(),
                        ))
                    }
                };
                let v2 = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "third argument must be a variable name".into(),
                        ))
                    }
                };
                let x = get_col_f64(&df, &v1)?.to_vec();
                let y = get_col_f64(&df, &v2)?.to_vec();
                let n = x.len().min(y.len());
                if n < 3 {
                    return Err(HayashiError::Runtime("spearman: n < 3".into()));
                }
                // Ranking with ties (average ranks)
                let rank = |vals: &[f64]| -> Vec<f64> {
                    let mut idx: Vec<usize> = (0..vals.len()).collect();
                    idx.sort_by(|&a, &b| {
                        vals[a]
                            .partial_cmp(&vals[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mut ranks = vec![0.0f64; vals.len()];
                    let mut i = 0;
                    while i < idx.len() {
                        let mut j = i + 1;
                        while j < idx.len() && (vals[idx[j]] - vals[idx[i]]).abs() < 1e-12 {
                            j += 1;
                        }
                        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-indexed
                        for k in i..j {
                            ranks[idx[k]] = avg_rank;
                        }
                        i = j;
                    }
                    ranks
                };
                let rx = rank(&x[..n]);
                let ry = rank(&y[..n]);
                let nf = n as f64;
                let mean_rx = rx.iter().sum::<f64>() / nf;
                let mean_ry = ry.iter().sum::<f64>() / nf;
                let num: f64 = rx
                    .iter()
                    .zip(ry.iter())
                    .map(|(a, b)| (a - mean_rx) * (b - mean_ry))
                    .sum();
                let dx: f64 = rx.iter().map(|a| (a - mean_rx).powi(2)).sum::<f64>();
                let dy: f64 = ry.iter().map(|b| (b - mean_ry).powi(2)).sum::<f64>();
                let rho = if dx * dy < 1e-15 {
                    0.0
                } else {
                    num / (dx * dy).sqrt()
                };
                let t_stat = rho * ((nf - 2.0) / (1.0 - rho * rho).max(1e-15)).sqrt();
                let p_val = greeners::t_pvalue_two(t_stat.abs(), nf - 2.0);
                println!("\n  Spearman ρ({v1}, {v2})");
                println!(
                    "  ρ = {rho:.6}   t = {t_stat:.4}   df = {}   p = {p_val:.4}",
                    n - 2
                );
                println!("  H₀: ρₛ = 0 (uncorrelated in ranks)");
                Ok(Value::Nil)
            }

            // ranksum(df, var, by=group) — Mann-Whitney U / Wilcoxon rank-sum
            "ranksum" | "mannwhitney" | "wilcoxon_rs" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ranksum(df, var, by=group_col)".into(),
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
                let df = match self.env.get(&df_name) {
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
                let by_name = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "ranksum requer by=coluna_grupo".into(),
                        ))
                    }
                };
                let y_col = get_col_f64(&df, &var_name)?;
                let grp_col = get_col_f64(&df, &by_name)?;
                let n_total = y_col.len();
                // Split into two groups by unique value
                let mut gvals: Vec<f64> = grp_col.to_vec();
                gvals.sort_by(nan_last_cmp);
                gvals.dedup();
                if gvals.len() != 2 {
                    return Err(self.rt_err(format!(
                        "ranksum: by= must have exactly 2 unique groups; found {}",
                        gvals.len()
                    )));
                }
                let g0: Vec<f64> = (0..n_total)
                    .filter(|&i| (grp_col[i] - gvals[0]).abs() < 1e-9)
                    .map(|i| y_col[i])
                    .collect();
                let g1: Vec<f64> = (0..n_total)
                    .filter(|&i| (grp_col[i] - gvals[1]).abs() < 1e-9)
                    .map(|i| y_col[i])
                    .collect();
                let n1 = g0.len();
                let n2 = g1.len();
                if n1 < 1 || n2 < 1 {
                    return Err(HayashiError::Runtime(
                        "ranksum: one of the groups is empty".into(),
                    ));
                }
                // Combined rank with ties
                let mut combined: Vec<(f64, usize)> = g0
                    .iter()
                    .map(|&v| (v, 0))
                    .chain(g1.iter().map(|&v| (v, 1)))
                    .collect();
                combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let m = combined.len();
                let mut ranks = vec![0.0f64; m];
                let mut i = 0;
                while i < m {
                    let mut j = i + 1;
                    while j < m && (combined[j].0 - combined[i].0).abs() < 1e-12 {
                        j += 1;
                    }
                    let avg_rank = (i + j + 1) as f64 / 2.0;
                    ranks[i..j].fill(avg_rank);
                    i = j;
                }
                let w1: f64 = (0..m)
                    .filter(|&k| combined[k].1 == 0)
                    .map(|k| ranks[k])
                    .sum();
                let u1 = w1 - (n1 * (n1 + 1)) as f64 / 2.0;
                let u2 = (n1 * n2) as f64 - u1;
                let u = u1.min(u2);
                // Normal approximation (large sample)
                let n1f = n1 as f64;
                let n2f = n2 as f64;
                let nf = m as f64;
                let mu_u = n1f * n2f / 2.0;
                let var_u = n1f * n2f * (nf + 1.0) / 12.0;
                let z_stat = (u - mu_u) / var_u.sqrt();
                // p-value via normal approximation
                let p_normal = 2.0 * (1.0 - norm_cdf(z_stat.abs()));
                println!("\n  Mann-Whitney U / Wilcoxon Rank-Sum");
                println!("  {}: n₁={n1}  {}: n₂={n2}", var_name, by_name);
                println!("  Group {}:  {var_name}", gvals[0] as i64);
                println!("  Group {}:  {var_name}", gvals[1] as i64);
                println!("  W (rank-sum group 0) = {w1:.1}");
                println!("  U₁ = {u1:.1}   U₂ = {u2:.1}   U = {u:.1}");
                println!("  z = {z_stat:.4}   p = {p_normal:.4}   (normal approx)");
                println!("  H₀: distribution of {var_name} equal in both groups");
                Ok(Value::Nil)
            }

            // kruskal(df, var, by=group) — Kruskal-Wallis (≥ 2 grupos)
            "kruskal" | "kwallis" | "kruskal_wallis" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "kruskal(df, var, by=group_col)".into(),
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
                let df = match self.env.get(&df_name) {
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
                let by_name = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "kruskal requer by=coluna_grupo".into(),
                        ))
                    }
                };
                let y_col = get_col_f64(&df, &var_name)?;
                let grp_col = get_col_f64(&df, &by_name)?;
                let n = y_col.len();
                let mut gvals: Vec<f64> = grp_col.to_vec();
                gvals.sort_by(nan_last_cmp);
                gvals.dedup();
                let k = gvals.len();
                if k < 2 {
                    return Err(HayashiError::Runtime(
                        "kruskal: precisa de pelo menos 2 grupos".into(),
                    ));
                }
                // Rank global com ties
                let mut indexed: Vec<(f64, usize)> = y_col
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(i, v)| (v, i))
                    .collect();
                indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut global_ranks = vec![0.0f64; n];
                let mut i = 0;
                let n_m = indexed.len();
                while i < n_m {
                    let mut j = i + 1;
                    while j < n_m && (indexed[j].0 - indexed[i].0).abs() < 1e-12 {
                        j += 1;
                    }
                    let avg_rank = (i + j + 1) as f64 / 2.0;
                    for k2 in i..j {
                        global_ranks[indexed[k2].1] = avg_rank;
                    }
                    i = j;
                }
                let nf = n as f64;
                let mean_rank = (nf + 1.0) / 2.0;
                // H statistic
                let h_num: f64 = gvals
                    .iter()
                    .map(|&gid| {
                        let idxs: Vec<usize> = (0..n)
                            .filter(|&i| (grp_col[i] - gid).abs() < 1e-9)
                            .collect();
                        let ni = idxs.len() as f64;
                        if ni == 0.0 {
                            return 0.0;
                        }
                        let rbar = idxs.iter().map(|&i| global_ranks[i]).sum::<f64>() / ni;
                        ni * (rbar - mean_rank).powi(2)
                    })
                    .sum();
                let h = 12.0 / (nf * (nf + 1.0)) * h_num;
                let df_kw = (k - 1) as f64;
                let p_val = greeners::chi2_pvalue(h, (k - 1) as f64);
                println!("\n  Kruskal-Wallis H");
                println!("  {var_name} por {by_name}  ({k} grupos, N={n})");
                for gid in &gvals {
                    let gdata: Vec<f64> = (0..n)
                        .filter(|&i| (grp_col[i] - gid).abs() < 1e-9)
                        .map(|i| y_col[i])
                        .collect();
                    let gn = gdata.len();
                    let gm = gdata.iter().sum::<f64>() / gn as f64;
                    let rbar = (0..n)
                        .filter(|&i| (grp_col[i] - gid).abs() < 1e-9)
                        .map(|i| global_ranks[i])
                        .sum::<f64>()
                        / gn as f64;
                    println!(
                        "    group {:>4}: n={gn:>4}  mean={gm:>8.4}  avg_rank={rbar:>8.2}",
                        *gid as i64
                    );
                }
                println!("  H = {h:.4}   df = {df_kw}   p = {p_val:.4}   χ² approx.");
                println!("  H₀: same distribution across all groups");
                Ok(Value::Nil)
            }

            // signrank(df, var, mu0=0) — Wilcoxon signed-rank (one sample or pairs)
            "signrank" | "wilcoxon_sr" | "wilcoxon_signed_rank" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "signrank(df, var, mu0=0)  or  signrank(df, d)  where d = x - y".into(),
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
                let df = match self.env.get(&df_name) {
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
                let mu0 = match opt_map.get("mu0") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.0,
                };
                let data = get_col_f64(&df, &var_name)?;
                let diffs: Vec<f64> = data
                    .iter()
                    .map(|&v| v - mu0)
                    .filter(|v| v.abs() > 1e-15)
                    .collect();
                let n = diffs.len();
                if n == 0 {
                    return Err(HayashiError::Runtime("signrank: all diffs are zero".into()));
                }
                // Rank of |diffs|
                let mut abs_indexed: Vec<(f64, usize, f64)> = diffs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| (d.abs(), i, d.signum()))
                    .collect();
                abs_indexed
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut ranks = vec![0.0f64; n];
                let mut i = 0;
                while i < n {
                    let mut j = i + 1;
                    while j < n && (abs_indexed[j].0 - abs_indexed[i].0).abs() < 1e-12 {
                        j += 1;
                    }
                    let avg_rank = (i + j + 1) as f64 / 2.0;
                    for k in i..j {
                        ranks[abs_indexed[k].1] = avg_rank;
                    }
                    i = j;
                }
                let w_plus: f64 = (0..n).filter(|&i| diffs[i] > 0.0).map(|i| ranks[i]).sum();
                let w_minus: f64 = (0..n).filter(|&i| diffs[i] < 0.0).map(|i| ranks[i]).sum();
                let w = w_plus.min(w_minus);
                // Normal approx (n ≥ 10)
                let nf = n as f64;
                let mu_w = nf * (nf + 1.0) / 4.0;
                let var_w = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0;
                let z_stat = (w - mu_w) / var_w.sqrt();
                let p_val = 2.0 * (1.0 - norm_cdf(z_stat.abs()));
                println!("\n  Wilcoxon Signed-Rank Test");
                println!("  H₀: mediana({var_name}) = {mu0}");
                println!("  n = {n}  (excluindo diffs ≈ 0)");
                println!("  W+ = {w_plus:.1}   W- = {w_minus:.1}   W = {w:.1}");
                println!("  z = {z_stat:.4}   p = {p_val:.4}   (normal approx)");
                Ok(Value::Nil)
            }

            // bitest(count, n, mu=0.5) — teste binomial (sinal)
            "bitest" | "signtest" | "binom_test" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "bitest(successes, n, mu=0.5)  ou  bitest(df, var, mu=0)".into(),
                    ));
                }
                // Suporta dois modos:
                // 1) bitest(count, n, mu=0.5)       — contagens diretas
                // 2) bitest(df, var, mu=0, by=group) — positivos vs negativos na coluna
                let first_val = self.eval_expr(&args[0])?;
                match first_val {
                    Value::Int(count) => {
                        // modo 1
                        let n_trials = match self.eval_expr(&args[1])? {
                            Value::Int(v) => v as usize,
                            Value::Float(v) => v as usize,
                            _ => return Err(HayashiError::Type("segundo arg: n (inteiro)".into())),
                        };
                        let mu = match opt_map.get("mu").or_else(|| opt_map.get("p")) {
                            Some(Value::Float(v)) => *v,
                            Some(Value::Int(v)) => *v as f64,
                            _ => 0.5,
                        };
                        let k = count as usize;
                        // p-value via normal approx (prop test)
                        let nf = n_trials as f64;
                        let phat = k as f64 / nf;
                        let se = (mu * (1.0 - mu) / nf).sqrt();
                        let z = (phat - mu) / se;
                        let p = 2.0 * (1.0 - norm_cdf(z.abs()));
                        println!("\n  Binomial / Sign Test");
                        println!(
                            "  Sucessos: {k}   n: {n_trials}   p̂ = {:.4}   H₀: p = {mu}",
                            phat
                        );
                        println!("  z = {z:.4}   p = {p:.4}");
                    }
                    Value::DataFrame(_) | Value::Nil => {
                        // Tentativa de modo 2: bitest(df, var, mu=0)
                        let df_name = match &args[0] {
                            Expr::Var(n) => n.clone(),
                            _ => {
                                return Err(HayashiError::Type(
                                    "first argument must be a DataFrame".into(),
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
                                    "second argument must be a variable name".into(),
                                ))
                            }
                        };
                        let mu0 = match opt_map.get("mu").or_else(|| opt_map.get("mu0")) {
                            Some(Value::Float(v)) => *v,
                            Some(Value::Int(v)) => *v as f64,
                            _ => 0.0,
                        };
                        let data = get_col_f64(&df, &var_name)?;
                        let pos = data.iter().filter(|&&v| v > mu0).count();
                        let neg = data.iter().filter(|&&v| v < mu0).count();
                        let ties = data.len() - pos - neg;
                        let n_eff = pos + neg;
                        let phat = pos as f64 / n_eff as f64;
                        let nf = n_eff as f64;
                        let z = (phat - 0.5) * nf.sqrt() / 0.5;
                        let p = 2.0 * (1.0 - norm_cdf(z.abs()));
                        println!("\n  Sign Test  ({var_name} vs {mu0})");
                        println!("  + : {pos}   - : {neg}   empates: {ties}   n efetivo: {n_eff}");
                        println!("  p̂(+) = {phat:.4}   z = {z:.4}   p = {p:.4}");
                        println!("  H₀: P(X > {mu0}) = 0.5");
                    }
                    _ => {
                        return Err(HayashiError::Type(
                            "bitest: first argument must be inteiro (count) ou DataFrame".into(),
                        ))
                    }
                }
                Ok(Value::Nil)
            }

            // ══════════════════════════════════════════════════════════════════
            // ── Unit-root tests ────────────────────────────────────────
            // ══════════════════════════════════════════════════════════════════

            // ══════════════════════════════════════════════════════════════════
            // ── Business-cycle filters ──────────────────────────────────
            // ══════════════════════════════════════════════════════════════════

            // hpfilter(df, var, lambda=1600)  →  cria df.var_trend e df.var_cycle
            "hpfilter" | "hp_filter" | "hprescott" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "hpfilter(df, var, lambda=1600)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "hpfilter: first argument must be a DataFrame".into(),
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
                            "hpfilter: second argument must be a variable name".into(),
                        ))
                    }
                };
                let lambda = match opt_map.get("lambda") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1600.0,
                };
                let series = ndarray::Array1::from(get_col_f64(&df, &var_name)?.to_vec());
                let (trend, cycle) = greeners::TimeSeries::hp_filter(&series, lambda)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let trend_name = format!("{var_name}_trend");
                let cycle_name = format!("{var_name}_cycle");
                Arc::make_mut(&mut df)
                    .insert(trend_name.clone(), trend)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                Arc::make_mut(&mut df)
                    .insert(cycle_name.clone(), cycle)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!(
                    "hpfilter: λ={lambda}  →  {trend_name} e {cycle_name} adicionadas a {df_name}"
                );
                self.env.set(&df_name, Value::DataFrame(df))?;
                Ok(Value::Nil)
            }

            // bkfilter(df, var, low=6, high=32, k=12)  →  cria df.var_cycle
            "bkfilter" | "bk_filter" | "baxter_king" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "bkfilter(df, var, low=6, high=32, k=12)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "bkfilter: first argument must be a DataFrame".into(),
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
                            "bkfilter: second argument must be a variable name".into(),
                        ))
                    }
                };
                let low = match opt_map.get("low") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 6,
                };
                let high = match opt_map.get("high") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 32,
                };
                let k = match opt_map.get("k") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };
                let series = ndarray::Array1::from(get_col_f64(&df, &var_name)?.to_vec());
                let cycle = greeners::TimeSeries::bk_filter(&series, low, high, k)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cycle_name = format!("{var_name}_cycle");
                Arc::make_mut(&mut df)
                    .insert(cycle_name.clone(), cycle)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!(
                    "bkfilter: periods [{low},{high}] k={k}  →  {cycle_name} added to {df_name}"
                );
                self.env.set(&df_name, Value::DataFrame(df))?;
                Ok(Value::Nil)
            }

            // cffilter(df, var, low=6, high=32, drift=false)  →  cria df.var_cycle
            "cffilter" | "cf_filter" | "christiano_fitzgerald" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "cffilter(df, var, low=6, high=32, drift=false)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "cffilter: first argument must be a DataFrame".into(),
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
                            "cffilter: second argument must be a variable name".into(),
                        ))
                    }
                };
                let low = match opt_map.get("low") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 6,
                };
                let high = match opt_map.get("high") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 32,
                };
                let drift = matches!(opt_map.get("drift"), Some(Value::Bool(true)));
                let series = ndarray::Array1::from(get_col_f64(&df, &var_name)?.to_vec());
                let cycle = greeners::TimeSeries::cf_filter(&series, low, high, drift)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let cycle_name = format!("{var_name}_cycle");
                Arc::make_mut(&mut df)
                    .insert(cycle_name.clone(), cycle)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!("cffilter: periods [{low},{high}] drift={drift}  →  {cycle_name} added to {df_name}");
                self.env.set(&df_name, Value::DataFrame(df))?;
                Ok(Value::Nil)
            }

            // ══════════════════════════════════════════════════════════════════
            // ── Penalized regression — Lasso / Ridge / ElasticNet ─────────────
            // ══════════════════════════════════════════════════════════════════

            // ridge(formula, df, alpha=1.0)
            // β_ridge = (X'X + αI)^{-1} X'y  (forma fechada)
            "ridge" | "ridge_reg" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ridge(formula, df, alpha=1.0)".into(),
                    ));
                }
                let formula = self.resolve_formula(&args[0])?;
                let df_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "ridge: second argument must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.0_f64,
                };
                let (df, gformula, _display) = self.prepare_formula(&formula, &df)?;
                let (y, x) = df
                    .to_design_matrix(&gformula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n = x.nrows();
                let k = x.ncols();
                // (X'X + αI)^{-1} X'y
                let xtx = x.t().dot(&x);
                let mut reg = xtx.clone();
                for i in 0..k {
                    reg[[i, i]] += alpha;
                }
                let reg_inv = reg
                    .inv()
                    .map_err(|_| HayashiError::Runtime("ridge: matriz singular".into()))?;
                let params = reg_inv.dot(&x.t().dot(&y));
                let y_hat = x.dot(&params);
                let resid: Vec<f64> = y.iter().zip(y_hat.iter()).map(|(a, b)| a - b).collect();
                let sse: f64 = resid.iter().map(|r| r * r).sum();
                let sst: f64 = {
                    let m = y.mean().unwrap_or(0.0);
                    y.iter().map(|v| (v - m).powi(2)).sum()
                };
                let r2 = 1.0 - sse / sst;
                let var_names: Vec<String> = gformula.independents.clone();
                let mut display_names = Vec::new();
                if gformula.intercept && var_names.len() < params.len() {
                    display_names.push("const".to_string());
                    display_names.extend(var_names.iter().cloned());
                } else {
                    display_names = var_names.clone();
                }
                println!("\n{:=^60}", " Ridge Regression ");
                println!(
                    "  Formula: {}   α = {alpha}",
                    Self::formula_to_string(&formula)
                );
                println!("  n = {n}   k = {k}   R² = {r2:.4}");
                println!("\n  {:<20} {:>12}", "Variable", "Coeff");
                println!("  {}", "─".repeat(33));
                for (i, name) in display_names.iter().enumerate() {
                    println!("  {:<20} {:>12.6}", name, params[i]);
                }
                let std_errors = ndarray::Array1::<f64>::zeros(params.len());
                Ok(Value::PenalizedResult(PenalizedModel {
                    params,
                    std_errors,
                    variable_names: display_names,
                    r_squared: r2,
                    n_obs: n,
                    alpha,
                    l1_ratio: None,
                    kind: "ridge".to_string(),
                }))
            }

            // lasso(formula, df, alpha=1.0, tol=1e-6, max_iter=10000)
            // Coordinate descent for Lasso (L1), with unpenalized intercept
            "lasso" | "lasso_reg" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "lasso(formula, df, alpha=1.0, tol=1e-6, max_iter=10000)".into(),
                    ));
                }
                let formula = self.resolve_formula(&args[0])?;
                let df_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "lasso: second argument must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.0_f64,
                };
                let tol = match opt_map.get("tol") {
                    Some(Value::Float(v)) => *v,
                    _ => 1e-6_f64,
                };
                let max_iter = match opt_map.get("max_iter") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10_000usize,
                };
                let (df, gformula, _display) = self.prepare_formula(&formula, &df)?;
                let (y, x) = df
                    .to_design_matrix(&gformula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n_obs = x.nrows();
                let k = x.ncols();
                let has_intercept = gformula.intercept;
                let start_col = if has_intercept { 1 } else { 0 };
                // Center y; standardize X (covariate columns)
                let y_mean = y.sum() / n_obs as f64;
                let y_c: ndarray::Array1<f64> = y.mapv(|v| v - y_mean);
                let mut x_std = x.clone();
                let mut col_mean: Vec<f64> = vec![0.0; k];
                let mut col_std_v: Vec<f64> = vec![1.0; k];
                for j in start_col..k {
                    let col = x.column(j);
                    let mj = col.sum() / n_obs as f64;
                    let sj =
                        (col.iter().map(|&v| (v - mj).powi(2)).sum::<f64>() / n_obs as f64).sqrt();
                    col_mean[j] = mj;
                    if sj > 1e-12 {
                        col_std_v[j] = sj;
                        for i in 0..n_obs {
                            x_std[[i, j]] = (x[[i, j]] - mj) / sj;
                        }
                    }
                }
                let soft = |z: f64, lam: f64| -> f64 {
                    if z > lam {
                        z - lam
                    } else if z < -lam {
                        z + lam
                    } else {
                        0.0
                    }
                };
                let xx_diag: Vec<f64> = (start_col..k)
                    .map(|j| x_std.column(j).dot(&x_std.column(j)))
                    .collect();
                // coordinate descent without intercept (y already centered)
                let mut beta = ndarray::Array1::<f64>::zeros(k - start_col);
                for _iter in 0..max_iter {
                    let mut max_delta = 0.0_f64;
                    let xb: ndarray::Array1<f64> = {
                        let mut v = ndarray::Array1::<f64>::zeros(n_obs);
                        for j in 0..beta.len() {
                            let col_j = x_std.column(start_col + j);
                            v = v + col_j.mapv(|x| x * beta[j]);
                        }
                        v
                    };
                    let r = &y_c - &xb;
                    for j in 0..beta.len() {
                        let denom = xx_diag[j];
                        if denom < 1e-12 {
                            continue;
                        }
                        let rho_j = r.dot(&x_std.column(start_col + j)) + denom * beta[j];
                        let new_b = soft(rho_j / denom, alpha * n_obs as f64 / denom);
                        let delta = (new_b - beta[j]).abs();
                        if delta > max_delta {
                            max_delta = delta;
                        }
                        beta[j] = new_b;
                    }
                    if max_delta < tol {
                        break;
                    }
                }
                // destandardizar: β_j_orig = β_j_std / std_j
                // intercept: intercept = y_mean - sum_j(β_j_orig * mean_j)
                let mut params = ndarray::Array1::<f64>::zeros(k);
                for j in 0..beta.len() {
                    params[start_col + j] = beta[j] / col_std_v[start_col + j];
                }
                if has_intercept {
                    params[0] =
                        y_mean - (start_col..k).map(|j| params[j] * col_mean[j]).sum::<f64>();
                }
                let y_hat = x.dot(&params);
                let sse: f64 = y
                    .iter()
                    .zip(y_hat.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let sst: f64 = {
                    let m = y.mean().unwrap_or(0.0);
                    y.iter().map(|v| (v - m).powi(2)).sum()
                };
                let r2 = 1.0 - sse / sst;
                let n_nonzero = params
                    .iter()
                    .skip(start_col)
                    .filter(|&&v| v.abs() > 1e-10)
                    .count();
                let var_names = &gformula.independents;
                let mut display_names = Vec::new();
                if gformula.intercept {
                    display_names.push("const".to_string());
                }
                display_names.extend(var_names.iter().cloned());
                println!("\n{:=^60}", " Lasso Regression ");
                println!(
                    "  Formula: {}   α = {alpha}",
                    Self::formula_to_string(&formula)
                );
                println!(
                    "  n = {}   k = {}   R² = {r2:.4}   vars ativas: {n_nonzero}",
                    x.nrows(),
                    k
                );
                println!("\n  {:<20} {:>12}", "Variable", "Coeff");
                println!("  {}", "─".repeat(33));
                for (i, name) in display_names.iter().enumerate() {
                    println!("  {:<20} {:>12.6}", name, params[i]);
                }
                let std_errors = ndarray::Array1::<f64>::zeros(params.len());
                Ok(Value::PenalizedResult(PenalizedModel {
                    params,
                    std_errors,
                    variable_names: display_names,
                    r_squared: r2,
                    n_obs: x.nrows(),
                    alpha,
                    l1_ratio: None,
                    kind: "lasso".to_string(),
                }))
            }

            // elasticnet(formula, df, alpha=1.0, l1_ratio=0.5, ...)
            // Combina L1 e L2: penalty = l1_ratio*α*|β| + (1-l1_ratio)*α/2*β²
            "elasticnet" | "elastic_net" | "enet" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "elasticnet(formula, df, alpha=1.0, l1_ratio=0.5)".into(),
                    ));
                }
                let formula = self.resolve_formula(&args[0])?;
                let df_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "elasticnet: second argument must be DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.0_f64,
                };
                let l1_ratio = match opt_map.get("l1_ratio") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.5_f64,
                };
                let tol = match opt_map.get("tol") {
                    Some(Value::Float(v)) => *v,
                    _ => 1e-6_f64,
                };
                let max_iter = match opt_map.get("max_iter") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10_000usize,
                };
                let (df, gformula, _display) = self.prepare_formula(&formula, &df)?;
                let (y, x) = df
                    .to_design_matrix(&gformula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n_obs = x.nrows();
                let k = x.ncols();
                let has_intercept = gformula.intercept;
                let start_col = if has_intercept { 1 } else { 0 };
                let y_mean = y.sum() / n_obs as f64;
                let y_c: ndarray::Array1<f64> = y.mapv(|v| v - y_mean);
                let mut x_std = x.clone();
                let mut col_mean: Vec<f64> = vec![0.0; k];
                let mut col_std_v: Vec<f64> = vec![1.0; k];
                for j in start_col..k {
                    let col = x.column(j);
                    let mj = col.sum() / n_obs as f64;
                    let sj =
                        (col.iter().map(|&v| (v - mj).powi(2)).sum::<f64>() / n_obs as f64).sqrt();
                    col_mean[j] = mj;
                    if sj > 1e-12 {
                        col_std_v[j] = sj;
                        for i in 0..n_obs {
                            x_std[[i, j]] = (x[[i, j]] - mj) / sj;
                        }
                    }
                }
                let l1 = alpha * l1_ratio;
                let l2 = alpha * (1.0 - l1_ratio);
                let soft = |z: f64, lam: f64| -> f64 {
                    if z > lam {
                        z - lam
                    } else if z < -lam {
                        z + lam
                    } else {
                        0.0
                    }
                };
                let xx_diag: Vec<f64> = (start_col..k)
                    .map(|j| x_std.column(j).dot(&x_std.column(j)))
                    .collect();
                let mut beta = ndarray::Array1::<f64>::zeros(k - start_col);
                for _iter in 0..max_iter {
                    let mut max_delta = 0.0_f64;
                    let xb: ndarray::Array1<f64> = {
                        let mut v = ndarray::Array1::<f64>::zeros(n_obs);
                        for j in 0..beta.len() {
                            let c = x_std.column(start_col + j);
                            v = v + c.mapv(|x| x * beta[j]);
                        }
                        v
                    };
                    let r = &y_c - &xb;
                    for j in 0..beta.len() {
                        let denom = xx_diag[j] + l2 * n_obs as f64;
                        if denom < 1e-12 {
                            continue;
                        }
                        let rho_j = r.dot(&x_std.column(start_col + j)) + xx_diag[j] * beta[j];
                        let new_b = soft(rho_j / denom, l1 * n_obs as f64 / denom);
                        let delta = (new_b - beta[j]).abs();
                        if delta > max_delta {
                            max_delta = delta;
                        }
                        beta[j] = new_b;
                    }
                    if max_delta < tol {
                        break;
                    }
                }
                let mut params = ndarray::Array1::<f64>::zeros(k);
                for j in 0..beta.len() {
                    params[start_col + j] = beta[j] / col_std_v[start_col + j];
                }
                if has_intercept {
                    params[0] =
                        y_mean - (start_col..k).map(|j| params[j] * col_mean[j]).sum::<f64>();
                }
                let y_hat = x.dot(&params);
                let sse: f64 = y
                    .iter()
                    .zip(y_hat.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let sst: f64 = {
                    let m = y.mean().unwrap_or(0.0);
                    y.iter().map(|v| (v - m).powi(2)).sum()
                };
                let r2 = 1.0 - sse / sst;
                let n_nonzero = params
                    .iter()
                    .skip(start_col)
                    .filter(|&&v| v.abs() > 1e-10)
                    .count();
                let var_names = &gformula.independents;
                let mut display_names = Vec::new();
                if gformula.intercept {
                    display_names.push("const".to_string());
                }
                display_names.extend(var_names.iter().cloned());
                println!("\n{:=^60}", " ElasticNet Regression ");
                println!(
                    "  Formula: {}   α={alpha}   l1_ratio={l1_ratio}",
                    Self::formula_to_string(&formula)
                );
                println!(
                    "  n={}  k={}  R²={r2:.4}  vars ativas: {n_nonzero}",
                    x.nrows(),
                    k
                );
                println!("\n  {:<20} {:>12}", "Variable", "Coeff");
                println!("  {}", "─".repeat(33));
                for (i, name) in display_names.iter().enumerate() {
                    println!("  {:<20} {:>12.6}", name, params[i]);
                }
                let std_errors = ndarray::Array1::<f64>::zeros(params.len());
                Ok(Value::PenalizedResult(PenalizedModel {
                    params,
                    std_errors,
                    variable_names: display_names,
                    r_squared: r2,
                    n_obs: x.nrows(),
                    alpha,
                    l1_ratio: Some(l1_ratio),
                    kind: "elasticnet".to_string(),
                }))
            }

            // adf(df, var, lags=N)
            "adf" | "dickey_fuller" | "augmented_df" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("adf(df, var, lags=N)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "adf: first argument must be a DataFrame".into(),
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
                            "adf: second argument must be a variable name".into(),
                        ))
                    }
                };
                let series = get_col_f64(&df, &var_name)?;
                let max_lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let arr = ndarray::Array1::from(series.to_vec());
                let r = greeners::TimeSeries::adf(&arr, max_lags)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Augmented Dickey-Fuller Test ");
                println!("  Variable: {var_name}   Lags used: {}", r.lags_used);
                println!("  H₀: series has a unit root (non-stationary)");
                println!("  Test statistic:  {:>10.4}", r.test_statistic);
                if let Some(p) = r.p_value {
                    println!("  p-value:         {:>10.4}", p);
                }
                let (cv1, cv5, cv10) = r.critical_values;
                println!("  Critical values:  1%={cv1:.3}  5%={cv5:.3}  10%={cv10:.3}");
                println!(
                    "  Conclusion: {}",
                    if r.is_stationary {
                        "REJECT H₀ — stationary"
                    } else {
                        "Does not reject H₀ — unit root present"
                    }
                );
                Ok(Value::Nil)
            }

            // kpss(df, var, regression=c, lags=N)
            "kpss" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "kpss(df, var, regression=c|ct, lags=N)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kpss: first argument must be a DataFrame".into(),
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
                            "kpss: second argument must be a variable name".into(),
                        ))
                    }
                };
                let series = get_col_f64(&df, &var_name)?;
                let regression = match opt_map.get("regression") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "c".to_string(),
                };
                let max_lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let arr = ndarray::Array1::from(series.to_vec());
                let r = greeners::TimeSeries::kpss(&arr, &regression, max_lags)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " KPSS Test ");
                println!(
                    "  Variable: {var_name}   Regression: {}   Lags: {}",
                    r.regression, r.lags_used
                );
                println!("  H₀: series is stationary");
                println!("  Test statistic:  {:>10.4}", r.test_statistic);
                let (cv10, cv5, cv25, cv1) = r.critical_values;
                println!(
                    "  Critical values:  10%={cv10:.3}  5%={cv5:.3}  2.5%={cv25:.3}  1%={cv1:.3}"
                );
                println!(
                    "  Conclusion: {}",
                    if r.is_stationary {
                        "Does not reject H₀ — stationary"
                    } else {
                        "REJECT H₀ — non-stationary"
                    }
                );
                Ok(Value::Nil)
            }

            // pp(df, var, lags=N)
            "pp" | "phillips_perron" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("pp(df, var, lags=N)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "pp: first argument must be a DataFrame".into(),
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
                            "pp: second argument must be a variable name".into(),
                        ))
                    }
                };
                let series = get_col_f64(&df, &var_name)?;
                let max_lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => Some(*v as usize),
                    Some(Value::Float(v)) => Some(*v as usize),
                    _ => None,
                };
                let arr = ndarray::Array1::from(series.to_vec());
                let r = greeners::TimeSeries::phillips_perron(&arr, max_lags)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Phillips-Perron Test ");
                println!("  Variable: {var_name}   Lags used: {}", r.lags_used);
                println!("  H₀: series has a unit root (non-stationary)");
                println!("  Zα statistic:    {:>10.4}", r.z_alpha);
                println!("  Zt statistic:    {:>10.4}", r.z_t);
                let (cv1, cv5, cv10) = r.critical_values;
                println!("  Critical values:  1%={cv1:.3}  5%={cv5:.3}  10%={cv10:.3}");
                println!(
                    "  Conclusion: {}",
                    if r.is_stationary {
                        "REJECT H₀ — stationary"
                    } else {
                        "Does not reject H₀ — unit root present"
                    }
                );
                Ok(Value::Nil)
            }

            // za(df, var, trim=0.15)
            "za" | "zivot_andrews" | "zivot" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("za(df, var, trim=0.15)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "za: first argument must be a DataFrame".into(),
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
                            "za: second argument must be a variable name".into(),
                        ))
                    }
                };
                let series = get_col_f64(&df, &var_name)?;
                let trim = match opt_map.get("trim") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.15,
                };
                let arr = ndarray::Array1::from(series.to_vec());
                let r = greeners::TimeSeries::zivot_andrews(&arr, trim)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Zivot-Andrews Test ");
                println!("  Variable: {var_name}");
                println!("  H₀: unit root (sem quebra estrutural)");
                println!("  Test statistic:  {:>10.4}", r.statistic);
                println!("  Break point:     obs {}", r.break_point);
                let (cv1, cv5, cv10) = r.critical_values;
                println!("  Critical values:  1%={cv1:.3}  5%={cv5:.3}  10%={cv10:.3}");
                println!(
                    "  Conclusion: {}",
                    if r.is_stationary {
                        "REJECT H₀ — stationary with break"
                    } else {
                        "Does not reject H₀ — unit root"
                    }
                );
                Ok(Value::Nil)
            }

            // ── Cointegration ──────────────────────────────────────────────────

            // granger(df, y, x, lags=N)
            "granger" | "granger_causality" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("granger(df, y, x, lags=N)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "granger: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let y_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "granger: second argument must be a variable name".into(),
                        ))
                    }
                };
                let x_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "granger: third argument must be variable name".into(),
                        ))
                    }
                };
                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 4,
                };
                let y_arr = ndarray::Array1::from(get_col_f64(&df, &y_name)?.to_vec());
                let x_arr = ndarray::Array1::from(get_col_f64(&df, &x_name)?.to_vec());
                let r = greeners::TimeSeries::granger_causality(&y_arr, &x_arr, lags)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Granger Causality Test ");
                println!("  H₀: {x_name} does not Granger-cause {y_name}   (lags={lags})");
                println!(
                    "  F({}, {}) = {:.4}   p = {:.4}",
                    r.df_num, r.df_denom, r.f_statistic, r.p_value
                );
                println!(
                    "  Conclusion: {}",
                    if r.p_value < 0.05 {
                        format!("REJECT H₀ — {x_name} Granger-causes {y_name}")
                    } else {
                        "Does not reject H₀".to_string()
                    }
                );
                Ok(Value::Nil)
            }

            // engle_granger(df, y1, y2)
            "engle_granger" | "coint" | "egtest" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("engle_granger(df, y1, y2)".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "engle_granger: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let y1_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "engle_granger: second argument must be a variable name".into(),
                        ))
                    }
                };
                let y2_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "engle_granger: third argument must be variable name".into(),
                        ))
                    }
                };
                let y1_arr = ndarray::Array1::from(get_col_f64(&df, &y1_name)?.to_vec());
                let y2_arr = ndarray::Array1::from(get_col_f64(&df, &y2_name)?.to_vec());
                let r = greeners::TimeSeries::engle_granger(&y1_arr, &y2_arr)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Engle-Granger Cointegration Test ");
                println!("  Variables: {y1_name}, {y2_name}");
                println!("  H₀: no cointegration");
                println!("  ADF statistic: {:>10.4}", r.adf_statistic);
                let (cv1, cv5, cv10) = r.critical_values;
                println!("  Critical values:  1%={cv1:.3}  5%={cv5:.3}  10%={cv10:.3}");
                let coef = &r.cointegrating_vector;
                if coef.len() >= 2 {
                    println!("  Vetor cointegrante: [{:.4}, {:.4}]", coef[0], coef[1]);
                }
                println!(
                    "  Conclusion: {}",
                    if r.is_cointegrated {
                        "REJECT H₀ — cointegrated series"
                    } else {
                        "Does not reject H₀ — no cointegration"
                    }
                );
                Ok(Value::Nil)
            }

            // johansen(df, [var1, var2, ...], lags=N, det=0)
            "johansen" | "johansen_trace" | "vecrank" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "johansen(df, [var1, var2, ...], lags=N, det=0)".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "johansen: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let var_names: Vec<String> = match self.eval_expr(&args[1])? {
                    Value::List(lst) => lst.iter().map(|v| format!("{v}")).collect(),
                    _ => {
                        return Err(HayashiError::Type(
                            "johansen: second argument must be list of variables".into(),
                        ))
                    }
                };
                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let det = match opt_map.get("det") {
                    Some(Value::Int(v)) => *v as i32,
                    Some(Value::Float(v)) => *v as i32,
                    _ => 0i32,
                };
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, name) in var_names.iter().enumerate() {
                    let col = get_col_f64(&df, name)?;
                    for i in 0..n {
                        data[[i, j]] = col[i];
                    }
                }
                let r = greeners::TimeSeries::johansen(&data, lags, det)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("\n{:=^60}", " Johansen Cointegration Test ");
                println!(
                    "  Variables: {}   Lags: {lags}   Det order: {det}",
                    var_names.join(", ")
                );
                println!("  Cointegrating rank: {}", r.cointegrating_rank);
                println!("\n  Trace Test:");
                println!(
                    "  {:>4}  {:>12}  {:>10}  {:>10}  {:>10}",
                    "Rank", "Trace stat", "10%", "5%", "1%"
                );
                for i in 0..r.n_vars {
                    let stat = r.trace_stats[i];
                    let cv = r.trace_critical_values.row(i);
                    println!(
                        "  {:>4}  {:>12.4}  {:>10.4}  {:>10.4}  {:>10.4}",
                        i, stat, cv[0], cv[1], cv[2]
                    );
                }
                println!("\n  Max-Eigenvalue Test:");
                println!(
                    "  {:>4}  {:>12}  {:>10}  {:>10}  {:>10}",
                    "Rank", "Max-eig", "10%", "5%", "1%"
                );
                for i in 0..r.n_vars {
                    let stat = r.max_eigen_stats[i];
                    let cv = r.max_eigen_critical_values.row(i);
                    println!(
                        "  {:>4}  {:>12.4}  {:>10.4}  {:>10.4}  {:>10.4}",
                        i, stat, cv[0], cv[1], cv[2]
                    );
                }
                Ok(Value::Nil)
            }

            // ══════════════════════════════════════════════════════════════════
            // ── xtset: declare panel structure ────────────────────────────
            // xtset(df, id_col, time_col)  — stores in panel_info
            // After xtset, fe/re/ab/etc. do not need id= and time=
            "xtset" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "xtset(df, id_col, time_col)  or  xtset(df, time_col) for time series"
                            .into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be nome do DataFrame".into(),
                        ))
                    }
                };
                match self.env.get(&df_name) {
                    Some(Value::DataFrame(_)) => {}
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let id_col = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "id_col deve ser identificador ou string".into(),
                        ))
                    }
                };
                let time_col = if args.len() >= 3 {
                    match &args[2] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(HayashiError::Type(
                                "time_col deve ser identificador ou string".into(),
                            ))
                        }
                    }
                } else {
                    String::new()
                };
                self.panel_info
                    .insert(df_name.clone(), (id_col.clone(), time_col.clone()));
                if time_col.is_empty() {
                    println!("xtset {df_name}  (time series: t={id_col})");
                } else {
                    println!("xtset {df_name}  id={id_col}  time={time_col}");
                }
                Ok(Value::Nil)
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
