use super::helpers::*;
use super::*;
use crate::lang::dap::model_expansion;

/// ASCII visualization (acfplot/pacf/qqplot/corrplot/scatter/histogram/boxplot/kdensity)
/// and coefplot. Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_visualization(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── ASCII visualization — ACF / PACF / QQ-plot / heatmap ──────────

            // acfplot(df, var, lags=20, width=50, title="")
            "acfplot" | "acf_plot" | "corrgram" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "acfplot(df, var, lags=20, width=50)".into(),
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
                let max_lag = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };
                let width = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 50,
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("ACF — {var_name}"),
                };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                let display = ascii_acf(&clean, max_lag, &title, width, false);
                let plot_lag = max_lag.min(clean.len() / 2);
                let summary = format!("{title} ({plot_lag} lags)");
                let fields = vec![
                    ("var".into(), Value::Str(var_name.clone())),
                    ("n".into(), Value::Int(clean.len() as i64)),
                    ("max_lag".into(), Value::Int(plot_lag as i64)),
                    ("width".into(), Value::Int(width as i64)),
                    ("title".into(), Value::Str(title.clone())),
                    ("partial".into(), Value::Bool(false)),
                    (
                        "data".into(),
                        model_expansion::series_from_vec("data", &clean),
                    ),
                    (
                        "lags".into(),
                        model_expansion::int_series("lags", &(1..=plot_lag).collect::<Vec<_>>()),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display, summary, "ACFPlot", fields,
                ))
            }

            // pacfplot(df, var, lags=20, width=50, title="")
            "pacfplot" | "pacf_plot" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "pacfplot(df, var, lags=20, width=50)".into(),
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
                let max_lag = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };
                let width = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 50,
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("PACF — {var_name}"),
                };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                let display = ascii_acf(&clean, max_lag, &title, width, true);
                let plot_lag = max_lag.min(clean.len() / 2);
                let summary = format!("{title} ({plot_lag} lags)");
                let fields = vec![
                    ("var".into(), Value::Str(var_name.clone())),
                    ("n".into(), Value::Int(clean.len() as i64)),
                    ("max_lag".into(), Value::Int(plot_lag as i64)),
                    ("width".into(), Value::Int(width as i64)),
                    ("title".into(), Value::Str(title.clone())),
                    ("partial".into(), Value::Bool(true)),
                    (
                        "data".into(),
                        model_expansion::series_from_vec("data", &clean),
                    ),
                    (
                        "lags".into(),
                        model_expansion::int_series("lags", &(1..=plot_lag).collect::<Vec<_>>()),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display, summary, "PACFPlot", fields,
                ))
            }

            // qqplot(df, var, width=50, height=20, dist="normal", title="")
            "qqplot" | "qnorm" | "pnorm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "qqplot(df, var, width=50, height=20)".into(),
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
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 50,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("QQ-plot normal — {var_name}"),
                };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                let display = ascii_qqplot(&clean, &title, &var_name, w, h);
                let summary = format!("QQ-plot normal — {var_name}");
                let fields = vec![
                    ("var".into(), Value::Str(var_name.clone())),
                    ("n".into(), Value::Int(clean.len() as i64)),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("title".into(), Value::Str(title.clone())),
                    (
                        "data".into(),
                        model_expansion::series_from_vec("data", &clean),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display, summary, "QQPlot", fields,
                ))
            }

            // corrplot(df, var1, var2, ...) — ASCII correlation matrix
            "corrplot" | "corr_heatmap" | "pwcorr_plot" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "corrplot(df, var1, var2, ...)".into(),
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
                let var_names: Vec<String> = {
                    let mut v = Vec::new();
                    for a in &args[1..] {
                        match a {
                            Expr::Var(n) | Expr::Str(n) => v.push(n.clone()),
                            _ => {
                                return Err(HayashiError::Type(
                                    "args must be variable names".into(),
                                ))
                            }
                        }
                    }
                    v
                };
                if var_names.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "corrplot: provide at least 2 variables".into(),
                    ));
                }
                let cols: Vec<Vec<f64>> = {
                    let mut v = Vec::new();
                    for n in &var_names {
                        v.push(get_col_f64(&df, n)?.to_vec());
                    }
                    v
                };
                let display = ascii_corrplot(&cols, &var_names);

                let n = cols[0].len();
                let means: Vec<f64> = cols
                    .iter()
                    .map(|c| c.iter().sum::<f64>() / n as f64)
                    .collect();
                let corr: Vec<Vec<f64>> = cols
                    .iter()
                    .enumerate()
                    .map(|(i, col_i)| {
                        let xi: Vec<f64> = col_i.iter().map(|x| x - means[i]).collect();
                        let di = xi.iter().map(|a| a * a).sum::<f64>().sqrt();
                        cols.iter()
                            .enumerate()
                            .map(|(j, col_j)| {
                                let xj: Vec<f64> = col_j.iter().map(|x| x - means[j]).collect();
                                let num: f64 = xi.iter().zip(&xj).map(|(a, b)| a * b).sum();
                                let dj = xj.iter().map(|b| b * b).sum::<f64>().sqrt();
                                if di * dj < 1e-15 {
                                    0.0
                                } else {
                                    num / (di * dj)
                                }
                            })
                            .collect()
                    })
                    .collect();
                let k = var_names.len();
                let corr_flat: Vec<f64> = corr.iter().flat_map(|r| r.iter().copied()).collect();
                let corr_arr =
                    ndarray::Array2::from_shape_vec((k, k), corr_flat).unwrap_or_else(|_| {
                        ndarray::Array2::from_shape_vec((k, k), vec![0.0; k * k]).unwrap()
                    });
                let correlation = model_expansion::array2_to_dataframe_named(&corr_arr, &var_names);

                let summary = format!("Correlation matrix ({} variables)", k);
                let fields = vec![
                    ("n".into(), Value::Int(n as i64)),
                    (
                        "variables".into(),
                        Value::List(Arc::new(
                            var_names.iter().map(|v| Value::Str(v.clone())).collect(),
                        )),
                    ),
                    ("correlation".into(), correlation),
                ];
                Ok(model_expansion::model_result(
                    display, summary, "Corrplot", fields,
                ))
            }

            // ── ASCII visualization ────────────────────────────────────────────

            // histogram(df, var, bins=20, width=50, title="")
            "histogram" | "hist" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "histogram(df, var, bins=20, width=50)".into(),
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
                let bins = match opt_map.get("bins") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };
                let width = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 50,
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("Histogram — {var_name}"),
                };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                let display = ascii_histogram(&clean, bins, &title, &var_name, width);
                let summary = format!("Histogram of {var_name}");
                let fields = vec![
                    ("var".into(), Value::Str(var_name.clone())),
                    ("n".into(), Value::Int(clean.len() as i64)),
                    ("bins".into(), Value::Int(bins as i64)),
                    ("width".into(), Value::Int(width as i64)),
                    ("title".into(), Value::Str(title.clone())),
                    (
                        "data".into(),
                        model_expansion::series_from_vec("data", &clean),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "HistogramPlot",
                    fields,
                ))
            }

            // scatter(df, x, y, width=60, height=20, title="")
            "scatter" | "scatterplot" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "scatter(df, x, y, width=60, height=20)".into(),
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
                let xname = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be a variable name (x)".into(),
                        ))
                    }
                };
                let yname = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "third argument must be variable name (y)".into(),
                        ))
                    }
                };
                let xs = get_col_f64(&df, &xname)?;
                let ys = get_col_f64(&df, &yname)?;
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 60,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("{yname} vs {xname}"),
                };
                let display =
                    ascii_scatter(&xs.to_vec(), &ys.to_vec(), &title, &xname, &yname, w, h);
                let summary = format!("{title} (n={})", xs.len());
                let fields = vec![
                    ("x".into(), Value::Str(xname.clone())),
                    ("y".into(), Value::Str(yname.clone())),
                    ("n".into(), Value::Int(xs.len() as i64)),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("title".into(), Value::Str(title.clone())),
                    (
                        "xs".into(),
                        model_expansion::series_from_vec("xs", &xs.to_vec()),
                    ),
                    (
                        "ys".into(),
                        model_expansion::series_from_vec("ys", &ys.to_vec()),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "ScatterPlot",
                    fields,
                ))
            }

            // lineplot(df, x, y, width=60, height=20, title="")
            "lineplot" | "tsplot" | "twoway" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "lineplot(df, x, y, width=60, height=20)".into(),
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
                let xname = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "second argument must be a variable name (x/time)".into(),
                        ))
                    }
                };
                let yname = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "third argument must be variable name (y)".into(),
                        ))
                    }
                };
                let xs = get_col_f64(&df, &xname)?;
                let ys = get_col_f64(&df, &yname)?;
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 60,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("{yname} — time series"),
                };
                let display =
                    ascii_lineplot(&xs.to_vec(), &ys.to_vec(), &title, &xname, &yname, w, h);
                let summary = format!("{title} (n={})", xs.len());
                let fields = vec![
                    ("x".into(), Value::Str(xname.clone())),
                    ("y".into(), Value::Str(yname.clone())),
                    ("n".into(), Value::Int(xs.len() as i64)),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("title".into(), Value::Str(title.clone())),
                    (
                        "xs".into(),
                        model_expansion::series_from_vec("xs", &xs.to_vec()),
                    ),
                    (
                        "ys".into(),
                        model_expansion::series_from_vec("ys", &ys.to_vec()),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display, summary, "LinePlot", fields,
                ))
            }

            // boxplot(df, var, width=60, title="")
            "boxplot" | "box" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("boxplot(df, var, width=60)".into()));
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
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 60,
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("Boxplot — {var_name}"),
                };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                let display = ascii_boxplot(&clean, &title, &var_name, w);
                let summary = format!("Boxplot of {var_name}");
                let fields = vec![
                    ("var".into(), Value::Str(var_name.clone())),
                    ("n".into(), Value::Int(clean.len() as i64)),
                    ("width".into(), Value::Int(w as i64)),
                    ("title".into(), Value::Str(title.clone())),
                    (
                        "data".into(),
                        model_expansion::series_from_vec("data", &clean),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display, summary, "Boxplot", fields,
                ))
            }

            // kdensity(df, var, width=60, height=20) — KDE via ascii_scatter dos pontos da densidade
            "kdensity" | "density" | "densityplot" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "kdensity(df, var, width=60, height=20)".into(),
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
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 60,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };
                // bw= option or Silverman
                let bw_opt = match opt_map.get("bw") {
                    Some(Value::Float(v)) => Some(*v),
                    Some(Value::Int(v)) => Some(*v as f64),
                    _ => None,
                };
                let kernel = match opt_map.get("kernel") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "epanechnikov" | "epa" => greeners::Kernel::Epanechnikov,
                        "triangular" => greeners::Kernel::Triangular,
                        "uniform" => greeners::Kernel::Uniform,
                        _ => greeners::Kernel::Gaussian,
                    },
                    _ => greeners::Kernel::Gaussian,
                };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                let n = clean.len();
                if n < 4 {
                    return Err(HayashiError::Runtime("kdensity: too few data".into()));
                }
                // Estimate KDE — uses support/density already calculated in fit (512 points)
                let result = greeners::KDEUnivariate::fit(
                    &ndarray::Array1::from(clean.clone()),
                    bw_opt,
                    kernel,
                )
                .map_err(|e| self.rt_err(format!("kdensity: {e}")))?;
                let xs: Vec<f64> = result.support.to_vec();
                let ys: Vec<f64> = result.density.to_vec();
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("KDE — {var_name}  (bw={:.4})", result.bandwidth),
                };
                let display = ascii_lineplot(&xs, &ys, &title, &var_name, "density", w, h);
                let summary = title.clone();
                let fields = vec![
                    ("var".into(), Value::Str(var_name.clone())),
                    ("n".into(), Value::Int(n as i64)),
                    ("bw".into(), Value::Float(result.bandwidth)),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("title".into(), Value::Str(title)),
                    ("xs".into(), model_expansion::series_from_vec("xs", &xs)),
                    ("ys".into(), model_expansion::series_from_vec("ys", &ys)),
                ];
                Ok(model_expansion::model_result(
                    display, summary, "KDEPlot", fields,
                ))
            }

            // residplot(model, width=60, height=20) — residuals vs ŷ
            "residplot" | "rvfplot" | "resid_plot" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "residplot(model, width=60, height=20)".into(),
                    ));
                }
                let (fitted, resids, mname) =
                    match self.eval_expr(&args[0])? {
                        Value::OlsResult(m) => {
                            let yhat = m.x.dot(&m.result.params).to_vec();
                            (yhat, m.residuals.to_vec(), "OLS".to_string())
                        }
                        _ => return Err(HayashiError::Type(
                            "residplot() only supports OLS models; for GLM use predict + scatter"
                                .into(),
                        )),
                    };
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 60,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 20,
                };
                let title = format!("Residuals vs Ŷ — {mname}");
                let display =
                    ascii_scatter(&fitted, &resids, &title, "ŷ (fitted)", "e (residual)", w, h);
                let summary = title.clone();
                let fields = vec![
                    ("model".into(), Value::Str(mname.clone())),
                    ("n".into(), Value::Int(fitted.len() as i64)),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("title".into(), Value::Str(title)),
                    (
                        "fitted".into(),
                        model_expansion::series_from_vec("fitted", &fitted),
                    ),
                    (
                        "residuals".into(),
                        model_expansion::series_from_vec("residuals", &resids),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "ResidPlot",
                    fields,
                ))
            }

            // ── coefplot: ASCII coefficient chart ──────────────────────
            "coefplot" | "coef_plot" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("coefplot(model [, width=50])".into()));
                }
                let model = self.eval_expr(&args[0])?;
                let width = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 50,
                };

                let params = extract_params(&model).ok_or_else(|| {
                    HayashiError::Runtime("coefplot: model without params".into())
                })?;
                let se = extract_se(&model).unwrap_or_default();
                let names = extract_var_names(&model);
                let z = 1.96_f64;

                // collect (name, coef, ci_lo, ci_hi) excluding constant
                let mut rows: Vec<(&str, f64, f64, f64)> = Vec::new();
                for (i, p) in params.iter().enumerate() {
                    let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                    if name == "_cons" || name == "const" {
                        continue;
                    }
                    let ci_lo = *p - z * se.get(i).unwrap_or(&0.0);
                    let ci_hi = *p + z * se.get(i).unwrap_or(&0.0);
                    rows.push((name, *p, ci_lo, ci_hi));
                }
                if rows.is_empty() {
                    let display = "(no coefficients to plot)\n".to_string();
                    let summary = "No coefficients to plot".to_string();
                    return Ok(Some(model_expansion::model_result(
                        display,
                        summary,
                        "CoefPlot",
                        vec![],
                    )));
                }

                let label_w = rows
                    .iter()
                    .map(|(n, _, _, _)| n.len())
                    .max()
                    .unwrap_or(4)
                    .max(8);
                let all_lo = rows
                    .iter()
                    .map(|(_, _, lo, _)| *lo)
                    .fold(f64::INFINITY, f64::min);
                let all_hi = rows
                    .iter()
                    .map(|(_, _, _, hi)| *hi)
                    .fold(f64::NEG_INFINITY, f64::max);
                let range = (all_hi - all_lo).max(1e-15);
                // expand to include zero if not in range
                let plot_lo = all_lo.min(0.0) - range * 0.05;
                let plot_hi = all_hi.max(0.0) + range * 0.05;
                let plot_range = (plot_hi - plot_lo).max(1e-15);

                let to_col = |v: f64| -> usize {
                    ((v - plot_lo) / plot_range * (width - 1) as f64)
                        .round()
                        .clamp(0.0, (width - 1) as f64) as usize
                };
                let zero_col = to_col(0.0);

                let mut display = String::new();
                display.push('\n');
                for (name, coef, ci_lo, ci_hi) in &rows {
                    let c_lo = to_col(*ci_lo);
                    let c_hi = to_col(*ci_hi);
                    let c_pt = to_col(*coef);
                    let mut line = vec![' '; width];
                    // zero line
                    if zero_col < width {
                        line[zero_col] = '│';
                    }
                    // CI bar
                    line[c_lo..=c_hi.min(width - 1)].fill('─');
                    // point estimate
                    if c_pt < width {
                        line[c_pt] = '●';
                    }
                    let bar: String = line.into_iter().collect();
                    display.push_str(&format!(
                        "{:>lw$} │{bar}  {coef:>8.3}\n",
                        name,
                        lw = label_w
                    ));
                }
                // axis
                let mut axis = vec![' '; width];
                if zero_col < width {
                    axis[zero_col] = '0';
                }
                display.push_str(&format!(
                    "{:>lw$} │{}\n",
                    "",
                    axis.iter().collect::<String>(),
                    lw = label_w
                ));
                // range labels
                let lo_s = format!("{:.1}", plot_lo);
                let hi_s = format!("{:.1}", plot_hi);
                let pad = width.saturating_sub(lo_s.len() + hi_s.len());
                display.push_str(&format!(
                    "{:>lw$}  {lo_s}{:>pad$}\n",
                    "",
                    hi_s,
                    lw = label_w,
                    pad = pad
                ));
                display.push('\n');

                let mut names_vec = Vec::new();
                let mut coefs_vec = Vec::new();
                let mut ci_lower_vec = Vec::new();
                let mut ci_upper_vec = Vec::new();
                for (name, coef, ci_lo, ci_hi) in &rows {
                    names_vec.push(name.to_string());
                    coefs_vec.push(*coef);
                    ci_lower_vec.push(*ci_lo);
                    ci_upper_vec.push(*ci_hi);
                }
                let summary = format!("Coefficient plot ({} coefficients)", rows.len());
                let fields = vec![
                    ("n".into(), Value::Int(rows.len() as i64)),
                    ("width".into(), Value::Int(width as i64)),
                    (
                        "names".into(),
                        Value::List(Arc::new(
                            names_vec.iter().map(|n| Value::Str(n.clone())).collect(),
                        )),
                    ),
                    (
                        "coefs".into(),
                        model_expansion::series_from_vec("coefs", &coefs_vec),
                    ),
                    (
                        "ci_lower".into(),
                        model_expansion::series_from_vec("ci_lower", &ci_lower_vec),
                    ),
                    (
                        "ci_upper".into(),
                        model_expansion::series_from_vec("ci_upper", &ci_upper_vec),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display, summary, "CoefPlot", fields,
                ))
            }

            // ══════════════════════════════════════════════════════════════════
            // SVG charts (publication-ready) — export to file
            // ══════════════════════════════════════════════════════════════════

            // graph_scatter(df, X, Y, path="plot.svg" [, title="", width=800, height=600])
            "graph_scatter" | "gscatter" => {
                if args.len() < 3 {
                    return Err(self.rt_err("graph_scatter(df, x_var, y_var, path=\"plot.svg\")"));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let x_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("x var".into())),
                };
                let y_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("y var".into())),
                };
                let path = match opt_map.get("path") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("{x_name}_{y_name}.svg"),
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("{y_name} vs {x_name}"),
                };
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as u32,
                    _ => 800,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as u32,
                    _ => 600,
                };
                let x = get_col_f64(&df, &x_name)?;
                let y = get_col_f64(&df, &y_name)?;
                crate::io::plot::Plot::scatter(
                    &x.to_vec(),
                    &y.to_vec(),
                    &x_name,
                    &y_name,
                    &title,
                    &path,
                    w,
                    h,
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let display = format!("graph saved: {path}\n");
                let summary = format!("Scatter plot saved to {path}");
                let fields = vec![
                    ("path".into(), Value::Str(path.clone())),
                    ("title".into(), Value::Str(title.clone())),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("x".into(), Value::Str(x_name.clone())),
                    ("y".into(), Value::Str(y_name.clone())),
                    ("n".into(), Value::Int(x.len() as i64)),
                    (
                        "xs".into(),
                        model_expansion::series_from_vec("xs", &x.to_vec()),
                    ),
                    (
                        "ys".into(),
                        model_expansion::series_from_vec("ys", &y.to_vec()),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "GraphScatter",
                    fields,
                ))
            }

            // graph_line(df, X, Y, path="plot.svg")
            "graph_line" | "gline" => {
                if args.len() < 3 {
                    return Err(self.rt_err("graph_line(df, x_var, y_var, path=\"plot.svg\")"));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let x_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("x var".into())),
                };
                let y_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("y var".into())),
                };
                let path = match opt_map.get("path") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("{y_name}_line.svg"),
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => y_name.to_string(),
                };
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as u32,
                    _ => 800,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as u32,
                    _ => 600,
                };
                let x = get_col_f64(&df, &x_name)?;
                let y = get_col_f64(&df, &y_name)?;
                crate::io::plot::Plot::line(
                    &x.to_vec(),
                    &y.to_vec(),
                    &x_name,
                    &y_name,
                    &title,
                    &path,
                    w,
                    h,
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let display = format!("graph saved: {path}\n");
                let summary = format!("Line plot saved to {path}");
                let fields = vec![
                    ("path".into(), Value::Str(path.clone())),
                    ("title".into(), Value::Str(title.clone())),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("x".into(), Value::Str(x_name.clone())),
                    ("y".into(), Value::Str(y_name.clone())),
                    ("n".into(), Value::Int(x.len() as i64)),
                    (
                        "xs".into(),
                        model_expansion::series_from_vec("xs", &x.to_vec()),
                    ),
                    (
                        "ys".into(),
                        model_expansion::series_from_vec("ys", &y.to_vec()),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "GraphLine",
                    fields,
                ))
            }

            // graph_hist(df, var, path="hist.svg" [, bins=30])
            "graph_hist" | "ghist" => {
                if args.len() < 2 {
                    return Err(self.rt_err("graph_hist(df, var, path=\"hist.svg\")"));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("var".into())),
                };
                let path = match opt_map.get("path") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("{var_name}_hist.svg"),
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => format!("Histogram — {var_name}"),
                };
                let bins = match opt_map.get("bins") {
                    Some(Value::Int(v)) => *v as usize,
                    _ => 30,
                };
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as u32,
                    _ => 800,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as u32,
                    _ => 600,
                };
                let vals = get_col_f64(&df, &var_name)?;
                crate::io::plot::Plot::histogram(
                    &vals.to_vec(),
                    &var_name,
                    &title,
                    &path,
                    w,
                    h,
                    bins,
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let display = format!("graph saved: {path}\n");
                let summary = format!("Histogram saved to {path}");
                let fields = vec![
                    ("path".into(), Value::Str(path.clone())),
                    ("title".into(), Value::Str(title.clone())),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("var".into(), Value::Str(var_name.clone())),
                    ("bins".into(), Value::Int(bins as i64)),
                    ("n".into(), Value::Int(vals.len() as i64)),
                    (
                        "data".into(),
                        model_expansion::series_from_vec("data", &vals.to_vec()),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "GraphHistogram",
                    fields,
                ))
            }

            // graph_coef(model, path="coef.svg")
            "graph_coef" | "gcoefplot" => {
                if args.is_empty() {
                    return Err(self.rt_err("graph_coef(model, path=\"coef.svg\")"));
                }
                let model = self.eval_expr(&args[0])?;
                let path = match opt_map.get("path") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "coefplot.svg".to_string(),
                };
                let title = match opt_map.get("title") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "Coefficient Plot".to_string(),
                };
                let w = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as u32,
                    _ => 700,
                };
                let h = match opt_map.get("height") {
                    Some(Value::Int(v)) => *v as u32,
                    _ => 500,
                };
                let params = extract_params(&model)
                    .ok_or_else(|| HayashiError::Runtime("model sem params".into()))?;
                let se = extract_se(&model).unwrap_or_default();
                let names = extract_var_names(&model);
                let z = 1.96_f64;
                let mut plot_names = Vec::new();
                let mut plot_coefs = Vec::new();
                let mut plot_lo = Vec::new();
                let mut plot_hi = Vec::new();
                for (i, p) in params.iter().enumerate() {
                    let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                    if name == "_cons" || name == "const" {
                        continue;
                    }
                    plot_names.push(name.to_string());
                    plot_coefs.push(*p);
                    plot_lo.push(*p - z * se.get(i).unwrap_or(&0.0));
                    plot_hi.push(*p + z * se.get(i).unwrap_or(&0.0));
                }
                crate::io::plot::Plot::coefplot(
                    &plot_names,
                    &plot_coefs,
                    &plot_lo,
                    &plot_hi,
                    &title,
                    &path,
                    w,
                    h,
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let display = format!("graph saved: {path}\n");
                let summary = format!("Coefficient plot saved to {path}");
                let fields = vec![
                    ("path".into(), Value::Str(path.clone())),
                    ("title".into(), Value::Str(title.clone())),
                    ("width".into(), Value::Int(w as i64)),
                    ("height".into(), Value::Int(h as i64)),
                    ("n".into(), Value::Int(plot_names.len() as i64)),
                    (
                        "names".into(),
                        Value::List(Arc::new(
                            plot_names.iter().map(|n| Value::Str(n.clone())).collect(),
                        )),
                    ),
                    (
                        "coefs".into(),
                        model_expansion::series_from_vec("coefs", &plot_coefs),
                    ),
                    (
                        "ci_lower".into(),
                        model_expansion::series_from_vec("ci_lower", &plot_lo),
                    ),
                    (
                        "ci_upper".into(),
                        model_expansion::series_from_vec("ci_upper", &plot_hi),
                    ),
                ];
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "GraphCoef",
                    fields,
                ))
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
