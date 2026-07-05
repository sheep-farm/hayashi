use super::*;
use super::helpers::*;

/// Visualização ASCII (acfplot/pacf/qqplot/corrplot/scatter/histogram/boxplot/kdensity)
/// e coefplot. Extraído de `eval_call` (ver src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_visualization(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── Visualização ASCII — ACF / PACF / QQ-plot / heatmap ──────────

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
                Self::ascii_acf(&clean, max_lag, &title, width, false);
                Ok(Value::Nil)
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
                Self::ascii_acf(&clean, max_lag, &title, width, true);
                Ok(Value::Nil)
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
                Self::ascii_qqplot(&clean, &title, &var_name, w, h);
                Ok(Value::Nil)
            }

            // corrplot(df, var1, var2, ...) — matriz de correlação ASCII
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
                                    "args devem ser nomes de variáveis".into(),
                                ))
                            }
                        }
                    }
                    v
                };
                if var_names.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "corrplot: forneça ao menos 2 variáveis".into(),
                    ));
                }
                let cols: Vec<Vec<f64>> = {
                    let mut v = Vec::new();
                    for n in &var_names {
                        v.push(get_col_f64(&df, n)?.to_vec());
                    }
                    v
                };
                Self::ascii_corrplot(&cols, &var_names);
                Ok(Value::Nil)
            }

            // ── Visualização ASCII ────────────────────────────────────────────

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
                    _ => format!("Histograma — {var_name}"),
                };
                let clean: Vec<f64> = data.iter().cloned().filter(|v| !v.is_nan()).collect();
                Self::ascii_histogram(&clean, bins, &title, &var_name, width);
                Ok(Value::Nil)
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
                Self::ascii_scatter(&xs.to_vec(), &ys.to_vec(), &title, &xname, &yname, w, h);
                Ok(Value::Nil)
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
                            "second argument must be a variable name (x/tempo)".into(),
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
                    _ => format!("{yname} — série temporal"),
                };
                Self::ascii_lineplot(&xs.to_vec(), &ys.to_vec(), &title, &xname, &yname, w, h);
                Ok(Value::Nil)
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
                Self::ascii_boxplot(&clean, &title, &var_name, w);
                Ok(Value::Nil)
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
                // bw= option ou Silverman
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
                    return Err(HayashiError::Runtime("kdensity: poucos dados".into()));
                }
                // Estimar KDE — usa support/density já calculados no fit (512 pontos)
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
                Self::ascii_lineplot(&xs, &ys, &title, &var_name, "densidade", w, h);
                Ok(Value::Nil)
            }

            // residplot(model, width=60, height=20) — resíduos vs ŷ
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
                let title = format!("Resíduos vs Ŷ — {mname}");
                Self::ascii_scatter(&fitted, &resids, &title, "ŷ (fitted)", "e (resíduo)", w, h);
                Ok(Value::Nil)
            }

            // ── coefplot: gráfico ASCII de coeficientes ──────────────────────
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

                let params = extract_params(&model)
                    .ok_or_else(|| HayashiError::Runtime("coefplot: modelo sem params".into()))?;
                let se = extract_se(&model).unwrap_or_default();
                let names = extract_var_names(&model);
                let z = 1.96_f64;

                // coletar (nome, coef, ci_lo, ci_hi) excluindo constante
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
                    println!("(no coefficients to plot)");
                    return Ok(Some(Value::Nil));
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
                // expandir para incluir zero se não está no range
                let plot_lo = all_lo.min(0.0) - range * 0.05;
                let plot_hi = all_hi.max(0.0) + range * 0.05;
                let plot_range = (plot_hi - plot_lo).max(1e-15);

                let to_col = |v: f64| -> usize {
                    ((v - plot_lo) / plot_range * (width - 1) as f64)
                        .round()
                        .clamp(0.0, (width - 1) as f64) as usize
                };
                let zero_col = to_col(0.0);

                println!();
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
                    println!("{:>lw$} │{bar}  {coef:>8.3}", name, lw = label_w);
                }
                // axis
                print!("{:>lw$} │", "", lw = label_w);
                let mut axis = vec![' '; width];
                if zero_col < width {
                    axis[zero_col] = '0';
                }
                println!("{}", axis.iter().collect::<String>());
                // range labels
                let lo_s = format!("{:.1}", plot_lo);
                let hi_s = format!("{:.1}", plot_hi);
                let pad = width.saturating_sub(lo_s.len() + hi_s.len());
                println!("{:>lw$}  {lo_s}{:>pad$}", "", hi_s, lw = label_w, pad = pad);
                println!();

                Ok(Value::Nil)
            }

            // ══════════════════════════════════════════════════════════════════
            // Gráficos SVG (publicáveis) — exporta para arquivo
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
                println!("graph saved: {path}");
                Ok(Value::Nil)
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
                println!("graph saved: {path}");
                Ok(Value::Nil)
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
                println!("graph saved: {path}");
                Ok(Value::Nil)
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
                println!("graph saved: {path}");
                Ok(Value::Nil)
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
