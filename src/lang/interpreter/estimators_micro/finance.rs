use super::super::helpers::*;
use super::super::*;
use crate::lang::dap::model_expansion;

impl Interpreter {
    pub(super) fn fmb(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime("fmb(formula, df, time=col)".into()));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
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
        let time_col = match opt_map.get("time") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(_, t)| t.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    HayashiError::Runtime("fmb requires time=col or xtset(df, id, time)".into())
                })?,
        };
        let nw_lags: usize = match opt_map.get("nw") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            Some(Value::Str(s)) => s.parse().unwrap_or(0),
            _ => 0,
        };

        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let result = greeners::FamaMacBeth::fit(&g_formula, &df, &time_col, nw_lags)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "FamaMacBeth(k={}, n_periods={}, n_obs_total={}), NWlags={}",
            result.params.len(),
            result.n_periods,
            result.n_obs_total,
            result.nw_lags
        );
        let fields = vec![
            (
                "coefficients".into(),
                model_expansion::coef_dataframe(
                    &result.variable_names,
                    &result.params,
                    &result.std_errors,
                    &result.t_values,
                    &result.p_values,
                    None,
                    None,
                ),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_periods", Value::Int(result.n_periods as i64)),
                    ("n_obs_total", Value::Int(result.n_obs_total as i64)),
                    ("nw_lags", Value::Int(result.nw_lags as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "FamaMacBethResult",
            fields,
        ))
    }

    pub(super) fn portsort(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "portsort(df, ret_var, sort_var, n=5)".into(),
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
        let df_raw = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let df = self.maybe_filter_df(&df_raw, opts)?;
        let ret_name = match &args[1] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "second argument must be return variable".into(),
                ))
            }
        };
        let sort_name = match &args[2] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "third argument must be sort variable".into(),
                ))
            }
        };
        let n_ports: usize = match opt_map.get("n") {
            Some(Value::Int(v)) => (*v).max(2) as usize,
            Some(Value::Float(v)) => (*v as usize).max(2),
            _ => 5,
        };

        let ret_col = get_col_f64(&df, &ret_name)?;
        let sort_col = get_col_f64(&df, &sort_name)?;

        // pares (sort_val, ret_val) — excluir NaN
        let mut pairs: Vec<(f64, f64)> = sort_col
            .iter()
            .zip(ret_col.iter())
            .filter(|(s, r)| s.is_finite() && r.is_finite())
            .map(|(&s, &r)| (s, r))
            .collect();
        pairs.sort_by(|a, b| nan_last_cmp(&a.0, &b.0));
        let n_valid = pairs.len();
        let per_port = n_valid / n_ports;

        if per_port < 1 {
            return Err(HayashiError::Runtime(format!(
                "portsort: {n_valid} valid obs insufficient for {n_ports} portfolios"
            )));
        }

        // assign portfolios
        struct PortStats {
            mean: f64,
            se: f64,
            n: usize,
        }
        let mut ports: Vec<PortStats> = Vec::new();
        for p in 0..n_ports {
            let start = p * per_port;
            let end = if p == n_ports - 1 {
                n_valid
            } else {
                (p + 1) * per_port
            };
            let rets: Vec<f64> = pairs[start..end].iter().map(|(_, r)| *r).collect();
            let n = rets.len();
            let mean = rets.iter().sum::<f64>() / n as f64;
            let var =
                rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
            let se = (var / n as f64).sqrt();
            ports.push(PortStats { mean, se, n });
        }

        // spread H-L
        let hl_mean = ports.last().unwrap().mean - ports[0].mean;
        let hl_se = (ports.last().unwrap().se.powi(2) + ports[0].se.powi(2)).sqrt();
        let hl_t = if hl_se > 1e-15 {
            hl_mean / hl_se
        } else {
            f64::NAN
        };
        let hl_p = t_pvalue_two(hl_t, (ports.last().unwrap().n + ports[0].n - 2) as f64);

        let thick = "═".repeat(60);
        let thin = "─".repeat(60);
        let mut display = String::new();
        display.push_str(&format!("\n{thick}\n"));
        display.push_str(&format!(
            "{:^60}\n",
            format!(" Portfolio Sort: {ret_name} by {sort_name} ({n_ports} groups) ")
        ));
        display.push_str(&format!("{thin}\n"));
        display.push_str(&format!(
            "{:<12} {:>8} {:>12} {:>10} {:>10}\n",
            "Portfolio", "N", "Mean", "SE", "t"
        ));
        display.push_str(&format!("{thin}\n"));
        for (i, ps) in ports.iter().enumerate() {
            let t = if ps.se > 1e-15 {
                ps.mean / ps.se
            } else {
                f64::NAN
            };
            let label = if i == 0 {
                "Low".to_string()
            } else if i == n_ports - 1 {
                "High".to_string()
            } else {
                format!("P{}", i + 1)
            };
            display.push_str(&format!(
                "{:<12} {:>8} {:>12.4} {:>10.4} {:>10.4}\n",
                label, ps.n, ps.mean, ps.se, t
            ));
        }
        display.push_str(&format!("{thin}\n"));
        let sig = if hl_p < 0.01 {
            "***"
        } else if hl_p < 0.05 {
            "**"
        } else if hl_p < 0.10 {
            "*"
        } else {
            ""
        };
        display.push_str(&format!(
            "{:<12} {:>8} {:>12.4} {:>10.4} {:>10.4} {sig}\n",
            "H-L", "", hl_mean, hl_se, hl_t
        ));
        display.push_str(&format!("{thick}\n"));

        let means: Vec<f64> = ports.iter().map(|p| p.mean).collect();
        let ses: Vec<f64> = ports.iter().map(|p| p.se).collect();
        let ns: Vec<usize> = ports.iter().map(|p| p.n).collect();
        let ts: Vec<f64> = ports
            .iter()
            .map(|p| {
                if p.se > 1e-15 {
                    p.mean / p.se
                } else {
                    f64::NAN
                }
            })
            .collect();
        let labels: Vec<String> = (0..n_ports)
            .map(|i| match i {
                0 => "Low".into(),
                i if i == n_ports - 1 => "High".into(),
                _ => format!("P{}", i + 1),
            })
            .collect();

        let summary = format!(
            "PortfolioSort(n_ports={}, n_valid={}), H-L={:.4}, p={:.4}",
            n_ports, n_valid, hl_mean, hl_p
        );
        let fields = vec![
            (
                "labels".into(),
                Value::List(Arc::new(labels.into_iter().map(Value::Str).collect())),
            ),
            (
                "means".into(),
                model_expansion::series_from_vec("means", &means),
            ),
            ("se".into(), model_expansion::series_from_vec("se", &ses)),
            ("n".into(), model_expansion::int_series("n", &ns)),
            ("t".into(), model_expansion::series_from_vec("t", &ts)),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_ports", Value::Int(n_ports as i64)),
                    ("n_valid", Value::Int(n_valid as i64)),
                    ("hl_mean", Value::Float(hl_mean)),
                    ("hl_se", Value::Float(hl_se)),
                    ("hl_t", Value::Float(hl_t)),
                    ("hl_p", Value::Float(hl_p)),
                    ("ret_name", Value::Str(ret_name.clone())),
                    ("sort_name", Value::Str(sort_name.clone())),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            display,
            summary,
            "PortfolioSortResult",
            fields,
        ))
    }

    pub(super) fn doublesort(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 4 {
            return Err(HayashiError::Runtime(
                "doublesort(df, ret, sort1, sort2, n1=5, n2=5)".into(),
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
        let df_raw = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let df = self.maybe_filter_df(&df_raw, opts)?;
        let ret_name = match &args[1] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => return Err(HayashiError::Type("ret var".into())),
        };
        let s1_name = match &args[2] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => return Err(HayashiError::Type("sort1 var".into())),
        };
        let s2_name = match &args[3] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => return Err(HayashiError::Type("sort2 var".into())),
        };
        let n1: usize = match opt_map.get("n1") {
            Some(Value::Int(v)) => (*v).max(2) as usize,
            _ => 5,
        };
        let n2: usize = match opt_map.get("n2") {
            Some(Value::Int(v)) => (*v).max(2) as usize,
            _ => 5,
        };

        let ret_col = get_col_f64(&df, &ret_name)?;
        let s1_col = get_col_f64(&df, &s1_name)?;
        let s2_col = get_col_f64(&df, &s2_name)?;

        // atribuir quantis independentes
        let assign_quantile = |vals: &[f64], n_q: usize| -> Vec<usize> {
            let mut indexed: Vec<(usize, f64)> = vals
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_by(|a, b| nan_last_cmp(&a.1, &b.1));
            let n = indexed.len();
            let mut q = vec![usize::MAX; vals.len()];
            for (rank, &(orig_i, _)) in indexed.iter().enumerate() {
                q[orig_i] = (rank * n_q / n).min(n_q - 1);
            }
            q
        };

        let s1_vec: Vec<f64> = s1_col.to_vec();
        let s2_vec: Vec<f64> = s2_col.to_vec();
        let q1 = assign_quantile(&s1_vec, n1);
        let q2 = assign_quantile(&s2_vec, n2);

        // means per cell (q1 x q2)
        let mut cell_sum = vec![vec![0.0; n2]; n1];
        let mut cell_n = vec![vec![0usize; n2]; n1];
        for i in 0..ret_col.len() {
            if q1[i] < n1 && q2[i] < n2 && ret_col[i].is_finite() {
                cell_sum[q1[i]][q2[i]] += ret_col[i];
                cell_n[q1[i]][q2[i]] += 1;
            }
        }

        let n_valid: usize = cell_n.iter().map(|row| row.iter().sum::<usize>()).sum();

        let row_labels: Vec<String> = (0..n1)
            .map(|i| match i {
                0 => "Low".into(),
                i if i == n1 - 1 => "High".into(),
                _ => format!("Q{}", i + 1),
            })
            .collect();
        let col_labels: Vec<String> = (0..n2)
            .map(|j| match j {
                0 => "Low".into(),
                j if j == n2 - 1 => "High".into(),
                _ => format!("Q{}", j + 1),
            })
            .collect();

        let means = Array2::from_shape_fn((n1, n2), |(i, j)| {
            if cell_n[i][j] > 0 {
                cell_sum[i][j] / cell_n[i][j] as f64
            } else {
                f64::NAN
            }
        });
        let counts = Array2::from_shape_fn((n1, n2), |(i, j)| cell_n[i][j] as f64);

        let thick = "═".repeat(12 + n2 * 10);
        let thin = "─".repeat(12 + n2 * 10);
        let mut display = String::new();
        display.push_str(&format!("\n{thick}\n"));
        display.push_str(&format!(
            " Double Sort: {ret_name} by {s1_name} (rows) × {s2_name} (cols)\n"
        ));
        display.push_str(&format!("{thin}\n"));
        display.push_str(&format!("{:<12}", format!("{s1_name}\\{s2_name}")));
        for j in 0..n2 {
            let label = if j == 0 {
                "Low"
            } else if j == n2 - 1 {
                "High"
            } else {
                &format!("Q{}", j + 1)
            };
            display.push_str(&format!("{:>10}", label));
        }
        display.push('\n');
        display.push_str(&format!("{thin}\n"));
        for i in 0..n1 {
            let label = if i == 0 {
                "Low".to_string()
            } else if i == n1 - 1 {
                "High".to_string()
            } else {
                format!("Q{}", i + 1)
            };
            display.push_str(&format!("{:<12}", label));
            for j in 0..n2 {
                let mean = if cell_n[i][j] > 0 {
                    cell_sum[i][j] / cell_n[i][j] as f64
                } else {
                    f64::NAN
                };
                if mean.is_nan() {
                    display.push_str(&format!("{:>10}", "."));
                } else {
                    display.push_str(&format!("{:>10.4}", mean));
                }
            }
            display.push('\n');
        }
        display.push_str(&format!("{thick}\n"));

        let summary = format!(
            "DoubleSort(n1={}, n2={}, n_valid={}), ret={}, s1={}, s2={}",
            n1, n2, n_valid, ret_name, s1_name, s2_name
        );
        let means_df = model_expansion::array2_to_dataframe_named(&means, &col_labels);
        let counts_df = model_expansion::array2_to_dataframe_named(&counts, &col_labels);
        let fields = vec![
            ("means".into(), means_df),
            ("counts".into(), counts_df),
            (
                "row_labels".into(),
                Value::List(Arc::new(row_labels.into_iter().map(Value::Str).collect())),
            ),
            (
                "col_labels".into(),
                Value::List(Arc::new(col_labels.into_iter().map(Value::Str).collect())),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n1", Value::Int(n1 as i64)),
                    ("n2", Value::Int(n2 as i64)),
                    ("n_valid", Value::Int(n_valid as i64)),
                    ("ret_name", Value::Str(ret_name.clone())),
                    ("s1_name", Value::Str(s1_name.clone())),
                    ("s2_name", Value::Str(s2_name.clone())),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            display,
            summary,
            "DoubleSortResult",
            fields,
        ))
    }
}
