use super::super::*;
impl Interpreter {
    pub(super) fn tidy(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "tidy" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "tidy(model) requires 1 argument".into(),
                    ));
                }
                let val = self.eval_expr(&args[0])?;
                let mut map = std::collections::HashMap::<String, Value>::new();

                match val {
                    Value::OlsResult(m) => {
                        let r = &m.result;
                        map = self.build_tidy_coef_map(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                            &r.conf_lower,
                            &r.conf_upper,
                        );
                    }
                    Value::RollingResult(r) => {
                        let dates = r.dates.clone();
                        let n = r.n_obs;
                        let k = r.params_history.ncols();
                        let names = r.variable_names.clone().unwrap_or_default();
                        let mut date_col = Vec::new();
                        let mut r2_col = Vec::new();
                        let mut coef_cols: Vec<(String, Vec<Value>)> = (0..k)
                            .map(|j| {
                                let name = names.get(j).cloned().unwrap_or_else(|| {
                                    if j == 0 {
                                        "const".into()
                                    } else {
                                        format!("x{j}")
                                    }
                                });
                                (name, Vec::new())
                            })
                            .collect();
                        for t in (r.window - 1)..n {
                            if r.params_history.row(t).iter().any(|v| v.is_nan()) {
                                continue;
                            }
                            let d = dates.get(t).cloned().unwrap_or_else(|| format!("{t}"));
                            date_col.push(Value::Str(d));
                            r2_col.push(Value::Float(r.r_squared_history[t]));
                            for (j, col) in coef_cols.iter_mut().enumerate().take(k) {
                                col.1.push(Value::Float(r.params_history[[t, j]]));
                            }
                        }
                        map.insert("date".into(), Value::List(Arc::new(date_col)));
                        map.insert("r2".into(), Value::List(Arc::new(r2_col)));
                        for (name, vals) in coef_cols {
                            map.insert(name, Value::List(Arc::new(vals)));
                        }
                    }
                    Value::IvResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::BinaryResult(m) => {
                        let r = &m.result;
                        map = self.build_tidy_simple(
                            m.coef_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::PanelResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::ReResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::GmmResult(r) => {
                        let names: Vec<String> =
                            (0..r.params.len()).map(|i| format!("x{i}")).collect();
                        map = self.build_tidy_simple(
                            names,
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::PoissonResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::NegBinResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::GlmResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::QuantileResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::TobitResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::HeckmanResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::OrderedResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::AbResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::PenalizedResult(m) => {
                        map = self.build_tidy_simple(
                            m.variable_names.clone(),
                            &m.params,
                            &m.std_errors,
                            &ndarray::Array1::from_vec(vec![0.0; m.params.len()]),
                            &ndarray::Array1::from_vec(vec![0.0; m.params.len()]),
                        );
                    }
                    Value::RlmResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::BetaResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::GeeResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.robust_se,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::ArimaResult(r) => {
                        // ARIMA has ar_params, ma_params, intercept — concatenate
                        let mut all_params = r.ar_params.to_vec();
                        all_params.extend(r.ma_params.iter().cloned());
                        all_params.push(r.intercept);
                        let params = ndarray::Array1::from_vec(all_params);
                        let p = r.p_values.len();
                        let se = if r.std_errors.len() >= p {
                            r.std_errors.slice(ndarray::s![..p]).to_owned()
                        } else {
                            ndarray::Array1::from_vec(vec![f64::NAN; p])
                        };
                        let tv = if r.t_values.len() >= p {
                            r.t_values.slice(ndarray::s![..p]).to_owned()
                        } else {
                            ndarray::Array1::from_vec(vec![f64::NAN; p])
                        };
                        let pv = if r.p_values.len() >= p {
                            r.p_values.slice(ndarray::s![..p]).to_owned()
                        } else {
                            ndarray::Array1::from_vec(vec![f64::NAN; p])
                        };
                        let names: Vec<String> = (0..params.len())
                            .map(|i| {
                                if i < r.ar_params.len() {
                                    format!("ar{}", i + 1)
                                } else if i < r.ar_params.len() + r.ma_params.len() {
                                    format!("ma{}", i - r.ar_params.len() + 1)
                                } else {
                                    "intercept".into()
                                }
                            })
                            .collect();
                        map = self.build_tidy_simple(names, &params, &se, &tv, &pv);
                    }
                    Value::GarchResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::VarResult(r) => {
                        // VAR: params is (1+k*p) x k matrix — flatten column by column
                        let k = r.n_vars;
                        let p = r.lags;
                        let n_coef = (1 + p * k) * k;
                        let mut params = ndarray::Array1::<f64>::zeros(n_coef);
                        let mut ses = ndarray::Array1::<f64>::zeros(n_coef);
                        let mut names: Vec<String> = Vec::with_capacity(n_coef);
                        let mut idx = 0;
                        for eq in 0..k {
                            for row in 0..(1 + p * k) {
                                params[idx] = r.params[(row, eq)];
                                ses[idx] = r.std_errors[(row, eq)];
                                if row == 0 {
                                    names.push(format!("const_{}", r.var_names[eq]));
                                } else {
                                    let lag = (row - 1) / k;
                                    let src = (row - 1) % k;
                                    names.push(format!(
                                        "L{}.{}_{}",
                                        lag + 1,
                                        r.var_names[src],
                                        r.var_names[eq]
                                    ));
                                }
                                idx += 1;
                            }
                        }
                        let tv = ndarray::Array1::<f64>::zeros(n_coef);
                        let pv = ndarray::Array1::<f64>::zeros(n_coef);
                        map = self.build_tidy_simple(names, &params, &ses, &tv, &pv);
                    }
                    Value::VecmResult(r) => {
                        // VECM: alpha (r x k), beta (r x k), gamma (k*(p-1) x k)
                        let k = r.n_vars;
                        let rank = r.rank;
                        let p = r.lags;
                        let n_alpha = rank * k;
                        let n_beta = rank * k;
                        let n_gamma = k * (p.saturating_sub(1)) * k;
                        let n_total = n_alpha + n_beta + n_gamma;
                        let mut params = ndarray::Array1::<f64>::zeros(n_total);
                        let mut ses = ndarray::Array1::<f64>::zeros(n_total);
                        let mut names: Vec<String> = Vec::with_capacity(n_total);
                        let mut idx = 0;
                        for j in 0..k {
                            for i in 0..rank {
                                params[idx] = r.alpha[(i, j)];
                                ses[idx] = r.std_errors_alpha[(i, j)];
                                let vn = r
                                    .variable_names
                                    .get(j)
                                    .cloned()
                                    .unwrap_or_else(|| format!("v{j}"));
                                names.push(format!("alpha_{}_{}", i + 1, vn));
                                idx += 1;
                            }
                        }
                        for j in 0..k {
                            for i in 0..rank {
                                params[idx] = r.beta[(i, j)];
                                ses[idx] = r.std_errors_beta[(i, j)];
                                let vn = r
                                    .variable_names
                                    .get(j)
                                    .cloned()
                                    .unwrap_or_else(|| format!("v{j}"));
                                names.push(format!("beta_{}_{}", i + 1, vn));
                                idx += 1;
                            }
                        }
                        for j in 0..k {
                            for i in 0..k * p.saturating_sub(1) {
                                params[idx] = r.gamma[(i, j)];
                                ses[idx] = r.std_errors_gamma[(i, j)];
                                let vn = r
                                    .variable_names
                                    .get(j)
                                    .cloned()
                                    .unwrap_or_else(|| format!("v{j}"));
                                names.push(format!("gamma_{}_{}", i + 1, vn));
                                idx += 1;
                            }
                        }
                        let tv = ndarray::Array1::<f64>::zeros(n_total);
                        let pv = ndarray::Array1::<f64>::zeros(n_total);
                        map = self.build_tidy_simple(names, &params, &ses, &tv, &pv);
                    }
                    Value::SysGmmResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::FE2SLSResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::PcseResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::PanelGlsResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::GlsarResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::RecursiveLSResult(r) => {
                        let names: Vec<String> =
                            (0..r.params.len()).map(|i| format!("beta{}", i)).collect();
                        let se = ndarray::Array1::<f64>::zeros(r.params.len());
                        let tv = ndarray::Array1::<f64>::zeros(r.params.len());
                        let pv = ndarray::Array1::<f64>::zeros(r.params.len());
                        map = self.build_tidy_simple(names, &r.params, &se, &tv, &pv);
                    }
                    Value::CoxResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::ConditionalResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::GamResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::MixedResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.fixed_effects,
                            &r.fixed_se,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::ZeroInflatedResult(r) => {
                        // Combine count and inflate params
                        let nc = r.count_params.len();
                        let ni = r.inflate_params.len();
                        let n = nc + ni;
                        let mut params = ndarray::Array1::<f64>::zeros(n);
                        let mut ses = ndarray::Array1::<f64>::zeros(n);
                        let mut tv = ndarray::Array1::<f64>::zeros(n);
                        let mut pv = ndarray::Array1::<f64>::zeros(n);
                        let mut names: Vec<String> = Vec::with_capacity(n);
                        let count_names = r.count_var_names.clone().unwrap_or_default();
                        let inflate_names = r.inflate_var_names.clone().unwrap_or_default();
                        for i in 0..nc {
                            params[i] = r.count_params[i];
                            ses[i] = r.count_std_errors[i];
                            tv[i] = r.count_z_values[i];
                            pv[i] = r.count_p_values[i];
                            let nm = count_names
                                .get(i)
                                .cloned()
                                .unwrap_or_else(|| format!("x{i}"));
                            names.push(format!("count_{nm}"));
                        }
                        for i in 0..ni {
                            params[nc + i] = r.inflate_params[i];
                            ses[nc + i] = r.inflate_std_errors[i];
                            tv[nc + i] = r.inflate_z_values[i];
                            pv[nc + i] = r.inflate_p_values[i];
                            let nm = inflate_names
                                .get(i)
                                .cloned()
                                .unwrap_or_else(|| format!("x{i}"));
                            names.push(format!("inflate_{nm}"));
                        }
                        map = self.build_tidy_simple(names, &params, &ses, &tv, &pv);
                    }
                    Value::AutoRegResult(r) => {
                        map = self.build_tidy_simple(
                            r.param_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::ArdlResult(r) => {
                        map = self.build_tidy_simple(
                            r.param_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::DidResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::ThresholdResult(r) => {
                        // Combine regime1 and regime2 params
                        let n1 = r.params_regime1.len();
                        let n2 = r.params_regime2.len();
                        let n = n1 + n2 + 1;
                        let mut params = ndarray::Array1::<f64>::zeros(n);
                        let mut names: Vec<String> = Vec::with_capacity(n);
                        for i in 0..n1 {
                            params[i] = r.params_regime1[i];
                            names.push(format!("regime1_x{}", i));
                        }
                        for i in 0..n2 {
                            params[n1 + i] = r.params_regime2[i];
                            names.push(format!("regime2_x{}", i));
                        }
                        params[n - 1] = r.threshold_gamma;
                        names.push("threshold".into());
                        let se = ndarray::Array1::<f64>::zeros(n);
                        let tv = ndarray::Array1::<f64>::zeros(n);
                        let pv = ndarray::Array1::<f64>::zeros(n);
                        map = self.build_tidy_simple(names, &params, &se, &tv, &pv);
                    }
                    Value::RdResult(r) => {
                        let mut var = vec!["tau".to_string()];
                        let mut coef = vec![r.tau];
                        let mut se = vec![r.se];
                        let mut t = vec![r.z];
                        let mut p = vec![r.p_value];
                        let mut cl = vec![r.ci_lower];
                        let mut cu = vec![r.ci_upper];
                        if r.is_fuzzy {
                            if let Some(ft) = r.first_stage_tau {
                                var.push("first_stage_tau".into());
                                coef.push(ft);
                                se.push(r.first_stage_se.unwrap_or(f64::NAN));
                                t.push(f64::NAN);
                                p.push(f64::NAN);
                                cl.push(f64::NAN);
                                cu.push(f64::NAN);
                            }
                        }
                        let params = ndarray::Array1::from_vec(coef);
                        let se = ndarray::Array1::from_vec(se);
                        let t = ndarray::Array1::from_vec(t);
                        let p = ndarray::Array1::from_vec(p);
                        let cl = ndarray::Array1::from_vec(cl);
                        let cu = ndarray::Array1::from_vec(cu);
                        map = self.build_tidy_coef_map(var, &params, &se, &t, &p, &cl, &cu);
                    }
                    Value::SynthResult(r) => {
                        let mut var = Vec::new();
                        let mut coef = Vec::new();
                        for (id, w) in &r.weights {
                            var.push(id.clone());
                            coef.push(*w);
                        }
                        let n = coef.len();
                        let params = ndarray::Array1::from_vec(coef);
                        let se = ndarray::Array1::from_vec(vec![f64::NAN; n]);
                        let t = ndarray::Array1::from_vec(vec![f64::NAN; n]);
                        let p = ndarray::Array1::from_vec(vec![f64::NAN; n]);
                        let names: Vec<String> = var.clone();
                        map = self.build_tidy_simple(names, &params, &se, &t, &p);
                    }
                    Value::PsmResult(r) => {
                        let mut cov = Vec::new();
                        let mut mt = Vec::new();
                        let mut mcr = Vec::new();
                        let mut mcm = Vec::new();
                        let mut smdb = Vec::new();
                        let mut smda = Vec::new();
                        for b in &r.balance {
                            cov.push(Value::Str(b.covariate.clone()));
                            mt.push(Value::Float(b.mean_treated));
                            mcr.push(Value::Float(b.mean_control_raw));
                            mcm.push(Value::Float(b.mean_control_matched));
                            smdb.push(Value::Float(b.smd_before));
                            smda.push(Value::Float(b.smd_after));
                        }
                        map.insert("covariate".into(), Value::List(Arc::new(cov)));
                        map.insert("mean_treated".into(), Value::List(Arc::new(mt)));
                        map.insert("mean_control_raw".into(), Value::List(Arc::new(mcr)));
                        map.insert("mean_control_matched".into(), Value::List(Arc::new(mcm)));
                        map.insert("smd_before".into(), Value::List(Arc::new(smdb)));
                        map.insert("smd_after".into(), Value::List(Arc::new(smda)));
                    }
                    Value::MNLogitResult(r) => {
                        let k = r.params.nrows();
                        let j = r.params.ncols();
                        let mut var = Vec::new();
                        let mut outcome = Vec::new();
                        let mut coef = Vec::new();
                        let mut se = Vec::new();
                        let mut z = Vec::new();
                        let mut p = Vec::new();
                        let vnames = r.variable_names.clone().unwrap_or_default();
                        for col in 0..j {
                            let out = r
                                .category_labels
                                .get(col)
                                .map(|v| format!("{v:.0}"))
                                .unwrap_or_else(|| format!("cat{col}"));
                            for row in 0..k {
                                let name = vnames
                                    .get(row)
                                    .cloned()
                                    .unwrap_or_else(|| format!("x{row}"));
                                var.push(name);
                                outcome.push(out.clone());
                                coef.push(r.params[[row, col]]);
                                se.push(r.std_errors[[row, col]]);
                                z.push(r.z_values[[row, col]]);
                                p.push(r.p_values[[row, col]]);
                            }
                        }
                        let n = coef.len();
                        map.insert(
                            "variable".into(),
                            Value::List(Arc::new(var.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "outcome".into(),
                            Value::List(Arc::new(outcome.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "coef".into(),
                            Value::List(Arc::new(coef.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "std_err".into(),
                            Value::List(Arc::new(se.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "z".into(),
                            Value::List(Arc::new(z.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "p_value".into(),
                            Value::List(Arc::new(p.into_iter().map(Value::Float).collect())),
                        );
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::KMResult(r) => {
                        let n = r.times.len();
                        let mut time = Vec::new();
                        let mut surv = Vec::new();
                        let mut se = Vec::new();
                        let mut cl = Vec::new();
                        let mut cu = Vec::new();
                        for i in 0..n {
                            time.push(Value::Float(r.times[i]));
                            surv.push(Value::Float(r.survival_probs[i]));
                            se.push(Value::Float(r.std_errors[i]));
                            cl.push(Value::Float(r.conf_lower[i]));
                            cu.push(Value::Float(r.conf_upper[i]));
                        }
                        map.insert("time".into(), Value::List(Arc::new(time)));
                        map.insert("survival".into(), Value::List(Arc::new(surv)));
                        map.insert("std_err".into(), Value::List(Arc::new(se)));
                        map.insert("conf_low".into(), Value::List(Arc::new(cl)));
                        map.insert("conf_high".into(), Value::List(Arc::new(cu)));
                    }
                    Value::SurResult(m) => {
                        let r = &m.result;
                        let mut eq_vec = Vec::new();
                        let mut var = Vec::new();
                        let mut coef = Vec::new();
                        let mut se = Vec::new();
                        let mut t = Vec::new();
                        let mut p = Vec::new();
                        for (ei, eq) in r.equations.iter().enumerate() {
                            let names = m.eq_var_names.get(ei).cloned().unwrap_or_default();
                            for i in 0..eq.params.len() {
                                eq_vec.push(eq.name.clone());
                                var.push(names.get(i).cloned().unwrap_or_else(|| format!("x{i}")));
                                coef.push(eq.params[i]);
                                se.push(eq.std_errors[i]);
                                t.push(eq.t_values[i]);
                                p.push(eq.p_values[i]);
                            }
                        }
                        let n = coef.len();
                        map.insert(
                            "equation".into(),
                            Value::List(Arc::new(eq_vec.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "variable".into(),
                            Value::List(Arc::new(var.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "coef".into(),
                            Value::List(Arc::new(coef.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "std_err".into(),
                            Value::List(Arc::new(se.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "t".into(),
                            Value::List(Arc::new(t.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "p_value".into(),
                            Value::List(Arc::new(p.into_iter().map(Value::Float).collect())),
                        );
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::ThreeSLSResult(m) => {
                        let r = &m.result;
                        let mut eq_vec = Vec::new();
                        let mut var = Vec::new();
                        let mut coef = Vec::new();
                        let mut se = Vec::new();
                        let mut t = Vec::new();
                        let mut p = Vec::new();
                        for (ei, eq) in r.equations.iter().enumerate() {
                            let names = m.eq_var_names.get(ei).cloned().unwrap_or_default();
                            for i in 0..eq.params.len() {
                                eq_vec.push(eq.name.clone());
                                var.push(names.get(i).cloned().unwrap_or_else(|| format!("x{i}")));
                                coef.push(eq.params[i]);
                                se.push(eq.std_errors[i]);
                                t.push(eq.t_values[i]);
                                p.push(eq.p_values[i]);
                            }
                        }
                        let n = coef.len();
                        map.insert(
                            "equation".into(),
                            Value::List(Arc::new(eq_vec.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "variable".into(),
                            Value::List(Arc::new(var.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "coef".into(),
                            Value::List(Arc::new(coef.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "std_err".into(),
                            Value::List(Arc::new(se.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "t".into(),
                            Value::List(Arc::new(t.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "p_value".into(),
                            Value::List(Arc::new(p.into_iter().map(Value::Float).collect())),
                        );
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::SVarResult(r) => {
                        let mut matrix = Vec::new();
                        let mut row = Vec::new();
                        let mut col = Vec::new();
                        let mut value = Vec::new();
                        let k = r.a_matrix.nrows();
                        let vnames = r.var_result.var_names.clone();
                        for i in 0..k {
                            for j in 0..k {
                                matrix.push(Value::Str("A".into()));
                                row.push(Value::Str(
                                    vnames.get(i).cloned().unwrap_or_else(|| format!("v{i}")),
                                ));
                                col.push(Value::Str(
                                    vnames.get(j).cloned().unwrap_or_else(|| format!("v{j}")),
                                ));
                                value.push(Value::Float(r.a_matrix[[i, j]]));
                            }
                        }
                        for i in 0..r.b_matrix.nrows() {
                            for j in 0..r.b_matrix.ncols() {
                                matrix.push(Value::Str("B".into()));
                                row.push(Value::Str(
                                    vnames.get(i).cloned().unwrap_or_else(|| format!("v{i}")),
                                ));
                                col.push(Value::Str(
                                    vnames.get(j).cloned().unwrap_or_else(|| format!("v{j}")),
                                ));
                                value.push(Value::Float(r.b_matrix[[i, j]]));
                            }
                        }
                        let n = value.len();
                        map.insert("matrix".into(), Value::List(Arc::new(matrix)));
                        map.insert("row".into(), Value::List(Arc::new(row)));
                        map.insert("col".into(), Value::List(Arc::new(col)));
                        map.insert("value".into(), Value::List(Arc::new(value)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::VarmaResult(r) => {
                        let mut type_vec = Vec::new();
                        let mut lag_vec = Vec::new();
                        let mut from_vec = Vec::new();
                        let mut to_vec = Vec::new();
                        let mut value_vec = Vec::new();
                        let k = r.n_vars;
                        let vnames: Vec<String> = (0..k).map(|i| format!("y{}", i + 1)).collect();
                        // AR: rows 0..1+p*k, cols 0..k
                        for (col, _vn) in vnames.iter().enumerate().take(k) {
                            type_vec.push(Value::Str("const".into()));
                            lag_vec.push(Value::Int(0));
                            from_vec.push(Value::Str("-".into()));
                            to_vec.push(Value::Str(vnames[col].clone()));
                            value_vec.push(Value::Float(r.ar_params[[0, col]]));
                        }
                        for l in 0..r.p_lags {
                            for src in 0..k {
                                for (col, _vn) in vnames.iter().enumerate().take(k) {
                                    let row = 1 + l * k + src;
                                    type_vec.push(Value::Str("AR".into()));
                                    lag_vec.push(Value::Int((l + 1) as i64));
                                    from_vec.push(Value::Str(vnames[src].clone()));
                                    to_vec.push(Value::Str(vnames[col].clone()));
                                    value_vec.push(Value::Float(r.ar_params[[row, col]]));
                                }
                            }
                        }
                        for l in 0..r.q_lags {
                            for src in 0..k {
                                for (col, _vn) in vnames.iter().enumerate().take(k) {
                                    let row = l * k + src;
                                    type_vec.push(Value::Str("MA".into()));
                                    lag_vec.push(Value::Int((l + 1) as i64));
                                    from_vec.push(Value::Str(vnames[src].clone()));
                                    to_vec.push(Value::Str(vnames[col].clone()));
                                    value_vec.push(Value::Float(r.ma_params[[row, col]]));
                                }
                            }
                        }
                        if let Some(ex) = &r.exog_params {
                            for ex_i in 0..ex.nrows() {
                                for (col, _vn) in vnames.iter().enumerate().take(k) {
                                    type_vec.push(Value::Str("exog".into()));
                                    lag_vec.push(Value::Int(0));
                                    from_vec.push(Value::Str(format!("ex{ex_i}")));
                                    to_vec.push(Value::Str(vnames[col].clone()));
                                    value_vec.push(Value::Float(ex[[ex_i, col]]));
                                }
                            }
                        }
                        let n = value_vec.len();
                        map.insert("type".into(), Value::List(Arc::new(type_vec)));
                        map.insert("lag".into(), Value::List(Arc::new(lag_vec)));
                        map.insert("from".into(), Value::List(Arc::new(from_vec)));
                        map.insert("to".into(), Value::List(Arc::new(to_vec)));
                        map.insert("value".into(), Value::List(Arc::new(value_vec)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::MarkovResult(r) => {
                        let mut regime = Vec::new();
                        let mut parameter = Vec::new();
                        let mut value = Vec::new();
                        for (i, params) in r.regime_params.iter().enumerate() {
                            for (j, &v) in params.iter().enumerate() {
                                regime.push(Value::Int((i + 1) as i64));
                                parameter.push(Value::Str(if j == 0 {
                                    "intercept".into()
                                } else {
                                    format!("ar{}", j - 1)
                                }));
                                value.push(Value::Float(v));
                            }
                            regime.push(Value::Int((i + 1) as i64));
                            parameter.push(Value::Str("variance".into()));
                            value.push(Value::Float(r.regime_variances[i]));
                        }
                        let n = value.len();
                        map.insert("regime".into(), Value::List(Arc::new(regime)));
                        map.insert("parameter".into(), Value::List(Arc::new(parameter)));
                        map.insert("value".into(), Value::List(Arc::new(value)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::MSARResult(r) => {
                        let mut regime = Vec::new();
                        let mut parameter = Vec::new();
                        let mut value = Vec::new();
                        for i in 0..r.k_regimes {
                            regime.push(Value::Int((i + 1) as i64));
                            parameter.push(Value::Str("intercept".into()));
                            value.push(Value::Float(r.regime_means[i]));
                            for p in 0..r.ar_order {
                                regime.push(Value::Int((i + 1) as i64));
                                parameter.push(Value::Str(format!("ar{}", p + 1)));
                                value.push(Value::Float(r.ar_params[[i, p]]));
                            }
                            regime.push(Value::Int((i + 1) as i64));
                            parameter.push(Value::Str("sigma".into()));
                            value.push(Value::Float(r.regime_sigmas[i]));
                        }
                        let n = value.len();
                        map.insert("regime".into(), Value::List(Arc::new(regime)));
                        map.insert("parameter".into(), Value::List(Arc::new(parameter)));
                        map.insert("value".into(), Value::List(Arc::new(value)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::PcaResult(m) => {
                        let r = &m.result;
                        let mut var = Vec::new();
                        let mut pc = Vec::new();
                        let mut loading = Vec::new();
                        let k = m.var_names.len();
                        let c = r.n_components;
                        for j in 0..c {
                            for i in 0..k {
                                var.push(Value::Str(m.var_names[i].clone()));
                                pc.push(Value::Str(format!("PC{}", j + 1)));
                                loading.push(Value::Float(r.loadings[[i, j]]));
                            }
                        }
                        let n = loading.len();
                        map.insert("variable".into(), Value::List(Arc::new(var)));
                        map.insert("component".into(), Value::List(Arc::new(pc)));
                        map.insert("loading".into(), Value::List(Arc::new(loading)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::FactorResult(m) => {
                        let r = &m.result;
                        let mut var = Vec::new();
                        let mut factor = Vec::new();
                        let mut loading = Vec::new();
                        let mut comm = Vec::new();
                        let mut uniq = Vec::new();
                        let k = m.var_names.len();
                        let f = r.n_factors;
                        for j in 0..f {
                            for i in 0..k {
                                var.push(Value::Str(m.var_names[i].clone()));
                                factor.push(Value::Str(format!("F{}", j + 1)));
                                loading.push(Value::Float(r.loadings[[i, j]]));
                                comm.push(Value::Float(r.communalities[i]));
                                uniq.push(Value::Float(r.uniquenesses[i]));
                            }
                        }
                        let n = loading.len();
                        map.insert("variable".into(), Value::List(Arc::new(var)));
                        map.insert("factor".into(), Value::List(Arc::new(factor)));
                        map.insert("loading".into(), Value::List(Arc::new(loading)));
                        map.insert("communality".into(), Value::List(Arc::new(comm)));
                        map.insert("uniqueness".into(), Value::List(Arc::new(uniq)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::DFMResult(m) => {
                        let r = &m.result;
                        let mut var = Vec::new();
                        let mut factor = Vec::new();
                        let mut loading = Vec::new();
                        let k = m.var_names.len();
                        let f = r.n_factors;
                        for j in 0..f {
                            for i in 0..k {
                                var.push(Value::Str(m.var_names[i].clone()));
                                factor.push(Value::Str(format!("F{}", j + 1)));
                                loading.push(Value::Float(r.factor_loadings[[i, j]]));
                            }
                        }
                        let n = loading.len();
                        map.insert("variable".into(), Value::List(Arc::new(var)));
                        map.insert("factor".into(), Value::List(Arc::new(factor)));
                        map.insert("loading".into(), Value::List(Arc::new(loading)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::DecompResult(_)
                    | Value::MstlResult(_)
                    | Value::UCResult(_)
                    | Value::MiceResult(_)
                    | Value::LowessResult(_) => {
                        // No coefficient-like parameters; return empty tidy table
                        map.insert("variable".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("coef".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("std_err".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("t".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("p_value".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("conf_low".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("conf_high".into(), Value::List(Arc::new(Vec::new())));
                    }
                    _ => return Err(HayashiError::Type("tidy: unsupported model type".into())),
                }

                let df = self.dict_to_dataframe(&map)?;
                Ok(Value::DataFrame(Arc::new(df)))
            }
            _ => unreachable!(),
        }
    }
}
