use super::super::helpers::*;
use super::super::*;
use crate::lang::dap::model_expansion;

impl Interpreter {
    pub(super) fn rd(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "rd() requer (formula, cutoff, df [, bw=..., poly=..., kernel=...])".into(),
            ));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
        let cutoff = match self.eval_expr(&args[1])? {
            Value::Float(v) => v,
            Value::Int(v) => v as f64,
            _ => {
                return Err(HayashiError::Type(
                    "rd(): second argument must be cutoff (number)".into(),
                ))
            }
        };
        let df = match self.eval_expr(&args[2])? {
            Value::DataFrame(df) => df,
            _ => {
                return Err(HayashiError::Type(
                    "rd(): third argument must be DataFrame".into(),
                ))
            }
        };

        // Extract names directly from Hayashi formula AST
        let outcome_name = formula_ast.lhs.clone();
        let running_name = formula_ast
            .rhs
            .first()
            .and_then(|t| t.as_var().map(|s| s.to_string()))
            .ok_or_else(|| {
                HayashiError::Runtime(
                    "rd(): formula must have exactly one variable on the right side (running var)"
                        .into(),
                )
            })?;

        let y = df
            .get(&outcome_name)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
            .to_owned();
        let x = df
            .get(&running_name)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
            .to_owned();

        let bw = match opt_map.get("bw") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            None => None,
            _ => return Err(HayashiError::Runtime("rd: bw must be numeric".into())),
        };
        let poly = match opt_map.get("poly") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            None => 1,
            _ => return Err(HayashiError::Runtime("rd: poly must be integer".into())),
        };
        let kernel = rd_kernel_opt(opt_map.get("kernel")).map_err(HayashiError::Runtime)?;

        let result = greeners::RD::fit(
            &y,
            &x,
            cutoff,
            bw,
            poly,
            kernel,
            Some((outcome_name, running_name)),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::RdResult(Rc::new(result)))
    }

    pub(super) fn fuzzy_rd(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 4 {
            return Err(HayashiError::Runtime(
                "fuzzy_rd() requer (formula, \"treatment\", cutoff, df [, bw=..., poly=...])"
                    .into(),
            ));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
        let treatment_name = match self.eval_expr(&args[1])? {
            Value::Str(s) => s,
            _ => {
                return Err(HayashiError::Type(
                    "fuzzy_rd(): second argument must be the treatment column name (string)".into(),
                ))
            }
        };
        let cutoff = match self.eval_expr(&args[2])? {
            Value::Float(v) => v,
            Value::Int(v) => v as f64,
            _ => {
                return Err(HayashiError::Type(
                    "fuzzy_rd(): third argument must be cutoff (number)".into(),
                ))
            }
        };
        let df = match self.eval_expr(&args[3])? {
            Value::DataFrame(df) => df,
            _ => {
                return Err(HayashiError::Type(
                    "fuzzy_rd(): fourth argument must be DataFrame".into(),
                ))
            }
        };

        let outcome_name = formula_ast.lhs.clone();
        let running_name = formula_ast.rhs.first()
        .and_then(|t| t.as_var().map(|s| s.to_string()))
        .ok_or_else(|| HayashiError::Runtime(
            "fuzzy_rd(): formula must have exactly one variable on the right side (running var)".into()
        ))?;

        let y = df
            .get(&outcome_name)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
            .to_owned();
        let d = df
            .get(&treatment_name)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
            .to_owned();
        let x = df
            .get(&running_name)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
            .to_owned();

        let bw = match opt_map.get("bw") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            None => None,
            _ => return Err(HayashiError::Runtime("fuzzy_rd: bw must be numeric".into())),
        };
        let poly = match opt_map.get("poly") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            None => 1,
            _ => {
                return Err(HayashiError::Runtime(
                    "fuzzy_rd: poly must be integer".into(),
                ))
            }
        };
        let kernel = rd_kernel_opt(opt_map.get("kernel")).map_err(HayashiError::Runtime)?;

        let result = greeners::RD::fit_fuzzy(
            &y,
            &d,
            &x,
            cutoff,
            bw,
            poly,
            kernel,
            Some((outcome_name, running_name, treatment_name)),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::RdResult(Rc::new(result)))
    }

    pub(super) fn psm(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "psm() requer (formula, df [, k=..., caliper=..., replace=..., boot=...])".into(),
            ));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
        let df = match self.eval_expr(&args[1])? {
            Value::DataFrame(df) => df,
            _ => {
                return Err(HayashiError::Type(
                    "psm(): second argument must be DataFrame".into(),
                ))
            }
        };

        let outcome_name = formula_ast.lhs.clone();
        // First RHS = treatment; remaining = covariates
        let mut rhs_names: Vec<String> = formula_ast
            .rhs
            .iter()
            .filter_map(|t| t.as_var().map(|s| s.to_string()))
            .collect();
        if rhs_names.is_empty() {
            return Err(HayashiError::Runtime(
                "psm(): formula must have at least 'outcome ~ treatment'".into(),
            ));
        }
        let treatment_name = rhs_names.remove(0);
        let covariate_names = rhs_names;

        if covariate_names.is_empty() {
            return Err(HayashiError::Runtime(
                "psm(): provide at least one covariate: outcome ~ treatment + cov1 + ...".into(),
            ));
        }

        let y = df
            .get(&outcome_name)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
            .to_owned();
        let d = df
            .get(&treatment_name)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
            .to_owned();

        let x = {
            let owned_cols: Vec<ndarray::Array1<f64>> = covariate_names
                .iter()
                .map(|c| {
                    df.get(c)
                        .map(|a| a.to_owned())
                        .map_err(|e| HayashiError::Runtime(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            let views: Vec<ndarray::ArrayView1<f64>> =
                owned_cols.iter().map(|a| a.view()).collect();
            ndarray::stack(ndarray::Axis(1), &views)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?
        };

        let k = match opt_map.get("k") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            None => 1,
            _ => return Err(HayashiError::Runtime("psm: k must be integer".into())),
        };
        let caliper: Option<f64> = match opt_map.get("caliper") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            None => None,
            _ => return Err(HayashiError::Runtime("psm: caliper must be numeric".into())),
        };
        let with_replacement = match opt_map.get("replace") {
            Some(Value::Bool(b)) => *b,
            None => false,
            _ => return Err(HayashiError::Runtime("psm: replace must be boolean".into())),
        };
        let n_boot = match opt_map.get("boot") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            None => 200,
            _ => return Err(HayashiError::Runtime("psm: boot must be integer".into())),
        };

        let result = greeners::PSM::fit(
            &y,
            &d,
            &x,
            k,
            caliper,
            with_replacement,
            n_boot,
            Some((outcome_name, treatment_name, covariate_names)),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::PsmResult(Rc::new(result)))
    }

    pub(super) fn synth(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 4 {
            return Err(HayashiError::Runtime(
                "synth() requer (outcome, treated_id, t0, df, id=col, time=col [, covs=[...]])"
                    .into(),
            ));
        }
        let outcome_col = match self.eval_expr(&args[0])? {
            Value::Str(s) => s,
            _ => {
                return Err(HayashiError::Type(
                    "synth(): first argument must be outcome column name (string)".into(),
                ))
            }
        };
        let treated_unit = match self.eval_expr(&args[1])? {
            Value::Str(s) => s,
            Value::Int(v) => v.to_string(),
            Value::Float(v) => (v as i64).to_string(),
            _ => {
                return Err(HayashiError::Type(
                    "synth(): second argument must be treated unit ID".into(),
                ))
            }
        };
        let t0 = match self.eval_expr(&args[2])? {
            Value::Float(v) => v,
            Value::Int(v) => v as f64,
            _ => {
                return Err(HayashiError::Type(
                    "synth(): third argument must be treatment start period (number)".into(),
                ))
            }
        };
        let df = match self.eval_expr(&args[3])? {
            Value::DataFrame(df) => df,
            _ => {
                return Err(HayashiError::Type(
                    "synth(): fourth argument must be DataFrame".into(),
                ))
            }
        };

        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(
                    "synth(): id=coluna option is required".into(),
                ))
            }
        };
        let time_col = match opt_map.get("time") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(
                    "synth(): time=coluna option is required".into(),
                ))
            }
        };
        let cov_cols: Option<Vec<String>> = match opt_map.get("covs") {
            Some(Value::List(lst)) => Some(
                lst.iter()
                    .map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type(
                            "synth(): covs must be a list of strings".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => None,
            _ => return Err(HayashiError::Runtime("synth(): covs must be a list".into())),
        };

        let result = greeners::SyntheticControl::fit(
            &outcome_col,
            &treated_unit,
            t0,
            &df,
            &id_col,
            &time_col,
            cov_cols.as_deref(),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::SynthResult(Rc::new(result)))
    }

    pub(super) fn did(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "did(outcome ~ treated + post, df) requires formula and DataFrame".into(),
            ));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
        let df = match self.eval_expr(&args[1])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "did(): second argument must be DataFrame".into(),
                ))
            }
        };
        // formula: outcome ~ treated_col + post_col
        let rhs_vars: Vec<String> = formula_ast
            .rhs
            .iter()
            .filter_map(|t| t.as_var().map(|s| s.to_string()))
            .collect();
        if rhs_vars.len() < 2 {
            return Err(HayashiError::Runtime(
                "did(): formula must have exactly 2 variables on RHS: treated + post".into(),
            ));
        }
        let y = get_col_f64(&df, &formula_ast.lhs)?;
        let treated = get_col_f64(&df, &rhs_vars[0])?;
        let post = get_col_f64(&df, &rhs_vars[1])?;
        let cov = resolve_cov_full(opt_map, &df)?;
        let result = greeners::DiffInDiff::fit(&y, &treated, &post, cov)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::DidResult(Rc::new(result)))
    }

    pub(super) fn eventstudy(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "eventstudy(y ~ event_time + controls, df) requires formula and DataFrame".into(),
            ));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
        let df = match self.eval_expr(&args[1])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "eventstudy(): second argument must be DataFrame".into(),
                ))
            }
        };
        let rhs_vars: Vec<String> = formula_ast
            .rhs
            .iter()
            .filter_map(|t| t.as_var().map(|s| s.to_string()))
            .collect();
        if rhs_vars.is_empty() {
            return Err(HayashiError::Runtime(
                "eventstudy(): formula must have at least event_time on RHS".into(),
            ));
        }
        let event_col = &rhs_vars[0];
        let y = get_col_f64(&df, &formula_ast.lhs)?;
        let event_vals = get_col_f64(&df, event_col)?;
        let event_time: Vec<i64> = event_vals.iter().map(|&v| v as i64).collect();

        // Controls: remaining RHS vars
        let n = y.len();
        let control_vars = &rhs_vars[1..];
        let x_controls = if control_vars.is_empty() {
            ndarray::Array2::zeros((n, 0))
        } else {
            let mut x = ndarray::Array2::zeros((n, control_vars.len()));
            for (j, v) in control_vars.iter().enumerate() {
                let col = get_col_f64(&df, v)?;
                for i in 0..n {
                    x[(i, j)] = col[i];
                }
            }
            x
        };

        let reference = match opt_map.get("ref") {
            Some(Value::Int(v)) => *v,
            Some(Value::Float(v)) => *v as i64,
            None => -1,
            _ => -1,
        };
        let min_t = match opt_map.get("min") {
            Some(Value::Int(v)) => *v,
            Some(Value::Float(v)) => *v as i64,
            None => -5,
            _ => -5,
        };
        let max_t = match opt_map.get("max") {
            Some(Value::Int(v)) => *v,
            Some(Value::Float(v)) => *v as i64,
            None => 5,
            _ => 5,
        };
        let cov = resolve_cov_full(opt_map, &df)?;
        let result =
            greeners::EventStudy::fit(&y, &event_time, &x_controls, reference, min_t, max_t, cov)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let event_names: Vec<String> = result
            .event_times
            .iter()
            .map(|&t| format!("t={t}"))
            .collect();
        let mut ols_names = vec!["const".to_string()];
        ols_names.extend(event_names.clone());
        ols_names.extend(control_vars.iter().cloned());

        let event_coef = model_expansion::coef_dataframe(
            &event_names,
            &result.event_coefs,
            &result.event_se,
            &result.event_t,
            &result.event_p,
            None,
            None,
        );
        let full_ols_coef = model_expansion::coef_dataframe(
            &ols_names,
            &result.ols.params,
            &result.ols.std_errors,
            &result.ols.t_values,
            &result.ols.p_values,
            None,
            None,
        );

        let summary = format!(
            "EventStudy(ref={}, k={}, n={})",
            result.reference,
            result.event_coefs.len(),
            result.ols.n_obs
        );
        let fields: Vec<(String, Value)> = vec![
            ("event_coefficients".into(), event_coef),
            ("full_coefficients".into(), full_ols_coef),
            (
                "event_col_indices".into(),
                model_expansion::int_series("event_col_indices", &result.event_col_indices),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("reference", Value::Int(result.reference)),
                    ("n_obs", Value::Int(result.ols.n_obs as i64)),
                    ("r2", Value::Float(result.ols.r_squared)),
                    ("adj_r2", Value::Float(result.ols.adj_r_squared)),
                    ("sigma", Value::Float(result.ols.sigma)),
                    ("df_resid", Value::Int(result.ols.df_resid as i64)),
                    ("df_model", Value::Int(result.ols.df_model as i64)),
                    ("f_statistic", Value::Float(result.ols.f_statistic)),
                    ("log_likelihood", Value::Float(result.ols.log_likelihood)),
                    ("aic", Value::Float(result.ols.aic)),
                    ("bic", Value::Float(result.ols.bic)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "EventStudyResult",
            fields,
        ))
    }

    pub(super) fn double_ml(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        // First RHS variable is treatment (d), rest are controls (x)
        let independents = &g_formula.independents;
        if independents.len() < 2 {
            return Err(HayashiError::Runtime(
                "double_ml() requires y ~ d + x1 + x2 + ... (treatment + controls)".into(),
            ));
        }
        let d_var = &independents[0];
        let x_vars = &independents[1..];

        let n = df.n_rows();
        let y_vec = get_col_f64(&df, &g_formula.dependent)?;
        let d_vec = get_col_f64(&df, d_var)?;
        let mut x_mat = ndarray::Array2::zeros((n, x_vars.len()));
        for (j, v) in x_vars.iter().enumerate() {
            let col = get_col_f64(&df, v)?;
            for i in 0..n {
                x_mat[(i, j)] = col[i];
            }
        }

        let n_folds = match opt_map.get("folds") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            None => 5,
            _ => 5,
        };
        let poly_degree = match opt_map.get("poly") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            None => 2,
            _ => 2,
        };

        let result = greeners::DoubleML::fit_plr(&y_vec, &d_vec, &x_mat, n_folds, poly_degree)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let theta_names = vec!["theta".to_string()];
        let theta_params = ndarray::Array1::from(vec![result.theta]);
        let theta_se = ndarray::Array1::from(vec![result.std_error]);
        let theta_t = ndarray::Array1::from(vec![result.t_value]);
        let theta_p = ndarray::Array1::from(vec![result.p_value]);
        let coefficients = model_expansion::coef_dataframe(
            &theta_names,
            &theta_params,
            &theta_se,
            &theta_t,
            &theta_p,
            None,
            None,
        );

        let summary = format!("DoubleML(theta={:.4}, n={})", result.theta, result.n_obs);
        let fields: Vec<(String, Value)> = vec![
            ("coefficients".into(), coefficients),
            (
                "y_tilde".into(),
                model_expansion::array1_to_series("y_tilde", &result.y_tilde),
            ),
            (
                "d_tilde".into(),
                model_expansion::array1_to_series("d_tilde", &result.d_tilde),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_folds", Value::Int(result.n_folds as i64)),
                    ("theta", Value::Float(result.theta)),
                    ("std_error", Value::Float(result.std_error)),
                    ("t_value", Value::Float(result.t_value)),
                    ("p_value", Value::Float(result.p_value)),
                    ("ci_low", Value::Float(result.ci_low)),
                    ("ci_high", Value::Float(result.ci_high)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "DoubleMLResult",
            fields,
        ))
    }

    pub(super) fn synthdid(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "synthdid(df, y_var, treated_var, treatment_period)".into(),
            ));
        }
        let df_name = match &args[0] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "synthdid: first arg must be DataFrame".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };

        // Parse: df, y_var, treated_var, treatment_period
        // args[0] = df, args[1] = y_var, args[2] = treated_var, args[3] = treatment_period
        let y_var = match &args[1] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "synthdid: y_var must be identifier".into(),
                ))
            }
        };
        let treated_var = match &args[2] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "synthdid: treated_var must be identifier".into(),
                ))
            }
        };
        let treatment_period = match &args[3] {
            Expr::Int(v) => *v as usize,
            Expr::Float(v) => *v as usize,
            _ => {
                return Err(HayashiError::Type(
                    "synthdid: treatment_period must be integer".into(),
                ))
            }
        };

        // Build outcome matrix (units x periods)
        // Need unit and period columns
        let unit_col = match opt_map.get("unit") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires unit=\"column\" option"
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

        // Extract unique units and periods
        let unit_col_data = df
            .get_column(unit_col.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let period_col_data = df
            .get_column(period_col.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let units: Vec<i64> = if let Some(i) = unit_col_data.as_int() {
            i.to_vec()
        } else if let Some(f) = unit_col_data.as_float() {
            f.iter().map(|v| *v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "{func}: unit must be numeric"
            )));
        };
        let periods: Vec<i64> = if let Some(i) = period_col_data.as_int() {
            i.to_vec()
        } else if let Some(f) = period_col_data.as_float() {
            f.iter().map(|v| *v as i64).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "{func}: period must be numeric"
            )));
        };

        let y_col = df
            .get_column(y_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;

        let treated_col = df
            .get_column(treated_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let treated_vals: Vec<bool> = if let Some(b) = treated_col.as_bool() {
            b.to_vec()
        } else if let Some(i) = treated_col.as_int() {
            i.iter().map(|&v| v != 0).collect()
        } else if let Some(f) = treated_col.as_float() {
            f.iter().map(|&v| v != 0.0).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "{func}: '{treated_var}' must be boolean or numeric"
            )));
        };

        // Get unique units and periods
        let mut unique_units: Vec<i64> = units
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_units.sort();
        let mut unique_periods: Vec<i64> = periods
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_periods.sort();

        let n_units = unique_units.len();
        let n_periods = unique_periods.len();

        // Build outcome matrix
        let mut y_mat = ndarray::Array2::<f64>::zeros((n_units, n_periods));
        let mut treated_vec = vec![false; n_units];

        let unit_to_idx: std::collections::HashMap<i64, usize> = unique_units
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect();
        let period_to_idx: std::collections::HashMap<i64, usize> = unique_periods
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();

        for row in 0..df.n_rows() {
            let ui = unit_to_idx[&units[row]];
            let pi = period_to_idx[&periods[row]];
            y_mat[(ui, pi)] = y_vals[row];
            if treated_vals[row] {
                treated_vec[ui] = true;
            }
        }

        let result = greeners::SyntheticDiD::fit(&y_mat, &treated_vec, treatment_period)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let att_names = vec!["ATT".to_string()];
        let att_params = ndarray::Array1::from(vec![result.att]);
        let att_se = ndarray::Array1::from(vec![result.se]);
        let att_t = ndarray::Array1::from(vec![result.t_stat]);
        let att_p = ndarray::Array1::from(vec![result.p_value]);
        let coefficients = model_expansion::coef_dataframe(
            &att_names,
            &att_params,
            &att_se,
            &att_t,
            &att_p,
            None,
            None,
        );
        let gap = &result.treated_avg - &result.synthetic_control;

        let summary = format!(
            "SyntheticDiD(att={:.4}, n_pre={}, n_post={})",
            result.att, result.n_pre, result.n_post
        );
        let fields: Vec<(String, Value)> = vec![
            ("coefficients".into(), coefficients),
            (
                "unit_weights".into(),
                model_expansion::array1_to_series("unit_weights", &result.unit_weights),
            ),
            (
                "time_weights".into(),
                model_expansion::array1_to_series("time_weights", &result.time_weights),
            ),
            (
                "treated_avg".into(),
                model_expansion::array1_to_series("treated_avg", &result.treated_avg),
            ),
            (
                "synthetic_control".into(),
                model_expansion::array1_to_series("synthetic_control", &result.synthetic_control),
            ),
            ("gap".into(), model_expansion::array1_to_series("gap", &gap)),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("att", Value::Float(result.att)),
                    ("se", Value::Float(result.se)),
                    ("t_stat", Value::Float(result.t_stat)),
                    ("p_value", Value::Float(result.p_value)),
                    ("n_treated", Value::Int(result.n_treated as i64)),
                    ("n_control", Value::Int(result.n_control as i64)),
                    ("n_pre", Value::Int(result.n_pre as i64)),
                    ("n_post", Value::Int(result.n_post as i64)),
                    ("n_periods", Value::Int(result.n_periods as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "SyntheticDidResult",
            fields,
        ))
    }

    pub(super) fn cuped(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        let treated_col = match opt_map.get("treated") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires treated=\"column\" option"
                )))
            }
        };

        // y = dependent, x = pre-treatment covariate
        let y_col = df
            .get_column(g_formula.dependent.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let y_vals = y_col.as_float().ok_or_else(|| {
            HayashiError::Runtime(format!("{func}: '{}' must be numeric", g_formula.dependent))
        })?;

        let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

        // Pre-treatment covariate: first independent
        let x_var = g_formula
            .independents
            .first()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: need at least 1 covariate")))?
            .clone();
        let x_col = df
            .get_column(x_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let x_vals = x_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{x_var}' must be numeric")))?;
        let x_arr = ndarray::Array1::from_vec(x_vals.to_vec());

        // Treatment indicator
        let treated_data = df
            .get_column(treated_col.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let treated_vec: Vec<bool> = if let Some(b) = treated_data.as_bool() {
            b.to_vec()
        } else if let Some(i) = treated_data.as_int() {
            i.iter().map(|&v| v != 0).collect()
        } else if let Some(f) = treated_data.as_float() {
            f.iter().map(|&v| v != 0.0).collect()
        } else {
            return Err(HayashiError::Runtime(format!(
                "{func}: '{treated_col}' must be boolean or numeric"
            )));
        };

        let result = greeners::CUPED::fit(&y_arr, &x_arr, &treated_vec)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let effect_names = vec!["treatment".to_string()];
        let effect_params = ndarray::Array1::from(vec![result.treatment_effect]);
        let effect_se = ndarray::Array1::from(vec![result.se]);
        let effect_t = ndarray::Array1::from(vec![result.t_stat]);
        let effect_p = ndarray::Array1::from(vec![result.p_value]);
        let coefficients = model_expansion::coef_dataframe(
            &effect_names,
            &effect_params,
            &effect_se,
            &effect_t,
            &effect_p,
            None,
            None,
        );

        let summary = format!(
            "CUPED(effect={:.4}, n={})",
            result.treatment_effect, result.n_obs
        );
        let fields: Vec<(String, Value)> = vec![
            ("coefficients".into(), coefficients),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("treatment_effect", Value::Float(result.treatment_effect)),
                    ("se", Value::Float(result.se)),
                    ("t_stat", Value::Float(result.t_stat)),
                    ("p_value", Value::Float(result.p_value)),
                    ("ci_low", Value::Float(result.ci[0])),
                    ("ci_high", Value::Float(result.ci[1])),
                    ("theta", Value::Float(result.theta)),
                    ("unadjusted_effect", Value::Float(result.unadjusted_effect)),
                    ("unadjusted_se", Value::Float(result.unadjusted_se)),
                    (
                        "variance_reduction",
                        Value::Float(result.variance_reduction),
                    ),
                    ("adjusted_variance", Value::Float(result.adjusted_variance)),
                    (
                        "unadjusted_variance",
                        Value::Float(result.unadjusted_variance),
                    ),
                    ("n_treatment", Value::Int(result.n_treatment as i64)),
                    ("n_control", Value::Int(result.n_control as i64)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "CupedResult",
            fields,
        ))
    }

    pub(super) fn dml_crossfit(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        // y = dependent, d = first independent (treatment)
        let y_var = g_formula.dependent.clone();
        let d_var = g_formula
            .independents
            .first()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: need treatment variable")))?
            .clone();

        // Confounders from x option
        let x_str = match opt_map.get("x") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires x=\"x1,x2\" option (confounders)"
                )))
            }
        };
        let x_vars: Vec<String> = x_str.split(',').map(|s| s.trim().to_string()).collect();
        let n_folds = match opt_map.get("folds") {
            Some(Value::Int(v)) => Some(*v as usize),
            Some(Value::Float(v)) => Some(*v as usize),
            _ => None,
        };

        let n = df.n_rows();
        let y_col = df
            .get_column(y_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
        let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

        let d_col = df
            .get_column(d_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let d_vals = d_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{d_var}' must be numeric")))?;
        let d_arr = ndarray::Array1::from_vec(d_vals.to_vec());

        let k = x_vars.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
        for (j, xname) in x_vars.iter().enumerate() {
            let col = df
                .get_column(xname.as_str())
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let vals = col.as_float().ok_or_else(|| {
                HayashiError::Runtime(format!("{func}: '{xname}' must be numeric"))
            })?;
            for i in 0..n {
                x_mat[(i, j)] = vals[i];
            }
        }

        let result = greeners::DMLCrossfit::fit(&y_arr, &d_arr, &x_mat, n_folds)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let theta_names = vec!["theta".to_string()];
        let theta_params = ndarray::Array1::from(vec![result.theta]);
        let theta_se = ndarray::Array1::from(vec![result.se]);
        let theta_t = ndarray::Array1::from(vec![result.t_stat]);
        let theta_p = ndarray::Array1::from(vec![result.p_value]);
        let coefficients = model_expansion::coef_dataframe(
            &theta_names,
            &theta_params,
            &theta_se,
            &theta_t,
            &theta_p,
            None,
            None,
        );

        let summary = format!("DMLCrossfit(theta={:.4}, n={})", result.theta, result.n_obs);
        let fields: Vec<(String, Value)> = vec![
            ("coefficients".into(), coefficients),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("theta", Value::Float(result.theta)),
                    ("se", Value::Float(result.se)),
                    ("t_stat", Value::Float(result.t_stat)),
                    ("p_value", Value::Float(result.p_value)),
                    ("ci_low", Value::Float(result.ci[0])),
                    ("ci_high", Value::Float(result.ci[1])),
                    ("n_folds", Value::Int(result.n_folds as i64)),
                    ("g_mse", Value::Float(result.g_mse)),
                    ("m_mse", Value::Float(result.m_mse)),
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_confounders", Value::Int(result.n_confounders as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "DmlResult",
            fields,
        ))
    }

    pub(super) fn bsc(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 4 {
            return Err(HayashiError::Runtime(
                "bsc(df, y_var, control_vars, treatment_period [, prior=1.0])".into(),
            ));
        }
        let df_name = match &args[0] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "bsc: first arg must be DataFrame".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let y_var = match &args[1] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => return Err(HayashiError::Type("bsc: y_var must be identifier".into())),
        };
        let control_vars_str = match &args[2] {
            Expr::Str(s) => s.clone(),
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "bsc: control_vars must be string".into(),
                ))
            }
        };
        let treatment_period = match &args[3] {
            Expr::Int(v) => *v as usize,
            Expr::Float(v) => *v as usize,
            _ => {
                return Err(HayashiError::Type(
                    "bsc: treatment_period must be integer".into(),
                ))
            }
        };
        let prior = match opt_map.get("prior") {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(*v as f64),
            _ => None,
        };

        let control_vars: Vec<String> = control_vars_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let n = df.n_rows();
        let n_controls = control_vars.len();

        let y_col = df
            .get_column(y_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
        let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

        let mut y_controls = ndarray::Array2::<f64>::zeros((n, n_controls));
        for (j, cname) in control_vars.iter().enumerate() {
            let col = df
                .get_column(cname.as_str())
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let vals = col.as_float().ok_or_else(|| {
                HayashiError::Runtime(format!("{func}: '{cname}' must be numeric"))
            })?;
            for i in 0..n {
                y_controls[(i, j)] = vals[i];
            }
        }

        let result = greeners::BayesianSC::fit(&y_arr, &y_controls, treatment_period, prior)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let tau_names = vec!["tau".to_string()];
        let tau_params = ndarray::Array1::from(vec![result.tau]);
        let tau_sd = ndarray::Array1::from(vec![result.tau_sd]);
        let tau_t = ndarray::Array1::from(vec![result.t_stat]);
        let tau_p = ndarray::Array1::from(vec![result.p_value]);
        let coefficients = model_expansion::coef_dataframe(
            &tau_names,
            &tau_params,
            &tau_sd,
            &tau_t,
            &tau_p,
            None,
            None,
        );
        let effect = &result.observed - &result.counterfactual;

        let summary = format!(
            "BayesianSC(tau={:.4}, n_controls={}, n_pre={})",
            result.tau, result.n_controls, result.n_pre
        );
        let fields: Vec<(String, Value)> = vec![
            ("coefficients".into(), coefficients),
            (
                "weights".into(),
                model_expansion::array1_to_series("weights", &result.weights),
            ),
            (
                "observed".into(),
                model_expansion::array1_to_series("observed", &result.observed),
            ),
            (
                "counterfactual".into(),
                model_expansion::array1_to_series("counterfactual", &result.counterfactual),
            ),
            (
                "effect".into(),
                model_expansion::array1_to_series("effect", &effect),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("tau", Value::Float(result.tau)),
                    ("tau_sd", Value::Float(result.tau_sd)),
                    ("sigma2", Value::Float(result.sigma2)),
                    ("p_value", Value::Float(result.p_value)),
                    ("t_stat", Value::Float(result.t_stat)),
                    ("log_marginal", Value::Float(result.log_marginal)),
                    ("cumulative_effect", Value::Float(result.cumulative_effect)),
                    ("tau_ci_low", Value::Float(result.tau_ci[0])),
                    ("tau_ci_high", Value::Float(result.tau_ci[1])),
                    ("n_controls", Value::Int(result.n_controls as i64)),
                    ("n_pre", Value::Int(result.n_pre as i64)),
                    ("n_post", Value::Int(result.n_post as i64)),
                    ("prior_precision", Value::Float(result.prior_precision)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "BayesianScResult",
            fields,
        ))
    }

    pub(super) fn causal_impact(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(format!(
                "{func}() requires (df, y_var)"
            )));
        }
        let df_name = match &args[0] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(format!(
                    "{func}: first arg must be DataFrame"
                )))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let y_var = match &args[1] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(format!(
                    "{func}: var must be identifier"
                )))
            }
        };

        let controls_str = match opt_map.get("controls") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires controls=\"c1,c2\" option"
                )))
            }
        };
        let control_vars: Vec<String> = controls_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let treatment_period = match opt_map.get("period") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "{func}() requires period=N option (treatment start index)"
                )))
            }
        };

        let n = df.n_rows();
        let y_col = df
            .get_column(y_var.as_str())
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let y_vals = y_col
            .as_float()
            .ok_or_else(|| HayashiError::Runtime(format!("{func}: '{y_var}' must be numeric")))?;
        let y_arr = ndarray::Array1::from_vec(y_vals.to_vec());

        let kk = control_vars.len();
        let mut controls_mat = ndarray::Array2::<f64>::zeros((n, kk));
        for (j, cname) in control_vars.iter().enumerate() {
            let col = df
                .get_column(cname.as_str())
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let vals = col.as_float().ok_or_else(|| {
                HayashiError::Runtime(format!("{func}: '{cname}' must be numeric"))
            })?;
            for i in 0..n {
                controls_mat[(i, j)] = vals[i];
            }
        }

        let result = greeners::CausalImpact::fit(
            &y_arr,
            &controls_mat,
            treatment_period,
            Some(control_vars),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let mut coef_names = vec!["const".to_string()];
        coef_names.extend(result.control_names.clone());
        let control_coef = model_expansion::coefficients_df(&coef_names, &result.coefficients);

        let avg_names = vec!["avg_effect".to_string()];
        let avg_mean = ndarray::Array1::from(vec![result.avg_effect]);
        let avg_sd = ndarray::Array1::from(vec![result.avg_effect_sd]);
        let avg_ci_low = ndarray::Array1::from(vec![result.avg_effect_ci[0]]);
        let avg_ci_high = ndarray::Array1::from(vec![result.avg_effect_ci[1]]);
        let avg_p = ndarray::Array1::from(vec![result.p_effect_positive]);
        let avg_effect_df = model_expansion::posterior_coef_df(
            &avg_names,
            &avg_mean,
            &avg_sd,
            &avg_ci_low,
            &avg_ci_high,
            &avg_p,
        );

        let summary = format!(
            "CausalImpact(avg_effect={:.4}, n_pre={}, n_post={})",
            result.avg_effect, result.n_pre, result.n_post
        );
        let fields: Vec<(String, Value)> = vec![
            ("control_coefficients".into(), control_coef),
            ("avg_effect".into(), avg_effect_df),
            (
                "y".into(),
                model_expansion::array1_to_series("y", &result.y),
            ),
            (
                "counterfactual".into(),
                model_expansion::array1_to_series("counterfactual", &result.counterfactual),
            ),
            (
                "counterfactual_sd".into(),
                model_expansion::array1_to_series("counterfactual_sd", &result.counterfactual_sd),
            ),
            (
                "pointwise_effect".into(),
                model_expansion::array1_to_series("pointwise_effect", &result.pointwise_effect),
            ),
            (
                "cumulative_effect".into(),
                model_expansion::array1_to_series("cumulative_effect", &result.cumulative_effect),
            ),
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("avg_effect", Value::Float(result.avg_effect)),
                    ("avg_effect_sd", Value::Float(result.avg_effect_sd)),
                    ("p_effect_positive", Value::Float(result.p_effect_positive)),
                    ("total_effect", Value::Float(result.total_effect)),
                    ("total_effect_sd", Value::Float(result.total_effect_sd)),
                    (
                        "total_effect_ci_low",
                        Value::Float(result.total_effect_ci[0]),
                    ),
                    (
                        "total_effect_ci_high",
                        Value::Float(result.total_effect_ci[1]),
                    ),
                    ("pre_r_squared", Value::Float(result.pre_r_squared)),
                    ("n_pre", Value::Int(result.n_pre as i64)),
                    ("n_post", Value::Int(result.n_post as i64)),
                ]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "CausalImpactResult",
            fields,
        ))
    }
}
