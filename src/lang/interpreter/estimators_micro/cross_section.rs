use super::super::helpers::*;
#[allow(unused_imports)]
use super::super::models::OlsModel;
use super::super::*;
use crate::lang::dap::model_expansion;

impl Interpreter {
    /// Expands categorical names and restores display names for materialised terms.
    pub(super) fn expanded_formula_names(
        df: &DataFrame,
        formula: &GFormula,
        display_names: &[String],
    ) -> Result<Vec<String>> {
        let mut names = df
            .formula_var_names(formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        for name in &mut names {
            for (column, display) in formula.independents.iter().zip(display_names) {
                if name == column {
                    *name = display.clone();
                    break;
                }
                if column.starts_with("C(") {
                    if let Some(suffix) = name.strip_prefix(&format!("{column}_")) {
                        *name = format!("{display}_{suffix}");
                        break;
                    }
                }
            }
        }
        Ok(names)
    }

    #[cfg(feature = "greeners-ols")]
    pub(super) fn ols(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "ols() requires (formula, dataframe)".into(),
            ));
        }
        let formula_ast = self.resolve_formula_allow_no_intercept(&args[0])?;
        let df_name = match &args[1] {
            Expr::Var(name) => name.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "second argument must be a DataFrame variable".into(),
                ))
            }
        };
        let df_raw = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let df_raw2 = self.maybe_filter_df(&df_raw, opts)?;
        let (df, g_formula, display_names) =
            self.prepare_formula_allow_no_intercept(&formula_ast, &df_raw2)?;
        let cov = resolve_cov_full(opt_map, &df)?;

        let (y, x) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let mut var_names = Self::expanded_formula_names(&df, &g_formula, &display_names)?;
        if g_formula.intercept {
            if let Some(intercept) = var_names.first_mut() {
                *intercept = "_cons".into();
            }
        }
        let result = OLS::fit_with_names(&y, &x, cov, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let fitted = result.x_clean.as_ref().unwrap_or(&x).dot(&result.params);
        let residuals = &y - &fitted;
        let x_used = result.x_clean.clone().unwrap_or(x);

        Ok(Value::Model(Rc::new(OlsModel {
            result: Rc::new(result),
            residuals,
            x: x_used,
        })))
    }

    #[cfg(feature = "greeners-ols")]
    pub(super) fn iv(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let mut prepared = self.prepare_iv(args, opts)?;
        let cov = resolve_cov_full(opt_map, &prepared.structural_frame)?;
        let result = IV::fit_with_names(
            &prepared.y,
            &prepared.x,
            &prepared.z,
            cov,
            // The backend owns its names; metadata retains the original order
            // for validating omitted positions and rematerialised predictions.
            Some(prepared.design.names.clone()),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        prepared.design.retain_fitted_columns(&result)?;
        Ok(Value::IvResult(Rc::new(super::super::models::IvModel {
            result,
            design: prepared.design,
        })))
    }

    #[cfg(feature = "greeners-ols")]
    pub(super) fn qreg(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let tau = match opt_map.get("tau") {
            Some(Value::Float(v)) => *v,
            Some(Value::Int(v)) => *v as f64,
            None => 0.5,
            _ => return Err(HayashiError::Type("tau= must be numeric".into())),
        };
        let n_boot = match opt_map.get("boot") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            None => 200,
            _ => return Err(HayashiError::Type("boot= must be integer".into())),
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result =
            greeners::QuantileReg::fit_with_names(&y_vec, &x_mat, tau, n_boot, Some(var_names))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::QuantileResult(Rc::new(result)))
    }

    #[cfg(feature = "greeners-ols")]
    pub(super) fn wls(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, display_names) = self.prepare_formula(&formula_ast, &df)?;
        let w_name = match opt_map.get("weights") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "wls() requires weights=\"weights_column\"".into(),
                ))
            }
            _ => return Err(HayashiError::Type("weights= must be string".into())),
        };
        let weights = get_col_f64(&df, &w_name)?;
        let cov = resolve_cov_full(opt_map, &df)?;
        let var_names = Self::expanded_formula_names(&df, &g_formula, &display_names)?;
        let (y, x) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result = greeners::WLS::fit_with_names(&y, &x, &weights, cov, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        // Omitted positions refer to the original columns, so retain those
        // columns directly without reusing the backend's weighted design.
        let x = if !result.omitted_vars.is_empty() {
            let retained: Vec<usize> = (0..x.ncols())
                .filter(|index| {
                    !result
                        .omitted_vars
                        .iter()
                        .any(|(position, _)| position == index)
                })
                .collect();
            x.select(Axis(1), &retained)
        } else {
            x
        };
        let fitted = x.dot(&result.params);
        let residuals = &y - &fitted;
        Ok(Value::Model(Rc::new(OlsModel {
            result: Rc::new(result),
            residuals,
            x,
        })))
    }

    pub(super) fn testparm(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "testparm(model, [\"x1\", \"x2\"]) requires model + list of variables".into(),
            ));
        }
        let model_val = self.eval_expr(&args[0])?;
        let tested: Vec<String> = match self.eval_expr(&args[1])? {
            Value::List(lst) => lst
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(HayashiError::Type(
                        "testparm: list must contain strings".into(),
                    )),
                })
                .collect::<Result<_>>()?,
            _ => {
                return Err(HayashiError::Type(
                    "testparm: second argument must be list of strings".into(),
                ))
            }
        };
        match &model_val {
        Value::Model(m) => {
            let m = m
                .as_any()
                .downcast_ref::<OlsModel>()
                .ok_or_else(|| self.rt_err("expected an OLS model".to_string()))?;
            let vnames = m.result.variable_names.as_deref().unwrap_or(&[]);
            let indices: Vec<usize> = tested.iter().map(|v| {
                vnames.iter().position(|n| n == v)
                    .ok_or_else(|| HayashiError::Runtime(
                        format!("testparm: variable '{v}' not found in model")
                    ))
            }).collect::<Result<_>>()?;
            let (f_stat, fitted_p_val) = m.result.f_test(&indices, &m.x)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let df1 = indices.len();
            let df2 = m.result.df_resid;
            let normal = matches!(m.result.inference_type, greeners::InferenceType::Normal);
            let (test_name, distribution, p_val) = if normal {
                let reference = statrs::distribution::ChiSquared::new(df1 as f64)
                    .map_err(|e| self.rt_err(format!("testparm: {e}")))?;
                ("testparm - Scaled Wald Test", "chi2", reference.sf(f_stat * df1 as f64))
            } else {
                ("testparm - Joint F Test", "F", fitted_p_val)
            };
            println!("\n{:=^62}", format!(" {test_name} "));
            println!(" H0: {} = 0 (simultaneously)", tested.join(" = "));
            println!("{:-^62}", "");
            if normal {
                println!(" Scaled Wald Q/{df1} = {f_stat:.4}");
                println!(" Reference: chi-square({df1}) at Q");
            } else {
                println!(" F({df1}, {df2})  =  {f_stat:.4}");
            }
            println!(" p-value       =  {p_val:.4}");
            let verdict = if p_val < 0.01 {
                "rejects H0 at 1%"
            } else if p_val < 0.05 {
                "rejects H0 at 5%"
            } else if p_val < 0.10 {
                "rejects H0 at 10%"
            } else {
                "does not reject H0 at 10%"
            };
            println!(" Result: {verdict}");
            println!("{:=^62}", "");
            let mut map = HashMap::new();
            map.insert("test".into(), Value::Str(test_name.into()));
            map.insert("reference_distribution".into(), Value::Str(distribution.into()));
            map.insert("f_stat".into(), Value::Float(f_stat));
            map.insert("df1".into(), Value::Int(df1 as i64));
            if !normal {
                map.insert("df2".into(), Value::Int(df2 as i64));
            }
            map.insert("p_value".into(), Value::Float(p_val));
            map.insert("variables".into(), Value::List(Arc::new(
                tested.into_iter().map(Value::Str).collect()
            )));
            map.insert("conclusion".into(), Value::Str(verdict.into()));
            Ok(Value::Dict(Arc::new(map)))
        }
        _ => Err(HayashiError::Runtime(
            "testparm: current support only for OLS/WLS — other models use chi2; implement via wald_test()".into()
        )),
    }
    }

    pub(super) fn anova(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime("anova(df, outcome, by=group)".into()));
        }
        let df_name = match &args[0] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "anova(): first argument must be a DataFrame".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let outcome_name = match &args[1] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "second argument must be outcome variable name".into(),
                ))
            }
        };
        let outcome = get_col_f64(&df, &outcome_name)?;
        let by_col = match opt_map.get("by") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "anova() requires by=\"group_column\"".into(),
                ))
            }
            _ => return Err(HayashiError::Type("by= must be string".into())),
        };
        let group_vals = get_col_f64(&df, &by_col)?;
        let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut next_g = 0usize;
        let groups: ndarray::Array1<usize> = group_vals
            .iter()
            .map(|&v| {
                let key = v as i64;
                *gmap.entry(key).or_insert_with(|| {
                    let g = next_g;
                    next_g += 1;
                    g
                })
            })
            .collect();
        let result = greeners::Stats::anova_oneway(&outcome, &groups)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        println!("{result}");
        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("One-Way ANOVA".into()));
        map.insert("ss_between".into(), Value::Float(result.ss_between));
        map.insert("ss_within".into(), Value::Float(result.ss_within));
        map.insert("ss_total".into(), Value::Float(result.ss_total));
        map.insert("df_between".into(), Value::Int(result.df_between as i64));
        map.insert("df_within".into(), Value::Int(result.df_within as i64));
        map.insert("ms_between".into(), Value::Float(result.ms_between));
        map.insert("ms_within".into(), Value::Float(result.ms_within));
        map.insert("f_stat".into(), Value::Float(result.f_statistic));
        map.insert("p_value".into(), Value::Float(result.p_value));
        map.insert("n_groups".into(), Value::Int(result.n_groups as i64));
        map.insert("n_obs".into(), Value::Int(result.n_obs as i64));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn manova(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "manova(df, y1, y2, ..., by=\"group_col\")".into(),
            ));
        }
        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "manova: first argument must be a DataFrame".into(),
                ))
            }
        };
        let group_col = match opt_map.get("by") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "manova() requires by=\"group_column\"".into(),
                ))
            }
            _ => return Err(HayashiError::Type("manova: by= must be string".into())),
        };
        let outcome_names = self.resolve_var_list(&args[1..], &df)?;
        let n = df.n_rows();
        let q = outcome_names.len();
        let mut y_mat = ndarray::Array2::<f64>::zeros((n, q));
        for (j, vname) in outcome_names.iter().enumerate() {
            let col = get_col_f64(&df, vname)?;
            for (i, &v) in col.iter().enumerate() {
                y_mat[[i, j]] = v;
            }
        }
        let group_vals = get_col_f64(&df, &group_col)?;
        let mut gmap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut gnext = 0usize;
        let groups: ndarray::Array1<usize> = ndarray::Array1::from(
            group_vals
                .iter()
                .map(|&v| {
                    let key = v as i64;
                    *gmap.entry(key).or_insert_with(|| {
                        let g = gnext;
                        gnext += 1;
                        g
                    })
                })
                .collect::<Vec<_>>(),
        );
        let result = greeners::MANOVA::fit(&y_mat, &groups)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let summary = format!(
            "MANOVA(groups={}, vars={}, n={}), Wilks={:.4}, Pillai={:.4}",
            result.n_groups, result.n_vars, result.n_obs, result.wilks_lambda, result.pillai_trace
        );
        let test_names: Vec<String> = vec![
            "Wilks' Lambda".into(),
            "Pillai's trace".into(),
            "Hotelling-Lawley".into(),
            "Roy's largest root".into(),
        ];
        let fields = vec![
            (
                "fit".into(),
                model_expansion::fit_dict(&[
                    ("n_obs", Value::Int(result.n_obs as i64)),
                    ("n_groups", Value::Int(result.n_groups as i64)),
                    ("n_vars", Value::Int(result.n_vars as i64)),
                    ("wilks_lambda", Value::Float(result.wilks_lambda)),
                    ("pillai_trace", Value::Float(result.pillai_trace)),
                    ("hotelling_lawley", Value::Float(result.hotelling_lawley)),
                    ("roys_largest_root", Value::Float(result.roys_largest_root)),
                ]),
            ),
            (
                "test_names".into(),
                Value::List(Arc::new(test_names.into_iter().map(Value::Str).collect())),
            ),
            (
                "f_values".into(),
                model_expansion::series_from_vec("f_values", &result.f_values[..]),
            ),
            (
                "p_values".into(),
                model_expansion::series_from_vec("p_values", &result.p_values[..]),
            ),
        ];
        Ok(model_expansion::model_result(
            result.to_string(),
            summary,
            "ManovaResult",
            fields,
        ))
    }
}
