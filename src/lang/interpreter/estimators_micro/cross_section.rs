use super::super::helpers::*;
use super::super::*;

impl Interpreter {
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
        let formula_ast = self.resolve_formula(&args[0])?;
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
        let (df, g_formula, display_names) = self.prepare_formula(&formula_ast, &df_raw2)?;
        let cov = resolve_cov_full(opt_map, &df)?;

        let (y, x) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        // Usa fit_with_names para preservar nomes legíveis (e.g. "log(K):log(L)")
        let var_names: Vec<String> = std::iter::once("_cons".to_string())
            .chain(display_names)
            .collect();
        let result = OLS::fit_with_names(&y, &x, cov, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let fitted = result.x_clean.as_ref().unwrap_or(&x).dot(&result.params);
        let residuals = &y - &fitted;
        let x_used = result.x_clean.clone().unwrap_or(x);

        Ok(Value::OlsResult(OlsModel {
            result: Rc::new(result),
            residuals,
            x: x_used,
        }))
    }

    pub(super) fn iv(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "iv() requires (endog_formula, instrument_formula, dataframe)".into(),
            ));
        }
        let endog_ast = self.resolve_formula(&args[0])?;
        let instr_ast = self.resolve_formula(&args[1])?;
        let df_name = match &args[2] {
            Expr::Var(name) => name.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "third argument must be a DataFrame variable".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
        };
        let (df_endog, g_endog, _) = self.prepare_formula(&endog_ast, &df)?;
        let cov = resolve_cov_full(opt_map, &df_endog)?;

        // Instrumento pode ter LHS vazio (~ z1 + z2)
        let g_instr = if instr_ast.lhs.is_empty() {
            let (df_instr, g_i, _) = self.prepare_formula(&instr_ast, &df)?;
            let _ = df_instr; // df aumentado não é necessário para instrumento
            GFormula {
                dependent: String::new(),
                independents: g_i.independents,
                intercept: true,
            }
        } else {
            let (_, g_i, _) = self.prepare_formula(&instr_ast, &df)?;
            g_i
        };

        let result = IV::from_formula(&g_endog, &g_instr, &df_endog, cov)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::IvResult(Rc::new(result)))
    }

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

    pub(super) fn wls(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let (y, x) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result = greeners::WLS::fit_with_names(&y, &x, &weights, cov, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let fitted = x.dot(&result.params);
        let residuals = &y - &fitted;
        Ok(Value::OlsResult(OlsModel {
            result: Rc::new(result),
            residuals,
            x,
        }))
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
        Value::OlsResult(m) => {
            let vnames = m.result.variable_names.as_deref().unwrap_or(&[]);
            let indices: Vec<usize> = tested.iter().map(|v| {
                vnames.iter().position(|n| n == v)
                    .ok_or_else(|| HayashiError::Runtime(
                        format!("testparm: variable '{v}' not found in model")
                    ))
            }).collect::<Result<_>>()?;
            let (f_stat, p_val) = m.result.f_test(&indices, &m.x)
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
            let df1 = indices.len();
            let df2 = m.result.df_resid;
            println!("\n{:=^62}", " testparm — Joint F Test ");
            println!(" H0: {} = 0 (simultaneously)", tested.join(" = "));
            println!("{:-^62}", "");
            println!(" F({df1}, {df2})  =  {f_stat:.4}");
            println!(" Prob > F      =  {p_val:.4}");
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
            map.insert("test".into(), Value::Str("testparm — Joint F Test".into()));
            map.insert("f_stat".into(), Value::Float(f_stat));
            map.insert("df1".into(), Value::Int(df1 as i64));
            map.insert("df2".into(), Value::Int(df2 as i64));
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
            return Err(HayashiError::Runtime("anova(df, outcome, by=grupo)".into()));
        }
        let df_name = match &args[0] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "primeiro argumento deve ser DataFrame".into(),
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
                    "anova() requer by=\"coluna_grupo\"".into(),
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
                    "manova requer by=\"coluna_grupo\"".into(),
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
        println!("{result}");
        Ok(Value::Nil)
    }
}
