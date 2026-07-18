use super::super::helpers::*;
use super::super::*;

impl Interpreter {
    pub(super) fn logit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y, x) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result = Logit::from_formula(&g_formula, &df)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let coef_names = coef_names_from_formula(&formula_ast, &df, x.ncols());
        Ok(Value::BinaryResult(BinaryModel {
            result: Rc::new(result),
            y,
            x,
            kind: "logit".into(),
            coef_names,
        }))
    }

    pub(super) fn probit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y, x) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result = Probit::from_formula(&g_formula, &df)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let coef_names = coef_names_from_formula(&formula_ast, &df, x.ncols());
        Ok(Value::BinaryResult(BinaryModel {
            result: Rc::new(result),
            y,
            x,
            kind: "probit".into(),
            coef_names,
        }))
    }

    pub(super) fn heckman(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "heckman() requires (outcome_formula, selection_formula, df)".into(),
            ));
        }
        let out_ast = self.resolve_formula(&args[0])?;
        let sel_ast = self.resolve_formula(&args[1])?;
        let df_name = match &args[2] {
            Expr::Var(name) => name.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "heckman(): third argument must be DataFrame name".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "heckman: '{df_name}' is not a DataFrame"
                )))
            }
        };

        // Outcome equation
        let (df_out, g_out, out_display) = self.prepare_formula(&out_ast, &df)?;
        let (y_vec_raw, x_out) = df_out
            .to_design_matrix(&g_out)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let out_names = {
            let mut n = vec!["_cons".to_string()];
            n.extend(out_display);
            n
        };

        // Selection equation
        let (df_sel, g_sel, sel_display) = self.prepare_formula(&sel_ast, &df)?;
        let (z_vec, x_sel) = df_sel
            .to_design_matrix(&g_sel)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let sel_names = {
            let mut n = vec!["_cons".to_string()];
            n.extend(sel_display);
            n
        };

        // Heckman: y and x_out may contain NaN for unselected obs (z=0).
        // Replace NaN/Inf with 0.0 in those rows (values are not used in outcome equation).
        let y_vec = y_vec_raw.mapv(|v| if v.is_finite() { v } else { 0.0 });
        let x_out = x_out.mapv(|v| if v.is_finite() { v } else { 0.0 });

        let result = greeners::Heckman::fit(
            &y_vec,
            &x_out,
            &z_vec,
            &x_sel,
            Some(out_names),
            Some(sel_names),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        Ok(Value::HeckmanResult(Rc::new(result)))
    }

    pub(super) fn tobit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let ll_limit = match opt_map.get("ll") {
            Some(Value::Float(v)) => *v,
            Some(Value::Int(v)) => *v as f64,
            None => 0.0,
            _ => return Err(HayashiError::Runtime("tobit(): ll must be numeric".into())),
        };
        let result = greeners::Tobit::fit(&y_vec, &x_mat, ll_limit, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::TobitResult(Rc::new(result)))
    }

    pub(super) fn poisson(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let cov = resolve_cov_full(opt_map, &df)?;
        let result = greeners::Poisson::fit_with_names(&y_vec, &x_mat, cov, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::PoissonResult(Rc::new(result)))
    }

    pub(super) fn nbreg(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let cov = resolve_cov_full(opt_map, &df)?;
        let result = greeners::NegBin::fit_with_names(&y_vec, &x_mat, cov, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::NegBinResult(Rc::new(result)))
    }

    pub(super) fn ologit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let result = greeners::OrderedLogit::from_formula(&g_formula, &df)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::OrderedResult(Rc::new(result)))
    }

    pub(super) fn oprobit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let result = greeners::OrderedProbit::from_formula(&g_formula, &df)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::OrderedResult(Rc::new(result)))
    }

    pub(super) fn mlogit(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let result = greeners::MNLogit::fit_with_names(&y_vec, &x_mat, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::MNLogitResult(Rc::new(result)))
    }

    pub(super) fn xtlogit(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(format!(
                    "{}() requires id=group_column option",
                    func
                )))
            }
            _ => return Err(HayashiError::Type("id= must be string".into())),
        };
        let (family, link) = match func {
            "xtlogit" => (greeners::Family::Binomial, greeners::Link::Logit),
            "xtprobit" => (greeners::Family::Binomial, greeners::Link::Probit),
            "xtpoisson" => (greeners::Family::Poisson, greeners::Link::Log),
            _ => {
                // xtgee — use family= option
                let family_str = match opt_map.get("family") {
                    None => "gaussian",
                    Some(Value::Str(s)) => s.as_str(),
                    _ => "gaussian",
                };
                match family_str {
                    "binomial" | "logit" => (greeners::Family::Binomial, greeners::Link::Logit),
                    "poisson" => (greeners::Family::Poisson, greeners::Link::Log),
                    _ => (greeners::Family::Gaussian, greeners::Link::Identity),
                }
            }
        };
        let corr_str = match opt_map.get("corr") {
            None => "exchangeable",
            Some(Value::Str(s)) => s.as_str(),
            _ => "exchangeable",
        };
        let corr = match corr_str {
            "independence" | "ind" => greeners::CorrStructure::Independence,
            "exchangeable" | "exch" => greeners::CorrStructure::Exchangeable,
            "ar1" | "ar(1)" => greeners::CorrStructure::AR1,
            "unstructured" | "uns" => greeners::CorrStructure::Unstructured,
            other => {
                return Err(HayashiError::Runtime(format!(
                    "corr='{other}' unknown — use: independence, exchangeable, ar1, unstructured"
                )))
            }
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let id_vals = get_col_f64(&df, &id_col)?;
        let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut next_id = 0usize;
        let groups: ndarray::Array1<usize> = id_vals
            .iter()
            .map(|&v| {
                let key = v as i64;
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();
        let result = greeners::GEE::fit_with_names(
            &y_vec,
            &x_mat,
            &groups,
            &family,
            &link,
            &corr,
            Some(var_names),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::GeeResult(Rc::new(result)))
    }

    pub(super) fn zip(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_count) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let count_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        // inflate= optional: list of column names for inflation equation
        // If omitted, uses the same X matrix as the count model
        let (x_inflate_opt, inflate_names_opt): (
            Option<ndarray::Array2<f64>>,
            Option<Vec<String>>,
        ) = match opt_map.get("inflate") {
            Some(Value::List(lst)) => {
                let inames: Vec<String> = lst
                    .iter()
                    .map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type(
                            "inflate= must be a list of strings".into(),
                        )),
                    })
                    .collect::<Result<_>>()?;
                // intercept + colunas especificadas
                let n = df.n_rows();
                let k = inames.len() + 1;
                let mut xi = ndarray::Array2::<f64>::ones((n, k));
                for (j, name) in inames.iter().enumerate() {
                    xi.column_mut(j + 1).assign(&get_col_f64(&df, name)?);
                }
                let mut full_names = vec!["_cons".to_string()];
                full_names.extend(inames);
                (Some(xi), Some(full_names))
            }
            None => (None, None),
            _ => {
                return Err(HayashiError::Type(
                    "inflate= must be a list of strings".into(),
                ))
            }
        };

        let use_negbin = func == "zinb";
        let result = if use_negbin {
            greeners::ZINB::fit_with_names(
                &y_vec,
                &x_count,
                x_inflate_opt.as_ref(),
                Some(count_names),
                inflate_names_opt,
            )
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
        } else {
            greeners::ZIP::fit_with_names(
                &y_vec,
                &x_count,
                x_inflate_opt.as_ref(),
                Some(count_names),
                inflate_names_opt,
            )
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
        };
        Ok(Value::ZeroInflatedResult(Rc::new(result)))
    }
}
