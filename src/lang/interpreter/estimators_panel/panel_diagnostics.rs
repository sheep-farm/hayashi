use super::*;
use crate::lang::dap::model_expansion;
use indexmap::IndexMap;

impl Interpreter {
    /// `wooldridge` / `xtserial` — Wooldridge serial correlation test.
    pub(super) fn eval_wooldridge(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(self.rt_err("wooldridge(df, y~x, id=\"entity\", time=\"time\")"));
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
        let formula_ast = self.resolve_formula(&args[1])?;
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(id, _)| id.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    self.rt_err(format!(
                        "wooldridge requires id= or xtset({df_name}, id, time)"
                    ))
                })?,
        };
        let time_col = match opt_map.get("time") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(_, t)| t.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    self.rt_err(format!(
                        "wooldridge requires time= or xtset({df_name}, id, time)"
                    ))
                })?,
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let id_vals: Vec<i64> = get_col_f64(&df, &id_col)?
            .iter()
            .map(|&v| v as i64)
            .collect();
        let time_vals: Vec<f64> = get_col_f64(&df, &time_col)?.to_vec();
        let (rho, t_stat, p, n_pairs) =
            greeners::PanelDiagnostics::wooldridge_serial(&y_vec, &x_mat, &id_vals, &time_vals)
                .map_err(HayashiError::Runtime)?;
        let sig = Self::panel_sig_stars(p);
        let conclusion = if p < 0.05 {
            "Reject H0 → serial correlation present → use robust SE"
        } else {
            "Do not reject H0 → no evidence of serial correlation"
        };
        let mut display = String::new();
        display.push_str(&format!(
            "\n{:=^62}\n",
            " Wooldridge Test — Panel Serial Correlation "
        ));
        display.push_str(" H0: ρ = -0.5 (no serial correlation)\n");
        display.push_str(&format!("{:-^62}\n", ""));
        display.push_str(&format!(
            " ρ̂ = {rho:.4}    t = {t_stat:.4}    p = {p:.4}  {sig}\n"
        ));
        display.push_str(&format!(" Residual pairs: {n_pairs}\n"));
        display.push_str(&format!(" Conclusion: {conclusion}\n"));
        display.push_str(&format!("{:=^62}\n", ""));

        let summary = format!("Wooldridge ρ={:.4}, t={:.4}, p={:.4}", rho, t_stat, p);
        let fit = model_expansion::fit_dict(&[
            ("test", Value::Str("Wooldridge".into())),
            ("rho", Value::Float(rho)),
            ("t_stat", Value::Float(t_stat)),
            ("p_value", Value::Float(p)),
            ("n_pairs", Value::Int(n_pairs as i64)),
            ("conclusion", Value::Str(conclusion.into())),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "WooldridgeTestResult",
            fields,
        ))
    }

    /// `pesaran` / `xtcd` — Pesaran cross-sectional dependence test.
    pub(super) fn eval_pesaran(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(self.rt_err("pesaran(df, y~x, id=\"entity\", time=\"time\")"));
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
        let formula_ast = self.resolve_formula(&args[1])?;
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(id, _)| id.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    self.rt_err(format!(
                        "pesaran requires id= or xtset({df_name}, id, time)"
                    ))
                })?,
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let ols_pooled = OLS::from_formula(&g_formula, &df, CovarianceType::NonRobust)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let resids = &y_vec - &x_mat.dot(&ols_pooled.params);
        let id_vals = get_col_f64(&df, &id_col)?;
        let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut next_id = 0usize;
        let entity_ids: Vec<usize> = id_vals
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
        let (cd, p) = greeners::PanelDiagnostics::pesaran_cd(&resids, &entity_ids)
            .map_err(HayashiError::Runtime)?;
        let sig = Self::panel_sig_stars(p);
        let conclusion = if p < 0.05 {
            "Reject H0 → CS dependence present → use cluster-robust SE"
        } else {
            "Do not reject H0 → no CS dependence detected"
        };
        let mut display = String::new();
        display.push_str(&format!(
            "\n{:=^62}\n",
            " Pesaran CD Test — Cross-Sectional Dependence "
        ));
        display.push_str(" H0: no cross-sectional dependence\n");
        display.push_str(&format!("{:-^62}\n", ""));
        display.push_str(&format!(" CD = {cd:.4}    p-value = {p:.4}  {sig}\n"));
        display.push_str(&format!(" Conclusion: {conclusion}\n"));
        display.push_str(&format!("{:=^62}\n", ""));

        let summary = format!("Pesaran CD={:.4}, p={:.4}", cd, p);
        let fit = model_expansion::fit_dict(&[
            ("test", Value::Str("Pesaran CD".into())),
            ("cd_stat", Value::Float(cd)),
            ("p_value", Value::Float(p)),
            ("conclusion", Value::Str(conclusion.into())),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "PesaranCDResult",
            fields,
        ))
    }

    /// `mundlak` — RE vs FE adequacy test.
    pub(super) fn eval_mundlak(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(self.rt_err("mundlak(df, y~x, id=\"entity\")"));
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
        let formula_ast = self.resolve_formula(&args[1])?;
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(id, _)| id.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    self.rt_err(format!(
                        "mundlak requires id= or xtset({df_name}, id, time)"
                    ))
                })?,
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let id_vals: Vec<i64> = get_col_f64(&df, &id_col)?
            .iter()
            .map(|&v| v as i64)
            .collect();
        let (f_stat, p, k, gamma, gamma_se) =
            greeners::PanelDiagnostics::mundlak(&y_vec, &x_mat, &id_vals)
                .map_err(HayashiError::Runtime)?;
        let sig = Self::panel_sig_stars(p);
        let conclusion = if p < 0.05 {
            "Reject H0 → RE is inconsistent → use FE or Hausman"
        } else {
            "Do not reject H0 → RE adequate"
        };

        // Names of time-varying variables (non-constants)
        let slope_names: Vec<&str> = var_names
            .iter()
            .filter(|n| n.as_str() != "_cons" && n.as_str() != "const")
            .map(|s| s.as_str())
            .collect();
        let n_gamma = gamma.len();

        let mut display = String::new();
        display.push_str(&format!(
            "\n{:=^62}\n",
            " Mundlak Test — RE vs FE (correlation of means) "
        ));
        display.push_str(" H0: γ = 0 (group means uncorrelated with X → RE ok)\n");
        display.push_str(&format!("{:-^62}\n", ""));
        display.push_str(&format!(" F({k}, .) = {f_stat:.4}    p = {p:.4}  {sig}\n"));
        display.push_str(&format!("{:-^62}\n", ""));
        display.push_str(&format!(
            " {:<20} {:>10}  {:>10}\n",
            "Variable (γ̂)", "Coef", "Std Err"
        ));

        let mut variable_col = Vec::with_capacity(n_gamma);
        let mut gamma_col = Vec::with_capacity(n_gamma);
        let mut gamma_se_col = Vec::with_capacity(n_gamma);
        for (i, g) in gamma.iter().enumerate().take(n_gamma) {
            let nm = slope_names.get(i).copied().unwrap_or("?").to_string();
            let se = gamma_se.get(i).copied().unwrap_or(f64::NAN);
            display.push_str(&format!(" {:<20} {:>10.4}  {:>10.4}\n", nm, *g, se));
            variable_col.push(nm);
            gamma_col.push(*g);
            gamma_se_col.push(se);
        }
        display.push_str(&format!("\n Conclusion: {conclusion}\n"));
        display.push_str(&format!("{:=^62}\n", ""));

        let mut gamma_columns = IndexMap::new();
        gamma_columns.insert(
            "variable".into(),
            greeners::Column::String(ndarray::Array1::from(variable_col)),
        );
        gamma_columns.insert(
            "gamma".into(),
            greeners::Column::Float(ndarray::Array1::from(gamma_col)),
        );
        gamma_columns.insert(
            "gamma_se".into(),
            greeners::Column::Float(ndarray::Array1::from(gamma_se_col)),
        );
        let gamma_df = Value::DataFrame(Arc::new(
            greeners::DataFrame::from_columns(gamma_columns)
                .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
        ));

        let summary = format!("Mundlak F={:.4}, p={:.4}, k={}", f_stat, p, k);
        let fit = model_expansion::fit_dict(&[
            ("test", Value::Str("Mundlak".into())),
            ("f_stat", Value::Float(f_stat)),
            ("p_value", Value::Float(p)),
            ("k", Value::Int(k as i64)),
            ("gamma", gamma_df),
            ("conclusion", Value::Str(conclusion.into())),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "MundlakTestResult",
            fields,
        ))
    }

    /// `abtest` / `arellano_bond` — Arellano-Bond m1/m2 tests.
    pub(super) fn eval_abtest(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(self.rt_err("abtest(df, y~x, id=\"entity\", time=\"time\")"));
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
        let formula_ast = self.resolve_formula(&args[1])?;
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(id, _)| id.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    self.rt_err(format!("abtest requires id= or xtset({df_name}, id, time)"))
                })?,
        };
        let time_col = match opt_map.get("time") {
            Some(Value::Str(s)) => s.clone(),
            _ => self
                .panel_info
                .get(&df_name)
                .map(|(_, t)| t.clone())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    self.rt_err(format!(
                        "abtest requires time= or xtset({df_name}, id, time)"
                    ))
                })?,
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let id_vals: Vec<i64> = get_col_f64(&df, &id_col)?
            .iter()
            .map(|&v| v as i64)
            .collect();
        let time_vals: Vec<f64> = get_col_f64(&df, &time_col)?.to_vec();
        let (m1, p1, m2, p2) =
            greeners::PanelDiagnostics::arellano_bond_test(&y_vec, &x_mat, &id_vals, &time_vals)
                .map_err(HayashiError::Runtime)?;
        let sig1 = Self::panel_sig_stars(p1);
        let sig2 = Self::panel_sig_stars(p2);
        let mut display = String::new();
        display.push_str(&format!(
            "\n{:=^62}\n",
            " Arellano-Bond Test — First-Difference Autocorrelation "
        ));
        display.push_str(" m1 SHOULD reject H0 (AR(1) induced by FD)\n");
        display.push_str(" m2 SHOULD NOT reject H0 (validates y_{t-2} instruments)\n");
        display.push_str(&format!("{:-^62}\n", ""));
        display.push_str(&format!(" m1 = {m1:.4}    p(m1) = {p1:.4}  {sig1}\n"));
        display.push_str(&format!(" m2 = {m2:.4}    p(m2) = {p2:.4}  {sig2}\n"));
        display.push_str(&format!("{:-^62}\n", ""));
        let m1_warning = p1 >= 0.05;
        let m2_warning = p2 < 0.05;
        if m1_warning {
            display.push_str(" [!] m1 does not reject H0 — model may be misspecified\n");
        }
        if m2_warning {
            display.push_str(" [!] m2 rejects H0 — y_{t-2} instruments may be invalid\n");
        }
        display.push_str(&format!("{:=^62}\n", ""));

        let summary = format!(
            "Arellano-Bond m1={:.4} (p={:.4}), m2={:.4} (p={:.4})",
            m1, p1, m2, p2
        );
        let fit = model_expansion::fit_dict(&[
            ("test", Value::Str("Arellano-Bond".into())),
            ("m1", Value::Float(m1)),
            ("p1", Value::Float(p1)),
            ("m2", Value::Float(m2)),
            ("p2", Value::Float(p2)),
            ("m1_warning", Value::Bool(m1_warning)),
            ("m2_warning", Value::Bool(m2_warning)),
        ]);
        let fields = vec![("fit".into(), fit)];
        Ok(model_expansion::model_result(
            display,
            summary,
            "ArellanoBondTestResult",
            fields,
        ))
    }
}
