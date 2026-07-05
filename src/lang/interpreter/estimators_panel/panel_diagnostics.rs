use super::*;

impl Interpreter {
    /// `wooldridge` / `xtserial` — teste de correlação serial de Wooldridge.
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
                        "wooldridge requer id= ou xtset({df_name}, id, time)"
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
                        "wooldridge requer time= ou xtset({df_name}, id, time)"
                    ))
                })?,
        };
        let formula_str = Self::formula_to_string(&formula_ast);
        let g_formula = GFormula::parse(&formula_str)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let id_vals: Vec<i64> = get_col_f64(&df, &id_col)?
            .iter()
            .map(|&v| v as i64)
            .collect();
        let time_vals: Vec<f64> = get_col_f64(&df, &time_col)?.to_vec();
        let (rho, t_stat, p, n_pairs) = greeners::PanelDiagnostics::wooldridge_serial(
            &y_vec, &x_mat, &id_vals, &time_vals,
        )
        .map_err(HayashiError::Runtime)?;
        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        println!(
            "\n{:=^62}",
            " Wooldridge Test — Correlação Serial em Painel "
        );
        println!(" H0: ρ = -0.5 (sem correlação serial)");
        println!("{:-^62}", "");
        println!(" ρ̂ = {rho:.4}    t = {t_stat:.4}    p = {p:.4}  {sig}");
        println!(" Pares de resíduos: {n_pairs}");
        if p < 0.05 {
            println!(
                " Conclusão: rejeita H0 → correlação serial presente → usar SE robustos"
            );
        } else {
            println!(" Conclusão: não rejeita H0 → sem evidência de correlação serial");
        }
        println!("{:=^62}", "");
        Ok(Value::Nil)
    }

    /// `pesaran` / `xtcd` — teste de dependência cross-sectional de Pesaran.
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
                    self.rt_err(format!("pesaran requer id= ou xtset({df_name}, id, time)"))
                })?,
        };
        let formula_str = Self::formula_to_string(&formula_ast);
        let g_formula = GFormula::parse(&formula_str)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let ols_pooled = OLS::from_formula(&g_formula, &df, CovarianceType::NonRobust)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let resids = &y_vec - &x_mat.dot(&ols_pooled.params);
        let id_vals = get_col_f64(&df, &id_col)?;
        let mut id_map: std::collections::HashMap<i64, usize> =
            std::collections::HashMap::new();
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
        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        println!(
            "\n{:=^62}",
            " Pesaran CD Test — Dependência Cross-Sectional "
        );
        println!(" H0: sem dependência cross-sectional");
        println!("{:-^62}", "");
        println!(" CD = {cd:.4}    p-valor = {p:.4}  {sig}");
        if p < 0.05 {
            println!(" Conclusão: rejeita H0 → dependência CS presente → usar SE robustos por cluster");
        } else {
            println!(" Conclusão: não rejeita H0 → sem dependência CS detectada");
        }
        println!("{:=^62}", "");
        Ok(Value::Nil)
    }

    /// `mundlak` — teste de adequação de RE vs FE.
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
                    self.rt_err(format!("mundlak requer id= ou xtset({df_name}, id, time)"))
                })?,
        };
        let formula_str = Self::formula_to_string(&formula_ast);
        let g_formula = GFormula::parse(&formula_str)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
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
        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        };
        println!(
            "\n{:=^62}",
            " Mundlak Test — RE vs FE (correlação das médias) "
        );
        println!(" H0: γ = 0 (médias de grupo não correlacionadas com X → RE ok)");
        println!("{:-^62}", "");
        println!(" F({k}, .) = {f_stat:.4}    p = {p:.4}  {sig}");
        println!("{:-^62}", "");
        // Nomes das variáveis variantes no tempo (não-constantes)
        let slope_names: Vec<&str> = var_names
            .iter()
            .filter(|n| n.as_str() != "_cons" && n.as_str() != "const")
            .map(|s| s.as_str())
            .collect();
        println!(" {:<20} {:>10}  {:>10}", "Variável (γ̂)", "Coef", "Std Err");
        for (i, g) in gamma.iter().enumerate().take(k.min(gamma.len())) {
            let nm = slope_names.get(i).copied().unwrap_or("?");
            println!(
                " {:<20} {:>10.4}  {:>10.4}",
                nm,
                g,
                gamma_se.get(i).copied().unwrap_or(f64::NAN)
            );
        }
        if p < 0.05 {
            println!("\n Conclusão: rejeita H0 → RE é inconsistente → usar FE ou Hausman");
        } else {
            println!("\n Conclusão: não rejeita H0 → RE adequado");
        }
        println!("{:=^62}", "");
        Ok(Value::Nil)
    }

    /// `abtest` / `arellano_bond` — testes m1/m2 de Arellano-Bond.
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
                    self.rt_err(format!("abtest requer id= ou xtset({df_name}, id, time)"))
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
                        "abtest requer time= ou xtset({df_name}, id, time)"
                    ))
                })?,
        };
        let formula_str = Self::formula_to_string(&formula_ast);
        let g_formula = GFormula::parse(&formula_str)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let id_vals: Vec<i64> = get_col_f64(&df, &id_col)?
            .iter()
            .map(|&v| v as i64)
            .collect();
        let time_vals: Vec<f64> = get_col_f64(&df, &time_col)?.to_vec();
        let (m1, p1, m2, p2) = greeners::PanelDiagnostics::arellano_bond_test(
            &y_vec, &x_mat, &id_vals, &time_vals,
        )
        .map_err(HayashiError::Runtime)?;
        let sig = |p: f64| {
            if p < 0.01 {
                "***"
            } else if p < 0.05 {
                "**"
            } else if p < 0.10 {
                "*"
            } else {
                ""
            }
        };
        println!(
            "\n{:=^62}",
            " Arellano-Bond Test — Autocorrelação em 1ª Diferença "
        );
        println!(" m1 DEVE rejeitar H0 (AR(1) induzido por FD)");
        println!(" m2 NÃO deve rejeitar H0 (valida instrumentos y_{{t-2}})");
        println!("{:-^62}", "");
        println!(" m1 = {m1:.4}    p(m1) = {p1:.4}  {}", sig(p1));
        println!(" m2 = {m2:.4}    p(m2) = {p2:.4}  {}", sig(p2));
        println!("{:-^62}", "");
        if p1 >= 0.05 {
            println!(" [!] m1 não rejeita H0 — modelo pode estar mal especificado");
        }
        if p2 < 0.05 {
            println!(" [!] m2 rejeita H0 — instrumentos y_{{t-2}} podem ser inválidos");
        }
        println!("{:=^62}", "");
        Ok(Value::Nil)
    }
}
