use super::*;
use super::helpers::*;

mod panel_diagnostics;
mod rolling_recursive;

/// bootstrap genérico/bootse, diagnósticos de painel, SUR, rolling/recursive
/// OLS, tabela de critérios de informação, Fixed Effects, Random Effects,
/// testes de painel (F-test, Pesaran CD, Breusch-Pagan LM, Chamberlain),
/// Arellano-Bond, GMM genérico, System GMM, FE-2SLS, PCSE, Panel GLS,
/// teste m1/m2, Hausman, especificação/Wald gerais.
/// Extraído de `eval_call` (ver src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_estimators_panel(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // bootse — Bootstrap standard errors para modelos OLS
            // bootse(model, n=1000)
            // Reamostral pares (y, X) com reposição para estimar distribuição amostral
            // Compara SE originais com bootstrap SE e IC percentil 95%
            // ── bootstrap genérico ────────────────────────────────────────────
            // bootstrap(estimator, formula, df, n=1000, alpha=0.05)
            // Reamostra linhas do DataFrame com reposição e re-estima.
            // Funciona com qualquer estimador: ols, logit, probit, iv, poisson, etc.
            // bootse(model, n=1000) mantido como alias para OLS pairs bootstrap.
            "bootstrap" | "boot" => {
                let n_boot = Self::bootstrap_reps(opt_map);
                let alpha = Self::bootstrap_alpha(opt_map);
                if args.len() >= 3 {
                    self.bootstrap_generic(args, opts, n_boot, alpha)
                } else {
                    self.bootstrap_pairs(args, n_boot, alpha)
                }
            }

            "bootse" => {
                return self.eval_call("bootstrap", args, opts).map(Some);
            }

            // markov — Markov-Switching AR (Hamilton 1989)
            // markov(df, y, k=2, p=1)
            // k=: número de regimes (padrão: 2)
            // p=: ordem AR dentro de cada regime (padrão: 1)
            // Algoritmo: EM via filtro de Hamilton (forward-backward)
            // Parâmetros por regime: intercept + AR coefficients + variance
            "markov" | "msar" | "markovswitching" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("markov(df, y_var, k=2, p=1)".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(
                            "markov: first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let y_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "markov: second argument must be variable name".into(),
                        ))
                    }
                };
                let y_vec = ndarray::Array1::from(get_col_f64(&df, &y_name)?);
                let k = match opt_map.get("k") {
                    Some(Value::Int(v)) => (*v as usize).max(2),
                    Some(Value::Float(v)) => (*v as usize).max(2),
                    _ => 2,
                };
                let p = match opt_map.get("p") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let result = greeners::MarkovSwitching::fit(&y_vec, k, p)
                    .map_err(|e| self.rt_err(format!("markov: {e}")))?;
                Ok(Value::MarkovResult(Rc::new(result)))
            }

            // clogit — Conditional Logit (Chamberlain 1980, FE logit)
            // clogit(y ~ x1 + x2, df, group="id_col")
            // Condiciona na soma de y por grupo → elimina efeitos fixos individuais
            // Grupos sem variação em y são automaticamente excluídos
            // Sem intercepto — absorvido pelo FE
            "clogit" | "xtlogit_fe" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let group_col = match opt_map.get("group") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "clogit requer group=\"coluna_id\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("clogit: group= must be string".into())),
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
                let group_vals = get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: Vec<usize> = group_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| {
                            let g = gnext;
                            gnext += 1;
                            g
                        })
                    })
                    .collect();
                let result = greeners::ConditionalLogit::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    Some(var_names),
                )
                .map_err(|e| self.rt_err(format!("clogit: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // cpoisson — Conditional Poisson (FE Poisson)
            // cpoisson(y ~ x1 + x2, df, group="id_col")
            // Equivalente a FE Poisson; consistente sob heterogeidade não observada
            // Só requer que E[y|x,c] = exp(c + xβ) — não requer y ~ Poisson (PPML)
            "cpoisson" | "xtpoisson_fe" | "ppml" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let group_col = match opt_map.get("group") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "cpoisson requer group=\"coluna_id\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("cpoisson: group= must be string".into())),
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
                let group_vals = get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: Vec<usize> = group_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| {
                            let g = gnext;
                            gnext += 1;
                            g
                        })
                    })
                    .collect();
                let result = greeners::ConditionalPoisson::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    Some(var_names),
                )
                .map_err(|e| self.rt_err(format!("cpoisson: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // cmnlogit — Conditional Multinomial Logit
            // cmnlogit(y ~ x1 + x2, df, group="id_col", alts=3)
            "cmnlogit" | "cmlogit" | "conditional_mlogit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let group_col = match opt_map.get("group") {
                    Some(Value::Str(s)) => s.clone(),
                    None => {
                        return Err(HayashiError::Runtime(
                            "cmnlogit requires group=\"id_col\"".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("cmnlogit: group= must be string".into())),
                };
                let n_alts = match opt_map.get("alts") {
                    Some(Value::Int(n)) => *n as usize,
                    Some(Value::Float(f)) => *f as usize,
                    None => {
                        return Err(HayashiError::Runtime(
                            "cmnlogit requires alts=N (number of alternatives)".into(),
                        ))
                    }
                    _ => return Err(HayashiError::Type("cmnlogit: alts= must be integer".into())),
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
                let group_vals = get_col_f64(&df, &group_col)?;
                let mut gmap: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                let mut gnext = 0usize;
                let groups: Vec<usize> = group_vals
                    .iter()
                    .map(|&v| {
                        let key = v as i64;
                        *gmap.entry(key).or_insert_with(|| {
                            let g = gnext;
                            gnext += 1;
                            g
                        })
                    })
                    .collect();
                let result = greeners::ConditionalMNLogit::fit_with_names(
                    &y_vec,
                    &x_mat,
                    &groups,
                    n_alts,
                    Some(var_names),
                )
                .map_err(|e| self.rt_err(format!("cmnlogit: {e}")))?;
                Ok(Value::ConditionalResult(Rc::new(result)))
            }

            // gqtest — Goldfeld-Quandt test (heteroskedasticidade)
            // gqtest(model, split=0.2)
            // H0: homocedasticidade
            // Divide os resíduos em dois grupos (descartando `split` do meio)
            // e testa se as variâncias diferem via F
            // split=: fração do meio a descartar (padrão: 0.2)
            // Mais potente que White quando heterocedasticidade é monotônica
            "gqtest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("gqtest(model, split=0.2)".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "gqtest(): only supports OLS models".into(),
                        ))
                    }
                };
                let split = match opt_map.get("split") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.2,
                };
                let (f, p, df1, df2) =
                    greeners::SpecificationTests::goldfeld_quandt_test(&ols.residuals, split)
                        .map_err(|e| self.rt_err(format!("gqtest: {e}")))?;
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
                let sep = "─".repeat(56);
                println!("\nGoldfeld-Quandt Test  —  split = {split:.2}");
                println!("{sep}");
                println!("H₀: homocedasticidade (σ²₁ = σ²₂)");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10.4} {:>10.4} {:>4}",
                    format!("F ~ F({df1},{df2})"),
                    f,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // bphet — Breusch-Pagan test (heteroskedasticidade, OLS)
            // bphet(model)
            // H0: homocedasticidade — LM = n·R² da regressão auxiliar de u² em X
            // Diferente de bptest() que é o LM de efeitos aleatórios (painel)
            "bphet" | "hettest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("bphet(model)".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "bphet(): only supports OLS models".into(),
                        ))
                    }
                };
                let (lm, p) = greeners::Diagnostics::breusch_pagan(&ols.residuals, &ols.x)
                    .map_err(|e| self.rt_err(format!("bphet: {e}")))?;
                let k = ols.x.ncols().saturating_sub(1);
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
                let sep = "─".repeat(56);
                println!("\nBreusch-Pagan Heteroskedasticity Test");
                println!("{sep}");
                println!("H₀: homocedasticidade (variância constante)");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10.4} {:>10.4} {:>4}",
                    format!("LM ~ χ²({k})"),
                    lm,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // ── Testes de diagnóstico para dados em painel ────────────────────

            // bptest — Breusch-Pagan LM test (H0: pooled OLS adequado, σ²_u = 0)
            // bptest(df, y ~ x1 + x2, id="entity_col")
            "bptest" | "xttest0" | "xtbp" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "bptest(df, y ~ x1+x2, id=\"entity_col\")".into(),
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
                let formula_ast = self.resolve_formula(&args[1])?;
                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => self
                        .panel_info
                        .get(&df_name)
                        .map(|(id, _)| id.clone())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            self.rt_err(format!("bptest requer id= ou xtset({df_name}, id, time)"))
                        })?,
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                // OLS pooled para obter resíduos
                let ols_pooled = OLS::from_formula(&g_formula, &df, CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let resids = &y_vec - &x_mat.dot(&ols_pooled.params);
                // Converter id para usize
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
                let (lm, p) = greeners::PanelDiagnostics::breusch_pagan_lm(&resids, &entity_ids)
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
                println!("\n{:=^62}", " Breusch-Pagan LM Test (RE) ");
                println!(" H0: σ²_u = 0 — pooled OLS adequado");
                println!("{:-^62}", "");
                println!(" LM = {lm:.4}    p-valor = {p:.4}  {sig}");
                if p < 0.05 {
                    println!(" Conclusão: rejeita H0 → usar RE ou FE");
                } else {
                    println!(" Conclusão: não rejeita H0 → pooled OLS adequado");
                }
                println!("{:=^62}", "");
                Ok(Value::Nil)
            }

            // wooldridge — Teste de Wooldridge para correlação serial em painel
            // H0: sem correlação serial de 1ª ordem nos erros idiossincráticos
            "wooldridge" | "xtserial" | "wooldridge_serial" | "xtwooldridge" => {
                self.eval_wooldridge(args, opt_map)
            }

            // pesaran — Pesaran CD test (cross-sectional dependence)
            "pesaran" | "xtcd" => self.eval_pesaran(args, opt_map),

            // mundlak — Teste de Mundlak (adequação de RE vs FE)
            "mundlak" => self.eval_mundlak(args, opt_map),

            // abtest — Arellano-Bond m1/m2 test (validação de instrumentos GMM)
            "abtest" | "abar" | "abond" | "xtabond_test" | "arellano_bond" => {
                self.eval_abtest(args, opt_map)
            }

            // ── SUR (Seemingly Unrelated Regressions) ─────────────────────────
            // sur(df, y1 ~ x1 + x2, y2 ~ x3 + x4, ...)
            // Estimador de Zellner (FGLS entre equações)
            // Cada equação pode ter regressores diferentes
            "sur" | "sureg" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "sur(df, y1~x1+x2, y2~x3+x4, ...) requer df + ao menos 2 fórmulas".into(),
                    ));
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
                let mut equations: Vec<greeners::SurEquation> = Vec::new();
                let mut eq_var_names: Vec<Vec<String>> = Vec::new();

                for arg in &args[1..] {
                    let formula_ast = self.resolve_formula(arg)?;
                    let formula_str = Self::formula_to_string(&formula_ast);
                    let g_formula = GFormula::parse(&formula_str)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let (y, x) = df
                        .to_design_matrix(&g_formula)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let var_names = df
                        .formula_var_names(&g_formula)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    eq_var_names.push(var_names);
                    equations.push(greeners::SurEquation {
                        y,
                        x,
                        name: formula_ast.lhs.clone(),
                    });
                }
                let result = greeners::SUR::fit(&equations)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::SurResult(SurModel {
                    result: Rc::new(result),
                    eq_var_names,
                }))
            }

            // ── Rolling OLS (janela deslizante) ───────────────────────────────
            "rolling" | "rols" => self.eval_rolling(args, opts, opt_map),

            // ── Recursive OLS (Kalman, acumula observações) ───────────────────
            "recursive" | "recols" => self.eval_recursive(args, opts),

            // ── ic — tabela de critérios de informação (AIC/BIC) ──────────────
            // ic(m1, m2, m3, ...)
            // Compara modelos pelo AIC e BIC; ordena do menor (melhor) para maior
            "ic" | "fitstat" | "estat" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "ic() requer ao menos um modelo".into(),
                    ));
                }
                struct IcRow {
                    label: String,
                    ll: f64,
                    k: usize,
                    n: usize,
                    aic: f64,
                    bic: f64,
                }
                let mut rows: Vec<IcRow> = Vec::new();
                for arg in args {
                    let label = match arg {
                        Expr::Var(name) => name.clone(),
                        _ => "model".to_string(),
                    };
                    let val = self.eval_expr(arg)?;
                    let (ll, k, n) = match &val {
                        Value::OlsResult(m)      => (m.result.log_likelihood, m.result.params.len(), m.result.n_obs),
                        Value::BinaryResult(b)   => (b.result.log_likelihood, b.result.params.len(), b.x.nrows()),
                        Value::PoissonResult(r)  => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::NegBinResult(r)   => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::OrderedResult(r)  => (r.log_likelihood, r.params.len() + r.thresholds.len(), r.n_obs),
                        Value::TobitResult(r)    => (r.log_likelihood, r.params.len(), r.n_obs),
                        Value::MixedResult(r)    => (r.log_likelihood, r.fixed_effects.len(), r.n_obs),
                        Value::ZeroInflatedResult(r) => (r.log_likelihood, r.count_params.len() + r.inflate_params.len(), r.n_obs),
                        Value::RollingResult(_) | Value::RecursiveLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                format!("ic(): '{label}' não tem log-verossimilhança — use print() para diagnósticos")
                            ));
                        }
                        _ => return Err(HayashiError::Runtime(
                            format!("ic(): modelo '{label}' não tem log-verossimilhança disponível para ic() — use print()")
                        )),
                    };
                    let aic = -2.0 * ll + 2.0 * k as f64;
                    let bic = -2.0 * ll + (k as f64) * (n as f64).ln();
                    rows.push(IcRow {
                        label,
                        ll,
                        k,
                        n,
                        aic,
                        bic,
                    });
                }
                // Ordenar por AIC
                rows.sort_by(|a, b| {
                    a.aic
                        .partial_cmp(&b.aic)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let min_aic = rows.first().map(|r| r.aic).unwrap_or(0.0);
                let _min_bic = rows
                    .iter()
                    .map(|r| r.bic)
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                println!("\n{:=^80}", " Critérios de Informação ");
                println!(
                    "{:<20} {:>6} {:>6} {:>12} {:>12} {:>8} {:>8}",
                    "Modelo", "N", "k", "Log-Lik", "AIC", "ΔAIC", "BIC"
                );
                println!("{:-^80}", "");
                for row in &rows {
                    println!(
                        "{:<20} {:>6} {:>6} {:>12.4} {:>12.4} {:>8.4} {:>12.4}",
                        row.label,
                        row.n,
                        row.k,
                        row.ll,
                        row.aic,
                        row.aic - min_aic,
                        row.bic
                    );
                }
                if rows.len() > 1 {
                    println!("{:-^80}", "");
                    println!(
                        " Melhor AIC: {}   Melhor BIC: {}",
                        rows.iter()
                            .min_by(|a, b| a.aic.partial_cmp(&b.aic).unwrap())
                            .unwrap()
                            .label,
                        rows.iter()
                            .min_by(|a, b| a.bic.partial_cmp(&b.bic).unwrap())
                            .unwrap()
                            .label
                    );
                    // Pesos de Akaike
                    let delta_aics: Vec<f64> = rows.iter().map(|r| r.aic - min_aic).collect();
                    let rel: Vec<f64> = delta_aics.iter().map(|d| (-d / 2.0).exp()).collect();
                    let sum_rel: f64 = rel.iter().sum();
                    println!(
                        " Pesos Akaike: {}",
                        rows.iter()
                            .zip(rel.iter())
                            .map(|(r, w)| format!("{}={:.3}", r.label, w / sum_rel))
                            .collect::<Vec<_>>()
                            .join("  ")
                    );
                }
                println!("{:=^80}", "");
                Ok(Value::Nil)
            }

            // ── Fixed Effects ─────────────────────────────────────────────────
            "fe" => {
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                // FE elimina o intercepto via within-transform; forçamos - 1
                // para evitar coluna de zeros pós-demeaning (singular matrix)
                let formula_no_const = if formula_str.contains("- 1") {
                    formula_str
                } else {
                    format!("{} - 1", formula_str)
                };
                let g_formula = GFormula::parse(&formula_no_const)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // tenta int; cai para float→int; cai para string
                let result = if let Ok(ids) = df.get_int(&id_col) {
                    let ids_vec: Vec<i64> = ids.to_vec();
                    FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else if let Ok(floats) = df.get(&id_col) {
                    let ids_vec: Vec<i64> = floats.iter().map(|&v| v as i64).collect();
                    FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else if let Ok(ids) = df.get_string(&id_col) {
                    let ids_vec: Vec<String> = ids.to_vec();
                    FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "column '{id_col}' not found or not usable as entity ID"
                    )));
                };

                Ok(Value::PanelResult(Rc::new(result)))
            }

            // ── Random Effects ────────────────────────────────────────────────
            "re" => {
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // aceita coluna float de valores inteiros (ex: idcode lido como f64)
                let ids_owned: ndarray::Array1<i64>;
                let ids = match df.get_int(&id_col) {
                    Ok(arr) => arr,
                    Err(_) => {
                        let floats = df.get(id_col.as_str()).map_err(|_| {
                            HayashiError::Runtime(format!(
                                "column '{id_col}' must be integer for re()"
                            ))
                        })?;
                        ids_owned = floats.mapv(|v| v as i64);
                        &ids_owned
                    }
                };

                let result = RandomEffects::from_formula(&g_formula, &df, ids)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::ReResult(Rc::new(result)))
            }

            // ── F-test para Efeitos Fixos (FE vs pooled OLS) ─────────────────
            "ftest_fe" => {
                // ftest_fe(formula, df, id=col)
                // H₀: todos os efeitos individuais são zero (pooled OLS adequado)
                // H₁: efeitos individuais existem (use FE)
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);

                // FE (within)
                let formula_no_const = if formula_str.contains("- 1") {
                    formula_str.clone()
                } else {
                    format!("{} - 1", formula_str)
                };
                let g_formula_fe = GFormula::parse(&formula_no_const)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids_fe: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ftest_fe: column '{id_col}' not found"
                    )));
                };

                let fe = FixedEffects::from_formula(&g_formula_fe, &df, &entity_ids_fe)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // Pooled OLS (com intercepto)
                let g_formula_ols = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_pool, x_pool) = df
                    .to_design_matrix(&g_formula_ols)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols = greeners::OLS::fit(&y_pool, &x_pool, greeners::CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let ssr_pooled = ols.sigma.powi(2) * ols.df_resid as f64;
                let ssr_fe = fe.sigma.powi(2) * fe.df_resid as f64;
                let n = fe.n_obs;
                let n_entities = fe.n_entities;
                let k = fe.params.len();

                let (f_stat, p) = greeners::PanelDiagnostics::f_test_fixed_effects(
                    ssr_pooled, ssr_fe, n, n_entities, k,
                )
                .map_err(HayashiError::Runtime)?;

                let df_num = n_entities - 1;
                let df_denom = n - n_entities - k;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos fixos individuais são significativos (use FE)"
                } else {
                    "Não rejeita H₀ → pooled OLS adequado (efeitos individuais não significativos)"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" F-test: Efeitos Fixos vs Pooled OLS\n");
                out.push_str(" H₀: todos os efeitos individuais são zero\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Soma dos Quadrados dos Resíduos\n");
                out.push_str(&format!("   SSR pooled = {:.6}\n", ssr_pooled));
                out.push_str(&format!("   SSR FE     = {:.6}\n", ssr_fe));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
                    df_num, df_denom, f_stat, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Pesaran CD: dependência cross-seccional ───────────────────────
            "pesaran_cd" | "cd_test" => {
                // pesaran_cd(formula, df, id=col)
                // H₀: resíduos independentes entre entidades (sem dependência cross-seccional)
                // H₁: dependência cross-seccional presente
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // OLS pooled para resíduos
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols = greeners::OLS::fit(&y_vec, &x_mat, greeners::CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let residuals = ols.residuals(&y_vec, &x_mat);

                // IDs de entidade
                let entity_ids: Vec<usize> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.iter().map(|&v| v as usize).collect()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as usize).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "pesaran_cd: column '{id_col}' not found"
                    )));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        s.insert(id);
                    }
                    s.len()
                };
                let t_bar = residuals.len() as f64 / n_entities as f64;

                let (cd, p) = greeners::PanelDiagnostics::pesaran_cd(&residuals, &entity_ids)
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
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → dependência cross-seccional presente"
                } else {
                    "Não rejeita H₀ → sem evidência de dependência cross-seccional"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Pesaran CD Test (dependência cross-seccional)\n");
                out.push_str(" H₀: ρ_ij = 0 para todo i≠j  (resíduos independentes)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: N={} entidades   T̄≈{:.1}\n",
                    n_entities, t_bar
                ));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   CD ~ N(0,1) = {:.4}   p = {:.4}  {}\n",
                    cd, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Breusch-Pagan LM test (efeitos individuais em painel) ────────
            "bplm" => {
                // bplm(formula, df, id=col)
                // H₀: sem efeitos individuais (σ²_u = 0) — pooled OLS adequado
                // H₁: efeitos individuais existem — use FE ou RE
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // OLS pooled para obter resíduos
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let ols = greeners::OLS::fit(&y_vec, &x_mat, greeners::CovarianceType::NonRobust)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let residuals = ols.residuals(&y_vec, &x_mat);

                // IDs de entidade → usize
                let entity_ids: Vec<usize> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.iter().map(|&v| v as usize).collect()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as usize).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "bplm: column '{id_col}' not found ou não usável como ID"
                    )));
                };

                let n = residuals.len();
                let n_entities = {
                    let mut ids_set = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        ids_set.insert(id);
                    }
                    ids_set.len()
                };
                let t_bar = n as f64 / n_entities as f64;

                let (lm, p) = greeners::PanelDiagnostics::breusch_pagan_lm(&residuals, &entity_ids)
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
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais presentes (use FE ou RE)"
                } else {
                    "Não rejeita H₀ → pooled OLS adequado (sem efeitos individuais)"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Breusch-Pagan LM Test (efeitos individuais)\n");
                out.push_str(" H₀: σ²_u = 0  (sem efeitos individuais)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Dados do Painel\n");
                out.push_str(&format!(
                    "   n = {}   N = {}   T̄ ≈ {:.1}\n",
                    n, n_entities, t_bar
                ));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   LM ~ χ²(1) = {:.4}   p = {:.4}  {}\n",
                    lm, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Chamberlain: correlação period-específica com efeitos individuais
            "chamberlain" => {
                // chamberlain(formula, df, id=col, time=col)
                // H₀: Π_s = 0 para todo s (RE consistente)
                // H₁: pelo menos um Π_s ≠ 0 (efeitos correlacionados com X — use FE)
                // Generalização do Mundlak: usa valores em TODOS os períodos, não só a média
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let time_col = self.get_time_col(&df_name, opt_map)?;

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "chamberlain: coluna id '{id_col}' not found"
                    )));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "chamberlain: coluna time '{time_col}' not found"
                    )));
                };

                let (f_stat, p, k_active, df_denom, n_entities, t_count) =
                    greeners::PanelDiagnostics::chamberlain(
                        &y_vec,
                        &x_mat,
                        &entity_ids,
                        &time_vals,
                    )
                    .map_err(HayashiError::Runtime)?;

                let n_obs = y_vec.len();
                let df1 = k_active;

                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais correlacionados com X (prefira FE)"
                } else {
                    "Não rejeita H₀ → RE consistente (sem correlação period-específica)"
                };

                let thick = "═".repeat(70);
                let thin = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(
                    " Chamberlain Test (correlação period-específica com efeitos individuais)\n",
                );
                out.push_str(" H₀: Π_s = 0 ∀s  (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: n={} obs   N={} entidades   T={} períodos\n",
                    n_obs, n_entities, t_count
                ));
                out.push_str(&format!("   Colunas de augmentação: {} de Chamberlain (k×T, após remover zero-variância)\n", k_active));
                if t_count > 6 {
                    out.push_str(&format!(
                        "   ⚠ T={} — com T grande o teste tem baixo poder em amostras finitas\n",
                        t_count
                    ));
                }
                out.push_str("\n── Teste conjunto H₀: todos os Π_s = 0\n");
                out.push_str(&format!(
                    "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
                    df1, df_denom, f_stat, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(
                    "   Teste mais geral que Mundlak — inclui valores em todos os T períodos\n",
                );
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Arellano-Bond Diff-GMM (OLD mundlak removed — use new mundlak above) ─
            "mundlak_OLD_REMOVED" => {
                let (formula_ast, df, _df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "mundlak: column '{id_col}' not found"
                    )));
                };

                // Nomes dos regressores variantes no tempo (excluindo "const")
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let non_const_names: Vec<&str> = var_names
                    .iter()
                    .filter(|n| n.as_str() != "const")
                    .map(|s| s.as_str())
                    .collect();

                let n = y_vec.len();
                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        s.insert(id);
                    }
                    s.len()
                };

                let (f_stat, p, k, gamma_hat, gamma_se) =
                    greeners::PanelDiagnostics::mundlak(&y_vec, &x_mat, &entity_ids)
                        .map_err(HayashiError::Runtime)?;

                let df1 = k;
                let df2_exact = if n > 2 * k + 1 { n - 2 * k - 1 } else { 1 };

                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → efeitos individuais correlacionados com X (prefira FE)"
                } else {
                    "Não rejeita H₀ → RE consistente (sem evidência de correlação com efeitos)"
                };

                let thick = "═".repeat(70);
                let thin = "─".repeat(70);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(
                    " Mundlak Test (correlação entre regressores e efeitos individuais)\n",
                );
                out.push_str(" H₀: γ = 0  (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: n={} obs   N={} entidades   k={} regressores variantes\n",
                    n, n_entities, k
                ));
                out.push_str("\n── Coeficientes sobre médias individuais (X̄_i)\n");
                out.push_str(&format!(
                    "   {:<18} {:>10}  {:>10}  {:>8}\n",
                    "Variável (X̄)", "γ̂", "SE", "t"
                ));
                out.push_str(&format!("   {}\n", "─".repeat(52)));
                for i in 0..k {
                    let t_i = if gamma_se[i] > 1e-15 {
                        gamma_hat[i] / gamma_se[i]
                    } else {
                        f64::NAN
                    };
                    let name = non_const_names.get(i).copied().unwrap_or("?");
                    out.push_str(&format!(
                        "   {:<18} {:>10.4}  {:>10.4}  {:>8.3}\n",
                        format!("{}̄", name),
                        gamma_hat[i],
                        gamma_se[i],
                        t_i
                    ));
                }
                out.push_str("\n── Teste conjunto H₀: γ = 0\n");
                out.push_str(&format!(
                    "   F({}, {}) = {:.4}   p = {:.4}  {}\n",
                    df1, df2_exact, f_stat, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Arellano-Bond Diff-GMM ────────────────────────────────────────
            // ab(formula, df, id=col, time=col [, lags=2 [, step=1]])
            // Estima y_it = ρ y_{i,t-1} + X_it'β + α_i + ε_it via Diff-GMM.
            // Instrumenta Δy_{i,t-1} com y_{i,t-2},...,y_{i,t-lags-1} (collapsed).
            "ab" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let time_col = self.get_time_col(&df_name, opt_map)?;

                let max_lags: usize = match opt_map.get("lags") {
                    Some(Value::Int(v)) => (*v).max(1) as usize,
                    Some(Value::Float(v)) => (*v as i64).max(1) as usize,
                    None => 2,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "ab(): lags must be integer positivo".into(),
                        ))
                    }
                };

                let two_step: bool = match opt_map.get("step") {
                    Some(Value::Int(2)) => true,
                    Some(Value::Float(v)) if *v as i64 == 2 => true,
                    Some(Value::Int(_)) | Some(Value::Float(_)) | None => false,
                    _ => return Err(HayashiError::Runtime("ab(): step deve ser 1 ou 2".into())),
                };

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ab: coluna id '{id_col}' not found"
                    )));
                };

                let time_ids: Vec<i64> = if let Ok(ids) = df.get_int(&time_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&time_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ab: coluna time '{time_col}' not found"
                    )));
                };

                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::ArellanoBond::fit(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_ids,
                    max_lags,
                    two_step,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::AbResult(Rc::new(result)))
            }

            // ── GMM genérico (Two-Step Efficient) ────────────────────────────
            // gmm(endog_formula, instrument_formula, df)
            "gmm" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "gmm(endog_formula, instrument_formula, dataframe)".into(),
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

                let endog_str = Self::formula_to_string(&endog_ast);
                let instr_str = Self::formula_to_string(&instr_ast);

                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let g_instr = if instr_ast.lhs.is_empty() {
                    let independents: Vec<String> = instr_ast
                        .rhs
                        .iter()
                        .map(|t| match t {
                            RhsTerm::Var(v) => v.clone(),
                            RhsTerm::Categorical(v) => format!("C({v})"),
                            RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                            RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
                        })
                        .collect();
                    GFormula {
                        dependent: String::new(),
                        independents,
                        intercept: true,
                    }
                } else {
                    GFormula::parse(&instr_str).map_err(|e| HayashiError::Runtime(e.to_string()))?
                };

                let (y, x) = df
                    .to_design_matrix(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let z = {
                    let n_rows = df.n_rows();
                    let n_cols = g_instr.independents.len() + if g_instr.intercept { 1 } else { 0 };
                    let mut z_mat = ndarray::Array2::<f64>::zeros((n_rows, n_cols));
                    let mut col_idx = 0;
                    if g_instr.intercept {
                        for i in 0..n_rows {
                            z_mat[[i, 0]] = 1.0;
                        }
                        col_idx = 1;
                    }
                    for (j, var_name) in g_instr.independents.iter().enumerate() {
                        let col_data = df
                            .get(var_name)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        for i in 0..n_rows {
                            z_mat[[i, col_idx + j]] = col_data[i];
                        }
                    }
                    z_mat
                };

                let result =
                    greeners::GMM::fit(&y, &x, &z).map_err(|e| self.rt_err(format!("gmm: {e}")))?;
                Ok(Value::GmmResult(Rc::new(result)))
            }

            // ── System GMM (Blundell-Bond 1998) ──────────────────────────────
            // sysgmm(formula, df, id=col, time=col [, lags=2 [, step=1]])
            // Empilha eq. em 1ª diferença (instrumentadas com níveis defasados)
            // + eq. em níveis (instrumentadas com Δy_{t-1} e ΔX_{t-1}).
            "sysgmm" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let time_col = self.get_time_col(&df_name, opt_map)?;

                let max_lags: usize = match opt_map.get("lags") {
                    Some(Value::Int(v)) => (*v).max(1) as usize,
                    Some(Value::Float(v)) => (*v as i64).max(1) as usize,
                    None => 2,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "sysgmm(): lags must be integer positivo".into(),
                        ))
                    }
                };

                let two_step: bool = match opt_map.get("step") {
                    Some(Value::Int(2)) => true,
                    Some(Value::Float(v)) if *v as i64 == 2 => true,
                    Some(Value::Int(_)) | Some(Value::Float(_)) | None => false,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "sysgmm(): step deve ser 1 ou 2".into(),
                        ))
                    }
                };

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "sysgmm: coluna id '{id_col}' not found"
                    )));
                };

                let time_ids: Vec<i64> = if let Ok(ids) = df.get_int(&time_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&time_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "sysgmm: coluna time '{time_col}' not found"
                    )));
                };

                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = greeners::SystemGmm::fit(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_ids,
                    max_lags,
                    two_step,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::SysGmmResult(Rc::new(result)))
            }

            // ── FE-2SLS (xtivreg, fe) — Hausman (1978) ───────────────────────
            // feiv(endog_formula, instrument_formula, df, id=col [, cov=...])
            // endog_formula: y ~ x1 + x2   (x2 é endógena)
            // instrument_formula: ~ x1 + z1 + z2  (exógenos incluídos + excluídos)
            "feiv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "feiv() requer (formula_estrutural, formula_instrumentos, df, id=col)"
                            .into(),
                    ));
                }

                let endog_ast = self.resolve_formula(&args[0])?;
                let instr_ast = self.resolve_formula(&args[1])?;
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "feiv(): terceiro argumento deve ser nome do DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "feiv: '{df_name}' is not a DataFrame"
                        )))
                    }
                };

                let id_col = match opt_map.get("id") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "feiv(): opção id=col é obrigatória".into(),
                        ))
                    }
                };

                // fórmula estrutural → y e X (sem constante, FE a absorve)
                let endog_str = Self::formula_to_string(&endog_ast);
                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // fórmula de instrumentos → Z (sem constante)
                let instr_vars: Vec<String> = instr_ast
                    .rhs
                    .iter()
                    .map(|t| match t {
                        RhsTerm::Var(v) => v.clone(),
                        RhsTerm::Categorical(v) => format!("C({v})"),
                        RhsTerm::Transform(fn_, v) => format!("{fn_}({v})"),
                        RhsTerm::Interaction(a, b) => format!("{a}:{b}"),
                    })
                    .collect();

                let n = y_vec.len();
                let l = instr_vars.len();
                if l == 0 {
                    return Err(HayashiError::Runtime(
                        "feiv(): formula de instrumentos deve ter ao menos um instrumento".into(),
                    ));
                }
                let mut z_mat = ndarray::Array2::<f64>::zeros((n, l));
                for (j, col_name) in instr_vars.iter().enumerate() {
                    let col = df.get(col_name).map_err(|_| {
                        HayashiError::Runtime(format!(
                            "feiv: instrumento '{col_name}' not found no DataFrame"
                        ))
                    })?;
                    for (i, &v) in col.iter().enumerate() {
                        z_mat[[i, j]] = v;
                    }
                }

                // entity IDs
                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "feiv: coluna id '{id_col}' not found"
                    )));
                };

                let var_names = df
                    .formula_var_names(&g_endog)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result =
                    greeners::FE2SLS::fit(&y_vec, &x_mat, &z_mat, &entity_ids, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::FE2SLSResult(Rc::new(result)))
            }

            // ── PCSE — Panel-Corrected Standard Errors (Beck & Katz 1995) ─────
            // pcse(formula, df, id=col, time=col)
            "pcse" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let time_col = self.get_time_col(&df_name, opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let entity_ids = Self::col_as_i64(&df, &id_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let time_ids = Self::col_as_i64(&df, &time_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result =
                    greeners::PCSE::fit(&y_vec, &x_mat, &entity_ids, &time_ids, Some(var_names))
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PcseResult(Rc::new(result)))
            }

            // ── Panel GLS — Parks (1967) / Stata xtgls ───────────────────────
            // xtgls(formula, df, id=col, time=col [, panels="hetero"|"corr"])
            "xtgls" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let time_col = self.get_time_col(&df_name, opt_map)?;
                let panels_opt = match opt_map.get("panels") {
                    Some(Value::Str(s)) if s == "corr" => greeners::GlsPanels::Correlated,
                    Some(Value::Str(s)) if s == "hetero" || s == "heteroscedastic" => {
                        greeners::GlsPanels::Hetero
                    }
                    None => greeners::GlsPanels::Hetero,
                    _ => {
                        return Err(HayashiError::Runtime(
                            "xtgls(): panels deve ser \"hetero\" ou \"corr\"".into(),
                        ))
                    }
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let entity_ids = Self::col_as_i64(&df, &id_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let time_ids = Self::col_as_i64(&df, &time_col)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let var_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = greeners::PanelGLS::fit(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_ids,
                    panels_opt,
                    Some(var_names),
                )
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::PanelGlsResult(Rc::new(result)))
            }

            // ── Arellano-Bond: teste m1/m2 para autocorrelação serial ─────────
            "ab_test" => {
                // ab_test(formula, df, id=col, time=col)
                // Testa autocorrelação serial nos resíduos da equação em 1ª diferença.
                // m1: DEVE rejeitar H₀ (FD induz AR(1) por construção)
                // m2: NÃO deve rejeitar H₀ (valida instrumentos y_{i,t-2} do GMM)
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let time_col = self.get_time_col(&df_name, opt_map)?;

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ab_test: coluna id '{id_col}' not found"
                    )));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "ab_test: coluna time '{time_col}' not found"
                    )));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        s.insert(id);
                    }
                    s.len()
                };

                let (m1, p1, m2, p2) = greeners::PanelDiagnostics::arellano_bond_test(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_vals,
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
                let n_obs = y_vec.len();

                let thick = "═".repeat(66);
                let thin = "─".repeat(66);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(
                    " Arellano-Bond Test (autocorrelação serial — resíduos em 1ª diferença)\n",
                );
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: n={} obs   N={} entidades\n",
                    n_obs, n_entities
                ));
                out.push_str("\n── Estatísticas  z ~ N(0,1)   H₀: sem autocorrelação de ordem p\n");
                out.push_str(&format!("   {:-^52}\n", ""));
                out.push_str(&format!(
                    "   {:>4}  {:>10}  {:>10}  {:>6}  {}\n",
                    "p", "z", "p-valor", "sig", "Interpretação"
                ));
                out.push_str(&format!("   {:-^52}\n", ""));
                let interp1 = if p1 < 0.05 {
                    "OK — FD induz AR(1) (esperado)"
                } else {
                    "Inesperado — verificar modelo"
                };
                let interp2 = if p2 >= 0.05 {
                    "OK — instrumentos válidos"
                } else {
                    "Atenção — AR(2) detectado"
                };
                out.push_str(&format!(
                    "   {:>4}  {:>10.4}  {:>10.4}  {:>6}  {}\n",
                    1,
                    m1,
                    p1,
                    sig(p1),
                    interp1
                ));
                out.push_str(&format!(
                    "   {:>4}  {:>10.4}  {:>10.4}  {:>6}  {}\n",
                    2,
                    m2,
                    p2,
                    sig(p2),
                    interp2
                ));
                out.push_str(&format!("   {:-^52}\n", ""));
                out.push_str("\n── Conclusão\n");
                if p1 < 0.05 && p2 >= 0.05 {
                    out.push_str(
                        "   m1 rejeita e m2 não rejeita → estrutura consistente com GMM válido\n",
                    );
                } else if p1 >= 0.05 {
                    out.push_str(
                        "   m1 não rejeita H₀ → checar especificação (AR(1) esperado em FD)\n",
                    );
                } else {
                    out.push_str("   m2 rejeita H₀ → AR(2) nos resíduos; instrumentos y_{t-2} podem ser inválidos\n");
                    out.push_str(
                        "   Considere usar lags mais distantes (y_{t-3}, ...) como instrumentos\n",
                    );
                }
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(
                    "   Variância estimada via sandwich (Σ_i dos produtos cruzados por entidade)\n",
                );
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── wooldridge_OLD_REMOVED (substituído pelo novo acima) ──────────
            "wooldridge_OLD_REMOVED" => {
                let (formula_ast, df, df_name, id_col) = self.extract_panel_args(args, opt_map)?;
                let time_col = self.get_time_col(&df_name, opt_map)?;

                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y_vec, x_mat) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let entity_ids: Vec<i64> = if let Ok(ids) = df.get_int(&id_col) {
                    ids.to_vec()
                } else if let Ok(floats) = df.get(&id_col) {
                    floats.iter().map(|&v| v as i64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "wooldridge: coluna id '{id_col}' not found"
                    )));
                };

                let time_vals: Vec<f64> = if let Ok(arr) = df.get(&time_col) {
                    arr.to_vec()
                } else if let Ok(arr) = df.get_int(&time_col) {
                    arr.iter().map(|&v| v as f64).collect()
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "wooldridge: coluna time '{time_col}' not found"
                    )));
                };

                let n_entities = {
                    let mut s = std::collections::HashSet::new();
                    for &id in &entity_ids {
                        s.insert(id);
                    }
                    s.len()
                };

                let (rho, t_stat, p, n_pairs) = greeners::PanelDiagnostics::wooldridge_serial(
                    &y_vec,
                    &x_mat,
                    &entity_ids,
                    &time_vals,
                )
                .map_err(HayashiError::Runtime)?;

                let df_t = n_entities - 1;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → autocorrelação serial de 1ª ordem presente"
                } else {
                    "Não rejeita H₀ → sem evidência de autocorrelação serial"
                };

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();
                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Wooldridge Test (autocorrelação serial em painel)\n");
                out.push_str(" H₀: ρ = -0.5  (sem autocorrelação nos erros idiossincráticos)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str(&format!(
                    "\n── Painel: N={} entidades   pares usados={}   df={}\n",
                    n_entities, n_pairs, df_t
                ));
                out.push_str("\n── Estimativa\n");
                out.push_str(&format!("   ρ̂ = {:.4}   (H₀: ρ = -0.500)\n", rho));
                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   t({}) = {:.4}   p = {:.4}  {}\n",
                    df_t, t_stat, p, sig
                ));
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(
                    "   (SE padrão OLS — use SE robustos clusterizados para inferência formal)\n",
                );
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Hausman FE vs RE ──────────────────────────────────────────────
            "hausman" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("hausman(fe_model, re_model)".into()));
                }

                let fe = match self.eval_expr(&args[0])? {
                    Value::PanelResult(r) => r,
                    _ => {
                        return Err(HayashiError::Type(
                            "hausman(): primeiro argumento deve ser um modelo FE".into(),
                        ))
                    }
                };
                let re = match self.eval_expr(&args[1])? {
                    Value::ReResult(r) => r,
                    _ => {
                        return Err(HayashiError::Type(
                            "hausman(): second argument must be um modelo RE".into(),
                        ))
                    }
                };

                // Variáveis comuns: FE não tem intercepto; RE tem.
                // Alinha por nome quando disponível; senão assume mesma ordem.
                let fe_names: Vec<String> =
                    fe.variable_names.as_ref().cloned().unwrap_or_else(|| {
                        (0..fe.params.len()).map(|i| format!("x{}", i)).collect()
                    });

                let re_names: Vec<String> =
                    re.variable_names.as_ref().cloned().unwrap_or_else(|| {
                        (0..re.params.len()).map(|i| format!("x{}", i)).collect()
                    });

                // Pares (β_FE, σ²_FE, β_RE, σ²_RE) para variáveis em comum (exclui intercepto)
                let mut pairs: Vec<(String, f64, f64, f64, f64)> = Vec::new();
                for (i, fe_name) in fe_names.iter().enumerate() {
                    if fe_name == "const" {
                        continue;
                    }
                    if let Some(j) = re_names.iter().position(|n| n == fe_name) {
                        pairs.push((
                            fe_name.clone(),
                            fe.params[i],
                            fe.std_errors[i].powi(2),
                            re.params[j],
                            re.std_errors[j].powi(2),
                        ));
                    }
                }

                if pairs.is_empty() {
                    return Err(HayashiError::Runtime(
                        "hausman: nenhuma variável comum entre FE e RE (verifique variable_names)"
                            .into(),
                    ));
                }

                // H = Σ (β_FE - β_RE)² / (σ²_FE - σ²_RE)  para pares onde σ²_FE > σ²_RE
                let mut chi2 = 0.0;
                let mut df = 0usize;
                let mut skipped = 0usize;

                let thick = "═".repeat(62);
                let thin = "─".repeat(62);
                let mut out = String::new();

                out.push_str(&format!("\n{thick}\n"));
                out.push_str(" Hausman Test: FE vs RE\n");
                out.push_str(" H₀: efeitos individuais não correlacionados com regressores (RE consistente)\n");
                out.push_str(&format!("{thick}\n"));
                out.push_str("\n── Coeficientes Comuns\n");
                out.push_str(&format!(
                    "   {:<20} {:>10} {:>10} {:>10}\n",
                    "Variável", "β_FE", "β_RE", "Δβ"
                ));
                out.push_str(&format!("   {thin}\n"));

                for (name, bfe, vfe, bre, vre) in &pairs {
                    let diff = bfe - bre;
                    let dvar = vfe - vre;
                    out.push_str(&format!(
                        "   {:<20} {:>10.4} {:>10.4} {:>10.4}\n",
                        name, bfe, bre, diff
                    ));
                    if dvar > 1e-15 {
                        chi2 += diff.powi(2) / dvar;
                        df += 1;
                    } else {
                        skipped += 1;
                    }
                }

                if df == 0 {
                    out.push_str("\n   [!] Var(β_FE) ≤ Var(β_RE) em todos os coeficientes.\n");
                    out.push_str(
                        "       Estatística indefinida — verifique especificação dos modelos.\n",
                    );
                    out.push_str(&format!("\n{thick}\n"));
                    return Ok(Some(diag(out)));
                }

                let p = greeners::chi2_pvalue(chi2, df as f64);

                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let verdict = if p < 0.05 {
                    "Rejeita H₀ → use EFEITOS FIXOS (RE pode ser inconsistente)"
                } else {
                    "Não rejeita H₀ → EFEITOS ALEATÓRIOS é consistente e eficiente"
                };

                out.push_str("\n── Estatística\n");
                out.push_str(&format!(
                    "   χ²({}) = {:.4}   p = {:.4}  {}\n",
                    df, chi2, p, sig
                ));
                if skipped > 0 {
                    out.push_str(&format!(
                        "   ({} coeficiente(s) excluídos: Var(β_FE) ≤ Var(β_RE))\n",
                        skipped
                    ));
                }
                out.push_str("\n── Conclusão\n");
                out.push_str(&format!("   {}\n", verdict));
                out.push_str(&format!("\n{thin}\n"));
                out.push_str("   *** p<0.01  ** p<0.05  * p<0.10\n");
                out.push_str(&format!("{thick}\n"));
                Ok(diag(out))
            }

            // ── Diagnósticos ──────────────────────────────────────────────────
            "test" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "test(model, name) requires 2 arguments".into(),
                    ));
                }
                let model = self.eval_expr(&args[0])?;

                let ols = match &model {
                    Value::OlsResult(m) => m.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "test() currently supports OLS models only".into(),
                        ))
                    }
                };

                let test_name = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    other => {
                        return Err(HayashiError::Type(format!(
                            "test name must be a string (e.g. \"white\"), got {other}"
                        )))
                    }
                };

                match test_name.as_str() {
                    // ── Specification tests ──────────────────────────────
                    "white" => match SpecificationTests::white_test(&ols.residuals, &ols.x) {
                        Ok((stat, p, df)) => {
                            println!("White Test for Heteroskedasticity");
                            println!("  LM statistic : {:.4}", stat);
                            println!("  p-value      : {:.4}", p);
                            println!("  df           : {}", df);
                            let verdict = if p < 0.05 {
                                "Reject H0 — evidence of heteroskedasticity"
                            } else {
                                "Fail to reject H0 — no evidence of heteroskedasticity"
                            };
                            println!("  Conclusion   : {}", verdict);
                        }
                        Err(e) => eprintln!("White test error: {e}"),
                    },
                    "bp" => match Diagnostics::breusch_pagan(&ols.residuals, &ols.x) {
                        Ok((stat, p)) => {
                            println!("Breusch-Pagan Test for Heteroskedasticity");
                            println!("  LM statistic : {:.4}", stat);
                            println!("  p-value      : {:.4}", p);
                            let verdict = if p < 0.05 {
                                "Reject H0 — evidence of heteroskedasticity"
                            } else {
                                "Fail to reject H0 — no evidence of heteroskedasticity"
                            };
                            println!("  Conclusion   : {}", verdict);
                        }
                        Err(e) => eprintln!("Breusch-Pagan test error: {e}"),
                    },
                    "dw" => {
                        let stat = Diagnostics::durbin_watson(&ols.residuals);
                        println!("Durbin-Watson Test for Autocorrelation");
                        println!("  DW statistic : {:.4}", stat);
                        let verdict = if stat < 1.5 {
                            "Positive autocorrelation suspected"
                        } else if stat > 2.5 {
                            "Negative autocorrelation suspected"
                        } else {
                            "No strong evidence of autocorrelation"
                        };
                        println!("  Conclusion   : {}", verdict);
                    }

                    // ── Wald / F-test sobre coeficientes ─────────────────
                    other => {
                        let names = ols.result.variable_names.as_ref().ok_or_else(|| {
                            HayashiError::Runtime("model has no variable names".into())
                        })?;
                        let k = ols.result.params.len();
                        let find_idx = |name: &str| -> Result<usize> {
                            let n = name.trim();
                            names
                                .iter()
                                .position(|v| v == n)
                                .or_else(|| {
                                    if n == "_cons" || n == "const" {
                                        Some(k - 1)
                                    } else {
                                        None
                                    }
                                })
                                .ok_or_else(|| {
                                    HayashiError::Runtime(format!(
                                        "variable '{n}' not found in model"
                                    ))
                                })
                        };

                        // "X1 = X2" ou "X1 = 0.5"
                        if let Some((lhs_s, rhs_s)) = other.split_once('=') {
                            let lhs_name = lhs_s.trim();
                            let rhs_trimmed = rhs_s.trim();
                            if let Ok(val) = rhs_trimmed.parse::<f64>() {
                                let idx = find_idx(lhs_name)?;
                                let mut r = ndarray::Array1::<f64>::zeros(k);
                                r[idx] = 1.0;
                                let (t, p) = ols
                                    .result
                                    .t_test(&r, val, &ols.x)
                                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                                println!("\n{:=^60}", " test ");
                                println!("  H₀: {lhs_name} = {val}");
                                println!("  t = {t:.4}   p = {p:.4}");
                                let sig = if p < 0.01 {
                                    "***"
                                } else if p < 0.05 {
                                    "**"
                                } else if p < 0.10 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!("  {sig}");
                            } else {
                                let idx1 = find_idx(lhs_name)?;
                                let idx2 = find_idx(rhs_trimmed)?;
                                let mut r = ndarray::Array1::<f64>::zeros(k);
                                r[idx1] = 1.0;
                                r[idx2] = -1.0;
                                let (t, p) = ols
                                    .result
                                    .t_test(&r, 0.0, &ols.x)
                                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                                println!("\n{:=^60}", " test ");
                                println!("  H₀: {lhs_name} = {rhs_trimmed}");
                                println!("  t = {t:.4}   p = {p:.4}");
                                let sig = if p < 0.01 {
                                    "***"
                                } else if p < 0.05 {
                                    "**"
                                } else if p < 0.10 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!("  {sig}");
                            }
                        } else {
                            let mut extra_names: Vec<String> = Vec::new();
                            for arg in &args[2..] {
                                let name = match self.eval_expr(arg)? {
                                    Value::Str(s) => s,
                                    other => {
                                        return Err(HayashiError::Type(format!(
                                            "test() variable names must be strings, got {other}"
                                        )))
                                    }
                                };
                                extra_names.push(name);
                            }
                            let mut indices = vec![find_idx(other)?];
                            for name in &extra_names {
                                indices.push(find_idx(name)?);
                            }
                            let (f, p) = ols
                                .result
                                .f_test(&indices, &ols.x)
                                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                            let var_list: Vec<&str> =
                                indices.iter().map(|&i| names[i].as_str()).collect();
                            let q = indices.len();
                            println!("\n{:=^60}", " test ");
                            if q == 1 {
                                println!("  H₀: {} = 0", var_list[0]);
                            } else {
                                println!("  H₀: {} = 0", var_list.join(" = "));
                            }
                            println!("  F({q}, {}) = {f:.4}   p = {p:.4}", ols.result.df_resid);
                            let sig = if p < 0.01 {
                                "***"
                            } else if p < 0.05 {
                                "**"
                            } else if p < 0.10 {
                                "*"
                            } else {
                                ""
                            };
                            println!("  {sig}");
                        }
                    }
                }

                Ok(Value::Nil)
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }

    // ── Bootstrap helpers ─────────────────────────────────────────────────────

    fn bootstrap_reps(opt_map: &HashMap<String, Value>) -> usize {
        match opt_map.get("n") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => match opt_map.get("reps") {
                Some(Value::Int(v)) => *v as usize,
                Some(Value::Float(v)) => *v as usize,
                _ => 1000,
            },
        }
    }

    fn bootstrap_alpha(opt_map: &HashMap<String, Value>) -> f64 {
        match opt_map.get("alpha") {
            Some(Value::Float(v)) => *v,
            _ => 0.05,
        }
    }

    fn bootstrap_generic(
        &mut self,
        args: &[Expr],
        opts: &[Opt],
        n_boot: usize,
        alpha: f64,
    ) -> Result<Value> {
        let estimator_name = match &args[0] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "bootstrap: first argument must be nome do estimador (ols, logit, ...)"
                        .into(),
                ))
            }
        };
        let formula_expr = args[1].clone();
        let df_name = match &args[2] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "bootstrap: third argument must be nome do DataFrame".into(),
                ))
            }
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(d)) => d.clone(),
            _ => {
                return Err(HayashiError::Runtime(format!(
                    "'{df_name}' is not a DataFrame"
                )))
            }
        };

        let extra_opts: Vec<Opt> = opts
            .iter()
            .filter(|o| !matches!(o.name.as_str(), "n" | "reps" | "alpha"))
            .cloned()
            .collect();
        let full_result = self.eval_call(
            &estimator_name,
            &[formula_expr.clone(), Expr::Var(df_name.clone())],
            &extra_opts,
        )?;
        let full_params = extract_params(&full_result).ok_or_else(|| {
            HayashiError::Runtime(
                "bootstrap: modelo not supportado (sem params extraíveis)".into(),
            )
        })?;
        let full_se = extract_se(&full_result).unwrap_or_default();
        let var_names = extract_var_names(&full_result);
        let k = full_params.len();

        use rand::seq::SliceRandom;
        let mut rng = self.get_rng();
        let n = df.n_rows();
        let indices: Vec<usize> = (0..n).collect();
        let mut boot_coefs = ndarray::Array2::<f64>::zeros((n_boot, k));
        let mut n_ok = 0usize;

        for b in 0..n_boot {
            let boot_idx: Vec<usize> =
                (0..n).map(|_| *indices.choose(&mut rng).unwrap()).collect();
            let boot_df = match df.iloc(Some(&boot_idx), None) {
                Ok(d) => d,
                Err(_) => continue,
            };
            self.env
                .set("__boot_df__", Value::DataFrame(Rc::new(boot_df)))?;
            if let Ok(ref result) = self.eval_call(
                &estimator_name,
                &[formula_expr.clone(), Expr::Var("__boot_df__".into())],
                &extra_opts,
            ) {
                if let Some(params) = extract_params(result) {
                    for j in 0..k.min(params.len()) {
                        boot_coefs[[b, j]] = params[j];
                    }
                    n_ok += 1;
                }
            }
        }
        self.env.remove("__boot_df__");

        if n_ok < 10 {
            return Err(HayashiError::Runtime(format!(
                "bootstrap: apenas {n_ok}/{n_boot} replicações convergiram"
            )));
        }

        let boot_se = greeners::Bootstrap::bootstrap_se(&boot_coefs);
        let (ci_lo, ci_hi) = greeners::Bootstrap::percentile_ci(&boot_coefs, alpha);

        let thick = "═".repeat(76);
        let thin = "─".repeat(76);
        println!("\n{thick}");
        println!(
            "{:^76}",
            format!(" Bootstrap SE — {} (n={n_ok}/{n_boot}) ", estimator_name)
        );
        println!("{thin}");
        println!(
            "{:<18} {:>10} {:>10} {:>10} {:>12} {:>12}",
            "Variável", "β̂", "SE orig.", "SE boot", "IC inf", "IC sup"
        );
        println!("{thin}");
        for i in 0..k {
            let vname = var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
            let orig_se = if i < full_se.len() { full_se[i] } else { f64::NAN };
            println!(
                "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>12.4}",
                vname, full_params[i], orig_se, boot_se[i], ci_lo[i], ci_hi[i]
            );
        }
        println!("{thick}");
        Ok(Value::Nil)
    }

    fn bootstrap_pairs(&mut self, args: &[Expr], n_boot: usize, alpha: f64) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "bootstrap(estimator, formula, df, n=1000) ou bootse(model, n=1000)".into(),
            ));
        }
        let model_val = self.eval_expr(&args[0])?;
        match &model_val {
            Value::OlsResult(m) => {
                let y_hat = m.x.dot(&m.result.params);
                let y_vec = &y_hat + &m.residuals;
                let boot_coefs = greeners::Bootstrap::pairs_bootstrap(&y_vec, &m.x, n_boot)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let boot_se = greeners::Bootstrap::bootstrap_se(&boot_coefs);
                let (ci_lo, ci_hi) = greeners::Bootstrap::percentile_ci(&boot_coefs, alpha);
                let vnames = m.result.variable_names.as_deref().unwrap_or(&[]);
                let k = m.result.params.len();
                let thick = "═".repeat(76);
                let thin = "─".repeat(76);
                println!("\n{thick}");
                println!("{:^76}", format!(" Bootstrap SE (n={n_boot}, pairs) "));
                println!("{thin}");
                println!(
                    "{:<18} {:>10} {:>10} {:>10} {:>12} {:>12}",
                    "Variável", "β̂", "SE orig.", "SE boot", "IC inf 95%", "IC sup 95%"
                );
                println!("{thin}");
                for i in 0..k {
                    let vname = vnames.get(i).map(|s| s.as_str()).unwrap_or("?");
                    println!(
                        "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>12.4}",
                        vname,
                        m.result.params[i],
                        m.result.std_errors[i],
                        boot_se[i],
                        ci_lo[i],
                        ci_hi[i]
                    );
                }
                println!("{thick}");
                Ok(Value::Nil)
            }
            _ => Err(HayashiError::Runtime(
                "bootse(model) suporta OLS. Para outros: bootstrap(estimator, formula, df, n=1000)"
                    .into(),
            )),
        }
    }
}
