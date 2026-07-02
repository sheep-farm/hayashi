use super::*;

/// GARCH/EGARCH/GJR-GARCH, VARMA, decomposição sazonal, MSTL, testes de
/// proporção, testes múltiplos, UCM, GAM, MICE, Markov Switching, SVAR, 3SLS,
/// DFM e diagnósticos menores de normalidade / forma funcional.
/// Extraído de `eval_call` (ver src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_estimators_timeseries(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── garch / egarch / gjrgarch ────────────────────────────────────
            // garch(df, varname, p=1, q=1)
            // garch(df, varname, p=1, q=1, dist=t)    — erros Student-t
            // egarch(df, varname, p=1, q=1)
            // gjrgarch(df, varname, p=1, q=1)
            "garch" | "egarch" | "gjrgarch" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(format!(
                        "{func}() requer df e variable name"
                    )));
                }

                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => {
                        return Err(HayashiError::Type(format!(
                            "{func}(): primeiro argumento deve ser um DataFrame"
                        )))
                    }
                };

                let col_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(format!(
                            "{func}(): second argument must be o nome de uma coluna"
                        )))
                    }
                };

                let p = match opt_map.get("p") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let q = match opt_map.get("q") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let use_t_dist = matches!(
                    opt_map.get("dist"),
                    Some(Value::Str(s)) if s == "t"
                );

                let y = Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?);

                let result = match (func, use_t_dist) {
                    ("garch", false) => greeners::GARCH::fit(&y, p, q),
                    ("garch", true) => greeners::GARCH::fit_t(&y, p, q),
                    ("egarch", false) => greeners::EGARCH::fit(&y, p, q),
                    ("egarch", true) => greeners::EGARCH::fit_t(&y, p, q),
                    ("gjrgarch", false) => greeners::GJRGARCH::fit(&y, p, q),
                    ("gjrgarch", true) => greeners::GJRGARCH::fit_t(&y, p, q),
                    _ => unreachable!(),
                };

                Ok(Value::GarchResult(Rc::new(
                    result.map_err(|e| self.rt_err(format!("{func}: {e}")))?,
                )))
            }

            // ljungbox(df, varname, lags=10)
            // ljungbox(model, lags=10)   — aceita GARCH, ARIMA, OLS
            // H₀: as primeiras `lags` autocorrelações são conjuntamente zero
            "ljungbox" | "ljung_box" | "portmanteau" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "ljungbox() requires a series or model".into(),
                    ));
                }

                let series = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => {
                        let col_name =
                            match args.get(1) {
                                Some(Expr::Var(n)) => n.clone(),
                                _ => return Err(HayashiError::Runtime(
                                    "ljungbox(df, varname): second argument must be a column name"
                                        .into(),
                                )),
                            };
                        Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?)
                    }
                    // resíduos padronizados de GARCH
                    Value::GarchResult(m) => m.standardized_residuals.clone(),
                    // resíduos de ARIMA
                    Value::ArimaResult(m) => Array1::from_vec(m.residuals().to_vec()),
                    // resíduos de OLS
                    Value::OlsResult(m) => m.residuals.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "ljungbox(): argument must be a DataFrame, GARCH, ARIMA, or OLS".into(),
                        ))
                    }
                };

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let res = greeners::Diagnostics::ljung_box(&series, lags)
                    .map_err(|e| self.rt_err(format!("ljungbox: {e}")))?;

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
                let sep = "─".repeat(62);
                println!(
                    "\nLjung-Box Test  —  lags = {}  n = {}",
                    res.lags, res.n_obs
                );
                println!("{sep}");
                println!("H₀: sem autocorrelação até lag {}", res.lags);
                println!("{sep}");
                println!("{:<6} {:>10} {:>10} {:>8}", "lag", "ACF", "Q", "p-value");
                println!("{sep}");
                let mut q_accum = 0.0_f64;
                let nf = res.n_obs as f64;
                for (i, &rho) in res.acf.iter().enumerate() {
                    let k = i + 1;
                    q_accum += nf * (nf + 2.0) * rho * rho / (nf - k as f64);
                    // p-value cumulativo para o Q até lag k
                    let p_k = greeners::chi2_pvalue(q_accum, k as f64);
                    println!(
                        "{:<6} {:>10.4} {:>10.4} {:>8.4} {:>3}",
                        k,
                        rho,
                        q_accum,
                        p_k,
                        sig(p_k)
                    );
                }
                println!("{sep}");
                println!(
                    "Q({lags}) = {:.4}   p = {:.4}  {}   (*** p<0.01  ** p<0.05  * p<0.10)",
                    res.q_stat,
                    res.p_value,
                    sig(res.p_value)
                );
                println!();

                Ok(Value::Nil)
            }

            // leverage(model)
            // leverage(model, threshold=2)   — múltiplo de k/n; padrão 2
            // Diagonal da hat matrix: h_i = x_i'(X'X)⁻¹x_i
            // Observações com h_i > threshold*k/n merecem atenção
            "leverage" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "leverage() requires an OLS model".into(),
                    ));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "leverage() only supports OLS models".into(),
                        ))
                    }
                };

                let threshold = match opt_map.get("threshold") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 2.0,
                };

                let h = greeners::Diagnostics::leverage(&ols.x)
                    .map_err(|e| self.rt_err(format!("leverage: {e}")))?;

                let n = h.len();
                let k = ols.x.ncols();
                let cutoff = threshold * k as f64 / n as f64;
                let h_mean = k as f64 / n as f64;

                // mostra apenas observações acima do cutoff (ou todas se poucas)
                let flagged: Vec<(usize, f64)> = h
                    .iter()
                    .enumerate()
                    .filter(|(_, &hi)| hi > cutoff)
                    .map(|(i, &hi)| (i + 1, hi))
                    .collect();

                let sep = "─".repeat(46);
                println!(
                    "\nLeverage  —  h̄ = {:.4}  cutoff = {:.4} ({}×k/n)",
                    h_mean, cutoff, threshold
                );
                println!("{sep}");
                if flagged.is_empty() {
                    println!("Nenhuma observação acima do cutoff.");
                } else {
                    println!("{:<8} {:>10}  ", "obs", "h_i");
                    println!("{sep}");
                    for (i, hi) in &flagged {
                        println!("{:<8} {:>10.4}  high leverage", i, hi);
                    }
                    println!("{sep}");
                    println!("{} observação(ões) com h_i > {:.4}", flagged.len(), cutoff);
                }
                println!();

                Ok(Value::Nil)
            }

            // cooks(model)
            // cooks(model, threshold=1)   — limiar absoluto padrão; ou usa 4/n
            // D_i = (e_i²·h_i) / (k·MSE·(1−h_i)²)
            "cooks" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "cooks() requires an OLS model".into(),
                    ));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "cooks() only supports OLS models".into(),
                        ))
                    }
                };

                let mse = ols.result.sigma * ols.result.sigma;
                let d = greeners::Diagnostics::cooks_distance(&ols.residuals, &ols.x, mse)
                    .map_err(|e| self.rt_err(format!("cooks: {e}")))?;

                let n = d.len();
                let k = ols.x.ncols();
                // cutoff configurável; padrão 4/n (regra de bolso mais comum)
                let cutoff = match opt_map.get("threshold") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 4.0 / n as f64,
                };

                let flagged: Vec<(usize, f64)> = d
                    .iter()
                    .enumerate()
                    .filter(|(_, &di)| di > cutoff)
                    .map(|(i, &di)| (i + 1, di))
                    .collect();

                let sep = "─".repeat(46);
                println!("\nCook's Distance  —  n={n}  k={k}  cutoff={cutoff:.4} (4/n)");
                println!("{sep}");
                if flagged.is_empty() {
                    println!("Nenhuma observação influente acima do cutoff.");
                } else {
                    println!("{:<8} {:>10}  ", "obs", "D_i");
                    println!("{sep}");
                    for (i, di) in &flagged {
                        let label = if *di > 1.0 {
                            "muito influente"
                        } else {
                            "influente"
                        };
                        println!("{:<8} {:>10.4}  {}", i, di, label);
                    }
                    println!("{sep}");
                    println!("{} observação(ões) com D_i > {:.4}", flagged.len(), cutoff);
                }
                println!();

                Ok(Value::Nil)
            }

            // vif(model)
            // Variance Inflation Factor — detecta multicolinearidade por variável
            // VIF_j = 1/(1−R²_j); VIF>10 indica multicolinearidade grave
            "vif" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("vif() requires an OLS model".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => return Err(HayashiError::Type("vif() only supports OLS models".into())),
                };

                let vifs = greeners::Diagnostics::vif(&ols.x)
                    .map_err(|e| self.rt_err(format!("vif: {e}")))?;

                let names = ols.result.variable_names.as_deref().unwrap_or(&[]);

                let sep = "─".repeat(40);
                println!("\nVariance Inflation Factor (VIF)");
                println!("{sep}");
                println!("{:<20} {:>8}  Diagnóstico", "Variável", "VIF");
                println!("{sep}");
                for (i, &v) in vifs.iter().enumerate() {
                    let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                    let diag = if v.is_nan() {
                        "constante"
                    } else if v.is_infinite() || v > 10.0 {
                        "multicolinearidade grave"
                    } else if v > 5.0 {
                        "moderada"
                    } else {
                        "ok"
                    };
                    if v.is_nan() {
                        println!("{:<20} {:>8}  {}", name, "—", diag);
                    } else if v.is_infinite() {
                        println!("{:<20} {:>8}  {}", name, "∞", diag);
                    } else {
                        println!("{:<20} {:>8.3}  {}", name, v, diag);
                    }
                }
                println!("{sep}");
                println!("Referência: VIF<5 ok  |  5-10 moderado  |  >10 grave");
                println!();

                Ok(Value::Nil)
            }

            // condnum(model)
            // Condition number da matriz X — diagnóstico global de multicolinearidade
            // κ = σ_max/σ_min; κ>30 indica problema sério
            "condnum" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "condnum() requires an OLS model".into(),
                    ));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "condnum() only supports OLS models".into(),
                        ))
                    }
                };

                let kappa = greeners::Diagnostics::condition_number(&ols.x)
                    .map_err(|e| self.rt_err(format!("condnum: {e}")))?;

                let diag = if kappa.is_infinite() || kappa > 100.0 {
                    "multicolinearidade severa"
                } else if kappa > 30.0 {
                    "multicolinearidade moderada"
                } else if kappa > 10.0 {
                    "atenção"
                } else {
                    "ok"
                };

                let sep = "─".repeat(44);
                println!("\nCondition Number (multicolinearidade global)");
                println!("{sep}");
                if kappa.is_infinite() {
                    println!("{:<20} {:>12}  {}", "κ(X)", "∞", diag);
                } else {
                    println!("{:<20} {:>12.2}  {}", "κ(X)", kappa, diag);
                }
                println!("{sep}");
                println!(
                    "Referência: κ<10 ok  |  10-30 atenção  |  30-100 moderado  |  >100 severo"
                );
                println!();

                Ok(Value::Nil)
            }

            // durbinwatson(model)
            // Durbin-Watson: detecta autocorrelação de primeira ordem nos resíduos OLS
            // DW ≈ 2 → sem autocorrelação; DW < 2 → positiva; DW > 2 → negativa
            "durbinwatson" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "durbinwatson() requires an OLS model".into(),
                    ));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "durbinwatson() only supports OLS models".into(),
                        ))
                    }
                };

                let dw = greeners::Diagnostics::durbin_watson(&ols.residuals);

                let interpretation = if dw < 1.5 {
                    "autocorrelação positiva provável"
                } else if dw > 2.5 {
                    "autocorrelação negativa provável"
                } else {
                    "sem autocorrelação evidente"
                };

                let sep = "─".repeat(50);
                println!("\nDurbin-Watson Test");
                println!("{sep}");
                println!("H₀: sem autocorrelação de primeira ordem");
                println!("{sep}");
                println!("{:<18} {:>10}", "DW statistic", format!("{dw:.4}"));
                println!("{:<18} {:>10}", "Interpretação", interpretation);
                println!("{sep}");
                println!("Referência: DW ≈ 2 (sem autocorr.) | <1.5 (positiva) | >2.5 (negativa)");
                println!();

                Ok(Value::Nil)
            }

            // white(model)
            // White (1980): H₀: homocedasticidade
            // Requer modelo OLS — regride u² nos regressores e seus quadrados
            "white" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "white() requires an OLS model".into(),
                    ));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "white() only supports OLS models".into(),
                        ))
                    }
                };

                let (lm, p, df) = greeners::SpecificationTests::white_test(&ols.residuals, &ols.x)
                    .map_err(|e| self.rt_err(format!("white: {e}")))?;

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
                let sep = "─".repeat(54);
                println!("\nWhite Test (heteroscedasticidade)");
                println!("{sep}");
                println!("H₀: homocedasticidade");
                println!("{sep}");
                println!(
                    "{:<24} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<24} {:>10.4} {:>10.4} {:>4}",
                    format!("LM ~ χ²({df})"),
                    lm,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();

                Ok(Value::Nil)
            }

            // reset(model)
            // reset(model, power=3)
            // Ramsey RESET: H₀: especificação linear correta
            // Requer modelo OLS — adiciona ŷ², ..., ŷ^power como regressores
            "reset" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "reset() requires an OLS model".into(),
                    ));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "reset() only supports OLS models".into(),
                        ))
                    }
                };

                let power = match opt_map.get("power") {
                    Some(Value::Int(v)) => (*v as usize).max(2),
                    Some(Value::Float(v)) => (*v as usize).max(2),
                    _ => 3,
                };

                let fitted = ols.result.fitted_values(&ols.x);
                // y = resíduos + valores ajustados
                let y = &ols.residuals + &fitted;

                let (f, p, df1, df2) =
                    greeners::SpecificationTests::reset_test(&y, &ols.x, &fitted, power)
                        .map_err(|e| self.rt_err(format!("reset: {e}")))?;

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
                let sep = "─".repeat(54);
                println!("\nRamsey RESET Test  —  power = {power}");
                println!("{sep}");
                println!("H₀: especificação linear correta");
                println!("{sep}");
                println!(
                    "{:<24} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<24} {:>10.4} {:>10.4} {:>4}",
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

            // jb(df, varname) | jb(model)
            // Jarque-Bera: H₀: resíduos normalmente distribuídos
            // Aceita série bruta, OLS, ARIMA, GARCH (resíduos padronizados)
            "jb" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "jb() requires a series or model".into(),
                    ));
                }

                let series = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => {
                        let col_name = match args.get(1) {
                            Some(Expr::Var(n)) => n.clone(),
                            _ => {
                                return Err(HayashiError::Runtime(
                                    "jb(df, varname): second argument must be a column name".into(),
                                ))
                            }
                        };
                        Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?)
                    }
                    Value::OlsResult(m) => m.residuals.clone(),
                    Value::ArimaResult(m) => Array1::from_vec(m.residuals().to_vec()),
                    Value::GarchResult(m) => m.standardized_residuals.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "jb(): argument must be a DataFrame, OLS, ARIMA, or GARCH".into(),
                        ))
                    }
                };

                let (jb, p) = greeners::Diagnostics::jarque_bera(&series)
                    .map_err(|e| self.rt_err(format!("jb: {e}")))?;

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
                let sep = "─".repeat(50);
                println!("\nJarque-Bera Test  —  n = {}", series.len());
                println!("{sep}");
                println!("H₀: resíduos normalmente distribuídos");
                println!("{sep}");
                println!("{:<18} {:>10} {:>10} {:>4}", "Teste", "JB", "p-value", "");
                println!("{sep}");
                println!(
                    "{:<18} {:>10.4} {:>10.4} {:>4}",
                    "Jarque-Bera ~ χ²(2)",
                    jb,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();

                Ok(Value::Nil)
            }

            // bgodfrey(model, lags=4)
            // Breusch-Godfrey: H₀: sem autocorrelação serial nos resíduos OLS
            // Requer modelo OLS (precisa da matriz X para a regressão auxiliar)
            "bgodfrey" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "bgodfrey() requires an OLS model".into(),
                    ));
                }

                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "bgodfrey() only supports OLS models".into(),
                        ))
                    }
                };

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 4,
                };

                let (lm, p, df) = greeners::SpecificationTests::breusch_godfrey_test(
                    &ols.residuals,
                    &ols.x,
                    lags,
                )
                .map_err(|e| self.rt_err(format!("bgodfrey: {e}")))?;

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
                let sep = "─".repeat(54);
                println!("\nBreusch-Godfrey LM Test  —  lags = {lags}");
                println!("{sep}");
                println!("H₀: sem autocorrelação serial de ordem {lags}");
                println!("{sep}");
                println!(
                    "{:<24} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<24} {:>10.4} {:>10.4} {:>4}",
                    format!("LM ~ χ²({df})"),
                    lm,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();

                Ok(Value::Nil)
            }

            // aliases para bgodfrey
            "bgtest" | "bg" | "breusch_godfrey" => {
                return self.eval_call("bgodfrey", args, opts).map(Some);
            }

            // archtest(df, varname, lags=5)
            // Engle (1982) LM test — H₀: sem efeitos ARCH de ordem `lags`
            // Também aceita resíduos de modelo GARCH: archtest(model, lags=5)
            "archtest" | "arch_test" | "engle_arch" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "archtest() requires a series or GARCH model".into(),
                    ));
                }

                let series = match self.eval_expr(&args[0])? {
                    // série bruta: archtest(df, varname, lags=5)
                    Value::DataFrame(df) => {
                        let col_name =
                            match args.get(1) {
                                Some(Expr::Var(n)) => n.clone(),
                                _ => return Err(HayashiError::Runtime(
                                    "archtest(df, varname): second argument must be a column name"
                                        .into(),
                                )),
                            };
                        Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?)
                    }
                    // resíduos de GARCH: archtest(model, lags=5)
                    // usa resíduos padronizados z_t = ε_t/√h_t — sob H₀ de
                    // especificação correta, z_t² não deve ter autocorrelação
                    Value::GarchResult(m) => m.standardized_residuals.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "archtest(): first argument must be a DataFrame or GARCH model".into(),
                        ))
                    }
                };

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 5,
                };

                let res = greeners::Diagnostics::arch_test(&series, lags)
                    .map_err(|e| self.rt_err(format!("archtest: {e}")))?;

                let sep = "─".repeat(54);
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
                    "\nARCH LM Test (Engle 1982)  —  lags = {}  n = {}",
                    res.lags, res.n_obs
                );
                println!("{sep}");
                println!("H₀: sem efeitos ARCH de ordem {}", res.lags);
                println!("{sep}");
                println!(
                    "{:<22} {:>10} {:>10} {:>8}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<22} {:>10.4} {:>10.4} {:>8}",
                    format!("LM  ~ χ²({})", res.lags),
                    res.lm_stat,
                    res.lm_pvalue,
                    sig(res.lm_pvalue)
                );
                println!(
                    "{:<22} {:>10.4} {:>10.4} {:>8}",
                    format!(
                        "F   ~ F({},{})",
                        res.lags,
                        res.n_obs.saturating_sub(res.lags + 1)
                    ),
                    res.f_stat,
                    res.f_pvalue,
                    sig(res.f_pvalue)
                );
                println!("{sep}");
                println!(
                    "R² aux = {:.4}   (*** p<0.01  ** p<0.05  * p<0.10)",
                    res.r_squared
                );
                println!();

                Ok(Value::Nil)
            }

            // forecast_vol(model, steps=10)
            "forecast_vol" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "forecast_vol() requires a GARCH model".into(),
                    ));
                }

                let model = match self.eval_expr(&args[0])? {
                    Value::GarchResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "forecast_vol() requires a GARCH/EGARCH/GJRGARCH model".into(),
                        ))
                    }
                };

                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let vol = model.forecast_volatility(steps);
                let model_label = match model.model_type {
                    greeners::GarchModelType::GARCH => "GARCH",
                    greeners::GarchModelType::EGARCH => "EGARCH",
                    greeners::GarchModelType::GJRGARCH => "GJR-GARCH",
                };
                let dist_label = match model.dist {
                    greeners::GarchDist::Normal => "Normal",
                    greeners::GarchDist::StudentT => "Student-t",
                };

                let sep = "─".repeat(40);
                println!("\nForecast de Volatilidade — {model_label}({}, {}) [{dist_label}]  {steps} passos",
                         model.p, model.q);
                println!("{sep}");
                println!(
                    "{:<6} {:>14} {:>14}",
                    "h", "var. condicional", "volatilidade"
                );
                println!("{sep}");
                for h in 0..steps {
                    println!("{:<6} {:>14.6} {:>14.6}", h + 1, vol[h], vol[h].sqrt());
                }
                println!("{sep}");
                println!();

                Ok(Value::Nil)
            }

            // diagnostics(model)
            // Roda todos os testes aplicáveis ao tipo de modelo e imprime relatório unificado.
            // OLS:  JB, DW, Breusch-Godfrey, White, RESET, VIF, Cook's D
            // GARCH: Ljung-Box, ARCH LM, JB nos resíduos padronizados
            // ARIMA: Ljung-Box, JB nos resíduos
            "diagnostics" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "diagnostics() requires a model (OLS, GARCH, or ARIMA)".into(),
                    ));
                }

                let sig = |p: f64| -> &'static str {
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
                let thick = "═".repeat(62);
                let thin = "─".repeat(62);

                match self.eval_expr(&args[0])? {
                    Value::OlsResult(ols) => {
                        println!("\n{thick}");
                        println!(
                            " DIAGNÓSTICOS — OLS  (n={}  k={})",
                            ols.residuals.len(),
                            ols.x.ncols()
                        );
                        println!("{thick}");

                        // ── Normalidade
                        println!("\n── Normalidade dos Resíduos (Jarque-Bera)");
                        match greeners::Diagnostics::jarque_bera(&ols.residuals) {
                            Ok((jb, p)) => {
                                println!("   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}", jb, p, sig(p))
                            }
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── Autocorrelação 1ª ordem
                        let dw = greeners::Diagnostics::durbin_watson(&ols.residuals);
                        let dw_label = if dw < 1.5 {
                            "autocorr. positiva"
                        } else if dw > 2.5 {
                            "autocorr. negativa"
                        } else {
                            "sem autocorr. evidente"
                        };
                        println!("\n── Autocorrelação 1ª Ordem (Durbin-Watson)");
                        println!("   DW = {:.4}  [{}]", dw, dw_label);

                        // ── Breusch-Godfrey
                        println!("\n── Autocorrelação Serial (Breusch-Godfrey, lags=4)");
                        match greeners::SpecificationTests::breusch_godfrey_test(
                            &ols.residuals,
                            &ols.x,
                            4,
                        ) {
                            Ok((lm, p, df)) => println!(
                                "   LM ~ χ²({})   = {:>9.4}   p = {:.4}  {}",
                                df,
                                lm,
                                p,
                                sig(p)
                            ),
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── White
                        println!("\n── Heteroscedasticidade (White)");
                        match greeners::SpecificationTests::white_test(&ols.residuals, &ols.x) {
                            Ok((lm, p, df)) => println!(
                                "   LM ~ χ²({})   = {:>9.4}   p = {:.4}  {}",
                                df,
                                lm,
                                p,
                                sig(p)
                            ),
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── RESET
                        println!("\n── Especificação Funcional (RESET, power=3)");
                        let fitted = ols.result.fitted_values(&ols.x);
                        let y = &ols.residuals + &fitted;
                        match greeners::SpecificationTests::reset_test(&y, &ols.x, &fitted, 3) {
                            Ok((f, p, df1, df2)) => println!(
                                "   F ~ F({},{}) = {:>9.4}   p = {:.4}  {}",
                                df1,
                                df2,
                                f,
                                p,
                                sig(p)
                            ),
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── VIF
                        println!("\n── Multicolinearidade (VIF)");
                        let names = ols.result.variable_names.as_deref().unwrap_or(&[]);
                        match greeners::Diagnostics::vif(&ols.x) {
                            Ok(vifs) => {
                                for (i, &v) in vifs.iter().enumerate() {
                                    if v.is_nan() {
                                        continue;
                                    }
                                    let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                                    let diag = if v.is_infinite() || v > 10.0 {
                                        "grave"
                                    } else if v > 5.0 {
                                        "moderado"
                                    } else {
                                        "ok"
                                    };
                                    println!("   {:<20} VIF = {:>7.3}  [{}]", name, v, diag);
                                }
                            }
                            Err(e) => println!("   erro: {e}"),
                        }

                        // ── Cook's D
                        let n = ols.residuals.len();
                        let mse = ols.result.sigma * ols.result.sigma;
                        let cutoff = 4.0 / n as f64;
                        println!("\n── Observações Influentes (Cook's D > {:.4})", cutoff);
                        match greeners::Diagnostics::cooks_distance(&ols.residuals, &ols.x, mse) {
                            Ok(d) => {
                                let flagged: Vec<(usize, f64)> = d
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, &di)| di > cutoff)
                                    .map(|(i, &di)| (i + 1, di))
                                    .collect();
                                if flagged.is_empty() {
                                    println!("   Nenhuma observação influente.");
                                } else {
                                    for (i, di) in &flagged {
                                        let label = if *di > 1.0 {
                                            "muito influente"
                                        } else {
                                            "influente"
                                        };
                                        println!("   obs {:>4}  D = {:.4}  [{}]", i, di, label);
                                    }
                                }
                            }
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n{thin}");
                        println!("  *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::GarchResult(m) => {
                        let model_label = match m.model_type {
                            greeners::GarchModelType::GARCH => "GARCH",
                            greeners::GarchModelType::EGARCH => "EGARCH",
                            greeners::GarchModelType::GJRGARCH => "GJR-GARCH",
                        };
                        println!("\n{thick}");
                        println!(
                            " DIAGNÓSTICOS — {model_label}({}, {})  (n={})",
                            m.p, m.q, m.n_obs
                        );
                        println!("{thick}");

                        let std_res = &m.standardized_residuals;

                        println!(
                            "\n── Autocorrelação nos Resíduos Padronizados (Ljung-Box, lags=10)"
                        );
                        match greeners::Diagnostics::ljung_box(std_res, 10) {
                            Ok(r) => println!(
                                "   Q(10) = {:>9.4}   p = {:.4}  {}",
                                r.q_stat,
                                r.p_value,
                                sig(r.p_value)
                            ),
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n── Efeitos ARCH Residuais (Engle LM, lags=5)");
                        match greeners::Diagnostics::arch_test(std_res, 5) {
                            Ok(r) => println!(
                                "   LM ~ χ²({}) = {:>9.4}   p = {:.4}  {}",
                                r.lags,
                                r.lm_stat,
                                r.lm_pvalue,
                                sig(r.lm_pvalue)
                            ),
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n── Normalidade dos Resíduos Padronizados (Jarque-Bera)");
                        match greeners::Diagnostics::jarque_bera(std_res) {
                            Ok((jb, p)) => {
                                println!("   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}", jb, p, sig(p))
                            }
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n{thin}");
                        println!("  *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::ArimaResult(m) => {
                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — ARIMA");
                        println!("{thick}");

                        let resid = Array1::from_vec(m.residuals().to_vec());

                        println!("\n── Autocorrelação nos Resíduos (Ljung-Box, lags=10)");
                        match greeners::Diagnostics::ljung_box(&resid, 10) {
                            Ok(r) => println!(
                                "   Q(10) = {:>9.4}   p = {:.4}  {}",
                                r.q_stat,
                                r.p_value,
                                sig(r.p_value)
                            ),
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n── Normalidade dos Resíduos (Jarque-Bera)");
                        match greeners::Diagnostics::jarque_bera(&resid) {
                            Ok((jb, p)) => {
                                println!("   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}", jb, p, sig(p))
                            }
                            Err(e) => println!("   erro: {e}"),
                        }

                        println!("\n{thin}");
                        println!("  *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::VarResult(m) => {
                        let k = m.n_vars;
                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — VAR({})  (n={}  k={})", m.lags, m.n_obs, k);
                        println!("{thick}");

                        // ── Critérios de informação
                        println!("\n── Critérios de Informação");
                        println!("   AIC = {:.4}   BIC = {:.4}", m.aic, m.bic);

                        // ── Desvio-padrão residual por equação (diagonal de Σ_u)
                        println!("\n── Desvio-Padrão Residual por Equação");
                        for (i, name) in m.var_names.iter().enumerate() {
                            println!("   {:<22} σ = {:.6}", name, m.sigma_u[[i, i]].sqrt());
                        }

                        // ── Matriz de correlação dos resíduos (Σ_u normalizada)
                        if k > 1 {
                            println!("\n── Correlação Contemporânea dos Resíduos");
                            // header
                            let col_w = m
                                .var_names
                                .iter()
                                .map(|n| n.len())
                                .max()
                                .unwrap_or(8)
                                .max(8)
                                + 2;
                            print!("   {:>col_w$}", "");
                            for name in &m.var_names {
                                print!(" {:>col_w$}", name);
                            }
                            println!();
                            for i in 0..k {
                                print!("   {:<col_w$}", m.var_names[i]);
                                for j in 0..k {
                                    let r = m.sigma_u[[i, j]]
                                        / (m.sigma_u[[i, i]] * m.sigma_u[[j, j]]).sqrt();
                                    if i == j {
                                        print!(" {:>col_w$.4}", 1.0_f64);
                                    } else {
                                        print!(" {:>col_w$.4}", r);
                                    }
                                }
                                println!();
                            }
                        }

                        println!("\n── Nota");
                        println!("   Resíduos não são armazenados em VarResult — para LB/JB por equação,");
                        println!("   extraia a série e rode ljungbox/jb diretamente.");
                        println!("\n{thin}");
                        println!("{thick}\n");
                    }

                    Value::VecmResult(m) => {
                        let k = m.n_vars;
                        let r = m.rank;
                        let n = m.n_obs as f64;
                        let eig = &m.eigenvalues; // ordenados decrescente

                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — VECM  (n={}  k={}  rank={})", m.n_obs, k, r);
                        println!("{thick}");

                        // ── Johansen trace test
                        // λ_trace(r₀) = -n Σ_{i=r₀}^{k-1} ln(1 - λ_i)  H₀: rank ≤ r₀
                        // CVs 5%: Osterwald-Lenum (1992) Tabela 1 — constante restrita
                        let cv_5pct: &[f64] = &[9.24, 19.96, 34.91, 53.12, 76.07, 102.56, 131.70];
                        println!("\n── Teste de Johansen (Trace)");
                        println!("   H₀: rank ≤ r   CVs 5%: Osterwald-Lenum (1992) Tabela 1");
                        println!(
                            "   {:<6} {:>10} {:>12} {:>10} {:>6}",
                            "H₀:r≤", "λ_max", "λ_trace", "CV 5%", ""
                        );
                        println!("   {}", "─".repeat(46));
                        for r0 in 0..k {
                            let lam_max = if r0 < eig.len() {
                                -n * (1.0 - eig[r0]).max(1e-15).ln()
                            } else {
                                0.0
                            };
                            let trace_stat: f64 = (r0..eig.len())
                                .map(|i| -n * (1.0 - eig[i]).max(1e-15).ln())
                                .sum();
                            let cv = cv_5pct.get(k - r0 - 1).copied().unwrap_or(f64::NAN);
                            let reject = if trace_stat > cv { "*" } else { " " };
                            println!(
                                "   {:<6} {:>10.4} {:>12.4} {:>10.2} {:>6}",
                                r0, lam_max, trace_stat, cv, reject
                            );
                        }
                        println!("   (* rejeita H₀ a 5%)");

                        // ── Velocidades de ajuste (alpha): k×rank
                        println!("\n── Velocidades de Ajuste (Alpha)  [sinal negativo = correção ao equilíbrio]");
                        for ec in 0..r {
                            println!("   Vetor EC{}", ec + 1);
                            for eq in 0..k {
                                println!(
                                    "     equação {:>2}   α = {:>9.4}",
                                    eq + 1,
                                    m.alpha[[eq, ec]]
                                );
                            }
                        }

                        // ── Vetores de cointegração (beta): k×rank
                        println!("\n── Vetores de Cointegração (Beta)");
                        for ec in 0..r {
                            println!("   EC{}:", ec + 1);
                            for var in 0..k {
                                println!(
                                    "     var {:>2}   β = {:>9.4}",
                                    var + 1,
                                    m.beta[[var, ec]]
                                );
                            }
                        }

                        println!("\n── Nota");
                        println!("   VecmResult não armazena nomes de variáveis nem resíduos.");
                        println!("   Para nomes, veja a ordem passada em vecm().");
                        println!("\n{thin}");
                        println!("{thick}\n");
                    }

                    Value::IvResult(iv) => {
                        let k = iv.params.len();
                        let n = iv.n_obs;
                        let df = iv.df_resid;
                        let mse = iv.sigma * iv.sigma;
                        let names = iv.variable_names.as_deref().unwrap_or(&[]);

                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — IV/2SLS  (n={}  k={}  df={})", n, k, df);
                        println!("{thick}");

                        println!("\n── Ajuste");
                        println!(
                            "   R²  = {:.4}   σ = {:.6}   MSE = {:.6}",
                            iv.r_squared, iv.sigma, mse
                        );

                        println!("\n── Significância dos Coeficientes");
                        let sig = |p: f64| -> &'static str {
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
                        println!("   {:<22} {:>8} {:>8}", "Variável", "p-value", "");
                        println!("   {}", "─".repeat(40));
                        for i in 0..k {
                            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                            println!(
                                "   {:<22} {:>8.4} {:>4}",
                                name,
                                iv.p_values[i],
                                sig(iv.p_values[i])
                            );
                        }

                        println!("\n── Testes Não Disponíveis");
                        println!("   Resíduos e matriz Z não armazenados em IvResult.");
                        println!("   • Sargan (sobreidentificação): precisa da matriz Z");
                        println!("   • Endogeneidade (Wu-Hausman): compare IV vs OLS manualmente");
                        println!("   • Instrumento fraco: verifique F da 1ª etapa (regra: F > 10)");
                        println!("\n{thin}");
                        println!("   *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::PanelResult(fe) => {
                        let k = fe.params.len();
                        let names = fe.variable_names.as_deref().unwrap_or(&[]);
                        let sig = |p: f64| -> &'static str {
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

                        println!("\n{thick}");
                        println!(
                            " DIAGNÓSTICOS — Efeitos Fixos  (n={}  N={}  T≈{:.1}  k={})",
                            fe.n_obs,
                            fe.n_entities,
                            fe.n_obs as f64 / fe.n_entities.max(1) as f64,
                            k
                        );
                        println!("{thick}");

                        println!("\n── Ajuste (Within)");
                        println!(
                            "   R² within = {:.4}   σ = {:.6}   df = {}",
                            fe.r_squared, fe.sigma, fe.df_resid
                        );

                        println!("\n── Significância dos Coeficientes");
                        println!(
                            "   {:<22} {:>10} {:>8} {:>4}",
                            "Variável", "coef", "p-value", ""
                        );
                        println!("   {}", "─".repeat(48));
                        for i in 0..k {
                            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                            println!(
                                "   {:<22} {:>10.4} {:>8.4} {:>4}",
                                name,
                                fe.params[i],
                                fe.p_values[i],
                                sig(fe.p_values[i])
                            );
                        }

                        println!("\n── Testes Não Disponíveis");
                        println!("   Resíduos não armazenados em PanelResult.");
                        println!("   • Hausman FE vs RE: use hausman(fe_model, re_model)");
                        println!("   • JB / Ljung-Box: rode sobre resíduos extraídos manualmente");
                        println!("\n{thin}");
                        println!("   *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    Value::ReResult(re) => {
                        let k = re.params.len();
                        let sig = |p: f64| -> &'static str {
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

                        // Decomposição de variância
                        let var_e = re.sigma_e * re.sigma_e; // variância dos efeitos individuais
                        let var_u = re.sigma_u * re.sigma_u; // variância idiossincrática
                        let var_tot = var_e + var_u;
                        let icc = if var_tot > 1e-15 {
                            var_e / var_tot
                        } else {
                            0.0
                        };

                        println!("\n{thick}");
                        println!(" DIAGNÓSTICOS — Efeitos Aleatórios  (k={})", k);
                        println!("{thick}");

                        println!("\n── Ajuste");
                        println!("   R² geral = {:.4}", re.r_squared_overall);

                        println!("\n── Decomposição de Variância");
                        println!(
                            "   σ_e  (efeitos individuais) = {:.6}   σ_e² = {:.6}",
                            re.sigma_e, var_e
                        );
                        println!(
                            "   σ_u  (idiossincrático)     = {:.6}   σ_u² = {:.6}",
                            re.sigma_u, var_u
                        );
                        println!("   ICC  = σ_e²/(σ_e²+σ_u²)   = {:.4}   ({:.1}% da variância é entre entidades)",
                            icc, icc * 100.0);
                        println!(
                            "   θ    (peso GLS)            = {:.4}   (0→OLS  1→FE)",
                            re.theta
                        );

                        println!("\n── Significância dos Coeficientes");
                        println!(
                            "   {:<22} {:>10} {:>8} {:>4}",
                            "Variável", "coef", "p-value", ""
                        );
                        println!("   {}", "─".repeat(48));
                        for i in 0..k {
                            let name = re
                                .variable_names
                                .as_ref()
                                .and_then(|v| v.get(i))
                                .map(|s| s.as_str())
                                .unwrap_or("const");
                            println!(
                                "   {:<22} {:>10.4} {:>8.4} {:>4}",
                                name,
                                re.params[i],
                                re.p_values[i],
                                sig(re.p_values[i])
                            );
                        }

                        println!("\n── Testes Não Disponíveis");
                        println!("   • Hausman FE vs RE: use hausman(fe_model, re_model)");
                        println!("   • BP LM test (H₀: sem efeitos individuais): σ_e²/σ_u² acima sugere efeitos");
                        println!("\n{thin}");
                        println!("   *** p<0.01  ** p<0.05  * p<0.10");
                        println!("{thick}\n");
                    }

                    _ => {
                        return Err(HayashiError::Type(
                            "diagnostics() suporta OLS, GARCH, ARIMA, VAR, VECM, IV, FE e RE"
                                .into(),
                        ))
                    }
                }

                Ok(Value::Nil)
            }

            // ── VARMA(p,q) ────────────────────────────────────────────────────
            // varma(df, y1, y2, ..., p=1, q=1)
            "varma" | "varmax" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "varma(df, y1, y2, ..., p=1, q=1)".into(),
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
                let var_names = self.resolve_var_list(&args[1..], &df)?;
                let p = match opt_map.get("p") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let q = match opt_map.get("q") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }
                let result = greeners::VARMA::fit(&data, p, q)
                    .map_err(|e| self.rt_err(format!("VARMA: {e}")))?;
                Ok(Value::VarmaResult(Rc::new(result)))
            }

            // ── Decomposição sazonal ──────────────────────────────────────────
            // decompose(df, var, period=12, model=additive)
            "decompose" | "seasonal_decompose" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "decompose(df, var, period=12, model=additive)".into(),
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
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let period = match opt_map.get("period") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };
                let model_str = match opt_map.get("model") {
                    Some(Value::Str(s)) => s.as_str(),
                    _ => "additive",
                };
                let result =
                    greeners::Decomposition::seasonal_decompose(&series, period, model_str)
                        .map_err(|e| self.rt_err(format!("decompose: {e}")))?;
                Ok(Value::DecompResult(Rc::new(result)))
            }

            // stl(df, var, period=12, sw=7, tw=0)
            "stl" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "stl(df, var, period=12, sw=7, tw=0)".into(),
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
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let period = match opt_map.get("period") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };
                let sw = match opt_map.get("sw") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 7,
                };
                let tw = match opt_map.get("tw") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 0,
                };
                let result = greeners::Decomposition::stl(&series, period, sw, tw)
                    .map_err(|e| self.rt_err(format!("stl: {e}")))?;
                Ok(Value::DecompResult(Rc::new(result)))
            }

            // ── MSTL ─────────────────────────────────────────────────────────
            // mstl(df, var, periods=[7, 365])
            "mstl" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "mstl(df, var, periods=[7,365])".into(),
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
                let series = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let periods: Vec<usize> = match opt_map.get("periods") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Int(i) => Ok(*i as usize),
                            Value::Float(f) => Ok(*f as usize),
                            _ => Err(HayashiError::Type(
                                "periods= must be a list de inteiros".into(),
                            )),
                        })
                        .collect::<Result<_>>()?,
                    Some(Value::Int(i)) => vec![*i as usize],
                    Some(Value::Float(f)) => vec![*f as usize],
                    _ => vec![7, 365],
                };
                let result = greeners::MSTL::fit(&series, &periods)
                    .map_err(|e| self.rt_err(format!("mstl: {e}")))?;
                Ok(Value::MstlResult(Rc::new(result)))
            }

            // ── Testes de proporção ───────────────────────────────────────────
            // proptest(count, n, mu=0.5)
            "proptest" | "prtest" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("proptest(count, n, mu=0.5)".into()));
                }
                let count = match self.eval_expr(&args[0])? {
                    Value::Int(v) => v as usize,
                    Value::Float(v) => v as usize,
                    _ => return Err(HayashiError::Type("count must be integer".into())),
                };
                let n = match self.eval_expr(&args[1])? {
                    Value::Int(v) => v as usize,
                    Value::Float(v) => v as usize,
                    _ => return Err(HayashiError::Type("n must be integer".into())),
                };
                let mu = match opt_map.get("mu") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.5,
                };
                let (z, p) = greeners::ProportionTests::proportions_ztest_1samp(count, n, mu)
                    .map_err(|e| self.rt_err(format!("proptest: {e}")))?;
                let p_hat = count as f64 / n as f64;
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
                println!("\nTeste de Proporção (1 amostra)");
                println!("{sep}");
                println!("  H₀: p = {mu:.4}");
                println!("  p̂ = {p_hat:.4}  (count={count}, n={n})");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "z", z, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // proptest2(count1, n1, count2, n2)
            "proptest2" | "prtest2" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime(
                        "proptest2(count1, n1, count2, n2)".into(),
                    ));
                }
                let to_usize = |v: Value| -> Result<usize> {
                    match v {
                        Value::Int(i) => Ok(i as usize),
                        Value::Float(f) => Ok(f as usize),
                        _ => Err(HayashiError::Type(
                            "argumentos de proptest2() devem ser inteiros".into(),
                        )),
                    }
                };
                let c1 = to_usize(self.eval_expr(&args[0])?)?;
                let n1 = to_usize(self.eval_expr(&args[1])?)?;
                let c2 = to_usize(self.eval_expr(&args[2])?)?;
                let n2 = to_usize(self.eval_expr(&args[3])?)?;
                let (z, p) = greeners::ProportionTests::proportions_ztest_2samp(c1, n1, c2, n2)
                    .map_err(|e| self.rt_err(format!("proptest2: {e}")))?;
                let p1 = c1 as f64 / n1 as f64;
                let p2 = c2 as f64 / n2 as f64;
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
                println!("\nTeste de Proporção (2 amostras)");
                println!("{sep}");
                println!("  H₀: p₁ = p₂");
                println!("  p̂₁ = {p1:.4}  (count={c1}, n={n1})");
                println!("  p̂₂ = {p2:.4}  (count={c2}, n={n2})");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10.4} {:>10.4} {:>4}",
                    "z (bicaudal)",
                    z,
                    p,
                    sig(p)
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // propci(count, n, alpha=0.05)
            "propci" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("propci(count, n, alpha=0.05)".into()));
                }
                let count = match self.eval_expr(&args[0])? {
                    Value::Int(v) => v as usize,
                    Value::Float(v) => v as usize,
                    _ => return Err(HayashiError::Type("count must be integer".into())),
                };
                let n = match self.eval_expr(&args[1])? {
                    Value::Int(v) => v as usize,
                    Value::Float(v) => v as usize,
                    _ => return Err(HayashiError::Type("n must be integer".into())),
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.05,
                };
                let (lo, hi) = greeners::ProportionTests::proportion_confint(count, n, alpha)
                    .map_err(|e| self.rt_err(format!("propci: {e}")))?;
                let p_hat = count as f64 / n as f64;
                let pct = (1.0 - alpha) * 100.0;
                let sep = "─".repeat(56);
                println!("\nIC de Proporção — Wilson Score ({pct:.0}%)");
                println!("{sep}");
                println!("  p̂ = {p_hat:.4}  (count={count}, n={n})");
                println!("  IC [{pct:.0}%]: [{lo:.4}, {hi:.4}]");
                println!("{sep}");
                println!();
                Ok(Value::Nil)
            }

            // chisq2x2(a, b, c, d)  — tabela 2×2
            "chisq2x2" | "chi2_2x2" => {
                if args.len() < 4 {
                    return Err(HayashiError::Runtime("chisq2x2(a, b, c, d)".into()));
                }
                let to_usize = |v: Value| -> Result<usize> {
                    match v {
                        Value::Int(i) => Ok(i as usize),
                        Value::Float(f) => Ok(f as usize),
                        _ => Err(HayashiError::Type(
                            "células da tabela devem ser inteiros".into(),
                        )),
                    }
                };
                let a = to_usize(self.eval_expr(&args[0])?)?;
                let b = to_usize(self.eval_expr(&args[1])?)?;
                let c = to_usize(self.eval_expr(&args[2])?)?;
                let d = to_usize(self.eval_expr(&args[3])?)?;
                let table = [[a, b], [c, d]];
                let (chi2, p) = greeners::ProportionTests::chi2_contingency(&table)
                    .map_err(|e| self.rt_err(format!("chisq2x2: {e}")))?;
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
                println!("\nTeste Qui-Quadrado — Tabela 2×2");
                println!("{sep}");
                println!("       | Col 0 | Col 1 |  Total");
                println!("  Row 0|  {:>5} |  {:>5} |  {:>5}", a, b, a + b);
                println!("  Row 1|  {:>5} |  {:>5} |  {:>5}", c, d, c + d);
                println!(
                    "  Total|  {:>5} |  {:>5} |  {:>5}",
                    a + c,
                    b + d,
                    a + b + c + d
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "χ²(1)", chi2, p, sig(p));
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // ── Múltiplos testes ──────────────────────────────────────────────
            // multipletests(pvalues, method=bonferroni, alpha=0.05)
            "multipletests" | "multtest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "multipletests(pvalues, method=bonferroni, alpha=0.05)".into(),
                    ));
                }
                let pvals_val = self.eval_expr(&args[0])?;
                let pvals: Vec<f64> = match pvals_val {
                    Value::List(lst) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Float(f) => Ok(*f),
                            Value::Int(i) => Ok(*i as f64),
                            _ => Err(HayashiError::Type(
                                "pvalues must be a list de floats".into(),
                            )),
                        })
                        .collect::<Result<_>>()?,
                    _ => {
                        return Err(HayashiError::Type(
                            "first argument must be lista de p-values".into(),
                        ))
                    }
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.05,
                };
                let method = match opt_map.get("method") {
                    Some(Value::Str(s)) => match s.to_lowercase().as_str() {
                        "bonferroni" => greeners::MultiTestMethod::Bonferroni,
                        "sidak" => greeners::MultiTestMethod::Sidak,
                        "holm" | "holm_bonferroni" | "holmbonferroni" => {
                            greeners::MultiTestMethod::HolmBonferroni
                        }
                        "bh" | "benjamini_hochberg" | "fdr_bh" => {
                            greeners::MultiTestMethod::BenjaminiHochberg
                        }
                        "by" | "benjamini_yekutieli" | "fdr_by" => {
                            greeners::MultiTestMethod::BenjaminiYekutieli
                        }
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "método unknown: '{other}' — use bonferroni, sidak, holm, bh, by"
                            )))
                        }
                    },
                    _ => greeners::MultiTestMethod::Bonferroni,
                };
                let method_name = format!("{:?}", method);
                let (rejects, pvals_adj) =
                    greeners::MultipleTests::multipletests(&pvals, alpha, method)
                        .map_err(|e| self.rt_err(format!("multipletests: {e}")))?;
                let sep = "─".repeat(64);
                println!("\nMúltiplos Testes — {method_name}  (α={alpha})");
                println!("{sep}");
                println!(
                    "{:>5}  {:>12}  {:>12}  {:>8}",
                    "#", "p original", "p ajustado", "Rejeitar?"
                );
                println!("{sep}");
                for (i, ((p_orig, p_adj), rej)) in pvals
                    .iter()
                    .zip(pvals_adj.iter())
                    .zip(rejects.iter())
                    .enumerate()
                {
                    let mark = if *rej { "  SIM ***" } else { "  não" };
                    println!("{:>5}  {:>12.6}  {:>12.6}  {}", i + 1, p_orig, p_adj, mark);
                }
                println!("{sep}");
                println!();
                Ok(Value::Nil)
            }

            // ── UCM — Unobserved Components Model ─────────────────────────────
            // ucm(df, var, level=local_linear, seasonal=stochastic, period=12)
            "ucm" | "uc" | "structural_ts" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ucm(df, var, level=local_linear, seasonal=stochastic, period=12)".into(),
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
                let y = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);

                let level = match opt_map.get("level") {
                    Some(Value::Str(s)) => match s.to_lowercase().as_str() {
                        "local_level" | "ll"            => greeners::UCLevel::LocalLevel,
                        "local_linear" | "local_linear_trend" | "llt" => greeners::UCLevel::LocalLinearTrend,
                        "smooth_trend" | "st"           => greeners::UCLevel::SmoothTrend,
                        "random_walk" | "rw"            => greeners::UCLevel::RandomWalk,
                        other => return Err(HayashiError::Runtime(format!(
                            "ucm: level='{other}' unknown — use: local_level, local_linear, smooth_trend, random_walk"
                        ))),
                    },
                    _ => greeners::UCLevel::LocalLinearTrend,
                };

                let period = match opt_map.get("period") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 12,
                };

                let seasonal = match opt_map.get("seasonal") {
                    Some(Value::Str(s)) => match s.to_lowercase().as_str() {
                        "none" => greeners::UCSeasonal::None,
                        "deterministic" => greeners::UCSeasonal::Deterministic(period),
                        "stochastic" => greeners::UCSeasonal::Stochastic(period),
                        other => {
                            return Err(HayashiError::Runtime(format!(
                            "ucm: seasonal='{other}' unknown — use: none, deterministic, stochastic"
                        )))
                        }
                    },
                    _ => greeners::UCSeasonal::None,
                };

                let result = greeners::UnobservedComponents::fit(&y, level, seasonal)
                    .map_err(|e| self.rt_err(format!("ucm: {e}")))?;
                Ok(Value::UCResult(Rc::new(result)))
            }

            // ── GAM — Generalized Additive Model (P-splines) ─────────────────
            // gam(y ~ x2, df, smooth="x1", spline_df=10, alpha=0.1, family=gaussian, link=log)
            "gam" | "gamfit" => {
                let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (y_vec, x_linear) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let linear_names = df
                    .formula_var_names(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n = y_vec.len();

                // Parse smooth= option
                let smooth_names: Vec<String> = match opt_map.get("smooth") {
                    Some(Value::Str(s)) => vec![s.clone()],
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type(
                                "smooth= must be string ou lista de strings".into(),
                            )),
                        })
                        .collect::<Result<_>>()?,
                    None => vec![],
                    _ => {
                        return Err(HayashiError::Type(
                            "smooth= must be string ou lista de strings".into(),
                        ))
                    }
                };

                if smooth_names.is_empty() && x_linear.ncols() == 0 {
                    return Err(HayashiError::Runtime(
                        "gam: especifique termos lineares (fórmula) e/ou smooth=".into(),
                    ));
                }

                let spline_df = match opt_map.get("spline_df") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };
                let degree = match opt_map.get("degree") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 3,
                };
                let alpha_pen = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.1,
                };

                // Build smooth basis matrix (concatenate across all smooth vars)
                let q_per = spline_df;
                let q_total = q_per * smooth_names.len().max(1);
                let mut x_smooth = ndarray::Array2::<f64>::zeros((n, q_total));
                for (k, sname) in smooth_names.iter().enumerate() {
                    let col = ndarray::Array1::from(Self::get_col_f64(&df, sname)?);
                    let basis = greeners::BSplineBasis::generate(&col, q_per, degree)
                        .map_err(|e| self.rt_err(format!("gam spline ({sname}): {e}")))?;
                    for i in 0..n {
                        for j in 0..q_per {
                            x_smooth[[i, k * q_per + j]] = basis[[i, j]];
                        }
                    }
                }
                // If no smooth vars, x_smooth must still be n×1 (placeholder)
                let x_smooth_ref = if smooth_names.is_empty() {
                    ndarray::Array2::<f64>::zeros((n, 1))
                } else {
                    x_smooth
                };

                let alpha_pen_used = if smooth_names.is_empty() {
                    0.0
                } else {
                    alpha_pen
                };

                // Parse family/link (same as GLM)
                let alpha_val = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.0,
                };
                let power_val = match opt_map.get("power") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 1.5,
                };
                let family = match opt_map.get("family") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "gaussian" | "normal"   => greeners::Family::Gaussian,
                        "binomial" | "logistic" => greeners::Family::Binomial,
                        "poisson"               => greeners::Family::Poisson,
                        "gamma"                 => greeners::Family::Gamma,
                        "inverse_gaussian"      => greeners::Family::InverseGaussian,
                        "negbin"                => greeners::Family::NegativeBinomial(alpha_val),
                        "tweedie"               => greeners::Family::Tweedie(power_val),
                        other => return Err(HayashiError::Runtime(format!(
                            "gam: family='{other}' unknown — use: gaussian, binomial, poisson, gamma, negbin"
                        ))),
                    },
                    _ => greeners::Family::Gaussian,
                };
                let link = match opt_map.get("link") {
                    Some(Value::Str(s)) => match s.as_str() {
                        "identity"  => greeners::Link::Identity,
                        "log"       => greeners::Link::Log,
                        "logit"     => greeners::Link::Logit,
                        "probit"    => greeners::Link::Probit,
                        "inverse"   => greeners::Link::InversePower,
                        "cloglog"   => greeners::Link::CLogLog,
                        other => return Err(HayashiError::Runtime(format!(
                            "gam: link='{other}' unknown — use: identity, log, logit, probit, inverse, cloglog"
                        ))),
                    },
                    _ => greeners::Link::Identity,
                };

                let result = greeners::GLMGam::fit_with_names(
                    &y_vec,
                    &x_linear,
                    &x_smooth_ref,
                    &family,
                    &link,
                    alpha_pen_used,
                    Some(linear_names),
                )
                .map_err(|e| self.rt_err(format!("gam: {e}")))?;
                Ok(Value::GamResult(Rc::new(result)))
            }

            // ── MICE — Multiple Imputation by Chained Equations ───────────────
            // mice(df, vars=["x1","x2"], m=5, iter=10)
            "mice" | "mi" | "multiple_imputation" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "mice(df, vars=[\"x1\",\"x2\"], m=5, iter=10)".into(),
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
                let var_names: Vec<String> = match opt_map.get("vars") {
                    Some(Value::List(lst)) => lst
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => Ok(s.clone()),
                            _ => Err(HayashiError::Type("vars= must be a list de strings".into())),
                        })
                        .collect::<Result<_>>()?,
                    Some(Value::Str(s)) => vec![s.clone()],
                    None => {
                        if args.len() > 1 {
                            self.resolve_var_list(&args[1..], &df)?
                        } else {
                            return Err(HayashiError::Runtime(
                                "mice: especifique vars=[\"x1\",\"x2\",...] ou liste variáveis após df".into()
                            ));
                        }
                    }
                    _ => return Err(HayashiError::Type("vars= must be a list de strings".into())),
                };
                let m = match opt_map.get("m") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 5,
                };
                let iter = match opt_map.get("iter") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };

                let mut data: std::collections::HashMap<String, ndarray::Array1<f64>> =
                    std::collections::HashMap::new();
                for vname in &var_names {
                    data.insert(
                        vname.clone(),
                        ndarray::Array1::from(Self::get_col_f64(&df, vname)?),
                    );
                }

                let result = greeners::MICE::impute(&data, m, iter)
                    .map_err(|e| self.rt_err(format!("mice: {e}")))?;
                println!("{result}");
                Ok(Value::MiceResult(Rc::new(result)))
            }

            // ── Markov Autoregression (Hamilton 1989 full MS-AR) ──────────────
            // msauto(df, var, k=2, p=1)
            "msauto" | "markov_ar" | "ms_ar" | "hamilton" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("msauto(df, var, k=2, p=1)".into()));
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
                let y = ndarray::Array1::from(Self::get_col_f64(&df, &var_name)?);
                let k = match opt_map.get("k") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let p = match opt_map.get("p") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let result = greeners::MarkovAutoregression::fit(&y, k, p)
                    .map_err(|e| self.rt_err(format!("msauto: {e}")))?;
                Ok(Value::MSARResult(Rc::new(result)))
            }

            // ── SVAR — Structural VAR ─────────────────────────────────────────
            // svar(df, y1, y2, ..., lags=1, id=cholesky)
            // id=cholesky  : identificação recursiva (Cholesky)
            // id=longrun   : restrições de longo prazo (Blanchard-Quah)
            "svar" | "svec" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "svar(df, y1, y2, ..., lags=1, id=cholesky)".into(),
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
                let var_names = self.resolve_var_list(&args[1..], &df)?;
                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }
                let identification = match opt_map.get("id") {
                    Some(Value::Str(s)) => match s.to_lowercase().as_str() {
                        "cholesky" | "recursive" => greeners::SVarIdentification::Cholesky,
                        "longrun" | "long_run" | "bq" | "blanchard_quah" => {
                            let mask = ndarray::Array2::from_elem((k, k), f64::NAN);
                            greeners::SVarIdentification::LongRun(mask)
                        }
                        other => {
                            return Err(HayashiError::Runtime(format!(
                                "svar: id='{other}' unknown — use: cholesky, longrun"
                            )))
                        }
                    },
                    _ => greeners::SVarIdentification::Cholesky,
                };
                let result = greeners::SVAR::fit(&data, lags, identification)
                    .map_err(|e| self.rt_err(format!("svar: {e}")))?;
                Ok(Value::SVarResult(Rc::new(result)))
            }

            // sirf(model, steps=10) — Structural IRF
            "sirf" | "svar_irf" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("sirf(model, steps=10)".into()));
                }
                let model = match self.eval_expr(&args[0])? {
                    Value::SVarResult(m) => m,
                    _ => return Err(HayashiError::Type("sirf() requires an SVAR model".into())),
                };
                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };
                let tensor = model
                    .structural_irf(steps)
                    .map_err(|e| self.rt_err(format!("sirf: {e}")))?;
                let k = model.var_result.n_vars;
                let names = &model.var_result.var_names;
                let sep = "─".repeat(14 + k * 12);
                println!(
                    "\nSVAR Structural IRF — {} — id: {} — {} passos",
                    format!("VAR({})", model.var_result.lags),
                    model.identification,
                    steps
                );
                for j in 0..k {
                    println!("\n  Impulso: {}", names[j]);
                    println!("  {sep}");
                    let header: String = names
                        .iter()
                        .map(|n| format!("{:>12}", n))
                        .collect::<Vec<_>>()
                        .join("");
                    println!("  {:>6}  {header}", "h");
                    println!("  {sep}");
                    for h in 0..steps {
                        let row: String = (0..k)
                            .map(|i| format!("{:>12.4}", tensor[[h, i, j]]))
                            .collect::<Vec<_>>()
                            .join("");
                        println!("  {:>6}  {row}", h);
                    }
                }
                println!();
                Ok(Value::Nil)
            }

            // sfevd(model, steps=10) — Structural FEVD
            "sfevd" | "svar_fevd" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("sfevd(model, steps=10)".into()));
                }
                let model = match self.eval_expr(&args[0])? {
                    Value::SVarResult(m) => m,
                    _ => return Err(HayashiError::Type("sfevd() requires an SVAR model".into())),
                };
                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 10,
                };
                let tensor = model
                    .structural_fevd(steps)
                    .map_err(|e| self.rt_err(format!("sfevd: {e}")))?;
                let k = model.var_result.n_vars;
                let names = &model.var_result.var_names;
                let sep = "─".repeat(14 + k * 12);
                println!(
                    "\nSVAR Structural FEVD — {} — id: {}",
                    format!("VAR({})", model.var_result.lags),
                    model.identification
                );
                for i in 0..k {
                    println!("\n  Resposta: {}", names[i]);
                    println!("  {sep}");
                    let header: String = names
                        .iter()
                        .map(|n| format!("{:>12}", n))
                        .collect::<Vec<_>>()
                        .join("");
                    println!("  {:>6}  {header}", "h");
                    println!("  {sep}");
                    for h in 0..steps {
                        let row: String = (0..k)
                            .map(|j| format!("{:>12.4}", tensor[[h, i, j]]))
                            .collect::<Vec<_>>()
                            .join("");
                        println!("  {:>6}  {row}", h);
                    }
                }
                println!();
                Ok(Value::Nil)
            }

            // ── 3SLS — Three Stage Least Squares ──────────────────────────────
            // threesl(df, y1~x1+z1, y2~x1+z2, instruments=["z1","z2"])
            "threesl" | "three_sls" | "3sls" | "reg3" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "threesl(df, y1~x1+z1, y2~x2+z2, instruments=[\"z1\",\"z2\"])".into(),
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

                // Parse instruments= option
                let instr_names: Vec<String> = match opt_map.get("instruments") {
                    Some(Value::List(lst)) => lst.iter().map(|v| match v {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(HayashiError::Type("instruments= must be a list de strings".into())),
                    }).collect::<Result<_>>()?,
                    Some(Value::Str(s)) => vec![s.clone()],
                    None => return Err(HayashiError::Runtime(
                        "threesl requer instruments=[\"z1\",\"z2\",...] — lista de variáveis exógenas".into()
                    )),
                    _ => return Err(HayashiError::Type("instruments= must be a list de strings".into())),
                };

                // Build global instrument matrix Z (n × q)
                let n = df.n_rows();
                let mut z_instr = ndarray::Array2::<f64>::zeros((n, instr_names.len()));
                for (j, zname) in instr_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, zname)?;
                    for (i, &v) in col.iter().enumerate() {
                        z_instr[[i, j]] = v;
                    }
                }

                // Build equations from formulas
                let mut equations: Vec<greeners::Equation> = Vec::new();
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
                    equations.push(greeners::Equation {
                        y,
                        x,
                        name: formula_ast.lhs.clone(),
                    });
                }
                let result = greeners::ThreeSLS::fit(&equations, &z_instr)
                    .map_err(|e| self.rt_err(format!("threesl: {e}")))?;
                Ok(Value::ThreeSLSResult(ThreeSLSModel {
                    result: Rc::new(result),
                    eq_var_names,
                }))
            }

            // ── DFM — Dynamic Factor Model ────────────────────────────────────
            // dfm(df, y1, y2, ..., factors=2, order=1)
            "dfm" | "dynamic_factor" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "dfm(df, y1, y2, ..., factors=2, order=1)".into(),
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
                let var_names: Vec<String> = args[1..]
                    .iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type(
                            "variáveis de dfm() devem ser identificadores".into(),
                        )),
                    })
                    .collect::<Result<_>>()?;
                let k_factors = match opt_map.get("factors") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 2,
                };
                let factor_order = match opt_map.get("order") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };
                let n = df.n_rows();
                let k = var_names.len();
                let mut data = ndarray::Array2::<f64>::zeros((n, k));
                for (j, vname) in var_names.iter().enumerate() {
                    let col = Self::get_col_f64(&df, vname)?;
                    for (i, &v) in col.iter().enumerate() {
                        data[[i, j]] = v;
                    }
                }
                let result = greeners::DynamicFactor::fit(&data, k_factors, factor_order)
                    .map_err(|e| self.rt_err(format!("dfm: {e}")))?;
                Ok(Value::DFMResult(DFMModel {
                    result: Rc::new(result),
                    var_names,
                }))
            }

            // ── Diagnósticos menores de normalidade / forma funcional ─────────

            // adtest(df, var) — Anderson-Darling test para normalidade
            "adtest" | "anderson_darling" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("adtest(df, var)".into()));
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
                let data = Self::get_col_f64(&df, &var_name)?;
                let r = greeners::Diagnostics::anderson_darling(&ndarray::Array1::from(data))
                    .map_err(|e| self.rt_err(format!("adtest: {e}")))?;
                let sep = "─".repeat(56);
                println!("\nAnderson-Darling Test (normalidade)");
                println!("{sep}");
                println!("  H₀: dados provêm de distribuição normal");
                println!("  A² (ajustado) = {:.4}  (n={})", r.statistic, r.n_obs);
                println!("{sep}");
                println!("{:<12} {:>10}", "α", "A²*_crítico");
                println!("{sep}");
                for (&sig, &cv) in r.significance_levels.iter().zip(r.critical_values.iter()) {
                    let mark = if r.statistic > cv { " ← REJEITA" } else { "" };
                    println!("{:<12.3} {:>10.3}{mark}", sig, cv);
                }
                println!("{sep}");
                println!("(Rejeita H₀ quando A²* > valor crítico)");
                println!();
                Ok(Value::Nil)
            }

            // lilliefors(df, var) — KS com parâmetros estimados
            "lilliefors" | "lillie" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("lilliefors(df, var)".into()));
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
                let data = Self::get_col_f64(&df, &var_name)?;
                let (stat, p) = greeners::Diagnostics::lilliefors(&ndarray::Array1::from(data))
                    .map_err(|e| self.rt_err(format!("lilliefors: {e}")))?;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let sep = "─".repeat(56);
                println!("\nLilliefors Test (normalidade — KS com parâmetros estimados)");
                println!("{sep}");
                println!("  H₀: dados provêm de distribuição normal");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!(
                    "{:<26} {:>10.4} {:>10.4} {:>4}",
                    "KS (Lilliefors)", stat, p, sig
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // omnibus(model) — D'Agostino-Pearson nos resíduos
            "omnibus" | "dagostino" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("omnibus(model)".into()));
                }
                let resids = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m.residuals.to_vec(),
                    _ => {
                        return Err(HayashiError::Type(
                            "omnibus() only supports OLS models".into(),
                        ))
                    }
                };
                let (k2, p) = greeners::Diagnostics::omnibus(&ndarray::Array1::from(resids))
                    .map_err(|e| self.rt_err(format!("omnibus: {e}")))?;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let sep = "─".repeat(56);
                println!("\nD'Agostino-Pearson Omnibus Test (normalidade dos resíduos)");
                println!("{sep}");
                println!("  H₀: resíduos são normalmente distribuídos");
                println!("  (combina assimetria e curtose via K² ~ χ²(2))");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "K² ~ χ²(2)", k2, p, sig);
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // swilk(df, var) — Shapiro-Wilk test for normality
            "swilk" | "shapiro_wilk" | "shapiro" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("swilk(df, var)".into()));
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
                let data = Self::get_col_f64(&df, &var_name)?;
                let res = greeners::Diagnostics::shapiro_wilk(&ndarray::Array1::from(data))
                    .map_err(|e| self.rt_err(format!("swilk: {e}")))?;
                let sig = if res.p_value < 0.01 {
                    "***"
                } else if res.p_value < 0.05 {
                    "**"
                } else if res.p_value < 0.10 {
                    "*"
                } else {
                    ""
                };
                let sep = "─".repeat(56);
                println!("\nShapiro-Wilk Test for Normality");
                println!("{sep}");
                println!("  H₀: {var_name} is normally distributed");
                println!("  n = {}", res.n_obs);
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Test", "W", "p-value", "");
                println!("{sep}");
                println!(
                    "{:<26} {:>10.6} {:>10.4} {:>4}",
                    "Shapiro-Wilk", res.w, res.p_value, sig
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // sfrancia(df, var) — Shapiro-Francia test for normality
            "sfrancia" | "shapiro_francia" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("sfrancia(df, var)".into()));
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
                let data = Self::get_col_f64(&df, &var_name)?;
                let res = greeners::Diagnostics::shapiro_francia(&ndarray::Array1::from(data))
                    .map_err(|e| self.rt_err(format!("sfrancia: {e}")))?;
                let sig = if res.p_value < 0.01 {
                    "***"
                } else if res.p_value < 0.05 {
                    "**"
                } else if res.p_value < 0.10 {
                    "*"
                } else {
                    ""
                };
                let sep = "─".repeat(56);
                println!("\nShapiro-Francia Test for Normality");
                println!("{sep}");
                println!("  H₀: {var_name} is normally distributed");
                println!("  n = {}", res.n_obs);
                println!("{sep}");
                println!("{:<26} {:>10} {:>10} {:>4}", "Test", "W'", "p-value", "");
                println!("{sep}");
                println!(
                    "{:<26} {:>10.6} {:>10.4} {:>4}",
                    "Shapiro-Francia", res.w_prime, res.p_value, sig
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            // sktest(df, var) — Skewness/Kurtosis test for normality
            "sktest" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("sktest(df, var)".into()));
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
                let data = Self::get_col_f64(&df, &var_name)?;
                let slice = data.as_slice().unwrap();
                let skew = greeners::MomentHelpers::skewness(slice);
                let kurt = greeners::MomentHelpers::kurtosis(slice);
                let (jb, jb_p) = greeners::MomentHelpers::jarque_bera(slice);
                let (k2, k2_p) = greeners::MomentHelpers::dagostino(slice);
                let n = data.len();
                let sep = "─".repeat(66);
                println!("\nSkewness/Kurtosis Tests for Normality");
                println!("{sep}");
                println!("  Variable: {var_name}    n = {n}");
                println!("{sep}");
                println!(
                    "{:<16} {:>10} {:>10} {:>12} {:>8}",
                    "", "Statistic", "Value", "chi2(2)", "p-value"
                );
                println!("{sep}");
                println!("{:<16} {:>10} {:>10.4}", "Skewness", "", skew);
                println!("{:<16} {:>10} {:>10.4}", "Kurtosis", "", kurt + 3.0);
                let jb_sig = if jb_p < 0.01 {
                    "***"
                } else if jb_p < 0.05 {
                    "**"
                } else if jb_p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let k2_sig = if k2_p < 0.01 {
                    "***"
                } else if k2_p < 0.05 {
                    "**"
                } else if k2_p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!("{sep}");
                println!(
                    "{:<16} {:>10} {:>10} {:>12.4} {:>8.4} {jb_sig}",
                    "Jarque-Bera", "JB", "", jb, jb_p
                );
                println!(
                    "{:<16} {:>10} {:>10} {:>12.4} {:>8.4} {k2_sig}",
                    "D'Agostino", "K²", "", k2, k2_p
                );
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!("(Kurtosis shown as excess+3, i.e. Normal=3)");
                println!();
                Ok(Value::Nil)
            }

            // harveycollier(model) — teste de linearidade via resíduos recursivos
            "harveycollier" | "harvey_collier" | "hctest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("harveycollier(model)".into()));
                }
                let ols = match self.eval_expr(&args[0])? {
                    Value::OlsResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "harveycollier() only supports OLS models".into(),
                        ))
                    }
                };
                // reconstruir y = ŷ + resíduos (OlsModel não armazena y diretamente)
                let y_hat = ols.x.dot(&ols.result.params);
                let y_obs = y_hat + &ols.residuals;
                let (t, p) = greeners::Diagnostics::harvey_collier(&y_obs, &ols.x)
                    .map_err(|e| self.rt_err(format!("harveycollier: {e}")))?;
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                let sep = "─".repeat(56);
                println!("\nHarvey-Collier Test (linearidade da especificação)");
                println!("{sep}");
                println!("  H₀: especificação funcional está correta (linear)");
                println!("  (testa se média dos resíduos recursivos é zero)");
                println!("{sep}");
                println!(
                    "{:<26} {:>10} {:>10} {:>4}",
                    "Teste", "Estatística", "p-value", ""
                );
                println!("{sep}");
                println!("{:<26} {:>10.4} {:>10.4} {:>4}", "t (HC)", t, p, sig);
                println!("{sep}");
                println!("(*** p<0.01  ** p<0.05  * p<0.10)");
                println!();
                Ok(Value::Nil)
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
