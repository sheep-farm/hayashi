use super::helpers::*;
use super::*;

/// GARCH/EGARCH/GJR-GARCH, VARMA, seasonal decomposition, MSTL, proportion
/// tests, multiple tests, UCM, GAM, MICE, Markov Switching, SVAR, 3SLS,
/// DFM and minor normality / functional-form diagnostics.
/// Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_estimators_timeseries(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            "garch" | "egarch" | "gjrgarch" => self.garch(func, args, opts, opt_map),
            "ljungbox" | "ljung_box" | "portmanteau" => self.ljungbox(func, args, opts, opt_map),
            "leverage" => self.leverage(func, args, opts, opt_map),
            "cooks" => self.cooks(func, args, opts, opt_map),
            "vif" => self.vif(func, args, opts, opt_map),
            "condnum" => self.condnum(func, args, opts, opt_map),
            "durbinwatson" | "dw" => self.durbinwatson(func, args, opts, opt_map),
            "white" => self.white(func, args, opts, opt_map),
            "reset" => self.reset(func, args, opts, opt_map),
            "jb" => self.jb(func, args, opts, opt_map),
            "bgodfrey" => self.bgodfrey(func, args, opts, opt_map),
            "bgtest" | "bg" | "breusch_godfrey" => {
                return self.eval_call("bgodfrey", args, opts).map(Some);
            }
            "archtest" | "arch_test" | "engle_arch" => self.archtest(func, args, opts, opt_map),
            "acf" => self.acf(func, args, opts, opt_map),
            "pacf" => self.pacf(func, args, opts, opt_map),
            "cusumtest" | "cusum_test" => self.cusumtest(func, args, opts, opt_map),
            "forecast_vol" => self.forecast_vol(func, args, opts, opt_map),
            "diagnostics" => self.diagnostics(func, args, opts, opt_map),
            "varma" | "varmax" => self.varma(func, args, opts, opt_map),
            "decompose" | "seasonal_decompose" => self.decompose(func, args, opts, opt_map),
            "stl" => self.stl(func, args, opts, opt_map),
            "mstl" => self.mstl(func, args, opts, opt_map),
            "proptest" | "prtest" => self.proptest(func, args, opts, opt_map),
            "proptest2" | "prtest2" => self.proptest2(func, args, opts, opt_map),
            "propci" => self.propci(func, args, opts, opt_map),
            "chisq2x2" | "chi2_2x2" => self.chisq2x2(func, args, opts, opt_map),
            "multipletests" | "multtest" => self.multipletests(func, args, opts, opt_map),
            "ucm" | "uc" | "structural_ts" => self.ucm(func, args, opts, opt_map),
            "gam" | "gamfit" => self.gam(func, args, opts, opt_map),
            "mice" | "mi" | "multiple_imputation" => self.mice(func, args, opts, opt_map),
            "msauto" | "markov_ar" | "ms_ar" | "hamilton" => self.msauto(func, args, opts, opt_map),
            "svar" | "svec" => self.svar(func, args, opts, opt_map),
            "sirf" | "svar_irf" => self.sirf(func, args, opts, opt_map),
            "sfevd" | "svar_fevd" => self.sfevd(func, args, opts, opt_map),
            "threesl" | "three_sls" | "3sls" | "reg3" => self.threesl(func, args, opts, opt_map),
            "dfm" | "dynamic_factor" => self.dfm(func, args, opts, opt_map),
            "adtest" | "anderson_darling" => self.adtest(func, args, opts, opt_map),
            "lilliefors" | "lillie" => self.lilliefors(func, args, opts, opt_map),
            "omnibus" | "dagostino" => self.omnibus(func, args, opts, opt_map),
            "swilk" | "shapiro_wilk" | "shapiro" => self.swilk(func, args, opts, opt_map),
            "sfrancia" | "shapiro_francia" => self.sfrancia(func, args, opts, opt_map),
            "sktest" => self.sktest(func, args, opts, opt_map),
            "harveycollier" | "harvey_collier" | "hctest" => {
                self.harveycollier(func, args, opts, opt_map)
            }
            _ => return Ok(None),
        };
        result.map(Some)
    }

    pub(super) fn garch(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(format!(
                "{func}() requires df and variable name"
            )));
        }

        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(format!(
                    "{func}(): first argument must be a DataFrame"
                )))
            }
        };

        let col_name = match &args[1] {
            Expr::Var(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(format!(
                    "{func}(): second argument must be a column name"
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

    pub(super) fn ljungbox(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "ljungbox() requires a series or model".into(),
            ));
        }

        let series = match self.eval_expr(&args[0])? {
            Value::DataFrame(df) => {
                let col_name = match args.get(1) {
                    Some(Expr::Var(n)) => n.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "ljungbox(df, varname): second argument must be a column name".into(),
                        ))
                    }
                };
                Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?)
            }
            // GARCH standardized residuals
            Value::GarchResult(m) => m.standardized_residuals.clone(),
            // ARIMA residuals
            Value::ArimaResult(m) => Array1::from_vec(m.residuals().to_vec()),
            // OLS residuals
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
        println!("H₀: no autocorrelation up to lag {}", res.lags);
        println!("{sep}");
        println!("{:<6} {:>10} {:>10} {:>8}", "lag", "ACF", "Q", "p-value");
        println!("{sep}");
        let mut q_accum = 0.0_f64;
        let nf = res.n_obs as f64;
        for (i, &rho) in res.acf.iter().enumerate() {
            let k = i + 1;
            q_accum += nf * (nf + 2.0) * rho * rho / (nf - k as f64);
            // cumulative p-value for Q up to lag k
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

    pub(super) fn leverage(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        // shows only observations above cutoff (or all if few)
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
            println!("No observations above cutoff.");
        } else {
            println!("{:<8} {:>10}  ", "obs", "h_i");
            println!("{sep}");
            for (i, hi) in &flagged {
                println!("{:<8} {:>10.4}  high leverage", i, hi);
            }
            println!("{sep}");
            println!("{} observation(s) with h_i > {:.4}", flagged.len(), cutoff);
        }
        println!();

        Ok(Value::Nil)
    }

    pub(super) fn cooks(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        // configurable cutoff; default 4/n (most common rule of thumb)
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
            println!("No influential observations above cutoff.");
        } else {
            println!("{:<8} {:>10}  ", "obs", "D_i");
            println!("{sep}");
            for (i, di) in &flagged {
                let label = if *di > 1.0 {
                    "very influential"
                } else {
                    "influential"
                };
                println!("{:<8} {:>10.4}  {}", i, di, label);
            }
            println!("{sep}");
            println!("{} observation(s) with D_i > {:.4}", flagged.len(), cutoff);
        }
        println!();

        Ok(Value::Nil)
    }

    pub(super) fn vif(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime("vif() requires an OLS model".into()));
        }
        let ols = match self.eval_expr(&args[0])? {
            Value::OlsResult(m) => m,
            _ => return Err(HayashiError::Type("vif() only supports OLS models".into())),
        };

        let vifs =
            greeners::Diagnostics::vif(&ols.x).map_err(|e| self.rt_err(format!("vif: {e}")))?;

        let names = ols.result.variable_names.as_deref().unwrap_or(&[]);

        let sep = "─".repeat(40);
        println!("\nVariance Inflation Factor (VIF)");
        println!("{sep}");
        println!("{:<20} {:>8}  Diagnostic", "Variable", "VIF");
        println!("{sep}");
        let mut var_vec = Vec::new();
        let mut vif_vec = Vec::new();
        let mut diag_vec = Vec::new();
        for (i, &v) in vifs.iter().enumerate() {
            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
            let diag = if v.is_nan() {
                "constant"
            } else if v.is_infinite() || v > 10.0 {
                "severe multicollinearity"
            } else if v > 5.0 {
                "moderate"
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
            var_vec.push(Value::Str(name.to_string()));
            vif_vec.push(Value::Float(if v.is_nan() || v.is_infinite() {
                f64::NAN
            } else {
                v
            }));
            diag_vec.push(Value::Str(diag.to_string()));
        }
        println!("{sep}");
        println!("Reference: VIF<5 ok  |  5-10 moderate  |  >10 severe");
        println!();

        let mut columns = HashMap::new();
        columns.insert("variable".into(), Value::List(Arc::new(var_vec)));
        columns.insert("vif".into(), Value::List(Arc::new(vif_vec)));
        columns.insert("diagnostic".into(), Value::List(Arc::new(diag_vec)));
        let df = self.dict_to_dataframe(&columns)?;
        Ok(Value::DataFrame(Arc::new(df)))
    }

    pub(super) fn condnum(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            "severe multicollinearity"
        } else if kappa > 30.0 {
            "moderate multicollinearity"
        } else if kappa > 10.0 {
            "attention"
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
        println!("Reference: κ<10 ok  |  10-30 attention  |  30-100 moderate  |  >100 severe");
        println!();

        let mut map = HashMap::new();
        map.insert(
            "condition_number".into(),
            Value::Float(if kappa.is_infinite() {
                f64::INFINITY
            } else {
                kappa
            }),
        );
        map.insert("diagnostic".into(), Value::Str(diag.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn durbinwatson(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            "likely positive autocorrelation"
        } else if dw > 2.5 {
            "likely negative autocorrelation"
        } else {
            "no evident autocorrelation"
        };

        let sep = "─".repeat(50);
        println!("\nDurbin-Watson Test");
        println!("{sep}");
        println!("H₀: no first-order autocorrelation");
        println!("{sep}");
        println!("{:<18} {:>10}", "DW statistic", format!("{dw:.4}"));
        println!("{:<18} {:>10}", "Interpretation", interpretation);
        println!("{sep}");
        println!("Reference: DW ≈ 2 (no autocorr.) | <1.5 (positive) | >2.5 (negative)");
        println!();

        let mut map = HashMap::new();
        map.insert("dw".into(), Value::Float(dw));
        map.insert("interpretation".into(), Value::Str(interpretation.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn white(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            "Test", "Statistic", "p-value", ""
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

        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("White Test".into()));
        map.insert("lm_stat".into(), Value::Float(lm));
        map.insert("df".into(), Value::Int(df as i64));
        map.insert("p_value".into(), Value::Float(p));
        let conclusion = if p < 0.05 {
            "reject H0 -> heteroscedasticity present"
        } else {
            "do not reject H0 -> homoscedastic"
        };
        map.insert("conclusion".into(), Value::Str(conclusion.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn reset(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        // y = residuals + fitted values
        let y = &ols.residuals + &fitted;

        let (f, p, df1, df2) = greeners::SpecificationTests::reset_test(&y, &ols.x, &fitted, power)
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
        println!("H₀: correct linear specification");
        println!("{sep}");
        println!(
            "{:<24} {:>10} {:>10} {:>4}",
            "Test", "Statistic", "p-value", ""
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

        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("Ramsey RESET Test".into()));
        map.insert("f_stat".into(), Value::Float(f));
        map.insert("df1".into(), Value::Int(df1 as i64));
        map.insert("df2".into(), Value::Int(df2 as i64));
        map.insert("p_value".into(), Value::Float(p));
        map.insert("power".into(), Value::Int(power as i64));
        let conclusion = if p < 0.05 {
            "reject H0 -> misspecification"
        } else {
            "do not reject H0 -> linear specification adequate"
        };
        map.insert("conclusion".into(), Value::Str(conclusion.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn jb(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        println!("H₀: residuals normally distributed");
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

        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("Jarque-Bera Test".into()));
        map.insert("jb_stat".into(), Value::Float(jb));
        map.insert("df".into(), Value::Int(2));
        map.insert("p_value".into(), Value::Float(p));
        map.insert("n".into(), Value::Int(series.len() as i64));
        let conclusion = if p < 0.05 {
            "reject H0 -> non-normal"
        } else {
            "do not reject H0 -> normal"
        };
        map.insert("conclusion".into(), Value::Str(conclusion.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn bgodfrey(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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

        let (lm, p, df) =
            greeners::SpecificationTests::breusch_godfrey_test(&ols.residuals, &ols.x, lags)
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
        println!("H₀: no serial autocorrelation of order {lags}");
        println!("{sep}");
        println!(
            "{:<24} {:>10} {:>10} {:>4}",
            "Test", "Statistic", "p-value", ""
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

        let mut map = HashMap::new();
        map.insert("test".into(), Value::Str("Breusch-Godfrey LM Test".into()));
        map.insert("lm_stat".into(), Value::Float(lm));
        map.insert("df".into(), Value::Int(df as i64));
        map.insert("p_value".into(), Value::Float(p));
        map.insert("lags".into(), Value::Int(lags as i64));
        let conclusion = if p < 0.05 {
            "reject H0 -> serial autocorrelation present"
        } else {
            "do not reject H0 -> no serial autocorrelation"
        };
        map.insert("conclusion".into(), Value::Str(conclusion.into()));
        Ok(Value::Dict(Arc::new(map)))
    }

    pub(super) fn archtest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "archtest() requires a series or GARCH model".into(),
            ));
        }

        let series = match self.eval_expr(&args[0])? {
            // raw series: archtest(df, varname, lags=5)
            Value::DataFrame(df) => {
                let col_name = match args.get(1) {
                    Some(Expr::Var(n)) => n.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "archtest(df, varname): second argument must be a column name".into(),
                        ))
                    }
                };
                Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?)
            }
            // GARCH residuals: archtest(model, lags=5)
            // uses standardized residuals z_t = ε_t/√h_t — under H₀ of
            // correct specification, z_t² should have no autocorrelation
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
            "Test", "Statistic", "p-value", ""
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

    pub(super) fn acf(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(self.rt_err("acf(df, var, lags=20) or acf(model, lags=20)"));
        }
        let series = match self.eval_expr(&args[0])? {
            Value::DataFrame(df) => {
                let col_name = match args.get(1) {
                    Some(Expr::Var(n)) => n.clone(),
                    _ => {
                        return Err(
                            self.rt_err("acf(df, var): second argument must be a column name")
                        )
                    }
                };
                Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?)
            }
            Value::OlsResult(m) => m.residuals.clone(),
            Value::GarchResult(m) => m.standardized_residuals.clone(),
            Value::ArimaResult(m) => Array1::from_vec(m.residuals().to_vec()),
            _ => {
                return Err(HayashiError::Type(
                    "acf(): argument must be a DataFrame or model".into(),
                ))
            }
        };
        let lags = match opt_map.get("lags") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 20,
        };
        let vals = greeners::TimeSeries::acf(&series, lags)
            .map_err(|e| self.rt_err(format!("acf: {e}")))?;
        let list: Vec<Value> = vals.iter().map(|&v| Value::Float(v)).collect();
        Ok(Value::List(Arc::new(list)))
    }

    pub(super) fn pacf(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(self.rt_err("pacf(df, var, lags=20) or pacf(model, lags=20)"));
        }
        let series = match self.eval_expr(&args[0])? {
            Value::DataFrame(df) => {
                let col_name = match args.get(1) {
                    Some(Expr::Var(n)) => n.clone(),
                    _ => {
                        return Err(
                            self.rt_err("pacf(df, var): second argument must be a column name")
                        )
                    }
                };
                Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?)
            }
            Value::OlsResult(m) => m.residuals.clone(),
            Value::GarchResult(m) => m.standardized_residuals.clone(),
            Value::ArimaResult(m) => Array1::from_vec(m.residuals().to_vec()),
            _ => {
                return Err(HayashiError::Type(
                    "pacf(): argument must be a DataFrame or model".into(),
                ))
            }
        };
        let lags = match opt_map.get("lags") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 20,
        };
        let vals = greeners::TimeSeries::pacf(&series, lags)
            .map_err(|e| self.rt_err(format!("pacf: {e}")))?;
        let list: Vec<Value> = vals.iter().map(|&v| Value::Float(v)).collect();
        Ok(Value::List(Arc::new(list)))
    }

    pub(super) fn cusumtest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(self.rt_err("cusumtest(model) requires an OLS model"));
        }
        let ols = match self.eval_expr(&args[0])? {
            Value::OlsResult(m) => m,
            _ => {
                return Err(HayashiError::Type(
                    "cusumtest(): only supports OLS models".into(),
                ))
            }
        };
        // Reconstruct y from residuals + fitted (x · params)
        let y = &ols.residuals + &ols.x.dot(&ols.result.params);
        let result = greeners::CUSUMTest::test(&y, &ols.x)
            .map_err(|e| self.rt_err(format!("cusumtest: {e}")))?;
        print!("{result}");
        Ok(Value::Nil)
    }

    pub(super) fn forecast_vol(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        println!(
            "\nForecast de Volatilidade — {model_label}({}, {}) [{dist_label}]  {steps} passos",
            model.p, model.q
        );
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

    fn sig_stars(p: f64) -> &'static str {
        if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        }
    }

    pub(super) fn diagnostics(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime(
                "diagnostics() requires a model (OLS, GARCH, or ARIMA)".into(),
            ));
        }

        let thick = "═".repeat(62);
        let thin = "─".repeat(62);

        let diagnostics = match self.eval_expr(&args[0])? {
            Value::OlsResult(ref ols) => self.diagnostics_ols(ols, &thick, &thin)?,
            Value::GarchResult(ref m) => self.diagnostics_garch(m, &thick, &thin)?,
            Value::ArimaResult(ref m) => self.diagnostics_arima(m, &thick, &thin)?,
            Value::VarResult(ref m) => self.diagnostics_var(m, &thick, &thin)?,
            Value::VecmResult(ref m) => self.diagnostics_vecm(m, &thick, &thin)?,
            Value::IvResult(ref iv) => self.diagnostics_iv(iv, &thick, &thin)?,
            Value::PanelResult(ref fe) => self.diagnostics_panel(fe, &thick, &thin)?,
            Value::ReResult(ref re) => self.diagnostics_re(re, &thick, &thin)?,
            _ => {
                return Err(HayashiError::Type(
                    "diagnostics() suporta OLS, GARCH, ARIMA, VAR, VECM, IV, FE e RE".into(),
                ))
            }
        };

        Ok(Value::Dict(Arc::new(diagnostics)))
    }

    pub(super) fn diagnostics_ols(
        &mut self,
        ols: &super::models::OlsModel,
        thick: &str,
        thin: &str,
    ) -> Result<HashMap<String, Value>> {
        let mut diagnostics: HashMap<String, Value> = HashMap::new();
        println!("\n{thick}");
        println!(
            " DIAGNOSTICS — OLS  (n={}  k={})",
            ols.residuals.len(),
            ols.x.ncols()
        );
        println!("{thick}");

        diagnostics.insert("model".into(), Value::Str("OLS".into()));
        diagnostics.insert("n".into(), Value::Int(ols.residuals.len() as i64));
        diagnostics.insert("k".into(), Value::Int(ols.x.ncols() as i64));
        diagnostics.insert("jarque_bera".into(), self.ols_jarque_bera(ols)?);
        diagnostics.insert("durbin_watson".into(), self.ols_durbin_watson(ols));
        diagnostics.insert("breusch_godfrey".into(), self.ols_breusch_godfrey(ols)?);
        diagnostics.insert("white".into(), self.ols_white(ols)?);
        diagnostics.insert("reset".into(), self.ols_reset(ols)?);
        if let Some(vif) = self.ols_vif(ols)? {
            diagnostics.insert("vif".into(), vif);
        }
        if let Some(cooks) = self.ols_cooks(ols)? {
            diagnostics.insert("cooks_d".into(), cooks);
        }

        println!("\n{thin}");
        println!("  *** p<0.01  ** p<0.05  * p<0.10");
        println!("{thick}\n");
        Ok(diagnostics)
    }

    fn ols_jarque_bera(&self, ols: &super::models::OlsModel) -> Result<Value> {
        println!("\n── Residual Normality (Jarque-Bera)");
        match greeners::Diagnostics::jarque_bera(&ols.residuals) {
            Ok((jb, p)) => {
                println!(
                    "   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}",
                    jb,
                    p,
                    Self::sig_stars(p)
                );
                let mut jb_map = HashMap::new();
                jb_map.insert("jb_stat".into(), Value::Float(jb));
                jb_map.insert("p_value".into(), Value::Float(p));
                jb_map.insert("df".into(), Value::Int(2));
                Ok(Value::Dict(Arc::new(jb_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }

    fn ols_durbin_watson(&self, ols: &super::models::OlsModel) -> Value {
        let dw = greeners::Diagnostics::durbin_watson(&ols.residuals);
        let dw_label = if dw < 1.5 {
            "positive autocorr."
        } else if dw > 2.5 {
            "negative autocorr."
        } else {
            "no evident autocorr."
        };
        println!("\n── First-Order Autocorrelation (Durbin-Watson)");
        println!("   DW = {:.4}  [{}]", dw, dw_label);
        let mut dw_map = HashMap::new();
        dw_map.insert("dw".into(), Value::Float(dw));
        dw_map.insert("interpretation".into(), Value::Str(dw_label.into()));
        Value::Dict(Arc::new(dw_map))
    }

    fn ols_breusch_godfrey(&self, ols: &super::models::OlsModel) -> Result<Value> {
        println!("\n── Serial Autocorrelation (Breusch-Godfrey, lags=4)");
        match greeners::SpecificationTests::breusch_godfrey_test(&ols.residuals, &ols.x, 4) {
            Ok((lm, p, df)) => {
                println!(
                    "   LM ~ χ²({})   = {:>9.4}   p = {:.4}  {}",
                    df,
                    lm,
                    p,
                    Self::sig_stars(p)
                );
                let mut bg_map = HashMap::new();
                bg_map.insert("lm_stat".into(), Value::Float(lm));
                bg_map.insert("df".into(), Value::Int(df as i64));
                bg_map.insert("p_value".into(), Value::Float(p));
                Ok(Value::Dict(Arc::new(bg_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }

    fn ols_white(&self, ols: &super::models::OlsModel) -> Result<Value> {
        println!("\n── Heteroscedasticidade (White)");
        match greeners::SpecificationTests::white_test(&ols.residuals, &ols.x) {
            Ok((lm, p, df)) => {
                println!(
                    "   LM ~ χ²({})   = {:>9.4}   p = {:.4}  {}",
                    df,
                    lm,
                    p,
                    Self::sig_stars(p)
                );
                let mut white_map = HashMap::new();
                white_map.insert("lm_stat".into(), Value::Float(lm));
                white_map.insert("df".into(), Value::Int(df as i64));
                white_map.insert("p_value".into(), Value::Float(p));
                Ok(Value::Dict(Arc::new(white_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }

    fn ols_reset(&self, ols: &super::models::OlsModel) -> Result<Value> {
        println!("\n── Functional Specification (RESET, power=3)");
        let fitted = ols.result.fitted_values(&ols.x);
        let y = &ols.residuals + &fitted;
        match greeners::SpecificationTests::reset_test(&y, &ols.x, &fitted, 3) {
            Ok((f, p, df1, df2)) => {
                println!(
                    "   F ~ F({},{}) = {:>9.4}   p = {:.4}  {}",
                    df1,
                    df2,
                    f,
                    p,
                    Self::sig_stars(p)
                );
                let mut reset_map = HashMap::new();
                reset_map.insert("f_stat".into(), Value::Float(f));
                reset_map.insert("df1".into(), Value::Int(df1 as i64));
                reset_map.insert("df2".into(), Value::Int(df2 as i64));
                reset_map.insert("p_value".into(), Value::Float(p));
                Ok(Value::Dict(Arc::new(reset_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }

    fn ols_vif(&self, ols: &super::models::OlsModel) -> Result<Option<Value>> {
        println!("\n── Multicolinearidade (VIF)");
        let names = ols.result.variable_names.as_deref().unwrap_or(&[]);
        let mut vif_var = Vec::new();
        let mut vif_val = Vec::new();
        let mut vif_diag = Vec::new();
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
                    vif_var.push(Value::Str(name.to_string()));
                    vif_val.push(Value::Float(if v.is_infinite() {
                        f64::INFINITY
                    } else {
                        v
                    }));
                    vif_diag.push(Value::Str(diag.to_string()));
                }
            }
            Err(e) => println!("   error: {e}"),
        }
        if vif_var.is_empty() {
            return Ok(None);
        }
        let mut vif_columns = HashMap::new();
        vif_columns.insert("variable".into(), Value::List(Arc::new(vif_var)));
        vif_columns.insert("vif".into(), Value::List(Arc::new(vif_val)));
        vif_columns.insert("diagnostic".into(), Value::List(Arc::new(vif_diag)));
        let vif_df = self.dict_to_dataframe(&vif_columns)?;
        Ok(Some(Value::DataFrame(Arc::new(vif_df))))
    }

    fn ols_cooks(&self, ols: &super::models::OlsModel) -> Result<Option<Value>> {
        let n = ols.residuals.len();
        let mse = ols.result.sigma * ols.result.sigma;
        let cutoff = 4.0 / n as f64;
        println!("\n── Influential Observations (Cook's D > {:.4})", cutoff);
        let mut cook_obs = Vec::new();
        let mut cook_d = Vec::new();
        let mut cook_label = Vec::new();
        match greeners::Diagnostics::cooks_distance(&ols.residuals, &ols.x, mse) {
            Ok(d) => {
                let flagged: Vec<(usize, f64)> = d
                    .iter()
                    .enumerate()
                    .filter(|(_, &di)| di > cutoff)
                    .map(|(i, &di)| (i + 1, di))
                    .collect();
                if flagged.is_empty() {
                    println!("   No influential observations.");
                } else {
                    for (i, di) in &flagged {
                        let label = if *di > 1.0 {
                            "very influential"
                        } else {
                            "influential"
                        };
                        println!("   obs {:>4}  D = {:.4}  [{}]", i, di, label);
                        cook_obs.push(Value::Int(*i as i64));
                        cook_d.push(Value::Float(*di));
                        cook_label.push(Value::Str(label.to_string()));
                    }
                }
            }
            Err(e) => println!("   error: {e}"),
        }
        if cook_obs.is_empty() {
            return Ok(None);
        }
        let mut cook_columns = HashMap::new();
        cook_columns.insert("observation".into(), Value::List(Arc::new(cook_obs)));
        cook_columns.insert("cooks_d".into(), Value::List(Arc::new(cook_d)));
        cook_columns.insert("label".into(), Value::List(Arc::new(cook_label)));
        let cook_df = self.dict_to_dataframe(&cook_columns)?;
        Ok(Some(Value::DataFrame(Arc::new(cook_df))))
    }
    pub(super) fn diagnostics_garch(
        &mut self,
        m: &greeners::GarchResult,
        thick: &str,
        thin: &str,
    ) -> Result<HashMap<String, Value>> {
        let mut diagnostics: HashMap<String, Value> = HashMap::new();
        let model_label = match m.model_type {
            greeners::GarchModelType::GARCH => "GARCH",
            greeners::GarchModelType::EGARCH => "EGARCH",
            greeners::GarchModelType::GJRGARCH => "GJR-GARCH",
        };
        println!("\n{thick}");
        println!(
            " DIAGNOSTICS — {model_label}({}, {})  (n={})",
            m.p, m.q, m.n_obs
        );
        println!("{thick}");

        diagnostics.insert("model".into(), Value::Str(model_label.into()));
        diagnostics.insert("p".into(), Value::Int(m.p as i64));
        diagnostics.insert("q".into(), Value::Int(m.q as i64));
        diagnostics.insert("n".into(), Value::Int(m.n_obs as i64));

        let std_res = &m.standardized_residuals;
        diagnostics.insert("ljung_box".into(), self.garch_ljung_box(std_res)?);
        diagnostics.insert("arch_test".into(), self.garch_arch_test(std_res)?);
        diagnostics.insert("jarque_bera".into(), self.garch_jarque_bera(std_res)?);

        println!("\n{thin}");
        println!("  *** p<0.01  ** p<0.05  * p<0.10");
        println!("{thick}\n");
        Ok(diagnostics)
    }

    fn garch_ljung_box(&self, std_res: &ndarray::Array1<f64>) -> Result<Value> {
        println!("\n── Autocorrelation in Standardized Residuals (Ljung-Box, lags=10)");
        match greeners::Diagnostics::ljung_box(std_res, 10) {
            Ok(r) => {
                println!(
                    "   Q(10) = {:>9.4}   p = {:.4}  {}",
                    r.q_stat,
                    r.p_value,
                    Self::sig_stars(r.p_value)
                );
                let mut lb_map = HashMap::new();
                lb_map.insert("q_stat".into(), Value::Float(r.q_stat));
                lb_map.insert("p_value".into(), Value::Float(r.p_value));
                lb_map.insert("lags".into(), Value::Int(10));
                Ok(Value::Dict(Arc::new(lb_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }

    fn garch_arch_test(&self, std_res: &ndarray::Array1<f64>) -> Result<Value> {
        println!("\n── Efeitos ARCH Residuais (Engle LM, lags=5)");
        match greeners::Diagnostics::arch_test(std_res, 5) {
            Ok(r) => {
                println!(
                    "   LM ~ χ²({}) = {:>9.4}   p = {:.4}  {}",
                    r.lags,
                    r.lm_stat,
                    r.lm_pvalue,
                    Self::sig_stars(r.lm_pvalue)
                );
                let mut arch_map = HashMap::new();
                arch_map.insert("lm_stat".into(), Value::Float(r.lm_stat));
                arch_map.insert("p_value".into(), Value::Float(r.lm_pvalue));
                arch_map.insert("lags".into(), Value::Int(r.lags as i64));
                Ok(Value::Dict(Arc::new(arch_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }

    fn garch_jarque_bera(&self, std_res: &ndarray::Array1<f64>) -> Result<Value> {
        println!("\n── Standardized Residual Normality (Jarque-Bera)");
        match greeners::Diagnostics::jarque_bera(std_res) {
            Ok((jb, p)) => {
                println!(
                    "   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}",
                    jb,
                    p,
                    Self::sig_stars(p)
                );
                let mut jb_map = HashMap::new();
                jb_map.insert("jb_stat".into(), Value::Float(jb));
                jb_map.insert("p_value".into(), Value::Float(p));
                jb_map.insert("df".into(), Value::Int(2));
                Ok(Value::Dict(Arc::new(jb_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }
    pub(super) fn diagnostics_arima(
        &mut self,
        m: &greeners::ArimaResult,
        thick: &str,
        thin: &str,
    ) -> Result<HashMap<String, Value>> {
        let mut diagnostics: HashMap<String, Value> = HashMap::new();
        println!("\n{thick}");
        println!(" DIAGNOSTICS — ARIMA");
        println!("{thick}");

        diagnostics.insert("model".into(), Value::Str("ARIMA".into()));

        let resid = Array1::from_vec(m.residuals().to_vec());
        diagnostics.insert("ljung_box".into(), self.arima_ljung_box(&resid)?);
        diagnostics.insert("jarque_bera".into(), self.arima_jarque_bera(&resid)?);

        println!("\n{thin}");
        println!("  *** p<0.01  ** p<0.05  * p<0.10");
        println!("{thick}\n");
        Ok(diagnostics)
    }

    fn arima_ljung_box(&self, resid: &ndarray::Array1<f64>) -> Result<Value> {
        println!("\n── Autocorrelation in Residuals (Ljung-Box, lags=10)");
        match greeners::Diagnostics::ljung_box(resid, 10) {
            Ok(r) => {
                println!(
                    "   Q(10) = {:>9.4}   p = {:.4}  {}",
                    r.q_stat,
                    r.p_value,
                    Self::sig_stars(r.p_value)
                );
                let mut lb_map = HashMap::new();
                lb_map.insert("q_stat".into(), Value::Float(r.q_stat));
                lb_map.insert("p_value".into(), Value::Float(r.p_value));
                lb_map.insert("lags".into(), Value::Int(10));
                Ok(Value::Dict(Arc::new(lb_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }

    fn arima_jarque_bera(&self, resid: &ndarray::Array1<f64>) -> Result<Value> {
        println!("\n── Residual Normality (Jarque-Bera)");
        match greeners::Diagnostics::jarque_bera(resid) {
            Ok((jb, p)) => {
                println!(
                    "   JB ~ χ²(2)  = {:>9.4}   p = {:.4}  {}",
                    jb,
                    p,
                    Self::sig_stars(p)
                );
                let mut jb_map = HashMap::new();
                jb_map.insert("jb_stat".into(), Value::Float(jb));
                jb_map.insert("p_value".into(), Value::Float(p));
                jb_map.insert("df".into(), Value::Int(2));
                Ok(Value::Dict(Arc::new(jb_map)))
            }
            Err(e) => {
                println!("   error: {e}");
                Ok(Value::Str(format!("error: {e}")))
            }
        }
    }
    pub(super) fn diagnostics_var(
        &mut self,
        m: &greeners::var::VarResult,
        thick: &str,
        thin: &str,
    ) -> Result<HashMap<String, Value>> {
        let mut diagnostics: HashMap<String, Value> = HashMap::new();
        let k = m.n_vars;
        println!("\n{thick}");
        println!(" DIAGNOSTICS — VAR({})  (n={}  k={})", m.lags, m.n_obs, k);
        println!("{thick}");

        diagnostics.insert("model".into(), Value::Str("VAR".into()));
        diagnostics.insert("lags".into(), Value::Int(m.lags as i64));
        diagnostics.insert("n".into(), Value::Int(m.n_obs as i64));
        diagnostics.insert("k".into(), Value::Int(k as i64));
        diagnostics.insert("aic".into(), Value::Float(m.aic));
        diagnostics.insert("bic".into(), Value::Float(m.bic));

        // ── Residual standard deviation by equation (diagonal of Σ_u)
        println!("\n── Residual Standard Deviation by Equation");
        let mut sd_var = Vec::new();
        let mut sd_val = Vec::new();
        for (i, name) in m.var_names.iter().enumerate() {
            let sd = m.sigma_u[[i, i]].sqrt();
            println!("   {:<22} σ = {:.6}", name, sd);
            sd_var.push(Value::Str(name.clone()));
            sd_val.push(Value::Float(sd));
        }
        if !sd_var.is_empty() {
            let mut sd_columns = HashMap::new();
            sd_columns.insert("variable".into(), Value::List(Arc::new(sd_var)));
            sd_columns.insert("residual_sd".into(), Value::List(Arc::new(sd_val)));
            let sd_df = self.dict_to_dataframe(&sd_columns)?;
            diagnostics.insert("residual_sd".into(), Value::DataFrame(Arc::new(sd_df)));
        }

        // ── Residual correlation matrix (normalized Σ_u)
        if k > 1 {
            println!("\n── Contemporaneous Residual Correlation");
            let mut corr_var1 = Vec::new();
            let mut corr_var2 = Vec::new();
            let mut corr_val = Vec::new();
            for i in 0..k {
                for j in 0..k {
                    let r = if i == j {
                        1.0
                    } else {
                        m.sigma_u[[i, j]] / (m.sigma_u[[i, i]] * m.sigma_u[[j, j]]).sqrt()
                    };
                    corr_var1.push(Value::Str(m.var_names[i].clone()));
                    corr_var2.push(Value::Str(m.var_names[j].clone()));
                    corr_val.push(Value::Float(r));
                }
            }
            let mut corr_columns = HashMap::new();
            corr_columns.insert("variable_1".into(), Value::List(Arc::new(corr_var1)));
            corr_columns.insert("variable_2".into(), Value::List(Arc::new(corr_var2)));
            corr_columns.insert("correlation".into(), Value::List(Arc::new(corr_val)));
            let corr_df = self.dict_to_dataframe(&corr_columns)?;
            diagnostics.insert(
                "residual_correlation".into(),
                Value::DataFrame(Arc::new(corr_df)),
            );
        }

        println!("\n── Note");
        println!("   Residuals are not stored in VarResult — for LB/JB by equation,");
        println!("   extract the series and run ljungbox/jb directly.");
        println!("\n{thin}");
        println!("{thick}\n");
        Ok(diagnostics)
    }
    pub(super) fn diagnostics_vecm(
        &mut self,
        m: &greeners::vecm::VecmResult,
        thick: &str,
        thin: &str,
    ) -> Result<HashMap<String, Value>> {
        let mut diagnostics: HashMap<String, Value> = HashMap::new();
        let k = m.n_vars;
        let r = m.rank;
        let n = m.n_obs;

        println!("\n{thick}");
        println!(" DIAGNOSTICS — VECM  (n={}  k={}  rank={})", n, k, r);
        println!("{thick}");

        diagnostics.insert("model".into(), Value::Str("VECM".into()));
        diagnostics.insert("n".into(), Value::Int(n as i64));
        diagnostics.insert("k".into(), Value::Int(k as i64));
        diagnostics.insert("rank".into(), Value::Int(r as i64));
        diagnostics.insert("johansen".into(), self.vecm_johansen(m)?);
        diagnostics.insert("alpha".into(), self.vecm_alpha(m)?);
        diagnostics.insert("beta".into(), self.vecm_beta(m)?);

        println!("\n── Note");
        println!("   VecmResult does not store variable names or residuals.");
        println!("   For names, see the order passed to vecm().");
        println!("\n{thin}");
        println!("{thick}\n");
        Ok(diagnostics)
    }

    fn vecm_johansen(&self, m: &greeners::vecm::VecmResult) -> Result<Value> {
        let k = m.n_vars;
        let n = m.n_obs as f64;
        let eig = &m.eigenvalues; // ordenados decrescente

        // λ_trace(r₀) = -n Σ_{i=r₀}^{k-1} ln(1 - λ_i)  H₀: rank ≤ r₀
        // CVs 5%: Osterwald-Lenum (1992) Tabela 1 — constant restrita
        let cv_5pct: &[f64] = &[9.24, 19.96, 34.91, 53.12, 76.07, 102.56, 131.70];
        println!("\n── Johansen Test (Trace)");
        println!("   H₀: rank ≤ r   CVs 5%: Osterwald-Lenum (1992) Tabela 1");
        println!(
            "   {:<6} {:>10} {:>12} {:>10} {:>6}",
            "H₀:r≤", "λ_max", "λ_trace", "CV 5%", ""
        );
        println!("   {}", "─".repeat(46));
        let mut joh_h0 = Vec::new();
        let mut joh_lam_max = Vec::new();
        let mut joh_trace = Vec::new();
        let mut joh_cv = Vec::new();
        let mut joh_reject = Vec::new();
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
            joh_h0.push(Value::Int(r0 as i64));
            joh_lam_max.push(Value::Float(lam_max));
            joh_trace.push(Value::Float(trace_stat));
            joh_cv.push(Value::Float(cv));
            joh_reject.push(Value::Str(reject.to_string()));
        }
        println!("   (* rejects H₀ at 5%)");
        let mut joh_columns = HashMap::new();
        joh_columns.insert("h0_rank".into(), Value::List(Arc::new(joh_h0)));
        joh_columns.insert("lambda_max".into(), Value::List(Arc::new(joh_lam_max)));
        joh_columns.insert("trace_stat".into(), Value::List(Arc::new(joh_trace)));
        joh_columns.insert("cv_5pct".into(), Value::List(Arc::new(joh_cv)));
        joh_columns.insert("reject".into(), Value::List(Arc::new(joh_reject)));
        let joh_df = self.dict_to_dataframe(&joh_columns)?;
        Ok(Value::DataFrame(Arc::new(joh_df)))
    }

    fn vecm_alpha(&self, m: &greeners::vecm::VecmResult) -> Result<Value> {
        let k = m.n_vars;
        let r = m.rank;
        println!("\n── Adjustment Speeds (Alpha)  [negative sign = correction to equilibrium]");
        let mut alpha_vec = Vec::new();
        let mut alpha_eq = Vec::new();
        let mut alpha_ec = Vec::new();
        for ec in 0..r {
            println!("   EC Vector {}", ec + 1);
            for eq in 0..k {
                println!(
                    "     equation {:>2}   α = {:>9.4}",
                    eq + 1,
                    m.alpha[[eq, ec]]
                );
                alpha_vec.push(Value::Float(m.alpha[[eq, ec]]));
                alpha_eq.push(Value::Int((eq + 1) as i64));
                alpha_ec.push(Value::Int((ec + 1) as i64));
            }
        }
        if alpha_vec.is_empty() {
            return Ok(Value::Nil);
        }
        let mut alpha_columns = HashMap::new();
        alpha_columns.insert("ec_vector".into(), Value::List(Arc::new(alpha_ec)));
        alpha_columns.insert("equation".into(), Value::List(Arc::new(alpha_eq)));
        alpha_columns.insert("alpha".into(), Value::List(Arc::new(alpha_vec)));
        let alpha_df = self.dict_to_dataframe(&alpha_columns)?;
        Ok(Value::DataFrame(Arc::new(alpha_df)))
    }

    fn vecm_beta(&self, m: &greeners::vecm::VecmResult) -> Result<Value> {
        let k = m.n_vars;
        let r = m.rank;
        println!("\n── Cointegration Vectors (Beta)");
        let mut beta_vec = Vec::new();
        let mut beta_var = Vec::new();
        let mut beta_ec = Vec::new();
        for ec in 0..r {
            println!("   EC{}:", ec + 1);
            for var in 0..k {
                println!("     var {:>2}   β = {:>9.4}", var + 1, m.beta[[var, ec]]);
                beta_vec.push(Value::Float(m.beta[[var, ec]]));
                beta_var.push(Value::Int((var + 1) as i64));
                beta_ec.push(Value::Int((ec + 1) as i64));
            }
        }
        if beta_vec.is_empty() {
            return Ok(Value::Nil);
        }
        let mut beta_columns = HashMap::new();
        beta_columns.insert("ec_vector".into(), Value::List(Arc::new(beta_ec)));
        beta_columns.insert("variable".into(), Value::List(Arc::new(beta_var)));
        beta_columns.insert("beta".into(), Value::List(Arc::new(beta_vec)));
        let beta_df = self.dict_to_dataframe(&beta_columns)?;
        Ok(Value::DataFrame(Arc::new(beta_df)))
    }
    pub(super) fn diagnostics_iv(
        &mut self,
        iv: &greeners::iv::IvResult,
        thick: &str,
        thin: &str,
    ) -> Result<HashMap<String, Value>> {
        let mut diagnostics: HashMap<String, Value> = HashMap::new();
        let k = iv.params.len();
        let n = iv.n_obs;
        let df = iv.df_resid;
        let mse = iv.sigma * iv.sigma;

        println!("\n{thick}");
        println!(" DIAGNOSTICS — IV/2SLS  (n={}  k={}  df={})", n, k, df);
        println!("{thick}");

        diagnostics.insert("model".into(), Value::Str("IV/2SLS".into()));
        diagnostics.insert("n".into(), Value::Int(n as i64));
        diagnostics.insert("k".into(), Value::Int(k as i64));
        diagnostics.insert("df".into(), Value::Int(df as i64));
        diagnostics.insert("r2".into(), Value::Float(iv.r_squared));
        diagnostics.insert("sigma".into(), Value::Float(iv.sigma));
        diagnostics.insert("mse".into(), Value::Float(mse));

        println!("\n── Fit");
        println!(
            "   R²  = {:.4}   σ = {:.6}   MSE = {:.6}",
            iv.r_squared, iv.sigma, mse
        );

        if let Some(coeffs) = self.iv_coefficients(iv)? {
            diagnostics.insert("coefficients".into(), coeffs);
        }

        println!("\n── Tests Not Available");
        println!("   Residuals and Z matrix not stored in IvResult.");
        println!("   • Sargan (overidentification): needs Z matrix");
        println!("   • Endogeneity (Wu-Hausman): compare IV vs OLS manually");
        println!("   • Weak instrument: check first-stage F (rule: F > 10)");
        println!("\n{thin}");
        println!("   *** p<0.01  ** p<0.05  * p<0.10");
        println!("{thick}\n");
        Ok(diagnostics)
    }

    fn iv_coefficients(&self, iv: &greeners::iv::IvResult) -> Result<Option<Value>> {
        let k = iv.params.len();
        let names = iv.variable_names.as_deref().unwrap_or(&[]);
        println!("\n── Coefficient Significance");
        println!("   {:<22} {:>8} {:>8}", "Variable", "p-value", "");
        println!("   {}", "─".repeat(40));
        let mut iv_var = Vec::new();
        let mut iv_p = Vec::new();
        let mut iv_sig = Vec::new();
        for i in 0..k {
            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
            let s = Self::sig_stars(iv.p_values[i]);
            println!("   {:<22} {:>8.4} {:>4}", name, iv.p_values[i], s);
            iv_var.push(Value::Str(name.to_string()));
            iv_p.push(Value::Float(iv.p_values[i]));
            iv_sig.push(Value::Str(s.to_string()));
        }
        if iv_var.is_empty() {
            return Ok(None);
        }
        let mut iv_columns = HashMap::new();
        iv_columns.insert("variable".into(), Value::List(Arc::new(iv_var)));
        iv_columns.insert("p_value".into(), Value::List(Arc::new(iv_p)));
        iv_columns.insert("significance".into(), Value::List(Arc::new(iv_sig)));
        let iv_df = self.dict_to_dataframe(&iv_columns)?;
        Ok(Some(Value::DataFrame(Arc::new(iv_df))))
    }

    pub(super) fn diagnostics_panel(
        &mut self,
        fe: &greeners::panel::PanelResult,
        thick: &str,
        thin: &str,
    ) -> Result<HashMap<String, Value>> {
        let mut diagnostics: HashMap<String, Value> = HashMap::new();
        let k = fe.params.len();

        println!("\n{thick}");
        println!(
            " DIAGNOSTICS — Fixed Effects  (n={}  N={}  T≈{:.1}  k={})",
            fe.n_obs,
            fe.n_entities,
            fe.n_obs as f64 / fe.n_entities.max(1) as f64,
            k
        );
        println!("{thick}");

        diagnostics.insert("model".into(), Value::Str("Fixed Effects".into()));
        diagnostics.insert("n".into(), Value::Int(fe.n_obs as i64));
        diagnostics.insert("n_entities".into(), Value::Int(fe.n_entities as i64));
        diagnostics.insert("k".into(), Value::Int(k as i64));
        diagnostics.insert("r2_within".into(), Value::Float(fe.r_squared));
        diagnostics.insert("sigma".into(), Value::Float(fe.sigma));
        diagnostics.insert("df_resid".into(), Value::Int(fe.df_resid as i64));

        println!("\n── Fit (Within)");
        println!(
            "   R² within = {:.4}   σ = {:.6}   df = {}",
            fe.r_squared, fe.sigma, fe.df_resid
        );

        if let Some(coeffs) = self.fe_coefficients(fe)? {
            diagnostics.insert("coefficients".into(), coeffs);
        }

        println!("\n── Tests Not Available");
        println!("   Residuals not stored in PanelResult.");
        println!("   • Hausman FE vs RE: use hausman(fe_model, re_model)");
        println!("   • JB / Ljung-Box: run on manually extracted residuals");
        println!("\n{thin}");
        println!("   *** p<0.01  ** p<0.05  * p<0.10");
        println!("{thick}\n");
        Ok(diagnostics)
    }

    fn fe_coefficients(&self, fe: &greeners::panel::PanelResult) -> Result<Option<Value>> {
        let k = fe.params.len();
        let names = fe.variable_names.as_deref().unwrap_or(&[]);
        println!("\n── Coefficient Significance");
        println!(
            "   {:<22} {:>10} {:>8} {:>4}",
            "Variable", "coef", "p-value", ""
        );
        println!("   {}", "─".repeat(48));
        let mut fe_var = Vec::new();
        let mut fe_coef = Vec::new();
        let mut fe_p = Vec::new();
        let mut fe_sig = Vec::new();
        for i in 0..k {
            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
            let s = Self::sig_stars(fe.p_values[i]);
            println!(
                "   {:<22} {:>10.4} {:>8.4} {:>4}",
                name, fe.params[i], fe.p_values[i], s
            );
            fe_var.push(Value::Str(name.to_string()));
            fe_coef.push(Value::Float(fe.params[i]));
            fe_p.push(Value::Float(fe.p_values[i]));
            fe_sig.push(Value::Str(s.to_string()));
        }
        if fe_var.is_empty() {
            return Ok(None);
        }
        let mut fe_columns = HashMap::new();
        fe_columns.insert("variable".into(), Value::List(Arc::new(fe_var)));
        fe_columns.insert("coef".into(), Value::List(Arc::new(fe_coef)));
        fe_columns.insert("p_value".into(), Value::List(Arc::new(fe_p)));
        fe_columns.insert("significance".into(), Value::List(Arc::new(fe_sig)));
        let fe_df = self.dict_to_dataframe(&fe_columns)?;
        Ok(Some(Value::DataFrame(Arc::new(fe_df))))
    }

    pub(super) fn diagnostics_re(
        &mut self,
        re: &greeners::panel::RandomEffectsResult,
        thick: &str,
        thin: &str,
    ) -> Result<HashMap<String, Value>> {
        let mut diagnostics: HashMap<String, Value> = HashMap::new();
        let k = re.params.len();

        // Variance decomposition
        let var_e = re.sigma_e * re.sigma_e; // variance of individual effects
        let var_u = re.sigma_u * re.sigma_u; // idiosyncratic variance
        let var_tot = var_e + var_u;
        let icc = if var_tot > 1e-15 {
            var_e / var_tot
        } else {
            0.0
        };

        println!("\n{thick}");
        println!(" DIAGNOSTICS — Random Effects  (k={})", k);
        println!("{thick}");

        diagnostics.insert("model".into(), Value::Str("Random Effects".into()));
        diagnostics.insert("k".into(), Value::Int(k as i64));
        diagnostics.insert("r2_overall".into(), Value::Float(re.r_squared_overall));
        diagnostics.insert("sigma_e".into(), Value::Float(re.sigma_e));
        diagnostics.insert("sigma_u".into(), Value::Float(re.sigma_u));
        diagnostics.insert("var_e".into(), Value::Float(var_e));
        diagnostics.insert("var_u".into(), Value::Float(var_u));
        diagnostics.insert("icc".into(), Value::Float(icc));
        diagnostics.insert("theta".into(), Value::Float(re.theta));

        println!("\n── Fit");
        println!("   Overall R² = {:.4}", re.r_squared_overall);

        println!("\n── Variance Decomposition");
        println!(
            "   σ_e  (individual effects) = {:.6}   σ_e² = {:.6}",
            re.sigma_e, var_e
        );
        println!(
            "   σ_u  (idiosyncratic)     = {:.6}   σ_u² = {:.6}",
            re.sigma_u, var_u
        );
        println!(
            "   ICC  = σ_e²/(σ_e²+σ_u²)   = {:.4}   ({:.1}% of variance is between entities)",
            icc,
            icc * 100.0
        );
        println!(
            "   θ    (GLS weight)            = {:.4}   (0→OLS  1→FE)",
            re.theta
        );

        if let Some(coeffs) = self.re_coefficients(re)? {
            diagnostics.insert("coefficients".into(), coeffs);
        }

        println!("\n── Tests Not Available");
        println!("   • Hausman FE vs RE: use hausman(fe_model, re_model)");
        println!("   • BP LM test (H₀: sem individual effects): σ_e²/σ_u² acima sugere efeitos");
        println!("\n{thin}");
        println!("   *** p<0.01  ** p<0.05  * p<0.10");
        println!("{thick}\n");
        Ok(diagnostics)
    }

    fn re_coefficients(&self, re: &greeners::panel::RandomEffectsResult) -> Result<Option<Value>> {
        let k = re.params.len();
        println!("\n── Coefficient Significance");
        println!(
            "   {:<22} {:>10} {:>8} {:>4}",
            "Variable", "coef", "p-value", ""
        );
        println!("   {}", "─".repeat(48));
        let mut re_var = Vec::new();
        let mut re_coef = Vec::new();
        let mut re_p = Vec::new();
        let mut re_sig = Vec::new();
        for i in 0..k {
            let name = re
                .variable_names
                .as_ref()
                .and_then(|v| v.get(i))
                .map(|s| s.as_str())
                .unwrap_or("const");
            let s = Self::sig_stars(re.p_values[i]);
            println!(
                "   {:<22} {:>10.4} {:>8.4} {:>4}",
                name, re.params[i], re.p_values[i], s
            );
            re_var.push(Value::Str(name.to_string()));
            re_coef.push(Value::Float(re.params[i]));
            re_p.push(Value::Float(re.p_values[i]));
            re_sig.push(Value::Str(s.to_string()));
        }
        if re_var.is_empty() {
            return Ok(None);
        }
        let mut re_columns = HashMap::new();
        re_columns.insert("variable".into(), Value::List(Arc::new(re_var)));
        re_columns.insert("coef".into(), Value::List(Arc::new(re_coef)));
        re_columns.insert("p_value".into(), Value::List(Arc::new(re_p)));
        re_columns.insert("significance".into(), Value::List(Arc::new(re_sig)));
        let re_df = self.dict_to_dataframe(&re_columns)?;
        Ok(Some(Value::DataFrame(Arc::new(re_df))))
    }

    pub(super) fn varma(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            let col = get_col_f64(&df, vname)?;
            for (i, &v) in col.iter().enumerate() {
                data[[i, j]] = v;
            }
        }
        let result =
            greeners::VARMA::fit(&data, p, q).map_err(|e| self.rt_err(format!("VARMA: {e}")))?;
        Ok(Value::VarmaResult(Rc::new(result)))
    }

    pub(super) fn decompose(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let series = ndarray::Array1::from(get_col_f64(&df, &var_name)?);
        let period = match opt_map.get("period") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 12,
        };
        let model_str = match opt_map.get("model") {
            Some(Value::Str(s)) => s.as_str(),
            _ => "additive",
        };
        let result = greeners::Decomposition::seasonal_decompose(&series, period, model_str)
            .map_err(|e| self.rt_err(format!("decompose: {e}")))?;
        Ok(Value::DecompResult(Rc::new(result)))
    }

    pub(super) fn stl(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let series = ndarray::Array1::from(get_col_f64(&df, &var_name)?);
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

    pub(super) fn mstl(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let series = ndarray::Array1::from(get_col_f64(&df, &var_name)?);
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

    pub(super) fn proptest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        println!("\nProportion Test (1 sample)");
        println!("{sep}");
        println!("  H₀: p = {mu:.4}");
        println!("  p̂ = {p_hat:.4}  (count={count}, n={n})");
        println!("{sep}");
        println!(
            "{:<26} {:>10} {:>10} {:>4}",
            "Test", "Statistic", "p-value", ""
        );
        println!("{sep}");
        println!("{:<26} {:>10.4} {:>10.4} {:>4}", "z", z, p, sig(p));
        println!("{sep}");
        println!("(*** p<0.01  ** p<0.05  * p<0.10)");
        println!();
        Ok(Value::Nil)
    }

    pub(super) fn proptest2(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        println!("\nProportion Test (2 samples)");
        println!("{sep}");
        println!("  H₀: p₁ = p₂");
        println!("  p̂₁ = {p1:.4}  (count={c1}, n={n1})");
        println!("  p̂₂ = {p2:.4}  (count={c2}, n={n2})");
        println!("{sep}");
        println!(
            "{:<26} {:>10} {:>10} {:>4}",
            "Test", "Statistic", "p-value", ""
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

    pub(super) fn propci(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        println!("\nProportion CI — Wilson Score ({pct:.0}%)");
        println!("{sep}");
        println!("  p̂ = {p_hat:.4}  (count={count}, n={n})");
        println!("  CI [{pct:.0}%]: [{lo:.4}, {hi:.4}]");
        println!("{sep}");
        println!();
        Ok(Value::Nil)
    }

    pub(super) fn chisq2x2(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 4 {
            return Err(HayashiError::Runtime("chisq2x2(a, b, c, d)".into()));
        }
        let to_usize = |v: Value| -> Result<usize> {
            match v {
                Value::Int(i) => Ok(i as usize),
                Value::Float(f) => Ok(f as usize),
                _ => Err(HayashiError::Type("table cells must be integers".into())),
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
        println!("\nChi-Square Test — 2×2 Table");
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
            "Test", "Statistic", "p-value", ""
        );
        println!("{sep}");
        println!("{:<26} {:>10.4} {:>10.4} {:>4}", "χ²(1)", chi2, p, sig(p));
        println!("{sep}");
        println!("(*** p<0.01  ** p<0.05  * p<0.10)");
        println!();
        Ok(Value::Nil)
    }

    pub(super) fn multipletests(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                        "unknown method: '{other}' — use bonferroni, sidak, holm, bh, by"
                    )))
                }
            },
            _ => greeners::MultiTestMethod::Bonferroni,
        };
        let method_name = format!("{:?}", method);
        let (rejects, pvals_adj) = greeners::MultipleTests::multipletests(&pvals, alpha, method)
            .map_err(|e| self.rt_err(format!("multipletests: {e}")))?;
        let sep = "─".repeat(64);
        println!("\nMultiple Tests — {method_name}  (α={alpha})");
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
            let mark = if *rej { "  YES ***" } else { "  no" };
            println!("{:>5}  {:>12.6}  {:>12.6}  {}", i + 1, p_orig, p_adj, mark);
        }
        println!("{sep}");
        println!();
        Ok(Value::Nil)
    }

    pub(super) fn ucm(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let y = ndarray::Array1::from(get_col_f64(&df, &var_name)?);

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

    pub(super) fn gam(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_linear) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let linear_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let n = y_vec.len();

        let (smooth_names, spline_df, degree, alpha_pen) =
            self.parse_gam_options(opt_map, &x_linear)?;
        let (x_smooth_ref, alpha_pen_used) =
            self.build_gam_smooth_basis(&df, &smooth_names, n, spline_df, degree, alpha_pen)?;
        let (family, link) = self.parse_gam_family_link(opt_map)?;

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

    fn parse_gam_options(
        &self,
        opt_map: &HashMap<String, Value>,
        x_linear: &ndarray::Array2<f64>,
    ) -> Result<(Vec<String>, usize, usize, f64)> {
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
                "gam: specify linear terms (formula) and/or smooth=".into(),
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

        Ok((smooth_names, spline_df, degree, alpha_pen))
    }

    fn build_gam_smooth_basis(
        &self,
        df: &greeners::DataFrame,
        smooth_names: &[String],
        n: usize,
        spline_df: usize,
        degree: usize,
        alpha_pen: f64,
    ) -> Result<(ndarray::Array2<f64>, f64)> {
        let q_per = spline_df;
        let q_total = q_per * smooth_names.len().max(1);
        let mut x_smooth = ndarray::Array2::<f64>::zeros((n, q_total));
        for (k, sname) in smooth_names.iter().enumerate() {
            let col = ndarray::Array1::from(get_col_f64(df, sname)?);
            let basis = greeners::BSplineBasis::generate(&col, q_per, degree)
                .map_err(|e| self.rt_err(format!("gam spline ({sname}): {e}")))?;
            for i in 0..n {
                for j in 0..q_per {
                    x_smooth[[i, k * q_per + j]] = basis[[i, j]];
                }
            }
        }
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
        Ok((x_smooth_ref, alpha_pen_used))
    }

    fn parse_gam_family_link(
        &self,
        opt_map: &HashMap<String, Value>,
    ) -> Result<(greeners::Family, greeners::Link)> {
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
                "gaussian" | "normal" => greeners::Family::Gaussian,
                "binomial" | "logistic" => greeners::Family::Binomial,
                "poisson" => greeners::Family::Poisson,
                "gamma" => greeners::Family::Gamma,
                "inverse_gaussian" => greeners::Family::InverseGaussian,
                "negbin" => greeners::Family::NegativeBinomial(alpha_val),
                "tweedie" => greeners::Family::Tweedie(power_val),
                other => {
                    return Err(HayashiError::Runtime(format!(
                "gam: family='{other}' unknown — use: gaussian, binomial, poisson, gamma, negbin"
            )))
                }
            },
            _ => greeners::Family::Gaussian,
        };
        let link = match opt_map.get("link") {
            Some(Value::Str(s)) => match s.as_str() {
                "identity" => greeners::Link::Identity,
                "log" => greeners::Link::Log,
                "logit" => greeners::Link::Logit,
                "probit" => greeners::Link::Probit,
                "inverse" => greeners::Link::InversePower,
                "cloglog" => greeners::Link::CLogLog,
                other => {
                    return Err(HayashiError::Runtime(format!(
                "gam: link='{other}' unknown — use: identity, log, logit, probit, inverse, cloglog"
            )))
                }
            },
            _ => greeners::Link::Identity,
        };
        Ok((family, link))
    }

    pub(super) fn mice(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                    _ => Err(HayashiError::Type("vars= must be a list of strings".into())),
                })
                .collect::<Result<_>>()?,
            Some(Value::Str(s)) => vec![s.clone()],
            None => {
                if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    return Err(HayashiError::Runtime(
                        "mice: specify vars=[\"x1\",\"x2\",...] or list variables after df".into(),
                    ));
                }
            }
            _ => return Err(HayashiError::Type("vars= must be a list of strings".into())),
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

        let mut data: indexmap::IndexMap<String, ndarray::Array1<f64>> = indexmap::IndexMap::new();
        for vname in &var_names {
            data.insert(
                vname.clone(),
                ndarray::Array1::from(get_col_f64(&df, vname)?),
            );
        }

        let result = greeners::MICE::impute(&data, m, iter)
            .map_err(|e| self.rt_err(format!("mice: {e}")))?;
        println!("{result}");
        Ok(Value::MiceResult(Rc::new(result)))
    }

    pub(super) fn msauto(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let y = ndarray::Array1::from(get_col_f64(&df, &var_name)?);
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

    pub(super) fn svar(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            let col = get_col_f64(&df, vname)?;
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

    pub(super) fn sirf(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            "\nSVAR Structural IRF — VAR({}) — id: {} — {} passos",
            model.var_result.lags, model.identification, steps
        );
        let mut h_vec = Vec::new();
        let mut impulse_vec = Vec::new();
        let mut response_vec = Vec::new();
        let mut irf_vec = Vec::new();
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
                    .map(|i| {
                        h_vec.push(h as i64);
                        impulse_vec.push(Value::Str(names[j].clone()));
                        response_vec.push(Value::Str(names[i].clone()));
                        irf_vec.push(Value::Float(tensor[[h, i, j]]));
                        format!("{:>12.4}", tensor[[h, i, j]])
                    })
                    .collect::<Vec<_>>()
                    .join("");
                println!("  {:>6}  {row}", h);
            }
        }
        println!();
        let mut columns = HashMap::new();
        columns.insert(
            "h".into(),
            Value::List(Arc::new(h_vec.into_iter().map(Value::Int).collect())),
        );
        columns.insert("impulse".into(), Value::List(Arc::new(impulse_vec)));
        columns.insert("response".into(), Value::List(Arc::new(response_vec)));
        columns.insert("sirf".into(), Value::List(Arc::new(irf_vec)));
        let df = self.dict_to_dataframe(&columns)?;
        Ok(Value::DataFrame(Arc::new(df)))
    }

    pub(super) fn sfevd(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            "\nSVAR Structural FEVD — VAR({}) — id: {}",
            model.var_result.lags, model.identification
        );
        let mut h_vec = Vec::new();
        let mut response_vec = Vec::new();
        let mut source_vec = Vec::new();
        let mut fevd_vec = Vec::new();
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
                    .map(|j| {
                        h_vec.push(h as i64);
                        response_vec.push(Value::Str(names[i].clone()));
                        source_vec.push(Value::Str(names[j].clone()));
                        fevd_vec.push(Value::Float(tensor[[h, i, j]]));
                        format!("{:>12.4}", tensor[[h, i, j]])
                    })
                    .collect::<Vec<_>>()
                    .join("");
                println!("  {:>6}  {row}", h);
            }
        }
        println!();
        let mut columns = HashMap::new();
        columns.insert(
            "h".into(),
            Value::List(Arc::new(h_vec.into_iter().map(Value::Int).collect())),
        );
        columns.insert("response".into(), Value::List(Arc::new(response_vec)));
        columns.insert("source".into(), Value::List(Arc::new(source_vec)));
        columns.insert("sfevd".into(), Value::List(Arc::new(fevd_vec)));
        let df = self.dict_to_dataframe(&columns)?;
        Ok(Value::DataFrame(Arc::new(df)))
    }

    pub(super) fn threesl(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
            Some(Value::List(lst)) => lst
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(HayashiError::Type(
                        "instruments= must be a list of strings".into(),
                    )),
                })
                .collect::<Result<_>>()?,
            Some(Value::Str(s)) => vec![s.clone()],
            None => return Err(HayashiError::Runtime(
                "threesl requires instruments=[\"z1\",\"z2\",...] — list of exogenous variables"
                    .into(),
            )),
            _ => {
                return Err(HayashiError::Type(
                    "instruments= must be a list of strings".into(),
                ))
            }
        };

        // Build global instrument matrix Z (n × q)
        let n = df.n_rows();
        let mut z_instr = ndarray::Array2::<f64>::zeros((n, instr_names.len()));
        for (j, zname) in instr_names.iter().enumerate() {
            let col = get_col_f64(&df, zname)?;
            for (i, &v) in col.iter().enumerate() {
                z_instr[[i, j]] = v;
            }
        }

        // Build equations from formulas
        let mut equations: Vec<greeners::Equation> = Vec::new();
        let mut eq_var_names: Vec<Vec<String>> = Vec::new();
        for arg in &args[1..] {
            let formula_ast = self.resolve_formula(arg)?;
            let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
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

    pub(super) fn dfm(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
                    "dfm() variables must be identifiers".into(),
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
            let col = get_col_f64(&df, vname)?;
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

    pub(super) fn adtest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let data = get_col_f64(&df, &var_name)?;
        let r = greeners::Diagnostics::anderson_darling(&ndarray::Array1::from(data))
            .map_err(|e| self.rt_err(format!("adtest: {e}")))?;
        let sep = "─".repeat(56);
        println!("\nAnderson-Darling Test (normality)");
        println!("{sep}");
        println!("  H₀: data come from normal distribution");
        println!("  A² (adjusted) = {:.4}  (n={})", r.statistic, r.n_obs);
        println!("{sep}");
        println!("{:<12} {:>10}", "α", "A²*_critical");
        println!("{sep}");
        for (&sig, &cv) in r.significance_levels.iter().zip(r.critical_values.iter()) {
            let mark = if r.statistic > cv { " ← REJECT" } else { "" };
            println!("{:<12.3} {:>10.3}{mark}", sig, cv);
        }
        println!("{sep}");
        println!("(Reject H₀ when A²* > critical value)");
        println!();
        Ok(Value::Nil)
    }

    pub(super) fn lilliefors(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let data = get_col_f64(&df, &var_name)?;
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
        println!("\nLilliefors Test (normality — KS with estimated parameters)");
        println!("{sep}");
        println!("  H₀: data come from normal distribution");
        println!("{sep}");
        println!(
            "{:<26} {:>10} {:>10} {:>4}",
            "Test", "Statistic", "p-value", ""
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

    pub(super) fn omnibus(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        println!("\nD'Agostino-Pearson Omnibus Test (normality of residuals)");
        println!("{sep}");
        println!("  H₀: residuals are normally distributed");
        println!("  (combines skewness and kurtosis via K² ~ χ²(2))");
        println!("{sep}");
        println!(
            "{:<26} {:>10} {:>10} {:>4}",
            "Test", "Statistic", "p-value", ""
        );
        println!("{sep}");
        println!("{:<26} {:>10.4} {:>10.4} {:>4}", "K² ~ χ²(2)", k2, p, sig);
        println!("{sep}");
        println!("(*** p<0.01  ** p<0.05  * p<0.10)");
        println!();
        Ok(Value::Nil)
    }

    pub(super) fn swilk(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let data = get_col_f64(&df, &var_name)?;
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

    pub(super) fn sfrancia(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let data = get_col_f64(&df, &var_name)?;
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

    pub(super) fn sktest(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        let data = get_col_f64(&df, &var_name)?;
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

    pub(super) fn harveycollier(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
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
        // reconstruct y = ŷ + residuals (OlsModel does not store y directly)
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
        println!("\nHarvey-Collier Test (linearity of specification)");
        println!("{sep}");
        println!("  H₀: functional specification is correct (linear)");
        println!("  (tests whether mean of recursive residuals is zero)");
        println!("{sep}");
        println!(
            "{:<26} {:>10} {:>10} {:>4}",
            "Test", "Statistic", "p-value", ""
        );
        println!("{sep}");
        println!("{:<26} {:>10.4} {:>10.4} {:>4}", "t (HC)", t, p, sig);
        println!("{sep}");
        println!("(*** p<0.01  ** p<0.05  * p<0.10)");
        println!();
        Ok(Value::Nil)
    }
}
