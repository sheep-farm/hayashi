use super::*;

impl Interpreter {
    /// `vecm(df, y1, y2, ..., lags=2, rank=1)` — Vector Error Correction.
    pub(super) fn eval_vecm(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "vecm() requires arguments: dataframe, var1, var2, ..., lags=p, rank=r".into(),
            ));
        }

        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "primeiro argumento deve ser um DataFrame".into(),
                ))
            }
        };

        let var_names = self.resolve_var_list(&args[1..], &df)?;

        let lags = match opt_map.get("lags") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 2,
        };
        let rank = match opt_map.get("rank") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 1,
        };

        // build T×k matrix
        let n = df.n_rows();
        let k = var_names.len();
        let mut data = ndarray::Array2::<f64>::zeros((n, k));
        for (j, vname) in var_names.iter().enumerate() {
            let col = self.eval_col_expr(&Expr::Var(vname.clone()), &df)?;
            for (i, &v) in col.iter().enumerate() {
                data[[i, j]] = v;
            }
        }

        let result = greeners::VECM::fit(&data, lags, rank)
            .map_err(|e| self.rt_err(format!("VECM: {e}")))?
            .with_inference(200)
            .map_err(|e| self.rt_err(format!("VECM inference: {e}")))?;

        Ok(Value::VecmResult(Rc::new(result)))
    }

    /// `var(df, y1, y2, ..., lags=2)` — Vetor Auto-regressivo.
    pub(super) fn eval_var(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "var() requires arguments: dataframe, var1, var2, ..., lags=p".into(),
            ));
        }

        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "primeiro argumento deve ser um DataFrame".into(),
                ))
            }
        };

        let var_names = self.resolve_var_list(&args[1..], &df)?;

        let lags = match opt_map.get("lags") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 1,
        };

        // build T×k matrix
        let n = df.n_rows();
        let k = var_names.len();
        let mut data = ndarray::Array2::<f64>::zeros((n, k));
        for (j, vname) in var_names.iter().enumerate() {
            let col = self.eval_col_expr(&Expr::Var(vname.clone()), &df)?;
            for (i, &v) in col.iter().enumerate() {
                data[[i, j]] = v;
            }
        }

        let result = greeners::VAR::fit(&data, lags, Some(var_names))
            .map_err(|e| self.rt_err(format!("VAR: {e}")))?;

        Ok(Value::VarResult(Rc::new(result)))
    }

    /// `irf(model, steps=10)` — Impulse Response Function.
    pub(super) fn eval_irf(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime("irf() requires a VAR model".into()));
        }
        let model = match self.eval_expr(&args[0])? {
            Value::VarResult(m) => m,
            _ => return Err(HayashiError::Type("irf() requires a VAR model".into())),
        };

        let steps = match opt_map.get("steps") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 10,
        };

        let tensor = model
            .irf(steps)
            .map_err(|e| self.rt_err(format!("IRF: {e}")))?;

        let k = model.n_vars;
        let names = &model.var_names;
        let sep = "─".repeat(14 + k * 12);

        println!("\nIRF — VAR({}) — {} passos", model.lags, steps);

        for j in 0..k {
            println!("\n  Impulso: {}", names[j]);
            println!("  {sep}");
            let header: String = names
                .iter()
                .map(|n| format!("{:>12}", n))
                .collect::<Vec<_>>()
                .join("");
            println!("  {:>6}{header}", "h");
            println!("  {sep}");
            for h in 0..steps {
                let row: String = (0..k)
                    .map(|i| format!("{:>12.4}", tensor[[h, i, j]]))
                    .collect::<Vec<_>>()
                    .join("");
                println!("  {:>6}{row}", h + 1);
            }
            println!("  {sep}");
        }
        println!();

        Ok(Value::Nil)
    }

    /// `fevd(model, steps=10)` — Forecast Error Variance Decomposition.
    pub(super) fn eval_fevd(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(HayashiError::Runtime("fevd() requires a VAR model".into()));
        }
        let model = match self.eval_expr(&args[0])? {
            Value::VarResult(m) => m,
            _ => return Err(HayashiError::Type("fevd() requires a VAR model".into())),
        };

        let steps = match opt_map.get("steps") {
            Some(Value::Int(v)) => *v as usize,
            Some(Value::Float(v)) => *v as usize,
            _ => 10,
        };

        let tensor = model
            .fevd(steps)
            .map_err(|e| self.rt_err(format!("FEVD: {e}")))?;

        let k = model.n_vars;
        let names = &model.var_names;
        let col_w = names.iter().map(|n| n.len()).max().unwrap_or(8).max(8) + 2;
        let sep = "─".repeat(8 + k * col_w);

        println!(
            "\nFEVD — VAR({}) — {} steps  (% of forecast error variance)",
            model.lags, steps
        );

        for i in 0..k {
            println!("\n  Variable: {}", names[i]);
            println!("  {sep}");
            let header: String = names
                .iter()
                .map(|n| format!("{:>col_w$}", n))
                .collect::<Vec<_>>()
                .join("");
            println!("  {:>6}{header}", "h");
            println!("  {sep}");
            for h in 0..steps {
                let row: String = (0..k)
                    .map(|j| format!("{:>col_w$.1}%", tensor[[h, i, j]] * 100.0))
                    .collect::<Vec<_>>()
                    .join("");
                println!("  {:>6}{row}", h + 1);
            }
            println!("  {sep}");
        }
        println!();

        Ok(Value::Nil)
    }

    /// `arima` / `sarima(df, varname, p=1, d=1, q=1, ...)`.
    pub(super) fn eval_arima(
        &mut self,
        func: &str,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "arima() requires arguments: dataframe, variable, p=, d=, q=".into(),
            ));
        }

        let df = match self.eval_expr(&args[0])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "primeiro argumento deve ser um DataFrame".into(),
                ))
            }
        };

        let col_name = match &args[1] {
            Expr::Var(n) | Expr::Str(n) => n.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "second argument must be o variable name".into(),
                ))
            }
        };

        // extract series as Array1<f64>
        let y = self.eval_col_expr(&Expr::Var(col_name.clone()), &df)?;
        let y = ndarray::Array1::from(y);

        // opts: p, d, q (ARIMA); P, D, Q, s (SARIMA); method ("hr" | "mle")
        let get_usize = |key: &str, default: usize| -> usize {
            match opt_map.get(key) {
                Some(Value::Int(v)) => *v as usize,
                Some(Value::Float(v)) => *v as usize,
                _ => default,
            }
        };

        let p = get_usize("p", 1);
        let d = get_usize("d", 1);
        let q = get_usize("q", 1);

        let method = match opt_map.get("method") {
            Some(Value::Str(s)) => s.clone(),
            _ => "hr".to_string(),
        };

        let result = if func == "sarima" {
            let sp = get_usize("P", 0);
            let sd = get_usize("D", 0);
            let sq = get_usize("Q", 0);
            let s = get_usize("s", 12);
            if method == "mle" && (sp > 0 || sd > 0 || sq > 0) {
                return Err(self.rt_err(
                    "SARIMA MLE is not supported; use method='hr' for seasonal models",
                ));
            }
            if method == "mle" {
                greeners::ARIMA::fit_mle(&y, (p, d, q))
                    .map_err(|e| self.rt_err(format!("ARIMA(MLE): {e}")))?
            } else {
                greeners::ARIMA::fit_sarimax(&y, (p, d, q), (sp, sd, sq, s), None)
                    .map_err(|e| self.rt_err(format!("SARIMA: {e}")))?
            }
        } else {
            if method == "mle" {
                greeners::ARIMA::fit_mle(&y, (p, d, q))
                    .map_err(|e| self.rt_err(format!("ARIMA(MLE): {e}")))?
            } else {
                greeners::ARIMA::fit(&y, (p, d, q))
                    .map_err(|e| self.rt_err(format!("ARIMA: {e}")))?
            }
        };

        Ok(Value::ArimaResult(Rc::new(result)))
    }
}
