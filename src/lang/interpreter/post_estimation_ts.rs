use super::helpers::*;
use super::*;
use crate::lang::dap::model_expansion;
use std::sync::Arc;

#[allow(unused_imports)]
use super::models::BinaryModel;
#[allow(unused_imports)]
use super::models::OlsModel;

#[cfg(feature = "greeners-timeseries")]
mod timeseries_models;

/// margins, VECM/VAR/IRF/FEVD, ARIMA/SARIMA/AutoReg/ARDL/Kalman/forecast,
/// lincom/nlcom. Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    /// Identifies explicit coefficient dependence in ordinary scalar expressions.
    /// Complex expressions and calls that can capture state remain conservative.
    fn nlcom_depends_on(expression: &Expr, name: &str) -> bool {
        match expression {
            Expr::Var(variable) => variable == name,
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Nil => false,
            Expr::Neg(inner) | Expr::Not(inner) => Self::nlcom_depends_on(inner, name),
            Expr::BinOp { lhs, rhs, .. } => {
                Self::nlcom_depends_on(lhs, name) || Self::nlcom_depends_on(rhs, name)
            }
            // Use the interpreter's existing scalar-transform recognition,
            // which precedes user-function lookup, rather than a new name list.
            Expr::Call { func, args, opts }
                if opts.is_empty()
                    && ((args.len() == 1
                        && greeners::transforms::Transforms::apply(&[1.0], func).is_ok())
                        || (args.len() == 2
                            && greeners::transforms::Transforms::apply2(&[1.0], &[1.0], func)
                                .is_ok())) =>
            {
                args.iter()
                    .any(|argument| Self::nlcom_depends_on(argument, name))
                    || opts
                        .iter()
                        .any(|option| Self::nlcom_depends_on(&option.value, name))
            }
            Expr::If {
                cond,
                then_expr,
                else_expr,
            } => {
                Self::nlcom_depends_on(cond, name)
                    || Self::nlcom_depends_on(then_expr, name)
                    || Self::nlcom_depends_on(else_expr, name)
            }
            _ => true,
        }
    }

    /// Evaluates a finite scalar nonlinear contrast in the temporary coefficient scope.
    fn nlcom_number(&mut self, expression: &Expr) -> Result<f64> {
        let value = match self.eval_expr(expression)? {
            Value::Float(value) => value,
            Value::Int(value) => value as f64,
            _ => {
                return Err(HayashiError::Type(
                    "nlcom: expression must evaluate to a number".into(),
                ))
            }
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(self.rt_err("nlcom: expression is outside its finite numerical domain"))
        }
    }

    /// Differentiates in coefficient units, checking domain and step convergence.
    fn nlcom_derivative(
        &mut self,
        expression: &Expr,
        name: &str,
        parameter: f64,
        coordinate_se: f64,
        centre: f64,
    ) -> Result<f64> {
        if !Self::nlcom_depends_on(expression, name) {
            return Ok(0.0);
        }
        if !parameter.is_finite() || !coordinate_se.is_finite() || coordinate_se < 0.0 {
            return Err(self.rt_err(format!("nlcom: invalid coefficient scale for '{name}'")));
        }
        let scale = parameter.abs().max(coordinate_se);
        // A zero covariance coordinate contributes no delta-method variance.
        if scale == 0.0 {
            return Ok(0.0);
        }
        let mut step = f64::EPSILON.cbrt() * scale;
        let mut previous_difference = None;
        let mut previous_extrapolation: Option<f64> = None;
        let mut previous_gap = None;
        for _ in 0..20 {
            let plus = parameter + step;
            let minus = parameter - step;
            if !plus.is_finite() || !minus.is_finite() || plus == parameter || minus == parameter {
                break;
            }
            self.env.set(name, Value::Float(plus))?;
            let plus_value = self.nlcom_number(expression);
            self.env.set(name, Value::Float(minus))?;
            let minus_value = self.nlcom_number(expression);
            let (Ok(plus_value), Ok(minus_value)) = (plus_value, minus_value) else {
                previous_difference = None;
                previous_extrapolation = None;
                previous_gap = None;
                step *= 0.5;
                continue;
            };
            if plus_value == centre && minus_value == centre {
                return Err(self.rt_err(format!(
                    "nlcom: derivative for '{name}' is unresolved at expression precision"
                )));
            }
            let span = plus - minus;
            let difference = (plus_value - minus_value) / span;
            let gap = ((plus_value - centre) / (plus - parameter)
                - (centre - minus_value) / (parameter - minus))
                .abs();
            // This budget measures subtraction roundoff in derivative units.
            let roundoff =
                8.0 * f64::EPSILON * (plus_value.abs() + minus_value.abs() + 2.0 * centre.abs())
                    / span;
            if !difference.is_finite() || !gap.is_finite() || !roundoff.is_finite() {
                break;
            }
            if let Some(previous) = previous_difference {
                let extrapolation: f64 = difference + (difference - previous) / 3.0;
                if let Some(previous_extrapolation) = previous_extrapolation {
                    let tolerance = f64::EPSILON.sqrt()
                        * extrapolation.abs().max(previous_extrapolation.abs())
                        + roundoff;
                    let smooth = gap <= tolerance
                        || previous_gap.is_some_and(|previous| gap <= 0.75 * previous + tolerance);
                    if (extrapolation - previous_extrapolation).abs() <= tolerance && smooth {
                        return Ok(extrapolation);
                    }
                }
                previous_extrapolation = Some(extrapolation);
            }
            previous_difference = Some(difference);
            previous_gap = Some(gap);
            step *= 0.5;
        }
        Err(self.rt_err(format!(
            "nlcom: derivative for '{name}' is unresolved in its domain or numerical precision"
        )))
    }

    pub(super) fn eval_call_post_estimation_ts(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── margins ──────────────────────────────────────────────────────
            "margins" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "margins() requires an estimated model as an argument".into(),
                    ));
                }
                let model = self.eval_expr(&args[0])?;

                // dydx=[X1, X2] — which variables to show (lazy, column names)
                let dydx_filter: Option<Vec<String>> =
                    opts.iter()
                        .find(|o| o.name == "dydx")
                        .map(|o| match &o.value {
                            Expr::List(items) => items
                                .iter()
                                .filter_map(|e| match e {
                                    Expr::Var(n) | Expr::Str(n) => Some(n.clone()),
                                    _ => None,
                                })
                                .collect(),
                            Expr::Var(n) | Expr::Str(n) => vec![n.clone()],
                            _ => vec![],
                        });
                let show_var = |name: &str| -> bool {
                    match &dydx_filter {
                        None => name != "_cons" && name != "const",
                        Some(list) => list.iter().any(|s| s == name),
                    }
                };

                // at_X=value — fixes variable X at the given value for margins calculation
                let at_vals: HashMap<String, f64> = opt_map
                    .iter()
                    .filter_map(|(k, v)| {
                        let var = k.strip_prefix("at_")?.to_string();
                        match v {
                            Value::Float(f) => Some((var, *f)),
                            Value::Int(i) => Some((var, *i as f64)),
                            _ => None,
                        }
                    })
                    .collect();

                let sep = "─".repeat(60);
                let sep2 = "═".repeat(60);

                let mut var_vec: Vec<Value> = Vec::new();
                let mut dy_dx_vec: Vec<Value> = Vec::new();
                let mut se_vec: Vec<Value> = Vec::new();
                let mut z_vec: Vec<Value> = Vec::new();
                let mut p_vec: Vec<Value> = Vec::new();
                let mut cat_vec: Vec<Value> = Vec::new();

                match model {
                    // ── Logit / Probit ────────────────────────────────────────
                    #[cfg(feature = "greeners-glm")]
                    Value::Model(m) => {
                        let bm = m.as_any().downcast_ref::<BinaryModel>().ok_or_else(|| {
                            HayashiError::Type("margins: expected binary model".into())
                        })?;
                        let mut x_use = bm.x.clone();
                        for (var, val) in &at_vals {
                            if let Some(idx) = bm.coef_names.iter().position(|n| n == var) {
                                x_use = greeners::margins::Margins::with_at(&x_use, idx, *val);
                            }
                        }
                        let vcov = Self::binary_mle_vcov(&bm.kind, &bm.result.params, &bm.y, &bm.x);
                        let mut ame_result = if bm.kind == "logit" {
                            match &vcov {
                                Some(v) => greeners::margins::Margins::ame_logit_with_vcov(
                                    &bm.result.params,
                                    &x_use,
                                    &bm.coef_names,
                                    v,
                                ),
                                None => greeners::margins::Margins::ame_logit(
                                    &bm.result.params,
                                    &x_use,
                                    &bm.coef_names,
                                ),
                            }
                        } else {
                            match &vcov {
                                Some(v) => greeners::margins::Margins::ame_probit_with_vcov(
                                    &bm.result.params,
                                    &x_use,
                                    &bm.coef_names,
                                    v,
                                ),
                                None => greeners::margins::Margins::ame_probit(
                                    &bm.result.params,
                                    &x_use,
                                    &bm.coef_names,
                                ),
                            }
                        };
                        if let Ok(normal_dist) = Normal::new(0.0, 1.0) {
                            for i in 0..ame_result.effects.len() {
                                let se = ame_result.std_errors[i];
                                if se.is_finite() && se > 1e-15 {
                                    let z = ame_result.effects[i] / se;
                                    ame_result.z_values[i] = z;
                                    ame_result.p_values[i] = 2.0 * (1.0 - normal_dist.cdf(z.abs()));
                                }
                            }
                        }
                        let at_label = if at_vals.is_empty() {
                            String::new()
                        } else {
                            format!(
                                "  at({})",
                                at_vals
                                    .iter()
                                    .map(|(k, v)| format!("{k}={v}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        };
                        let has_se = ame_result.std_errors.iter().any(|s| s.is_finite());
                        println!("\n{sep2}");
                        println!(
                            " Average Marginal Effects — {}{at_label}",
                            bm.kind.to_uppercase()
                        );
                        println!("{sep2}");
                        if has_se {
                            println!(
                                "{:<18} {:>10} {:>10} {:>8} {:>8}",
                                "Variable", "dy/dx", "Std.Err.", "z", "P>|z|"
                            );
                        } else {
                            println!("{:<22} {:>14}", "Variable", "dy/dx");
                        }
                        println!("{sep}");
                        for (i, name) in ame_result.variable_names.iter().enumerate() {
                            if !show_var(name) {
                                continue;
                            }
                            if has_se {
                                let sig = if ame_result.p_values[i] < 0.01 {
                                    "***"
                                } else if ame_result.p_values[i] < 0.05 {
                                    "**"
                                } else if ame_result.p_values[i] < 0.10 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!(
                                    "{:<18} {:>10.6} {:>10.6} {:>8.3} {:>8.4} {sig}",
                                    name,
                                    ame_result.effects[i],
                                    ame_result.std_errors[i],
                                    ame_result.z_values[i],
                                    ame_result.p_values[i]
                                );
                            } else {
                                println!("{:<22} {:>14.6}", name, ame_result.effects[i]);
                            }
                            var_vec.push(Value::Str(name.clone()));
                            dy_dx_vec.push(Value::Float(ame_result.effects[i]));
                            se_vec.push(Value::Float(if has_se {
                                ame_result.std_errors[i]
                            } else {
                                f64::NAN
                            }));
                            z_vec.push(Value::Float(if has_se {
                                ame_result.z_values[i]
                            } else {
                                f64::NAN
                            }));
                            p_vec.push(Value::Float(if has_se {
                                ame_result.p_values[i]
                            } else {
                                f64::NAN
                            }));
                            cat_vec.push(Value::Str("".into()));
                        }
                        println!("{sep}");
                        println!("n = {}", ame_result.n_obs);
                        println!("{sep2}\n");
                    }

                    // ── Poisson / NegBin ──────────────────────────────────────
                    #[cfg(feature = "greeners-glm")]
                    Value::PoissonResult(r) => {
                        let x = r.x_data();
                        let fb: Vec<String> =
                            (0..r.params.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        let mut x_use = x.to_owned();
                        for (var, val) in &at_vals {
                            if let Some(idx) = names.iter().position(|n| n == var) {
                                x_use = greeners::margins::Margins::with_at(&x_use, idx, *val);
                            }
                        }
                        let ame_result =
                            greeners::margins::Margins::ame_exponential(&r.params, &x_use, names);
                        let at_label = if at_vals.is_empty() {
                            String::new()
                        } else {
                            format!(
                                "  at({})",
                                at_vals
                                    .iter()
                                    .map(|(k, v)| format!("{k}={v}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        };
                        println!("\n{sep2}");
                        println!(" Average Marginal Effects — POISSON{at_label}  (dy/dx = β·μ̄)");
                        println!("{sep2}");
                        println!("{:<22} {:>14}", "Variable", "dy/dx");
                        println!("{sep}");
                        for (k_idx, name) in ame_result.variable_names.iter().enumerate() {
                            if !show_var(name) {
                                continue;
                            }
                            let ame = ame_result.effects[k_idx];
                            println!("{:<22} {:>14.6}", name, ame);
                            var_vec.push(Value::Str(name.clone()));
                            dy_dx_vec.push(Value::Float(ame));
                            se_vec.push(Value::Float(f64::NAN));
                            z_vec.push(Value::Float(f64::NAN));
                            p_vec.push(Value::Float(f64::NAN));
                            cat_vec.push(Value::Str("".into()));
                        }
                        println!("{sep}");
                        println!("n = {}", ame_result.n_obs);
                        println!("{sep2}\n");
                    }
                    #[cfg(feature = "greeners-glm")]
                    Value::NegBinResult(r) => {
                        let x = r.x_data();
                        let fb: Vec<String> =
                            (0..r.params.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        let mut x_use = x.to_owned();
                        for (var, val) in &at_vals {
                            if let Some(idx) = names.iter().position(|n| n == var) {
                                x_use = greeners::margins::Margins::with_at(&x_use, idx, *val);
                            }
                        }
                        let ame_result =
                            greeners::margins::Margins::ame_exponential(&r.params, &x_use, names);
                        let at_label = if at_vals.is_empty() {
                            String::new()
                        } else {
                            format!(
                                "  at({})",
                                at_vals
                                    .iter()
                                    .map(|(k, v)| format!("{k}={v}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        };
                        println!("\n{sep2}");
                        println!(
                            " Average Marginal Effects — NEG. BINOMIAL{at_label}  (dy/dx = β·μ̄)"
                        );
                        println!("{sep2}");
                        println!("{:<22} {:>14}", "Variable", "dy/dx");
                        println!("{sep}");
                        for (k_idx, name) in ame_result.variable_names.iter().enumerate() {
                            if !show_var(name) {
                                continue;
                            }
                            let ame = ame_result.effects[k_idx];
                            println!("{:<22} {:>14.6}", name, ame);
                            var_vec.push(Value::Str(name.clone()));
                            dy_dx_vec.push(Value::Float(ame));
                            se_vec.push(Value::Float(f64::NAN));
                            z_vec.push(Value::Float(f64::NAN));
                            p_vec.push(Value::Float(f64::NAN));
                            cat_vec.push(Value::Str("".into()));
                        }
                        println!("{sep}");
                        println!("n = {}   α = {:.4}", ame_result.n_obs, r.alpha);
                        println!("{sep2}\n");
                    }

                    // ── Ordered Logit / Probit ────────────────────────────────
                    // AME_k(Y=j) = (1/n) Σ_i [f(κ_{j-1} - X_iβ) - f(κ_j - X_iβ)] * β_k
                    // (com κ_0 = -∞ → f(κ_0 - ·) = 0;  κ_J = +∞ → f(κ_J - ·) = 0)
                    #[cfg(feature = "greeners-glm")]
                    Value::OrderedResult(r) => {
                        let x = r.x_data();
                        let n = x.nrows();
                        let beta = &r.params;
                        let cuts = &r.thresholds;
                        let j = r.n_categories;
                        let is_logit = r.model_name.to_lowercase().contains("logit");
                        let link_pdf = |u: f64| -> f64 {
                            if is_logit {
                                let p = logistic(u);
                                p * (1.0 - p)
                            } else {
                                norm_pdf(u)
                            }
                        };
                        let fb: Vec<String> = (0..beta.len()).map(|i| format!("x{i}")).collect();
                        let names = r.variable_names.as_ref().unwrap_or(&fb);
                        // Xβ for each observation
                        let xb: Vec<f64> = (0..n).map(|i| x.row(i).dot(beta)).collect();
                        // AME[var_k, cat_j]
                        let k = beta.len();
                        println!("\n{sep2}");
                        println!(
                            " Average Marginal Effects — {}",
                            r.model_name.to_uppercase()
                        );
                        println!(" dP(Y=j)/dx — one panel per category");
                        println!("{sep2}");
                        // header
                        print!("{:<22}", "Variable");
                        for cat_j in 0..j {
                            print!("  {:>10}", format!("P(Y={})", cat_j + 1));
                        }
                        println!();
                        println!("{sep}");
                        for k_idx in 0..k {
                            let name = names.get(k_idx).map(String::as_str).unwrap_or("?");
                            if name == "_cons" || name == "const" {
                                continue;
                            }
                            print!("{:<22}", name);
                            for cat_j in 0..j {
                                // f(κ_{j-1} - Xβ) — zero para cat_j=0 (sem threshold inferior)
                                let f_lo: f64 = if cat_j == 0 {
                                    0.0
                                } else {
                                    (0..n)
                                        .map(|i| link_pdf(cuts[cat_j - 1] - xb[i]))
                                        .sum::<f64>()
                                        / n as f64
                                };
                                // f(κ_j - Xβ) — zero para cat_j=J-1 (sem threshold superior)
                                let f_hi: f64 = if cat_j == j - 1 {
                                    0.0
                                } else {
                                    (0..n).map(|i| link_pdf(cuts[cat_j] - xb[i])).sum::<f64>()
                                        / n as f64
                                };
                                let ame = (f_lo - f_hi) * beta[k_idx];
                                print!("  {:>10.5}", ame);
                                var_vec.push(Value::Str(name.to_string()));
                                dy_dx_vec.push(Value::Float(ame));
                                se_vec.push(Value::Float(f64::NAN));
                                z_vec.push(Value::Float(f64::NAN));
                                p_vec.push(Value::Float(f64::NAN));
                                cat_vec.push(Value::Str(format!("P(Y={})", cat_j + 1)));
                            }
                            println!();
                        }
                        println!("{sep}");
                        println!("n = {n}   Categories: {j}   Model: {}", r.model_name);
                        println!("{sep2}\n");
                    }

                    _ => {
                        return Err(HayashiError::Type(
                            "margins() suporta: logit, probit, poisson, negbin, ologit, oprobit"
                                .into(),
                        ))
                    }
                }
                let mut columns = HashMap::new();
                columns.insert("variable".into(), Value::List(Arc::new(var_vec)));
                columns.insert("dy_dx".into(), Value::List(Arc::new(dy_dx_vec)));
                columns.insert("std_err".into(), Value::List(Arc::new(se_vec)));
                columns.insert("z".into(), Value::List(Arc::new(z_vec)));
                columns.insert("p".into(), Value::List(Arc::new(p_vec)));
                columns.insert("category".into(), Value::List(Arc::new(cat_vec)));
                let df = self.dict_to_dataframe(&columns)?;
                Ok(Value::DataFrame(Arc::new(df)))
            }

            // ── marginsplot ─────────────────────────────────────────────────
            // marginsplot(model [, width=50])
            "marginsplot" | "margins_plot" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "marginsplot(model) requires an estimated model".into(),
                    ));
                }
                let model = self.eval_expr(&args[0])?;
                let width = match opt_map.get("width") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 50,
                };

                match model {
                    #[cfg(feature = "greeners-glm")]
                    Value::Model(m) => {
                        let bm = m.as_any().downcast_ref::<BinaryModel>().ok_or_else(|| {
                            HayashiError::Type("margins: expected binary model".into())
                        })?;
                        let vcov = Self::binary_mle_vcov(&bm.kind, &bm.result.params, &bm.y, &bm.x);
                        let ame = if bm.kind == "logit" {
                            match &vcov {
                                Some(v) => greeners::margins::Margins::ame_logit_with_vcov(
                                    &bm.result.params,
                                    &bm.x,
                                    &bm.coef_names,
                                    v,
                                ),
                                None => greeners::margins::Margins::ame_logit(
                                    &bm.result.params,
                                    &bm.x,
                                    &bm.coef_names,
                                ),
                            }
                        } else {
                            match &vcov {
                                Some(v) => greeners::margins::Margins::ame_probit_with_vcov(
                                    &bm.result.params,
                                    &bm.x,
                                    &bm.coef_names,
                                    v,
                                ),
                                None => greeners::margins::Margins::ame_probit(
                                    &bm.result.params,
                                    &bm.x,
                                    &bm.coef_names,
                                ),
                            }
                        };

                        // Collect rows (name, effect, ci_lo, ci_hi) excluding constant
                        let z = 1.96_f64;
                        let mut rows: Vec<(String, f64, f64, f64)> = Vec::new();
                        for (i, name) in ame.variable_names.iter().enumerate() {
                            if name == "_cons" || name == "const" {
                                continue;
                            }
                            let eff = ame.effects[i];
                            let se = ame.std_errors[i];
                            rows.push((name.clone(), eff, eff - z * se, eff + z * se));
                        }

                        if rows.is_empty() {
                            return Ok(Some(model_expansion::model_result(
                                "(no marginal effects to plot)",
                                "marginsplot: no marginal effects",
                                "MarginsPlotResult",
                                vec![],
                            )));
                        }

                        let label_w = rows
                            .iter()
                            .map(|(n, _, _, _)| n.len())
                            .max()
                            .unwrap_or(4)
                            .max(8);
                        let all_lo = rows
                            .iter()
                            .map(|(_, _, lo, _)| *lo)
                            .fold(f64::INFINITY, f64::min);
                        let all_hi = rows
                            .iter()
                            .map(|(_, _, _, hi)| *hi)
                            .fold(f64::NEG_INFINITY, f64::max);
                        let range = (all_hi - all_lo).max(1e-15);
                        let plot_lo = all_lo.min(0.0) - range * 0.05;
                        let plot_hi = all_hi.max(0.0) + range * 0.05;
                        let plot_range = (plot_hi - plot_lo).max(1e-15);

                        let to_col = |v: f64| -> usize {
                            ((v - plot_lo) / plot_range * (width - 1) as f64)
                                .round()
                                .clamp(0.0, (width - 1) as f64) as usize
                        };
                        let zero_col = to_col(0.0);

                        let mut display = String::new();
                        display.push_str(&format!("\n  Marginal Effects Plot ({})\n", bm.kind));
                        display.push_str(&format!("  {}\n", "─".repeat(width + 4)));
                        let mut var_vec = Vec::new();
                        let mut effect_vec = Vec::new();
                        let mut ci_lo_vec = Vec::new();
                        let mut ci_hi_vec = Vec::new();
                        for (name, eff, ci_lo, ci_hi) in &rows {
                            let c_lo = to_col(*ci_lo);
                            let c_hi = to_col(*ci_hi).min(width - 1);
                            let c_pt = to_col(*eff);
                            let mut line = vec![' '; width];
                            if zero_col < width {
                                line[zero_col] = '│';
                            }
                            if c_lo <= c_hi {
                                line[c_lo..=c_hi].fill('─');
                            }
                            if c_pt < width {
                                line[c_pt] = '●';
                            }
                            let bar: String = line.into_iter().collect();
                            display.push_str(&format!(
                                "{:>lw$} │{bar}  {eff:>8.4}\n",
                                name,
                                lw = label_w,
                                bar = bar,
                                eff = eff
                            ));
                            var_vec.push(Value::Str(name.clone()));
                            effect_vec.push(Value::Float(*eff));
                            ci_lo_vec.push(Value::Float(*ci_lo));
                            ci_hi_vec.push(Value::Float(*ci_hi));
                        }
                        display.push_str(&format!("  {}\n", "─".repeat(width + 4)));
                        display.push_str(&format!("  {zero_col} (zero reference)\n"));

                        let mut columns = HashMap::new();
                        columns.insert("variable".into(), Value::List(Arc::new(var_vec)));
                        columns.insert("effect".into(), Value::List(Arc::new(effect_vec)));
                        columns.insert("ci_lo".into(), Value::List(Arc::new(ci_lo_vec)));
                        columns.insert("ci_hi".into(), Value::List(Arc::new(ci_hi_vec)));
                        let effects_df = self.dict_to_dataframe(&columns)?;

                        let fields: Vec<(String, Value)> = vec![
                            ("effects".into(), Value::DataFrame(Arc::new(effects_df))),
                            (
                                "fit".into(),
                                model_expansion::fit_dict(&[
                                    ("model_kind", Value::Str(bm.kind.clone())),
                                    ("n", Value::Int(bm.y.len() as i64)),
                                    ("width", Value::Int(width as i64)),
                                ]),
                            ),
                        ];
                        let summary = format!(
                            "Marginal effects plot ({}): {} variables",
                            bm.kind,
                            rows.len()
                        );
                        return Ok(Some(model_expansion::model_result(
                            display,
                            summary,
                            "MarginsPlotResult",
                            fields,
                        )));
                    }
                    _ => {
                        return Err(HayashiError::Runtime(
                            "marginsplot: supports logit/probit models".into(),
                        ))
                    }
                }
            }

            // ── vecm ─────────────────────────────────────────────────────────
            #[cfg(feature = "greeners-timeseries")]
            "vecm" => self.eval_vecm(args, opt_map),

            // ── var ──────────────────────────────────────────────────────────
            #[cfg(feature = "greeners-timeseries")]
            "var" => self.eval_var(args, opt_map),

            // ── irf ──────────────────────────────────────────────────────────
            #[cfg(feature = "greeners-timeseries")]
            "irf" => self.eval_irf(args, opt_map),

            // ── fevd ─────────────────────────────────────────────────────────
            #[cfg(feature = "greeners-timeseries")]
            "fevd" => self.eval_fevd(args, opt_map),

            // ── arima / sarima ───────────────────────────────────────────────
            #[cfg(feature = "greeners-timeseries")]
            "arima" | "sarima" => self.eval_arima(func, args, opt_map),

            // ── autoreg ──────────────────────────────────────────────────────
            // autoreg(df, y, lags=p, trend="c")
            #[cfg(feature = "greeners-timeseries")]
            "autoreg" | "ar" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "autoreg(df, var, lags=p, trend=\"c\"|\"ct\"|\"t\"|\"n\")".into(),
                    ));
                }

                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "autoreg(): first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df, opts)?;

                let col_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "autoreg: second argument must be variable name".into(),
                        ))
                    }
                };

                let y = ndarray::Array1::from(self.eval_col_expr(&Expr::Var(col_name), &df)?);

                let lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                let trend = match opt_map.get("trend") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "c".to_string(),
                };

                let result = greeners::AutoReg::fit(&y, lags, None, &trend)
                    .map_err(|e| self.rt_err(format!("autoreg: {e}")))?;

                Ok(Value::AutoRegResult(Rc::new(result)))
            }

            // ── ardl ─────────────────────────────────────────────────────────
            // ardl(y ~ x1 + x2, df, lags=p, xlags=q)
            #[cfg(feature = "greeners-timeseries")]
            "ardl" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "ardl(y ~ x1 + x2, df, lags=p, xlags=q)".into(),
                    ));
                }

                let formula_ast = self.resolve_formula(&args[0])?;

                let df_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "ardl(): second argument must be a DataFrame".into(),
                        ))
                    }
                };
                let df_raw = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };
                let df = self.maybe_filter_df(&df_raw, opts)?;

                let y_lags = match opt_map.get("lags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                let x_lags = match opt_map.get("xlags") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 1,
                };

                let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

                // to_design_matrix retorna (y, x_com_constante)
                let (y_vec, x_with_const) = df
                    .to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // ARDL::fit adds its own constant; remove intercept column
                let x_no_const = if x_with_const.ncols() > 1 {
                    x_with_const.slice(ndarray::s![.., 1..]).to_owned()
                } else {
                    return Err(HayashiError::Runtime(
                        "ardl: formula must have at least one regressor besides intercept".into(),
                    ));
                };

                let y_arr = ndarray::Array1::from_vec(y_vec.to_vec());

                let result = greeners::ARDL::fit(&y_arr, &x_no_const, y_lags, x_lags)
                    .map_err(|e| self.rt_err(format!("ardl: {e}")))?;

                Ok(Value::ArdlResult(Rc::new(result)))
            }

            // ── kalman ───────────────────────────────────────────────────────
            // kalman(df, var, model="ll"|"llt", sigma_obs=s, sigma_state=s)
            //
            // Predefined models (State Space Form):
            //   "ll"  — Local Level:        y_t = mu_t + e_t
            //                               mu_t = mu_{t-1} + eta_t
            //   "llt" — Local Linear Trend: y_t = mu_t + e_t
            //                               mu_t = mu_{t-1} + nu_{t-1} + eta_t
            //                               nu_t = nu_{t-1} + zeta_t
            //
            // Adiciona colunas {var}_filtered e {var}_smoothed ao DataFrame.
            #[cfg(feature = "greeners-timeseries")]
            "kalman" | "kfilter" | "ssm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "kalman(df, var, model=\"ll\"|\"llt\", sigma_obs=s, sigma_state=s)".into(),
                    ));
                }

                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kalman(): first argument must be a DataFrame".into(),
                        ))
                    }
                };
                let mut df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                };

                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "kalman: second argument must be variable name".into(),
                        ))
                    }
                };

                let model_kind = match opt_map.get("model") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => "ll".to_string(),
                };

                let y_vec: Vec<f64> = get_col_f64(&df, &var_name)?.to_vec();
                let n = y_vec.len();
                if n < 4 {
                    return Err(HayashiError::Runtime(
                        "kalman: series too short (minimum 4 observations)".into(),
                    ));
                }

                // Estimate sigma_obs from diff(y) if not provided
                let diff_var: f64 = {
                    let diffs: Vec<f64> = y_vec.windows(2).map(|w| w[1] - w[0]).collect();
                    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
                    diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64
                };
                let sigma_obs_default = (diff_var / 2.0).sqrt().max(1e-6);

                let sigma_obs = match opt_map.get("sigma_obs") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => sigma_obs_default,
                };
                let sigma_state = match opt_map.get("sigma_state") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => sigma_obs * 0.1,
                };
                let sigma_slope = match opt_map.get("sigma_slope") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => sigma_state * 0.1,
                };

                // Observations as Vec<Array1<f64>> (scalar-wrapped)
                let obs: Vec<ndarray::Array1<f64>> = y_vec
                    .iter()
                    .map(|&v| ndarray::Array1::from_vec(vec![v]))
                    .collect();

                let ss_result = match model_kind.as_str() {
                    "ll" | "local_level" => {
                        // Local-level model: fit variances by MLE and return a result object.
                        let result = greeners::statespace::LocalLevel::fit(&y_vec)
                            .map_err(|e| self.rt_err(format!("kalman (ll): {e}")))?;
                        return Ok(Some(Value::LocalLevelResult(Rc::new(result))));
                    }
                    "llt" | "local_linear_trend" => {
                        // States: [level, slope]
                        // H = [[1, 0]]
                        // F = [[1, 1], [0, 1]]
                        // R = I_2, Q = diag(sigma_state^2, sigma_slope^2)
                        let h = ndarray::array![[1.0_f64, 0.0]];
                        let f = ndarray::array![[1.0_f64, 1.0], [0.0, 1.0]];
                        let r = ndarray::Array2::<f64>::eye(2);
                        let mut q = ndarray::Array2::<f64>::zeros((2, 2));
                        q[[0, 0]] = sigma_state.powi(2);
                        q[[1, 1]] = sigma_slope.powi(2);
                        let r_obs = ndarray::Array2::from_elem((1, 1), sigma_obs.powi(2));
                        let init_slope = if n > 1 { y_vec[1] - y_vec[0] } else { 0.0 };
                        let model = greeners::StateSpaceModel {
                            h,
                            f,
                            r,
                            q,
                            r_obs,
                            s0: ndarray::Array1::from_vec(vec![y_vec[0], init_slope]),
                            p0: {
                                let mut p = ndarray::Array2::<f64>::zeros((2, 2));
                                p[[0, 0]] = sigma_obs.powi(2) * 10.0;
                                p[[1, 1]] = sigma_slope.powi(2) * 10.0;
                                p
                            },
                        };
                        greeners::state_space_estimate(&model, &obs)
                            .map_err(|e| self.rt_err(format!("kalman (llt): {e}")))?
                    }
                    other => {
                        return Err(HayashiError::Runtime(format!(
                            "kalman(): unknown model '{other}' — use \"ll\" or \"llt\""
                        )))
                    }
                };

                // Extract filtered and smoothed level (state 0 in both models)
                let filtered: ndarray::Array1<f64> = ndarray::Array1::from_vec(
                    ss_result.filtered_states.iter().map(|s| s[0]).collect(),
                );
                let smoothed: ndarray::Array1<f64> = ndarray::Array1::from_vec(
                    ss_result.smoothed_states.iter().map(|s| s[0]).collect(),
                );

                let filt_name = format!("{var_name}_filtered");
                let smooth_name = format!("{var_name}_smoothed");

                let mut fields: Vec<(String, Value)> = vec![
                    (
                        "fit".into(),
                        model_expansion::fit_dict(&[
                            ("model_kind", Value::Str(model_kind.clone())),
                            ("n", Value::Int(n as i64)),
                            ("log_likelihood", Value::Float(ss_result.log_likelihood)),
                            ("sigma_obs", Value::Float(sigma_obs)),
                            ("sigma_state", Value::Float(sigma_state)),
                        ]),
                    ),
                    (
                        "filtered".into(),
                        model_expansion::array1_to_series(&filt_name, &filtered),
                    ),
                    (
                        "smoothed".into(),
                        model_expansion::array1_to_series(&smooth_name, &smoothed),
                    ),
                ];

                Arc::make_mut(&mut df)
                    .insert(filt_name.clone(), filtered)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Arc::make_mut(&mut df)
                    .insert(smooth_name.clone(), smoothed)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // For LLT, also add trend (slope = state 1)
                let display = if matches!(model_kind.as_str(), "llt" | "local_linear_trend") {
                    let slope_filt: ndarray::Array1<f64> = ndarray::Array1::from_vec(
                        ss_result.filtered_states.iter().map(|s| s[1]).collect(),
                    );
                    let slope_smooth: ndarray::Array1<f64> = ndarray::Array1::from_vec(
                        ss_result.smoothed_states.iter().map(|s| s[1]).collect(),
                    );
                    let sf_name = format!("{var_name}_slope_filtered");
                    let ss_name = format!("{var_name}_slope_smoothed");

                    fields.push((
                        "slope_filtered".into(),
                        model_expansion::array1_to_series(&sf_name, &slope_filt),
                    ));
                    fields.push((
                        "slope_smoothed".into(),
                        model_expansion::array1_to_series(&ss_name, &slope_smooth),
                    ));
                    if let Some(Value::Dict(fit)) =
                        fields.iter_mut().find(|(k, _)| k == "fit").map(|(_, v)| v)
                    {
                        Arc::make_mut(fit).insert("sigma_slope".into(), Value::Float(sigma_slope));
                    }

                    Arc::make_mut(&mut df)
                        .insert(sf_name.clone(), slope_filt)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    Arc::make_mut(&mut df)
                        .insert(ss_name.clone(), slope_smooth)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    format!(
                        "\nKalman ({}):  T={}  loglik={:.4}  σ_obs={:.4}  σ_state={:.4}  σ_slope={:.4}\n  → {filt_name}, {smooth_name}, {sf_name}, {ss_name} added to {df_name}\n",
                        model_kind, n, ss_result.log_likelihood, sigma_obs, sigma_state, sigma_slope
                    )
                } else {
                    format!(
                        "\nKalman ({}):  T={}  loglik={:.4}  σ_obs={:.4}  σ_state={:.4}\n  → {filt_name}, {smooth_name} added to {df_name}\n",
                        model_kind, n, ss_result.log_likelihood, sigma_obs, sigma_state
                    )
                };

                self.env.set(&df_name, Value::DataFrame(df))?;
                let summary = format!(
                    "Kalman {}: T={}, loglik={:.4}",
                    model_kind, n, ss_result.log_likelihood
                );
                Ok(model_expansion::model_result(
                    display,
                    summary,
                    "KalmanResult",
                    fields,
                ))
            }

            // ── forecast ─────────────────────────────────────────────────────
            // forecast(model, steps=8)
            // forecast(model, steps=8, alpha=0.05)
            #[cfg(feature = "greeners-timeseries")]
            "forecast" | "fcast" | "predict_h" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "forecast() requires an ARIMA model".into(),
                    ));
                }

                let model = match self.eval_expr(&args[0])? {
                    #[cfg(feature = "greeners-timeseries")]
                    Value::ArimaResult(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "forecast() requires an ARIMA model".into(),
                        ))
                    }
                };

                let steps = match opt_map.get("steps") {
                    Some(Value::Int(v)) => *v as usize,
                    Some(Value::Float(v)) => *v as usize,
                    _ => 8,
                };
                let alpha = match opt_map.get("alpha") {
                    Some(Value::Float(v)) => *v,
                    Some(Value::Int(v)) => *v as f64,
                    _ => 0.05,
                };

                let (fc, lo, hi) = model
                    .predict_with_ci(steps, None, alpha)
                    .map_err(|e| self.rt_err(format!("forecast: {e}")))?;

                let sep = "─".repeat(52);
                println!(
                    "\nForecast — {} steps ahead  (CI {}%)",
                    steps,
                    ((1.0 - alpha) * 100.0) as usize
                );
                println!("{sep}");
                println!(
                    "{:<6} {:>12} {:>12} {:>12}",
                    "h", "forecast", "lower", "upper"
                );
                println!("{sep}");
                let mut h_vec = Vec::new();
                let mut fc_vec = Vec::new();
                let mut lo_vec = Vec::new();
                let mut hi_vec = Vec::new();
                for h in 0..steps {
                    println!(
                        "{:<6} {:>12.4} {:>12.4} {:>12.4}",
                        h + 1,
                        fc[h],
                        lo[h],
                        hi[h]
                    );
                    h_vec.push((h + 1) as i64);
                    fc_vec.push(Value::Float(fc[h]));
                    lo_vec.push(Value::Float(lo[h]));
                    hi_vec.push(Value::Float(hi[h]));
                }
                println!("{sep}");
                println!();

                let mut columns = HashMap::new();
                columns.insert(
                    "h".into(),
                    Value::List(Arc::new(h_vec.into_iter().map(Value::Int).collect())),
                );
                columns.insert("forecast".into(), Value::List(Arc::new(fc_vec)));
                columns.insert("lower".into(), Value::List(Arc::new(lo_vec)));
                columns.insert("upper".into(), Value::List(Arc::new(hi_vec)));
                let df = self.dict_to_dataframe(&columns)?;
                Ok(Value::DataFrame(Arc::new(df)))
            }

            // ── lincom ───────────────────────────────────────────────────────
            // lincom(model, var1=mult1, var2=mult2, ...)
            // Delegates algebra to Greeners via OlsResult::t_test(r, q, x)
            // ── nlcom: non-linear combination of coefs (delta method) ────────
            // nlcom(model, expr) — expr uses coefficient names as variables
            // Examples: nlcom(m, X1 / X2)   nlcom(m, exp(_cons))   nlcom(m, X1 * X2)
            "nlcom" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("nlcom(model, expression)".into()));
                }
                let m = match self.eval_expr(&args[0])? {
                    Value::Model(m) => m,
                    _ => return Err(HayashiError::Type("nlcom() requires an OLS model".into())),
                };
                let ols = m
                    .as_any()
                    .downcast_ref::<OlsModel>()
                    .ok_or_else(|| HayashiError::Type("nlcom() requires an OLS model".into()))?;
                let names =
                    ols.result.variable_names.as_ref().ok_or_else(|| {
                        HayashiError::Runtime("model has no variable names".into())
                    })?;
                let params = &ols.result.params;
                let k = params.len();
                let expr = &args[1];

                if names.len() != k || ols.result.std_errors.len() != k {
                    return Err(self.rt_err("nlcom: coefficient dimensions disagree"));
                }
                // A temporary scope shadows mutable caller bindings. Pop it
                // before propagating any binding, expression or gradient error.
                let scope_depth = self.env.scope_count();
                let call_depth = self.call_stack.len();
                self.env.push_scope();
                let evaluation = (|| -> Result<(f64, Array1<f64>)> {
                    for (name, &parameter) in names.iter().zip(params.iter()) {
                        self.env.declare(name, Value::Float(parameter))?;
                    }
                    let g = self.nlcom_number(expr)?;
                    let mut gradient = Array1::<f64>::zeros(k);
                    for (((name, &parameter), &coordinate_se), derivative) in names
                        .iter()
                        .zip(params.iter())
                        .zip(ols.result.std_errors.iter())
                        .zip(gradient.iter_mut())
                    {
                        *derivative =
                            self.nlcom_derivative(expr, name, parameter, coordinate_se, g)?;
                        self.env.set(name, Value::Float(parameter))?;
                    }
                    Ok((g, gradient))
                })();
                while self.env.scope_count() > scope_depth {
                    self.env.pop_scope();
                }
                self.call_stack.truncate(call_depth);
                let (g, gradient) = evaluation?;
                let contrast = gradient.view().insert_axis(Axis(0)).to_owned();
                let prediction = ols
                    .result
                    .get_prediction(&contrast, &ols.x, 0.05)
                    .map_err(|e| self.rt_err(format!("nlcom: {e}")))?;
                let se = prediction
                    .se
                    .first()
                    .copied()
                    .ok_or_else(|| self.rt_err("nlcom: invalid prediction dimensions"))?;
                if !se.is_finite() || se < 0.0 {
                    return Err(self.rt_err("nlcom: invalid propagated standard error"));
                }
                // Avoid the original engine's absolute SE cutoff and offset
                // subtraction: the delta null is g(beta)=0, not gradient*beta=0.
                let statistic = if se > 0.0 {
                    g / se
                } else if g == 0.0 {
                    0.0
                } else {
                    g.signum() * f64::INFINITY
                };
                let normal = matches!(ols.result.inference_type, greeners::InferenceType::Normal);
                let df = ols.result.df_resid as f64;
                let (critical, p) = if normal {
                    let distribution =
                        Normal::new(0.0, 1.0).map_err(|e| self.rt_err(format!("nlcom: {e}")))?;
                    (
                        distribution.inverse_cdf(0.975),
                        2.0 * distribution.sf(statistic.abs()),
                    )
                } else {
                    let distribution = statrs::distribution::StudentsT::new(0.0, 1.0, df)
                        .map_err(|e| self.rt_err(format!("nlcom: {e}")))?;
                    (
                        distribution.inverse_cdf(0.975),
                        2.0 * distribution.sf(statistic.abs()),
                    )
                };
                let ci_lower = g - critical * se;
                let ci_upper = g + critical * se;
                if !ci_lower.is_finite()
                    || !ci_upper.is_finite()
                    || !p.is_finite()
                    || !(0.0..=1.0).contains(&p)
                {
                    return Err(
                        self.rt_err("nlcom: interval or probability exceeds numerical precision")
                    );
                }
                let statistic_name = if normal { "z" } else { "t" };

                println!("\n{:=^60}", " nlcom ");
                println!("  g(β̂) = {g:.6}");
                println!("  SE    = {se:.6}   (delta method)");
                println!("  {statistic_name}     = {statistic:.4}   p = {p:.4}");
                if !normal {
                    println!("  df    = {df}");
                }
                println!("  95% CI: [{ci_lower:.6}, {ci_upper:.6}]");
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                if !sig.is_empty() {
                    println!("  {sig}");
                }
                println!("{:=^60}\n", "");
                let mut fields = HashMap::new();
                fields.insert("estimate".into(), Value::Float(g));
                fields.insert("std_err".into(), Value::Float(se));
                fields.insert(statistic_name.into(), Value::Float(statistic));
                fields.insert("p_value".into(), Value::Float(p));
                fields.insert("ci_lower".into(), Value::Float(ci_lower));
                fields.insert("ci_upper".into(), Value::Float(ci_upper));
                fields.insert(
                    "reference_distribution".into(),
                    Value::Str(if normal { "normal" } else { "student_t" }.into()),
                );
                if !normal {
                    fields.insert("df".into(), Value::Float(df));
                }
                Ok(Value::Dict(Arc::new(fields)))
            }

            "lincom" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "lincom() requires an OLS model".into(),
                    ));
                }

                let m = match self.eval_expr(&args[0])? {
                    Value::Model(m) => m,
                    _ => {
                        return Err(HayashiError::Type(
                            "lincom() only supports OLS models".into(),
                        ))
                    }
                };
                let ols = m.as_any().downcast_ref::<OlsModel>().ok_or_else(|| {
                    HayashiError::Type("lincom() only supports OLS models".into())
                })?;

                // nomes dos coeficientes via API do Greeners (sem parse de CSV)
                let var_names: Vec<String> =
                    ols.result.variable_names.clone().ok_or_else(|| {
                        HayashiError::Runtime(
                            "model has no variable_names — use from_formula".into(),
                        )
                    })?;

                let k = var_names.len();

                // monta vetor de contraste c alinhado com var_names
                // aceita "const" (Greeners) e "_cons" (Stata-compat) como aliases
                let mut c = Array1::<f64>::zeros(k);
                let mut found = false;
                for (idx, greeners_name) in var_names.iter().enumerate() {
                    let lookup = if greeners_name == "const" {
                        "_cons"
                    } else {
                        greeners_name.as_str()
                    };
                    let val = opt_map
                        .get(lookup)
                        .or_else(|| opt_map.get(greeners_name.as_str()));
                    if let Some(v) = val {
                        let mult = match v {
                            Value::Float(f) => *f,
                            Value::Int(i) => *i as f64,
                            _ => {
                                return Err(HayashiError::Type(format!(
                                    "{greeners_name}= must be numeric"
                                )))
                            }
                        };
                        c[idx] = mult;
                        found = true;
                    }
                }

                if !found {
                    let available: Vec<&str> = var_names
                        .iter()
                        .map(|n| if n == "const" { "_cons" } else { n.as_str() })
                        .collect();
                    return Err(HayashiError::Runtime(format!(
                        "no coefficients found — available: {}",
                        available.join(", ")
                    )));
                }

                // A contrast is a one-row prediction of the conditional mean.
                // Obtain its uncertainty directly, including when c'β is zero.
                let contrast = c.view().insert_axis(Axis(0)).to_owned();
                let prediction = ols
                    .result
                    .get_prediction(&contrast, &ols.x, 0.05)
                    .map_err(|e| self.rt_err(format!("lincom: {e}")))?;
                let (Some(&estimate), Some(&se), Some(&ci_lower), Some(&ci_upper)) = (
                    prediction.mean.first(),
                    prediction.se.first(),
                    prediction.ci_lower.first(),
                    prediction.ci_upper.first(),
                ) else {
                    return Err(self.rt_err("lincom: invalid prediction dimensions"));
                };
                let (statistic, p) = ols
                    .result
                    .t_test(&c, 0.0, &ols.x)
                    .map_err(|e| self.rt_err(format!("lincom: {e}")))?;
                let normal = matches!(ols.result.inference_type, greeners::InferenceType::Normal);
                let statistic_name = if normal { "z" } else { "t" };
                let df = ols.result.df_resid as f64;

                // readable label for the combination
                let display_name = |n: &str| {
                    if n == "const" {
                        "_cons".to_string()
                    } else {
                        n.to_string()
                    }
                };
                let expr_label: String = var_names
                    .iter()
                    .zip(c.iter())
                    .filter(|(_, &m)| m != 0.0)
                    .enumerate()
                    .map(|(i, (name, &mult))| {
                        let dname = display_name(name);
                        let term = if mult == 1.0 {
                            dname
                        } else if mult == -1.0 {
                            format!("-{dname}")
                        } else {
                            format!("{mult}*{dname}")
                        };
                        if i == 0 {
                            term
                        } else if mult < 0.0 {
                            format!(" - {}", &term[1..])
                        } else {
                            format!(" + {term}")
                        }
                    })
                    .collect();

                let sep = "─".repeat(64);
                println!("\nlincom: {expr_label}");
                println!("{sep}");
                if normal {
                    println!(
                        "{:<12} {:>10} {:>10} {:>10}",
                        "Estimate", "Std.Err.", statistic_name, "p"
                    );
                    println!("{sep}");
                    println!("{estimate:<12.6} {se:>10.6} {statistic:>10.4} {p:>10.4}");
                } else {
                    println!(
                        "{:<12} {:>10} {:>10} {:>8} {:>10}",
                        "Estimate", "Std.Err.", statistic_name, "df", "p"
                    );
                    println!("{sep}");
                    println!("{estimate:<12.6} {se:>10.6} {statistic:>10.4} {df:>8.1} {p:>10.4}");
                }
                println!("{sep}");
                println!("95% CI: [{:.6},  {:.6}]", ci_lower, ci_upper);
                println!();

                let mut map = HashMap::new();
                map.insert("estimate".into(), Value::Float(estimate));
                map.insert("std_err".into(), Value::Float(se));
                map.insert(statistic_name.into(), Value::Float(statistic));
                if !normal {
                    map.insert("df".into(), Value::Float(df));
                }
                map.insert(
                    "reference_distribution".into(),
                    Value::Str(if normal { "normal" } else { "student_t" }.into()),
                );
                map.insert("p_value".into(), Value::Float(p));
                map.insert("ci_lower".into(), Value::Float(ci_lower));
                map.insert("ci_upper".into(), Value::Float(ci_upper));
                map.insert("expression".into(), Value::Str(expr_label));
                Ok(Value::Dict(Arc::new(map)))
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
