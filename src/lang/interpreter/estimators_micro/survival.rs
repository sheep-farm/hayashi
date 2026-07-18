use super::super::helpers::*;
use super::super::*;

impl Interpreter {
    pub(super) fn km(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 3 {
            return Err(HayashiError::Runtime(
                "km(time, event, df) requires 3 arguments".into(),
            ));
        }
        let time_name = match &args[0] {
            Expr::Var(v) | Expr::Str(v) => v.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "km(): first argument must be nome da coluna de tempo".into(),
                ))
            }
        };
        let event_name = match &args[1] {
            Expr::Var(v) | Expr::Str(v) => v.clone(),
            _ => {
                return Err(HayashiError::Type(
                    "km(): second argument must be nome da coluna de evento".into(),
                ))
            }
        };
        let df = match self.eval_expr(&args[2])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "km(): third argument must be DataFrame".into(),
                ))
            }
        };
        let times = get_col_f64(&df, &time_name)?;
        let events_f = get_col_f64(&df, &event_name)?;
        let events: ndarray::Array1<u8> = events_f.iter().map(|&v| v as u8).collect();
        let result = greeners::KaplanMeier::fit(&times, &events)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::KMResult(Rc::new(result)))
    }

    pub(super) fn cox(
        &mut self,
        _func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "cox(time ~ x1 + x2, df, event=col) requires formula and DataFrame".into(),
            ));
        }
        let formula_ast = self.resolve_formula(&args[0])?;
        let df = match self.eval_expr(&args[1])? {
            Value::DataFrame(d) => d,
            _ => {
                return Err(HayashiError::Type(
                    "cox(): second argument must be DataFrame".into(),
                ))
            }
        };
        let event_col = match opt_map.get("event") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "cox() requires event=coluna option".into(),
                ))
            }
            _ => return Err(HayashiError::Type("event= must be string".into())),
        };
        let times = get_col_f64(&df, &formula_ast.lhs)?;
        let events_f = get_col_f64(&df, &event_col)?;
        let events: ndarray::Array1<u8> = events_f.iter().map(|&v| v as u8).collect();
        // build covariate matrix from RHS variables
        let rhs_vars: Vec<String> = formula_ast
            .rhs
            .iter()
            .filter_map(|t| t.as_var().map(|s| s.to_string()))
            .collect();
        if rhs_vars.is_empty() {
            return Err(HayashiError::Runtime(
                "cox(): formula needs at least one covariate on RHS".into(),
            ));
        }
        let cols: Vec<ndarray::Array1<f64>> = rhs_vars
            .iter()
            .map(|v| get_col_f64(&df, v))
            .collect::<Result<_>>()?;
        let n = times.len();
        let k = cols.len();
        let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
        for (j, col) in cols.iter().enumerate() {
            x_mat.column_mut(j).assign(col);
        }
        let result = greeners::CoxPH::fit_with_names(&times, &events, &x_mat, Some(rhs_vars))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::CoxResult(Rc::new(result)))
    }
}
