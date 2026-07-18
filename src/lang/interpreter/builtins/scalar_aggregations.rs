use super::super::helpers::*;
use super::super::*;
impl Interpreter {
    pub(super) fn scalar_aggregations(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "median" => {
                // median(list) | median(df, x) | median(df, x, if = cond)
                let nums: Vec<f64> = if args.len() >= 2 {
                    let df_name = match &args[0] {
                        Expr::Var(n) => n.clone(),
                        _ => return Err(self.rt_err("median: first argument must be a DataFrame")),
                    };
                    let df = match self.env.get(&df_name) {
                        Some(Value::DataFrame(d)) => d.clone(),
                        _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                    };
                    let var_name = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(
                                self.rt_err("median: second argument must be a variable name")
                            )
                        }
                    };
                    let col = get_col_f64(&df, &var_name)?;
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        col.iter()
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|(&v, _)| v)
                            .collect()
                    } else {
                        col.to_vec()
                    }
                } else if args.len() == 1 {
                    match self.eval_expr(&args[0])? {
                        Value::List(lst) => lst.iter().map(value_as_f64).collect::<Result<_>>()?,
                        other => {
                            return Err(self
                                .type_err(format!("median() requires numeric list, got {other}")))
                        }
                    }
                } else {
                    return Err(self.rt_err("median() requires at least 1 argument"));
                };
                if nums.is_empty() {
                    return Err(self.rt_err("median(): empty list"));
                }
                let mut sorted = nums.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                let result = if n.is_multiple_of(2) {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                } else {
                    sorted[n / 2]
                };
                Ok(Value::Float(result))
            }
            "variance" => {
                // variance(list) | variance(df, x) | variance(df, x, if = cond) — sample (/ n-1)
                let nums: Vec<f64> = if args.len() >= 2 {
                    let df_name = match &args[0] {
                        Expr::Var(n) => n.clone(),
                        _ => {
                            return Err(self.rt_err("variance: first argument must be a DataFrame"))
                        }
                    };
                    let df = match self.env.get(&df_name) {
                        Some(Value::DataFrame(d)) => d.clone(),
                        _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                    };
                    let var_name = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(
                                self.rt_err("variance: second argument must be a variable name")
                            )
                        }
                    };
                    let col = get_col_f64(&df, &var_name)?;
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        col.iter()
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|(&v, _)| v)
                            .collect()
                    } else {
                        col.to_vec()
                    }
                } else if args.len() == 1 {
                    match self.eval_expr(&args[0])? {
                        Value::List(lst) => lst.iter().map(value_as_f64).collect::<Result<_>>()?,
                        other => {
                            return Err(self.type_err(format!(
                                "variance() requires numeric list, got {other}"
                            )))
                        }
                    }
                } else {
                    return Err(self.rt_err("variance() requires at least 1 argument"));
                };
                let n = nums.len();
                if n < 2 {
                    return Err(self.rt_err("variance(): requires at least 2 observations"));
                }
                let mean = nums.iter().sum::<f64>() / n as f64;
                let v = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                Ok(Value::Float(v))
            }
            _ => unreachable!(),
        }
    }
}
