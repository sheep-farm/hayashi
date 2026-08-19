use super::super::helpers::*;
use super::super::*;
impl Interpreter {
    pub(super) fn aggregations_list(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "sum" | "mean" | "sd" | "std" | "min" | "max" | "total" => {
                // Form 1: mean(list)  /  sd(list)  /  std(list)  etc.
                // Form 2: mean(df, var)  or  mean(df, var, if=cond)
                let nums: Vec<f64> = if args.len() >= 2 {
                    // DataFrame form
                    let df_name = match &args[0] {
                        Expr::Var(n) => n.clone(),
                        _ => {
                            return Err(self
                                .type_err(format!("{func}: first argument must be a DataFrame")))
                        }
                    };
                    let df = match self.env.get(&df_name) {
                        Some(Value::DataFrame(d)) => d.clone(),
                        _ => return Err(self.rt_err(format!("'{df_name}' is not a DataFrame"))),
                    };
                    let var_name = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => {
                            return Err(self.type_err(format!(
                                "{func}: second argument must be a variable name"
                            )))
                        }
                    };
                    let col = get_col_f64(&df, &var_name)?;
                    // optional filter: if=cond
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
                    let v = self.eval_expr(&args[0])?;
                    match v {
                        Value::List(lst) => lst.iter().map(value_as_f64).collect::<Result<_>>()?,
                        Value::Series(s) => {
                            if s.is_empty() {
                                return Err(self.rt_err(format!("{func}(): empty series")));
                            }
                            let v = s.numeric_values();
                            let val = match func {
                                "sum" | "total" => v.iter().sum::<f64>(),
                                "mean" => s.mean(),
                                "min" => s.min(),
                                "max" => s.max(),
                                "sd" | "std" => {
                                    if s.len() < 2 {
                                        return Err(self.rt_err(format!(
                                            "{func}(): series needs at least 2 observations"
                                        )));
                                    }
                                    s.sd()
                                }
                                _ => unreachable!(),
                            };
                            return Ok(Value::Float(val));
                        }
                        other => {
                            return Err(self.type_err(format!(
                                "{func}() requires numeric list or series, got {other}"
                            )))
                        }
                    }
                } else {
                    return Err(self.rt_err(format!("{func}() requires at least 1 argument")));
                };
                if nums.is_empty() {
                    return Err(self.rt_err(format!(
                        "{func}(): no values (empty list or filter excluded everything)"
                    )));
                }
                let result = match func {
                    "sum" | "total" => nums.iter().sum::<f64>(),
                    "mean" => nums.iter().sum::<f64>() / nums.len() as f64,
                    "min" => nums.iter().cloned().fold(f64::INFINITY, f64::min),
                    "max" => nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    "sd" | "std" => {
                        let n = nums.len() as f64;
                        let m = nums.iter().sum::<f64>() / n;
                        (nums.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
                    }
                    _ => unreachable!(),
                };
                Ok(Value::Float(result))
            }
            _ => unreachable!(),
        }
    }
}
