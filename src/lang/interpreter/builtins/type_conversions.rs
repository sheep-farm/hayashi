use super::super::helpers::*;
use super::super::*;
impl Interpreter {
    pub(super) fn type_conversions(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "int" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("int(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Int(match v {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    Value::Bool(b) => {
                        if b {
                            1
                        } else {
                            0
                        }
                    }
                    Value::Str(s) => s
                        .trim()
                        .parse::<i64>()
                        .or_else(|_| s.trim().parse::<f64>().map(|f| f as i64))
                        .map_err(|_| self.type_err(format!("cannot convert '{s}' to int")))?,
                    other => return Err(self.type_err(format!("cannot convert {other} to int"))),
                }))
            }
            "float" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("float(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Float(match v {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    Value::Bool(b) => {
                        if b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Value::Str(s) => s
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| self.type_err(format!("cannot convert '{s}' to float")))?,
                    other => return Err(self.type_err(format!("cannot convert {other} to float"))),
                }))
            }
            "str" | "string" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("str(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Str(format!("{v}")))
            }
            "bool" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("bool(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(value_as_bool(&v)))
            }
            "is_nil" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_nil(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Nil)))
            }
            "is_int" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_int(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Int(_))))
            }
            "is_float" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_float(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Float(_))))
            }
            "is_bool" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_bool(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Bool(_))))
            }
            "is_str" | "is_string" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_str(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Str(_))))
            }
            "is_list" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_list(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::List(_))))
            }
            "is_dict" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_dict(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::Dict(_))))
            }
            "is_df" | "is_dataframe" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_dataframe(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::DataFrame(_))))
            }
            "is_fn" | "is_function" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("is_function(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Bool(matches!(v, Value::UserFn(_))))
            }
            "type" | "typeof" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("type(x)".into()));
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::Str(
                    match v {
                        Value::Float(_) => "float",
                        Value::Int(_) => "int",
                        Value::Bool(_) => "bool",
                        Value::Str(_) => "string",
                        Value::List(_) => "list",
                        Value::Dict(_) => "dict",
                        Value::DataFrame(_) => "dataframe",
                        Value::UserFn(_) => "function",
                        Value::Nil => "nil",
                        _ => "model",
                    }
                    .to_string(),
                ))
            }
            _ => unreachable!(),
        }
    }
}
