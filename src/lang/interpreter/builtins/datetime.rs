use super::super::*;
impl Interpreter {
    pub(super) fn datetime(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "date" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("date(\"YYYY-MM-DD\")".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("date() requires a string".into())),
                };
                let nd = chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d")
                    .map_err(|e| HayashiError::Runtime(format!("date parse error: {e}")))?;
                let dt = nd.and_hms_opt(0, 0, 0).unwrap();
                Ok(Value::Float(dt.and_utc().timestamp() as f64))
            }
            "datetime" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "datetime(\"YYYY-MM-DD HH:MM:SS\")".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("datetime() requires a string".into())),
                };
                let dt = chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                    .or_else(|_| chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%dT%H:%M:%S"))
                    .map_err(|e| HayashiError::Runtime(format!("datetime parse error: {e}")))?;
                Ok(Value::Float(dt.and_utc().timestamp() as f64))
            }
            _ => unreachable!(),
        }
    }
}
