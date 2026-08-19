use super::super::*;
impl Interpreter {
    pub(super) fn names(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "names" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "names(df) requires 1 argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("names() requires a DataFrame".into())),
                };
                let names: Vec<Value> = df.column_names().into_iter().map(Value::Str).collect();
                Ok(Value::List(Arc::new(names)))
            }
            _ => unreachable!(),
        }
    }
}
