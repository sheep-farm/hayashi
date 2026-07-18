use super::super::*;
impl Interpreter {
    pub(super) fn regex(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "regexm" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("regexm(string, pattern)".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("regexm: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexm: pattern must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Bool(greeners::Transforms::regexm(&s, &pat)))
            }
            "regexr" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "regexr(string, pattern, replacement)".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("regexr: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexr: pattern must be string, got {v}"))
                        )
                    }
                };
                let rep = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexr: replacement must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Str(greeners::Transforms::regexr(&s, &pat, &rep)))
            }
            "regexra" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "regexra(string, pattern, replacement)".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("regexra: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexra: pattern must be string, got {v}"))
                        )
                    }
                };
                let rep = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexra: replacement must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Str(greeners::Transforms::regexra(&s, &pat, &rep)))
            }
            "regexs" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("regexs(string, pattern)".into()));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("regexs: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("regexs: pattern must be string, got {v}"))
                        )
                    }
                };
                match greeners::Transforms::regexs(&s, &pat) {
                    Some(m) => Ok(Value::Str(m)),
                    None => Ok(Value::Str(String::new())),
                }
            }
            _ => unreachable!(),
        }
    }
}
