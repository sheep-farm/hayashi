use super::super::*;
impl Interpreter {
    pub(super) fn string_functions(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "upper" | "lower" | "trim" => {
                let s =
                    match self
                        .eval_expr(args.first().ok_or_else(|| {
                            self.rt_err(format!("{func}() requires 1 argument"))
                        })?)? {
                        Value::Str(s) => s,
                        v => {
                            return Err(HayashiError::Type(format!(
                                "{func}() requires string, got {v}"
                            )))
                        }
                    };
                Ok(Value::Str(match func {
                    "upper" => s.to_uppercase(),
                    "lower" => s.to_lowercase(),
                    "trim" => s.trim().to_string(),
                    _ => unreachable!(),
                }))
            }
            "write" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "write(content, path) requires 2 arguments".into(),
                    ));
                }
                let content = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(self.type_err(format!("write: content must be string, got {v}")))
                    }
                };
                let path = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("write: path must be string, got {v}"))),
                };
                std::fs::write(&path, &content)
                    .map_err(|e| HayashiError::Io(format!("Failed to write file '{path}': {e}")))?;
                Ok(Value::Nil)
            }
            "file_exists" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "file_exists(path) requires 1 argument".into(),
                    ));
                }
                let path = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("file_exists: path must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Bool(std::path::Path::new(&path).exists()))
            }
            "ensure_dir" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "ensure_dir(path) requires 1 argument".into(),
                    ));
                }
                let path = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("ensure_dir: path must be string, got {v}"))
                        )
                    }
                };
                std::fs::create_dir_all(&path).map_err(|e| {
                    HayashiError::Io(format!("Failed to create directory '{path}': {e}"))
                })?;
                Ok(Value::Nil)
            }
            "contains" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "contains(s, pattern) requires 2 arguments".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("contains: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("contains: pattern must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Bool(s.contains(pat.as_str())))
            }
            "starts_with" | "ends_with" => {
                if args.len() != 2 {
                    return Err(self.rt_err(format!("{func}(s, pattern) requires 2 arguments")));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("{func}: expected string, got {v}"))),
                };
                let pat = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("{func}: pattern must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Bool(match func {
                    "starts_with" => s.starts_with(pat.as_str()),
                    "ends_with" => s.ends_with(pat.as_str()),
                    _ => unreachable!(),
                }))
            }
            "substr" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime(
                        "substr(s, start [, length]) requires 2 or 3 arguments".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("substr: expected string, got {v}"))),
                };
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len() as i64;
                let start = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    v => {
                        return Err(self.type_err(format!("substr: start must be integer, got {v}")))
                    }
                };
                let real_start =
                    (if start < 0 { len + start } else { start }).clamp(0, len) as usize;
                let count = if args.len() == 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Int(i) => i.max(0) as usize,
                        Value::Float(f) => f.max(0.0) as usize,
                        v => {
                            return Err(
                                self.type_err(format!("substr: length must be integer, got {v}"))
                            )
                        }
                    }
                } else {
                    chars.len().saturating_sub(real_start)
                };
                let end = (real_start + count).min(chars.len());
                Ok(Value::Str(chars[real_start..end].iter().collect()))
            }
            "split" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "split(s, delimiter) requires 2 arguments".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => return Err(self.type_err(format!("split: expected string, got {v}"))),
                };
                let delim = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("split: delimiter must be string, got {v}"))
                        )
                    }
                };
                let parts: Vec<Value> = s
                    .split(delim.as_str())
                    .map(|p| Value::Str(p.to_string()))
                    .collect();
                Ok(Value::List(Arc::new(parts)))
            }
            "str_replace" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "str_replace(s, from, to) requires 3 arguments".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(self.type_err(format!("str_replace: expected string, got {v}")))
                    }
                };
                let from = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("str_replace: 'from' must be string, got {v}"))
                        )
                    }
                };
                let to = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    v => {
                        return Err(
                            self.type_err(format!("str_replace: 'to' must be string, got {v}"))
                        )
                    }
                };
                Ok(Value::Str(s.replace(from.as_str(), to.as_str())))
            }
            _ => unreachable!(),
        }
    }
}
