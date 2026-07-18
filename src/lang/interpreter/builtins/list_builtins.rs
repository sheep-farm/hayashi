use super::super::*;
impl Interpreter {
    pub(super) fn list_builtins(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "len" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "len() requires exactly 1 argument".into(),
                    ));
                }
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::List(lst) => Ok(Value::Int(lst.len() as i64)),
                    Value::Dict(m) => Ok(Value::Int(m.len() as i64)),
                    Value::Str(s) => Ok(Value::Int(s.chars().count() as i64)),
                    Value::Series(s) => Ok(Value::Int(s.len() as i64)),
                    _ => Err(HayashiError::Type(
                        "len() requires list, dict, series, or string".into(),
                    )),
                }
            }
            "keys" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("keys(dict)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::Dict(m) => {
                        let mut ks: Vec<String> = m.keys().cloned().collect();
                        ks.sort();
                        Ok(Value::List(Arc::new(
                            ks.into_iter().map(Value::Str).collect(),
                        )))
                    }
                    _ => Err(HayashiError::Type("keys() requires dict".into())),
                }
            }
            "values" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("values(dict)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::Dict(m) => {
                        let mut pairs: Vec<_> = m.iter().collect();
                        pairs.sort_by_key(|(k, _)| (*k).clone());
                        Ok(Value::List(Arc::new(
                            pairs.into_iter().map(|(_, v)| v.clone()).collect(),
                        )))
                    }
                    _ => Err(HayashiError::Type("values() requires dict".into())),
                }
            }
            "has_key" => {
                if args.len() != 2 {
                    return Err(self.rt_err("has_key(dict, \"key\")"));
                }
                let d = self.eval_expr(&args[0])?;
                let k = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("has_key: key must be string".into())),
                };
                match d {
                    Value::Dict(m) => Ok(Value::Bool(m.contains_key(&k))),
                    _ => Err(HayashiError::Type("has_key() requires dict".into())),
                }
            }
            "dict_merge" | "dmerge" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("dict_merge(dict1, dict2)".into()));
                }
                let d1 = self.eval_expr(&args[0])?;
                let d2 = self.eval_expr(&args[1])?;
                match (d1, d2) {
                    (Value::Dict(m1), Value::Dict(m2)) => {
                        let mut merged = (*m1).clone();
                        for (k, v) in m2.iter() {
                            merged.insert(k.clone(), v.clone());
                        }
                        Ok(Value::Dict(Arc::new(merged)))
                    }
                    _ => Err(HayashiError::Type("dict_merge() requires two dicts".into())),
                }
            }
            "dict_set" | "dset" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "dict_set(dict, \"key\", value)".into(),
                    ));
                }
                let d = self.eval_expr(&args[0])?;
                let k = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("dict_set: key must be string".into())),
                };
                let v = self.eval_expr(&args[2])?;
                match d {
                    Value::Dict(m) => {
                        let mut new_m = (*m).clone();
                        new_m.insert(k, v);
                        Ok(Value::Dict(Arc::new(new_m)))
                    }
                    _ => Err(HayashiError::Type("dict_set() requires dict".into())),
                }
            }
            "dict_remove" | "dremove" => {
                if args.len() != 2 {
                    return Err(self.rt_err("dict_remove(dict, \"key\")"));
                }
                let d = self.eval_expr(&args[0])?;
                let k = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("dict_remove: key must be string".into())),
                };
                match d {
                    Value::Dict(m) => {
                        let mut new_m = (*m).clone();
                        new_m.remove(&k);
                        Ok(Value::Dict(Arc::new(new_m)))
                    }
                    _ => Err(HayashiError::Type("dict_remove() requires dict".into())),
                }
            }
            "dataframe" => {
                if args.len() != 1 {
                    return Err(self.rt_err("dataframe(dict)"));
                }
                let d = self.eval_expr(&args[0])?;
                match d {
                    Value::Dict(m) => {
                        let df = self.dict_to_dataframe(&m)?;
                        Ok(Value::DataFrame(Arc::new(df)))
                    }
                    _ => Err(HayashiError::Type("dataframe() requires dict".into())),
                }
            }
            _ => unreachable!(),
        }
    }
}
