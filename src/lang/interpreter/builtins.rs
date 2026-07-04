use super::*;

/// Conversões de tipo (int/float/str/bool), date/time, builtins de lista e
/// dict, funções de string, regex, agregações sobre List, e as agregações
/// escalares (mean/sum/min/max/std/...) com suporte a `if=`.
/// Extraído de `eval_call` (ver src/lang/interpreter.rs).
impl Interpreter {
    /// Helper for `tidy`: build a tidy coefficient map from model result vectors.
    fn build_tidy_coef_map(
        &self,
        names: Vec<String>,
        params: &ndarray::Array1<f64>,
        std_errors: &ndarray::Array1<f64>,
        t_values: &ndarray::Array1<f64>,
        p_values: &ndarray::Array1<f64>,
        conf_lower: &ndarray::Array1<f64>,
        conf_upper: &ndarray::Array1<f64>,
    ) -> std::collections::HashMap<String, Value> {
        let n = params.len();
        let name_col: Vec<Value> = (0..n)
            .map(|i| {
                Value::Str(
                    names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("x{i}")),
                )
            })
            .collect();
        let coef_col: Vec<Value> = params.iter().map(|&v| Value::Float(v)).collect();
        let se_col: Vec<Value> = std_errors.iter().map(|&v| Value::Float(v)).collect();
        let t_col: Vec<Value> = t_values.iter().map(|&v| Value::Float(v)).collect();
        let p_col: Vec<Value> = p_values.iter().map(|&v| Value::Float(v)).collect();
        let cl_col: Vec<Value> = conf_lower.iter().map(|&v| Value::Float(v)).collect();
        let cu_col: Vec<Value> = conf_upper.iter().map(|&v| Value::Float(v)).collect();

        let mut map = std::collections::HashMap::new();
        map.insert("variable".into(), Value::List(Rc::new(name_col)));
        map.insert("coef".into(), Value::List(Rc::new(coef_col)));
        map.insert("std_err".into(), Value::List(Rc::new(se_col)));
        map.insert("t".into(), Value::List(Rc::new(t_col)));
        map.insert("p_value".into(), Value::List(Rc::new(p_col)));
        map.insert("conf_low".into(), Value::List(Rc::new(cl_col)));
        map.insert("conf_high".into(), Value::List(Rc::new(cu_col)));
        map
    }

    pub(super) fn eval_call_builtins(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── Type conversions ─────────────────────────────────────────────
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
                Ok(Value::Bool(Self::value_as_bool(&v)))
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

            // ── Date/time ─────────────────────────────────────────────────────
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

            // ── Builtins de lista ─────────────────────────────────────────────
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
                    _ => Err(HayashiError::Type(
                        "len() requires list, dict, or string".into(),
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
                        Ok(Value::List(Rc::new(
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
                        Ok(Value::List(Rc::new(
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
                        Ok(Value::Dict(Rc::new(merged)))
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
                        Ok(Value::Dict(Rc::new(new_m)))
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
                        Ok(Value::Dict(Rc::new(new_m)))
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
                        Ok(Value::DataFrame(Rc::new(df)))
                    }
                    _ => Err(HayashiError::Type("dataframe() requires dict".into())),
                }
            }

            // ── tidy: coefficient table from a model ───────────────────────────
            "tidy" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("tidy(model) requires 1 argument".into()));
                }
                let val = self.eval_expr(&args[0])?;
                let mut map = std::collections::HashMap::<String, Value>::new();

                match val {
                    Value::OlsResult(m) => {
                        let r = &m.result;
                        map = self.build_tidy_coef_map(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                            &r.conf_lower,
                            &r.conf_upper,
                        );
                    }
                    Value::RollingResult(r) => {
                        let dates = r.dates.clone();
                        let n = r.n_obs;
                        let k = r.params_history.ncols();
                        let names = r.variable_names.clone().unwrap_or_default();
                        let mut date_col = Vec::new();
                        let mut r2_col = Vec::new();
                        let mut coef_cols: Vec<(String, Vec<Value>)> = (0..k)
                            .map(|j| {
                                let name = names.get(j).cloned().unwrap_or_else(|| {
                                    if j == 0 { "const".into() } else { format!("x{j}") }
                                });
                                (name, Vec::new())
                            })
                            .collect();
                        for t in (r.window - 1)..n {
                            if r.params_history.row(t).iter().any(|v| v.is_nan()) {
                                continue;
                            }
                            let d = dates.get(t).cloned().unwrap_or_else(|| format!("{t}"));
                            date_col.push(Value::Str(d));
                            r2_col.push(Value::Float(r.r_squared_history[t]));
                            for j in 0..k {
                                coef_cols[j].1.push(Value::Float(r.params_history[[t, j]]));
                            }
                        }
                        map.insert("date".into(), Value::List(Rc::new(date_col)));
                        map.insert("r2".into(), Value::List(Rc::new(r2_col)));
                        for (name, vals) in coef_cols {
                            map.insert(name, Value::List(Rc::new(vals)));
                        }
                    }
                    _ => return Err(HayashiError::Type("tidy: unsupported model type".into())),
                }

                let df = self.dict_to_dataframe(&map)?;
                Ok(Value::DataFrame(Rc::new(df)))
            }

            // ── glance: model fit summary ──────────────────────────────────────
            "glance" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("glance(model) requires 1 argument".into()));
                }
                let val = self.eval_expr(&args[0])?;
                let mut map = std::collections::HashMap::<String, Value>::new();

                match val {
                    Value::OlsResult(m) => {
                        let r = &m.result;
                        let scalar = |v: f64| Value::List(Rc::new(vec![Value::Float(v)]));
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert("adj_r2".into(), scalar(r.adj_r_squared));
                        map.insert("n".into(), Value::List(Rc::new(vec![Value::Int(r.n_obs as i64)])));
                        map.insert("f_stat".into(), scalar(r.f_statistic));
                        map.insert("prob_f".into(), scalar(r.prob_f));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("sigma".into(), scalar(r.sigma));
                    }
                    _ => return Err(HayashiError::Type("glance: unsupported model type".into())),
                }

                let df = self.dict_to_dataframe(&map)?;
                Ok(Value::DataFrame(Rc::new(df)))
            }

            // ── Funções de string ─────────────────────────────────────────────
            "upper" | "lower" | "trim" => {
                let s =
                    match self
                        .eval_expr(args.first().ok_or_else(|| {
                            self.rt_err(format!("{func}() requires 1 argument"))
                        })?)? {
                        Value::Str(s) => s,
                        v => {
                            return Err(HayashiError::Type(format!(
                                "{func}() requires string, recebeu {v}"
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
                    v => return Err(self.type_err(format!("file_exists: path must be string, got {v}"))),
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
                    v => return Err(self.type_err(format!("ensure_dir: path must be string, got {v}"))),
                };
                std::fs::create_dir_all(&path)
                    .map_err(|e| HayashiError::Io(format!("Failed to create directory '{path}': {e}")))?;
                Ok(Value::Nil)
            }

            "contains" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "contains(s, padrão) requires 2 arguments".into(),
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
                    return Err(self.rt_err(format!("{func}(s, padrão) requires 2 arguments")));
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

            // substr(s, início [, comprimento]) — índice 0-based em chars
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

            // split(s, delimitador) → List de Str
            "split" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "split(s, delimitador) requires 2 arguments".into(),
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
                Ok(Value::List(Rc::new(parts)))
            }

            // str_replace(s, de, para) — "replace" é palavra-chave
            "str_replace" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "str_replace(s, de, para) requires 3 arguments".into(),
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

            // ── Regex ─────────────────────────────────────────────────────────
            // regexm(s, pattern)            → 1 se match, 0 se não
            // regexr(s, pattern, replace)   → substitui primeira ocorrência
            // regexra(s, pattern, replace)  → substitui todas
            // regexs(s, pattern)            → extrai primeiro grupo de captura
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

            // ── Agregações sobre List ─────────────────────────────────────────
            // "sum" fica para summarize(df) — Stata-style
            // "total" é a soma de uma lista numérica
            "sum" | "mean" | "sd" | "std" | "min" | "max" | "total" => {
                // Forma 1: mean(list)  /  sd(list)  /  std(list)  etc.
                // Forma 2: mean(df, var)  ou  mean(df, var, if=cond)
                let nums: Vec<f64> = if args.len() >= 2 {
                    // forma DataFrame
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
                    let col = Self::get_col_f64(&df, &var_name)?;
                    // filtro opcional: if=cond
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
                        Value::List(lst) => {
                            lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?
                        }
                        other => {
                            return Err(self
                                .type_err(format!("{func}() requires numeric list, got {other}")))
                        }
                    }
                } else {
                    return Err(self.rt_err(format!("{func}() requires at least 1 argument")));
                };
                if nums.is_empty() {
                    return Err(self.rt_err(format!(
                        "{func}(): nenhum valor (empty list ou filtro excluiu tudo)"
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

            // ── Novas agregações escalares (todas suportam if = cond) ────────
            "median" => {
                // median(lista) | median(df, x) | median(df, x, if = cond)
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
                    let col = Self::get_col_f64(&df, &var_name)?;
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
                        Value::List(lst) => {
                            lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?
                        }
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
                // variance(lista) | variance(df, x) | variance(df, x, if = cond) — amostral (/ n-1)
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
                    let col = Self::get_col_f64(&df, &var_name)?;
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
                        Value::List(lst) => {
                            lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?
                        }
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

            "quantile" => {
                // quantile(df, x, p) | quantile(lista, p) | quantile(df, x, p, if = cond) — p ∈ [0,1]
                let (nums, p) = if args.len() >= 3 {
                    let df_name = match &args[0] {
                        Expr::Var(n) => n.clone(),
                        _ => {
                            return Err(self.rt_err("quantile: first argument must be a DataFrame"))
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
                                self.rt_err("quantile: second argument must be a variable name")
                            )
                        }
                    };
                    let col = Self::get_col_f64(&df, &var_name)?;
                    let nums = if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        col.iter()
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|(&v, _)| v)
                            .collect()
                    } else {
                        col.to_vec()
                    };
                    let p = match self.eval_expr(&args[2])? {
                        Value::Float(f) => f,
                        Value::Int(i) => i as f64,
                        other => return Err(self.type_mismatch("Float", &other)),
                    };
                    (nums, p)
                } else if args.len() == 2 {
                    let v = self.eval_expr(&args[0])?;
                    let nums = match v {
                        Value::List(lst) => {
                            lst.iter().map(Self::value_as_f64).collect::<Result<_>>()?
                        }
                        other => {
                            return Err(self.type_err(format!(
                                "quantile() requires numeric list, got {other}"
                            )))
                        }
                    };
                    let p = match self.eval_expr(&args[1])? {
                        Value::Float(f) => f,
                        Value::Int(i) => i as f64,
                        other => return Err(self.type_mismatch("Float", &other)),
                    };
                    (nums, p)
                } else {
                    return Err(self.rt_err("quantile(df, x, p) ou quantile(lista, p)"));
                };
                if !(0.0..=1.0).contains(&p) {
                    return Err(self.rt_err("quantile(): p deve estar em [0, 1]"));
                }
                let mut sorted: Vec<f64> = nums.into_iter().filter(|x| x.is_finite()).collect();
                if sorted.is_empty() {
                    return Err(self.rt_err("quantile(): nenhum valor finito"));
                }
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = p * (sorted.len() - 1) as f64;
                let lo = idx.floor() as usize;
                let hi = idx.ceil() as usize;
                let result = if lo == hi {
                    sorted[lo]
                } else {
                    let w = idx - lo as f64;
                    sorted[lo] * (1.0 - w) + sorted[hi] * w
                };
                Ok(Value::Float(result))
            }

            "cov" => {
                // cov(df, x, y) | cov(df, x, y, if = cond) — covariância amostral (/ n-1)
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("cov(df, x, y)".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };
                let x_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(self.rt_err("cov(): second argument must be a variable name")),
                };
                let y_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(self.rt_err("cov(): third argument must be a variable name")),
                };
                let x_col = Self::get_col_f64(&df, &x_name)?;
                let y_col = Self::get_col_f64(&df, &y_name)?;
                let (x_vals, y_vals): (Vec<f64>, Vec<f64>) =
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        x_col
                            .iter()
                            .zip(y_col.iter())
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|((&xi, &yi), _)| (xi, yi))
                            .unzip()
                    } else {
                        (x_col.to_vec(), y_col.to_vec())
                    };
                let n = x_vals.len();
                if n < 2 {
                    return Err(self.rt_err("cov(): requires at least 2 observations"));
                }
                let mx = x_vals.iter().sum::<f64>() / n as f64;
                let my = y_vals.iter().sum::<f64>() / n as f64;
                let c = x_vals
                    .iter()
                    .zip(y_vals.iter())
                    .map(|(&xi, &yi)| (xi - mx) * (yi - my))
                    .sum::<f64>()
                    / (n - 1) as f64;
                Ok(Value::Float(c))
            }

            "corr_pair" => {
                // corr_pair(df, x, y) | corr_pair(df, x, y, if = cond) — Pearson escalar
                if args.len() < 3 {
                    return Err(HayashiError::Runtime("corr_pair(df, x, y)".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };
                let x_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(
                            self.rt_err("corr_pair(): second argument must be a variable name")
                        )
                    }
                };
                let y_name = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(
                            self.rt_err("corr_pair(): third argument must be a variable name")
                        )
                    }
                };
                let x_col = Self::get_col_f64(&df, &x_name)?;
                let y_col = Self::get_col_f64(&df, &y_name)?;
                let (x_vals, y_vals): (Vec<f64>, Vec<f64>) =
                    if let Some(cond_opt) = opts.iter().find(|o| o.name == "if") {
                        let mask = self.eval_col_expr(&cond_opt.value, &df)?;
                        x_col
                            .iter()
                            .zip(y_col.iter())
                            .zip(mask.iter())
                            .filter(|(_, &m)| m != 0.0)
                            .map(|((&xi, &yi), _)| (xi, yi))
                            .unzip()
                    } else {
                        (x_col.to_vec(), y_col.to_vec())
                    };
                let n = x_vals.len();
                if n < 2 {
                    return Err(self.rt_err("corr_pair(): requires at least 2 observations"));
                }
                let mx = x_vals.iter().sum::<f64>() / n as f64;
                let my = y_vals.iter().sum::<f64>() / n as f64;
                let mut num = 0.0f64;
                let mut dx2 = 0.0f64;
                let mut dy2 = 0.0f64;
                for (&xi, &yi) in x_vals.iter().zip(y_vals.iter()) {
                    let dx = xi - mx;
                    let dy = yi - my;
                    num += dx * dy;
                    dx2 += dx * dx;
                    dy2 += dy * dy;
                }
                let r = if dx2 > 0.0 && dy2 > 0.0 {
                    num / (dx2.sqrt() * dy2.sqrt())
                } else {
                    0.0
                };
                Ok(Value::Float(r))
            }

            "push" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("push(lista, item)".into()));
                }
                let var_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "push() first argument must be a variable".into(),
                        ))
                    }
                };
                let item = self.eval_expr(&args[1])?;
                let lst = self
                    .env
                    .get(&var_name)
                    .cloned()
                    .ok_or_else(|| self.rt_err(format!("undefined variable '{var_name}'")))?;
                match lst {
                    Value::List(v) => {
                        let mut new_v = (*v).clone();
                        new_v.push(item);
                        self.env.set(&var_name, Value::List(Rc::new(new_v)))?;
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Type("push() requires list".into())),
                }
            }

            "pop" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("pop(lista)".into()));
                }
                let var_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(
                            "pop() argument must be a variable".into(),
                        ))
                    }
                };
                let lst = self
                    .env
                    .get(&var_name)
                    .cloned()
                    .ok_or_else(|| self.rt_err(format!("undefined variable '{var_name}'")))?;
                match lst {
                    Value::List(v) => {
                        if v.is_empty() {
                            return Err(HayashiError::Runtime("pop() on empty list".into()));
                        }
                        let mut new_v = (*v).clone();
                        let removed = new_v.pop().unwrap();
                        self.env.set(&var_name, Value::List(Rc::new(new_v)))?;
                        Ok(removed)
                    }
                    _ => Err(HayashiError::Type("pop() requires list".into())),
                }
            }

            "insert" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime("insert(lista, indice, item)".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let idx = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(HayashiError::Type("insert: index must be integer".into())),
                };
                let item = self.eval_expr(&args[2])?;
                match lst {
                    Value::List(v) => {
                        let mut new_v = (*v).clone();
                        if idx > new_v.len() {
                            return Err(HayashiError::Runtime(format!(
                                "insert: index out of range (len={})",
                                new_v.len()
                            )));
                        }
                        new_v.insert(idx, item);
                        Ok(Value::List(Rc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("insert() requires list".into())),
                }
            }

            "remove" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("remove(lista, indice)".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let idx = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(HayashiError::Type("remove: index must be integer".into())),
                };
                match lst {
                    Value::List(v) => {
                        if idx >= v.len() {
                            return Err(HayashiError::Runtime(format!(
                                "remove: index out of range (len={})",
                                v.len()
                            )));
                        }
                        let mut new_v = (*v).clone();
                        new_v.remove(idx);
                        Ok(Value::List(Rc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("remove() requires list".into())),
                }
            }

            "clear" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("clear(lista)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(_) => Ok(Value::List(Rc::new(Vec::new()))),
                    _ => Err(HayashiError::Type("clear() requires list".into())),
                }
            }

            "reverse" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("reverse(lista)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(v) => {
                        let mut new_v = (*v).clone();
                        new_v.reverse();
                        Ok(Value::List(Rc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("reverse() requires list".into())),
                }
            }

            "index" | "indexof" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "index(lista, item) → posição ou -1".into(),
                    ));
                }
                let lst = self.eval_expr(&args[0])?;
                let needle = self.eval_expr(&args[1])?;
                match lst {
                    Value::List(v) => {
                        let pos = v.iter().position(|x| format!("{x}") == format!("{needle}"));
                        Ok(Value::Int(pos.map(|p| p as i64).unwrap_or(-1)))
                    }
                    _ => Err(HayashiError::Type("index() requires list".into())),
                }
            }

            "slice" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime("slice(lista, inicio [, fim])".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let start = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(HayashiError::Type("slice: start must be integer".into())),
                };
                match lst {
                    Value::List(v) => {
                        let end = if args.len() == 3 {
                            match self.eval_expr(&args[2])? {
                                Value::Int(i) => (i as usize).min(v.len()),
                                Value::Float(f) => (f as usize).min(v.len()),
                                _ => {
                                    return Err(HayashiError::Type(
                                        "slice: end must be integer".into(),
                                    ))
                                }
                            }
                        } else {
                            v.len()
                        };
                        let s = start.min(v.len());
                        Ok(Value::List(Rc::new(v[s..end].to_vec())))
                    }
                    _ => Err(HayashiError::Type("slice() requires list".into())),
                }
            }

            "join" => {
                if args.is_empty() || args.len() > 2 {
                    return Err(HayashiError::Runtime("join(lista [, separador])".into()));
                }
                let lst = self.eval_expr(&args[0])?;
                let sep = if args.len() == 2 {
                    match self.eval_expr(&args[1])? {
                        Value::Str(s) => s,
                        _ => {
                            return Err(HayashiError::Type("join: separator must be string".into()))
                        }
                    }
                } else {
                    ", ".to_string()
                };
                match lst {
                    Value::List(v) => {
                        let strs: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                        Ok(Value::Str(strs.join(&sep)))
                    }
                    _ => Err(HayashiError::Type("join() requires list".into())),
                }
            }

            "map" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "map(list, fn) or map(list, |x| expr)".into(),
                    ));
                }
                let lst = match self.eval_expr(&args[0])? {
                    Value::List(v) => v,
                    _ => return Err(HayashiError::Type("map() requires list".into())),
                };
                let fn_val = self.eval_expr(&args[1])?;
                let mut result = Vec::with_capacity(lst.len());
                for item in lst.iter() {
                    let val = self.call_value_fn(&fn_val, std::slice::from_ref(item))?;
                    result.push(val);
                }
                Ok(Value::List(Rc::new(result)))
            }

            "unique" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("unique(lista)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(v) => {
                        let mut seen = Vec::new();
                        let mut result = Vec::new();
                        for item in v.iter() {
                            let key = format!("{item}");
                            if !seen.contains(&key) {
                                seen.push(key);
                                result.push(item.clone());
                            }
                        }
                        Ok(Value::List(Rc::new(result)))
                    }
                    _ => Err(HayashiError::Type("unique() requires list".into())),
                }
            }

            "flatten" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("flatten(lista)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(v) => {
                        let mut result = Vec::new();
                        for item in v.iter() {
                            match item {
                                Value::List(inner) => result.extend(inner.iter().cloned()),
                                other => result.push(other.clone()),
                            }
                        }
                        Ok(Value::List(Rc::new(result)))
                    }
                    _ => Err(HayashiError::Type("flatten() requires list".into())),
                }
            }

            "chain" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("chain(seq1, seq2, ...)".into()));
                }
                let mut result = Vec::new();
                for arg in args {
                    match self.eval_expr(arg)? {
                        Value::List(v) => result.extend(v.iter().cloned()),
                        other => return Err(self.type_mismatch("List", &other)),
                    }
                }
                Ok(Value::List(Rc::new(result)))
            }

            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(HayashiError::Runtime(
                        "range(start, end [, step]) requires 2 or 3 arguments".into(),
                    ));
                }
                let start = match self.eval_expr(&args[0])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(HayashiError::Type("range: start must be integer".into())),
                };
                let end = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(HayashiError::Type("range: end must be integer".into())),
                };
                let step: i64 = if args.len() == 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Int(i) => i,
                        Value::Float(f) => f as i64,
                        _ => return Err(HayashiError::Type("range: step must be integer".into())),
                    }
                } else if start <= end {
                    1
                } else {
                    -1
                };
                if step == 0 {
                    return Err(HayashiError::Runtime("range: step cannot be zero".into()));
                }
                let mut v = Vec::new();
                let mut cur = start;
                while if step > 0 { cur < end } else { cur > end } {
                    v.push(Value::Int(cur));
                    cur += step;
                }
                Ok(Value::List(Rc::new(v)))
            }
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
