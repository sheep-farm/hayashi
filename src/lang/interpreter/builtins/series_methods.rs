use super::super::helpers::*;
use super::super::*;
impl Interpreter {
    pub(super) fn series_methods(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "first" => {
                if args.len() != 1 {
                    return Err(self.rt_err("first(series) requires 1 argument"));
                }
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::Series(s) => s
                        .first()
                        .ok_or_else(|| self.rt_err("first(): empty series")),
                    Value::List(lst) => lst
                        .first()
                        .cloned()
                        .ok_or_else(|| self.rt_err("first(): empty list")),
                    other => {
                        Err(self.type_err(format!("first() requires series or list, got {other}")))
                    }
                }
            }
            "last" => {
                if args.len() != 1 {
                    return Err(self.rt_err("last(series) requires 1 argument"));
                }
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::Series(s) => s.last().ok_or_else(|| self.rt_err("last(): empty series")),
                    Value::List(lst) => lst
                        .last()
                        .cloned()
                        .ok_or_else(|| self.rt_err("last(): empty list")),
                    other => {
                        Err(self.type_err(format!("last() requires series or list, got {other}")))
                    }
                }
            }
            "shift" => {
                if args.len() != 2 {
                    return Err(self.rt_err("shift(series, n) requires 2 arguments"));
                }
                let v = self.eval_expr(&args[0])?;
                let n = match self.eval_expr(&args[1])? {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    other => {
                        return Err(
                            self.type_err(format!("shift(): n must be integer, got {other}"))
                        )
                    }
                };
                match v {
                    Value::Series(s) => Ok(Value::Series(Arc::new(s.shift(n)))),
                    Value::List(lst) => {
                        let shifted = if n > 0 {
                            let mut v = vec![Value::Nil; n as usize];
                            v.extend_from_slice(&lst[..lst.len().saturating_sub(n as usize)]);
                            v
                        } else if n < 0 {
                            let n_abs = (-n) as usize;
                            let mut v = lst[n_abs.min(lst.len())..].to_vec();
                            v.extend(vec![Value::Nil; n_abs.min(lst.len())]);
                            v
                        } else {
                            lst.to_vec()
                        };
                        Ok(Value::List(Arc::new(shifted)))
                    }
                    other => {
                        Err(self.type_err(format!("shift() requires series or list, got {other}")))
                    }
                }
            }
            "quantile" => {
                // quantile(df, x, p) | quantile(list, p) | quantile(df, x, p, if = cond) — p ∈ [0,1]
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
                    let col = get_col_f64(&df, &var_name)?;
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
                        Value::List(lst) => lst.iter().map(value_as_f64).collect::<Result<_>>()?,
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
                    return Err(self.rt_err("quantile(df, x, p) or quantile(list, p)"));
                };
                if !(0.0..=1.0).contains(&p) {
                    return Err(self.rt_err("quantile(): p must be in [0, 1]"));
                }
                let mut sorted: Vec<f64> = nums.into_iter().filter(|x| x.is_finite()).collect();
                if sorted.is_empty() {
                    return Err(self.rt_err("quantile(): no finite value"));
                }
                sorted.sort_by(nan_last_cmp);
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
                // cov(df, x, y) | cov(df, x, y, if = cond) — sample covariance (/ n-1)
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
                let x_col = get_col_f64(&df, &x_name)?;
                let y_col = get_col_f64(&df, &y_name)?;
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
                // corr_pair(df, x, y) | corr_pair(df, x, y, if = cond) — scalar Pearson
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
                let x_col = get_col_f64(&df, &x_name)?;
                let y_col = get_col_f64(&df, &y_name)?;
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
                    return Err(HayashiError::Runtime("push(list, item)".into()));
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
                        self.env.set(&var_name, Value::List(Arc::new(new_v)))?;
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Type("push() requires list".into())),
                }
            }
            "pop" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("pop(list)".into()));
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
                        self.env.set(&var_name, Value::List(Arc::new(new_v)))?;
                        Ok(removed)
                    }
                    _ => Err(HayashiError::Type("pop() requires list".into())),
                }
            }
            "insert" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime("insert(list, index, item)".into()));
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
                        Ok(Value::List(Arc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("insert() requires list".into())),
                }
            }
            "remove" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("remove(list, index)".into()));
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
                        Ok(Value::List(Arc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("remove() requires list".into())),
                }
            }
            "clear" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("clear(list)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(_) => Ok(Value::List(Arc::new(Vec::new()))),
                    _ => Err(HayashiError::Type("clear() requires list".into())),
                }
            }
            "reverse" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("reverse(list)".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::List(v) => {
                        let mut new_v = (*v).clone();
                        new_v.reverse();
                        Ok(Value::List(Arc::new(new_v)))
                    }
                    _ => Err(HayashiError::Type("reverse() requires list".into())),
                }
            }
            "index" | "indexof" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "index(list, item) → position or -1".into(),
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
                    return Err(HayashiError::Runtime("slice(list, start [, end])".into()));
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
                        Ok(Value::List(Arc::new(v[s..end].to_vec())))
                    }
                    _ => Err(HayashiError::Type("slice() requires list".into())),
                }
            }
            "join" => {
                if args.is_empty() || args.len() > 2 {
                    return Err(HayashiError::Runtime("join(list [, separator])".into()));
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
                Ok(Value::List(Arc::new(result)))
            }
            "unique" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("unique(list)".into()));
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
                        Ok(Value::List(Arc::new(result)))
                    }
                    _ => Err(HayashiError::Type("unique() requires list".into())),
                }
            }
            "flatten" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("flatten(list)".into()));
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
                        Ok(Value::List(Arc::new(result)))
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
                Ok(Value::List(Arc::new(result)))
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
                Ok(Value::List(Arc::new(v)))
            }
            _ => unreachable!(),
        }
    }
}
