use super::helpers::*;
use super::*;
use std::sync::Arc;

pub const BUILTIN_NAMES: &[&str] = &[
    "mean",
    "sd",
    "min",
    "max",
    "sum",
    "total",
    "median",
    "variance",
    "quantile",
    "cov",
    "corr_pair",
    "abs",
    "sqrt",
    "ln",
    "log",
    "exp",
    "list_files",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "ceil",
    "floor",
    "round",
    "sign",
    "factorial",
    "comb",
    "int",
    "float",
    "str",
    "bool",
    "len",
    "first",
    "last",
    "shift",
    "typeof",
    "ols",
    "iv",
    "logit",
    "probit",
    "poisson",
    "nbreg",
    "tobit",
    "heckman",
    "fe",
    "re",
    "be",
    "fe2sls",
    "ab",
    "sysgmm",
    "pcse",
    "xtgls",
    "qreg",
    "rlm",
    "lasso",
    "ridge",
    "elasticnet",
    "cox",
    "arima",
    "autoreg",
    "ardl",
    "kalman",
    "var",
    "vecm",
    "varma",
    "svar",
    "garch",
    "glm",
    "gee",
    "mixed",
    "mlogit",
    "ologit",
    "oprobit",
    "clogit",
    "cpoisson",
    "gmm",
    "sur",
    "three_sls",
    "fmb",
    "did",
    "rd",
    "psm",
    "synth",
    "summarize",
    "tabulate",
    "tabstat",
    "correlate",
    "corr",
    "pwcorr",
    "describe",
    "codebook",
    "ttest",
    "ci",
    "centile",
    "count",
    "nrow",
    "filter",
    "sort",
    "drop",
    "keep",
    "select",
    "dropna",
    "rename",
    "merge",
    "append",
    "rbind",
    "collapse",
    "group_by",
    "reshape",
    "mutate",
    "generate",
    "pivot_longer",
    "pivot_wider",
    "anova",
    "pca",
    "factor",
    "manova",
    "cancorr",
    "kde",
    "lowess",
    "swilk",
    "sfrancia",
    "sktest",
    "omnibus",
    "dagostino",
    "vif",
    "predict",
    "esttab",
    "eststo",
    "margins",
    "test",
    "lincom",
    "nlcom",
    "bootstrap",
    "bootse",
    "acf",
    "pacf",
    "cusumtest",
    "akaike_weights",
    "lrtest",
    "estat_overid",
    "estat_endog",
    "estat_classification",
    "lroc",
    "estat_gof",
    "linktest",
    "xtlogit",
    "xtprobit",
    "xtpoisson",
    "eventstudy",
    "nls_exp",
    "nls_power",
    "nls_logistic",
    "nls_cobb_douglas",
    "nls_ces",
    "marginsplot",
    "spatial_sar",
    "spatial_sem",
    "double_ml",
    "dml",
    "sfa_production",
    "sfa_cost",
    "frontier",
    "panel_tobit",
    "panel_heckman",
    "spatial_panel_sar",
    "spatial_panel_sem",
    "bayes_sfa_production",
    "bayes_sfa_cost",
    "bayes_frontier",
    "midas",
    "tvp",
    "setar",
    "panel_qreg",
    "panel_quantile",
    "msvar",
    "ms_var",
    "favar",
    "spatial_durbin",
    "sdm",
    "johansen_break",
    "tvp_var",
    "spatial_durbin_error",
    "sdem",
    "fmols",
    "qvar",
    "quantile_var",
    "pstr",
    "modwt",
    "copula",
    "nardl",
    "pvar",
    "panel_var",
    "fcoef",
    "functional_coef",
    "dcc_garch",
    "dcc",
    "tvar",
    "threshold_var",
    "bvar",
    "bayesian_var",
    "mfvar",
    "mixed_freq_var",
    "tvcopula",
    "tv_copula",
    "sv",
    "stochastic_vol",
    "fapanel",
    "fa_panel",
    "hawkes",
    "rf",
    "random_forest",
    "gbm",
    "gradient_boosting",
    "mlp",
    "neural_net",
    "synthdid",
    "synthetic_did",
    "cuped",
    "qrf",
    "quantile_forest",
    "xgboost",
    "xgb",
    "dml_crossfit",
    "dml_cf",
    "bsc",
    "bayesian_sc",
    "lstm",
    "causalforest",
    "causal_forest",
    "grf",
    "generalized_rf",
    "conformal",
    "conformal_pred",
    "histogram",
    "boxplot",
    "kdensity",
    "qqplot",
    "scatter",
    "recode",
    "destring",
    "winsor",
    "label",
    "format",
    "print",
    "display",
    "source",
    "import",
    "install",
    "assert",
    "timer",
    "push",
    "pop",
    "reverse",
    "unique",
    "flatten",
    "chain",
    "join",
    "split",
    "contains",
    "starts_with",
    "ends_with",
    "lower",
    "upper",
    "trim",
    "substr",
    "replace",
    "regexm",
    "regexr",
    "regexs",
    "input",
    "load",
    "export",
    "write",
];

/// Type conversions (int/float/str/bool), date/time, list and dict builtins,
/// string functions, regex, aggregations over List, and scalar aggregations
/// (mean/sum/min/max/std/...) with `if=` support.
/// Extracted from `eval_call` (see src/lang/interpreter.rs).
impl Interpreter {
    /// Helper for `tidy`: build a tidy coefficient map from model result vectors.
    #[allow(clippy::too_many_arguments)]
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
            .map(|i| Value::Str(names.get(i).cloned().unwrap_or_else(|| format!("x{i}"))))
            .collect();
        let coef_col: Vec<Value> = params.iter().map(|&v| Value::Float(v)).collect();
        let se_col: Vec<Value> = std_errors.iter().map(|&v| Value::Float(v)).collect();
        let t_col: Vec<Value> = t_values.iter().map(|&v| Value::Float(v)).collect();
        let p_col: Vec<Value> = p_values.iter().map(|&v| Value::Float(v)).collect();
        let cl_col: Vec<Value> = conf_lower.iter().map(|&v| Value::Float(v)).collect();
        let cu_col: Vec<Value> = conf_upper.iter().map(|&v| Value::Float(v)).collect();

        let mut map = std::collections::HashMap::new();
        map.insert("variable".into(), Value::List(Arc::new(name_col)));
        map.insert("coef".into(), Value::List(Arc::new(coef_col)));
        map.insert("std_err".into(), Value::List(Arc::new(se_col)));
        map.insert("t".into(), Value::List(Arc::new(t_col)));
        map.insert("p_value".into(), Value::List(Arc::new(p_col)));
        map.insert("conf_low".into(), Value::List(Arc::new(cl_col)));
        map.insert("conf_high".into(), Value::List(Arc::new(cu_col)));
        map
    }

    /// Helper for `tidy`: build a tidy coefficient map without confidence intervals.
    fn build_tidy_simple(
        &self,
        names: Vec<String>,
        params: &ndarray::Array1<f64>,
        std_errors: &ndarray::Array1<f64>,
        stat_values: &ndarray::Array1<f64>,
        p_values: &ndarray::Array1<f64>,
    ) -> std::collections::HashMap<String, Value> {
        let n = params.len();
        let name_col: Vec<Value> = (0..n)
            .map(|i| Value::Str(names.get(i).cloned().unwrap_or_else(|| format!("x{i}"))))
            .collect();
        let coef_col: Vec<Value> = params.iter().map(|&v| Value::Float(v)).collect();
        let se_col: Vec<Value> = std_errors.iter().map(|&v| Value::Float(v)).collect();
        let stat_col: Vec<Value> = stat_values.iter().map(|&v| Value::Float(v)).collect();
        let p_col: Vec<Value> = p_values.iter().map(|&v| Value::Float(v)).collect();
        let nan_col: Vec<Value> = vec![Value::Float(f64::NAN); n];

        let mut map = std::collections::HashMap::new();
        map.insert("variable".into(), Value::List(Arc::new(name_col)));
        map.insert("coef".into(), Value::List(Arc::new(coef_col)));
        map.insert("std_err".into(), Value::List(Arc::new(se_col)));
        map.insert("t".into(), Value::List(Arc::new(stat_col)));
        map.insert("p_value".into(), Value::List(Arc::new(p_col)));
        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
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

            // ── List builtins ─────────────────────────────────────────────────
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

            // ── tidy: coefficient table from a model ───────────────────────────
            "tidy" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "tidy(model) requires 1 argument".into(),
                    ));
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
                                    if j == 0 {
                                        "const".into()
                                    } else {
                                        format!("x{j}")
                                    }
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
                            for (j, col) in coef_cols.iter_mut().enumerate().take(k) {
                                col.1.push(Value::Float(r.params_history[[t, j]]));
                            }
                        }
                        map.insert("date".into(), Value::List(Arc::new(date_col)));
                        map.insert("r2".into(), Value::List(Arc::new(r2_col)));
                        for (name, vals) in coef_cols {
                            map.insert(name, Value::List(Arc::new(vals)));
                        }
                    }
                    Value::IvResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::BinaryResult(m) => {
                        let r = &m.result;
                        map = self.build_tidy_simple(
                            m.coef_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::PanelResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::ReResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::GmmResult(r) => {
                        let names: Vec<String> =
                            (0..r.params.len()).map(|i| format!("x{i}")).collect();
                        map = self.build_tidy_simple(
                            names,
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::PoissonResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::NegBinResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::GlmResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::QuantileResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::TobitResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::HeckmanResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::OrderedResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::AbResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::PenalizedResult(m) => {
                        map = self.build_tidy_simple(
                            m.variable_names.clone(),
                            &m.params,
                            &m.std_errors,
                            &ndarray::Array1::from_vec(vec![0.0; m.params.len()]),
                            &ndarray::Array1::from_vec(vec![0.0; m.params.len()]),
                        );
                    }
                    Value::RlmResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::BetaResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::GeeResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.robust_se,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::ArimaResult(r) => {
                        // ARIMA has ar_params, ma_params, intercept — concatenate
                        let mut all_params = r.ar_params.to_vec();
                        all_params.extend(r.ma_params.iter().cloned());
                        all_params.push(r.intercept);
                        let params = ndarray::Array1::from_vec(all_params);
                        let p = r.p_values.len();
                        let se = if r.std_errors.len() >= p {
                            r.std_errors.slice(ndarray::s![..p]).to_owned()
                        } else {
                            ndarray::Array1::from_vec(vec![f64::NAN; p])
                        };
                        let tv = if r.t_values.len() >= p {
                            r.t_values.slice(ndarray::s![..p]).to_owned()
                        } else {
                            ndarray::Array1::from_vec(vec![f64::NAN; p])
                        };
                        let pv = if r.p_values.len() >= p {
                            r.p_values.slice(ndarray::s![..p]).to_owned()
                        } else {
                            ndarray::Array1::from_vec(vec![f64::NAN; p])
                        };
                        let names: Vec<String> = (0..params.len())
                            .map(|i| {
                                if i < r.ar_params.len() {
                                    format!("ar{}", i + 1)
                                } else if i < r.ar_params.len() + r.ma_params.len() {
                                    format!("ma{}", i - r.ar_params.len() + 1)
                                } else {
                                    "intercept".into()
                                }
                            })
                            .collect();
                        map = self.build_tidy_simple(names, &params, &se, &tv, &pv);
                    }
                    Value::GarchResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    _ => return Err(HayashiError::Type("tidy: unsupported model type".into())),
                }

                let df = self.dict_to_dataframe(&map)?;
                Ok(Value::DataFrame(Arc::new(df)))
            }

            // ── glance: model fit summary ──────────────────────────────────────
            "glance" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "glance(model) requires 1 argument".into(),
                    ));
                }
                let val = self.eval_expr(&args[0])?;
                let mut map = std::collections::HashMap::<String, Value>::new();

                match val {
                    Value::OlsResult(m) => {
                        let r = &m.result;
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert("adj_r2".into(), scalar(r.adj_r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert("f_stat".into(), scalar(r.f_statistic));
                        map.insert("prob_f".into(), scalar(r.prob_f));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("sigma".into(), scalar(r.sigma));
                    }
                    Value::IvResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert("sigma".into(), scalar(r.sigma));
                    }
                    Value::BinaryResult(m) => {
                        let r = &m.result;
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("n".into(), Value::List(Arc::new(vec![Value::Int(0)])));
                        // n not stored
                    }
                    Value::PanelResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_entities".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_entities as i64)])),
                        );
                        map.insert("sigma".into(), scalar(r.sigma));
                    }
                    Value::ReResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("r2".into(), scalar(r.r_squared_overall));
                        map.insert("sigma_u".into(), scalar(r.sigma_u));
                        map.insert("sigma_e".into(), scalar(r.sigma_e));
                        map.insert("theta".into(), scalar(r.theta));
                    }
                    Value::GmmResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("j_stat".into(), scalar(r.j_stat));
                        map.insert("j_p_value".into(), scalar(r.j_p_value));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "df_overid".into(),
                            Value::List(Arc::new(vec![Value::Int(r.df_overid as i64)])),
                        );
                    }
                    Value::PoissonResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::NegBinResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                        map.insert("alpha".into(), scalar(r.alpha));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::GlmResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                        map.insert("deviance".into(), scalar(r.deviance));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::QuantileResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("tau".into(), scalar(r.tau));
                        map.insert("pseudo_r2".into(), scalar(r.r_squared));
                    }
                    Value::TobitResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_censored".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_censored as i64)])),
                        );
                    }
                    Value::HeckmanResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("rho".into(), scalar(r.rho));
                        map.insert("delta".into(), scalar(r.delta));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::OrderedResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                    }
                    Value::PenalizedResult(m) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("r2".into(), scalar(m.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(m.n_obs as i64)])),
                        );
                        map.insert("alpha".into(), scalar(m.alpha));
                    }
                    Value::ArimaResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("sigma2".into(), scalar(r.sigma2));
                    }
                    Value::GarchResult(r) => {
                        let scalar = |v: f64| Value::List(Arc::new(vec![Value::Float(v)]));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                    }
                    _ => return Err(HayashiError::Type("glance: unsupported model type".into())),
                }

                let df = self.dict_to_dataframe(&map)?;
                Ok(Value::DataFrame(Arc::new(df)))
            }

            // ── names: column names of a DataFrame ─────────────────────────────
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

            // ── String functions ────────────────────────────────────────────
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

            // substr(s, start [, length]) — 0-based char index
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

            // split(s, delimiter) → List of Str
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

            // str_replace(s, from, to) — "replace" is a keyword
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

            // ── Regex ─────────────────────────────────────────────────────────
            // regexm(s, pattern)            → 1 if match, 0 otherwise
            // regexr(s, pattern, replace)   → replace first occurrence
            // regexra(s, pattern, replace)  → replace all
            // regexs(s, pattern)            → extract first capture group
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

            // ── Aggregations over List ────────────────────────────────────────
            // "sum" is reserved for summarize(df) — Stata-style
            // "total" is the sum of a numeric list
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
                            return Ok(Some(Value::Float(val)));
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

            // ── New scalar aggregations (all support if = cond) ─────────────
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

            // ── Series methods (first-class column) ──────────────────────────
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

            // ── list_files(dir, [pattern]) ─────────────────────────────────────
            "list_files" => {
                let dir = if args.is_empty() {
                    ".".to_string()
                } else {
                    match self.eval_expr(&args[0])? {
                        Value::Str(s) => s,
                        _ => return Err(self.type_err("list_files: directory must be a string")),
                    }
                };
                let pattern = if args.len() > 1 {
                    match self.eval_expr(&args[1])? {
                        Value::Str(s) => Some(s),
                        _ => return Err(self.type_err("list_files: pattern must be a string")),
                    }
                } else {
                    None
                };

                let entries = std::fs::read_dir(&dir)
                    .map_err(|e| self.rt_err(format!("list_files: cannot read '{dir}': {e}")))?;

                let mut files: Vec<String> = Vec::new();
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if let Some(ref pat) = pattern {
                            if !name.contains(pat) {
                                continue;
                            }
                        }
                        files.push(path.to_string_lossy().to_string());
                    }
                }
                files.sort();
                Ok(Value::List(Arc::new(
                    files.into_iter().map(Value::Str).collect(),
                )))
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
