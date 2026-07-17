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
    "hausman_robust",
    "hausman_r",
    "ftest_robust",
    "f_robust",
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
    "transformer",
    "transformer_ts",
    "dr_learner",
    "drlearner",
    "bart",
    "bayesian_trees",
    "gp",
    "gaussian_process",
    "tmle",
    "orf",
    "orthogonal_forest",
    "spectral",
    "spectral_clustering",
    "isotonic",
    "isotonic_reg",
    "causal_impact",
    "causalimpact",
    "mice_chained",
    "mice_eq",
    "kmeans",
    "k_means",
    "bayes_lm",
    "bayesian_lm",
    "causal_impact",
    "causalimpact",
    "dbscan",
    "dbscan_clust",
    "gmm_clust",
    "gmm_clustering",
    "reg_path",
    "regpath",
    "qrf_inf",
    "qrf_inference",
    "hclust",
    "hierarchical",
    "tsne",
    "t_sne",
    "umap",
    "biplot",
    "pca_biplot",
    "hausman_robust",
    "hausman_r",
    "ftest_robust",
    "f_robust",
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

    fn gf(&self, v: f64) -> Value {
        Value::List(Arc::new(vec![Value::Float(v)]))
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
                    Value::VarResult(r) => {
                        // VAR: params is (1+k*p) x k matrix — flatten column by column
                        let k = r.n_vars;
                        let p = r.lags;
                        let n_coef = (1 + p * k) * k;
                        let mut params = ndarray::Array1::<f64>::zeros(n_coef);
                        let mut ses = ndarray::Array1::<f64>::zeros(n_coef);
                        let mut names: Vec<String> = Vec::with_capacity(n_coef);
                        let mut idx = 0;
                        for eq in 0..k {
                            for row in 0..(1 + p * k) {
                                params[idx] = r.params[(row, eq)];
                                ses[idx] = r.std_errors[(row, eq)];
                                if row == 0 {
                                    names.push(format!("const_{}", r.var_names[eq]));
                                } else {
                                    let lag = (row - 1) / k;
                                    let src = (row - 1) % k;
                                    names.push(format!(
                                        "L{}.{}_{}",
                                        lag + 1,
                                        r.var_names[src],
                                        r.var_names[eq]
                                    ));
                                }
                                idx += 1;
                            }
                        }
                        let tv = ndarray::Array1::<f64>::zeros(n_coef);
                        let pv = ndarray::Array1::<f64>::zeros(n_coef);
                        map = self.build_tidy_simple(names, &params, &ses, &tv, &pv);
                    }
                    Value::VecmResult(r) => {
                        // VECM: alpha (r x k), beta (r x k), gamma (k*(p-1) x k)
                        let k = r.n_vars;
                        let rank = r.rank;
                        let p = r.lags;
                        let n_alpha = rank * k;
                        let n_beta = rank * k;
                        let n_gamma = k * (p.saturating_sub(1)) * k;
                        let n_total = n_alpha + n_beta + n_gamma;
                        let mut params = ndarray::Array1::<f64>::zeros(n_total);
                        let mut ses = ndarray::Array1::<f64>::zeros(n_total);
                        let mut names: Vec<String> = Vec::with_capacity(n_total);
                        let mut idx = 0;
                        for j in 0..k {
                            for i in 0..rank {
                                params[idx] = r.alpha[(i, j)];
                                ses[idx] = r.std_errors_alpha[(i, j)];
                                let vn = r
                                    .variable_names
                                    .get(j)
                                    .cloned()
                                    .unwrap_or_else(|| format!("v{j}"));
                                names.push(format!("alpha_{}_{}", i + 1, vn));
                                idx += 1;
                            }
                        }
                        for j in 0..k {
                            for i in 0..rank {
                                params[idx] = r.beta[(i, j)];
                                ses[idx] = r.std_errors_beta[(i, j)];
                                let vn = r
                                    .variable_names
                                    .get(j)
                                    .cloned()
                                    .unwrap_or_else(|| format!("v{j}"));
                                names.push(format!("beta_{}_{}", i + 1, vn));
                                idx += 1;
                            }
                        }
                        for j in 0..k {
                            for i in 0..k * p.saturating_sub(1) {
                                params[idx] = r.gamma[(i, j)];
                                ses[idx] = r.std_errors_gamma[(i, j)];
                                let vn = r
                                    .variable_names
                                    .get(j)
                                    .cloned()
                                    .unwrap_or_else(|| format!("v{j}"));
                                names.push(format!("gamma_{}_{}", i + 1, vn));
                                idx += 1;
                            }
                        }
                        let tv = ndarray::Array1::<f64>::zeros(n_total);
                        let pv = ndarray::Array1::<f64>::zeros(n_total);
                        map = self.build_tidy_simple(names, &params, &ses, &tv, &pv);
                    }
                    Value::SysGmmResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::FE2SLSResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::PcseResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::PanelGlsResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::GlsarResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::RecursiveLSResult(r) => {
                        let names: Vec<String> =
                            (0..r.params.len()).map(|i| format!("beta{}", i)).collect();
                        let se = ndarray::Array1::<f64>::zeros(r.params.len());
                        let tv = ndarray::Array1::<f64>::zeros(r.params.len());
                        let pv = ndarray::Array1::<f64>::zeros(r.params.len());
                        map = self.build_tidy_simple(names, &r.params, &se, &tv, &pv);
                    }
                    Value::CoxResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::ConditionalResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::GamResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.params,
                            &r.std_errors,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::MixedResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone().unwrap_or_default(),
                            &r.fixed_effects,
                            &r.fixed_se,
                            &r.z_values,
                            &r.p_values,
                        );
                    }
                    Value::ZeroInflatedResult(r) => {
                        // Combine count and inflate params
                        let nc = r.count_params.len();
                        let ni = r.inflate_params.len();
                        let n = nc + ni;
                        let mut params = ndarray::Array1::<f64>::zeros(n);
                        let mut ses = ndarray::Array1::<f64>::zeros(n);
                        let mut tv = ndarray::Array1::<f64>::zeros(n);
                        let mut pv = ndarray::Array1::<f64>::zeros(n);
                        let mut names: Vec<String> = Vec::with_capacity(n);
                        let count_names = r.count_var_names.clone().unwrap_or_default();
                        let inflate_names = r.inflate_var_names.clone().unwrap_or_default();
                        for i in 0..nc {
                            params[i] = r.count_params[i];
                            ses[i] = r.count_std_errors[i];
                            tv[i] = r.count_z_values[i];
                            pv[i] = r.count_p_values[i];
                            let nm = count_names
                                .get(i)
                                .cloned()
                                .unwrap_or_else(|| format!("x{i}"));
                            names.push(format!("count_{nm}"));
                        }
                        for i in 0..ni {
                            params[nc + i] = r.inflate_params[i];
                            ses[nc + i] = r.inflate_std_errors[i];
                            tv[nc + i] = r.inflate_z_values[i];
                            pv[nc + i] = r.inflate_p_values[i];
                            let nm = inflate_names
                                .get(i)
                                .cloned()
                                .unwrap_or_else(|| format!("x{i}"));
                            names.push(format!("inflate_{nm}"));
                        }
                        map = self.build_tidy_simple(names, &params, &ses, &tv, &pv);
                    }
                    Value::AutoRegResult(r) => {
                        map = self.build_tidy_simple(
                            r.param_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::ArdlResult(r) => {
                        map = self.build_tidy_simple(
                            r.param_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::DidResult(r) => {
                        map = self.build_tidy_simple(
                            r.variable_names.clone(),
                            &r.params,
                            &r.std_errors,
                            &r.t_values,
                            &r.p_values,
                        );
                    }
                    Value::ThresholdResult(r) => {
                        // Combine regime1 and regime2 params
                        let n1 = r.params_regime1.len();
                        let n2 = r.params_regime2.len();
                        let n = n1 + n2 + 1;
                        let mut params = ndarray::Array1::<f64>::zeros(n);
                        let mut names: Vec<String> = Vec::with_capacity(n);
                        for i in 0..n1 {
                            params[i] = r.params_regime1[i];
                            names.push(format!("regime1_x{}", i));
                        }
                        for i in 0..n2 {
                            params[n1 + i] = r.params_regime2[i];
                            names.push(format!("regime2_x{}", i));
                        }
                        params[n - 1] = r.threshold_gamma;
                        names.push("threshold".into());
                        let se = ndarray::Array1::<f64>::zeros(n);
                        let tv = ndarray::Array1::<f64>::zeros(n);
                        let pv = ndarray::Array1::<f64>::zeros(n);
                        map = self.build_tidy_simple(names, &params, &se, &tv, &pv);
                    }
                    Value::RdResult(r) => {
                        let mut var = vec!["tau".to_string()];
                        let mut coef = vec![r.tau];
                        let mut se = vec![r.se];
                        let mut t = vec![r.z];
                        let mut p = vec![r.p_value];
                        let mut cl = vec![r.ci_lower];
                        let mut cu = vec![r.ci_upper];
                        if r.is_fuzzy {
                            if let Some(ft) = r.first_stage_tau {
                                var.push("first_stage_tau".into());
                                coef.push(ft);
                                se.push(r.first_stage_se.unwrap_or(f64::NAN));
                                t.push(f64::NAN);
                                p.push(f64::NAN);
                                cl.push(f64::NAN);
                                cu.push(f64::NAN);
                            }
                        }
                        let params = ndarray::Array1::from_vec(coef);
                        let se = ndarray::Array1::from_vec(se);
                        let t = ndarray::Array1::from_vec(t);
                        let p = ndarray::Array1::from_vec(p);
                        let cl = ndarray::Array1::from_vec(cl);
                        let cu = ndarray::Array1::from_vec(cu);
                        map = self.build_tidy_coef_map(var, &params, &se, &t, &p, &cl, &cu);
                    }
                    Value::SynthResult(r) => {
                        let mut var = Vec::new();
                        let mut coef = Vec::new();
                        for (id, w) in &r.weights {
                            var.push(id.clone());
                            coef.push(*w);
                        }
                        let n = coef.len();
                        let params = ndarray::Array1::from_vec(coef);
                        let se = ndarray::Array1::from_vec(vec![f64::NAN; n]);
                        let t = ndarray::Array1::from_vec(vec![f64::NAN; n]);
                        let p = ndarray::Array1::from_vec(vec![f64::NAN; n]);
                        let names: Vec<String> = var.clone();
                        map = self.build_tidy_simple(names, &params, &se, &t, &p);
                    }
                    Value::PsmResult(r) => {
                        let mut cov = Vec::new();
                        let mut mt = Vec::new();
                        let mut mcr = Vec::new();
                        let mut mcm = Vec::new();
                        let mut smdb = Vec::new();
                        let mut smda = Vec::new();
                        for b in &r.balance {
                            cov.push(Value::Str(b.covariate.clone()));
                            mt.push(Value::Float(b.mean_treated));
                            mcr.push(Value::Float(b.mean_control_raw));
                            mcm.push(Value::Float(b.mean_control_matched));
                            smdb.push(Value::Float(b.smd_before));
                            smda.push(Value::Float(b.smd_after));
                        }
                        map.insert("covariate".into(), Value::List(Arc::new(cov)));
                        map.insert("mean_treated".into(), Value::List(Arc::new(mt)));
                        map.insert("mean_control_raw".into(), Value::List(Arc::new(mcr)));
                        map.insert("mean_control_matched".into(), Value::List(Arc::new(mcm)));
                        map.insert("smd_before".into(), Value::List(Arc::new(smdb)));
                        map.insert("smd_after".into(), Value::List(Arc::new(smda)));
                    }
                    Value::MNLogitResult(r) => {
                        let k = r.params.nrows();
                        let j = r.params.ncols();
                        let mut var = Vec::new();
                        let mut outcome = Vec::new();
                        let mut coef = Vec::new();
                        let mut se = Vec::new();
                        let mut z = Vec::new();
                        let mut p = Vec::new();
                        let vnames = r.variable_names.clone().unwrap_or_default();
                        for col in 0..j {
                            let out = r
                                .category_labels
                                .get(col)
                                .map(|v| format!("{v:.0}"))
                                .unwrap_or_else(|| format!("cat{col}"));
                            for row in 0..k {
                                let name = vnames
                                    .get(row)
                                    .cloned()
                                    .unwrap_or_else(|| format!("x{row}"));
                                var.push(name);
                                outcome.push(out.clone());
                                coef.push(r.params[[row, col]]);
                                se.push(r.std_errors[[row, col]]);
                                z.push(r.z_values[[row, col]]);
                                p.push(r.p_values[[row, col]]);
                            }
                        }
                        let n = coef.len();
                        map.insert(
                            "variable".into(),
                            Value::List(Arc::new(var.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "outcome".into(),
                            Value::List(Arc::new(outcome.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "coef".into(),
                            Value::List(Arc::new(coef.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "std_err".into(),
                            Value::List(Arc::new(se.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "z".into(),
                            Value::List(Arc::new(z.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "p_value".into(),
                            Value::List(Arc::new(p.into_iter().map(Value::Float).collect())),
                        );
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::KMResult(r) => {
                        let n = r.times.len();
                        let mut time = Vec::new();
                        let mut surv = Vec::new();
                        let mut se = Vec::new();
                        let mut cl = Vec::new();
                        let mut cu = Vec::new();
                        for i in 0..n {
                            time.push(Value::Float(r.times[i]));
                            surv.push(Value::Float(r.survival_probs[i]));
                            se.push(Value::Float(r.std_errors[i]));
                            cl.push(Value::Float(r.conf_lower[i]));
                            cu.push(Value::Float(r.conf_upper[i]));
                        }
                        map.insert("time".into(), Value::List(Arc::new(time)));
                        map.insert("survival".into(), Value::List(Arc::new(surv)));
                        map.insert("std_err".into(), Value::List(Arc::new(se)));
                        map.insert("conf_low".into(), Value::List(Arc::new(cl)));
                        map.insert("conf_high".into(), Value::List(Arc::new(cu)));
                    }
                    Value::SurResult(m) => {
                        let r = &m.result;
                        let mut eq_vec = Vec::new();
                        let mut var = Vec::new();
                        let mut coef = Vec::new();
                        let mut se = Vec::new();
                        let mut t = Vec::new();
                        let mut p = Vec::new();
                        for (ei, eq) in r.equations.iter().enumerate() {
                            let names = m.eq_var_names.get(ei).cloned().unwrap_or_default();
                            for i in 0..eq.params.len() {
                                eq_vec.push(eq.name.clone());
                                var.push(names.get(i).cloned().unwrap_or_else(|| format!("x{i}")));
                                coef.push(eq.params[i]);
                                se.push(eq.std_errors[i]);
                                t.push(eq.t_values[i]);
                                p.push(eq.p_values[i]);
                            }
                        }
                        let n = coef.len();
                        map.insert(
                            "equation".into(),
                            Value::List(Arc::new(eq_vec.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "variable".into(),
                            Value::List(Arc::new(var.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "coef".into(),
                            Value::List(Arc::new(coef.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "std_err".into(),
                            Value::List(Arc::new(se.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "t".into(),
                            Value::List(Arc::new(t.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "p_value".into(),
                            Value::List(Arc::new(p.into_iter().map(Value::Float).collect())),
                        );
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::ThreeSLSResult(m) => {
                        let r = &m.result;
                        let mut eq_vec = Vec::new();
                        let mut var = Vec::new();
                        let mut coef = Vec::new();
                        let mut se = Vec::new();
                        let mut t = Vec::new();
                        let mut p = Vec::new();
                        for (ei, eq) in r.equations.iter().enumerate() {
                            let names = m.eq_var_names.get(ei).cloned().unwrap_or_default();
                            for i in 0..eq.params.len() {
                                eq_vec.push(eq.name.clone());
                                var.push(names.get(i).cloned().unwrap_or_else(|| format!("x{i}")));
                                coef.push(eq.params[i]);
                                se.push(eq.std_errors[i]);
                                t.push(eq.t_values[i]);
                                p.push(eq.p_values[i]);
                            }
                        }
                        let n = coef.len();
                        map.insert(
                            "equation".into(),
                            Value::List(Arc::new(eq_vec.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "variable".into(),
                            Value::List(Arc::new(var.into_iter().map(Value::Str).collect())),
                        );
                        map.insert(
                            "coef".into(),
                            Value::List(Arc::new(coef.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "std_err".into(),
                            Value::List(Arc::new(se.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "t".into(),
                            Value::List(Arc::new(t.into_iter().map(Value::Float).collect())),
                        );
                        map.insert(
                            "p_value".into(),
                            Value::List(Arc::new(p.into_iter().map(Value::Float).collect())),
                        );
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::SVarResult(r) => {
                        let mut matrix = Vec::new();
                        let mut row = Vec::new();
                        let mut col = Vec::new();
                        let mut value = Vec::new();
                        let k = r.a_matrix.nrows();
                        let vnames = r.var_result.var_names.clone();
                        for i in 0..k {
                            for j in 0..k {
                                matrix.push(Value::Str("A".into()));
                                row.push(Value::Str(
                                    vnames.get(i).cloned().unwrap_or_else(|| format!("v{i}")),
                                ));
                                col.push(Value::Str(
                                    vnames.get(j).cloned().unwrap_or_else(|| format!("v{j}")),
                                ));
                                value.push(Value::Float(r.a_matrix[[i, j]]));
                            }
                        }
                        for i in 0..r.b_matrix.nrows() {
                            for j in 0..r.b_matrix.ncols() {
                                matrix.push(Value::Str("B".into()));
                                row.push(Value::Str(
                                    vnames.get(i).cloned().unwrap_or_else(|| format!("v{i}")),
                                ));
                                col.push(Value::Str(
                                    vnames.get(j).cloned().unwrap_or_else(|| format!("v{j}")),
                                ));
                                value.push(Value::Float(r.b_matrix[[i, j]]));
                            }
                        }
                        let n = value.len();
                        map.insert("matrix".into(), Value::List(Arc::new(matrix)));
                        map.insert("row".into(), Value::List(Arc::new(row)));
                        map.insert("col".into(), Value::List(Arc::new(col)));
                        map.insert("value".into(), Value::List(Arc::new(value)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::VarmaResult(r) => {
                        let mut type_vec = Vec::new();
                        let mut lag_vec = Vec::new();
                        let mut from_vec = Vec::new();
                        let mut to_vec = Vec::new();
                        let mut value_vec = Vec::new();
                        let k = r.n_vars;
                        let vnames: Vec<String> = (0..k).map(|i| format!("y{}", i + 1)).collect();
                        // AR: rows 0..1+p*k, cols 0..k
                        for (col, _vn) in vnames.iter().enumerate().take(k) {
                            type_vec.push(Value::Str("const".into()));
                            lag_vec.push(Value::Int(0));
                            from_vec.push(Value::Str("-".into()));
                            to_vec.push(Value::Str(vnames[col].clone()));
                            value_vec.push(Value::Float(r.ar_params[[0, col]]));
                        }
                        for l in 0..r.p_lags {
                            for src in 0..k {
                                for (col, _vn) in vnames.iter().enumerate().take(k) {
                                    let row = 1 + l * k + src;
                                    type_vec.push(Value::Str("AR".into()));
                                    lag_vec.push(Value::Int((l + 1) as i64));
                                    from_vec.push(Value::Str(vnames[src].clone()));
                                    to_vec.push(Value::Str(vnames[col].clone()));
                                    value_vec.push(Value::Float(r.ar_params[[row, col]]));
                                }
                            }
                        }
                        for l in 0..r.q_lags {
                            for src in 0..k {
                                for (col, _vn) in vnames.iter().enumerate().take(k) {
                                    let row = l * k + src;
                                    type_vec.push(Value::Str("MA".into()));
                                    lag_vec.push(Value::Int((l + 1) as i64));
                                    from_vec.push(Value::Str(vnames[src].clone()));
                                    to_vec.push(Value::Str(vnames[col].clone()));
                                    value_vec.push(Value::Float(r.ma_params[[row, col]]));
                                }
                            }
                        }
                        if let Some(ex) = &r.exog_params {
                            for ex_i in 0..ex.nrows() {
                                for (col, _vn) in vnames.iter().enumerate().take(k) {
                                    type_vec.push(Value::Str("exog".into()));
                                    lag_vec.push(Value::Int(0));
                                    from_vec.push(Value::Str(format!("ex{ex_i}")));
                                    to_vec.push(Value::Str(vnames[col].clone()));
                                    value_vec.push(Value::Float(ex[[ex_i, col]]));
                                }
                            }
                        }
                        let n = value_vec.len();
                        map.insert("type".into(), Value::List(Arc::new(type_vec)));
                        map.insert("lag".into(), Value::List(Arc::new(lag_vec)));
                        map.insert("from".into(), Value::List(Arc::new(from_vec)));
                        map.insert("to".into(), Value::List(Arc::new(to_vec)));
                        map.insert("value".into(), Value::List(Arc::new(value_vec)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::MarkovResult(r) => {
                        let mut regime = Vec::new();
                        let mut parameter = Vec::new();
                        let mut value = Vec::new();
                        for (i, params) in r.regime_params.iter().enumerate() {
                            for (j, &v) in params.iter().enumerate() {
                                regime.push(Value::Int((i + 1) as i64));
                                parameter.push(Value::Str(if j == 0 {
                                    "intercept".into()
                                } else {
                                    format!("ar{}", j - 1)
                                }));
                                value.push(Value::Float(v));
                            }
                            regime.push(Value::Int((i + 1) as i64));
                            parameter.push(Value::Str("variance".into()));
                            value.push(Value::Float(r.regime_variances[i]));
                        }
                        let n = value.len();
                        map.insert("regime".into(), Value::List(Arc::new(regime)));
                        map.insert("parameter".into(), Value::List(Arc::new(parameter)));
                        map.insert("value".into(), Value::List(Arc::new(value)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::MSARResult(r) => {
                        let mut regime = Vec::new();
                        let mut parameter = Vec::new();
                        let mut value = Vec::new();
                        for i in 0..r.k_regimes {
                            regime.push(Value::Int((i + 1) as i64));
                            parameter.push(Value::Str("intercept".into()));
                            value.push(Value::Float(r.regime_means[i]));
                            for p in 0..r.ar_order {
                                regime.push(Value::Int((i + 1) as i64));
                                parameter.push(Value::Str(format!("ar{}", p + 1)));
                                value.push(Value::Float(r.ar_params[[i, p]]));
                            }
                            regime.push(Value::Int((i + 1) as i64));
                            parameter.push(Value::Str("sigma".into()));
                            value.push(Value::Float(r.regime_sigmas[i]));
                        }
                        let n = value.len();
                        map.insert("regime".into(), Value::List(Arc::new(regime)));
                        map.insert("parameter".into(), Value::List(Arc::new(parameter)));
                        map.insert("value".into(), Value::List(Arc::new(value)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::PcaResult(m) => {
                        let r = &m.result;
                        let mut var = Vec::new();
                        let mut pc = Vec::new();
                        let mut loading = Vec::new();
                        let k = m.var_names.len();
                        let c = r.n_components;
                        for j in 0..c {
                            for i in 0..k {
                                var.push(Value::Str(m.var_names[i].clone()));
                                pc.push(Value::Str(format!("PC{}", j + 1)));
                                loading.push(Value::Float(r.loadings[[i, j]]));
                            }
                        }
                        let n = loading.len();
                        map.insert("variable".into(), Value::List(Arc::new(var)));
                        map.insert("component".into(), Value::List(Arc::new(pc)));
                        map.insert("loading".into(), Value::List(Arc::new(loading)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::FactorResult(m) => {
                        let r = &m.result;
                        let mut var = Vec::new();
                        let mut factor = Vec::new();
                        let mut loading = Vec::new();
                        let mut comm = Vec::new();
                        let mut uniq = Vec::new();
                        let k = m.var_names.len();
                        let f = r.n_factors;
                        for j in 0..f {
                            for i in 0..k {
                                var.push(Value::Str(m.var_names[i].clone()));
                                factor.push(Value::Str(format!("F{}", j + 1)));
                                loading.push(Value::Float(r.loadings[[i, j]]));
                                comm.push(Value::Float(r.communalities[i]));
                                uniq.push(Value::Float(r.uniquenesses[i]));
                            }
                        }
                        let n = loading.len();
                        map.insert("variable".into(), Value::List(Arc::new(var)));
                        map.insert("factor".into(), Value::List(Arc::new(factor)));
                        map.insert("loading".into(), Value::List(Arc::new(loading)));
                        map.insert("communality".into(), Value::List(Arc::new(comm)));
                        map.insert("uniqueness".into(), Value::List(Arc::new(uniq)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::DFMResult(m) => {
                        let r = &m.result;
                        let mut var = Vec::new();
                        let mut factor = Vec::new();
                        let mut loading = Vec::new();
                        let k = m.var_names.len();
                        let f = r.n_factors;
                        for j in 0..f {
                            for i in 0..k {
                                var.push(Value::Str(m.var_names[i].clone()));
                                factor.push(Value::Str(format!("F{}", j + 1)));
                                loading.push(Value::Float(r.factor_loadings[[i, j]]));
                            }
                        }
                        let n = loading.len();
                        map.insert("variable".into(), Value::List(Arc::new(var)));
                        map.insert("factor".into(), Value::List(Arc::new(factor)));
                        map.insert("loading".into(), Value::List(Arc::new(loading)));
                        let nan_col = vec![Value::Float(f64::NAN); n];
                        map.insert("conf_low".into(), Value::List(Arc::new(nan_col.clone())));
                        map.insert("conf_high".into(), Value::List(Arc::new(nan_col)));
                    }
                    Value::DecompResult(_)
                    | Value::MstlResult(_)
                    | Value::UCResult(_)
                    | Value::MiceResult(_)
                    | Value::LowessResult(_) => {
                        // No coefficient-like parameters; return empty tidy table
                        map.insert("variable".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("coef".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("std_err".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("t".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("p_value".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("conf_low".into(), Value::List(Arc::new(Vec::new())));
                        map.insert("conf_high".into(), Value::List(Arc::new(Vec::new())));
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
                        let scalar = |v: f64| self.gf(v);
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
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert("sigma".into(), scalar(r.sigma));
                    }
                    Value::BinaryResult(m) => {
                        let r = &m.result;
                        let scalar = |v: f64| self.gf(v);
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("n".into(), Value::List(Arc::new(vec![Value::Int(0)])));
                        // n not stored
                    }
                    Value::PanelResult(r) => {
                        let scalar = |v: f64| self.gf(v);
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
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(r.r_squared_overall));
                        map.insert("sigma_u".into(), scalar(r.sigma_u));
                        map.insert("sigma_e".into(), scalar(r.sigma_e));
                        map.insert("theta".into(), scalar(r.theta));
                    }
                    Value::GmmResult(r) => {
                        let scalar = |v: f64| self.gf(v);
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
                        let scalar = |v: f64| self.gf(v);
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
                        let scalar = |v: f64| self.gf(v);
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
                        let scalar = |v: f64| self.gf(v);
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
                        let scalar = |v: f64| self.gf(v);
                        map.insert("tau".into(), scalar(r.tau));
                        map.insert("pseudo_r2".into(), scalar(r.r_squared));
                    }
                    Value::TobitResult(r) => {
                        let scalar = |v: f64| self.gf(v);
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
                        let scalar = |v: f64| self.gf(v);
                        map.insert("rho".into(), scalar(r.rho));
                        map.insert("delta".into(), scalar(r.delta));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::OrderedResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                    }
                    Value::PenalizedResult(m) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(m.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(m.n_obs as i64)])),
                        );
                        map.insert("alpha".into(), scalar(m.alpha));
                    }
                    Value::ArimaResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("sigma2".into(), scalar(r.sigma2));
                    }
                    Value::GarchResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                    }
                    Value::VarResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::VecmResult(r) => {
                        map.insert(
                            "rank".into(),
                            Value::List(Arc::new(vec![Value::Int(r.rank as i64)])),
                        );
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::SysGmmResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("sargan_stat".into(), scalar(r.sargan_stat));
                        map.insert("sargan_p".into(), scalar(r.sargan_pvalue));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(
                                (r.n_obs_fd + r.n_obs_lev) as i64,
                            )])),
                        );
                    }
                    Value::FE2SLSResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert("sigma".into(), scalar(r.sigma));
                    }
                    Value::PcseResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert("sigma".into(), scalar(r.sigma));
                    }
                    Value::PanelGlsResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert("sigma".into(), scalar(r.sigma));
                    }
                    Value::GlsarResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::RecursiveLSResult(r) => {
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::CoxResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("concordance".into(), scalar(r.concordance));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::ConditionalResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::GamResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("gcv".into(), scalar(r.gcv_score));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::MixedResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_groups".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_groups as i64)])),
                        );
                    }
                    Value::ZeroInflatedResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        if let Some(a) = r.alpha {
                            map.insert("alpha".into(), scalar(a));
                        }
                    }
                    Value::AutoRegResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert("adj_r2".into(), scalar(r.adj_r_squared));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::ArdlResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert("adj_r2".into(), scalar(r.adj_r_squared));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::DidResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("att".into(), scalar(r.att));
                        map.insert("r2".into(), scalar(r.r_squared));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::ThresholdResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("threshold".into(), scalar(r.threshold_gamma));
                        map.insert("r2".into(), scalar(r.r_squared));
                    }
                    Value::EtsResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("sse".into(), scalar(r.sse));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::LocalLevelResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("sigma_obs".into(), scalar(r.sigma_obs));
                        map.insert("sigma_state".into(), scalar(r.sigma_state));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::BetaResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                        map.insert("precision".into(), scalar(r.precision_param));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::GeeResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("scale".into(), scalar(r.scale));
                        map.insert("qic".into(), scalar(r.qic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_groups".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_groups as i64)])),
                        );
                    }
                    Value::RlmResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("scale".into(), scalar(r.scale));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "converged".into(),
                            Value::List(Arc::new(vec![Value::Bool(r.converged)])),
                        );
                    }
                    Value::AbResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("sargan_stat".into(), scalar(r.sargan_stat));
                        map.insert("sargan_p".into(), scalar(r.sargan_pvalue));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_entities".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_entities as i64)])),
                        );
                        map.insert(
                            "n_instruments".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_instruments as i64)])),
                        );
                    }
                    Value::RollingResult(r) => {
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "window".into(),
                            Value::List(Arc::new(vec![Value::Int(r.window as i64)])),
                        );
                    }
                    Value::RdResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("tau".into(), scalar(r.tau));
                        map.insert("se".into(), scalar(r.se));
                        map.insert("p_value".into(), scalar(r.p_value));
                        map.insert("bandwidth".into(), scalar(r.bandwidth));
                        map.insert("cutoff".into(), scalar(r.cutoff));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_total as i64)])),
                        );
                        map.insert(
                            "n_left".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_left as i64)])),
                        );
                        map.insert(
                            "n_right".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_right as i64)])),
                        );
                        map.insert(
                            "is_fuzzy".into(),
                            Value::List(Arc::new(vec![Value::Bool(r.is_fuzzy)])),
                        );
                    }
                    Value::PsmResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("att".into(), scalar(r.att));
                        map.insert("se".into(), scalar(r.se));
                        map.insert("p_value".into(), scalar(r.p_value));
                        map.insert(
                            "n_treated".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_treated as i64)])),
                        );
                        map.insert(
                            "n_control".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_control as i64)])),
                        );
                        map.insert(
                            "n_matched_treated".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_matched_treated as i64)])),
                        );
                        map.insert(
                            "k".into(),
                            Value::List(Arc::new(vec![Value::Int(r.k as i64)])),
                        );
                    }
                    Value::MNLogitResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_categories".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_categories as i64)])),
                        );
                    }
                    Value::SurResult(m) => {
                        let r = &m.result;
                        let scalar = |v: f64| self.gf(v);
                        map.insert("system_r2".into(), scalar(r.system_r2));
                        map.insert(
                            "n_equations".into(),
                            Value::List(Arc::new(vec![Value::Int(r.equations.len() as i64)])),
                        );
                    }
                    Value::ThreeSLSResult(m) => {
                        let r = &m.result;
                        let scalar = |v: f64| self.gf(v);
                        map.insert("system_r2".into(), scalar(r.system_r2));
                        map.insert(
                            "n_equations".into(),
                            Value::List(Arc::new(vec![Value::Int(r.equations.len() as i64)])),
                        );
                    }
                    Value::SVarResult(r) => {
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.var_result.n_obs as i64)])),
                        );
                        map.insert(
                            "n_vars".into(),
                            Value::List(Arc::new(vec![Value::Int(r.var_result.n_vars as i64)])),
                        );
                        map.insert(
                            "lags".into(),
                            Value::List(Arc::new(vec![Value::Int(r.var_result.lags as i64)])),
                        );
                        map.insert(
                            "identification".into(),
                            Value::List(Arc::new(vec![Value::Str(r.identification.clone())])),
                        );
                    }
                    Value::VarmaResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_vars".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_vars as i64)])),
                        );
                        map.insert(
                            "p_lags".into(),
                            Value::List(Arc::new(vec![Value::Int(r.p_lags as i64)])),
                        );
                        map.insert(
                            "q_lags".into(),
                            Value::List(Arc::new(vec![Value::Int(r.q_lags as i64)])),
                        );
                    }
                    Value::MarkovResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_regimes".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_regimes as i64)])),
                        );
                    }
                    Value::MSARResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "k_regimes".into(),
                            Value::List(Arc::new(vec![Value::Int(r.k_regimes as i64)])),
                        );
                        map.insert(
                            "ar_order".into(),
                            Value::List(Arc::new(vec![Value::Int(r.ar_order as i64)])),
                        );
                    }
                    Value::PcaResult(m) => {
                        let r = &m.result;
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_components".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_components as i64)])),
                        );
                        map.insert(
                            "n_variables".into(),
                            Value::List(Arc::new(vec![Value::Int(m.var_names.len() as i64)])),
                        );
                    }
                    Value::FactorResult(m) => {
                        let r = &m.result;
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_factors".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_factors as i64)])),
                        );
                        map.insert(
                            "n_variables".into(),
                            Value::List(Arc::new(vec![Value::Int(m.var_names.len() as i64)])),
                        );
                    }
                    Value::DFMResult(m) => {
                        let r = &m.result;
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_vars".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_vars as i64)])),
                        );
                        map.insert(
                            "n_factors".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_factors as i64)])),
                        );
                        map.insert(
                            "factor_order".into(),
                            Value::List(Arc::new(vec![Value::Int(r.factor_order as i64)])),
                        );
                    }
                    Value::DecompResult(r) => {
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.observed.len() as i64)])),
                        );
                        map.insert(
                            "model".into(),
                            Value::List(Arc::new(vec![Value::Str(r.model.clone())])),
                        );
                    }
                    Value::MstlResult(r) => {
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_periods".into(),
                            Value::List(Arc::new(vec![Value::Int(r.periods.len() as i64)])),
                        );
                    }
                    Value::UCResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert("log_lik".into(), scalar(r.log_likelihood));
                        map.insert("aic".into(), scalar(r.aic));
                        map.insert("bic".into(), scalar(r.bic));
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                    }
                    Value::MiceResult(r) => {
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_vars".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_vars as i64)])),
                        );
                        map.insert(
                            "n_imputations".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_imputations as i64)])),
                        );
                    }
                    Value::LowessResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert("frac".into(), scalar(r.frac));
                    }
                    Value::KMResult(r) => {
                        let scalar = |v: f64| self.gf(v);
                        map.insert(
                            "n".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                        );
                        map.insert(
                            "n_events".into(),
                            Value::List(Arc::new(vec![Value::Int(r.n_events as i64)])),
                        );
                        map.insert("median_survival".into(), scalar(r.median_survival));
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
