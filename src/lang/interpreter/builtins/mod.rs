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
    "lpdid",
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
    "rnormal",
    "rlognormal",
    "rskewnormal",
    "rcauchy",
    "rstudentt",
    "rt",
    "rchisq",
    "rf",
    "rbeta",
    "rgamma",
    "rexponential",
    "rweibull",
    "rpareto",
    "rpert",
    "rtriangular",
    "rfrechet",
    "rgumbel",
    "rinversegaussian",
    "rnig",
    "runiform",
    "rbernoulli",
    "rbinomial",
    "rpoisson",
    "rgeometric",
    "rhypergeometric",
    "rzeta",
    "rzipf",
];

mod aggregations_list;
mod datetime;
mod glance;
mod list_builtins;
mod list_files;
mod names;
mod regex;
mod scalar_aggregations;
mod series_methods;
mod string_functions;
mod tidy;
mod type_conversions;

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
            "sum" | "mean" | "sd" | "std" | "min" | "max" | "total" => {
                self.aggregations_list(func, args, opts, _opt_map)
            }
            "date" | "datetime" => self.datetime(func, args, opts, _opt_map),
            "glance" => self.glance(func, args, opts, _opt_map),
            "len" | "keys" | "values" | "has_key" | "dict_merge" | "dmerge" | "dict_set"
            | "dset" | "dict_remove" | "dremove" | "dataframe" => {
                self.list_builtins(func, args, opts, _opt_map)
            }
            "list_files" => self.list_files(func, args, opts, _opt_map),
            "names" => self.names(func, args, opts, _opt_map),
            "regexm" | "regexr" | "regexra" | "regexs" => self.regex(func, args, opts, _opt_map),
            "median" | "variance" => self.scalar_aggregations(func, args, opts, _opt_map),
            "first" | "last" | "shift" | "quantile" | "cov" | "corr_pair" | "push" | "pop"
            | "insert" | "remove" | "clear" | "reverse" | "index" | "indexof" | "slice"
            | "join" | "map" | "unique" | "flatten" | "chain" | "range" => {
                self.series_methods(func, args, opts, _opt_map)
            }
            "upper" | "lower" | "trim" | "write" | "file_exists" | "ensure_dir" | "contains"
            | "starts_with" | "ends_with" | "substr" | "split" | "str_replace" => {
                self.string_functions(func, args, opts, _opt_map)
            }
            "tidy" => self.tidy(func, args, opts, _opt_map),
            "int" | "float" | "str" | "string" | "bool" | "is_nil" | "is_int" | "is_float"
            | "is_bool" | "is_str" | "is_string" | "is_list" | "is_dict" | "is_df"
            | "is_dataframe" | "is_fn" | "is_function" | "type" | "typeof" => {
                self.type_conversions(func, args, opts, _opt_map)
            }
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
