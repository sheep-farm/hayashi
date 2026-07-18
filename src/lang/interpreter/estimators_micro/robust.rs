use super::super::helpers::*;
use super::super::*;

impl Interpreter {
    pub(super) fn rlm(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let norm = match opt_map.get("norm") {
            None => greeners::RobustNorm::Huber(1.345),
            Some(Value::Str(s)) => match s.as_str() {
                "huber" => greeners::RobustNorm::Huber(1.345),
                "tukey" | "bisquare" => greeners::RobustNorm::Tukey(4.685),
                "andrews" | "wave" => greeners::RobustNorm::AndrewWave(std::f64::consts::PI),
                "hampel" => greeners::RobustNorm::Hampel(2.0, 4.0, 8.0),
                "ols" | "leastsq" => greeners::RobustNorm::LeastSquares,
                other => {
                    return Err(HayashiError::Runtime(format!(
                        "norm='{other}' unknown — use: huber, tukey, andrews, hampel, ols"
                    )))
                }
            },
            _ => return Err(HayashiError::Type("norm= must be string".into())),
        };
        let cov = resolve_cov_full(opt_map, &df)?;
        let result = greeners::RLM::fit_with_names(&y_vec, &x_mat, &norm, cov, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::RlmResult(Rc::new(result)))
    }

    pub(super) fn gee(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "gee() requires id=group_column option".into(),
                ))
            }
            _ => return Err(HayashiError::Type("id= must be string".into())),
        };
        let family_str = match opt_map.get("family") {
            None => "gaussian",
            Some(Value::Str(s)) => match s.as_str() {
                "gaussian" | "normal" => "gaussian",
                "binomial" | "logit" => "binomial",
                "poisson" => "poisson",
                other => {
                    return Err(HayashiError::Runtime(format!(
                        "family='{other}' unknown — use: gaussian, binomial, poisson"
                    )))
                }
            },
            _ => return Err(HayashiError::Type("family= must be string".into())),
        };
        let corr_str = match opt_map.get("corr") {
            None => "independence",
            Some(Value::Str(s)) => s.as_str(),
            _ => "independence",
        };
        let corr = match corr_str {
            "independence" | "ind" => greeners::CorrStructure::Independence,
            "exchangeable" | "exch" => greeners::CorrStructure::Exchangeable,
            "ar1" | "ar(1)" => greeners::CorrStructure::AR1,
            "unstructured" | "uns" => greeners::CorrStructure::Unstructured,
            other => {
                return Err(HayashiError::Runtime(format!(
                    "corr='{other}' unknown — use: independence, exchangeable, ar1, unstructured"
                )))
            }
        };
        let (family, link) = match family_str {
            "binomial" => (greeners::Family::Binomial, greeners::Link::Logit),
            "poisson" => (greeners::Family::Poisson, greeners::Link::Log),
            _ => (greeners::Family::Gaussian, greeners::Link::Identity),
        };
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        // convert id column to group indices (usize)
        let id_vals = get_col_f64(&df, &id_col)?;
        let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut next_id = 0usize;
        let groups: ndarray::Array1<usize> = id_vals
            .iter()
            .map(|&v| {
                let key = v as i64;
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();
        let result = greeners::GEE::fit_with_names(
            &y_vec,
            &x_mat,
            &groups,
            &family,
            &link,
            &corr,
            Some(var_names),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::GeeResult(Rc::new(result)))
    }

    pub(super) fn mixed(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;

        // id= required: group column
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            None => {
                return Err(HayashiError::Runtime(
                    "mixed() requires id=\"group_column\"".into(),
                ))
            }
            _ => return Err(HayashiError::Type("id= must be string".into())),
        };

        // re= optional: list of variables with random slope effect
        // Se omitido, modelo de random intercept apenas (re = [1])
        let re_vars: Vec<String> = match opt_map.get("re") {
            Some(Value::List(lst)) => lst
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(HayashiError::Type("re= must be a list of strings".into())),
                })
                .collect::<Result<_>>()?,
            None => vec![],
            _ => return Err(HayashiError::Type("re= must be a list of strings".into())),
        };

        let (y_vec, x_fixed) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        // Convert id to group indices
        let id_vals = get_col_f64(&df, &id_col)?;
        let mut id_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut next_id = 0usize;
        let groups: ndarray::Array1<usize> = id_vals
            .iter()
            .map(|&v| {
                let key = v as i64;
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();

        // Build x_random: intercept + specified slopes
        let n = df.n_rows();
        let q = re_vars.len() + 1; // +1 para random intercept
        let mut x_random = ndarray::Array2::<f64>::ones((n, q));
        for (j, name) in re_vars.iter().enumerate() {
            x_random.column_mut(j + 1).assign(&get_col_f64(&df, name)?);
        }

        let result = greeners::MixedLM::fit_with_names(
            &y_vec,
            &x_fixed,
            &groups,
            &x_random,
            Some(var_names),
        )
        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::MixedResult(Rc::new(result)))
    }

    pub(super) fn glsar(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let ar_order = match opt_map.get("ar") {
            Some(Value::Int(n)) => *n as usize,
            None => 1,
            _ => return Err(HayashiError::Type("ar= must be integer".into())),
        };
        let max_iter = match opt_map.get("iter") {
            Some(Value::Int(n)) => *n as usize,
            None => 50,
            _ => return Err(HayashiError::Type("iter= must be integer".into())),
        };
        let result =
            greeners::GLSAR::fit_with_names(&y_vec, &x_mat, ar_order, max_iter, Some(var_names))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::GlsarResult(Rc::new(result)))
    }

    pub(super) fn betareg(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let link = match opt_map.get("link") {
            None => greeners::BetaLink::Logit,
            Some(Value::Str(s)) => match s.as_str() {
                "logit" => greeners::BetaLink::Logit,
                "probit" => greeners::BetaLink::Probit,
                "cloglog" => greeners::BetaLink::CLogLog,
                other => {
                    return Err(HayashiError::Runtime(format!(
                        "betareg: link='{other}' unknown — use: logit, probit, cloglog"
                    )))
                }
            },
            _ => greeners::BetaLink::Logit,
        };
        let result = greeners::BetaModel::fit_with_names(&y_vec, &x_mat, &link, Some(var_names))
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        Ok(Value::BetaResult(Rc::new(result)))
    }

    pub(super) fn glm(
        &mut self,
        _func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        let (formula_ast, df) = self.extract_binary_args_filtered(args, opts)?;
        let (df, g_formula, _display) = self.prepare_formula(&formula_ast, &df)?;
        let (y_vec, x_mat) = df
            .to_design_matrix(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let var_names = df
            .formula_var_names(&g_formula)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let cov = resolve_cov_full(opt_map, &df)?;

        let alpha_val = match opt_map.get("alpha") {
            Some(Value::Float(v)) => *v,
            Some(Value::Int(v)) => *v as f64,
            _ => 1.0,
        };
        let power_val = match opt_map.get("power") {
            Some(Value::Float(v)) => *v,
            Some(Value::Int(v)) => *v as f64,
            _ => 1.5,
        };

        let family = match opt_map.get("family") {
        None | Some(Value::Str(_)) if opt_map.get("family").is_none() => {
            greeners::Family::Gaussian
        }
        Some(Value::Str(s)) => match s.as_str() {
            "gaussian" | "normal" => greeners::Family::Gaussian,
            "binomial" | "logistic" => greeners::Family::Binomial,
            "poisson"  => greeners::Family::Poisson,
            "gamma"    => greeners::Family::Gamma,
            "inverse_gaussian" | "inversegaussian" => greeners::Family::InverseGaussian,
            "negbin" | "negative_binomial" => greeners::Family::NegativeBinomial(alpha_val),
            "tweedie" => greeners::Family::Tweedie(power_val),
            other => return Err(HayashiError::Runtime(
                format!("glm: family='{other}' unknown — use: gaussian, binomial, poisson, gamma, inverse_gaussian, negbin, tweedie")
            )),
        },
        _ => greeners::Family::Gaussian,
    };

        let result = match opt_map.get("link") {
            None => greeners::GLM::fit_with_names(&y_vec, &x_mat, family, cov, Some(var_names))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?,
            Some(Value::Str(s)) => {
                let link = match s.as_str() {
                "identity"  => greeners::Link::Identity,
                "log"       => greeners::Link::Log,
                "logit"     => greeners::Link::Logit,
                "probit"    => greeners::Link::Probit,
                "inverse"   => greeners::Link::InversePower,
                "cloglog"   => greeners::Link::CLogLog,
                other => return Err(HayashiError::Runtime(
                    format!("glm: link='{other}' unknown — use: identity, log, logit, probit, inverse, cloglog")
                )),
            };
                // fit_with_link does not accept var_names; set after
                let mut r = greeners::GLM::fit_with_link(&y_vec, &x_mat, family, link, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                r.variable_names = Some(var_names);
                r
            }
            _ => greeners::GLM::fit_with_names(&y_vec, &x_mat, family, cov, Some(var_names))
                .map_err(|e| HayashiError::Runtime(e.to_string()))?,
        };
        Ok(Value::GlmResult(Rc::new(result)))
    }
}
