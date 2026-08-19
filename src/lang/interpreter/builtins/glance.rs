use super::super::*;
impl Interpreter {
    #[cfg(all(feature = "greeners-bayesian", feature = "greeners-causal", feature = "greeners-glm", feature = "greeners-imputation", feature = "greeners-ols", feature = "greeners-panel", feature = "greeners-survival", feature = "greeners-timeseries"))]
    pub(super) fn glance(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "glance" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime(
                        "glance(model) requires 1 argument".into(),
                    ));
                }
                let val = self.eval_expr(&args[0])?;

                // Diagnostic and generic model results already expose their
                // structured fields as a dict; glance() just returns it.
                if let Value::DiagResult(r) = &val {
                    return Ok(Value::Dict(Arc::new(r.fields.clone())));
                }
                if let Value::ModelResult { fields, .. } = &val {
                    return Ok(Value::Dict(Arc::new(fields.as_ref().clone())));
                }
                if let Value::Dict(d) = val {
                    return Ok(Value::Dict(d));
                }

                let mut map = std::collections::HashMap::<String, Value>::new();

                if let Some(mv) = val.to_model_view() {
                    map = mv.to_glance_map();
                } else {
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
                        #[cfg(feature = "greeners-ols")]
                        Value::IvResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("r2".into(), scalar(r.r_squared));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                            map.insert("sigma".into(), scalar(r.sigma));
                        }
                        #[cfg(feature = "greeners-glm")]
                        Value::BinaryResult(m) => {
                            let r = &m.result;
                            let scalar = |v: f64| self.gf(v);
                            map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                            map.insert("log_lik".into(), scalar(r.log_likelihood));
                            map.insert("n".into(), Value::List(Arc::new(vec![Value::Int(0)])));
                            // n not stored
                        }
                        #[cfg(feature = "greeners-panel")]
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
                        #[cfg(feature = "greeners-panel")]
                        Value::ReResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("r2".into(), scalar(r.r_squared_overall));
                            map.insert("sigma_u".into(), scalar(r.sigma_u));
                            map.insert("sigma_e".into(), scalar(r.sigma_e));
                            map.insert("theta".into(), scalar(r.theta));
                        }
                        #[cfg(feature = "greeners-ols")]
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
                        #[cfg(feature = "greeners-glm")]
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
                        #[cfg(feature = "greeners-glm")]
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
                        #[cfg(feature = "greeners-glm")]
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
                        #[cfg(feature = "greeners-ols")]
                        Value::QuantileResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("tau".into(), scalar(r.tau));
                            map.insert("pseudo_r2".into(), scalar(r.r_squared));
                        }
                        #[cfg(feature = "greeners-ols")]
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
                        #[cfg(feature = "greeners-ols")]
                        Value::HeckmanResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("rho".into(), scalar(r.rho));
                            map.insert("delta".into(), scalar(r.delta));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-glm")]
                        Value::OrderedResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("log_lik".into(), scalar(r.log_likelihood));
                            map.insert("aic".into(), scalar(r.aic));
                            map.insert("bic".into(), scalar(r.bic));
                            map.insert("pseudo_r2".into(), scalar(r.pseudo_r2));
                        }
                        #[cfg(feature = "greeners-ols")]
                        Value::PenalizedResult(m) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("r2".into(), scalar(m.r_squared));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(m.n_obs as i64)])),
                            );
                            map.insert("alpha".into(), scalar(m.alpha));
                        }
                        #[cfg(feature = "greeners-timeseries")]
                        Value::ArimaResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("aic".into(), scalar(r.aic));
                            map.insert("bic".into(), scalar(r.bic));
                            map.insert("log_lik".into(), scalar(r.log_likelihood));
                            map.insert("sigma2".into(), scalar(r.sigma2));
                        }
                        #[cfg(feature = "greeners-timeseries")]
                        Value::GarchResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("log_lik".into(), scalar(r.log_likelihood));
                            map.insert("aic".into(), scalar(r.aic));
                            map.insert("bic".into(), scalar(r.bic));
                        }
                        #[cfg(feature = "greeners-timeseries")]
                        Value::VarResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("aic".into(), scalar(r.aic));
                            map.insert("bic".into(), scalar(r.bic));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-panel")]
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
                        #[cfg(feature = "greeners-panel")]
                        Value::FE2SLSResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("r2".into(), scalar(r.r_squared));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                            map.insert("sigma".into(), scalar(r.sigma));
                        }
                        #[cfg(feature = "greeners-panel")]
                        Value::PcseResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("r2".into(), scalar(r.r_squared));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                            map.insert("sigma".into(), scalar(r.sigma));
                        }
                        #[cfg(feature = "greeners-panel")]
                        Value::PanelGlsResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("r2".into(), scalar(r.r_squared));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                            map.insert("sigma".into(), scalar(r.sigma));
                        }
                        #[cfg(feature = "greeners-ols")]
                        Value::GlsarResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("r2".into(), scalar(r.r_squared));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-ols")]
                        Value::RecursiveLSResult(r) => {
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-survival")]
                        Value::CoxResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("log_lik".into(), scalar(r.log_likelihood));
                            map.insert("concordance".into(), scalar(r.concordance));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-glm")]
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
                        #[cfg(feature = "greeners-glm")]
                        Value::GamResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("gcv".into(), scalar(r.gcv_score));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-bayesian")]
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
                        #[cfg(feature = "greeners-glm")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-causal")]
                        Value::DidResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("att".into(), scalar(r.att));
                            map.insert("r2".into(), scalar(r.r_squared));
                            map.insert(
                                "n".into(),
                                Value::List(Arc::new(vec![Value::Int(r.n_obs as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-panel")]
                        Value::ThresholdResult(r) => {
                            let scalar = |v: f64| self.gf(v);
                            map.insert("threshold".into(), scalar(r.threshold_gamma));
                            map.insert("r2".into(), scalar(r.r_squared));
                        }
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-glm")]
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
                        #[cfg(feature = "greeners-glm")]
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
                        #[cfg(feature = "greeners-ols")]
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
                        #[cfg(feature = "greeners-panel")]
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
                        #[cfg(feature = "greeners-ols")]
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
                        #[cfg(feature = "greeners-causal")]
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
                        #[cfg(feature = "greeners-causal")]
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
                        #[cfg(feature = "greeners-glm")]
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
                        #[cfg(feature = "greeners-ols")]
                        Value::SurResult(m) => {
                            let r = &m.result;
                            let scalar = |v: f64| self.gf(v);
                            map.insert("system_r2".into(), scalar(r.system_r2));
                            map.insert(
                                "n_equations".into(),
                                Value::List(Arc::new(vec![Value::Int(r.equations.len() as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-ols")]
                        Value::ThreeSLSResult(m) => {
                            let r = &m.result;
                            let scalar = |v: f64| self.gf(v);
                            map.insert("system_r2".into(), scalar(r.system_r2));
                            map.insert(
                                "n_equations".into(),
                                Value::List(Arc::new(vec![Value::Int(r.equations.len() as i64)])),
                            );
                        }
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-timeseries")]
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
                        #[cfg(feature = "greeners-imputation")]
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
                        #[cfg(feature = "greeners-survival")]
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
                        _ => {
                            return Err(HayashiError::Type("glance: unsupported model type".into()))
                        }
                    }
                }

                let df = self.dict_to_dataframe(&map)?;
                Ok(Value::DataFrame(Arc::new(df)))
            }
            _ => unreachable!(),
        }
    }
}