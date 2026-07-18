use super::super::*;
impl Interpreter {
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
            _ => unreachable!(),
        }
    }
}
