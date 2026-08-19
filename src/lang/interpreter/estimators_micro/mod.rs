use super::*;
#[cfg(feature = "greeners-causal")]
mod causal;
mod cross_section;
#[cfg(feature = "greeners-diagnostics")]
mod diagnostics;
#[cfg(feature = "greeners-glm")]
mod discrete;
mod finance;
#[cfg(feature = "greeners-ml")]
mod ml;
#[cfg(feature = "greeners-ols")]
mod nls;
#[cfg(feature = "greeners-panel")]
mod panel;
mod production;
mod robust;
#[cfg(feature = "greeners-spatial")]
mod spatial;
#[cfg(feature = "greeners-survival")]
mod survival;
#[cfg(feature = "greeners-timeseries")]
mod timeseries;

impl Interpreter {
    pub(super) fn eval_call_estimators_micro(
        &mut self,
        func: &str,
        args: &[Expr],
        opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            "reg" | "regress" => self.eval_call("ols", args, opts),
            #[cfg(feature = "greeners-diagnostics")]
            "fmb" | "fama_macbeth" | "xtfmb" => self.fmb(func, args, opts, opt_map),
            "portsort" | "portfolio_sort" | "psort" => self.portsort(func, args, opts, opt_map),
            "doublesort" | "double_sort" | "bivariate_sort" => {
                self.doublesort(func, args, opts, opt_map)
            }
            #[cfg(feature = "greeners-ols")]
            "ols" => self.ols(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "iv" => self.iv(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "weak_iv" => self.weak_iv(func, args, opts, opt_map),
            "estat_overid" | "sargan" | "overid" | "sargan_test" => {
                self.estat_overid(func, args, opts, opt_map)
            }
            #[cfg(feature = "greeners-ols")]
            "estat_endog" | "endog_test" | "dwh" => self.estat_endog(func, args, opts, opt_map),
            "estat_classification" | "classification" => {
                self.estat_classification(func, args, opts, opt_map)
            }
            #[cfg(feature = "greeners-glm")]
            "lroc" | "roc" | "estat_roc" => self.lroc(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "estat_gof" | "hosmer_lemeshow" | "hltest" => self.estat_gof(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "linktest" => self.linktest(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "logit" => self.logit(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "probit" => self.probit(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "heckman" | "heckit" => self.heckman(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "tobit" => self.tobit(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "rd" => self.rd(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "fuzzy_rd" => self.fuzzy_rd(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "psm" => self.psm(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "synth" => self.synth(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "poisson" => self.poisson(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "nbreg" | "negbin" => self.nbreg(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "ologit" => self.ologit(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "oprobit" => self.oprobit(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "mlogit" => self.mlogit(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "did" => self.did(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "lpdid" => self.lpdid(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "eventstudy" | "event_study" | "es" => self.eventstudy(func, args, opts, opt_map),
            "nls_exp" | "nls_power" | "nls_logistic" | "nls_cobb_douglas" | "nls_ces" => {
                self.nls_exp(func, args, opts, opt_map)
            }
            #[cfg(feature = "greeners-causal")]
            "double_ml" | "dml" => self.double_ml(func, args, opts, opt_map),
            "sfa_production" | "sfa_cost" | "frontier" => {
                self.sfa_production(func, args, opts, opt_map)
            }
            #[cfg(feature = "greeners-panel")]
            "panel_tobit" => self.panel_tobit(func, args, opts, opt_map),
            #[cfg(feature = "greeners-panel")]
            "panel_heckman" => self.panel_heckman(func, args, opts, opt_map),
            "spatial_panel_sar" | "spatial_panel_sem" => {
                self.spatial_panel_sar(func, args, opts, opt_map)
            }
            "bayes_sfa_production" | "bayes_sfa_cost" | "bayes_frontier" => {
                self.bayes_sfa_production(func, args, opts, opt_map)
            }
            #[cfg(feature = "greeners-timeseries")]
            "midas" => self.midas(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "tvp" => self.tvp(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "setar" => self.setar(func, args, opts, opt_map),
            #[cfg(feature = "greeners-panel")]
            "panel_qreg" | "panel_quantile" => self.panel_qreg(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "msvar" | "ms_var" => self.msvar(func, args, opts, opt_map),
            #[cfg(feature = "greeners-bayesian")]
            "favar" => self.favar(func, args, opts, opt_map),
            #[cfg(feature = "greeners-spatial")]
            "spatial_durbin" | "sdm" => self.spatial_durbin(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "johansen_break" => self.johansen_break(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "tvp_var" => self.tvp_var(func, args, opts, opt_map),
            #[cfg(feature = "greeners-spatial")]
            "spatial_durbin_error" | "sdem" => self.spatial_durbin_error(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "fmols" => self.fmols(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "qvar" | "quantile_var" => self.qvar(func, args, opts, opt_map),
            #[cfg(feature = "greeners-panel")]
            "pstr" => self.pstr(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "modwt" => self.modwt(func, args, opts, opt_map),
            "copula" => self.copula(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "nardl" => self.nardl(func, args, opts, opt_map),
            #[cfg(feature = "greeners-panel")]
            "pvar" | "panel_var" => self.pvar(func, args, opts, opt_map),
            "fcoef" | "functional_coef" => self.fcoef(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "dcc_garch" | "dcc" => self.dcc_garch(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "tvar" | "threshold_var" => self.tvar(func, args, opts, opt_map),
            #[cfg(feature = "greeners-bayesian")]
            "bvar" | "bayesian_var" => self.bvar(func, args, opts, opt_map),
            #[cfg(feature = "greeners-bayesian")]
            "mfvar" | "mixed_freq_var" => self.mfvar(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "tvcopula" | "tv_copula" => self.tvcopula(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "sv" | "stochastic_vol" => self.sv(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "fapanel" | "fa_panel" => self.fapanel(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "hawkes" => self.hawkes(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "rf" | "random_forest" => self.rf(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "gbm" | "gradient_boosting" => self.gbm(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "mlp" | "neural_net" => self.mlp(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "synthdid" | "synthetic_did" => self.synthdid(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "cuped" => self.cuped(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "qrf" | "quantile_forest" => self.qrf(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "xgboost" | "xgb" => self.xgboost(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "dml_crossfit" | "dml_cf" => self.dml_crossfit(func, args, opts, opt_map),
            #[cfg(feature = "greeners-bayesian")]
            "bsc" | "bayesian_sc" => self.bsc(func, args, opts, opt_map),
            #[cfg(feature = "greeners-timeseries")]
            "lstm" => self.lstm(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "causalforest" | "causal_forest" => self.causalforest(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "grf" | "generalized_rf" => self.grf(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "conformal" | "conformal_pred" => self.conformal(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "transformer" | "transformer_ts" => self.transformer(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "dr_learner" | "drlearner" => self.dr_learner(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "bart" | "bayesian_trees" => self.bart(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "gp" | "gaussian_process" => self.gp(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "tmle" => self.tmle(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "orf" | "orthogonal_forest" => self.orf(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "spectral" | "spectral_clustering" => self.spectral(func, args, opts, opt_map),
            "isotonic" | "isotonic_reg" => self.isotonic(func, args, opts, opt_map),
            #[cfg(feature = "greeners-causal")]
            "causal_impact" | "causalimpact" => self.causal_impact(func, args, opts, opt_map),
            #[cfg(feature = "greeners-imputation")]
            "mice_chained" | "mice_eq" => self.mice_chained(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "kmeans" | "k_means" => self.kmeans(func, args, opts, opt_map),
            #[cfg(feature = "greeners-bayesian")]
            "bayes_lm" | "bayesian_lm" => self.bayes_lm(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "dbscan" | "dbscan_clust" => self.dbscan(func, args, opts, opt_map),
            "gmm_clust" | "gmm_clustering" => self.gmm_clust(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "reg_path" | "regpath" => self.reg_path(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "qrf_inf" | "qrf_inference" => self.qrf_inf(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "hclust" | "hierarchical" => self.hclust(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "tsne" | "t_sne" => self.tsne(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ml")]
            "umap" => self.umap(func, args, opts, opt_map),
            "biplot" | "pca_biplot" => self.biplot(func, args, opts, opt_map),
            #[cfg(feature = "greeners-spatial")]
            "spatial_sar" | "spatial_sem" => self.spatial_sar(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "qreg" => self.qreg(func, args, opts, opt_map),
            #[cfg(feature = "greeners-survival")]
            "km" => self.km(func, args, opts, opt_map),
            #[cfg(feature = "greeners-survival")]
            "cox" => self.cox(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "rlm" => self.rlm(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "gee" => self.gee(func, args, opts, opt_map),
            "xtlogit" | "xtprobit" | "xtpoisson" | "xtgee" => {
                self.xtlogit(func, args, opts, opt_map)
            }
            #[cfg(feature = "greeners-ols")]
            "wls" => self.wls(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "zip" | "zinb" => self.zip(func, args, opts, opt_map),
            #[cfg(feature = "greeners-bayesian")]
            "mixed" | "mixedlm" => self.mixed(func, args, opts, opt_map),
            #[cfg(feature = "greeners-diagnostics")]
            "testparm" => self.testparm(func, args, opts, opt_map),
            #[cfg(feature = "greeners-ols")]
            "glsar" | "prais" => self.glsar(func, args, opts, opt_map),
            "anova" => self.anova(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "betareg" | "beta" => self.betareg(func, args, opts, opt_map),
            #[cfg(feature = "greeners-glm")]
            "glm" => self.glm(func, args, opts, opt_map),
            #[cfg(feature = "greeners-diagnostics")]
            "influence" => self.influence(func, args, opts, opt_map),
            "lowess" => self.lowess(func, args, opts, opt_map),
            "kde" => self.kde(func, args, opts, opt_map),
            "pca" | "princomp" => self.pca(func, args, opts, opt_map),
            "factor" => self.factor(func, args, opts, opt_map),
            "manova" => self.manova(func, args, opts, opt_map),
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
