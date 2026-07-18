use super::*;
mod causal;
mod cross_section;
mod diagnostics;
mod discrete;
mod finance;
mod ml;
mod nls;
mod panel;
mod production;
mod robust;
mod spatial;
mod survival;
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
            "fmb" | "fama_macbeth" | "xtfmb" => self.fmb(func, args, opts, opt_map),
            "portsort" | "portfolio_sort" | "psort" => self.portsort(func, args, opts, opt_map),
            "doublesort" | "double_sort" | "bivariate_sort" => {
                self.doublesort(func, args, opts, opt_map)
            }
            "ols" => self.ols(func, args, opts, opt_map),
            "iv" => self.iv(func, args, opts, opt_map),
            "weak_iv" => self.weak_iv(func, args, opts, opt_map),
            "estat_overid" | "sargan" | "overid" | "sargan_test" => {
                self.estat_overid(func, args, opts, opt_map)
            }
            "estat_endog" | "endog_test" | "dwh" => self.estat_endog(func, args, opts, opt_map),
            "estat_classification" | "classification" => {
                self.estat_classification(func, args, opts, opt_map)
            }
            "lroc" | "roc" | "estat_roc" => self.lroc(func, args, opts, opt_map),
            "estat_gof" | "hosmer_lemeshow" | "hltest" => self.estat_gof(func, args, opts, opt_map),
            "linktest" => self.linktest(func, args, opts, opt_map),
            "logit" => self.logit(func, args, opts, opt_map),
            "probit" => self.probit(func, args, opts, opt_map),
            "heckman" | "heckit" => self.heckman(func, args, opts, opt_map),
            "tobit" => self.tobit(func, args, opts, opt_map),
            "rd" => self.rd(func, args, opts, opt_map),
            "fuzzy_rd" => self.fuzzy_rd(func, args, opts, opt_map),
            "psm" => self.psm(func, args, opts, opt_map),
            "synth" => self.synth(func, args, opts, opt_map),
            "poisson" => self.poisson(func, args, opts, opt_map),
            "nbreg" | "negbin" => self.nbreg(func, args, opts, opt_map),
            "ologit" => self.ologit(func, args, opts, opt_map),
            "oprobit" => self.oprobit(func, args, opts, opt_map),
            "mlogit" => self.mlogit(func, args, opts, opt_map),
            "did" => self.did(func, args, opts, opt_map),
            "eventstudy" | "event_study" | "es" => self.eventstudy(func, args, opts, opt_map),
            "nls_exp" | "nls_power" | "nls_logistic" | "nls_cobb_douglas" | "nls_ces" => {
                self.nls_exp(func, args, opts, opt_map)
            }
            "double_ml" | "dml" => self.double_ml(func, args, opts, opt_map),
            "sfa_production" | "sfa_cost" | "frontier" => {
                self.sfa_production(func, args, opts, opt_map)
            }
            "panel_tobit" => self.panel_tobit(func, args, opts, opt_map),
            "panel_heckman" => self.panel_heckman(func, args, opts, opt_map),
            "spatial_panel_sar" | "spatial_panel_sem" => {
                self.spatial_panel_sar(func, args, opts, opt_map)
            }
            "bayes_sfa_production" | "bayes_sfa_cost" | "bayes_frontier" => {
                self.bayes_sfa_production(func, args, opts, opt_map)
            }
            "midas" => self.midas(func, args, opts, opt_map),
            "tvp" => self.tvp(func, args, opts, opt_map),
            "setar" => self.setar(func, args, opts, opt_map),
            "panel_qreg" | "panel_quantile" => self.panel_qreg(func, args, opts, opt_map),
            "msvar" | "ms_var" => self.msvar(func, args, opts, opt_map),
            "favar" => self.favar(func, args, opts, opt_map),
            "spatial_durbin" | "sdm" => self.spatial_durbin(func, args, opts, opt_map),
            "johansen_break" => self.johansen_break(func, args, opts, opt_map),
            "tvp_var" => self.tvp_var(func, args, opts, opt_map),
            "spatial_durbin_error" | "sdem" => self.spatial_durbin_error(func, args, opts, opt_map),
            "fmols" => self.fmols(func, args, opts, opt_map),
            "qvar" | "quantile_var" => self.qvar(func, args, opts, opt_map),
            "pstr" => self.pstr(func, args, opts, opt_map),
            "modwt" => self.modwt(func, args, opts, opt_map),
            "copula" => self.copula(func, args, opts, opt_map),
            "nardl" => self.nardl(func, args, opts, opt_map),
            "pvar" | "panel_var" => self.pvar(func, args, opts, opt_map),
            "fcoef" | "functional_coef" => self.fcoef(func, args, opts, opt_map),
            "dcc_garch" | "dcc" => self.dcc_garch(func, args, opts, opt_map),
            "tvar" | "threshold_var" => self.tvar(func, args, opts, opt_map),
            "bvar" | "bayesian_var" => self.bvar(func, args, opts, opt_map),
            "mfvar" | "mixed_freq_var" => self.mfvar(func, args, opts, opt_map),
            "tvcopula" | "tv_copula" => self.tvcopula(func, args, opts, opt_map),
            "sv" | "stochastic_vol" => self.sv(func, args, opts, opt_map),
            "fapanel" | "fa_panel" => self.fapanel(func, args, opts, opt_map),
            "hawkes" => self.hawkes(func, args, opts, opt_map),
            "rf" | "random_forest" => self.rf(func, args, opts, opt_map),
            "gbm" | "gradient_boosting" => self.gbm(func, args, opts, opt_map),
            "mlp" | "neural_net" => self.mlp(func, args, opts, opt_map),
            "synthdid" | "synthetic_did" => self.synthdid(func, args, opts, opt_map),
            "cuped" => self.cuped(func, args, opts, opt_map),
            "qrf" | "quantile_forest" => self.qrf(func, args, opts, opt_map),
            "xgboost" | "xgb" => self.xgboost(func, args, opts, opt_map),
            "dml_crossfit" | "dml_cf" => self.dml_crossfit(func, args, opts, opt_map),
            "bsc" | "bayesian_sc" => self.bsc(func, args, opts, opt_map),
            "lstm" => self.lstm(func, args, opts, opt_map),
            "causalforest" | "causal_forest" => self.causalforest(func, args, opts, opt_map),
            "grf" | "generalized_rf" => self.grf(func, args, opts, opt_map),
            "conformal" | "conformal_pred" => self.conformal(func, args, opts, opt_map),
            "transformer" | "transformer_ts" => self.transformer(func, args, opts, opt_map),
            "dr_learner" | "drlearner" => self.dr_learner(func, args, opts, opt_map),
            "bart" | "bayesian_trees" => self.bart(func, args, opts, opt_map),
            "gp" | "gaussian_process" => self.gp(func, args, opts, opt_map),
            "tmle" => self.tmle(func, args, opts, opt_map),
            "orf" | "orthogonal_forest" => self.orf(func, args, opts, opt_map),
            "spectral" | "spectral_clustering" => self.spectral(func, args, opts, opt_map),
            "isotonic" | "isotonic_reg" => self.isotonic(func, args, opts, opt_map),
            "causal_impact" | "causalimpact" => self.causal_impact(func, args, opts, opt_map),
            "mice_chained" | "mice_eq" => self.mice_chained(func, args, opts, opt_map),
            "kmeans" | "k_means" => self.kmeans(func, args, opts, opt_map),
            "bayes_lm" | "bayesian_lm" => self.bayes_lm(func, args, opts, opt_map),
            "dbscan" | "dbscan_clust" => self.dbscan(func, args, opts, opt_map),
            "gmm_clust" | "gmm_clustering" => self.gmm_clust(func, args, opts, opt_map),
            "reg_path" | "regpath" => self.reg_path(func, args, opts, opt_map),
            "qrf_inf" | "qrf_inference" => self.qrf_inf(func, args, opts, opt_map),
            "hclust" | "hierarchical" => self.hclust(func, args, opts, opt_map),
            "tsne" | "t_sne" => self.tsne(func, args, opts, opt_map),
            "umap" => self.umap(func, args, opts, opt_map),
            "biplot" | "pca_biplot" => self.biplot(func, args, opts, opt_map),
            "spatial_sar" | "spatial_sem" => self.spatial_sar(func, args, opts, opt_map),
            "qreg" => self.qreg(func, args, opts, opt_map),
            "km" => self.km(func, args, opts, opt_map),
            "cox" => self.cox(func, args, opts, opt_map),
            "rlm" => self.rlm(func, args, opts, opt_map),
            "gee" => self.gee(func, args, opts, opt_map),
            "xtlogit" | "xtprobit" | "xtpoisson" | "xtgee" => {
                self.xtlogit(func, args, opts, opt_map)
            }
            "wls" => self.wls(func, args, opts, opt_map),
            "zip" | "zinb" => self.zip(func, args, opts, opt_map),
            "mixed" | "mixedlm" => self.mixed(func, args, opts, opt_map),
            "testparm" => self.testparm(func, args, opts, opt_map),
            "glsar" | "prais" => self.glsar(func, args, opts, opt_map),
            "anova" => self.anova(func, args, opts, opt_map),
            "betareg" | "beta" => self.betareg(func, args, opts, opt_map),
            "glm" => self.glm(func, args, opts, opt_map),
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
