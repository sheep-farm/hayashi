use super::*;

pub fn ols_fit_dict(r: &greeners::OlsResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("adj_r2", Value::Float(r.adj_r_squared)),
        ("f_stat", Value::Float(r.f_statistic)),
        ("prob_f", Value::Float(r.prob_f)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("sigma", Value::Float(r.sigma)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("cov_type", Value::Str(format!("{:?}", r.cov_type))),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn iv_fit_dict(r: &greeners::iv::IvResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("cov_type", Value::Str(format!("{:?}", r.cov_type))),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn panel_fit_dict(r: &greeners::panel::PanelResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn re_fit_dict(r: &greeners::panel::RandomEffectsResult) -> Value {
    fit_dict(&[
        ("r2_overall", Value::Float(r.r_squared_overall)),
        ("sigma_u", Value::Float(r.sigma_u)),
        ("sigma_e", Value::Float(r.sigma_e)),
        ("theta", Value::Float(r.theta)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn binary_fit_dict(r: &greeners::discrete::BinaryModelResult) -> Value {
    fit_dict(&[
        ("model_name", Value::Str(r.model_name.clone())),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn quantile_fit_dict(r: &greeners::QuantileResult) -> Value {
    fit_dict(&[
        ("tau", Value::Float(r.tau)),
        ("r2", Value::Float(r.r_squared)),
        ("iterations", Value::Int(r.iterations as i64)),
    ])
}

pub fn tobit_fit_dict(r: &greeners::TobitResult) -> Value {
    fit_dict(&[
        ("sigma", Value::Float(r.sigma)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_censored", Value::Int(r.n_censored as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
    ])
}

pub fn poisson_fit_dict(r: &greeners::PoissonResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("deviance", Value::Float(r.deviance)),
        ("null_deviance", Value::Float(r.null_deviance)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("pearson_chi2", Value::Float(r.pearson_chi2)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn negbin_fit_dict(r: &greeners::NegBinResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("deviance", Value::Float(r.deviance)),
        ("null_deviance", Value::Float(r.null_deviance)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("pearson_chi2", Value::Float(r.pearson_chi2)),
        ("alpha", Value::Float(r.alpha)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn glm_fit_dict(r: &greeners::GlmResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("deviance", Value::Float(r.deviance)),
        ("null_deviance", Value::Float(r.null_deviance)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("pearson_chi2", Value::Float(r.pearson_chi2)),
        ("dispersion", Value::Float(r.dispersion)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("family", Value::Str(format!("{:?}", r.family))),
    ])
}

pub fn rlm_fit_dict(r: &greeners::RlmResult) -> Value {
    fit_dict(&[
        ("scale", Value::Float(r.scale)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

pub fn beta_fit_dict(r: &greeners::BetaResult) -> Value {
    fit_dict(&[
        ("precision_param", Value::Float(r.precision_param)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

pub fn gmm_fit_dict(r: &greeners::GmmResult) -> Value {
    fit_dict(&[
        ("j_stat", Value::Float(r.j_stat)),
        ("j_p_value", Value::Float(r.j_p_value)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("df_overid", Value::Int(r.df_overid as i64)),
    ])
}

pub fn ab_fit_dict(r: &greeners::ArellanoBondResult) -> Value {
    fit_dict(&[
        ("sargan_stat", Value::Float(r.sargan_stat)),
        ("sargan_pvalue", Value::Float(r.sargan_pvalue)),
        ("sargan_df", Value::Int(r.sargan_df as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("t_bar", Value::Float(r.t_bar)),
        ("n_instruments", Value::Int(r.n_instruments as i64)),
        ("max_lags", Value::Int(r.max_lags as i64)),
        ("step", Value::Int(r.step as i64)),
        ("m1_stat", Value::Float(r.m1_stat)),
        ("m1_pval", Value::Float(r.m1_pval)),
        ("m2_stat", Value::Float(r.m2_stat)),
        ("m2_pval", Value::Float(r.m2_pval)),
    ])
}

pub fn sysgmm_fit_dict(r: &greeners::SystemGmmResult) -> Value {
    fit_dict(&[
        ("sargan_stat", Value::Float(r.sargan_stat)),
        ("sargan_pvalue", Value::Float(r.sargan_pvalue)),
        ("sargan_df", Value::Int(r.sargan_df as i64)),
        ("n_obs_fd", Value::Int(r.n_obs_fd as i64)),
        ("n_obs_lev", Value::Int(r.n_obs_lev as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("n_instruments", Value::Int(r.n_instruments as i64)),
        ("max_lags", Value::Int(r.max_lags as i64)),
        ("step", Value::Int(r.step as i64)),
        ("m1_stat", Value::Float(r.m1_stat)),
        ("m1_pval", Value::Float(r.m1_pval)),
        ("m2_stat", Value::Float(r.m2_stat)),
        ("m2_pval", Value::Float(r.m2_pval)),
    ])
}

pub fn pcse_fit_dict(r: &greeners::PcseResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("t_periods", Value::Int(r.t_periods as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
    ])
}

pub fn panel_gls_fit_dict(r: &greeners::PanelGlsResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("t_periods", Value::Int(r.t_periods as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("panels", Value::Str(format!("{:?}", r.panels))),
    ])
}

pub fn fe2sls_fit_dict(r: &greeners::PanelIvResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_entities", Value::Int(r.n_entities as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn ordered_fit_dict(r: &greeners::OrderedResult) -> Value {
    fit_dict(&[
        ("model_name", Value::Str(r.model_name.clone())),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_categories", Value::Int(r.n_categories as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("converged", Value::Bool(r.converged)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ])
}

pub fn zero_inflated_children(r: &greeners::ZeroInflatedResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let count_names = r.count_var_names.clone().unwrap_or_default();
    let inflate_names = r.inflate_var_names.clone().unwrap_or_default();

    let count_coef = coef_dataframe(
        &count_names,
        &r.count_params,
        &r.count_std_errors,
        &r.count_z_values,
        &r.count_p_values,
        None,
        None,
    );
    vars.push(("count_coefficients".into(), count_coef));

    let inflate_coef = coef_dataframe(
        &inflate_names,
        &r.inflate_params,
        &r.inflate_std_errors,
        &r.inflate_z_values,
        &r.inflate_p_values,
        None,
        None,
    );
    vars.push(("inflate_coefficients".into(), inflate_coef));

    vars.push(("fit".into(), zero_inflated_fit_dict(r)));
    vars
}

pub fn zero_inflated_fit_dict(r: &greeners::ZeroInflatedResult) -> Value {
    let mut entries: Vec<(&str, Value)> = vec![
        ("model_name", Value::Str(r.model_name.clone())),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("converged", Value::Bool(r.converged)),
        ("inference", Value::Str(format!("{:?}", r.inference_type))),
    ];
    if let Some(alpha) = r.alpha {
        entries.push(("alpha", Value::Float(alpha)));
    }
    fit_dict(&entries)
}

pub fn mixed_children(r: &greeners::MixedResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let names = r.variable_names.clone().unwrap_or_default();
    let fixed = coef_dataframe(
        &names,
        &r.fixed_effects,
        &r.fixed_se,
        &r.z_values,
        &r.p_values,
        None,
        None,
    );
    vars.push(("fixed_effects".into(), fixed));

    let mut re_cols: IndexMap<String, greeners::Column> = IndexMap::new();
    for (group, vals) in r.random_effects.iter() {
        re_cols.insert(
            format!("group_{group}"),
            greeners::Column::Float(Array1::from(vals.iter().copied().collect::<Vec<_>>())),
        );
    }
    let re_df = Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(re_cols)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ));
    vars.push(("random_effects".into(), re_df));

    vars.push(("fit".into(), mixed_fit_dict(r)));
    vars
}

pub fn mixed_fit_dict(r: &greeners::MixedResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_groups", Value::Int(r.n_groups as i64)),
        ("var_resid", Value::Float(r.var_resid)),
        ("iterations", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

pub fn glsar_fit_dict(r: &greeners::GlsarResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
    ])
}

pub fn sur_children(m: &SurModel) -> Vec<(String, Value)> {
    let r = &m.result;
    let mut vars = Vec::new();
    vars.push(("system_r2".into(), Value::Float(r.system_r2)));
    vars.push((
        "sigma_cross".into(),
        array2_to_dataframe("sigma", &r.sigma_cross),
    ));
    for (i, eq) in r.equations.iter().enumerate() {
        let name = if eq.name.is_empty() {
            format!("equation_{i}")
        } else {
            eq.name.clone()
        };
        let eq_fit = fit_dict(&[("r2", Value::Float(eq.r_squared))]);
        let eq_val = coef_dataframe(
            &(0..eq.params.len())
                .map(|j| format!("x{j}"))
                .collect::<Vec<_>>(),
            &eq.params,
            &eq.std_errors,
            &eq.t_values,
            &eq.p_values,
            None,
            None,
        );
        let mut wrap = HashMap::new();
        wrap.insert("coefficients".into(), eq_val);
        wrap.insert("fit".into(), eq_fit);
        vars.push((name, Value::Dict(Arc::new(wrap))));
    }
    vars
}

pub fn three_sls_children(m: &ThreeSLSModel) -> Vec<(String, Value)> {
    let r = &m.result;
    let mut vars = Vec::new();
    for (i, eq) in r.equations.iter().enumerate() {
        let name = m
            .eq_var_names
            .get(i)
            .and_then(|v| v.first())
            .cloned()
            .unwrap_or_else(|| format!("equation_{i}"));
        let eq_val = coef_dataframe(
            &(0..eq.params.len())
                .map(|j| format!("x{j}"))
                .collect::<Vec<_>>(),
            &eq.params,
            &eq.std_errors,
            &eq.t_values,
            &eq.p_values,
            None,
            None,
        );
        let mut wrap = HashMap::new();
        wrap.insert("coefficients".into(), eq_val);
        wrap.insert(
            "fit".into(),
            fit_dict(&[("r2", Value::Float(eq.r_squared))]),
        );
        vars.push((name, Value::Dict(Arc::new(wrap))));
    }
    vars
}

pub fn mnlogit_children(r: &greeners::MNLogitResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let names = r.variable_names.clone().unwrap_or_default();
    let categories = &r.category_labels;
    for (j, cat) in categories.iter().enumerate().skip(1) {
        let cat_name = format!("category_{cat:.0}");
        let cat_params = r.params.column(j - 1).to_owned();
        let cat_se = r.std_errors.column(j - 1).to_owned();
        let cat_z = r.z_values.column(j - 1).to_owned();
        let cat_p = r.p_values.column(j - 1).to_owned();
        let coef = coef_dataframe(&names, &cat_params, &cat_se, &cat_z, &cat_p, None, None);
        vars.push((cat_name, coef));
    }
    vars.push(("fit".into(), mnlogit_fit_dict(r)));
    vars
}

pub fn mnlogit_fit_dict(r: &greeners::MNLogitResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("pseudo_r2", Value::Float(r.pseudo_r2)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_categories", Value::Int(r.n_categories as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

pub fn gee_children(r: &greeners::GeeResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let names = r.variable_names.clone().unwrap_or_default();
    let coef = coef_dataframe(
        &names,
        &r.params,
        &r.robust_se,
        &r.z_values,
        &r.p_values,
        None,
        None,
    );
    vars.push(("coefficients".into(), coef));
    let n = r.working_correlation.nrows();
    let wc_names: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    vars.push((
        "working_correlation".into(),
        array2_to_dataframe_named(&r.working_correlation, &wc_names),
    ));
    vars.push(("fit".into(), gee_fit_dict(r)));
    vars
}

pub fn gee_fit_dict(r: &greeners::GeeResult) -> Value {
    fit_dict(&[
        ("scale", Value::Float(r.scale)),
        ("qic", Value::Float(r.qic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_groups", Value::Int(r.n_groups as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

pub fn conditional_children(r: &greeners::ConditionalResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let names = r.variable_names.clone().unwrap_or_default();
    let coef = coef_dataframe(
        &names,
        &r.params,
        &r.std_errors,
        &r.z_values,
        &r.p_values,
        None,
        None,
    );
    vars.push(("coefficients".into(), coef));
    vars.push(("fit".into(), conditional_fit_dict(r)));
    vars
}

pub fn conditional_fit_dict(r: &greeners::ConditionalResult) -> Value {
    fit_dict(&[
        ("model_name", Value::Str(r.model_name.clone())),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_groups", Value::Int(r.n_groups as i64)),
        ("iterations", Value::Int(r.iterations as i64)),
        ("converged", Value::Bool(r.converged)),
        (
            "inference_type",
            Value::Str(format!("{:?}", r.inference_type)),
        ),
    ])
}
