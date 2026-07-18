use super::*;

pub fn arima_children(r: &greeners::ArimaResult) -> Vec<(String, Value)> {
    let mut params = vec![r.intercept];
    params.extend(r.ar_params.iter());
    params.extend(r.ma_params.iter());
    params.extend(r.seasonal_ar_params.iter());
    params.extend(r.seasonal_ma_params.iter());
    if let Some(exog) = &r.exog_params {
        params.extend(exog.iter());
    }
    let params = Array1::from_vec(params);
    let names = r.param_names.clone();
    regression_children(RegressionCtx {
        names,
        params: &params,
        std_errors: &r.std_errors,
        test_values: &r.t_values,
        p_values: &r.p_values,
        conf_lower: Some(&r.conf_lower),
        conf_upper: Some(&r.conf_upper),
        fit: arima_fit_dict(r),
        residuals: Some(&r.residuals),
        fitted_values: None,
        x: None,
    })
}

pub fn arima_fit_dict(r: &greeners::ArimaResult) -> Value {
    let mut entries = vec![
        ("sigma2", Value::Float(r.sigma2)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("df_model", Value::Int(r.df_model as i64)),
        ("df_resid", Value::Int(r.df_resid as i64)),
        ("p", Value::Int(r.order.p as i64)),
        ("d", Value::Int(r.order.d as i64)),
        ("q", Value::Int(r.order.q as i64)),
        ("method", Value::Str(r.estimation_method.clone())),
    ];
    if let Some(so) = &r.seasonal_order {
        entries.push(("seasonal_p", Value::Int(so.p as i64)));
        entries.push(("seasonal_d", Value::Int(so.d as i64)));
        entries.push(("seasonal_q", Value::Int(so.q as i64)));
        entries.push(("seasonal_period", Value::Int(so.s as i64)));
    }
    fit_dict(&entries)
}

pub fn garch_children(r: &greeners::GarchResult) -> Vec<(String, Value)> {
    let mut vars = regression_children(RegressionCtx {
        names: r.variable_names.clone(),
        params: &r.params,
        std_errors: &r.std_errors,
        test_values: &r.z_values,
        p_values: &r.p_values,
        conf_lower: Some(&r.conf_lower),
        conf_upper: Some(&r.conf_upper),
        fit: garch_fit_dict(r),
        residuals: Some(&r.residuals),
        fitted_values: None,
        x: None,
    });
    vars.push((
        "conditional_variance".into(),
        array1_to_series("conditional_variance", &r.conditional_variance),
    ));
    vars.push((
        "standardized_residuals".into(),
        array1_to_series("standardized_residuals", &r.standardized_residuals),
    ));
    vars
}

pub fn garch_fit_dict(r: &greeners::GarchResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("p", Value::Int(r.p as i64)),
        ("q", Value::Int(r.q as i64)),
        ("model_type", Value::Str(format!("{:?}", r.model_type))),
        ("dist", Value::Str(format!("{:?}", r.dist))),
    ])
}

pub fn ets_children(r: &greeners::ETSResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    vars.push(("level".into(), array1_to_series("level", &r.level)));
    if !r.trend.is_empty() {
        vars.push(("trend".into(), array1_to_series("trend", &r.trend)));
    }
    if !r.seasonal.is_empty() {
        vars.push(("seasonal".into(), array1_to_series("seasonal", &r.seasonal)));
    }
    if !r.fitted_values.is_empty() {
        vars.push((
            "fitted_values".into(),
            array1_to_series("fitted_values", &r.fitted_values),
        ));
    }
    if !r.residuals.is_empty() {
        vars.push((
            "residuals".into(),
            array1_to_series("residuals", &r.residuals),
        ));
    }
    vars.push(("fit".into(), ets_fit_dict(r)));
    vars
}

pub fn ets_fit_dict(r: &greeners::ETSResult) -> Value {
    let mut entries = vec![
        ("alpha", Value::Float(r.alpha)),
        ("sse", Value::Float(r.sse)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("last_level", Value::Float(r.last_level)),
        ("seasonal_periods", Value::Int(r.seasonal_periods as i64)),
        ("trend_type", Value::Str(r.trend_type.clone())),
        ("seasonal_type", Value::Str(r.seasonal_type.clone())),
        ("damped", Value::Bool(r.damped)),
    ];
    if let Some(beta) = r.beta {
        entries.push(("beta", Value::Float(beta)));
    }
    if let Some(gamma) = r.gamma {
        entries.push(("gamma", Value::Float(gamma)));
    }
    if let Some(phi) = r.phi {
        entries.push(("phi", Value::Float(phi)));
    }
    if !r.last_trend.is_nan() {
        entries.push(("last_trend", Value::Float(r.last_trend)));
    }
    fit_dict(&entries)
}

pub fn mstl_children(r: &greeners::MSTLResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    vars.push(("trend".into(), array1_to_series("trend", &r.trend)));
    for (i, s) in r.seasonal.iter().enumerate() {
        let period = r.periods.get(i).copied().unwrap_or(i);
        let name = format!("seasonal_{period}");
        vars.push((name.clone(), array1_to_series(&name, s)));
    }
    if !r.resid.is_empty() {
        vars.push(("residuals".into(), array1_to_series("residuals", &r.resid)));
    }
    vars.push((
        "observed".into(),
        array1_to_series("observed", &r.observed()),
    ));
    vars.push(("fit".into(), mstl_fit_dict(r)));
    vars
}

pub fn mstl_fit_dict(r: &greeners::MSTLResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        (
            "periods",
            Value::List(Arc::new(
                r.periods.iter().map(|&p| Value::Int(p as i64)).collect(),
            )),
        ),
    ])
}

pub fn uc_children(r: &greeners::UCResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let n = r.params.len();
    let params = if n > 0 {
        Array1::from_vec(r.params.clone())
    } else {
        Array1::zeros(0)
    };
    let names = r.param_names.clone();
    if !params.is_empty() {
        let coef = coef_dataframe(
            &names,
            &params,
            &Array1::zeros(n),
            &Array1::zeros(n),
            &Array1::zeros(n),
            None,
            None,
        );
        vars.push(("coefficients".into(), coef));
    }
    if !r.level.is_empty() {
        vars.push(("level".into(), array1_to_series("level", &r.level)));
    }
    if let Some(trend) = &r.trend {
        if !trend.is_empty() {
            vars.push(("trend".into(), array1_to_series("trend", trend)));
        }
    }
    if let Some(seasonal) = &r.seasonal {
        if !seasonal.is_empty() {
            vars.push(("seasonal".into(), array1_to_series("seasonal", seasonal)));
        }
    }
    if !r.residuals.is_empty() {
        vars.push((
            "residuals".into(),
            array1_to_series("residuals", &r.residuals),
        ));
    }
    vars.push(("fit".into(), uc_fit_dict(r)));
    vars
}

pub fn uc_fit_dict(r: &greeners::UCResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("level_type", Value::Str(format!("{:?}", r.level_type))),
        (
            "seasonal_type",
            Value::Str(format!("{:?}", r.seasonal_type)),
        ),
    ])
}

pub fn local_level_children(r: &greeners::LocalLevelResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let filtered: Vec<f64> = r
        .filtered_states
        .iter()
        .map(|s| s.first().copied().unwrap_or(f64::NAN))
        .collect();
    let smoothed: Vec<f64> = r
        .smoothed_states
        .iter()
        .map(|s| s.first().copied().unwrap_or(f64::NAN))
        .collect();
    if !filtered.is_empty() {
        vars.push((
            "filtered".into(),
            array1_to_series("filtered", &Array1::from_vec(filtered)),
        ));
    }
    if !smoothed.is_empty() {
        vars.push((
            "smoothed".into(),
            array1_to_series("smoothed", &Array1::from_vec(smoothed)),
        ));
    }
    vars.push(("fit".into(), local_level_fit_dict(r)));
    vars
}

pub fn local_level_fit_dict(r: &greeners::LocalLevelResult) -> Value {
    fit_dict(&[
        ("sigma_obs", Value::Float(r.sigma_obs)),
        ("sigma_state", Value::Float(r.sigma_state)),
        ("log_lik", Value::Float(r.log_likelihood)),
        ("n_obs", Value::Int(r.n_obs as i64)),
    ])
}

pub fn autoreg_children(r: &greeners::AutoRegResult) -> Vec<(String, Value)> {
    regression_children(RegressionCtx {
        names: r.param_names.clone(),
        params: &r.params,
        std_errors: &r.std_errors,
        test_values: &r.t_values,
        p_values: &r.p_values,
        conf_lower: None,
        conf_upper: None,
        fit: autoreg_fit_dict(r),
        residuals: Some(&r.residuals),
        fitted_values: Some(&r.fitted_values),
        x: None,
    })
}

pub fn autoreg_fit_dict(r: &greeners::AutoRegResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("adj_r2", Value::Float(r.adj_r_squared)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("lags", Value::Int(r.lags as i64)),
        ("trend", Value::Str(r.trend.clone())),
    ])
}

pub fn ardl_children(r: &greeners::ARDLResult) -> Vec<(String, Value)> {
    regression_children(RegressionCtx {
        names: r.param_names.clone(),
        params: &r.params,
        std_errors: &r.std_errors,
        test_values: &r.t_values,
        p_values: &r.p_values,
        conf_lower: None,
        conf_upper: None,
        fit: ardl_fit_dict(r),
        residuals: Some(&r.residuals),
        fitted_values: Some(&r.fitted_values),
        x: None,
    })
}

pub fn ardl_fit_dict(r: &greeners::ARDLResult) -> Value {
    fit_dict(&[
        ("r2", Value::Float(r.r_squared)),
        ("adj_r2", Value::Float(r.adj_r_squared)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("y_lags", Value::Int(r.y_lags as i64)),
        ("x_lags", Value::Int(r.x_lags as i64)),
    ])
}

pub fn threshold_children(r: &greeners::threshold::ThresholdResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let regime1_names: Vec<String> = (0..r.params_regime1.len())
        .map(|i| format!("x{i}"))
        .collect();
    let regime2_names: Vec<String> = (0..r.params_regime2.len())
        .map(|i| format!("x{i}"))
        .collect();
    let r1 = coef_dataframe(
        &regime1_names,
        &r.params_regime1,
        &Array1::zeros(r.params_regime1.len()),
        &Array1::zeros(r.params_regime1.len()),
        &Array1::zeros(r.params_regime1.len()),
        None,
        None,
    );
    let r2 = coef_dataframe(
        &regime2_names,
        &r.params_regime2,
        &Array1::zeros(r.params_regime2.len()),
        &Array1::zeros(r.params_regime2.len()),
        &Array1::zeros(r.params_regime2.len()),
        None,
        None,
    );
    vars.push(("regime1".into(), r1));
    vars.push(("regime2".into(), r2));
    vars.push(("fit".into(), threshold_fit_dict(r)));
    vars
}

pub fn threshold_fit_dict(r: &greeners::threshold::ThresholdResult) -> Value {
    fit_dict(&[
        ("threshold_gamma", Value::Float(r.threshold_gamma)),
        ("r2", Value::Float(r.r_squared)),
        ("ssr_min", Value::Float(r.ssr_min)),
        ("n_search", Value::Int(r.n_search as i64)),
    ])
}

pub fn var_children(r: &greeners::var::VarResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let k = r.n_vars;
    let p = r.lags;
    let param_names = var_param_names(k, p, &r.var_names);
    let total_rows = r.params.nrows();
    for i in 0..k {
        let dep = r
            .var_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("var{i}"));
        let params_col = r.params.column(i).to_owned();
        let se_col = r.std_errors.column(i).to_owned();
        let zeros = Array1::zeros(total_rows);
        let coef = coef_dataframe(
            &param_names,
            &params_col,
            &se_col,
            &zeros,
            &zeros,
            None,
            None,
        );
        let mut wrap = HashMap::new();
        wrap.insert("coefficients".into(), coef);
        vars.push((dep, Value::Dict(Arc::new(wrap))));
    }
    vars.push((
        "sigma_u".into(),
        array2_to_dataframe_named(&r.sigma_u, &r.var_names),
    ));
    vars.push(("fit".into(), var_fit_dict(r)));
    vars
}

pub fn var_fit_dict(r: &greeners::var::VarResult) -> Value {
    fit_dict(&[
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("lags", Value::Int(r.lags as i64)),
        ("n_vars", Value::Int(r.n_vars as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
    ])
}

pub fn vecm_children(r: &greeners::vecm::VecmResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let row_labels: Vec<String> = (0..r.beta.nrows()).map(|i| format!("eq{i}")).collect();
    let col_labels = r.variable_names.clone();
    vars.push((
        "beta".into(),
        array2_to_dataframe_named(&r.beta, &col_labels),
    ));
    vars.push((
        "alpha".into(),
        array2_to_dataframe_named(&r.alpha, &col_labels),
    ));
    vars.push((
        "gamma".into(),
        array2_to_dataframe_named(&r.gamma, &row_labels),
    ));
    vars.push((
        "std_errors_beta".into(),
        array2_to_dataframe_named(&r.std_errors_beta, &col_labels),
    ));
    vars.push((
        "std_errors_alpha".into(),
        array2_to_dataframe_named(&r.std_errors_alpha, &col_labels),
    ));
    vars.push((
        "std_errors_gamma".into(),
        array2_to_dataframe_named(&r.std_errors_gamma, &row_labels),
    ));
    vars.push((
        "residuals".into(),
        array2_to_dataframe_named(&r.residuals, &col_labels),
    ));
    vars.push((
        "eigenvalues".into(),
        array1_to_series("eigenvalues", &r.eigenvalues),
    ));
    vars.push(("fit".into(), vecm_fit_dict(r)));
    vars
}

pub fn vecm_fit_dict(r: &greeners::vecm::VecmResult) -> Value {
    fit_dict(&[
        ("rank", Value::Int(r.rank as i64)),
        ("n_vars", Value::Int(r.n_vars as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("lags", Value::Int(r.lags as i64)),
    ])
}

pub fn varma_children(r: &greeners::varma::VarmaResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let col_names: Vec<String> = (0..r.n_vars).map(|i| format!("var{i}")).collect();
    let ar_col_names: Vec<String> = (0..r.ar_params.ncols())
        .map(|i| format!("col{i}"))
        .collect();
    let ma_col_names: Vec<String> = (0..r.ma_params.ncols())
        .map(|i| format!("col{i}"))
        .collect();
    vars.push((
        "ar_params".into(),
        array2_to_dataframe_named(&r.ar_params, &ar_col_names),
    ));
    vars.push((
        "ma_params".into(),
        array2_to_dataframe_named(&r.ma_params, &ma_col_names),
    ));
    if let Some(exog) = &r.exog_params {
        let exog_names: Vec<String> = (0..exog.ncols()).map(|i| format!("x{i}")).collect();
        vars.push((
            "exog_params".into(),
            array2_to_dataframe_named(exog, &exog_names),
        ));
    }
    vars.push((
        "sigma_u".into(),
        array2_to_dataframe_named(&r.sigma_u, &col_names),
    ));
    vars.push(("fit".into(), varma_fit_dict(r)));
    vars
}

pub fn varma_fit_dict(r: &greeners::varma::VarmaResult) -> Value {
    fit_dict(&[
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("p_lags", Value::Int(r.p_lags as i64)),
        ("q_lags", Value::Int(r.q_lags as i64)),
        ("n_vars", Value::Int(r.n_vars as i64)),
        ("n_exog", Value::Int(r.n_exog as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
    ])
}

pub fn svar_children(r: &greeners::svar::SVarResult) -> Vec<(String, Value)> {
    let mut vars = var_children(&r.var_result);
    let k = r.var_result.n_vars;
    let names: Vec<String> = (0..k).map(|i| format!("var{i}")).collect();
    vars.push((
        "a_matrix".into(),
        array2_to_dataframe_named(&r.a_matrix, &names),
    ));
    vars.push((
        "b_matrix".into(),
        array2_to_dataframe_named(&r.b_matrix, &names),
    ));
    vars.push((
        "identification".into(),
        Value::Str(r.identification.clone()),
    ));
    vars
}

pub fn msar_children(r: &greeners::markov_autoreg::MarkovAutoregResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    for j in 0..r.k_regimes {
        let regime_name = format!("regime_{j}");
        let mut map = HashMap::new();
        map.insert("mean".into(), Value::Float(r.regime_means[j]));
        map.insert("sigma".into(), Value::Float(r.regime_sigmas[j]));
        let ar_names: Vec<String> = (0..r.ar_order).map(|l| format!("AR.L{}", l + 1)).collect();
        let ar = r.ar_params.row(j).to_owned();
        let zeros = Array1::zeros(ar.len());
        let ar_df = coef_dataframe(&ar_names, &ar, &zeros, &zeros, &zeros, None, None);
        map.insert("ar_coefficients".into(), ar_df);
        vars.push((regime_name, Value::Dict(Arc::new(map))));
    }
    let regime_labels: Vec<String> = (0..r.k_regimes).map(|i| format!("regime_{i}")).collect();
    vars.push((
        "transition_matrix".into(),
        array2_to_dataframe_named(&r.transition_matrix, &regime_labels),
    ));
    vars.push((
        "smoothed_probs".into(),
        array2_to_dataframe_named(&r.smoothed_probs, &regime_labels),
    ));
    vars.push((
        "filtered_probs".into(),
        array2_to_dataframe_named(&r.filtered_probs, &regime_labels),
    ));
    vars.push(("fit".into(), msar_fit_dict(r)));
    vars
}

pub fn msar_fit_dict(r: &greeners::markov_autoreg::MarkovAutoregResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("k_regimes", Value::Int(r.k_regimes as i64)),
        ("ar_order", Value::Int(r.ar_order as i64)),
    ])
}

pub fn dfm_children(m: &DFMModel) -> Vec<(String, Value)> {
    let r = &*m.result;
    let mut vars = Vec::new();
    let factor_names: Vec<String> = (0..r.n_factors).map(|i| format!("F{}", i + 1)).collect();
    vars.push((
        "factors".into(),
        array2_to_dataframe_named(&r.factors, &factor_names),
    ));
    vars.push((
        "loadings".into(),
        array2_to_dataframe_named(&r.factor_loadings, &factor_names),
    ));
    let ar_names: Vec<String> = (0..r.factor_order)
        .map(|i| format!("AR{}", i + 1))
        .collect();
    for (i, ar) in r.factor_ar_params.iter().enumerate() {
        let name = ar_names.get(i).cloned().unwrap_or_else(|| format!("AR{i}"));
        vars.push((name.clone(), array2_to_dataframe_named(ar, &factor_names)));
    }
    vars.push((
        "factor_cov".into(),
        array2_to_dataframe_named(&r.sigma_factor, &factor_names),
    ));
    vars.push((
        "obs_variances".into(),
        array1_to_series("obs_variances", &r.sigma_obs),
    ));
    vars.push(("fit".into(), dfm_fit_dict(r)));
    vars
}

pub fn dfm_fit_dict(r: &greeners::DynamicFactorResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_series", Value::Int(r.n_vars as i64)),
        ("n_factors", Value::Int(r.n_factors as i64)),
        ("factor_order", Value::Int(r.factor_order as i64)),
    ])
}

pub fn markov_children(r: &greeners::markov::MarkovSwitchingResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    for j in 0..r.n_regimes {
        let regime_name = format!("regime_{j}");
        let mut map = HashMap::new();
        let params = &r.regime_params[j];
        let ar_names: Vec<String> = (0..r.ar_order).map(|l| format!("AR.L{}", l + 1)).collect();
        let zeros = Array1::zeros(params.len());
        let coef = coef_dataframe(&ar_names, params, &zeros, &zeros, &zeros, None, None);
        map.insert("coefficients".into(), coef);
        map.insert("variance".into(), Value::Float(r.regime_variances[j]));
        vars.push((regime_name, Value::Dict(Arc::new(map))));
    }
    let regime_labels: Vec<String> = (0..r.n_regimes).map(|i| format!("regime_{i}")).collect();
    vars.push((
        "transition_matrix".into(),
        array2_to_dataframe_named(&r.transition_matrix, &regime_labels),
    ));
    vars.push((
        "smoothed_probs".into(),
        array2_to_dataframe_named(&r.smoothed_probs, &regime_labels),
    ));
    vars.push((
        "filtered_probs".into(),
        array2_to_dataframe_named(&r.filtered_probs, &regime_labels),
    ));
    vars.push(("fit".into(), markov_fit_dict(r)));
    vars
}

pub fn markov_fit_dict(r: &greeners::markov::MarkovSwitchingResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_regimes", Value::Int(r.n_regimes as i64)),
        ("ar_order", Value::Int(r.ar_order as i64)),
    ])
}

pub fn rolling_children(r: &greeners::RollingResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let k = r.params_history.ncols();
    let param_names = r
        .variable_names
        .clone()
        .unwrap_or_else(|| (0..k).map(|i| format!("beta{i}")).collect());
    vars.push((
        "params_history".into(),
        array2_to_dataframe_named(&r.params_history, &param_names),
    ));
    vars.push((
        "r_squared_history".into(),
        array1_to_series("r_squared_history", &r.r_squared_history),
    ));
    vars.push((
        "residuals".into(),
        array1_to_series("residuals", &r.residuals),
    ));
    vars.push(("fit".into(), rolling_fit_dict(r)));
    vars
}

pub fn rolling_fit_dict(r: &greeners::RollingResult) -> Value {
    fit_dict(&[
        ("window", Value::Int(r.window as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
    ])
}

pub fn recursive_ls_children(r: &greeners::RecursiveLSResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let k = r.params_history.ncols();
    let param_names: Vec<String> = (0..k).map(|i| format!("beta{i}")).collect();
    vars.push((
        "params_history".into(),
        array2_to_dataframe_named(&r.params_history, &param_names),
    ));
    vars.push(("params".into(), array1_to_series("params", &r.params)));
    vars.push((
        "residuals".into(),
        array1_to_series("residuals", &r.residuals),
    ));
    vars.push(("cusum".into(), array1_to_series("cusum", &r.cusum)));
    vars.push((
        "cusum_squares".into(),
        array1_to_series("cusum_squares", &r.cusum_squares),
    ));
    vars.push(("fit".into(), recursive_ls_fit_dict(r)));
    vars
}

pub fn recursive_ls_fit_dict(r: &greeners::RecursiveLSResult) -> Value {
    fit_dict(&[("n_obs", Value::Int(r.n_obs as i64))])
}

pub fn decomp_children(r: &greeners::DecompositionResult) -> Vec<(String, Value)> {
    vec![
        ("observed".into(), array1_to_series("observed", &r.observed)),
        ("trend".into(), array1_to_series("trend", &r.trend)),
        ("seasonal".into(), array1_to_series("seasonal", &r.seasonal)),
        ("residual".into(), array1_to_series("residual", &r.residual)),
        ("fit".into(), decomp_fit_dict(r)),
    ]
}

pub fn decomp_fit_dict(r: &greeners::DecompositionResult) -> Value {
    fit_dict(&[("model", Value::Str(r.model.clone()))])
}
