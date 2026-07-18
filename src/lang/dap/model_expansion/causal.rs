use super::*;

pub fn rd_children(r: &greeners::RdResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let mut map = HashMap::new();
    map.insert("tau".into(), Value::Float(r.tau));
    map.insert("se".into(), Value::Float(r.se));
    map.insert("z".into(), Value::Float(r.z));
    map.insert("p_value".into(), Value::Float(r.p_value));
    map.insert("ci_lower".into(), Value::Float(r.ci_lower));
    map.insert("ci_upper".into(), Value::Float(r.ci_upper));
    map.insert("bandwidth".into(), Value::Float(r.bandwidth));
    map.insert("cutoff".into(), Value::Float(r.cutoff));
    map.insert("poly_order".into(), Value::Int(r.poly_order as i64));
    map.insert("n_left".into(), Value::Int(r.n_left as i64));
    map.insert("n_right".into(), Value::Int(r.n_right as i64));
    map.insert("n_total".into(), Value::Int(r.n_total as i64));
    map.insert("is_fuzzy".into(), Value::Bool(r.is_fuzzy));
    if let (Some(tau), Some(se)) = (r.first_stage_tau, r.first_stage_se) {
        map.insert("first_stage_tau".into(), Value::Float(tau));
        map.insert("first_stage_se".into(), Value::Float(se));
    }
    vars.push(("fit".into(), Value::Dict(Arc::new(map))));
    vars
}

pub fn synth_children(r: &greeners::SynthResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let mut weights = HashMap::new();
    for (unit, w) in &r.weights {
        weights.insert(unit.clone(), Value::Float(*w));
    }
    vars.push(("weights".into(), Value::Dict(Arc::new(weights))));
    vars.push((
        "synthetic_series".into(),
        Value::Series(Arc::new(Series::new(
            "synthetic_series",
            r.synthetic_series
                .iter()
                .map(|&v| Value::Float(v))
                .collect(),
        ))),
    ));
    vars.push((
        "actual_series".into(),
        Value::Series(Arc::new(Series::new(
            "actual_series",
            r.actual_series.iter().map(|&v| Value::Float(v)).collect(),
        ))),
    ));
    vars.push((
        "time_index".into(),
        Value::Series(Arc::new(Series::new(
            "time_index",
            r.time_index.iter().map(|&v| Value::Float(v)).collect(),
        ))),
    ));
    vars.push(("fit".into(), synth_fit_dict(r)));
    vars
}

pub fn synth_fit_dict(r: &greeners::SynthResult) -> Value {
    let mut entries = vec![
        ("treated_unit".into(), Value::Str(r.treated_unit.clone())),
        ("outcome_name".into(), Value::Str(r.outcome_name.clone())),
        ("t0".into(), Value::Float(r.t0)),
        ("rmspe_pre".into(), Value::Float(r.rmspe_pre)),
        ("n_donors".into(), Value::Int(r.n_donors as i64)),
        ("t_pre".into(), Value::Int(r.t_pre as i64)),
        ("t_post".into(), Value::Int(r.t_post as i64)),
    ];
    if let Some(rp) = r.rmspe_post {
        entries.push(("rmspe_post".into(), Value::Float(rp)));
    }
    Value::Dict(Arc::new(HashMap::from_iter(entries)))
}

pub fn psm_children(r: &greeners::PsmResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let mut map = HashMap::new();
    map.insert("att".into(), Value::Float(r.att));
    map.insert("se".into(), Value::Float(r.se));
    map.insert("z".into(), Value::Float(r.z));
    map.insert("p_value".into(), Value::Float(r.p_value));
    map.insert("ci_lower".into(), Value::Float(r.ci_lower));
    map.insert("ci_upper".into(), Value::Float(r.ci_upper));
    map.insert("n_treated".into(), Value::Int(r.n_treated as i64));
    map.insert("n_control".into(), Value::Int(r.n_control as i64));
    map.insert(
        "n_matched_treated".into(),
        Value::Int(r.n_matched_treated as i64),
    );
    map.insert("k".into(), Value::Int(r.k as i64));
    map.insert("n_boot".into(), Value::Int(r.n_boot as i64));
    if let Some(c) = r.caliper {
        map.insert("caliper".into(), Value::Float(c));
    }
    vars.push(("fit".into(), Value::Dict(Arc::new(map))));
    vars.push((
        "propensity_scores".into(),
        array1_to_series("propensity_scores", &r.propensity_scores),
    ));

    let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();
    columns.insert(
        "covariate".into(),
        greeners::Column::from_strings(r.balance.iter().map(|b| b.covariate.clone()).collect()),
    );
    columns.insert(
        "mean_treated".into(),
        f64_array_column(&Array1::from(
            r.balance.iter().map(|b| b.mean_treated).collect::<Vec<_>>(),
        )),
    );
    columns.insert(
        "mean_control_raw".into(),
        f64_array_column(&Array1::from(
            r.balance
                .iter()
                .map(|b| b.mean_control_raw)
                .collect::<Vec<_>>(),
        )),
    );
    columns.insert(
        "mean_control_matched".into(),
        f64_array_column(&Array1::from(
            r.balance
                .iter()
                .map(|b| b.mean_control_matched)
                .collect::<Vec<_>>(),
        )),
    );
    columns.insert(
        "smd_before".into(),
        f64_array_column(&Array1::from(
            r.balance.iter().map(|b| b.smd_before).collect::<Vec<_>>(),
        )),
    );
    columns.insert(
        "smd_after".into(),
        f64_array_column(&Array1::from(
            r.balance.iter().map(|b| b.smd_after).collect::<Vec<_>>(),
        )),
    );
    let balance_df = Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(columns)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ));
    vars.push(("balance".into(), balance_df));
    vars
}

pub fn did_children(r: &greeners::DidResult) -> Vec<(String, Value)> {
    regression_children(RegressionCtx {
        names: r.variable_names.clone(),
        params: &r.params,
        std_errors: &r.std_errors,
        test_values: &r.t_values,
        p_values: &r.p_values,
        conf_lower: None,
        conf_upper: None,
        fit: did_fit_dict(r),
        residuals: None,
        fitted_values: None,
        x: None,
    })
}

pub fn did_fit_dict(r: &greeners::DidResult) -> Value {
    fit_dict(&[
        ("att", Value::Float(r.att)),
        ("std_error", Value::Float(r.std_error)),
        ("t_stat", Value::Float(r.t_stat)),
        ("p_value", Value::Float(r.p_value)),
        ("r2", Value::Float(r.r_squared)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("control_pre_mean", Value::Float(r.control_pre_mean)),
        ("control_post_mean", Value::Float(r.control_post_mean)),
        ("treated_pre_mean", Value::Float(r.treated_pre_mean)),
        ("treated_post_mean", Value::Float(r.treated_post_mean)),
        ("cov_type", Value::Str(format!("{:?}", r.cov_type))),
    ])
}

pub fn km_children(r: &greeners::KMResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let n = r.times.len();
    let names: Vec<String> = (0..n).map(|i| format!("t{i}")).collect();
    let coef = coef_dataframe(
        &names,
        &r.survival_probs,
        &r.std_errors,
        &r.conf_lower,
        &r.conf_upper,
        None,
        None,
    );
    vars.push(("survival_table".into(), coef));
    vars.push(("times".into(), array1_to_series("times", &r.times)));
    vars.push((
        "survival_probs".into(),
        array1_to_series("survival_probs", &r.survival_probs),
    ));
    vars.push(("fit".into(), km_fit_dict(r)));
    vars
}

pub fn km_fit_dict(r: &greeners::KMResult) -> Value {
    fit_dict(&[
        ("median_survival", Value::Float(r.median_survival)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_events", Value::Int(r.n_events as i64)),
    ])
}

pub fn cox_children(r: &greeners::CoxResult) -> Vec<(String, Value)> {
    let mut vars = regression_children(RegressionCtx {
        names: r.variable_names.clone().unwrap_or_default(),
        params: &r.params,
        std_errors: &r.std_errors,
        test_values: &r.z_values,
        p_values: &r.p_values,
        conf_lower: None,
        conf_upper: None,
        fit: cox_fit_dict(r),
        residuals: None,
        fitted_values: None,
        x: None,
    });
    vars.push((
        "hazard_ratios".into(),
        array1_to_series("hazard_ratios", &r.hazard_ratios),
    ));
    vars
}

pub fn cox_fit_dict(r: &greeners::CoxResult) -> Value {
    fit_dict(&[
        ("log_lik", Value::Float(r.log_likelihood)),
        ("concordance", Value::Float(r.concordance)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_events", Value::Int(r.n_events as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

pub fn heckman_children(r: &greeners::HeckmanResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let outcome_names = r.variable_names.clone().unwrap_or_default();
    let select_names = r.select_names.clone().unwrap_or_default();
    let outcome_coef = coef_dataframe(
        &outcome_names,
        &r.params,
        &r.std_errors,
        &r.t_values,
        &r.p_values,
        None,
        None,
    );
    let select_zeros: Array1<f64> = Array1::zeros(r.select_params.len());
    let select_coef = coef_dataframe(
        &select_names,
        &r.select_params,
        &r.select_se,
        &select_zeros,
        &select_zeros,
        None,
        None,
    );
    vars.push(("outcome_coefficients".into(), outcome_coef));
    vars.push(("selection_coefficients".into(), select_coef));
    vars.push(("fit".into(), heckman_fit_dict(r)));
    vars
}

pub fn heckman_fit_dict(r: &greeners::HeckmanResult) -> Value {
    fit_dict(&[
        ("delta", Value::Float(r.delta)),
        ("delta_se", Value::Float(r.delta_se)),
        ("rho", Value::Float(r.rho)),
        ("sigma", Value::Float(r.sigma)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_selected", Value::Int(r.n_selected as i64)),
    ])
}

pub fn lowess_children(r: &greeners::LowessResult) -> Vec<(String, Value)> {
    vec![
        ("smoothed".into(), array1_to_series("smoothed", &r.smoothed)),
        (
            "residuals".into(),
            array1_to_series("residuals", &r.residuals),
        ),
        ("fit".into(), lowess_fit_dict(r)),
    ]
}

pub fn lowess_fit_dict(r: &greeners::LowessResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("frac", Value::Float(r.frac)),
    ])
}
