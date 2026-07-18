use super::*;

pub fn penalized_children(m: &PenalizedModel) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let zeros: Array1<f64> = Array1::zeros(m.params.len());
    let coef = coef_dataframe(
        &m.variable_names,
        &m.params,
        &m.std_errors,
        &zeros,
        &zeros,
        None,
        None,
    );
    vars.push(("coefficients".into(), coef));
    vars.push(("fit".into(), penalized_fit_dict(m)));
    vars
}

pub fn penalized_fit_dict(m: &PenalizedModel) -> Value {
    let mut entries = vec![
        ("kind", Value::Str(m.kind.clone())),
        ("n_obs", Value::Int(m.n_obs as i64)),
        ("alpha", Value::Float(m.alpha)),
        ("r2", Value::Float(m.r_squared)),
    ];
    if let Some(l1r) = m.l1_ratio {
        entries.push(("l1_ratio", Value::Float(l1r)));
    }
    fit_dict(&entries)
}

pub fn pca_children(m: &PcaModel) -> Vec<(String, Value)> {
    let r = &*m.result;
    let mut vars = Vec::new();
    let comp_names: Vec<String> = (0..r.n_components)
        .map(|i| format!("PC{}", i + 1))
        .collect();
    vars.push((
        "components".into(),
        array2_to_dataframe_named(&r.components, &comp_names),
    ));
    vars.push((
        "loadings".into(),
        array2_to_dataframe_named(&r.loadings, &comp_names),
    ));
    vars.push((
        "scores".into(),
        array2_to_dataframe_named(&r.scores, &comp_names),
    ));
    vars.push((
        "explained_variance".into(),
        array1_to_series("explained_variance", &r.explained_variance),
    ));
    vars.push((
        "explained_variance_ratio".into(),
        array1_to_series("explained_variance_ratio", &r.explained_variance_ratio),
    ));
    vars.push(("mean".into(), array1_to_series("mean", &r.mean)));
    vars.push(("std".into(), array1_to_series("std", &r.std)));
    vars.push(("fit".into(), pca_fit_dict(r)));
    vars
}

pub fn pca_fit_dict(r: &greeners::PCAResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_components", Value::Int(r.n_components as i64)),
    ])
}

pub fn factor_children(m: &FactorModel) -> Vec<(String, Value)> {
    let r = &*m.result;
    let mut vars = Vec::new();
    let factor_names: Vec<String> = (0..r.n_factors).map(|i| format!("F{}", i + 1)).collect();
    vars.push((
        "loadings".into(),
        array2_to_dataframe_named(&r.loadings, &factor_names),
    ));
    vars.push((
        "communalities".into(),
        array1_to_series("communalities", &r.communalities),
    ));
    vars.push((
        "uniquenesses".into(),
        array1_to_series("uniquenesses", &r.uniquenesses),
    ));
    vars.push((
        "eigenvalues".into(),
        array1_to_series("eigenvalues", &r.eigenvalues),
    ));
    vars.push(("fit".into(), factor_fit_dict(r)));
    vars
}

pub fn factor_fit_dict(r: &greeners::FactorResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_factors", Value::Int(r.n_factors as i64)),
    ])
}

pub fn mice_children(r: &greeners::MICEResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let mut imputations = Vec::new();
    for (i, ds) in r.datasets.iter().enumerate() {
        let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();
        for (name, arr) in ds {
            let col: Vec<f64> = arr.iter().copied().collect();
            columns.insert(name.clone(), greeners::Column::Float(Array1::from(col)));
        }
        let df = Value::DataFrame(Arc::new(
            greeners::DataFrame::from_columns(columns)
                .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
        ));
        imputations.push((format!("imputation_{i}"), df));
    }
    vars.push((
        "imputations".into(),
        Value::Dict(Arc::new(HashMap::from_iter(imputations))),
    ));
    vars.push(("fit".into(), mice_fit_dict(r)));
    vars
}

pub fn mice_fit_dict(r: &greeners::MICEResult) -> Value {
    fit_dict(&[
        ("n_imputations", Value::Int(r.n_imputations as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_vars", Value::Int(r.n_vars as i64)),
    ])
}

pub fn gam_children(r: &greeners::GamResult) -> Vec<(String, Value)> {
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
    vars.push(("fit".into(), gam_fit_dict(r)));
    vars
}

pub fn gam_fit_dict(r: &greeners::GamResult) -> Value {
    fit_dict(&[
        ("n_linear", Value::Int(r.n_linear as i64)),
        ("n_smooth", Value::Int(r.n_smooth as i64)),
        ("edf", Value::Float(r.edf)),
        ("gcv_score", Value::Float(r.gcv_score)),
        ("scale", Value::Float(r.scale)),
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
    ])
}

pub fn kmeans_children(r: &greeners::KmeansResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let labels: Vec<Value> = r.labels.iter().map(|&l| Value::Int(l as i64)).collect();
    vars.push((
        "labels".into(),
        Value::Series(Arc::new(Series::new("labels", labels))),
    ));
    vars.push((
        "centroids".into(),
        array2_to_dataframe("centroids", &r.centroids),
    ));
    let sizes: Vec<Value> = r
        .cluster_sizes
        .iter()
        .map(|&s| Value::Int(s as i64))
        .collect();
    vars.push((
        "cluster_sizes".into(),
        Value::Series(Arc::new(Series::new("cluster_sizes", sizes))),
    ));
    vars.push(("fit".into(), kmeans_fit_dict(r)));
    vars
}

pub fn kmeans_fit_dict(r: &greeners::KmeansResult) -> Value {
    let pct = if r.total_ss > 1e-15 {
        r.between_ss / r.total_ss * 100.0
    } else {
        0.0
    };
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_features", Value::Int(r.n_features as i64)),
        ("n_clusters", Value::Int(r.n_clusters as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("inertia", Value::Float(r.inertia)),
        ("between_ss", Value::Float(r.between_ss)),
        ("total_ss", Value::Float(r.total_ss)),
        ("pct_explained", Value::Float(pct)),
    ])
}

pub fn dbscan_children(r: &greeners::DbscanResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let labels: Vec<Value> = r.labels.iter().map(|&l| Value::Int(l)).collect();
    vars.push((
        "labels".into(),
        Value::Series(Arc::new(Series::new("labels", labels))),
    ));
    let sizes: Vec<Value> = r
        .cluster_sizes
        .iter()
        .map(|&s| Value::Int(s as i64))
        .collect();
    vars.push((
        "cluster_sizes".into(),
        Value::Series(Arc::new(Series::new("cluster_sizes", sizes))),
    ));
    vars.push(("fit".into(), dbscan_fit_dict(r)));
    vars
}

pub fn dbscan_fit_dict(r: &greeners::DbscanResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_features", Value::Int(r.n_features as i64)),
        ("n_clusters", Value::Int(r.n_clusters as i64)),
        ("n_noise", Value::Int(r.n_noise as i64)),
        ("n_core", Value::Int(r.n_core as i64)),
        ("eps", Value::Float(r.eps)),
        ("min_pts", Value::Int(r.min_pts as i64)),
    ])
}
