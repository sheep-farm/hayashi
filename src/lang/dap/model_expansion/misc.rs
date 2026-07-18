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

pub fn isotonic_children(r: &greeners::IsotonicResult) -> Vec<(String, Value)> {
    vec![
        ("x".into(), array1_to_series("x", &r.x)),
        ("y".into(), array1_to_series("y", &r.y)),
        ("weights".into(), array1_to_series("weights", &r.weights)),
        ("fitted".into(), array1_to_series("fitted", &r.fitted)),
        ("x_steps".into(), series_from_vec("x_steps", &r.x_steps)),
        ("y_steps".into(), series_from_vec("y_steps", &r.y_steps)),
        ("fit".into(), isotonic_fit_dict(r)),
    ]
}

pub fn isotonic_fit_dict(r: &greeners::IsotonicResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_blocks", Value::Int(r.n_blocks as i64)),
        ("increasing", Value::Bool(r.increasing)),
        ("r_squared", Value::Float(r.r_squared)),
        ("mse", Value::Float(r.mse)),
    ])
}

pub fn kde_children(r: &greeners::KDEResult) -> Vec<(String, Value)> {
    vec![
        ("support".into(), array1_to_series("support", &r.support)),
        ("density".into(), array1_to_series("density", &r.density)),
        ("fit".into(), kde_fit_dict(r)),
    ]
}

pub fn kde_fit_dict(r: &greeners::KDEResult) -> Value {
    let support_min = r.support.iter().cloned().fold(f64::INFINITY, f64::min);
    let support_max = r.support.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (peak_idx, peak_density) = r
        .density
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));
    let peak_x = r.support.get(peak_idx).copied().unwrap_or(0.0);
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("bandwidth", Value::Float(r.bandwidth)),
        ("n_points", Value::Int(r.support.len() as i64)),
        ("support_min", Value::Float(support_min)),
        ("support_max", Value::Float(support_max)),
        ("peak_density", Value::Float(peak_density)),
        ("peak_x", Value::Float(peak_x)),
    ])
}

pub fn bart_children(r: &greeners::BartResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    vars.push(("fitted".into(), array1_to_series("fitted", &r.fitted)));

    let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();
    let names = r.variable_names.clone();
    let var_col: Vec<String> = (0..r.variable_inclusion.len())
        .map(|i| names.get(i).cloned().unwrap_or_else(|| format!("x{i}")))
        .collect();
    columns.insert(
        "variable".into(),
        greeners::Column::String(ndarray::Array1::from(var_col)),
    );
    let incl_col: Vec<f64> = r.variable_inclusion.iter().copied().collect();
    columns.insert(
        "inclusion".into(),
        greeners::Column::Float(ndarray::Array1::from(incl_col)),
    );
    let var_df = Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(columns)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ));
    vars.push(("variable_inclusion".into(), var_df));

    let samples: Vec<Value> = r.sigma2_samples.iter().map(|&v| Value::Float(v)).collect();
    vars.push((
        "sigma2_samples".into(),
        Value::Series(Arc::new(Series::new("sigma2_samples", samples))),
    ));
    vars.push(("fit".into(), bart_fit_dict(r)));
    vars
}

pub fn bart_fit_dict(r: &greeners::BartResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_features", Value::Int(r.n_features as i64)),
        ("n_trees", Value::Int(r.n_trees as i64)),
        ("max_depth", Value::Int(r.max_depth as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("burn_in", Value::Int(r.burn_in as i64)),
        ("sigma2", Value::Float(r.sigma2)),
        ("r_squared", Value::Float(r.r_squared)),
        ("mse", Value::Float(r.mse)),
    ])
}

pub fn gp_children(r: &greeners::GpResult) -> Vec<(String, Value)> {
    vec![
        ("fitted".into(), array1_to_series("fitted", &r.fitted)),
        (
            "fitted_sd".into(),
            array1_to_series("fitted_sd", &r.fitted_sd),
        ),
        ("fit".into(), gp_fit_dict(r)),
    ]
}

pub fn gp_fit_dict(r: &greeners::GpResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_features", Value::Int(r.n_features as i64)),
        ("length_scale", Value::Float(r.length_scale)),
        ("signal_variance", Value::Float(r.signal_variance)),
        ("noise_variance", Value::Float(r.noise_variance)),
        ("log_marginal", Value::Float(r.log_marginal)),
        ("r_squared", Value::Float(r.r_squared)),
        ("mse", Value::Float(r.mse)),
    ])
}

pub fn gmm_clustering_children(r: &greeners::GmmClusteringResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let labels: Vec<Value> = r.labels.iter().map(|&l| Value::Int(l as i64)).collect();
    vars.push((
        "labels".into(),
        Value::Series(Arc::new(Series::new("labels", labels))),
    ));
    vars.push(("means".into(), array2_to_dataframe("means", &r.means)));
    let covariances: Vec<Value> = r
        .covariances
        .iter()
        .enumerate()
        .map(|(i, m)| array2_to_dataframe(&format!("covariance_{i}"), m))
        .collect();
    vars.push(("covariances".into(), Value::List(Arc::new(covariances))));
    vars.push(("weights".into(), array1_to_series("weights", &r.weights)));
    vars.push((
        "responsibilities".into(),
        array2_to_dataframe("responsibilities", &r.responsibilities),
    ));
    vars.push(("fit".into(), gmm_clustering_fit_dict(r)));
    vars
}

pub fn gmm_clustering_fit_dict(r: &greeners::GmmClusteringResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_features", Value::Int(r.n_features as i64)),
        ("n_clusters", Value::Int(r.n_clusters as i64)),
        ("n_iter", Value::Int(r.n_iter as i64)),
        ("converged", Value::Bool(r.converged)),
        ("log_likelihood", Value::Float(r.log_likelihood)),
        ("aic", Value::Float(r.aic)),
        ("bic", Value::Float(r.bic)),
    ])
}

pub fn hierarchical_children(r: &greeners::HierarchicalResult) -> Vec<(String, Value)> {
    let mut columns: IndexMap<String, greeners::Column> = IndexMap::new();
    let cluster_a: Vec<i64> = r.merges.iter().map(|m| m.cluster_a as i64).collect();
    let cluster_b: Vec<i64> = r.merges.iter().map(|m| m.cluster_b as i64).collect();
    let distance: Vec<f64> = r.merges.iter().map(|m| m.distance).collect();
    let size: Vec<i64> = r.merges.iter().map(|m| m.size as i64).collect();
    columns.insert(
        "cluster_a".into(),
        greeners::Column::Int(ndarray::Array1::from(cluster_a)),
    );
    columns.insert(
        "cluster_b".into(),
        greeners::Column::Int(ndarray::Array1::from(cluster_b)),
    );
    columns.insert(
        "distance".into(),
        greeners::Column::Float(ndarray::Array1::from(distance)),
    );
    columns.insert(
        "size".into(),
        greeners::Column::Int(ndarray::Array1::from(size)),
    );
    let merges_df = Value::DataFrame(Arc::new(
        greeners::DataFrame::from_columns(columns)
            .unwrap_or_else(|_| greeners::DataFrame::from_columns(IndexMap::new()).unwrap()),
    ));

    let mut vars = Vec::new();
    vars.push(("merges".into(), merges_df));
    let labels: Vec<Value> = r.labels.iter().map(|&l| Value::Int(l as i64)).collect();
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
    vars.push(("fit".into(), hierarchical_fit_dict(r)));
    vars
}

pub fn hierarchical_fit_dict(r: &greeners::HierarchicalResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_features", Value::Int(r.n_features as i64)),
        ("n_clusters", Value::Int(r.n_clusters as i64)),
        ("cut_height", Value::Float(r.cut_height)),
        ("linkage", Value::Str(format!("{:?}", r.linkage))),
        ("cophenetic_corr", Value::Float(r.cophenetic_corr)),
    ])
}

pub fn spectral_children(r: &greeners::SpectralResult) -> Vec<(String, Value)> {
    let mut vars = Vec::new();
    let labels: Vec<Value> = r.labels.iter().map(|&l| Value::Int(l as i64)).collect();
    vars.push((
        "labels".into(),
        Value::Series(Arc::new(Series::new("labels", labels))),
    ));
    vars.push((
        "affinity".into(),
        array2_to_dataframe("affinity", &r.affinity),
    ));
    vars.push((
        "eigenvalues".into(),
        array1_to_series("eigenvalues", &r.eigenvalues),
    ));
    vars.push((
        "eigenvectors".into(),
        array2_to_dataframe("eigenvectors", &r.eigenvectors),
    ));
    vars.push((
        "centroids".into(),
        array2_to_dataframe("centroids", &r.centroids),
    ));
    vars.push(("fit".into(), spectral_fit_dict(r)));
    vars
}

pub fn spectral_fit_dict(r: &greeners::SpectralResult) -> Value {
    fit_dict(&[
        ("n_obs", Value::Int(r.n_obs as i64)),
        ("n_features", Value::Int(r.n_features as i64)),
        ("n_clusters", Value::Int(r.n_clusters as i64)),
        ("sigma", Value::Float(r.sigma)),
        ("inertia", Value::Float(r.inertia)),
    ])
}
