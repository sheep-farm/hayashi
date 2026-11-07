//! IV formula, diagnostic sample and prediction-schema regressions.
#![cfg(feature = "greeners-ols")]

use hayashi_lang::lang::dap::model_expansion;
use hayashi_lang::lang::interpreter::{Interpreter, Value};
use hayashi_lang::lang::plugin::value_to_json;
use hayashi_lang::run_source;
use std::sync::Arc;

const DATA: &str = "input df\ny x z\n1.1 1.1 1\n1.7 2 1.2\n3.2 1.9 2.1\n2.8 3.2 1.7\n4.6 2.7 2.8\n4.1 4.1 2.3\n6.3 3.6 3.8\n5.9 5 3\nend\n";

fn execute(source: &str) -> Interpreter {
    let mut interpreter = Interpreter::new();
    interpreter.set_auto_display(false);
    run_source(source, &mut interpreter).expect("IV contract script must succeed");
    interpreter
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() <= 1e-8 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn number(value: &Value) -> f64 {
    match value {
        Value::Float(v) => *v,
        Value::Int(v) => *v as f64,
        other => panic!("expected number, got {other}"),
    }
}

fn diagnostic(interpreter: &Interpreter, name: &str, field: &str) -> f64 {
    let Value::Dict(result) = interpreter.env.get(name).unwrap() else {
        panic!("diagnostic result")
    };
    number(result.get(field).unwrap())
}

fn frame<'a>(interpreter: &'a Interpreter, name: &str) -> &'a greeners::DataFrame {
    let Value::DataFrame(frame) = interpreter.env.get(name).unwrap() else {
        panic!("dataframe")
    };
    frame
}

fn view(
    interpreter: &Interpreter,
    name: &str,
) -> hayashi_lang::lang::interpreter::model_view::ModelView {
    interpreter.env.get(name).unwrap().to_model_view().unwrap()
}

fn categorical_data() -> String {
    let mut source = "input df\ny x z g t s\n".to_string();
    for i in 0..24 {
        let g = 1 + i % 3;
        let x = 1.0 + 0.15 * i as f64 + 0.1 * ((i * 7) % 5) as f64;
        let z = 1.0 + 0.1 * i as f64 + 0.2 * ((i * 3) % 7) as f64;
        let t = 1.0 + 0.05 * i as f64;
        let s = 1.0 + ((i * 7) % 11) as f64;
        let y = 3.0 + 0.8 * x.ln() + g as f64 + 0.05 * t.ln() * s + 0.2 * (i as f64).sin();
        source.push_str(&format!("{y} {x} {z} {g} {t} {s}\n"));
    }
    source.push_str("end\n");
    source
}

#[test]
fn transformed_sargan_matches_independent_classical_reference() {
    for (suffix, j, p) in [
        ("", 2.912546320160687, 0.08789300526484993),
        (" - 1", 0.03509244010955295, 0.8514020885185064),
    ] {
        let interpreter = execute(&format!(
            "{DATA}let d=estat_overid(y ~ log(x){suffix}, ~ log(z)+z{suffix},df)\n"
        ));
        close(diagnostic(&interpreter, "d", "j_stat"), j);
        close(diagnostic(&interpreter, "d", "p_value"), p);
        close(diagnostic(&interpreter, "d", "df"), 1.0);
        let Value::Dict(result) = interpreter.env.get("d").unwrap() else {
            panic!("diagnostic")
        };
        assert!(!result.get("test").unwrap().to_string().contains("Hansen"));
        assert!(!result
            .get("conclusion")
            .unwrap()
            .to_string()
            .contains("instruments are valid"));
    }
}

#[test]
fn sargan_preserves_a_representable_chi_square_survival_tail() {
    let rows = (0..60)
        .flat_map(|pair| [-1, 1].map(move |sign| format!("{sign} {pair} {sign}\n")))
        .collect::<String>();
    let interpreter = execute(&format!(
        "input df\ny x z\n{rows}end\nlet d=estat_overid(y ~ x,~ x+z,df)\n"
    ));
    close(diagnostic(&interpreter, "d", "j_stat"), 120.0);
    let p = diagnostic(&interpreter, "d", "p_value");
    assert!(
        p > 0.0 && (p / 6.326068263677272e-28 - 1.0).abs() < 1e-9,
        "{p}"
    );
}

#[test]
fn classical_iv_diagnostics_preserve_response_unit_invariance() {
    for scale in [1e-100, 1e-9, 1.0, 1e100] {
        let mut interpreter = execute(DATA);
        let mut data = frame(&interpreter, "df").clone();
        data.insert(
            "y".into(),
            frame(&interpreter, "df").get("y").unwrap() * scale,
        )
        .unwrap();
        interpreter
            .env
            .set("df", Value::DataFrame(Arc::new(data)))
            .unwrap();
        run_source("let over=estat_overid(y ~ log(x),~ log(z)+z,df)\nlet endog=estat_endog(y ~ log(x),~ log(z)+z,df)\n",&mut interpreter).unwrap();
        close(
            diagnostic(&interpreter, "over", "j_stat"),
            2.912546320160687,
        );
        close(
            diagnostic(&interpreter, "over", "p_value"),
            0.08789300526484993,
        );
        close(
            diagnostic(&interpreter, "endog", "f_stat"),
            5.126355896346584,
        );
        close(
            diagnostic(&interpreter, "endog", "p_value"),
            0.0729717561736179,
        );
    }
    let mut interpreter = execute(DATA);
    let mut data = frame(&interpreter, "df").clone();
    let almost_exact = ndarray::Array1::from_iter(
        frame(&interpreter, "df")
            .get("x")
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, &x)| x + if i % 2 == 0 { 1e-12 } else { -1e-12 }),
    );
    data.insert("y".into(), almost_exact).unwrap();
    interpreter
        .env
        .set("df", Value::DataFrame(Arc::new(data)))
        .unwrap();
    let error = run_source("estat_overid(y ~ x,~ z+x:x,df)\n", &mut interpreter)
        .expect_err("backend residual floor must not fabricate a resolved Sargan result");
    assert!(error.to_string().contains("residual"), "{error}");
}

#[test]
fn transformed_dwh_matches_independent_reference_with_and_without_intercept() {
    for (suffix, f, p, df2) in [
        ("", 5.126355896346584, 0.0729717561736179, 5.0),
        (" - 1", 102.04421710325921, 5.467104537002613e-5, 6.0),
    ] {
        let interpreter = execute(&format!(
            "{DATA}let d=estat_endog(y ~ log(x){suffix},~ log(z)+z{suffix},df)\n"
        ));
        close(diagnostic(&interpreter, "d", "f_stat"), f);
        close(diagnostic(&interpreter, "d", "p_value"), p);
        close(diagnostic(&interpreter, "d", "df_resid"), df2);
        let Value::Dict(result) = interpreter.env.get("d").unwrap() else {
            panic!("diagnostic")
        };
        let Value::List(names) = result.get("endogenous_vars").unwrap() else {
            panic!("endogenous names")
        };
        assert_eq!(
            names.iter().map(ToString::to_string).collect::<Vec<_>>(),
            ["log(x)"]
        );
        assert!(!result
            .get("conclusion")
            .unwrap()
            .to_string()
            .contains("OLS consistent"));
    }
}

#[test]
fn dwh_real_const_term_is_not_an_instrument_intercept() {
    let interpreter=execute(&format!("{DATA}let expanded=mutate(df,const=log(x))\nlet d=estat_endog(y ~ const-1,~ log(z)+z,expanded)\n"));
    close(diagnostic(&interpreter, "d", "f_stat"), 6.420874495820468);
    close(
        diagnostic(&interpreter, "d", "p_value"),
        0.04444248692826059,
    );
    close(diagnostic(&interpreter, "d", "df_resid"), 6.0);
    let Value::Dict(d) = interpreter.env.get("d").unwrap() else {
        panic!("DWH")
    };
    let Value::List(names) = d.get("endogenous_vars").unwrap() else {
        panic!("names")
    };
    assert_eq!(
        names.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["const"]
    );
    let interpreter=execute(&format!("{DATA}let expanded=mutate(df,const=log(z))\nlet d=estat_endog(y ~ const+log(x)-1,~ const+z+z:z-1,expanded)\n"));
    let Value::Dict(d) = interpreter.env.get("d").unwrap() else {
        panic!("DWH")
    };
    let Value::List(names) = d.get("endogenous_vars").unwrap() else {
        panic!("names")
    };
    assert_eq!(
        names.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["log(x)"]
    );
}

#[test]
fn iv_diagnostics_filter_the_same_estimation_rows_as_fit() {
    let interpreter = execute(&format!("{DATA}let m=iv(y ~ log(x),~ log(z)+z,df,if=x<5)\nlet over=estat_overid(y ~ log(x),~ log(z)+z,df,if=x<5)\nlet endog=estat_endog(y ~ log(x),~ log(z)+z,df,if=x<5)\n"));
    close(
        number(view(&interpreter, "m").fit.get("n_obs").unwrap()),
        7.0,
    );
    close(
        diagnostic(&interpreter, "over", "j_stat"),
        2.6801667463590846,
    );
    close(
        diagnostic(&interpreter, "over", "p_value"),
        0.10160508885120606,
    );
    close(
        diagnostic(&interpreter, "endog", "f_stat"),
        4.060702223816359,
    );
    close(
        diagnostic(&interpreter, "endog", "p_value"),
        0.11413090713462241,
    );
    close(diagnostic(&interpreter, "endog", "df_resid"), 4.0);
}

#[test]
fn dwh_excludes_common_expanded_categories_transforms_and_interactions() {
    let source = categorical_data();
    let interpreter = execute(&format!("{source}let expanded=mutate(df,lx=log(x),lz=log(z),exog=log(t)*s)\nlet d=estat_endog(y ~ C(g)+log(x)+log(t):s,~ C(g)+log(z)+log(t):s+z,df)\nlet reference=estat_endog(y ~ C(g)+lx+exog,~ C(g)+lz+exog+z,expanded)\n"));
    close(
        diagnostic(&interpreter, "d", "f_stat"),
        diagnostic(&interpreter, "reference", "f_stat"),
    );
    close(
        diagnostic(&interpreter, "d", "p_value"),
        diagnostic(&interpreter, "reference", "p_value"),
    );
    close(diagnostic(&interpreter, "d", "df"), 1.0);
    let Value::Dict(d) = interpreter.env.get("d").unwrap() else {
        panic!("DWH")
    };
    let Value::List(names) = d.get("endogenous_vars").unwrap() else {
        panic!("names")
    };
    assert_eq!(
        names.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["log(x)"]
    );
}

#[test]
fn transformed_iv_predicts_original_and_new_structural_only_data() {
    let mut interpreter=execute(&format!("{DATA}let m=iv(y ~ log(x),~ log(z),df)\npredict df fitted=m\ninput new\nx\n2\n4\nend\npredict new fitted=m\n"));
    let coefficients = view(&interpreter, "m").params;
    for name in ["df", "new"] {
        let x = frame(&interpreter, name).get("x").unwrap();
        let predicted = frame(&interpreter, name).get("fitted").unwrap();
        for (&x, &actual) in x.iter().zip(predicted.iter()) {
            close(actual, coefficients[0] + coefficients[1] * x.ln());
        }
    }
    // Formula metadata must survive model value cloning without an identity cache.
    run_source(
        "let copied=m\npredict new cloned=copied\n",
        &mut interpreter,
    )
    .unwrap();
    for (&a, &b) in frame(&interpreter, "new")
        .get("fitted")
        .unwrap()
        .iter()
        .zip(frame(&interpreter, "new").get("cloned").unwrap().iter())
    {
        close(a, b);
    }
}

#[test]
fn iv_prediction_metadata_takes_priority_over_misleading_display_columns() {
    let mut interpreter = execute(&format!(
        "{DATA}let m=iv(y ~ log(x),~ log(z),df)\ninput new\nx\n2\n4\nend\n"
    ));
    let mut data = frame(&interpreter, "new").clone();
    data.insert("log(x)".into(), ndarray::array![99., 99.])
        .unwrap();
    data.insert("__term_0".into(), ndarray::array![-99., -99.])
        .unwrap();
    interpreter
        .env
        .set("new", Value::DataFrame(Arc::new(data)))
        .unwrap();
    run_source("predict new fitted=m\n", &mut interpreter).unwrap();
    let coefficients = view(&interpreter, "m").params;
    for (&x, &actual) in frame(&interpreter, "new")
        .get("x")
        .unwrap()
        .iter()
        .zip(frame(&interpreter, "new").get("fitted").unwrap().iter())
    {
        close(actual, coefficients[0] + coefficients[1] * x.ln());
    }
}

#[test]
fn iv_no_intercept_predictions_rematerialise_transformed_interactions() {
    let mut interpreter=execute(&format!("{DATA}let augmented=mutate(df,t=x+z)\nlet m=iv(y ~ log(x)+x:t-1,~ log(z)+z:t-1,augmented)\ninput new\nx t\n2 3\n4 2\nend\npredict new fitted=m\n"));
    let coefficients = view(&interpreter, "m").params;
    assert_eq!(view(&interpreter, "m").variable_names, ["log(x)", "x:t"]);
    let expected = [
        coefficients[0] * 2.0_f64.ln() + coefficients[1] * 6.0,
        coefficients[0] * 4.0_f64.ln() + coefficients[1] * 8.0,
    ];
    for (&a, &b) in frame(&interpreter, "new")
        .get("fitted")
        .unwrap()
        .iter()
        .zip(expected.iter())
    {
        close(a, b);
    }
    run_source("predict augmented original=m\n", &mut interpreter).unwrap();
}

#[test]
fn iv_prediction_preserves_retained_column_order_after_collinearity_removal() {
    let interpreter = execute(&format!("{DATA}let m=iv(y ~ log(x)+log(x):2,~ log(z)+z,df)\nlet expanded=mutate(df,lx=log(x),duplicate=2*log(x))\nlet reference=iv(y ~ lx+duplicate,~ log(z)+z,expanded)\ninput new\nx\n2\n4\nend\nlet new_expanded=mutate(new,lx=log(x),duplicate=2*log(x))\npredict new fitted=m\npredict new_expanded fitted=reference\n"));
    assert_eq!(view(&interpreter, "m").params.len(), 2);
    for (&actual, &expected) in frame(&interpreter, "new")
        .get("fitted")
        .unwrap()
        .iter()
        .zip(
            frame(&interpreter, "new_expanded")
                .get("fitted")
                .unwrap()
                .iter(),
        )
    {
        close(actual, expected);
    }
}

#[test]
fn iv_categorical_predictions_keep_training_dummy_columns_and_baseline() {
    let interpreter=execute(&format!("{}let m=iv(y ~ C(g)+log(x),~ C(g)+log(z),df,if=g>1)\ninput new\ng x\n3 2\n3 4\nend\npredict new fitted=m\n",categorical_data()));
    let model = view(&interpreter, "m");
    assert_eq!(model.variable_names, ["const", "C(g)_3", "log(x)"]);
    for (&x, &actual) in frame(&interpreter, "new")
        .get("x")
        .unwrap()
        .iter()
        .zip(frame(&interpreter, "new").get("fitted").unwrap().iter())
    {
        close(
            actual,
            model.params[0] + model.params[1] + model.params[2] * x.ln(),
        );
    }
}

#[test]
fn iv_typed_category_predictions_remap_labels_and_reject_unseen_levels() {
    let mut interpreter = execute(&categorical_data());
    let mut data = frame(&interpreter, "df").clone();
    data.insert_column(
        "g".into(),
        greeners::Column::from_strings(
            (0..24)
                .map(|i| ["B", "A", "C"][i % 3].to_string())
                .collect(),
        ),
    )
    .unwrap();
    interpreter
        .env
        .set("df", Value::DataFrame(Arc::new(data)))
        .unwrap();
    run_source(
        "let m=iv(y ~ C(g)+log(x),~ C(g)+log(z),df)\ninput new\nx\n2\n4\nend\n",
        &mut interpreter,
    )
    .unwrap();
    let model = view(&interpreter, "m");
    assert_eq!(model.variable_names, ["const", "g=A", "g=C", "log(x)"]);
    let mut new = frame(&interpreter, "new").clone();
    new.insert_column(
        "g".into(),
        greeners::Column::from_strings(vec!["C".into(), "A".into()]),
    )
    .unwrap();
    interpreter
        .env
        .set("new", Value::DataFrame(Arc::new(new)))
        .unwrap();
    run_source("predict new fitted=m\n", &mut interpreter).unwrap();
    let predicted = frame(&interpreter, "new").get("fitted").unwrap();
    close(
        predicted[0],
        model.params[0] + model.params[2] + model.params[3] * 2.0_f64.ln(),
    );
    close(
        predicted[1],
        model.params[0] + model.params[1] + model.params[3] * 4.0_f64.ln(),
    );
    let mut new = frame(&interpreter, "new").clone();
    new.insert_column(
        "g".into(),
        greeners::Column::from_strings(vec!["unknown".into(), "B".into()]),
    )
    .unwrap();
    interpreter
        .env
        .set("new", Value::DataFrame(Arc::new(new)))
        .unwrap();
    let error = run_source("predict new invalid=m\n", &mut interpreter)
        .expect_err("unseen label must fail");
    assert!(error.to_string().contains("unseen"), "{error}");
}

#[test]
fn iv_existing_result_readers_preserve_tidy_glance_dap_and_plugin_contracts() {
    let interpreter=execute(&format!("{DATA}let m=iv(y ~ x,~ z,df)\nlet coefficients=tidy(m)\nlet fit=glance(m)\nlet summary=m.summary()\nesttab(m)\ndiagnostics(m)\n"));
    let value = interpreter.env.get("m").unwrap();
    let model = view(&interpreter, "m");
    assert_eq!(model.variable_names, ["const", "x"]);
    assert_eq!(frame(&interpreter, "coefficients").n_rows(), 2);
    assert_eq!(frame(&interpreter, "fit").n_rows(), 1);
    assert!(interpreter
        .env
        .get("summary")
        .unwrap()
        .to_string()
        .contains("2SLS"));
    let children = model_expansion::value_children(value);
    assert!(children.iter().any(|(name, _)| name == "coefficients"));
    assert_eq!(model_expansion::value_summary(value).1, "IvResult");
    let json = value_to_json(value, false, &mut Vec::new());
    assert_eq!(json["__model_type__"], "iv");
    assert_eq!(json["variable"], serde_json::json!(["const", "x"]));
    close(json["coef"][1].as_f64().unwrap(), model.params[1]);
}

#[test]
fn dwh_insufficient_augmented_residual_degrees_of_freedom_fail_explicitly() {
    let mut interpreter =
        execute("input df\ny x1 x2 z1 z2\n1 1 1 1 2\n2 2 4 2 1\n3 3 2 3 4\n5 4 3 4 3\nend\n");
    let error = run_source("estat_endog(y ~ x1+x2,~ z1+z2,df)\n", &mut interpreter)
        .expect_err("DWH requires positive augmented residual df");
    assert!(error.to_string().contains("observations"), "{error}");
}

#[test]
fn iv_unrepresentable_design_moments_return_a_rescaling_error() {
    for scale in [1e160, 1e200, 1e-200] {
        let mut interpreter = execute(DATA);
        let mut data = frame(&interpreter, "df").clone();
        data.insert(
            "x".into(),
            frame(&interpreter, "df").get("x").unwrap() * scale,
        )
        .unwrap();
        interpreter
            .env
            .set("df", Value::DataFrame(Arc::new(data)))
            .unwrap();
        for estimator in ["iv", "estat_overid", "estat_endog"] {
            let error = run_source(
                &format!("{estimator}(y ~ x,~ log(z)+z,df)\n"),
                &mut interpreter,
            )
            .expect_err("unrepresentable normal-equation moments must fail before the backend");
            assert!(error.to_string().contains("rescale"), "{error}");
        }
    }
}

#[test]
fn classical_iv_diagnostics_reject_covariance_controls() {
    for diagnostic in ["estat_overid", "estat_endog"] {
        for option in ["cov=HC3", "cluster=x", "cluster2=x", "nw=1", "robust=true"] {
            let mut interpreter = execute(DATA);
            let error = run_source(
                &format!("{diagnostic}(y ~ x,~ z+x:x,df,{option})\n"),
                &mut interpreter,
            )
            .expect_err("classical diagnostic must not silently accept covariance controls");
            assert!(error.to_string().contains("classical"), "{error}");
        }
    }
}

#[test]
fn iv_prediction_retains_linear_aliases_and_empty_samples_fail() {
    let mut interpreter = execute(&format!(
        "{DATA}let m=iv(y ~ x,~ z,df)\ninput new\nx\n2\n4\nend\n"
    ));
    for alias in ["xb", "fitted", "linear", "yhat"] {
        run_source(
            &format!("predict new {alias}=m,\"{alias}\"\n"),
            &mut interpreter,
        )
        .unwrap();
    }
    for alias in ["fitted", "linear", "yhat"] {
        for (&actual, &expected) in frame(&interpreter, "new")
            .get(alias)
            .unwrap()
            .iter()
            .zip(frame(&interpreter, "new").get("xb").unwrap().iter())
        {
            close(actual, expected);
        }
    }
    for estimator in ["iv", "estat_overid", "estat_endog"] {
        let error = run_source(
            &format!("{estimator}(y ~ x,~ z+x:x,df,if=x>100)\n"),
            &mut interpreter,
        )
        .expect_err("empty filtered sample must fail");
        assert!(error.to_string().contains("no observations"), "{error}");
    }
}
