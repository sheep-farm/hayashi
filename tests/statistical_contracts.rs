//! Regression tests for formula, sample, design and reporting contracts.

use hayashi_lang::lang::interpreter::models::OlsModel;
use hayashi_lang::lang::interpreter::{Interpreter, Value};
use hayashi_lang::run_source;
use statrs::distribution::ContinuousCDF;
use std::rc::Rc;

const IV_DATA: &str = "input df\ny x z\n1.1 1.1 1\n1.7 2 1.2\n3.2 1.9 2.1\n2.8 3.2 1.7\n4.6 2.7 2.8\n4.1 4.1 2.3\n6.3 3.6 3.8\n5.9 5 3\nend\n";
const CATEGORICAL_DATA: &str =
    "input df\ny g x\n2 1 1\n4 1 2\n5 1 3\n10 2 1\n13 2 2\n14 2 3\n20 3 1\n21 3 2\n23 3 3\nend\n";
const CLUSTER_DATA: &str = "input df\nY X decimal integer\n10 2 1.1 1\n12 3 1.1 1\n8 1 1.2 2\n15 5 1.2 2\n11 2 2.1 3\n14 4 2.1 3\n9 1 2.2 4\n13 4 2.2 4\nend\n";

fn execute(source: &str) -> Interpreter {
    let mut interpreter = Interpreter::new();
    interpreter.set_auto_display(false);
    run_source(source, &mut interpreter).expect("regression script must succeed");
    interpreter
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "expected {expected}, got {actual}"
    );
}

fn numeric(value: &Value) -> f64 {
    match value {
        Value::Float(value) => *value,
        Value::Int(value) => *value as f64,
        other => panic!("expected number, got {other}"),
    }
}

fn model(interpreter: &Interpreter, name: &str) -> Rc<dyn hayashi_lang::lang::interpreter::Model> {
    match interpreter.env.get(name).expect("model variable exists") {
        Value::Model(model) => model.clone(),
        other => panic!("expected model, got {other}"),
    }
}

#[test]
fn iv_transformed_instruments_match_explicit_columns_and_names() {
    for suffix in ["", " - 1"] {
        let interpreter = execute(&format!(
            "{IV_DATA}let m=iv(y ~ log(x){suffix}, ~ log(z){suffix}, df, cov=robust)\nlet expanded=mutate(df, lx=log(x), lz=log(z))\nlet reference=iv(y ~ lx{suffix}, ~ lz{suffix}, expanded, cov=robust)\n"
        ));
        let Value::IvResult(actual) = interpreter.env.get("m").unwrap() else {
            panic!("expected IV result");
        };
        let Value::IvResult(expected) = interpreter.env.get("reference").unwrap() else {
            panic!("expected reference IV result");
        };
        assert_eq!(actual.result.params.len(), expected.result.params.len());
        for (&a, &b) in actual
            .result
            .params
            .iter()
            .zip(expected.result.params.iter())
        {
            close(a, b);
        }
        for (&a, &b) in actual
            .result
            .std_errors
            .iter()
            .zip(expected.result.std_errors.iter())
        {
            close(a, b);
        }
        assert_eq!(
            actual
                .result
                .variable_names
                .as_ref()
                .unwrap()
                .last()
                .unwrap(),
            "log(x)"
        );
    }
}

#[test]
fn iv_filter_and_cluster_labels_share_the_explicitly_filtered_sample() {
    let interpreter = execute("input df\ny x z group firm\n1 1 2 1 1\n2 2 1 1 1\n4 3 4 1 2\n4 4 3 1 2\n5 5 6 1 3\n6 6 5 1 3\n100 7 8 2 4\n200 8 7 2 4\nend\nlet m=iv(y ~ x, ~ z, df, if=group == 1, cluster=firm)\nlet kept=filter(df, group == 1)\nlet reference=iv(y ~ x, ~ z, kept, cluster=firm)\n");
    let Value::IvResult(actual) = interpreter.env.get("m").unwrap() else {
        panic!("IV result");
    };
    let Value::IvResult(expected) = interpreter.env.get("reference").unwrap() else {
        panic!("IV reference");
    };
    assert_eq!(actual.result.n_obs, 6);
    for (&a, &b) in actual
        .result
        .params
        .iter()
        .zip(expected.result.params.iter())
    {
        close(a, b);
    }
    for (&a, &b) in actual
        .result
        .std_errors
        .iter()
        .zip(expected.result.std_errors.iter())
    {
        close(a, b);
    }
}

#[test]
fn categorical_names_and_named_restrictions_target_the_actual_slope() {
    let interpreter = execute(&format!(
        "{CATEGORICAL_DATA}let m=ols(y ~ C(g) + x, df)\nlet joint=testparm(m, [\"x\"])\n"
    ));
    let view = model(&interpreter, "m").to_model_view();
    assert_eq!(view.variable_names, ["_cons", "C(g)_2", "C(g)_3", "x"]);
    close(*view.params.last().unwrap(), 5.0 / 3.0);
    let Value::Dict(joint) = interpreter.env.get("joint").unwrap() else {
        panic!("test result");
    };
    close(numeric(joint.get("f_stat").unwrap()), 62.5);
}

#[test]
fn expanded_categorical_names_preserve_transforms_and_collinearity_mapping() {
    let interpreter = execute(&format!("{CATEGORICAL_DATA}let m=ols(y ~ C(g) + log(x), df)\nlet expanded=mutate(df, lx=log(x), x2=2*x)\nlet reference=ols(y ~ C(g) + lx, expanded)\nlet redundant=ols(y ~ C(g) + x + x2, expanded)\n"));
    let actual = model(&interpreter, "m").to_model_view();
    let reference = model(&interpreter, "reference").to_model_view();
    assert_eq!(actual.variable_names.last().unwrap(), "log(x)");
    assert_eq!(actual.variable_names.len(), actual.params.len());
    for (&a, &b) in actual.params.iter().zip(reference.params.iter()) {
        close(a, b);
    }
    let retained = model(&interpreter, "redundant").to_model_view();
    assert_eq!(retained.variable_names.len(), retained.params.len());
}

#[test]
fn cluster_partition_is_invariant_to_float_integer_and_string_labels() {
    let mut interpreter = execute(CLUSTER_DATA);
    let Value::DataFrame(original) = interpreter.env.get("df").unwrap() else {
        panic!("dataframe");
    };
    let mut frame = original.as_ref().clone();
    frame
        .insert_column(
            "label".into(),
            greeners::Column::from_strings(
                ["a", "a", "b", "b", "c", "c", "d", "d"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
        )
        .unwrap();
    interpreter
        .env
        .set("df", Value::DataFrame(std::sync::Arc::new(frame)))
        .unwrap();
    run_source("let a=ols(Y ~ X, df, cluster=decimal)\nlet b=ols(Y ~ X, df, cluster=integer)\nlet c=ols(Y ~ X, df, cluster=label)\n", &mut interpreter).unwrap();
    let expected = model(&interpreter, "b").to_model_view();
    // Independent statsmodels clustered fixture with its small-sample correction.
    close(expected.std_errors[1], 0.12278435237854593);
    for name in ["a", "c"] {
        let actual = model(&interpreter, name).to_model_view();
        for (&a, &b) in actual.std_errors.iter().zip(expected.std_errors.iter()) {
            close(a, b);
        }
    }
}

#[test]
fn cluster_signed_zero_is_one_label_and_nonfinite_labels_are_rejected() {
    let interpreter = execute("input df\ny x g canonical\n1 1 -0.0 0\n3 2 0.0 0\n4 3 -0.0 0\n6 4 1.0 1\n7 5 1.0 1\n9 6 1.0 1\nend\nlet a=ols(y ~ x, df, cluster=g)\nlet b=ols(y ~ x, df, cluster=canonical)\n");
    let a = model(&interpreter, "a").to_model_view();
    let b = model(&interpreter, "b").to_model_view();
    for (&a, &b) in a.std_errors.iter().zip(b.std_errors.iter()) {
        close(a, b);
    }
    for replacement in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut interpreter =
            execute("input df\ny x g\n1 1 1\n3 2 1\n4 3 2\n6 4 2\n7 5 3\n9 6 3\nend\n");
        let Value::DataFrame(original) = interpreter.env.get("df").unwrap() else {
            panic!("dataframe");
        };
        let mut frame = original.as_ref().clone();
        frame
            .insert("g".into(), ndarray::array![1., replacement, 2., 2., 3., 3.])
            .unwrap();
        interpreter
            .env
            .set("df", Value::DataFrame(std::sync::Arc::new(frame)))
            .unwrap();
        let error = run_source("let m=ols(y ~ x, df, cluster=g)\n", &mut interpreter)
            .expect_err("nonfinite cluster label must fail");
        assert!(error.to_string().contains("cluster"), "{error}");
    }
}

#[test]
fn collinear_wls_preserves_original_scale_predictions_and_residuals() {
    let interpreter = execute("input df\ny x x2 w\n1 1 2 1\n3 2 4 2\n4 3 6 3\n6 4 8 2\n7 5 10 1\nend\nlet m=wls(y ~ x + x2, df, weights=\"w\")\nlet reference=wls(y ~ x2, df, weights=\"w\")\n");
    let actual = model(&interpreter, "m");
    let reference = model(&interpreter, "reference");
    for (&a, &b) in actual
        .fitted_values()
        .unwrap()
        .iter()
        .zip(reference.fitted_values().unwrap().iter())
    {
        close(a, b);
    }
    for (&a, &b) in actual
        .residuals()
        .unwrap()
        .iter()
        .zip(reference.residuals().unwrap().iter())
    {
        close(a, b);
    }
    let actual = actual.to_model_view();
    let reference = reference.to_model_view();
    for (&a, &b) in actual.std_errors.iter().zip(reference.std_errors.iter()) {
        close(a, b);
    }
}

fn coefficient_frame<'a>(interpreter: &'a Interpreter, name: &str) -> &'a greeners::DataFrame {
    let Value::ModelResult { fields, .. } = interpreter.env.get(name).unwrap() else {
        panic!("model result");
    };
    let Value::DataFrame(frame) = fields.get("coefficients").unwrap() else {
        panic!("coefficient frame");
    };
    frame
}

#[test]
fn bayesian_wrapper_inserts_one_intercept_and_names_expanded_predictors() {
    for rhs in ["x", "C(g) + log(x)"] {
        let interpreter = execute(&format!(
            "{CATEGORICAL_DATA}let m=bayes_lm(y ~ {rhs}, df)\n"
        ));
        let frame = coefficient_frame(&interpreter, "m");
        let names = frame.get_string("variable").unwrap();
        assert_eq!(names.first().unwrap(), "Intercept");
        if rhs == "x" {
            assert_eq!(names, ["Intercept", "x"]);
        } else {
            assert_eq!(names, ["Intercept", "C(g)_2", "C(g)_3", "log(x)"]);
        }
        assert_eq!(frame.n_rows(), names.len());
        let display = interpreter.env.get("m").unwrap().to_string();
        // Each parameter appears once in the coefficient table and once in
        // the separate sign-probability table, with the same expanded name.
        for name in &names {
            assert_eq!(
                display
                    .lines()
                    .filter(|line| line.split_whitespace().next() == Some(name.as_str()))
                    .count(),
                2,
                "rendered coefficient name mismatch: {display}"
            );
        }
    }
}

#[test]
fn normal_testparm_reports_scaled_wald_with_chi_square_reference() {
    let x = ndarray::array![[1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.]];
    let y = ndarray::array![1., 3., 2., 3., 2.];
    let result = greeners::OLS::fit_with_names(
        &y,
        &x,
        greeners::CovarianceType::NonRobust,
        Some(vec!["_cons".into(), "x".into()]),
    )
    .unwrap()
    .with_inference(greeners::InferenceType::Normal)
    .unwrap();
    let fitted = x.dot(&result.params);
    let expected = (result.params[1] / result.std_errors[1]).powi(2);
    let mut interpreter = Interpreter::new();
    interpreter.set_auto_display(false);
    interpreter
        .env
        .declare(
            "m",
            Value::Model(Rc::new(OlsModel {
                result: Rc::new(result),
                residuals: &y - &fitted,
                x,
            })),
        )
        .unwrap();
    run_source("let joint=testparm(m, [\"x\"])\n", &mut interpreter).unwrap();
    let Value::Dict(joint) = interpreter.env.get("joint").unwrap() else {
        panic!("test result");
    };
    assert_eq!(
        joint.get("reference_distribution").unwrap().to_string(),
        "chi2"
    );
    assert!(joint
        .get("test")
        .unwrap()
        .to_string()
        .contains("Scaled Wald"));
    close(numeric(joint.get("f_stat").unwrap()), expected);
    close(
        numeric(joint.get("p_value").unwrap()),
        statrs::distribution::ChiSquared::new(1.0)
            .unwrap()
            .sf(expected),
    );
    assert!(!joint.contains_key("df2"));
}

#[test]
#[cfg(feature = "experimental")]
fn sfa_posterior_table_omits_unavailable_sign_probability() {
    let interpreter = execute("input df\ny k l\n10 5 5\n12 6 6\n15 8 7\n18 10 8\n20 12 10\n22 13 11\n25 15 12\n28 17 13\n30 18 14\n32 20 15\nend\nlet m=bayes_sfa_production(y ~ k + l, df, burn=20, draws=50)\n");
    let frame = coefficient_frame(&interpreter, "m");
    assert!(!frame.column_names().iter().any(|name| name == "p_positive"));
    for name in ["variable", "mean", "sd", "ci_low", "ci_high"] {
        assert!(frame.get_column(name).is_ok(), "missing {name}");
    }
}

#[test]
fn zero_linear_contrast_retains_positive_uncertainty() {
    let interpreter = execute("input df\ny x\n1 -2\n-2 -1\n2 0\n-2 1\n1 2\nend\nlet m=ols(y ~ x, df)\nlet combination=lincom(m, x=1)\n");
    let Value::Dict(combination) = interpreter.env.get("combination").unwrap() else {
        panic!("linear contrast result");
    };
    let expected_se = (14.0_f64 / 30.0).sqrt();
    close(numeric(combination.get("estimate").unwrap()), 0.0);
    close(numeric(combination.get("std_err").unwrap()), expected_se);
    let critical = statrs::distribution::StudentsT::new(0.0, 1.0, 3.0)
        .unwrap()
        .inverse_cdf(0.975);
    close(
        numeric(combination.get("ci_lower").unwrap()),
        -critical * expected_se,
    );
    close(
        numeric(combination.get("ci_upper").unwrap()),
        critical * expected_se,
    );
    close(numeric(combination.get("p_value").unwrap()), 1.0);
    close(numeric(combination.get("t").unwrap()), 0.0);
    close(numeric(combination.get("df").unwrap()), 3.0);
    assert_eq!(
        combination
            .get("reference_distribution")
            .unwrap()
            .to_string(),
        "student_t"
    );
}

#[test]
fn normal_linear_contrast_uses_normal_interval_and_reports_z() {
    let x = ndarray::array![[1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.]];
    let y = ndarray::array![1., 3., 2., 3., 2.];
    let result = greeners::OLS::fit_with_names(
        &y,
        &x,
        greeners::CovarianceType::NonRobust,
        Some(vec!["_cons".into(), "x".into()]),
    )
    .unwrap()
    .with_inference(greeners::InferenceType::Normal)
    .unwrap();
    let estimate = result.params[1];
    let expected_se = result.std_errors[1];
    let fitted = x.dot(&result.params);
    let mut interpreter = Interpreter::new();
    interpreter.set_auto_display(false);
    interpreter
        .env
        .declare(
            "m",
            Value::Model(Rc::new(OlsModel {
                result: Rc::new(result),
                residuals: &y - &fitted,
                x,
            })),
        )
        .unwrap();
    run_source("let combination=lincom(m, x=1)\n", &mut interpreter).unwrap();
    let Value::Dict(combination) = interpreter.env.get("combination").unwrap() else {
        panic!("linear contrast result");
    };
    let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let margin = normal.inverse_cdf(0.975) * expected_se;
    close(
        numeric(combination.get("ci_lower").unwrap()),
        estimate - margin,
    );
    close(
        numeric(combination.get("ci_upper").unwrap()),
        estimate + margin,
    );
    close(
        numeric(combination.get("z").unwrap()),
        estimate / expected_se,
    );
    close(
        numeric(combination.get("p_value").unwrap()),
        2.0 * normal.sf((estimate / expected_se).abs()),
    );
    assert_eq!(
        combination
            .get("reference_distribution")
            .unwrap()
            .to_string(),
        "normal"
    );
    assert!(!combination.contains_key("df"));
    assert!(!combination.contains_key("t"));
}

#[test]
fn nonlinear_identity_matches_linear_inference_for_each_fitted_covariance() {
    for fit in [
        "ols(y ~ x, df, cov=HC3)",
        "ols(y ~ x, df, cluster=group)",
        "ols(y ~ x, df, nw=2)",
        "wls(y ~ x, df, weights=\"w\")",
    ] {
        let interpreter = execute(&format!("input df\ny x group w\n1 1 1 1\n3 2 1 2\n2 3 2 1\n5 4 2 4\n3 5 3 1\n9 6 3 8\n6 7 4 2\n15 8 4 10\nend\nlet m={fit}\nlet linear=lincom(m,x=1)\nlet nonlinear=nlcom(m,x)\n"));
        let Value::Dict(linear) = interpreter.env.get("linear").unwrap() else {
            panic!("linear inference result");
        };
        let Value::Dict(nonlinear) = interpreter.env.get("nonlinear").unwrap() else {
            panic!("nonlinear inference must expose its uncertainty");
        };
        for field in [
            "estimate", "std_err", "t", "p_value", "ci_lower", "ci_upper", "df",
        ] {
            close(
                numeric(nonlinear.get(field).unwrap()),
                numeric(linear.get(field).unwrap()),
            );
        }
        assert_eq!(
            nonlinear.get("reference_distribution").unwrap().to_string(),
            "student_t"
        );
        assert!(interpreter.env.get("x").is_none());
        assert!(interpreter.env.get("_cons").is_none());
    }
}

#[test]
fn nonlinear_zero_estimate_retains_uncertainty_for_both_distributions() {
    for inference in [
        greeners::InferenceType::StudentT,
        greeners::InferenceType::Normal,
    ] {
        let student_t = matches!(inference, greeners::InferenceType::StudentT);
        let x = ndarray::array![[1., -2.], [1., -1.], [1., 0.], [1., 1.], [1., 2.]];
        let y = ndarray::array![1., -2., 2., -2., 1.];
        let result = greeners::OLS::fit_with_names(
            &y,
            &x,
            greeners::CovarianceType::NonRobust,
            Some(vec!["_cons".into(), "x".into()]),
        )
        .unwrap()
        .with_inference(inference)
        .unwrap();
        let mut interpreter = Interpreter::new();
        interpreter.set_auto_display(false);
        let fitted = x.dot(&result.params);
        interpreter
            .env
            .declare(
                "m",
                Value::Model(Rc::new(OlsModel {
                    result: Rc::new(result),
                    residuals: &y - &fitted,
                    x,
                })),
            )
            .unwrap();
        run_source(
            "let linear=lincom(m,x=1)\nlet nonlinear=nlcom(m,x)\nlet offset=nlcom(m,x+2)\n",
            &mut interpreter,
        )
        .unwrap();
        let Value::Dict(linear) = interpreter.env.get("linear").unwrap() else {
            panic!("linear result")
        };
        let Value::Dict(nonlinear) = interpreter.env.get("nonlinear").unwrap() else {
            panic!("nonlinear result")
        };
        for field in ["estimate", "std_err", "p_value", "ci_lower", "ci_upper"] {
            close(
                numeric(nonlinear.get(field).unwrap()),
                numeric(linear.get(field).unwrap()),
            );
        }
        close(numeric(nonlinear.get("p_value").unwrap()), 1.0);
        assert_eq!(nonlinear.contains_key("df"), student_t);
        assert_eq!(
            nonlinear.get("reference_distribution").unwrap().to_string(),
            linear.get("reference_distribution").unwrap().to_string()
        );
        let Value::Dict(offset) = interpreter.env.get("offset").unwrap() else {
            panic!("offset result");
        };
        let statistic = 2.0 / (14.0_f64 / 30.0).sqrt();
        let expected_p = if student_t {
            2.0 * statrs::distribution::StudentsT::new(0.0, 1.0, 3.0)
                .unwrap()
                .sf(statistic)
        } else {
            2.0 * statrs::distribution::Normal::new(0.0, 1.0)
                .unwrap()
                .sf(statistic)
        };
        close(numeric(offset.get("p_value").unwrap()), expected_p);
        close(
            numeric(offset.get(if student_t { "t" } else { "z" }).unwrap()),
            statistic,
        );
    }
}

#[test]
fn nonlinear_small_positive_uncertainty_and_units_are_preserved() {
    let mut probabilities = Vec::new();
    for scale in [1.0_f64, 1e5] {
        let amplitude = 1.5e-6 * 19.0_f64.sqrt();
        let rows = (0..20)
            .map(|i| {
                format!(
                    "{}\n",
                    scale * (1e-5 + if i % 2 == 0 { amplitude } else { -amplitude })
                )
            })
            .collect::<String>();
        let interpreter = execute(&format!("input df\ny\n{rows}end\nlet m=ols(y ~ 1, df, cov=HC1)\nlet nonlinear=nlcom(m,_cons*_cons*_cons)\n"));
        let Value::Dict(result) = interpreter.env.get("nonlinear").unwrap() else {
            panic!("nonlinear result")
        };
        let expected_se = 4.5e-16 * scale.powi(3);
        let se = numeric(result.get("std_err").unwrap());
        assert!(
            (se / expected_se - 1.0).abs() < 1e-5,
            "SE {se} versus {expected_se}"
        );
        let p = numeric(result.get("p_value").unwrap());
        close(p, 0.03860856171287706);
        probabilities.push(p);
    }
    close(probabilities[0], probabilities[1]);
}

#[test]
fn nonlinear_expression_errors_and_domain_failures_restore_environment() {
    let mut interpreter = execute("input df\ny x\n1 -2\n-2 -1\n2 0\n-2 1\n1 2\nend\nlet m=ols(y ~ x,df)\nlet x=999\nlet _cons=777\n");
    let scopes = interpreter.env.scope_count();
    run_source(
        "fn incomplete(a) { return a }\nfn captured() { return x }\n",
        &mut interpreter,
    )
    .unwrap();
    for expression in [
        "unknown",
        "[1,2]",
        "sqrt(x)",
        "log(x)",
        "abs(x)",
        "incomplete()",
        "captured()",
    ] {
        assert!(
            run_source(&format!("nlcom(m,{expression})\n"), &mut interpreter).is_err(),
            "{expression}"
        );
        close(numeric(interpreter.env.get("x").unwrap()), 999.0);
        close(numeric(interpreter.env.get("_cons").unwrap()), 777.0);
        assert_eq!(interpreter.env.scope_count(), scopes);
    }
    run_source("nlcom(m,x)\n", &mut interpreter).unwrap();
    close(numeric(interpreter.env.get("x").unwrap()), 999.0);
    close(numeric(interpreter.env.get("_cons").unwrap()), 777.0);
    assert_eq!(interpreter.env.scope_count(), scopes);
    assert!(run_source("nlcom(m,1e308*x)\n", &mut interpreter).is_err());
    close(numeric(interpreter.env.get("x").unwrap()), 999.0);
    assert_eq!(interpreter.env.scope_count(), scopes);

    // Binding x fails after _cons has been shadowed in the temporary scope.
    interpreter.env.declare_const("x", Value::Int(123));
    assert!(run_source("nlcom(m,x)\n", &mut interpreter).is_err());
    close(numeric(interpreter.env.get("x").unwrap()), 123.0);
    close(numeric(interpreter.env.get("_cons").unwrap()), 777.0);
    assert_eq!(interpreter.env.scope_count(), scopes);
    assert!(interpreter.env.set("x", Value::Int(456)).is_err());
}

#[test]
fn nonlinear_rounded_flat_expression_reports_unresolved_uncertainty() {
    let mut interpreter = execute(
        "input df\ny x\n1 -2\n-2 -1\n2 0\n-2 1\n1 2\nend\nlet m=ols(y ~ x,df)\nlet x=999\n",
    );
    let error = run_source("nlcom(m,1e16+x)\n", &mut interpreter)
        .expect_err("rounded finite differences must not fabricate zero uncertainty");
    assert!(error.to_string().contains("precision"), "{error}");
    close(numeric(interpreter.env.get("x").unwrap()), 999.0);
}

#[test]
fn nonlinear_offset_and_exact_zero_variance_boundaries_use_the_expression_null() {
    let interpreter = execute("input df\ny x\n1 -2\n-2 -1\n2 0\n-2 1\n1 2\nend\nlet m=ols(y ~ x,df)\nlet offset=nlcom(m,x+2)\nlet zero=nlcom(m,0)\nlet nonzero=nlcom(m,2)\nlet stationary=nlcom(m,x*x)\nlet exponential=nlcom(m,exp(_cons))\n");
    let Value::Dict(offset) = interpreter.env.get("offset").unwrap() else {
        panic!("offset result")
    };
    let se = (14.0_f64 / 30.0).sqrt();
    close(numeric(offset.get("estimate").unwrap()), 2.0);
    close(numeric(offset.get("std_err").unwrap()), se);
    close(numeric(offset.get("t").unwrap()), 2.0 / se);
    for name in ["zero", "stationary"] {
        let Value::Dict(result) = interpreter.env.get(name).unwrap() else {
            panic!("zero variance result")
        };
        close(numeric(result.get("std_err").unwrap()), 0.0);
        close(numeric(result.get("t").unwrap()), 0.0);
        close(numeric(result.get("p_value").unwrap()), 1.0);
    }
    let Value::Dict(nonzero) = interpreter.env.get("nonzero").unwrap() else {
        panic!("nonzero result")
    };
    close(numeric(nonzero.get("std_err").unwrap()), 0.0);
    assert_eq!(numeric(nonzero.get("t").unwrap()), f64::INFINITY);
    close(numeric(nonzero.get("p_value").unwrap()), 0.0);
    close(numeric(nonzero.get("ci_lower").unwrap()), 2.0);
    close(numeric(nonzero.get("ci_upper").unwrap()), 2.0);
    let Value::Dict(exponential) = interpreter.env.get("exponential").unwrap() else {
        panic!("exponential result");
    };
    close(numeric(exponential.get("estimate").unwrap()), 1.0);
    close(
        numeric(exponential.get("std_err").unwrap()),
        (14.0_f64 / 15.0).sqrt(),
    );
}

#[test]
#[cfg(feature = "experimental")]
fn sfa_categorical_transformed_names_match_production_and_cost_designs() {
    for estimator in ["bayes_sfa_production", "bayes_sfa_cost"] {
        let interpreter = execute(&format!(
            "{CATEGORICAL_DATA}let m={estimator}(y ~ C(g)+log(x),df,burn=20,draws=50)\n"
        ));
        let frame = coefficient_frame(&interpreter, "m");
        assert_eq!(
            frame.get_string("variable").unwrap(),
            ["const", "C(g)_2", "C(g)_3", "log(x)"]
        );
        assert_eq!(frame.n_rows(), 4);
        assert!(!frame.column_names().iter().any(|name| name == "p_positive"));
        let display = interpreter.env.get("m").unwrap().to_string();
        for name in ["const", "C(g)_2", "C(g)_3", "log(x)"] {
            assert_eq!(
                display
                    .lines()
                    .filter(|line| line.split_whitespace().next() == Some(name))
                    .count(),
                1,
                "{display}"
            );
        }
    }
}

#[test]
fn normal_joint_test_preserves_representable_survival_tail() {
    let x = ndarray::array![
        [1., 1.],
        [1., 2.],
        [1., 3.],
        [1., 4.],
        [1., 5.],
        [1., 6.],
        [1., 7.],
        [1., 8.]
    ];
    let y = ndarray::array![1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5];
    let result = greeners::OLS::fit_with_names(
        &y,
        &x,
        greeners::CovarianceType::NonRobust,
        Some(vec!["_cons".into(), "x".into()]),
    )
    .unwrap()
    .with_inference(greeners::InferenceType::Normal)
    .unwrap();
    let fitted = x.dot(&result.params);
    let mut interpreter = Interpreter::new();
    interpreter.set_auto_display(false);
    interpreter
        .env
        .declare(
            "m",
            Value::Model(Rc::new(OlsModel {
                result: Rc::new(result),
                residuals: &y - &fitted,
                x,
            })),
        )
        .unwrap();
    run_source("let joint=testparm(m,[\"x\"])\n", &mut interpreter).unwrap();
    let Value::Dict(joint) = interpreter.env.get("joint").unwrap() else {
        panic!("joint result")
    };
    close(numeric(joint.get("f_stat").unwrap()), 120.0);
    let p = numeric(joint.get("p_value").unwrap());
    assert!(p > 0.0);
    assert!((p / 6.326068263677272e-28 - 1.0).abs() < 1e-9, "p={p}");
}
