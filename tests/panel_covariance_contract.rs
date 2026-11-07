//! Panel inference rejects raw two-way covariance with a material negative variance.

#![cfg(feature = "greeners-panel")]

use hayashi_lang::lang::interpreter::{Interpreter, Value};
use hayashi_lang::run_source;
use std::collections::BTreeSet;

#[test]
fn fixed_effects_rejects_materially_negative_two_way_covariance() {
    // Synthetic integer fixture selected with NumPy seed 20261006 (candidate 8).
    // Runtime data and independent score-sandwich calculation are deterministic.
    let x = [2., 3., 3., 4., -4., -4., 1., -2., -3., -3., 1., -1.];
    let y = [1., -2., -3., 4., 4., 0., 3., -4., 4., -3., 2., -3.];
    let entity = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let time = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    assert!(x.iter().chain(y.iter()).all(|v: &f64| v.is_finite()));
    let entity_count = entity.iter().collect::<BTreeSet<_>>().len();
    let time_count = time.iter().collect::<BTreeSet<_>>().len();
    assert_eq!(entity_count, 4);
    assert_eq!(time_count, 3);
    let n = x.len();
    let k = 1;
    let within_df = n - k - (entity_count - 1);
    assert!(within_df > 0);

    let demean = |values: &[f64; 12]| {
        let mut transformed = [0.; 12];
        for group in 0..4 {
            let start = 3 * group;
            let mean = values[start..start + 3].iter().sum::<f64>() / 3.;
            for row in start..start + 3 {
                transformed[row] = values[row] - mean;
            }
        }
        transformed
    };
    let within_x = demean(&x);
    let within_y = demean(&y);
    let xtx = within_x.iter().map(|v| v * v).sum::<f64>();
    assert!((xtx - 60.).abs() < 1e-12); // One nonzero within column has full rank.
    let beta = within_x
        .iter()
        .zip(within_y)
        .map(|(x, y)| x * y)
        .sum::<f64>()
        / xtx;
    assert!((beta - 16. / 45.).abs() < 1e-12);
    let scores: Vec<_> = within_x
        .iter()
        .zip(within_y)
        .map(|(&x, y)| x * (y - x * beta))
        .collect();
    let grouped_meat = |labels: &[i32; 12]| {
        labels
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .iter()
            .map(|group| {
                labels
                    .iter()
                    .zip(&scores)
                    .filter(|(label, _)| *label == group)
                    .map(|(_, score)| score)
                    .sum::<f64>()
                    .powi(2)
            })
            .sum::<f64>()
    };
    let intersection: [i32; 12] = std::array::from_fn(|i| 3 * entity[i] + time[i]);
    assert_eq!(intersection.iter().collect::<BTreeSet<_>>().len(), 12);
    let g = entity_count.min(time_count) as f64;
    let correction = g / (g - 1.) * (n - 1) as f64 / (n - k) as f64;
    let raw_covariance =
        (grouped_meat(&entity) + grouped_meat(&time) - grouped_meat(&intersection)) / xtx.powi(2)
            * correction;
    assert!(raw_covariance.is_finite());
    assert!((raw_covariance - (-0.028555098308184736)).abs() < 1e-12);
    assert!(raw_covariance < -0.02); // Materially negative, not floating-point roundoff.

    let mut source = String::from("input df\ny x entity time\n");
    for i in 0..12 {
        source.push_str(&format!("{} {} {} {}\n", y[i], x[i], entity[i], time[i]));
    }
    source.push_str("end\nxtset(df, entity, time)\nlet baseline=fe(y ~ x, df)\n");
    let mut interpreter = Interpreter::new();
    interpreter.set_auto_display(false);
    run_source(&source, &mut interpreter).expect("valid full-rank FE baseline must fit");
    let Value::PanelResult(baseline) = interpreter.env.get("baseline").unwrap() else {
        panic!("expected FE result");
    };
    assert_eq!(baseline.n_obs, 12);
    assert_eq!(baseline.n_entities, 4);
    assert_eq!(baseline.df_resid, within_df);
    assert!((baseline.params[0] - beta).abs() < 1e-12);
    assert!(baseline.std_errors.iter().all(|v| v.is_finite() && *v > 0.));

    let error = run_source(
        "let unavailable=fe(y ~ x, df, cluster=entity, cluster2=time)\n",
        &mut interpreter,
    )
    .expect_err("materially negative two-way covariance must not provide fitted inference");
    let message = error.to_string().to_lowercase();
    assert!(message.contains("covariance"), "{message}");
    assert!(
        message.contains("negative") || message.contains("indefinite"),
        "{message}"
    );
}
