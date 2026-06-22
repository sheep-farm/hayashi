use plotters::prelude::*;

pub struct Plot;

impl Plot {
    pub fn scatter(
        x: &[f64], y: &[f64],
        xlabel: &str, ylabel: &str, title: &str,
        path: &str, width: u32, height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        root.fill(&WHITE)?;

        let (xmin, xmax) = min_max(x);
        let (ymin, ymax) = min_max(y);
        let xpad = (xmax - xmin).max(1e-10) * 0.05;
        let ypad = (ymax - ymin).max(1e-10) * 0.05;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (xmin - xpad)..(xmax + xpad),
                (ymin - ypad)..(ymax + ypad),
            )?;

        chart.configure_mesh().x_desc(xlabel).y_desc(ylabel).draw()?;

        chart.draw_series(
            x.iter().zip(y.iter())
                .filter(|(a, b)| a.is_finite() && b.is_finite())
                .map(|(&xi, &yi)| Circle::new((xi, yi), 3, BLUE.filled())),
        )?;

        root.present()?;
        Ok(())
    }

    pub fn line(
        x: &[f64], y: &[f64],
        xlabel: &str, ylabel: &str, title: &str,
        path: &str, width: u32, height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        root.fill(&WHITE)?;

        let (xmin, xmax) = min_max(x);
        let (ymin, ymax) = min_max(y);
        let xpad = (xmax - xmin).max(1e-10) * 0.05;
        let ypad = (ymax - ymin).max(1e-10) * 0.05;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (xmin - xpad)..(xmax + xpad),
                (ymin - ypad)..(ymax + ypad),
            )?;

        chart.configure_mesh().x_desc(xlabel).y_desc(ylabel).draw()?;

        let points: Vec<(f64, f64)> = x.iter().zip(y.iter())
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(&a, &b)| (a, b))
            .collect();

        chart.draw_series(LineSeries::new(points, BLUE.stroke_width(2)))?;

        root.present()?;
        Ok(())
    }

    pub fn histogram(
        vals: &[f64], xlabel: &str, title: &str,
        path: &str, width: u32, height: u32, n_bins: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let clean: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
        if clean.is_empty() { return Ok(()); }
        let (vmin, vmax) = min_max(&clean);
        let range = (vmax - vmin).max(1e-10);
        let bin_w = range / n_bins as f64;

        let mut counts = vec![0u32; n_bins];
        for &v in &clean {
            let idx = ((v - vmin) / bin_w).floor() as usize;
            counts[idx.min(n_bins - 1)] += 1;
        }
        let max_count = *counts.iter().max().unwrap_or(&1);

        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(vmin..(vmax + bin_w), 0u32..(max_count + 1))?;

        chart.configure_mesh().x_desc(xlabel).y_desc("Frequency").draw()?;

        chart.draw_series(
            counts.iter().enumerate().map(|(i, &c)| {
                let x0 = vmin + i as f64 * bin_w;
                let x1 = x0 + bin_w;
                Rectangle::new([(x0, 0), (x1, c)], BLUE.mix(0.7).filled())
            }),
        )?;

        root.present()?;
        Ok(())
    }

    pub fn coefplot(
        names: &[String], coefs: &[f64], ci_lo: &[f64], ci_hi: &[f64],
        title: &str, path: &str, width: u32, height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let k = coefs.len();
        if k == 0 { return Ok(()); }

        let all_lo = ci_lo.iter().copied().fold(f64::INFINITY, f64::min).min(0.0);
        let all_hi = ci_hi.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.0);
        let pad = (all_hi - all_lo).max(1e-10) * 0.1;

        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 18))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(120)
            .build_cartesian_2d((all_lo - pad)..(all_hi + pad), 0.0..(k as f64))?;

        chart.configure_mesh()
            .disable_y_mesh()
            .y_label_formatter(&|y| {
                let idx = *y as usize;
                if idx < k { names[k - 1 - idx].clone() } else { String::new() }
            })
            .x_desc("Coefficient")
            .draw()?;

        chart.draw_series(LineSeries::new(
            vec![(0.0, 0.0), (0.0, k as f64)], BLACK.stroke_width(1),
        ))?;

        for i in 0..k {
            let y = (k - 1 - i) as f64 + 0.5;
            chart.draw_series(LineSeries::new(
                vec![(ci_lo[i], y), (ci_hi[i], y)], BLUE.stroke_width(2),
            ))?;
            chart.draw_series(std::iter::once(
                Circle::new((coefs[i], y), 4, BLUE.filled()),
            ))?;
        }

        root.present()?;
        Ok(())
    }
}

fn min_max(v: &[f64]) -> (f64, f64) {
    let clean: Vec<f64> = v.iter().filter(|x| x.is_finite()).copied().collect();
    if clean.is_empty() { return (0.0, 1.0); }
    let mn = clean.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = clean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (mx - mn).abs() < 1e-15 { (mn - 1.0, mx + 1.0) } else { (mn, mx) }
}
