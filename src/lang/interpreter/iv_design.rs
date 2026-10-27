//! IV structural/instrument preparation and retained prediction design.

use super::*;
use greeners::Column;
use std::collections::BTreeSet;

pub(super) struct PreparedIv {
    pub structural_frame: Arc<DataFrame>,
    pub y: Array1<f64>,
    pub x: Array2<f64>,
    pub z: Array2<f64>,
    /// Expanded instrument term identities, excluding its actual intercept.
    pub instrument_terms: Vec<String>,
    pub design: IvPredictionDesign,
}

pub(super) struct IvPredictionDesign {
    pub formula: Formula,
    pub names: Vec<String>,
    terms: Vec<IvTermSchema>,
    retained_columns: Vec<usize>,
}

enum IvTermSchema {
    Scalar,
    NumericCategory(Vec<i32>),
    LabelCategory(Vec<String>),
}

impl IvTermSchema {
    fn width(&self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::NumericCategory(levels) => levels.len() - 1,
            Self::LabelCategory(levels) => levels.len() - 1,
        }
    }
}

impl IvPredictionDesign {
    fn from_frame(
        formula: Formula,
        materialised: &GFormula,
        frame: &DataFrame,
        display_names: &[String],
    ) -> Result<Self> {
        if materialised.independents.len() != display_names.len() {
            return Err(HayashiError::Runtime(
                "IV formula term names disagree".into(),
            ));
        }
        let mut names = Vec::new();
        let mut terms = Vec::new();
        if materialised.intercept {
            names.push("const".into());
        }
        for (column, display) in materialised.independents.iter().zip(display_names) {
            if let Some(variable) = category_column(column) {
                let values = frame
                    .get_column(variable)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let schema = if let Column::Categorical(category) = values {
                    // Filtering retains declared levels; the backend design uses
                    // observed codes and drops the lowest observed code.
                    let levels = category
                        .codes
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .map(|code| {
                            category.get_level(code).map(str::to_string).ok_or_else(|| {
                                HayashiError::Runtime(format!(
                                    "IV: invalid category code in '{variable}'"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    names.extend(
                        levels
                            .iter()
                            .skip(1)
                            .map(|level| format!("{variable}={level}")),
                    );
                    IvTermSchema::LabelCategory(levels)
                } else {
                    let values = values.to_float();
                    ensure_finite(&values, variable)?;
                    // Retain the backend's finite numeric categorical encoding.
                    let levels = values
                        .iter()
                        .map(|value| value.round() as i32)
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>();
                    names.extend(
                        levels
                            .iter()
                            .skip(1)
                            .map(|level| format!("{display}_{level}")),
                    );
                    IvTermSchema::NumericCategory(levels)
                };
                let level_count = match &schema {
                    IvTermSchema::NumericCategory(levels) => levels.len(),
                    IvTermSchema::LabelCategory(levels) => levels.len(),
                    IvTermSchema::Scalar => {
                        return Err(HayashiError::Runtime(
                            "IV category schema is invalid".into(),
                        ))
                    }
                };
                if level_count < 2 {
                    return Err(HayashiError::Runtime(format!(
                        "IV categorical term '{display}' requires at least two observed levels"
                    )));
                }
                terms.push(schema);
            } else {
                names.push(display.clone());
                terms.push(IvTermSchema::Scalar);
            }
        }
        let retained_columns = (0..names.len()).collect();
        Ok(Self {
            formula,
            names,
            terms,
            retained_columns,
        })
    }

    /// Validate the backend's retained original positions before storing them.
    pub(super) fn retain_fitted_columns(&mut self, result: &greeners::iv::IvResult) -> Result<()> {
        if result
            .omitted_vars
            .iter()
            .any(|(position, _)| *position >= self.names.len())
        {
            return Err(HayashiError::Runtime(
                "IV omitted column positions disagree with its design".into(),
            ));
        }
        let retained = (0..self.names.len())
            .filter(|index| {
                !result
                    .omitted_vars
                    .iter()
                    .any(|(position, _)| position == index)
            })
            .collect::<Vec<_>>();
        let names = retained
            .iter()
            .map(|index| {
                self.names.get(*index).cloned().ok_or_else(|| {
                    HayashiError::Runtime("IV retained column is outside its design".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if retained.len() != result.params.len() || result.variable_names.as_ref() != Some(&names) {
            return Err(HayashiError::Runtime(
                "IV fitted coefficient order disagrees with its design".into(),
            ));
        }
        self.retained_columns = retained;
        Ok(())
    }

    fn matrix(&self, frame: &DataFrame, materialised: &GFormula) -> Result<Array2<f64>> {
        if self.terms.len() != materialised.independents.len()
            || self.formula.intercept != materialised.intercept
        {
            return Err(HayashiError::Runtime(
                "IV prediction formula disagrees with its fitted design".into(),
            ));
        }
        let mut columns = Vec::new();
        if materialised.intercept {
            columns.push(Array1::ones(frame.n_rows()));
        }
        for (column, schema) in materialised.independents.iter().zip(&self.terms) {
            match schema {
                IvTermSchema::Scalar => {
                    let values = frame
                        .get_column(column)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                        .to_float();
                    ensure_finite(&values, column)?;
                    columns.push(values);
                }
                IvTermSchema::NumericCategory(levels) => {
                    let variable = category_column(column).ok_or_else(|| {
                        HayashiError::Runtime(
                            "IV numeric category term disagrees with its design".into(),
                        )
                    })?;
                    let column = frame
                        .get_column(variable)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if matches!(column, Column::Categorical(_) | Column::String(_)) {
                        return Err(HayashiError::Runtime(format!(
                            "IV category '{variable}' requires numeric codes"
                        )));
                    }
                    let values = column.to_float();
                    ensure_finite(&values, variable)?;
                    let codes = values
                        .iter()
                        .map(|value| value.round() as i32)
                        .collect::<Vec<_>>();
                    if codes.iter().any(|code| !levels.contains(code)) {
                        return Err(HayashiError::Runtime(format!(
                            "IV prediction contains an unseen level of '{variable}'"
                        )));
                    }
                    for level in levels.iter().skip(1) {
                        columns.push(Array1::from_iter(
                            codes.iter().map(|code| f64::from(code == level)),
                        ));
                    }
                }
                IvTermSchema::LabelCategory(levels) => {
                    let variable = category_column(column).ok_or_else(|| {
                        HayashiError::Runtime(
                            "IV labelled category term disagrees with its design".into(),
                        )
                    })?;
                    let labels = frame
                        .get_string(variable)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    if labels.iter().any(|label| !levels.contains(label)) {
                        return Err(HayashiError::Runtime(format!(
                            "IV prediction contains an unseen level of '{variable}'"
                        )));
                    }
                    for level in levels.iter().skip(1) {
                        columns.push(Array1::from_iter(
                            labels.iter().map(|label| f64::from(label == level)),
                        ));
                    }
                }
            }
        }
        let width = usize::from(materialised.intercept)
            + self.terms.iter().map(IvTermSchema::width).sum::<usize>();
        if width != self.names.len() || columns.len() != width {
            return Err(HayashiError::Runtime(
                "IV expanded column names disagree with its matrix".into(),
            ));
        }
        let mut matrix = Array2::zeros((frame.n_rows(), width));
        for (mut destination, column) in matrix.columns_mut().into_iter().zip(columns) {
            destination.assign(&column);
        }
        Ok(matrix)
    }
}

fn category_column(column: &str) -> Option<&str> {
    column.strip_prefix("C(")?.strip_suffix(')').map(str::trim)
}

fn ensure_finite(values: &Array1<f64>, column: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(HayashiError::Runtime(format!(
            "IV column '{column}' contains non-finite values"
        )));
    }
    Ok(())
}

/// Reject unrepresentable moments before the unchanged normal-equation backend.
/// This is a range check, not a condition-number policy or column normaliser.
fn ensure_representable_moments(matrix: &Array2<f64>, names: &[String]) -> Result<()> {
    for (column, name) in matrix.columns().into_iter().zip(names) {
        let norm = column
            .iter()
            .fold(0.0_f64, |norm, value| norm.hypot(*value));
        let moment = norm * norm;
        // Exactly zero columns retain the backend's existing omission rules.
        if !moment.is_finite() || (norm > 0.0 && moment == 0.0) {
            return Err(HayashiError::Runtime(format!(
                "IV normal-equation moment for '{name}' is not representable; rescale predictors"
            )));
        }
    }
    Ok(())
}

impl Interpreter {
    /// Materialise X and Z independently on a single filtered estimation sample.
    pub(super) fn prepare_iv(&mut self, args: &[Expr], opts: &[Opt]) -> Result<PreparedIv> {
        if args.len() < 3 {
            return Err(
                self.rt_err("IV requires (structural formula, instrument formula, dataframe)")
            );
        }
        let structural = self.resolve_formula_allow_no_intercept(&args[0])?;
        let instruments = self.resolve_formula_allow_no_intercept(&args[1])?;
        let Expr::Var(name) = &args[2] else {
            return Err(self.rt_err("third argument must be a DataFrame variable"));
        };
        let frame = match self.env.get(name) {
            Some(Value::DataFrame(frame)) => frame.clone(),
            _ => return Err(self.rt_err(format!("'{name}' is not a DataFrame"))),
        };
        let frame = self.maybe_filter_df(&frame, opts)?;
        if frame.n_rows() == 0 {
            return Err(self.rt_err("IV: no observations remain after filtering"));
        }
        let (structural_frame, structural_formula, structural_display) =
            self.prepare_formula_allow_no_intercept(&structural, &frame)?;
        let (instrument_frame, instrument_formula, instrument_display) =
            self.prepare_formula_allow_no_intercept(&instruments, &frame)?;
        let design = IvPredictionDesign::from_frame(
            structural,
            &structural_formula,
            &structural_frame,
            &structural_display,
        )?;
        let instrument_design = IvPredictionDesign::from_frame(
            instruments,
            &instrument_formula,
            &instrument_frame,
            &instrument_display,
        )?;
        let y = structural_frame
            .get_column(&structural_formula.dependent)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?
            .to_float();
        ensure_finite(&y, &structural_formula.dependent)?;
        let x = design.matrix(&structural_frame, &structural_formula)?;
        let z = instrument_design.matrix(&instrument_frame, &instrument_formula)?;
        ensure_representable_moments(&x, &design.names)?;
        ensure_representable_moments(&z, &instrument_design.names)?;
        if y.len() != x.nrows() || y.len() != z.nrows() {
            return Err(self.rt_err("IV structural and instrument sample rows disagree"));
        }
        Ok(PreparedIv {
            structural_frame,
            y,
            x,
            z,
            instrument_terms: instrument_design
                .names
                .into_iter()
                .skip(usize::from(instrument_formula.intercept))
                .collect(),
            design,
        })
    }

    /// Rematerialise only structural predictors using the fitted category schema.
    pub(super) fn iv_prediction_matrix(
        &mut self,
        design: &IvPredictionDesign,
        frame: &Arc<DataFrame>,
    ) -> Result<Array2<f64>> {
        let (materialised_frame, formula, _) =
            self.prepare_formula_allow_no_intercept(&design.formula, frame)?;
        let matrix = design.matrix(&materialised_frame, &formula)?;
        Ok(matrix.select(Axis(1), &design.retained_columns))
    }
}
