use ndarray::{Array1, Array2};
use std::rc::Rc;

// ── Wrappers that preserve the X matrix for diagnostics and predict ─────────

#[derive(Clone)]
pub struct OlsModel {
    pub result: Rc<greeners::OlsResult>,
    pub residuals: Array1<f64>,
    pub x: Array2<f64>,
}

impl std::fmt::Display for OlsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

#[derive(Clone)]
pub struct PenalizedModel {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub variable_names: Vec<String>,
    pub r_squared: f64,
    pub n_obs: usize,
    pub alpha: f64,
    pub l1_ratio: Option<f64>,
    pub kind: String,
}

impl std::fmt::Display for PenalizedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let title = match self.kind.as_str() {
            "ridge" => "Ridge Regression",
            "lasso" => "Lasso Regression",
            "elasticnet" => "ElasticNet Regression",
            _ => "Penalized Regression",
        };
        writeln!(f, "\n{:=^60}", format!(" {title} "))?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.6}", "Alpha:", self.alpha)?;
        if let Some(l1r) = self.l1_ratio {
            writeln!(f, "{:<20} {:>10.6}", "L1 ratio:", l1r)?;
        }
        writeln!(f, "{:<20} {:>10.4}", "R-squared:", self.r_squared)?;

        writeln!(f, "\n{:-^60}", " Coefficients ")?;
        writeln!(
            f,
            "{:<15} {:>12} {:>12} {:>12} {:>12}",
            "Variable", "coef", "std err", "t", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(60))?;
        for i in 0..self.params.len() {
            writeln!(
                f,
                "{:<15} {:>12.6} {:>12.6} {:>12.4} {:>12.4}",
                self.variable_names[i], self.params[i], self.std_errors[i], 0.0, 0.0
            )?;
        }
        writeln!(f, "{:=^60}", "")
    }
}

#[derive(Clone)]
pub struct BinaryModel {
    pub result: Rc<greeners::discrete::BinaryModelResult>,
    pub y: Array1<f64>,
    pub x: Array2<f64>,
    pub kind: String,            // "logit" | "probit"
    pub coef_names: Vec<String>, // coefficient names for margins
}

impl std::fmt::Display for BinaryModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── SUR wrapper (preserves variable names per equation) ─────────────────────

#[derive(Clone)]
pub struct SurModel {
    pub result: Rc<greeners::sur::SurResult>,
    pub eq_var_names: Vec<Vec<String>>, // names per equation
}

impl std::fmt::Display for SurModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = &self.result;
        let thick = "═".repeat(78);
        let thin = "─".repeat(78);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^78}", " Seemingly Unrelated Regressions (SUR) ")?;
        writeln!(f, "{:^78}", "Zellner's Efficient Estimator")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " Cross-Equation Error Correlation (Σ):")?;
        for row in r.sigma_cross.rows() {
            write!(f, "  [")?;
            for v in row {
                write!(f, " {:>8.4}", v)?;
            }
            writeln!(f, " ]")?;
        }
        for (eq, vnames) in r.equations.iter().zip(self.eq_var_names.iter()) {
            writeln!(f, "\n{:-^78}", format!(" Equation: {} ", eq.name))?;
            writeln!(
                f,
                "{:<20} {:>10} {:>10} {:>8} {:>8}",
                "Variable", "Coef", "Std Err", "t", "P>|t|"
            )?;
            writeln!(f, "{thin}")?;
            for i in 0..eq.params.len() {
                let vname: &str = vnames.get(i).map(|s| s.as_str()).unwrap_or("?");
                writeln!(
                    f,
                    "{:<20} {:>10.4} {:>10.4} {:>8.3} {:>8.3}",
                    vname, eq.params[i], eq.std_errors[i], eq.t_values[i], eq.p_values[i]
                )?;
            }
            writeln!(f, " R² = {:.4}", eq.r_squared)?;
        }
        writeln!(f, "{thick}")
    }
}

// ── PCA wrapper (adds variable names to PCAResult) ───────────────────────────
#[derive(Clone)]
pub struct PcaModel {
    pub result: Rc<greeners::PCAResult>,
    pub var_names: Vec<String>,
}

impl std::fmt::Display for PcaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = &self.result;
        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^62}", " Principal Component Analysis ")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " {:>20}  {:>10}", "Observations:", r.n_obs)?;
        writeln!(f, " {:>20}  {:>10}", "Components:", r.n_components)?;
        writeln!(f, " {:>20}  {:>10}", "Variables:", self.var_names.len())?;
        writeln!(
            f,
            "\n{:^12} {:>12} {:>12} {:>10}",
            "Component", "Var Expl.", "% Cum.", "Eigenvalue"
        )?;
        writeln!(f, "{thin}")?;
        let mut cum = 0.0;
        for i in 0..r.n_components {
            cum += r.explained_variance_ratio[i];
            writeln!(
                f,
                " PC{:<9} {:>12.4} {:>12.4} {:>10.4}",
                i + 1,
                r.explained_variance_ratio[i],
                cum,
                r.explained_variance[i]
            )?;
        }
        writeln!(f, "\n{:^62}", " Loadings ")?;
        writeln!(f, "{thin}")?;
        let hdr: String = (0..r.n_components)
            .map(|i| format!(" {:>8}", format!("PC{}", i + 1)))
            .collect();
        writeln!(f, "{:<18}{hdr}", "Variable")?;
        for (j, vname) in self.var_names.iter().enumerate() {
            let row: String = (0..r.n_components)
                .map(|i| format!(" {:>8.4}", r.loadings[[j, i]]))
                .collect();
            writeln!(f, "{:<18}{row}", vname)?;
        }
        writeln!(f, "{thick}")
    }
}

// ── Factor Analysis wrapper ───────────────────────────────────────────────────
#[derive(Clone)]
pub struct FactorModel {
    pub result: Rc<greeners::FactorResult>,
    pub var_names: Vec<String>,
}

impl std::fmt::Display for FactorModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = &self.result;
        let thick = "═".repeat(62);
        let thin = "─".repeat(62);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^62}", " Factor Analysis (Principal Axis) ")?;
        writeln!(f, "{thin}")?;
        writeln!(f, " {:>20}  {:>10}", "Observations:", r.n_obs)?;
        writeln!(f, " {:>20}  {:>10}", "Factors:", r.n_factors)?;
        writeln!(f, "\n{:^62}", " Factor Loadings ")?;
        writeln!(f, "{thin}")?;
        let hdr: String = (0..r.n_factors)
            .map(|i| format!(" {:>8}", format!("F{}", i + 1)))
            .collect();
        writeln!(f, "{:<18}{hdr}  {:>10}", "Variable", "Communality")?;
        for (j, vname) in self.var_names.iter().enumerate() {
            let row: String = (0..r.n_factors)
                .map(|i| format!(" {:>8.4}", r.loadings[[j, i]]))
                .collect();
            writeln!(f, "{:<18}{row}  {:>10.4}", vname, r.communalities[j])?;
        }
        writeln!(f, "\n{:<12} {:>10}", "Eigenvalues:", "")?;
        for (i, &ev) in r.eigenvalues.iter().enumerate() {
            writeln!(f, "  F{:<10} {:>10.4}", i + 1, ev)?;
        }
        writeln!(f, "{thick}")
    }
}

// ── DFM wrapper ───────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct DFMModel {
    pub result: Rc<greeners::DynamicFactorResult>,
    #[allow(dead_code)]
    pub var_names: Vec<String>,
}

impl std::fmt::Display for DFMModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── 3SLS wrapper ──────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct ThreeSLSModel {
    pub result: Rc<greeners::three_sls::ThreeSLSResult>,
    #[allow(dead_code)]
    pub eq_var_names: Vec<Vec<String>>,
}

impl std::fmt::Display for ThreeSLSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── Diagnostic test result (print-on-demand) ───────────────────────────────

#[derive(Debug, Clone)]
pub struct DiagResult {
    pub rendered: String, // pre-rendered output by the test
}

impl std::fmt::Display for DiagResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.rendered)
    }
}
