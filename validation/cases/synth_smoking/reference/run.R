# Reference implementation in R for synthetic control.
#
# Uses the same simulated panel as the Hayashi script (10 donors + 1 treated
# unit, 20 periods).  The treated unit is unit 1; intervention starts in
# period 11.  Donor weights are obtained by minimising the pre-treatment
# squared prediction error subject to simplex constraints.

library(jsonlite)

CASE_DIR <- "validation/cases/synth_smoking"
DATA_DIR <- file.path(CASE_DIR, "data")
CSV_PATH <- file.path(DATA_DIR, "synth_smoking.csv")

if (!file.exists(CSV_PATH)) {
  stop("synth_smoking.csv not found; run the Python reference first")
}

df <- read.csv(CSV_PATH)

y_wide <- reshape(df[, c("year", "unit", "y")],
                  idvar = "year", timevar = "unit", direction = "wide")

T0 <- 10
y_pre <- y_wide[1:T0, -1]
y_post <- y_wide[(T0 + 1):nrow(y_wide), -1]

y_t_pre <- y_pre[, "y.1"]
Y_d_pre <- as.matrix(y_pre[, -1])
Y_d_post <- as.matrix(y_post[, -1])

n_d <- ncol(Y_d_pre)
if (n_d < 2) {
  stop("At least two donor units are required for the constrained reference")
}

# Solve the basic outcome-only SCM quadratic programme on the unit simplex.
# Eliminate the final weight so constrOptim can enforce non-negativity with
# linear inequalities while preserving the exact sum-to-one constraint.
objective <- function(theta) {
  w <- c(theta, 1 - sum(theta))
  residuals <- y_t_pre - Y_d_pre %*% w
  sum(residuals^2)
}

gradient <- function(theta) {
  w <- c(theta, 1 - sum(theta))
  residuals <- y_t_pre - Y_d_pre %*% w
  donor_differences <- Y_d_pre[, -n_d, drop = FALSE] - Y_d_pre[, n_d]
  as.vector(-2 * crossprod(donor_differences, residuals))
}

ui <- rbind(diag(n_d - 1), -rep(1, n_d - 1))
ci <- c(rep(0, n_d - 1), -1)
opt <- constrOptim(
  theta = rep(1 / n_d, n_d - 1),
  f = objective,
  grad = gradient,
  ui = ui,
  ci = ci,
  method = "BFGS",
  control = list(reltol = 1e-12, maxit = 10000)
)
if (opt$convergence != 0) {
  stop(sprintf("Simplex optimisation failed with code %s", opt$convergence))
}

w <- c(opt$par, 1 - sum(opt$par))

y_sc_post <- Y_d_post %*% w
att <- mean(y_post[, "y.1"] - y_sc_post)

result <- list(coefficients = list(ATT = as.numeric(att)))

out_dir <- file.path(CASE_DIR, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
