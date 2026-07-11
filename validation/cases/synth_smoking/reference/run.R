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

# Optimise weights with L-BFGS-B inside [0,1], then project onto the simplex.
objective <- function(w) {
  sum((y_t_pre - Y_d_pre %*% w) ^ 2)
}

init <- rep(1 / n_d, n_d)
opt <- optim(init, objective, method = "L-BFGS-B", lower = rep(0, n_d), upper = rep(1, n_d))

# Projection onto the unit simplex (non-negative, sum to 1).
project_simplex <- function(v) {
  v <- pmax(v, 0)
  s <- sum(v)
  if (s > 0) v / s else rep(1 / length(v), length(v))
}

w <- project_simplex(opt$par)

y_sc_post <- Y_d_post %*% w
att <- mean(y_post[, "y.1"] - y_sc_post)

result <- list(coefficients = list(ATT = as.numeric(att)))

out_dir <- file.path(CASE_DIR, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
