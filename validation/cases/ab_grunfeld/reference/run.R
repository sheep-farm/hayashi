# Reference implementation in R for the Arellano-Bond Grunfeld case.
#
# One-step difference GMM (Arellano-Bond 1991) implemented in base R, without
# the plm package, so the case can run in environments where plm is not installed.

# Ensure the data directory exists.
data_dir <- "validation/cases/ab_grunfeld/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "grunfeld.csv")
if (!file.exists(csv_path)) {
  # Fallback to the panel_fe_grunfeld dataset if this case has not downloaded yet.
  fallback <- "validation/cases/panel_fe_grunfeld/data/grunfeld.csv"
  if (file.exists(fallback)) {
    file.copy(fallback, csv_path)
  } else {
    stop("Grunfeld CSV not found. Run panel_fe_grunfeld reference first.")
  }
}

grunfeld <- read.csv(csv_path)
grunfeld <- grunfeld[order(grunfeld$firm, grunfeld$year), ]

# First differences by firm.
grunfeld$d_inv <- ave(grunfeld$inv, grunfeld$firm, FUN = function(x) c(NA, diff(x)))
grunfeld$d_value <- ave(grunfeld$value, grunfeld$firm, FUN = function(x) c(NA, diff(x)))
grunfeld$d_capital <- ave(grunfeld$capital, grunfeld$firm, FUN = function(x) c(NA, diff(x)))
grunfeld$lag_inv <- ave(grunfeld$inv, grunfeld$firm, FUN = function(x) c(NA, x[-length(x)]))
grunfeld$lag2_inv <- ave(grunfeld$inv, grunfeld$firm, FUN = function(x) c(NA, NA, x[-c(length(x)-1, length(x))]))
# Δy_{t-1} = y_{t-1} - y_{t-2}
grunfeld$d_lag_inv <- grunfeld$lag_inv - grunfeld$lag2_inv

# Keep observations usable for t >= 3.
idx <- with(grunfeld, !is.na(d_inv) & !is.na(d_lag_inv) & !is.na(lag2_inv)
              & !is.na(d_value) & !is.na(d_capital))

df <- grunfeld[idx, ]

Y <- df$d_inv
n <- length(Y)
# X: [LD.y, Δvalue, Δcapital]
X <- cbind(df$d_lag_inv, df$d_value, df$d_capital)
# Z: [inv_{t-2}, Δvalue, Δcapital] — exactly identified (one instrument for the
# endogenous lagged difference, plus the differenced exogenous regressors as their
# own instruments).
Z <- cbind(df$lag2_inv, df$d_value, df$d_capital)

# One-step GMM with the Arellano-Bond weight matrix H.
# H is block-diagonal: for each firm the first-difference errors have
# covariance matrix tridiag(2, -1, -1).

# Build block-diagonal H based on firm/time ordering.
entity_labels <- df$firm
blocks <- rle(entity_labels)
H <- matrix(0, nrow = n, ncol = n)
ptr <- 1
for (b_len in blocks$lengths) {
  if (b_len >= 2) {
    for (s in 1:b_len) {
      H[ptr + s - 1, ptr + s - 1] <- 2
      if (s > 1) H[ptr + s - 1, ptr + s - 2] <- -1
      if (s < b_len) H[ptr + s - 1, ptr + s] <- -1
    }
  }
  ptr <- ptr + b_len
}

ZHZ <- t(Z) %*% H %*% Z
W <- solve(ZHZ)
ZX <- t(Z) %*% X
ZY <- t(Z) %*% Y
beta <- solve(t(ZX) %*% W %*% ZX) %*% (t(ZX) %*% W %*% ZY)

# Entity-clustered robust (sandwich) standard errors matching the one-step GMM
# variance estimator in Hayashi/Greeners.
resid <- as.vector(Y - X %*% beta)
entity_labels <- df$firm
blocks <- rle(entity_labels)
S <- matrix(0, nrow = ncol(Z), ncol = ncol(Z))
ptr <- 1
for (b_len in blocks$lengths) {
  zi <- Z[ptr:(ptr + b_len - 1), , drop = FALSE]
  ui <- resid[ptr:(ptr + b_len - 1)]
  zui <- t(zi) %*% ui
  S <- S + zui %*% t(zui)
  ptr <- ptr + b_len
}
A <- t(ZX) %*% W %*% ZX
V <- solve(A) %*% (t(ZX) %*% W %*% S %*% W %*% ZX) %*% solve(A)
se <- sqrt(pmax(diag(V), 0))

# Use the same labels as the Hayashi text output.
coefs <- c("LD.y" = beta[1], "Δvalue" = beta[2], "Δcapital" = beta[3])
std_errors <- c("LD.y" = se[1], "Δvalue" = se[2], "Δcapital" = se[3])

# Helper to write a JSON object with numeric values in full precision on one line.
fmt_json <- function(obj) {
  keys <- names(obj)
  parts <- vapply(seq_along(keys), function(i) {
    sprintf('"%s": %.10g', keys[i], obj[[i]])
  }, character(1))
  paste0("{", paste(parts, collapse = ", "), "}")
}

result <- paste0(
  '{"coefficients": ', fmt_json(setNames(as.list(coefs), names(coefs))),
  ', "standard_errors": ', fmt_json(setNames(as.list(std_errors), names(std_errors))), "}"
)

out_dir <- "validation/cases/ab_grunfeld/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

writeLines(result, file.path(out_dir, "expected.json"), sep = "")

cat(result)
