# Explicit R reference for the Wooldridge wagepan panel FE two-way-clustered-SE case.
#
# Within-transformed OLS with two-way (entity + time) clustered covariance.
# Mirrors the Greeners implementation:
#   V = sandwich(X, meat_1 + meat_2 - meat_12, X) * g/(g-1) * (n-1)/(n-k)
# where g = min(G_entity, G_time).

library(jsonlite)

checked_covariance_rejection <- function(result) {
  # Require the fixed fixture's finite, material d81/eigenvalue rejection.
  covariance <- do.call(rbind, lapply(result$covariance, function(row) as.numeric(unlist(row))))
  dimnames(covariance) <- list(names(result$covariance), names(result$covariance))
  diagnostics <- result$covariance_diagnostics
  if (is.null(diagnostics$eigenvalues) || !all(is.finite(covariance))) {
    stop("Availability requires finite raw covariance and its spectrum")
  }
  eigenvalues <- as.numeric(unlist(diagnostics$eigenvalues))
  bounds <- list(diagnostics$symmetry_error, diagnostics$symmetry_bound, diagnostics$eigenvalue_bound)
  if (any(vapply(bounds, is.null, logical(1))) || !all(is.finite(unlist(bounds))) || !all(is.finite(eigenvalues))) {
    stop("Availability requires finite covariance diagnostics")
  }
  reason <- if (is.null(result$inference_reason)) "" else result$inference_reason
  expected_rejection <- identical(result$inference_available, FALSE) && isTRUE(diagnostics$finite) &&
    diagnostics$symmetry_error <= diagnostics$symmetry_bound &&
    covariance["d81", "d81"] < -diagnostics$symmetry_bound &&
    any(eigenvalues < -diagnostics$eigenvalue_bound) &&
    grepl("negative diagonal variance for d81", reason, fixed = TRUE) &&
    grepl("materially negative eigenvalues", reason, fixed = TRUE)
  if (!expected_rejection) {
    stop("Expected finite materially indefinite d81 covariance; this availability contract is not met")
  }
  as.integer(expected_rejection)
}

case_dir <- "validation/cases/panel_fe_twoway_cluster_wagepan"
csv_path <- file.path(case_dir, "data", "wagepan.csv")

if (!file.exists(csv_path)) {
  stop("wagepan.csv is missing; run data/gen.py before the reference script")
}

df <- read.csv(csv_path)
variables <- c("lwage", "union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87", "nr", "year")
df <- df[complete.cases(df[, variables]), variables]

y_name <- "lwage"
x_names <- c("union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87")
entity_name <- "nr"
time_name <- "year"

within_vector <- function(values, groups) {
  values - ave(values, groups, FUN = mean)
}

y <- within_vector(df[[y_name]], df[[entity_name]])
X <- as.matrix(df[, x_names])
for (j in seq_along(x_names)) {
  X[, j] <- within_vector(X[, j], df[[entity_name]])
}

if (!all(is.finite(X)) || !all(is.finite(y)) || nrow(X) <= ncol(X) || qr(X)$rank != ncol(X)) {
  stop("Ordinary within design must be finite, full-rank and have residual degrees of freedom")
}
xtx_inv <- solve(crossprod(X))
beta <- as.numeric(xtx_inv %*% crossprod(X, y))
names(beta) <- x_names
if (!all(is.finite(beta))) stop("Ordinary within coefficients must be finite")

residuals <- as.numeric(y - X %*% beta)
n <- nrow(X)
k <- ncol(X)

clustered_meat <- function(cluster_col) {
  clusters <- df[[cluster_col]]
  unique_clusters <- unique(clusters)
  g <- length(unique_clusters)
  meat <- matrix(0, nrow = k, ncol = k)
  for (cluster in unique_clusters) {
    idx <- clusters == cluster
    x_g <- X[idx, , drop = FALSE]
    u_g <- residuals[idx]
    meat <- meat + t(x_g) %*% u_g %*% t(u_g) %*% x_g
  }
  list(meat = meat, g = g)
}

df$inter <- interaction(df[[entity_name]], df[[time_name]])

m1 <- clustered_meat(entity_name)
m2 <- clustered_meat(time_name)
m12 <- clustered_meat("inter")

meat <- m1$meat + m2$meat - m12$meat
sandwich <- xtx_inv %*% meat %*% xtx_inv

g <- min(m1$g, m2$g)
if (g <= 1) stop("Two-way covariance requires at least two groups in each dimension")
finite_sample_correction <- (g / (g - 1)) * ((n - 1) / (n - k))
vcov <- finite_sample_correction * sandwich
# Assess admissibility without modifying raw covariance. The declared budget is
# 128*k*eps relative to matrix/spectral scale, matching the Python reference.
roundoff_multiplier <- 128
finite_covariance <- all(is.finite(vcov))
negative_variance_terms <- x_names[diag(vcov) < 0 & is.finite(diag(vcov))]
eigenvalues <- NULL
symmetry_error <- NULL
symmetry_bound <- NULL
eigenvalue_bound <- NULL
reasons <- character(0)
if (!finite_covariance) {
  reasons <- c(reasons, "covariance contains non-finite entries")
} else {
  symmetry_error <- max(abs(vcov - t(vcov)))
  symmetry_bound <- roundoff_multiplier * k * .Machine$double.eps * max(abs(vcov))
  # Only the spectral diagnostic uses the symmetric part; raw vcov is retained.
  eigenvalues <- sort(eigen(0.5 * vcov + 0.5 * t(vcov), symmetric = TRUE, only.values = TRUE)$values)
  eigenvalue_bound <- roundoff_multiplier * k * .Machine$double.eps * max(abs(eigenvalues))
  if (symmetry_error > symmetry_bound) {
    reasons <- c(reasons, "covariance is materially asymmetric")
  }
  if (length(negative_variance_terms) > 0) {
    reasons <- c(reasons, paste("negative diagonal variance for", paste(negative_variance_terms, collapse = ", ")))
  }
  if (any(eigenvalues < -eigenvalue_bound)) {
    reasons <- c(reasons, "covariance has materially negative eigenvalues")
  }
}

json_number <- function(value) {
  # Preserve finite numbers and label IEEE non-finite values in valid JSON.
  if (is.finite(value)) return(value)
  if (is.nan(value)) return("NaN")
  if (value > 0) "Infinity" else "-Infinity"
}

inference_available <- length(reasons) == 0
named_covariance <- setNames(lapply(seq_along(x_names), function(i) {
  setNames(lapply(vcov[i, ], json_number), x_names)
}), x_names)
result <- list(
  coefficients = as.list(beta),
  covariance = named_covariance,
  inference_available = inference_available,
  inference_reason = if (inference_available) NULL else paste(reasons, collapse = "; "),
  covariance_diagnostics = list(
    finite = finite_covariance,
    eigenvalues = if (is.null(eigenvalues)) NULL else as.list(eigenvalues),
    symmetry_error = symmetry_error,
    symmetry_bound = symmetry_bound,
    eigenvalue_bound = eigenvalue_bound,
    roundoff_multiplier = roundoff_multiplier,
    negative_variance_terms = as.list(negative_variance_terms)
  )
)
result$availability <- list(covariance_rejected = checked_covariance_rejection(result))

out_dir <- file.path(case_dir, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(
  result,
  file.path(out_dir, "expected.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 16
)

cat(toJSON(result, auto_unbox = TRUE, digits = 16))
