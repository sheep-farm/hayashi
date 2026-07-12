# Reference implementation in R for xtgls on Wooldridge wagepan.
# Replicates the Stata-style panel feasible GLS with panels(heteroskedastic)
# as implemented in Hayashi/Greeners, using only base-R.

library(jsonlite)

case_dir <- "validation/cases/xtgls_wagepan"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(file.path(data_dir, "wagepan.csv"), stringsAsFactors = FALSE)

# Sort by entity and time to match the panel extraction order in Greeners.
df <- df[order(df$nr, df$year), ]

y <- df$lwage
X <- cbind(const = 1,
           educ = df$educ,
           exper = df$exper,
           expersq = df$expersq,
           married = df$married,
           union = df$union)
x_cols <- c("const", "educ", "exper", "expersq", "married", "union")

entity_ids <- df$nr
time_ids <- df$year

unique_entities <- sort(unique(entity_ids))
unique_times <- sort(unique(time_ids))
n_entities <- length(unique_entities)
big_t <- length(unique_times)
k <- ncol(X)

# Extract balanced panels.
entity_to_idx <- match(entity_ids, unique_entities)

y_panels <- list()
x_panels <- list()
for (i in seq_along(unique_entities)) {
  e <- unique_entities[i]
  sub <- df[df$nr == e, ]
  sub <- sub[order(sub$year), ]
  if (nrow(sub) != big_t) {
    stop(sprintf("Panel %s is unbalanced: %d observations", e, nrow(sub)))
  }
  y_panels[[i]] <- sub$lwage
  x_panels[[i]] <- cbind(1,
                         sub$educ,
                         sub$exper,
                         sub$expersq,
                         sub$married,
                         sub$union)
}

# Step 1: pooled OLS for initial residuals.
xtx0 <- matrix(0, k, k)
xty0 <- numeric(k)
for (i in seq_len(n_entities)) {
  xi <- x_panels[[i]]
  yi <- y_panels[[i]]
  xtx0 <- xtx0 + t(xi) %*% xi
  xty0 <- xty0 + t(xi) %*% yi
}
beta0 <- solve(xtx0, xty0)
resid0 <- lapply(seq_len(n_entities), function(i) {
  y_panels[[i]] - x_panels[[i]] %*% beta0
})

# Step 2: estimate diagonal Omega and compute (X' Omega^{-1} X) and (X' Omega^{-1} y).
# For panels=hetero: sigma2_i = e_i' e_i / T.
xtox <- matrix(0, k, k)
xtoy <- numeric(k)
for (i in seq_len(n_entities)) {
  ei <- resid0[[i]]
  sigma2_i <- as.numeric(t(ei) %*% ei) / big_t
  if (sigma2_i < 1e-15) {
    stop(sprintf("sigma2_i is near zero for entity %d", i))
  }
  w <- 1.0 / sigma2_i
  xi <- x_panels[[i]]
  yi <- y_panels[[i]]
  xtox <- xtox + t(xi) %*% xi * w
  xtoy <- xtoy + t(xi) %*% yi * w
}

# Step 3: beta_fgls = (X' Omega^{-1} X)^{-1} X' Omega^{-1} y.
xtox_inv <- solve(xtox)
beta <- xtox_inv %*% xtoy

# Residuals and sigma.
resid_gls <- lapply(seq_len(n_entities), function(i) {
  y_panels[[i]] - x_panels[[i]] %*% beta
})
ssr_gls <- sum(sapply(resid_gls, function(e) as.numeric(t(e) %*% e)))
df_resid <- n_entities * big_t - k
sigma <- sqrt(ssr_gls / df_resid)

# Standard errors from (X' Omega^{-1} X)^{-1}, asymptotic (Normal).
std_errors <- sqrt(pmax(diag(xtox_inv), 0))
z_values <- beta / std_errors
p_values <- 2 * (1 - pnorm(abs(z_values)))

result <- list(
  coefficients = as.list(setNames(as.numeric(beta), x_cols)),
  standard_errors = as.list(setNames(as.numeric(std_errors), x_cols))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
