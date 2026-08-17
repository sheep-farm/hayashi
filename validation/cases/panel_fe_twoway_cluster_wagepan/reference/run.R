# Explicit R reference for the Wooldridge wagepan panel FE two-way-clustered-SE case.
#
# Within-transformed OLS with two-way (entity + time) clustered covariance.
# Mirrors the Greeners implementation:
#   V = sandwich(X, meat_1 + meat_2 - meat_12, X) * g/(g-1) * (n-1)/(n-k)
# where g = min(G_entity, G_time).

library(jsonlite)

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

xtx_inv <- solve(crossprod(X))
beta <- as.numeric(xtx_inv %*% crossprod(X, y))
names(beta) <- x_names

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
finite_sample_correction <- (g / (g - 1)) * ((n - 1) / (n - k))
vcov <- finite_sample_correction * sandwich
se <- sqrt(pmax(0, diag(vcov)))
names(se) <- x_names

result <- list(
  coefficients = as.list(beta),
  standard_errors = as.list(se)
)

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
