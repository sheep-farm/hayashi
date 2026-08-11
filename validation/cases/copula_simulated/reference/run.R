df <- read.csv(file.path("validation", "cases", "copula_simulated", "data", "data.csv"))

y <- df$y
x <- df$x

# Kendall's tau and Spearman's rho (with sin transformation to theta)
tau <- cor(y, x, method = "kendall")
rho <- sin(pi / 2 * tau)
spearman <- cor(y, x, method = "spearman")

result <- list(
  theta = as.numeric(rho),
  kendall_tau_yx = as.numeric(tau),
  spearman_rho_yx = as.numeric(spearman)
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, pretty = TRUE), "\n")
