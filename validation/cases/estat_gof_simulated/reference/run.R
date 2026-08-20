library(jsonlite)
case_dir <- "validation/cases/estat_gof_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
fit <- glm(y ~ x, data = df, family = binomial)
p <- fitted(fit)
n_groups <- 10
g <- as.integer(cut(p, breaks = quantile(p, probs = seq(0, 1, length.out = n_groups + 1), na.rm = TRUE), include.lowest = TRUE))
hl <- 0
for (k in unique(g)) {
  idx <- which(g == k)
  o <- sum(df$y[idx])
  n <- length(idx)
  e <- sum(p[idx])
  if (e > 0 && (n - e) > 0) {
    hl <- hl + (o - e)^2 / (e * (n - e) / n)
  }
}
df_used <- length(unique(g))
df_gof <- max(1, df_used - 2)
pval <- 1 - pchisq(hl, df_gof)
cat(toJSON(list(coefficients = list(hl_stat = hl, p_value = pval), standard_errors = list(hl_stat = 0, p_value = 0)), auto_unbox = TRUE, digits = 15))
