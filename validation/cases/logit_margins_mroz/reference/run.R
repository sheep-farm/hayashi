# Reference implementation in R for logit average marginal effects.

library(wooldridge)
library(jsonlite)

data(mroz)

data_dir <- "validation/cases/logit_margins_mroz/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(mroz, file.path(data_dir, "mroz.csv"), row.names = FALSE)

formula <- inlf ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6
model <- glm(formula, data = mroz, family = binomial(link = "logit"))

beta <- coef(model)
vcov_beta <- vcov(model)
x <- model.matrix(model)
p <- fitted(model)
g <- p * (1 - p)
mean_g <- mean(g)
dg <- colMeans(g * (1 - 2 * p) * x)

vars <- names(beta)[names(beta) != "(Intercept)"]
marginal_effects <- setNames(numeric(length(vars)), vars)
standard_errors <- setNames(numeric(length(vars)), vars)

for (var in vars) {
  gradient <- beta[var] * dg
  gradient[var] <- gradient[var] + mean_g
  marginal_effects[var] <- beta[var] * mean_g
  standard_errors[var] <- sqrt(as.numeric(t(gradient) %*% vcov_beta %*% gradient))
}

result <- list(
  marginal_effects = as.list(marginal_effects),
  standard_errors = as.list(standard_errors)
)

out_dir <- "validation/cases/logit_margins_mroz/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE, digits = NA))
