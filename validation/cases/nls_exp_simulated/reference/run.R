df <- read.csv(file.path("validation", "cases", "nls_exp_simulated", "data", "data.csv"))

fit <- nls(y ~ a * exp(b * x), data = df, start = list(a = 2.0, b = -1.0), control = nls.control(maxiter = 200, tol = 1e-8))
sm <- summary(fit)

a <- as.numeric(coef(fit)["a"])
b <- as.numeric(coef(fit)["b"])
se_a <- as.numeric(sm$coefficients["a", "Std. Error"])
se_b <- as.numeric(sm$coefficients["b", "Std. Error"])

cat(sprintf(
  '{"coefficients":{"a":%.17g,"b":%.17g},"standard_errors":{"a":%.17g,"b":%.17g}}',
  a, b, se_a, se_b
), "\n")
