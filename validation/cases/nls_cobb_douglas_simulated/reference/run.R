df <- read.csv(file.path("validation", "cases", "nls_cobb_douglas_simulated", "data", "data.csv"))

fit <- nls(y ~ a * I(x1^b0) * I(x2^b1), data = df, start = list(a = 1.0, b0 = 0.3, b1 = 0.5), control = nls.control(maxiter = 200, tol = 1e-8))
sm <- summary(fit)

a <- as.numeric(coef(fit)["a"])
b0 <- as.numeric(coef(fit)["b0"])
b1 <- as.numeric(coef(fit)["b1"])
se_a <- as.numeric(sm$coefficients["a", "Std. Error"])
se_b0 <- as.numeric(sm$coefficients["b0", "Std. Error"])
se_b1 <- as.numeric(sm$coefficients["b1", "Std. Error"])

cat(sprintf(
  '{"coefficients":{"a":%.17g,"b0":%.17g,"b1":%.17g},"standard_errors":{"a":%.17g,"b0":%.17g,"b1":%.17g}}',
  a, b0, b1, se_a, se_b0, se_b1
), "\n")
