df <- read.csv(file.path("validation", "cases", "nls_ces_simulated", "data", "data.csv"))

fit <- nls(
  y ~ a * (b1 * I(x1^rho) + (1 - b1) * I(x2^rho))^(1 / rho),
  data = df,
  start = list(a = 1.0, b1 = 0.5, rho = 0.5),
  lower = c(a = 0.01, b1 = 0.01, rho = 0.01),
  upper = c(a = 5.0, b1 = 0.99, rho = 2.0),
  algorithm = "port",
  control = nls.control(maxiter = 400, tol = 1e-8)
)
sm <- summary(fit)

a <- as.numeric(coef(fit)["a"])
b1 <- as.numeric(coef(fit)["b1"])
rho <- as.numeric(coef(fit)["rho"])
se_a <- as.numeric(sm$coefficients["a", "Std. Error"])
se_b1 <- as.numeric(sm$coefficients["b1", "Std. Error"])
se_rho <- as.numeric(sm$coefficients["rho", "Std. Error"])

cat(sprintf(
  '{"coefficients":{"a":%.17g,"b1":%.17g,"rho":%.17g},"standard_errors":{"a":%.17g,"b1":%.17g,"rho":%.17g}}',
  a, b1, rho, se_a, se_b1, se_rho
), "\n")
