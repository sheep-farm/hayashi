library(jsonlite)
case_dir <- "validation/cases/linktest_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
y <- df$y
fit <- glm(y ~ x, data = df, family = binomial)
X <- cbind(1, df$x)
yhat <- as.vector(X %*% coef(fit))
fit2 <- glm(y ~ yhat + I(yhat^2), family = binomial)
s <- summary(fit2)$coefficients
cat(toJSON(list(
  coefficients = list(yhat = s[2, 1], yhat2 = s[3, 1]),
  standard_errors = list(yhat = s[2, 2], yhat2 = s[3, 2])
), auto_unbox = TRUE, digits = 15))
