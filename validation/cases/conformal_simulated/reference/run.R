options(warn=-1)
library(jsonlite)

case_dir <- "validation/cases/conformal_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

n <- nrow(df)
calib_n <- floor(0.3 * n)
train_n <- n - calib_n

train <- df[1:train_n, ]
calib <- df[(train_n + 1):n, ]

mod <- lm(y ~ x1 + x2, data = train)
mu_calib <- predict(mod, newdata = calib)
scores <- abs(calib$y - mu_calib)
q <- quantile(scores, probs = 0.9, names = FALSE)

mu_test <- predict(mod, newdata = df)
lower <- mu_test - q
upper <- mu_test + q
coverage <- mean(df$y >= lower & df$y <= upper)

result <- list(
  coefficients = list(
    empirical_coverage = as.numeric(coverage),
    conformal_quantile = as.numeric(q)
  ),
  standard_errors = list(
    empirical_coverage = 0,
    conformal_quantile = 0
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
