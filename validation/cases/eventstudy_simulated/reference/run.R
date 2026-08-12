library(jsonlite)
library(sandwich)

case_dir <- "validation/cases/eventstudy_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

# Construct event-study dummies, dropping reference period -1.
df$t_m2 <- as.integer(df$event_time == -2)
df$t_0  <- as.integer(df$event_time == 0)
df$t_1  <- as.integer(df$event_time == 1)
df$t_2  <- as.integer(df$event_time == 2)

fit <- lm(y ~ t_m2 + t_0 + t_1 + t_2, data = df)

# Cluster-robust variance by unit (HC1-style finite-sample correction).
V <- vcovCL(fit, cluster = df$unit)
se <- sqrt(diag(V))

# Match the variable names emitted by Hayashi.
names <- c("t=-2", "t=0", "t=1", "t=2")
vals <- as.numeric(coef(fit)[-1])  # drop intercept
names(vals) <- names

coefficients <- as.list(vals)
standard_errors <- as.list(setNames(se[-1], names))

result <- list(
  coefficients = coefficients,
  standard_errors = standard_errors
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
