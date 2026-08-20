# Reference implementation in R for Panel Heckman selection model.

library(jsonlite)

# Simulate data following the same DGP as Hayashi.
set.seed(42)
n_entities <- 100
n_periods <- 10
n <- n_entities * n_periods

# Generate entity and time identifiers
entity <- floor((0:(n-1)) / n_periods)
time <- (0:(n-1)) %% n_periods

# Generate covariates
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)

# Generate selection equation
gamma0 <- 0.5
gamma1 <- 0.3
gamma2 <- -0.2
gamma3 <- 0.1
u1 <- rnorm(n) * 0.5

selection_star <- gamma0 + gamma1 * x1 + gamma2 * x2 + gamma3 * x3 + u1
selection <- as.numeric(selection_star > 0)

# Generate outcome equation
beta0 <- 1.0
beta1 <- 0.5
beta2 <- 0.3
beta3 <- -0.1
u2 <- rnorm(n) * 0.5

y <- beta0 + beta1 * x1 + beta2 * x2 + beta3 * x3 + u2

# Set y to NA when selection = 0
y[selection == 0] <- NA

# Create DataFrame
df <- data.frame(
  y = y,
  selection = selection,
  x1 = x1,
  x2 = x2,
  x3 = x3,
  entity = entity,
  time = time
)

# Ensure the data directory exists.
data_dir <- "validation/cases/panel_heckman_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read (empty cells for missing y).
write.csv(df, file.path(data_dir, "panel_heckman.csv"), row.names = FALSE, na = "")

# Simplified reference: estimate selection (probit) then OLS with inverse Mills ratio.
selection_model <- glm(selection ~ x1 + x2 + x3, data = df, family = binomial(link = "probit"))

# Compute inverse Mills ratio manually
xb <- predict(selection_model, type = "link")
imr <- dnorm(xb) / pnorm(xb)
df$imr <- imr

outcome_data <- df[df$selection == 1, ]
outcome_model <- lm(y ~ x1 + x2 + x3 + imr, data = outcome_data)

selection_coefs <- coef(selection_model)
names(selection_coefs)[1] <- "(Intercept)"
outcome_coefs <- coef(outcome_model)
# Rename the IMR coefficient so the parser can identify it consistently.
names(outcome_coefs)[length(outcome_coefs)] <- "lambda_IMR"
inverse_mills_ratio <- mean(imr[selection == 1], na.rm = TRUE)

result <- list(
  selection_coefficients = as.list(selection_coefs),
  outcome_coefficients = as.list(outcome_coefs),
  inverse_mills_ratio = inverse_mills_ratio
)

out_dir <- "validation/cases/panel_heckman_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
