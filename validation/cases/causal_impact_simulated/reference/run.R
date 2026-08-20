# Reference implementation in R for Causal Impact.

library(jsonlite)

# Simulate data following the same DGP as Hayashi.
set.seed(42)
n <- 200
treatment_start <- 100
treatment_effect <- 10.0

y <- numeric(n)
treatment <- numeric(n)

for (t in 1:n) {
  trend <- 0.1 * t
  seasonal <- 5.0 * sin(2 * pi * t / 50)
  noise <- rnorm(1) * 2.0
  
  if (t >= treatment_start) {
    treatment[t] <- 1
    y[t] <- trend + seasonal + treatment_effect + noise
  } else {
    treatment[t] <- 0
    y[t] <- trend + seasonal + noise
  }
}

# Create DataFrame
df <- data.frame(y = y, treatment = treatment, time = 1:n, time_num = as.numeric(1:n))

# Ensure the data directory exists.
data_dir <- "validation/cases/causal_impact_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(df, file.path(data_dir, "causal_impact.csv"), row.names = FALSE)

treatment_start <- 100
n <- nrow(df)
n_post <- n - treatment_start

# Pre-treatment period: observations 1..treatment_start
pre_period <- df[df$time <= treatment_start, ]
# Post-treatment period: observations (treatment_start+1)..n
post_period <- df[df$time > treatment_start, ]

# Estimate counterfactual using pre-treatment trend
pre_trend <- lm(y ~ time, data = pre_period)

# Predict counterfactual for post-period
counterfactual <- predict(pre_trend, newdata = post_period)

# Point effect: actual - counterfactual
point_effects <- post_period$y - counterfactual
avg_effect <- mean(point_effects)

# Cumulative effect
total_effect <- sum(point_effects)

# CIs using standard deviation of point effects
point_effect_sd <- sd(point_effects)
point_effect_se <- point_effect_sd / sqrt(length(point_effects))
avg_effect_sd <- point_effect_se
avg_effect_lower <- avg_effect - 1.96 * avg_effect_sd
avg_effect_upper <- avg_effect + 1.96 * avg_effect_sd

cumulative_se <- sqrt(length(point_effects)) * point_effect_sd
cumulative_lower <- total_effect - 1.96 * cumulative_se
cumulative_upper <- total_effect + 1.96 * cumulative_se

result <- list(
  avg_effect = avg_effect,
  avg_effect_lower = avg_effect_lower,
  avg_effect_upper = avg_effect_upper,
  total_effect = total_effect,
  total_effect_lower = cumulative_lower,
  total_effect_upper = cumulative_upper
)

out_dir <- "validation/cases/causal_impact_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
