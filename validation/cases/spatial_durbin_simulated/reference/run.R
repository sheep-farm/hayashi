# Reference implementation in R for Spatial Durbin model.

library(jsonlite)

# Simulate data following the same DGP as Hayashi.
set.seed(42)
n <- 25

# Generate spatial coordinates
row <- floor((0:(n-1)) / 5)
col <- (0:(n-1)) %% 5

# Generate covariates
x1 <- rnorm(n)
x2 <- rnorm(n)

# Simplified spatial lag: average of neighbors
wx1 <- numeric(n)
wx2 <- numeric(n)

for (i in 1:n) {
  r <- row[i]
  c <- col[i]
  
  sum_x1 <- 0
  sum_x2 <- 0
  count <- 0
  
  # Check 4 neighbors
  if (r > 0) {
    idx <- (r - 1) * 5 + c + 1
    sum_x1 <- sum_x1 + x1[idx]
    sum_x2 <- sum_x2 + x2[idx]
    count <- count + 1
  }
  if (r < 4) {
    idx <- (r + 1) * 5 + c + 1
    sum_x1 <- sum_x1 + x1[idx]
    sum_x2 <- sum_x2 + x2[idx]
    count <- count + 1
  }
  if (c > 0) {
    idx <- r * 5 + (c - 1) + 1
    sum_x1 <- sum_x1 + x1[idx]
    sum_x2 <- sum_x2 + x2[idx]
    count <- count + 1
  }
  if (c < 4) {
    idx <- r * 5 + (c + 1) + 1
    sum_x1 <- sum_x1 + x1[idx]
    sum_x2 <- sum_x2 + x2[idx]
    count <- count + 1
  }
  
  if (count > 0) {
    wx1[i] <- sum_x1 / count
    wx2[i] <- sum_x2 / count
  } else {
    wx1[i] <- 0
    wx2[i] <- 0
  }
}

# Generate outcome with spatial Durbin structure
beta1 <- 0.5
beta2 <- 0.3
theta1 <- 0.2
theta2 <- 0.1

noise <- rnorm(n) * 0.5
y <- beta1 * x1 + beta2 * x2 + theta1 * wx1 + theta2 * wx2 + noise

# Create DataFrame including spatial lags so Hayashi can read them.
df <- data.frame(y = y, x1 = x1, x2 = x2, wx1 = wx1, wx2 = wx2, row = row, col = col)

# Ensure the data directory exists.
data_dir <- "validation/cases/spatial_durbin_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(df, file.path(data_dir, "spatial_durbin.csv"), row.names = FALSE)

# Simplified reference: estimate OLS with spatial lags as covariates
model <- lm(y ~ x1 + x2 + wx1 + wx2, data = df)

# Extract coefficients
coefs <- coef(model)

result <- list(
  coefficients = as.list(coefs)
)

out_dir <- "validation/cases/spatial_durbin_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
