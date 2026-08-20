# Independent synthetic-control-style ATT reference using constrained MSE weights.

case_dir <- "validation/cases/synthdid_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

units <- sort(unique(df$unit))
periods <- sort(unique(df$period))
T <- length(periods)
n_units <- length(units)

Y <- matrix(0, n_units, T)
for (i in seq_len(nrow(df))) {
  r <- df[i, ]
  Y[r$unit + 1, r$period + 1] <- r$y
}

treated <- tapply(df$treated, df$unit, max) > 0
treat_period <- min(df$period[df$treated == 1])

treated_idx <- which(treated)
control_idx <- which(!treated)

y_treated <- colMeans(Y[treated_idx, , drop = FALSE])

Y_pre <- Y[control_idx, 1:treat_period, drop = FALSE]
y_treated_pre <- y_treated[1:treat_period]
Y_post <- Y[control_idx, (treat_period + 1):T, drop = FALSE]
y_treated_post <- y_treated[(treat_period + 1):T]

n_control <- length(control_idx)

# Parameterise weights with softmax and minimise pre-treatment MSE.
mse <- function(theta) {
  w <- exp(theta)
  w <- w / sum(w)
  y_syn <- as.numeric(w %*% Y_pre)
  mean((y_treated_pre - y_syn) ^ 2)
}

init <- rep(0, n_control)
res <- optim(init, mse, method = "L-BFGS-B", lower = rep(-10, n_control), upper = rep(10, n_control))
w <- exp(res$par)
w <- w / sum(w)

y_syn_post <- as.numeric(w %*% Y_post)
att <- mean(y_treated_post - y_syn_post)

result <- list(
  coefficients = list(ATT = att),
  standard_errors = list(ATT = NaN)
)

json_str <- jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null", pretty = FALSE)
json_str <- gsub(":null", ":NaN", json_str, fixed = TRUE)
cat(json_str)

