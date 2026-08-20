# Reference implementation in R for the Wooldridge card clustered-IV case.
#
# The IV clustered covariance is computed manually to match the Hayashi/Greeners
# convention and avoid package-specific finite-sample defaults.

library(wooldridge)
library(jsonlite)

data(card)

case_dir <- "validation/cases/iv_cluster_card"
data_dir <- file.path(case_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

region_cols <- paste0("reg66", 1:9)
vars <- c(
  "lwage", "educ", "exper", "expersq", "black", "south", "smsa",
  "nearc4", region_cols
)
card_clean <- card[complete.cases(card[, vars]), vars]
if (any(rowSums(card_clean[, region_cols]) != 1)) {
  stop("card region indicators are not mutually exclusive and exhaustive")
}
card_clean$region <- max.col(card_clean[, region_cols], ties.method = "first")
card_clean$feduc <- as.numeric(card_clean$educ)
card_clean$fexper <- as.numeric(card_clean$exper)
card_clean$fexpersq <- as.numeric(card_clean$expersq)
card_clean$fblack <- as.numeric(card_clean$black)
card_clean$fsouth <- as.numeric(card_clean$south)
card_clean$fsmsa <- as.numeric(card_clean$smsa)
card_clean$fnearc4 <- as.numeric(card_clean$nearc4)

write.csv(card_clean, file.path(data_dir, "card.csv"), row.names = FALSE)

y <- as.matrix(card_clean[, "lwage"])
x <- model.matrix(~ feduc + fexper + fexpersq + fblack + fsouth + fsmsa, data = card_clean)
z <- model.matrix(~ fnearc4 + fexper + fexpersq + fblack + fsouth + fsmsa, data = card_clean)
clusters <- card_clean$region

ztz_inv <- solve(crossprod(z))
x_hat <- z %*% ztz_inv %*% crossprod(z, x)
xhx_inv <- solve(crossprod(x_hat))
beta <- xhx_inv %*% crossprod(x_hat, y)

resid <- y - x %*% beta
n <- nrow(x)
k <- ncol(x)
cluster_levels <- unique(clusters)
g <- length(cluster_levels)

meat <- matrix(0, nrow = k, ncol = k)
for (cluster in cluster_levels) {
  idx <- clusters == cluster
  score <- crossprod(x_hat[idx, , drop = FALSE], resid[idx, , drop = FALSE])
  meat <- meat + score %*% t(score)
}

finite_sample_correction <- (g / (g - 1)) * ((n - 1) / (n - k))
vcov_cluster <- finite_sample_correction * xhx_inv %*% meat %*% xhx_inv
std_errors <- sqrt(diag(vcov_cluster))

coefs <- as.numeric(beta)
names(coefs) <- colnames(x)
names(coefs)[names(coefs) == "(Intercept)"] <- "Intercept"

names(std_errors) <- colnames(x)
names(std_errors)[names(std_errors) == "(Intercept)"] <- "Intercept"

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- file.path(case_dir, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
