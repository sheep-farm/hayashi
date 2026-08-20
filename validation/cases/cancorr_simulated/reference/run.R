args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)][1])
case_dir <- normalizePath(file.path(dirname(script_path), ".."))
df <- read.csv(file.path(case_dir, "data", "data.csv"))

X <- df[, c("x1", "x2")]
Y <- df[, c("y1", "y2")]

res <- cancor(X, Y)

cancorrs <- res$cor

n <- nrow(X)
p <- ncol(X)
q <- ncol(Y)

# Wilks' lambda and Rao's F approximation
lambda_prod <- prod(1 - cancorrs^2)

# Bartlett chi-square
m <- n - 0.5 * (p + q + 1)
chi2 <- -m * log(lambda_prod)
df_chi2 <- p * q
p_val <- pchisq(chi2, df_chi2, lower.tail = FALSE)

out <- list(
  coefficients = list(
    cancorr_1 = cancorrs[1],
    cancorr_2 = cancorrs[2],
    wilks_lambda = lambda_prod,
    f_stat = chi2,
    p_value = p_val
  ),
  standard_errors = list(
    cancorr_1 = 0.0,
    cancorr_2 = 0.0,
    wilks_lambda = 0.0,
    f_stat = 0.0,
    p_value = 0.0
  )
)

json_str <- jsonlite::toJSON(out, auto_unbox = TRUE, digits = 15, na = "null", pretty = FALSE)
json_str <- gsub(":null", ":NaN", json_str, fixed = TRUE)
cat(json_str)
