# Reference implementation in R for the book VAR(1) case.
# Estimates equation-by-equation OLS to match Hayashi's VAR output.

df <- read.csv("validation/cases/var_book/data/var.csv")
n <- nrow(df)

# Build lags.
y1 <- df$y1
y2 <- df$y2
y1_l1 <- c(NA, y1[1:(n - 1)])
y2_l1 <- c(NA, y2[1:(n - 1)])

# Equation 1: y1 on const, y1.L1, y2.L1
idx1 <- 2:n
X1 <- cbind(1, y1_l1[idx1], y2_l1[idx1])
b1 <- solve(t(X1) %*% X1) %*% t(X1) %*% y1[idx1]
r1 <- y1[idx1] - X1 %*% b1
se1 <- sqrt(diag(sum(r1^2) / (n - 1 - 3) * solve(t(X1) %*% X1)))

# Equation 2: y2 on const, y1.L1, y2.L1
X2 <- cbind(1, y1_l1[idx1], y2_l1[idx1])
b2 <- solve(t(X2) %*% X2) %*% t(X2) %*% y2[idx1]
r2 <- y2[idx1] - X2 %*% b2
se2 <- sqrt(diag(sum(r2^2) / (n - 1 - 3) * solve(t(X2) %*% X2)))

result <- list(
  coefficients = list(
    y1_const = as.numeric(b1[1]),
    y1_y1.L1 = as.numeric(b1[2]),
    y1_y2.L1 = as.numeric(b1[3]),
    y2_const = as.numeric(b2[1]),
    y2_y1.L1 = as.numeric(b2[2]),
    y2_y2.L1 = as.numeric(b2[3])
  ),
  standard_errors = list(
    y1_const = as.numeric(se1[1]),
    y1_y1.L1 = as.numeric(se1[2]),
    y1_y2.L1 = as.numeric(se1[3]),
    y2_const = as.numeric(se2[1]),
    y2_y1.L1 = as.numeric(se2[2]),
    y2_y2.L1 = as.numeric(se2[3])
  )
)

out_dir <- "validation/cases/var_book/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fmt_json <- function(obj) {
  keys <- names(obj)
  parts <- vapply(seq_along(keys), function(i) {
    sprintf('"%s": %.10g', keys[i], obj[[i]])
  }, character(1))
  paste0("{", paste(parts, collapse = ", "), "}")
}

json <- paste0(
  '{"coefficients": ', fmt_json(result$coefficients),
  ', "standard_errors": ', fmt_json(result$standard_errors), "}"
)

writeLines(json, file.path(out_dir, "expected.json"), sep = "")
cat(json)
