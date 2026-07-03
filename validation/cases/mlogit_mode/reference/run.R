# Reference implementation in R for the multinomial logit mode-choice case.

library(nnet)

case_dir <- "validation/cases/mlogit_mode"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

raw_csv <- file.path(data_dir, "TravelMode.csv")
mode_csv <- file.path(data_dir, "mode.csv")

raw <- read.csv(raw_csv)
chosen <- raw[raw$choice == "yes", c("individual", "mode", "income")]
rownames(chosen) <- NULL

# The alternative-specific attributes (wait, vcost, travel) are averaged per
# individual so that they become individual-specific covariates suitable for a
# standard multinomial logit.
avg <- aggregate(cbind(wait, vcost, travel) ~ individual, data = raw, FUN = mean)
chosen <- merge(chosen, avg, by = "individual")

# Encode mode as numeric: air=1, train=2, bus=3, car=4 (base = 4).
mode_map <- c(air = 1, train = 2, bus = 3, car = 4)
chosen$mode <- mode_map[chosen$mode]
df <- chosen[, c("mode", "income", "wait", "vcost", "travel")]

# Standardise covariates to improve numerical stability of the Newton-Raphson solver.
for (col in c("income", "wait", "vcost", "travel")) {
  df[[col]] <- as.numeric(scale(df[[col]]))
}

write.csv(df, mode_csv, row.names = FALSE)

# Make car (4) the reference category so the non-base categories are 1, 2, 3.
df$mode <- factor(df$mode, levels = c(4, 1, 2, 3))

model <- multinom(mode ~ income + wait + vcost + travel, data = df, trace = FALSE)

summary_model <- summary(model)
coefs <- coef(model)
se <- summary_model$standard.errors

# Flatten to {category}:{variable} keys, matching the Hayashi output format.
result <- list(coefficients = list(), standard_errors = list())
for (cat in rownames(coefs)) {
  cat_key <- as.character(cat)
  for (var in colnames(coefs)) {
    var_key <- if (var == "(Intercept)") "Intercept" else var
    key <- paste0(cat_key, ":", var_key)
    result$coefficients[[key]] <- as.numeric(coefs[cat, var])
    result$standard_errors[[key]] <- as.numeric(se[cat, var])
  }
}

# Build a JSON string without requiring jsonlite.
format_json <- function(x) {
  if (is.list(x)) {
    pairs <- vapply(names(x), function(n) {
      sprintf("\"%s\": %s", n, format_json(x[[n]]))
    }, character(1))
    sprintf("{%s}", paste(pairs, collapse = ", "))
  } else {
    format(x, digits = 17, scientific = FALSE)
  }
}

out <- format_json(result)
writeLines(out, file.path(ref_dir, "expected.json"))
cat(out)
