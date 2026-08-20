# Reference implementation in R for the SUR Grunfeld case.

library(systemfit)

case_dir <- "validation/cases/sur_grunfeld"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "grunfeld.csv")
if (!file.exists(csv_path)) {
  fallback <- "validation/cases/panel_fe_grunfeld/data/grunfeld.csv"
  if (file.exists(fallback)) {
    file.copy(fallback, csv_path)
  }
}
df <- read.csv(csv_path)

# Two-equation SUR.
eq1 <- value ~ inv + capital
eq2 <- inv ~ value + capital

model <- systemfit(list(eq1 = eq1, eq2 = eq2), data = df, method = "SUR")

coefs1 <- coef(model$eq[[1]])
coefs2 <- coef(model$eq[[2]])
se1 <- summary(model$eq[[1]])$coefficients[, "Std. Error"]
se2 <- summary(model$eq[[2]])$coefficients[, "Std. Error"]

result <- list(
  coefficients = list(
    "value:Intercept" = as.numeric(coefs1["(Intercept)"]),
    "value:inv" = as.numeric(coefs1["inv"]),
    "value:capital" = as.numeric(coefs1["capital"]),
    "inv:Intercept" = as.numeric(coefs2["(Intercept)"]),
    "inv:value" = as.numeric(coefs2["value"]),
    "inv:capital" = as.numeric(coefs2["capital"])
  ),
  standard_errors = list(
    "value:Intercept" = as.numeric(se1["(Intercept)"]),
    "value:inv" = as.numeric(se1["inv"]),
    "value:capital" = as.numeric(se1["capital"]),
    "inv:Intercept" = as.numeric(se2["(Intercept)"]),
    "inv:value" = as.numeric(se2["value"]),
    "inv:capital" = as.numeric(se2["capital"])
  )
)

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
