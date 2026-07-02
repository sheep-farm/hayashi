# Reference implementation in R for the Arellano-Bond Grunfeld case.

library(plm)
library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/ab_grunfeld/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "grunfeld.csv")
if (!file.exists(csv_path)) {
  # Fallback to the panel_fe_grunfeld dataset if this case has not downloaded yet.
  fallback <- "validation/cases/panel_fe_grunfeld/data/grunfeld.csv"
  if (file.exists(fallback)) {
    file.copy(fallback, csv_path)
  } else {
    stop("Grunfeld CSV not found. Run panel_fe_grunfeld reference first.")
  }
}

grunfeld <- read.csv(csv_path)

# Arellano-Bond difference GMM for dynamic panel investment demand.
model <- pgmm(
  inv ~ lag(inv, 1) + value + capital | lag(inv, 2:3),
  data = grunfeld,
  effect = "individual",
  model = "onestep",
  transformation = "d"
)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- rownames(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- rownames(summary_model$coefficients)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/ab_grunfeld/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
