# Reference implementation in R for Wooldridge wagepan GEE, Chapter 14.

library(wooldridge)
library(geepack)
library(jsonlite)

data(wagepan)

# Ensure the data directory exists.
data_dir <- "validation/cases/gee_wooldridge_wagepan_ch14/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Keep only observations with complete model variables.
wagepan_clean <- wagepan[complete.cases(wagepan[, c("lwage", "union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87", "nr", "year")]), ]

# Write CSV for Hayashi to read.
write.csv(wagepan_clean, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

# Estimate GEE with independent working correlation.
model <- geeglm(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87,
                id = nr,
                data = wagepan_clean,
                family = gaussian,
                corstr = "independence")

# Extract comparable quantities.
coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))
std_errors <- as.numeric(summary(model)$coefficients[, "Std.err"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors),
  nobs = nrow(wagepan_clean)
)

out_dir <- "validation/cases/gee_wooldridge_wagepan_ch14/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
