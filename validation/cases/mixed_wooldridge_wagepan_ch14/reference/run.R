# Reference implementation in R for Wooldridge wagepan mixed model, Chapter 14.

library(wooldridge)
library(lme4)
library(jsonlite)

data(wagepan)

# Ensure the data directory exists.
data_dir <- "validation/cases/mixed_wooldridge_wagepan_ch14/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Keep only observations with complete model variables.
wagepan_clean <- wagepan[complete.cases(wagepan[, c("lwage", "union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87", "nr", "year")]), ]

# Write CSV for Hayashi to read.
write.csv(wagepan_clean, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

# Estimate mixed linear model with random intercept by nr.
model <- lmer(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87 + (1 | nr),
              data = wagepan_clean,
              REML = TRUE)

# Extract comparable quantities (fixed effects only).
coefs <- as.numeric(fixef(model))
names(coefs) <- names(fixef(model))

summary_model <- summary(model)
std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(fixef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors),
  nobs = nrow(wagepan_clean)
)

out_dir <- "validation/cases/mixed_wooldridge_wagepan_ch14/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
