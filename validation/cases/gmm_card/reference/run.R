# Reference implementation in R for the GMM card case.

library(wooldridge)
library(AER)
library(jsonlite)

data(card)

# Ensure the data directory exists.
data_dir <- "validation/cases/gmm_card/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(card, file.path(data_dir, "card.csv"), row.names = FALSE)

# GMM returns-to-schooling with nearc4 as instrument for education.
model <- ivreg(
  lwage ~ educ + exper + expersq + smsa + black + south
    | nearc4 + exper + expersq + smsa + black + south,
  data = card
)

summary_model <- summary(model)

name_map <- c(
  "(Intercept)" = "x0",
  "educ" = "x1",
  "exper" = "x2",
  "expersq" = "x3",
  "smsa" = "x4",
  "black" = "x5",
  "south" = "x6"
)

coefs <- as.numeric(coef(model))
names(coefs) <- name_map[names(coef(model))]

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- name_map[rownames(summary_model$coefficients)]

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/gmm_card/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
