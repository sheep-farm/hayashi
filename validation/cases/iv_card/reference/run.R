# Reference implementation in R for the IV Card returns-to-schooling case.

library(wooldridge)
library(jsonlite)

data(card)

# Ensure the data directory exists.
data_dir <- "validation/cases/iv_card/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Keep only complete observations for the variables used.
card <- card[complete.cases(card[, c("lwage", "educ", "exper", "expersq", "black", "south", "smsa", "nearc4")]), ]

# Write CSV for Hayashi to read.
write.csv(card, file.path(data_dir, "card.csv"), row.names = FALSE)

# Estimate 2SLS: lwage ~ educ + controls, instrumented by nearc4.
model <- AER::ivreg(lwage ~ educ + exper + expersq + black + south + smsa | nearc4 + exper + expersq + black + south + smsa, data = card)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/iv_card/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
