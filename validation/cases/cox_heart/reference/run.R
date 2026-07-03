# Reference implementation in R for the Cox survival heart case.

library(survival)
library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/cox_heart/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Load the heart transplant dataset from the Rdatasets mirror.
url <- "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/statsmodels/heart.csv"
heart <- read.csv(url)
heart <- heart[, c("survival", "age", "censors")]
names(heart)[names(heart) == "survival"] <- "time"
names(heart)[names(heart) == "censors"] <- "censored"

# Write CSV for Hayashi to read.
write.csv(heart, file.path(data_dir, "heart.csv"), row.names = FALSE)

# Cox proportional hazards: survival time after heart transplant.
model <- coxph(Surv(time, censored) ~ age, data = heart)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "se(coef)"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/cox_heart/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
