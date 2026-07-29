library(jsonlite)
library(survival)

csv_path <- "validation/cases/km_aml/data/aml.csv"
if (!file.exists(csv_path)) {
  stop("missing generated AML CSV: run validation/cases/km_aml/data/gen.py first")
}

aml <- read.csv(csv_path)
fit <- survfit(Surv(time, event) ~ 1, data = aml)

checkpoints <- c(10, 20, 30, 40, 50, 60, 70)
surv <- summary(fit, times = checkpoints, extend = TRUE)$surv

expected <- c(
  t10 = 0.7826086956521741,
  t20 = 0.6459627329192548,
  t30 = 0.44168391994478956,
  t40 = 0.27605244996549344,
  t50 = 0.08281573498964803,
  t60 = 0.08281573498964803,
  t70 = 0.08281573498964803
)

names(surv) <- names(expected)
if (any(abs(surv - expected) > 1e-12)) {
  stop("Kaplan-Meier checkpoint values differ from fixed expected values")
}

result <- list(survival_probabilities = as.list(surv))
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
