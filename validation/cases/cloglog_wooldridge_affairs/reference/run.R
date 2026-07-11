# Reference implementation in R for complementary log-log on Wooldridge affairs.

library(wooldridge)
library(jsonlite)

data(affairs)

case_dir <- "validation/cases/cloglog_wooldridge_affairs"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(affairs, file.path(data_dir, "affairs.csv"), row.names = FALSE)

model <- glm(affair ~ age + yrsmarr + kids + educ + relig + ratemarr,
             data = affairs, family = binomial(link = "cloglog"))

coefs <- coef(model)
bse <- summary(model)$coefficients[, "Std. Error"]

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(bse)
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
