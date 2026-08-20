# Reference implementation in R for ZINB on Wooldridge affairs.

library(wooldridge)
library(jsonlite)
library(pscl)

data(affairs)

case_dir <- "validation/cases/zinb_affairs"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(affairs, file.path(data_dir, "affairs.csv"), row.names = FALSE)

formula_count <- naffairs ~ age + yrsmarr + kids + educ + relig + ratemarr
formula_inflate <- ~ age + yrsmarr + kids + educ + relig + ratemarr
model <- zeroinfl(formula_count, data = affairs, dist = "negbin", link = "logit")
sm <- summary(model)

count_vars <- c("(Intercept)", "age", "yrsmarr", "kids", "educ", "relig", "ratemarr")
count_names <- c("const", "age", "yrsmarr", "kids", "educ", "relig", "ratemarr")
inflate_vars <- c("(Intercept)", "age", "yrsmarr", "kids", "educ", "relig", "ratemarr")
inflate_names <- c("const", "age", "yrsmarr", "kids", "educ", "relig", "ratemarr")

count_coefs <- coef(model, model = "count")
inflate_coefs <- coef(model, model = "zero")
count_se <- sm$coefficients$count[, "Std. Error"]
inflate_se <- sm$coefficients$zero[, "Std. Error"]

coefs <- c(setNames(as.numeric(count_coefs[count_vars]), paste0("count_", count_names)),
          setNames(as.numeric(inflate_coefs[inflate_vars]), paste0("inflate_", inflate_names)))
ses <- c(setNames(as.numeric(count_se[count_vars]), paste0("count_", count_names)),
         setNames(as.numeric(inflate_se[inflate_vars]), paste0("inflate_", inflate_names)))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(ses)
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
