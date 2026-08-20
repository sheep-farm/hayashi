options(warn=-1)
suppressMessages(library(tmle))
library(jsonlite)

case_dir <- "validation/cases/tmle_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

Y <- df$y
A <- df$t
W <- df[, c("x1", "x2")]

res <- suppressWarnings(
  tmle(Y, A, W, family = "gaussian", Q.SL.library = "SL.glm", g.SL.library = "SL.glm")
)

ate <- as.numeric(res$estimates$ATE$psi)
se <- as.numeric(sqrt(res$estimates$ATE$var.psi))

result <- list(
  coefficients = list(ate = ate, se = se),
  standard_errors = list(ate = 0, se = 0)
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
