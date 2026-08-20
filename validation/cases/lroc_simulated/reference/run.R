library(jsonlite)
case_dir <- "validation/cases/lroc_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
y <- df$y
p <- 1 / (1 + exp(-2 * df$x))
n1 <- sum(y == 1)
n0 <- sum(y == 0)
r <- rank(p, ties.method = "average")
auc <- (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
gini <- 2 * auc - 1
cat(toJSON(list(coefficients = list(auc = auc, gini = gini), standard_errors = list(auc = 0, gini = 0)), auto_unbox = TRUE, digits = 15))
