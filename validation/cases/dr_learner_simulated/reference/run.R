library(jsonlite)

if (!requireNamespace("gbm", quietly = TRUE)) {
  install.packages("gbm", repos = "https://cloud.r-project.org")
}
library(gbm)

set.seed(42)

case_dir <- "validation/cases/dr_learner_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

y <- df$y
d <- df$d
x <- df$x

# propensity score
ps <- glm(d ~ x, data = df, family = binomial(link = "logit"))
phat <- predict(ps, type = "response")

# outcome model E[y | d, x]
out_mod <- gbm(y ~ d + x, data = df, distribution = "gaussian",
               n.trees = 100, interaction.depth = 3, shrinkage = 0.1,
               bag.fraction = 1, n.minobsinnode = 10, keep.data = TRUE,
               cv.folds = 0, verbose = FALSE)

new1 <- df; new1$d <- 1
new0 <- df; new0$d <- 0
mu1 <- predict(out_mod, newdata = new1, n.trees = 100, type = "response")
mu0 <- predict(out_mod, newdata = new0, n.trees = 100, type = "response")

aipw <- (mu1 - mu0) + d * (y - mu1) / phat - (1 - d) * (y - mu0) / (1 - phat)
ate <- mean(aipw)
ate_se <- sd(aipw) / sqrt(length(aipw))

result <- list(
  coefficients = list(ate = as.numeric(ate)),
  standard_errors = list(ate = as.numeric(ate_se))
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
