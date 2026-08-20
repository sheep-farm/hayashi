library(jsonlite)
case_dir <- "validation/cases/estat_classification_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
fit <- glm(y ~ x, data = df, family = binomial)
p <- fitted(fit)
pred <- as.integer(p >= 0.5)
y <- df$y
tp <- sum(pred == 1 & y == 1)
tn <- sum(pred == 0 & y == 0)
fp <- sum(pred == 1 & y == 0)
fn <- sum(pred == 0 & y == 1)
sens <- tp / (tp + fn)
spec <- tn / (tn + fp)
acc <- (tp + tn) / (tp + tn + fp + fn)
cat(toJSON(list(
  coefficients = list(sensitivity = sens, specificity = spec, correct_rate = acc),
  standard_errors = list(sensitivity = 0, specificity = 0, correct_rate = 0)
), auto_unbox = TRUE, digits = 15))
