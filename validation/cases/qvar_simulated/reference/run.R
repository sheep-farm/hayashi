library(jsonlite)
library(quantreg)

case_dir <- "validation/cases/qvar_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

n <- nrow(df)
Y <- df[-1, ]
Ylag <- df[-n, ]

coef_list <- list()

for (v in c("y1", "y2")) {
  eq_data <- data.frame(
    y = Y[[v]],
    const = 1,
    L1.y1 = Ylag[["y1"]],
    L1.y2 = Ylag[["y2"]]
  )
  fit <- rq(y ~ -1 + const + L1.y1 + L1.y2, data = eq_data, tau = 0.5)
  b <- as.numeric(coef(fit))
  names(b) <- c("const", "L1.y1", "L1.y2")
  for (p in names(b)) {
    coef_list[[paste0(v, "_", p)]] <- b[[p]]
  }
}

result <- list(coefficients = coef_list)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
