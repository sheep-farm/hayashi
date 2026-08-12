df <- read.csv(file.path("validation", "cases", "tvar_simulated", "data", "data.csv"))
suppressPackageStartupMessages(library(tsDyn))
suppressPackageStartupMessages(library(jsonlite))

Y <- as.matrix(df[, c("y1", "y2")])
q <- df$q

sink(tempfile())
fit <- TVAR(Y, lag=1, nthresh=1, thDelay=1, thVar=q, include="none")
sink()

c <- fit$model.specific$Thresh

n <- nrow(df)
y1_t <- df$y1[-1]
y2_t <- df$y2[-1]
y1_l1 <- df$y1[-n]
y2_l1 <- df$y2[-n]
q_l1 <- q[-n]

low <- q_l1 < c
high <- !low

coef_se <- function(y, x1, x2, idx) {
  dat <- data.frame(y = y[idx], x1 = x1[idx], x2 = x2[idx])
  m <- lm(y ~ x1 + x2 - 1, data = dat)
  c(coef(m)[1], coef(m)[2], sqrt(diag(vcov(m)))[1], sqrt(diag(vcov(m)))[2])
}

r1 <- coef_se(y1_t, y1_l1, y2_l1, low)
r2 <- coef_se(y2_t, y1_l1, y2_l1, low)
r3 <- coef_se(y1_t, y1_l1, y2_l1, high)
r4 <- coef_se(y2_t, y1_l1, y2_l1, high)

result <- list(
  coefficients = list(
    low_y1_L1.y1 = r1[1], low_y1_L1.y2 = r1[2],
    low_y2_L1.y1 = r2[1], low_y2_L1.y2 = r2[2],
    high_y1_L1.y1 = r3[1], high_y1_L1.y2 = r3[2],
    high_y2_L1.y1 = r4[1], high_y2_L1.y2 = r4[2]
  ),
  standard_errors = list(
    low_y1_L1.y1 = r1[3], low_y1_L1.y2 = r1[4],
    low_y2_L1.y1 = r2[3], low_y2_L1.y2 = r2[4],
    high_y1_L1.y1 = r3[3], high_y1_L1.y2 = r3[4],
    high_y2_L1.y1 = r4[3], high_y2_L1.y2 = r4[4]
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
