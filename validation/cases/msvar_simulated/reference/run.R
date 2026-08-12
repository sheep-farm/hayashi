options(warn=-1)
suppressMessages(library(MSwM))
library(jsonlite)

case_dir <- "validation/cases/msvar_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

n <- nrow(df)
y1 <- df$y1
y2 <- df$y2
y1_lag <- c(NA, y1[-n])
y2_lag <- c(NA, y2[-n])

mod <- lm(y1 ~ y1_lag + y2_lag)
msm <- msmFit(mod, k = 2, sw = c(TRUE, FALSE, FALSE, TRUE), control = list(parallel = FALSE))

# Regime intercepts are in the first column of Coef; sort by magnitude so mu0 < mu1
intercepts <- as.numeric(msm@Coef[, 1])
ord <- order(intercepts)
low <- ord[1]
high <- ord[2]
mu0_y1 <- intercepts[low]
mu1_y1 <- intercepts[high]

tm <- msm@transMat
p00 <- as.numeric(tm[low, low])
p01 <- as.numeric(tm[high, low])
p10 <- as.numeric(tm[low, high])
p11 <- as.numeric(tm[high, high])

result <- list(
  coefficients = list(
    mu0_y1 = mu0_y1,
    mu1_y1 = mu1_y1,
    p00 = p00,
    p01 = p01,
    p10 = p10,
    p11 = p11
  ),
  standard_errors = list(
    mu0_y1 = 0,
    mu1_y1 = 0,
    p00 = 0,
    p01 = 0,
    p10 = 0,
    p11 = 0
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
