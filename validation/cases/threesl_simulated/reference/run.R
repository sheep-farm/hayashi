library(jsonlite)
library(systemfit)

case_dir <- "validation/cases/threesl_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

eq1 <- y1 ~ x1 + y2
eq2 <- y2 ~ x2 + y1
inst <- ~ x1 + x2

fit <- systemfit(list(y1 = eq1, y2 = eq2), method = "3SLS", inst = inst, data = df)

get_eq <- function(idx) {
  cobj <- fit$eq[[idx]]
  c <- as.numeric(coef(cobj))
  names(c) <- names(coef(cobj))
  s <- summary(fit)$eq[[idx]]$coefficients
  se <- as.numeric(s[, 2])
  names(se) <- rownames(s)
  list(c = c, se = se)
}

e1 <- get_eq(1)
e2 <- get_eq(2)

result <- list(
  coefficients = list(
    "y1:const" = e1$c["(Intercept)"],
    "y1:x1"    = e1$c["x1"],
    "y1:y2"    = e1$c["y2"],
    "y2:const" = e2$c["(Intercept)"],
    "y2:x2"    = e2$c["x2"],
    "y2:y1"    = e2$c["y1"]
  ),
  standard_errors = list(
    "y1:const" = e1$se["(Intercept)"],
    "y1:x1"    = e1$se["x1"],
    "y1:y2"    = e1$se["y2"],
    "y2:const" = e2$se["(Intercept)"],
    "y2:x2"    = e2$se["x2"],
    "y2:y1"    = e2$se["y1"]
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
