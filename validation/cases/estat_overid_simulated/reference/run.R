case_dir <- "validation/cases/estat_overid_simulated"
library(jsonlite)
df <- read.csv(file.path(case_dir, "data", "data.csv"))
Y <- as.matrix(df$y)
X <- cbind(1, df$x)
Z <- cbind(1, df$z1, df$z2)
n <- nrow(df)
Pi <- solve(t(Z) %*% Z) %*% t(Z) %*% X
Xhat <- Z %*% Pi
b2sls <- solve(t(Xhat) %*% X) %*% t(Xhat) %*% Y
u <- as.vector(Y - X %*% b2sls)
Pz <- Z %*% solve(t(Z) %*% Z) %*% t(Z)
stat <- as.numeric(n * (t(u) %*% Pz %*% u) / (t(u) %*% u))
pval <- as.numeric(1 - pchisq(stat, 1))
cat(toJSON(list(coefficients = list(j_stat = stat, p_value = pval), standard_errors = list(j_stat = 0, p_value = 0)), auto_unbox = TRUE, digits = 15))