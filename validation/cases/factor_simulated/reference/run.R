options(warn=-1)
library(jsonlite)

case_dir <- "validation/cases/factor_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

eig <- eigen(cor(df))$values
eig <- sort(eig, decreasing = TRUE)

result <- list(
  coefficients = list(
    eigen_1 = as.numeric(eig[1]),
    eigen_2 = as.numeric(eig[2])
  ),
  standard_errors = list(
    eigen_1 = 0,
    eigen_2 = 0
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
