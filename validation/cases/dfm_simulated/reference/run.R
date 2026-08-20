library(jsonlite)

case_dir <- "validation/cases/dfm_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

# Static factor analysis with two factors.
fit <- factanal(df, factors = 2, scores = "regression")

# Communality is the fraction of each (standardised) variable's variance
# explained by the common factors.
comm <- 1 - fit$uniquenesses
names(comm) <- colnames(df)

result <- list(
  coefficients = as.list(comm),
  standard_errors = as.list(setNames(rep(NaN, length(comm)), names(comm)))
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
