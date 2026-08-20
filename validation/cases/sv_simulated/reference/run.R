options(warn=-1)
suppressMessages(library(stochvol))
library(jsonlite)

case_dir <- "validation/cases/sv_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

set.seed(42)
res <- svsample(df$y, draws = 1000, burnin = 500, quiet = TRUE)

latent <- as.matrix(res$latent[[1]])
mean_ht <- as.numeric(mean(latent))

result <- list(
  coefficients = list(mean_ht = mean_ht),
  standard_errors = list(mean_ht = 0)
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
