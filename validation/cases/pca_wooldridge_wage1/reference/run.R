# R reference for standardised PCA on Wooldridge wage1.

library(wooldridge)
library(jsonlite)

data(wage1)
vars <- c("educ", "exper", "tenure", "wage")
df <- wage1[complete.cases(wage1[, vars]), vars]

data_dir <- "validation/cases/pca_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(df, file.path(data_dir, "wage1.csv"), row.names = FALSE)

fit <- prcomp(df, center = TRUE, scale. = TRUE, rank. = 2)
eigenvalues <- fit$sdev^2
ratios <- eigenvalues / sum(fit$sdev^2)
loadings <- sweep(fit$rotation[, seq_len(2), drop = FALSE], 2, fit$sdev[seq_len(2)], "*")

result <- list(
  explained_variance = as.list(setNames(eigenvalues[seq_len(2)], paste0("PC", seq_len(2)))),
  explained_variance_ratio = as.list(setNames(ratios[seq_len(2)], paste0("PC", seq_len(2)))),
  absolute_loadings = as.list(setNames(
    as.vector(abs(loadings)),
    as.vector(outer(rownames(loadings), paste0("PC", seq_len(2)), paste, sep = ":"))
  ))
)

out_dir <- "validation/cases/pca_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
