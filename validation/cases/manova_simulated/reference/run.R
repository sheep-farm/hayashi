options(warn=-1)
library(jsonlite)

case_dir <- "validation/cases/manova_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))
df$group <- as.factor(df$group)

m <- manova(cbind(y1, y2) ~ group, data = df)

result <- list(
  coefficients = list(
    pillai = as.numeric(summary(m, test = "Pillai")$stats[1, "Pillai"]),
    wilks = as.numeric(summary(m, test = "Wilks")$stats[1, "Wilks"]),
    lh = as.numeric(summary(m, test = "Hotelling-Lawley")$stats[1, "Hotelling-Lawley"]),
    roy = as.numeric(summary(m, test = "Roy")$stats[1, "Roy"])
  ),
  standard_errors = list(
    pillai = 0,
    wilks = 0,
    lh = 0,
    roy = 0
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
