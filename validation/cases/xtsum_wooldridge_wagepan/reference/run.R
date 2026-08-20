# Reference implementation in R for the Wooldridge wagepan xtsum case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/xtsum_wooldridge_wagepan/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wagepan)

# Write CSV for Hayashi to read.
write.csv(wagepan, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

y <- wagepan$lwage
id <- wagepan$nr
t <- wagepan$year

n_total <- length(y)
n_entities <- length(unique(id))

# Overall stats.
overall_mean <- mean(y)
overall_sd <- sd(y)
overall_min <- min(y)
overall_max <- max(y)

# Between stats: entity means.
entity_means <- tapply(y, id, mean)
between_mean <- mean(entity_means)
between_sd <- sd(entity_means)
between_min <- min(entity_means)
between_max <- max(entity_means)

# Within stats: deviations from entity means.
y_within <- y - entity_means[as.character(id)]
within_mean <- mean(y_within)
within_sd <- sd(y_within)
within_min <- min(y_within)
within_max <- max(y_within)

result <- list(
  variable = rep("lwage", 3),
  type = c("overall", "between", "within"),
  n = c(n_total, n_entities, n_total),
  mean = c(overall_mean, between_mean, within_mean),
  sd = c(overall_sd, between_sd, within_sd),
  min = c(overall_min, between_min, within_min),
  max = c(overall_max, between_max, within_max)
)

out_dir <- "validation/cases/xtsum_wooldridge_wagepan/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
