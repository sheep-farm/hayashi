# Reference implementation in R for the Wooldridge wage1 one-way ANOVA case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/anova_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wage1)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

# One-way ANOVA of wage by educ.
fit <- aov(wage ~ as.factor(educ), data = wage1)
s <- summary(fit)[[1]]

result <- list(
  test = "One-Way ANOVA",
  ss_between = s$`Sum Sq`[1],
  ss_within = s$`Sum Sq`[2],
  ss_total = sum(s$`Sum Sq`, na.rm = TRUE),
  df_between = s$Df[1],
  df_within = s$Df[2],
  ms_between = s$`Mean Sq`[1],
  ms_within = s$`Mean Sq`[2],
  f_stat = s$`F value`[1],
  p_value = s$`Pr(>F)`[1],
  n_groups = length(unique(wage1$educ)),
  n_obs = nrow(wage1)
)

out_dir <- "validation/cases/anova_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
