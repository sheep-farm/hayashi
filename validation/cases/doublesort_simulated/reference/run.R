options(warn=-1)
library(jsonlite)
library(dplyr)

case_dir <- "validation/cases/doublesort_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

df <- df %>%
  mutate(size_q = ntile(size, 5), bm_q = ntile(bm, 5)) %>%
  group_by(size_q, bm_q) %>%
  summarise(mean_ret = mean(ret), .groups = "drop")

val <- df %>% filter(size_q == 1, bm_q == 5) %>% pull(mean_ret)

result <- list(
  coefficients = list(
    low_size_high_bm = as.numeric(val)
  ),
  standard_errors = list(
    low_size_high_bm = 0
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
