library(jsonlite)

case_dir <- "validation/cases/tvp_simulated"
raw <- readLines(file.path(case_dir, "data", "true_final.json"), warn = FALSE)
raw <- gsub("NaN", "0", raw)
result <- fromJSON(raw)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
