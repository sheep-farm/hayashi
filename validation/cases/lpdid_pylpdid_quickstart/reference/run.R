library(jsonlite)

case_dir <- "validation/cases/lpdid_pylpdid_quickstart"
df <- read.csv(file.path(case_dir, "data", "panel.csv"))

df$rel <- ifelse(df$g > 0, df$t - df$g, NA)
pre <- c(-5, -4, -3, -2)
post <- c(0, 1, 2, 3, 4, 5, 6, 7, 8)

varnames <- c(paste0("h_n", -pre), paste0("h_", post))
for (k in c(pre, post)) {
  if (k < 0) {
    name <- paste0("h_n", -k)
  } else {
    name <- paste0("h_", k)
  }
  df[[name]] <- as.integer(!is.na(df$rel) & df$rel == k)
}

rhs <- paste(varnames, collapse = " + ")
form <- as.formula(paste("y ~ factor(id) + factor(t) +", rhs))
fit <- lm(form, data = df)
s <- summary(fit)$coefficients
rows <- rownames(s)
event_rows <- rows[grepl("^h_", rows)]

coefs <- as.list(s[event_rows, 1])
ses <- as.list(s[event_rows, 2])

map_name <- function(nm) {
  if (grepl("^h_n", nm)) {
    paste0("h=-", gsub("^h_n", "", nm))
  } else if (grepl("^h_", nm)) {
    paste0("h=", gsub("^h_", "", nm))
  } else {
    nm
  }
}
names(coefs) <- vapply(event_rows, map_name, character(1))
names(ses) <- names(coefs)

coefs[["h=-1"]] <- 0.0
ses[["h=-1"]] <- 0.0

result <- list(
  coefficients = coefs,
  standard_errors = ses,
  n_obs = nrow(df)
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
