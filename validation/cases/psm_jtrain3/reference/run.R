# Reference implementation in R for PSM on Wooldridge jtrain3.
#
# 1:1 nearest-neighbor propensity score matching with caliper and bootstrap SE.
# If the wooldridge package is unavailable, the CSV written by the Python
# reference is used.

library(jsonlite)

CASE_DIR <- "validation/cases/psm_jtrain3"
DATA_DIR <- file.path(CASE_DIR, "data")
CSV_PATH <- file.path(DATA_DIR, "jtrain3.csv")

dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)

# Prefer wooldridge if installed; otherwise rely on the CSV from Python.
if (requireNamespace("wooldridge", quietly = TRUE)) {
  data(jtrain3, package = "wooldridge")
  write.csv(jtrain3, CSV_PATH, row.names = FALSE)
} else if (!file.exists(CSV_PATH)) {
  stop("jtrain3.csv not found and R package wooldridge is not installed")
}

df <- read.csv(CSV_PATH)

outcome <- "re78"
treatment <- "train"
covariates <- c("age", "educ", "black", "hisp", "married",
                "unem74", "unem75", "re74", "re75")

# Propensity score via logit.
formula <- as.formula(paste(treatment, "~", paste(covariates, collapse = " + ")))
ps_model <- glm(formula, data = df, family = binomial(link = "logit"))
ps <- predict(ps_model, type = "response")

caliper <- 0.2 * sd(ps)

treated_idx <- which(df[[treatment]] == 1)
control_idx <- which(df[[treatment]] == 0)

ps_t <- ps[treated_idx]
ps_c <- ps[control_idx]

# 1:1 nearest-neighbor matching without replacement.
match_c <- rep(NA_integer_, length(treated_idx))
valid <- rep(FALSE, length(treated_idx))
for (i in seq_along(treated_idx)) {
  d <- abs(ps_c - ps_t[i])
  j <- which.min(d)
  if (d[j] <= caliper) {
    match_c[i] <- j
    valid[i] <- TRUE
  }
}

matched_treated <- treated_idx[valid]
matched_control <- control_idx[match_c[valid]]

att <- mean(df[[outcome]][matched_treated] - df[[outcome]][matched_control])

# Bootstrap SE.
set.seed(42)
N <- nrow(df)
B <- 200
boot_atts <- numeric(B)
for (b in seq_len(B)) {
  boot_idx <- sample(N, N, replace = TRUE)
  boot_df <- df[boot_idx, ]
  boot_ps <- predict(ps_model, newdata = boot_df, type = "response")
  boot_t <- which(boot_df[[treatment]] == 1)
  boot_c <- which(boot_df[[treatment]] == 0)
  if (length(boot_t) == 0 || length(boot_c) == 0) {
    boot_atts[b] <- NA
    next
  }
  boot_ps_t <- boot_ps[boot_t]
  boot_ps_c <- boot_ps[boot_c]
  boot_match <- rep(NA_integer_, length(boot_t))
  boot_valid <- rep(FALSE, length(boot_t))
  for (i in seq_along(boot_t)) {
    d <- abs(boot_ps_c - boot_ps_t[i])
    j <- which.min(d)
    if (d[j] <= caliper) {
      boot_match[i] <- j
      boot_valid[i] <- TRUE
    }
  }
  if (any(boot_valid)) {
    boot_yt <- boot_df[[outcome]][boot_t[boot_valid]]
    boot_yc <- boot_df[[outcome]][boot_c[boot_match[boot_valid]]]
    boot_atts[b] <- mean(boot_yt - boot_yc)
  } else {
    boot_atts[b] <- NA
  }
}

se <- sd(boot_atts, na.rm = TRUE)

result <- list(
  coefficients = list(ATT = att),
  standard_errors = list(ATT = se)
)

out_dir <- file.path(CASE_DIR, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
