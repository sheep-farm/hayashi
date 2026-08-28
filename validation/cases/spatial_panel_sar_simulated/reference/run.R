df <- read.csv(file.path("validation", "cases", "spatial_panel_sar_simulated", "data", "data.csv"))
W <- as.matrix(read.csv(file.path("validation", "cases", "spatial_panel_sar_simulated", "data", "W.csv"), header = FALSE))

suppressPackageStartupMessages(library(splm))
suppressPackageStartupMessages(library(spdep))
suppressPackageStartupMessages(library(jsonlite))

listw <- mat2listw(W, style = "W")
fit <- spml(y ~ x, data = df, index = c("entity", "time"), listw = listw,
            model = "within", lag = TRUE, spatial.error = "none")

rho <- as.numeric(fit$arcoef[1])
rho_se <- as.numeric(sqrt(fit$vcov.arcoef[1, 1]))
beta <- as.numeric(fit$coefficients["x"])
beta_se <- as.numeric(sqrt(fit$vcov["x", "x"]))

result <- list(
  coefficients = list(rho = rho, x = beta),
  standard_errors = list(rho = rho_se, x = beta_se)
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
