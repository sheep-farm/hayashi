# Dataset: auto.dta — Stata Press (n=74)
# Fonte: https://www.stata-press.com/data/r9/auto.dta
# Variáveis: price, mpg, weight, length, foreign, ...

load "https://www.stata-press.com/data/r9/auto.dta" as auto

summarize(auto, price, mpg, weight)
correlate(auto, price, mpg, weight)

# OLS com erros padrão robustos
let m = ols(price ~ mpg + weight, auto, cov=HC3)
print(m)

test(m, "bp")
test(m, "dw")

# Valores ajustados e resíduos
predict auto yhat = m
predict auto ehat = m, "residuals"

summarize(auto, price, yhat, ehat)

# Transformações
generate auto log_price  = log(price)
generate auto weight_ton = weight / 2000

# Exportar dados e tabela
export(auto, "csv", "/tmp/auto.csv")
export(m,    "latex", "/tmp/auto_ols.tex")
