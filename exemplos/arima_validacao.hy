# Validação ARIMA — Wholesale Price Index (WPI)
# Dataset: wpi1.dta — índice de preços no atacado trimestral, EUA 1960q1-1990q4
# Referência clássica do Box-Jenkins; resultado esperado próximo ao Stata

load "https://www.stata-press.com/data/r14/wpi1.dta" as wpi

# inspeciona estrutura
list(wpi, 8)
count wpi

# declara série temporal
tsset wpi t

# ── ARIMA(1,1,1) — especificação padrão para séries com tendência
# Em Stata: arima wpi, arima(1,1,1)
# Coeficientes esperados: AR≈0.7, MA≈-0.4 (aproximados)
let m111 = arima(wpi, wpi, p=1, d=1, q=1)
print(m111)

# ── previsão 8 trimestres à frente com IC 95%
forecast(m111, steps=8)

# ── ARIMA(0,1,1) — random walk com componente MA
# Equivalente ao exponential smoothing
let m011 = arima(wpi, wpi, p=0, d=1, q=1)
print(m011)

# ── ARIMA(1,1,0) — AR sobre primeiras diferenças
let m110 = arima(wpi, wpi, p=1, d=1, q=0)
print(m110)

# compara AIC/BIC implicitamente nos prints acima
