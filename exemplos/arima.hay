# ARIMA e SARIMA: estimação e previsão
# Dataset: cancer.dta — usa studytime como série temporal (fins didáticos)
# Em uso real: série macroeconômica com tsset declarado

load "https://www.stata-press.com/data/r8/cancer.dta" as cancer

# ── ordena por tempo antes de estimar
tsset cancer studytime

# ── ARIMA(1,1,1)
let m_arima = arima(cancer, studytime, p=1, d=1, q=1)
print(m_arima)

# ── previsão 8 passos à frente com IC 95%
forecast(m_arima, steps=8)

# ── previsão 12 passos com IC 90%
forecast(m_arima, steps=12, alpha=0.10)

# ── ARIMA(2,1,0) — AR puro
let m_ar2 = arima(cancer, studytime, p=2, d=1, q=0)
print(m_ar2)
forecast(m_ar2, steps=6)

# ── ARIMA(0,1,1) — MA puro (random walk + MA)
let m_ma1 = arima(cancer, studytime, p=0, d=1, q=1)
print(m_ma1)

# ── SARIMA(1,1,1)(1,0,1,4) — sazonalidade trimestral
let m_sarima = sarima(cancer, studytime, p=1, d=1, q=1, P=1, D=0, Q=1, s=4)
print(m_sarima)
forecast(m_sarima, steps=8)
