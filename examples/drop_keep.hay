# Manipulação de colunas e filtragem de linhas
# Dataset: auto.dta — Stata Press (n=74)
# Fonte: https://www.stata-press.com/data/r9/auto.dta

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# drop: remove colunas desnecessárias
let auto = drop(auto, make, rep78, trunk, turn, displacement, gear_ratio)

summarize(auto)

# keep: mantém apenas as colunas de interesse (seleção de variáveis)
let base = keep(auto, price, mpg, weight, length, foreign)

summarize(base)
correlate(base, price, mpg, weight)

# ── filter: seleção de LINHAS por condição ──────────────────────────────────────
# filter(df, condição) → DataFrame com obs onde condição ≠ 0

# Filtro numérico simples
let economicos = filter(base, mpg > 25)
print(economicos)

# Filtro composto
let leves_economicos = filter(base, mpg > 20 & weight < 3000)
summarize(leves_economicos, mpg, weight, price)

# Filtro por dummy (foreign == 1 → importados)
let importados = filter(base, foreign == 1)
summarize(importados, mpg, weight, price)

# Comparar OLS em subamostras
let m_importados  = ols(price ~ mpg + weight, importados)
let nacionais     = filter(base, foreign == 0)
let m_nacionais   = ols(price ~ mpg + weight, nacionais)
esttab(m_importados, m_nacionais)
