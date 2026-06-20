# Dataset: auto.dta — Stata Press (n=74)
# Fonte: https://www.stata-press.com/data/r9/auto.dta

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# drop: remove colunas desnecessárias
let auto = drop(auto, make, rep78, trunk, turn, displacement, gear_ratio)

summarize(auto)

# keep: mantém apenas as variáveis de interesse
let base = keep(auto, price, mpg, weight, length, foreign)

summarize(base)
correlate(base, price, mpg, weight)
