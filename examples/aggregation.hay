# Agregação: collapse com múltiplas funções
# Dataset: auto.dta — Stata Press R9 (n=74)

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# médias por origem (0=nacional, 1=estrangeiro)
let por_origem = collapse(auto, mean, price, mpg, weight, by=foreign)
print(por_origem)

# desvio padrão — dispersão de preço por origem
let sd_origem = collapse(auto, sd, price, mpg, by=foreign)
print(sd_origem)

# contagem de modelos por origem
let n_origem = collapse(auto, count, price, by=foreign)
print(n_origem)

# mínimo e máximo
let range_origem = collapse(auto, min, price, mpg, by=foreign)
print(range_origem)

let range_origem = collapse(auto, max, price, mpg, by=foreign)
print(range_origem)

# mediana de preço por origem
let med_origem = collapse(auto, median, price, mpg, by=foreign)
print(med_origem)

# ── análise mais profunda ──────────────────────────────────────────────────
# Dataset: cancer.dta — por tipo de tratamento
load "https://www.stata-press.com/data/r8/cancer.dta" as cancer

let por_droga = collapse(cancer, mean, studytime, died, age, by=drug)
print(por_droga)

let n_droga = collapse(cancer, count, studytime, by=drug)
print(n_droga)
