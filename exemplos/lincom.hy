# lincom: combinações lineares de coeficientes OLS
# Dataset: auto.dta — Stata Press R9 (n=74)

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# modelo base: preço em função de mpg, weight e headroom
let m = ols(price ~ mpg + weight + headroom, auto, cov=HC3)
print(m)

# ── testa β_mpg (equivale a H0: β_mpg = 0, IC 95%)
lincom(m, mpg=1)

# ── testa diferença de efeitos: H0: β_mpg = β_weight
# Δ = β_mpg - β_weight = 0
lincom(m, mpg=1, weight=-1)

# ── variação de preço esperada: mpg sobe 5, weight cai 100
# Δprice = 5*β_mpg + (-100)*β_weight
lincom(m, mpg=5, weight=-100)

# ── dois modelos para comparar lincom
let m2 = ols(price ~ mpg + weight, auto, cov=HC3)
print(m2)

# efeito conjunto de 10 unidades extras de mpg e peso constante
lincom(m2, mpg=10)

# combinação mpg=3, weight=-50: qual o efeito líquido?
lincom(m2, mpg=3, weight=-50)
