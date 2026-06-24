# Operadores de série temporal: tsset, L.x, F.x, D.x
# Demonstração com auto.dta — ordena por mpg como proxy de "período"

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# ── tsset: declara variável de tempo e ordena o dataset
tsset auto mpg

# ── Lag L.: valor defasado 1 e 2 períodos
generate auto L_price   = L.price
generate auto L2_price  = L2.price
generate auto L_weight  = L.weight

# ── Lead F.: valor adiantado 1 período
generate auto F_price   = F.price

# ── Diferença D.: variação em relação ao período anterior
generate auto D_price   = D.price
generate auto D2_price  = D2.price

list(auto, 12, mpg, price, L_price, L2_price, F_price, D_price)

# ── Combinação: média móvel de 3 períodos
generate auto ma3_price = (price + L_price + L2_price) / 3
list(auto, 10, mpg, price, L_price, L2_price, ma3_price)

# ── Caso econométrico: variação de preço em função do peso defasado
# crescimento = D.price / L.price (requer L_price != 0)
generate auto growth_price = D_price / L_price
generate auto lag_weight   = L.weight

# descarta NaN da primeira observação antes de estimar
let auto_clean = dropna(auto, growth_price, lag_weight)
let m = ols(growth_price ~ lag_weight + weight, auto_clean, cov=HC3)
print(m)
