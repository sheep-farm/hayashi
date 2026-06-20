# egen: extensões de generate — rowmean, rank, cumsum, group, rowsum
# Dataset: auto.dta — Stata Press R9 (n=74)

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# ── rowmean: média de peso e comprimento (proxy de "tamanho médio")
generate auto size_proxy = rowmean(weight, length)
list(auto, 8, weight, length, size_proxy)

# ── rowsum: soma de atributos numéricos por linha
generate auto rep_trunk = rowsum(rep78, trunk)
list(auto, 8, rep78, trunk, rep_trunk)

# ── rowmin / rowmax: valores extremos por linha
generate auto best_attr  = rowmin(mpg, rep78)
generate auto worst_attr = rowmax(weight, turn)
list(auto, 8, mpg, rep78, weight, turn, best_attr, worst_attr)

# ── rank: classificação de preço (1 = mais barato)
generate auto price_rank = rank(price)
# ordena para visualizar ranking
let auto_sorted = sort(auto, price)
list(auto_sorted, 10, price, price_rank)

# ── cumsum: soma acumulada de mpg após ordenar por eficiência
let auto_eff = sort(auto, mpg)
generate auto_eff cum_mpg = cumsum(mpg)
list(auto_eff, 10, mpg, cum_mpg)

# ── group: identifica grupos únicos de foreign (0=nacional, 1=estrangeiro)
generate auto grp_foreign = group(foreign)
# verifica que grp_foreign == foreign + 1 (grupos 1 e 2 em vez de 0 e 1)
list(auto, 6, foreign, grp_foreign)

# ── combinação: rank dentro de grupo (desvio de rank por origem)
generate auto mpg_rank = rank(mpg)
let por_origem = collapse(auto, mean, mpg_rank, by=foreign)
print(por_origem)
