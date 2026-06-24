# Inspeção de dados: sort, list, count
# Dataset: auto.dta — Stata Press R9 (n=74)

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# list: primeiras 8 linhas, colunas selecionadas
list(auto, 8, price, mpg, weight, foreign)

# sort: ordena por preço decrescente
let auto = sort(auto, price, desc=true)
list(auto, 5, price, mpg, weight)

# sort: múltiplas chaves — origem e depois preço
let auto = sort(auto, foreign, price)
list(auto, 10, price, mpg, foreign)

# count: total de observações
count auto

# count: observações que satisfazem condição
count auto if foreign == 1
count auto if price > 5000
count auto if mpg > 25

# sort ascendente por mpg e lista os mais eficientes
let auto = sort(auto, mpg, desc=true)
list(auto, 5, mpg, price, weight, foreign)
