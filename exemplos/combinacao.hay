# Combinação de datasets: append e merge
# Dataset: auto.dta — Stata Press R9 (n=74)

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# ── MERGE ────────────────────────────────────────────────────────────────
# Caso clássico: une estatísticas de grupo de volta ao painel individual
# Útil para construir variáveis de desvio em relação à média do grupo

let media_grupo = collapse(auto, mean, price, mpg, weight, by=foreign)
let media_grupo = rename(media_grupo, price,  preco_medio)
let media_grupo = rename(media_grupo, mpg,    mpg_medio)
let media_grupo = rename(media_grupo, weight, peso_medio)

let auto = merge(auto, media_grupo, key=foreign)

# desvio individual em relação à média do grupo
generate auto desvio_preco = price - preco_medio
generate auto desvio_mpg   = mpg   - mpg_medio

list(auto, 8, foreign, price, preco_medio, desvio_preco)
summarize(auto, desvio_preco, desvio_mpg)

# ── APPEND ────────────────────────────────────────────────────────────────
# Exemplo: empilha carros nacionais (foreign=0) com estrangeiros (foreign=1)
# Na prática, seriam arquivos de fontes distintas
load "https://www.stata-press.com/data/r9/auto.dta" as lote1
load "https://www.stata-press.com/data/r9/auto.dta" as lote2

let lote1 = keep(lote1, price, mpg, weight, foreign)
let lote2 = keep(lote2, price, mpg, weight, foreign)

let painel = append(lote1, lote2)
count painel
summarize(painel, price, mpg)
