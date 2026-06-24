# Testes t: uni-amostral, dois grupos, pareado
# Datasets: auto.dta e cancer.dta

load "https://www.stata-press.com/data/r9/auto.dta" as auto
load "https://www.stata-press.com/data/r8/cancer.dta" as cancer

# ── UNI-AMOSTRAL ─────────────────────────────────────────────────────────
# H0: preço médio = 6000
ttest(auto, price, mu=6000)

# H0: eficiência média = 21 mpg
ttest(auto, mpg, mu=21)

# H0: probabilidade de óbito = 0.5
ttest(cancer, died, mu=0.5)

# ── DOIS GRUPOS ───────────────────────────────────────────────────────────
# H0: preço médio igual entre nacionais e estrangeiros (Welch)
ttest(auto, price, by=foreign)

# H0: mpg igual entre nacionais e estrangeiros
ttest(auto, mpg, by=foreign)

# H0: semanas de sobrevivência iguais entre tratamentos
#     (drug: 1=placebo, 2=droga1, 3=droga2 — aqui comparando drug vs died)
ttest(cancer, studytime, by=died)

# ── PAREADO ───────────────────────────────────────────────────────────────
# Simula teste pareado: compara peso vs comprimento (escala similar, só demo)
# Na prática: antes vs depois de uma intervenção
generate auto price_norm  = price  / 1000
generate auto weight_norm = weight / 1000
ttest(auto, price_norm, weight_norm, paired=true)
