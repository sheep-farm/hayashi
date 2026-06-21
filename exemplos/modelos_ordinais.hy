# Modelos de escolha discreta: Ordered Logit/Probit e Multinomial Logit
# Dataset: auto.dta — Stata Press (n=74)
# Variáveis: price, mpg, weight, rep78, foreign, ...
# rep78: registro de qualidade de reparo, ordinal 1–5

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# rep78 tem missing (. em Stata) — filtrar antes de estimar
let auto = dropna(auto, rep78)

summarize(auto, rep78, mpg, weight, foreign)
tabulate(auto, rep78)

# ── Ordered Logit ──────────────────────────────────────────────────────────────
# P(rep78 ≤ j | X) = σ(α_j − X'β)  com limiares α₁ < α₂ < α₃ < α₄
let m_ologit = ologit(rep78 ~ mpg + weight + foreign, auto)
print(m_ologit)

# ── Ordered Probit ─────────────────────────────────────────────────────────────
# Idêntico mas com Φ no lugar de σ — normalmente resulta em coeficientes menores
let m_oprobit = oprobit(rep78 ~ mpg + weight + foreign, auto)
print(m_oprobit)

# ── Multinomial Logit ──────────────────────────────────────────────────────────
# Trata rep78 como nominal (ignora ordenação)
# Categoria base: menor valor
# Interprete odds ratios exp(β) como razão de prob categoria j vs base
let m_mlogit = mlogit(rep78 ~ mpg + weight + foreign, auto)
print(m_mlogit)

# ── Comparação ordered ─────────────────────────────────────────────────────────
esttab(m_ologit, m_oprobit)

# ── Efeitos marginais para cada categoria ──────────────────────────────────────
# AME_k(Y=j) = (1/n) Σ_i [f(κ_{j-1} - X_iβ) - f(κ_j - X_iβ)] · β_k
# Sinal oposto entre categoria 1 (lowest) e categoria J (highest)
margins(m_ologit)
margins(m_oprobit)
