# Testes de hipótese e diagnósticos pós-estimação
# hausman, testparm, glsar, ttest, anova

load "https://www.stata-press.com/data/r9/auto.dta" as auto
load "painel.csv" as painel

# ══════════════════════════════════════════════════════════════════════════════
# 1. HAUSMAN TEST (FE vs RE)
# ══════════════════════════════════════════════════════════════════════════════
# H0: efeitos individuais não correlacionados com regressores → RE é consistente
# H1: correlação existe → usar FE (RE seria inconsistente)
# χ²(k) = (β_FE - β_RE)' [Var(β_FE) - Var(β_RE)]⁻¹ (β_FE - β_RE)

let m_fe = fe(lucro ~ alavancagem + tamanho, painel, id=empresa, time=ano)
let m_re = re(lucro ~ alavancagem + tamanho, painel, id=empresa, time=ano)

# p < 0.05 → rejeita RE → usar FE
# p ≥ 0.05 → RE é preferível (mais eficiente)
hausman(m_fe, m_re)

# ══════════════════════════════════════════════════════════════════════════════
# 2. TESTPARM — Wald F-test conjunto (OLS)
# ══════════════════════════════════════════════════════════════════════════════
# H0: β_x1 = β_x2 = ... = 0 (todos os coeficientes listados = 0 simultaneamente)
# Complementa o F-geral: testa subconjunto específico de variáveis

let m_ols = ols(price ~ mpg + weight + length + C(foreign), auto, cov=nonrobust)

# Teste se mpg e weight são conjuntamente insignificantes
testparm(m_ols, ["mpg", "weight"])

# Teste de significância individual via testparm
testparm(m_ols, ["mpg"])

# ══════════════════════════════════════════════════════════════════════════════
# 3. GLSAR — GLS com Erros AR(p) (Cochrane-Orcutt / Prais-Winsten)
# ══════════════════════════════════════════════════════════════════════════════
# Corrige autocorrelação serial dos erros em séries temporais
# Algoritmo: estima ρ → transforma y e X → OLS → itera até convergência
#
# ar=1  → AR(1): ε_t = ρ·ε_{t-1} + u_t  (Cochrane-Orcutt padrão)
# ar=2  → AR(2): inclui dois lags
# iter= → máximo de iterações (padrão: 50)

load "serie.csv" as serie
# Variáveis: pib, consumo, investimento, exportacoes

# OLS como referência (provavelmente com autocorrelação nos resíduos)
let m_ols_ts = ols(pib ~ consumo + investimento, serie, cov=nonrobust)
print(m_ols_ts)

# GLSAR(1) — Cochrane-Orcutt
let m_ar1 = glsar(pib ~ consumo + investimento, serie, ar=1)
print(m_ar1)
# Output inclui ρ estimado e número de iterações

# GLSAR(2) — AR(2) nos resíduos
let m_ar2 = glsar(pib ~ consumo + investimento, serie, ar=2)
print(m_ar2)

# Comparação: SE do GLSAR devem ser mais precisos que OLS com AC
esttab(m_ols_ts, m_ar1, m_ar2)

# Predição pós-GLSAR
predict serie yhat_ar1  = m_ar1, "xb"
predict serie ehat_ar1  = m_ar1, "residuals"
summarize(serie, pib, yhat_ar1, ehat_ar1)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TTEST — testes t
# ══════════════════════════════════════════════════════════════════════════════

# Uni-amostral: H0: preço médio = 6000
ttest(auto, price, mu=6000)

# Dois grupos: H0: preço médio = entre domésticos e importados
ttest(auto, price, by="foreign")

# Dois grupos com variável numérica (0/1)
generate auto nacional = (foreign == 0)
ttest(auto, mpg, by="nacional")

# Pareado: H0: preço = peso (apenas por curiosidade sintática)
# ttest(auto, price, mpg, paired=true)

# ══════════════════════════════════════════════════════════════════════════════
# 5. ANOVA one-way
# ══════════════════════════════════════════════════════════════════════════════
# H0: μ_1 = μ_2 = ... = μ_k (todas as médias de grupo iguais)
# F = MS_entre / MS_dentro  ~  F(k-1, n-k) sob H0

# Qualidade de reparo (rep78: 1-5) → afeta preço?
let auto2 = dropna(auto, rep78)
anova(auto2, price, by="rep78")

# Interpretação: p < 0.05 → pelo menos um grupo tem preço médio diferente
# Para saber QUAIS pares diferem: usar ttest com by= par a par
