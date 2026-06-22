# GLM — Modelos Lineares Generalizados
# influence — Diagnósticos de influência (Cook's D, leverage, DFBETAS)
# lowess / kde — Estimação não-paramétrica

# ══════════════════════════════════════════════════════════════════════════════
# 1. GLM — Interface unificada para famílias exponenciais
# ══════════════════════════════════════════════════════════════════════════════
# Famílias: gaussian, binomial, poisson, gamma, inverse_gaussian, negbin, tweedie
# Links:    identity, log, logit, probit, inverse, cloglog
#           (omitido → link canônico da família)
#
# Relação família ↔ link canônico:
#   gaussian        → identity
#   binomial        → logit
#   poisson         → log
#   gamma           → inverse
#   inverse_gaussian → inverse_squared
#   negbin          → log
#
# Quando usar GLM em vez de logit/poisson separados?
# glm(y~x, df, family=binomial) ≡ logit(y~x, df)
# glm(y~x, df, family=poisson) ≡ poisson(y~x, df)
# Use glm() quando quiser link não-canônico ou família menos comum

load "https://www.stata-press.com/data/r9/auto.dta" as auto

input saude
visitas renda doenca
0.0 3.2 0.0
3.0 4.1 1.0
0.0 2.8 0.0
1.0 3.5 0.0
5.0 4.5 1.0
2.0 3.9 1.0
0.0 2.5 0.0
4.0 4.3 1.0
1.0 3.0 0.0
0.0 2.7 0.0
6.0 4.8 1.0
0.0 3.1 0.0
2.0 3.7 1.0
1.0 3.3 0.0
3.0 4.0 1.0
0.0 2.6 0.0
7.0 5.0 1.0
1.0 3.4 0.0
4.0 4.2 1.0
0.0 2.9 0.0
end

input seguro
sinistros exposure severidade n_sinistros renda
2.0 100.0 5000.0 2.0 3.5
5.0 200.0 12000.0 5.0 4.2
1.0 80.0 3000.0 1.0 3.1
3.0 150.0 8000.0 3.0 3.8
0.0 50.0 1500.0 0.0 2.9
4.0 180.0 9500.0 4.0 4.0
2.0 120.0 5500.0 2.0 3.4
6.0 250.0 15000.0 6.0 4.5
1.0 90.0 2800.0 1.0 3.0
3.0 160.0 7500.0 3.0 3.7
7.0 300.0 18000.0 7.0 4.8
2.0 110.0 4800.0 2.0 3.3
4.0 170.0 9000.0 4.0 3.9
1.0 70.0 2500.0 1.0 2.8
5.0 220.0 13000.0 5.0 4.3
3.0 140.0 7000.0 3.0 3.6
0.0 60.0 1200.0 0.0 2.7
6.0 260.0 16000.0 6.0 4.6
2.0 130.0 5200.0 2.0 3.5
4.0 190.0 10000.0 4.0 4.1
end

# seguro: sinistros (contagem) e exposure (número apólices), severidade (custo total)
# auto: price, mpg, weight, foreign

# ── Binomial com link logit (≡ logit()) ──────────────────────────────────────
let m_glm_logit = glm(foreign ~ price + mpg + weight, auto, family=binomial)
print(m_glm_logit)

# Comparar com logit dedicado — resultados idênticos
let m_logit = logit(foreign ~ price + mpg + weight, auto)
esttab(m_glm_logit, m_logit)

# ── Binomial com link probit (link não-canônico) ─────────────────────────────
let m_glm_probit = glm(foreign ~ price + mpg + weight, auto,
                       family=binomial, link=probit)
print(m_glm_probit)

# ── Poisson com link log (≡ poisson()) ───────────────────────────────────────
let m_glm_pois = glm(visitas ~ renda + doenca, saude, family=poisson)
print(m_glm_pois)

# ── Gamma com link log (não-canônico) ────────────────────────────────────────
# Gamma é adequado para variáveis contínuas positivas com variância ∝ μ²
# Ex: custos de saúde, salários, indenizações
let m_gamma_log = glm(severidade ~ n_sinistros + exposure, seguro,
                      family=gamma, link=log)
print(m_gamma_log)

# Gamma com link canônico (inverse)
let m_gamma_inv = glm(severidade ~ n_sinistros + exposure, seguro,
                      family=gamma)
print(m_gamma_inv)

# Comparar AIC: qual link melhor ajusta os dados?
# ic(m_gamma_log, m_gamma_inv)  # ic() não suporta GLM gamma ainda

# ── NegBin via GLM (alpha como parâmetro de dispersão) ──────────────────────
# alpha: parâmetro de sobredispersão; Var(Y) = μ + α·μ²
# Aumentar alpha → mais sobredispersão (padrão: 1.0)
let m_glm_nb = glm(visitas ~ renda + doenca, saude,
                   family=negbin, alpha=1.0)
print(m_glm_nb)

# ── Tweedie (família flexível: p=1→Poisson, p=2→Gamma, 1<p<2→compound Poisson-Gamma)
let m_tw = glm(sinistros ~ exposure + renda, seguro,
               family=tweedie, power=1.5)
print(m_tw)

# ── Predições GLM ─────────────────────────────────────────────────────────────
# pr/mu/fitted → μ̂ = g⁻¹(Xβ) — resposta média (escala da variável)
# xb           → Xβ — preditor linear (escala do link)
# residuals    → resíduos de desvio (deviance residuals)
# pearson      → resíduos de Pearson (y-μ)/√V(μ)
# working      → resíduos de trabalho do IRLS

predict auto   p_hat     = m_glm_logit, "pr"        # P(foreign=1|X)
predict saude  mu_pois   = m_glm_pois,  "mu"        # E[visitas|X]
predict saude  xb_pois   = m_glm_pois,  "xb"        # log(E[visitas|X])
predict saude  dev_resid = m_glm_pois,  "residuals" # deviance residuals
predict saude  prs_resid = m_glm_pois,  "pearson"   # Pearson residuals
predict saude  wrk_resid = m_glm_pois,  "working"   # working residuals

summarize(saude, visitas, mu_pois, dev_resid, prs_resid)

# ── esttab com GLM ────────────────────────────────────────────────────────────
esttab(m_glm_logit, m_glm_probit)
esttab(m_glm_pois, m_glm_nb)

# ══════════════════════════════════════════════════════════════════════════════
# 2. INFLUENCE — Diagnósticos de influência para OLS
# ══════════════════════════════════════════════════════════════════════════════
# DFBetas : mudança em β_j quando obs i é removida; threshold = 2/√n
# DFFITS  : mudança no ŷ_i quando obs i é removida; threshold = 2√(k/n)
# Leverage: diagonal da hat matrix h_ii; alto quando > 2k/n
# Resíduos studentizados internos e externos

load "https://www.stata-press.com/data/r9/auto.dta" as auto2

let m_ols = ols(price ~ mpg + weight + length, auto2, cov=nonrobust)
print(m_ols)

# Calcular diagnósticos de influência
influence(m_ols)
# Saída: tabela com n, k, thresholds e lista de obs influentes por DFFITS
# Se influential_dffits está vazia → sem obs problemáticas

# Fluxo recomendado:
#  1. Estimar OLS
#  2. influence() → identificar obs suspeitas
#  3. Reestimar excluindo essas obs e comparar coeficientes
#  4. Se coeficientes mudarem muito → resultado não é robusto a outliers

# ── IMPORTANTE: influence() não modifica o DataFrame ─────────────────────────
# Os índices retornados são 0-based (posição na ordem de carregamento)
# Para ver a obs: filtrar o df pelo índice (via for ou replace)

# ══════════════════════════════════════════════════════════════════════════════
# 3. LOWESS — Suavização não-paramétrica local (Cleveland, 1979)
# ══════════════════════════════════════════════════════════════════════════════
# Ajusta uma linha de regressão local ponderada em cada ponto
# frac: fração dos dados usada em cada ajuste (0 < frac ≤ 1)
#       menor frac → curva mais flexível (mais ruído)
#       maior frac → curva mais suave (mais viés)
# it: iterações de robustificação (reduz influência de outliers)
#     it=0 → sem robustificação (mais rápido)
#     it=3 → padrão (mais robusto a outliers)
#
# Uso típico:
#  1. Explorar relação entre Y e X antes de especificar forma funcional
#  2. Verificar se relação é linear ou não-linear
#  3. Visualização residual: LOWESS(resíduos, ŷ) — deve ser plana se bem especificado

input macro
pib_growth inflation unemployment
2.1 3.5 5.2
1.8 4.0 5.8
3.2 2.8 4.5
0.5 5.1 6.3
2.5 3.2 5.0
1.2 4.5 6.0
3.0 2.5 4.2
0.8 4.8 6.5
2.8 3.0 4.8
1.5 4.2 5.5
3.5 2.2 4.0
0.3 5.5 7.0
2.0 3.8 5.3
1.0 4.6 6.2
2.7 3.1 4.9
1.6 4.3 5.7
3.3 2.6 4.3
0.6 5.0 6.8
2.3 3.4 5.1
1.9 4.1 5.6
end
# Variáveis: pib_growth, inflation, unemployment

# LOWESS simples: inflation vs pib_growth
let m_low = lowess(macro, pib_growth, inflation, frac=0.5)

# Extrair valores suavizados
predict macro yhat_low = m_low, "smoothed"
predict macro resid_low = m_low, "residuals"
summarize(macro, pib_growth, yhat_low, resid_low)

# LOWESS com mais suavização (frac maior)
let m_low_smooth = lowess(macro, pib_growth, inflation, frac=0.8, it=0)
predict macro yhat_smooth = m_low_smooth, "smoothed"

# LOWESS vs OLS — comparar resíduos
let m_ols_macro = ols(pib_growth ~ inflation, macro, cov=nonrobust)
predict macro yhat_ols = m_ols_macro, "xb"
predict macro resid_ols = m_ols_macro, "residuals"

# Se LOWESS ≈ OLS → relação linear bem capturada
# Se LOWESS difere muito de OLS → forma funcional inadequada
summarize(macro, yhat_ols, yhat_low)

# LOWESS dos resíduos vs fitted (especificação)
let m_ols2 = ols(pib_growth ~ inflation + unemployment, macro, cov=nonrobust)
predict macro fitted2 = m_ols2, "xb"
predict macro resid2  = m_ols2, "residuals"

# Se m_resid_lowess for plana → regressão bem especificada
let m_resid_low = lowess(macro, resid2, fitted2, frac=0.5)
predict macro smoothed_resid = m_resid_low, "smoothed"
# smoothed_resid deveria ser ≈ 0 por toda parte

# ══════════════════════════════════════════════════════════════════════════════
# 4. KDE — Estimativa de Densidade por Kernel (univariada)
# ══════════════════════════════════════════════════════════════════════════════
# Suavização não-paramétrica da distribuição empírica
# bandwidth (bw): largura de banda
#   bw=auto → regra de Silverman: 0.9 * min(σ, IQR/1.34) * n^(-1/5)
#   bw=0.5  → bandwidth manual
# kernels: gaussian (padrão), epanechnikov, triangular, uniform

load "https://www.stata-press.com/data/r9/auto.dta" as auto3

# KDE com bandwidth automático (Silverman)
kde(auto3, price)

# KDE com bandwidth manual (mais suave)
kde(auto3, price, bw=500)

# KDE com bandwidth estreito (mais rugoso)
kde(auto3, price, bw=100)

# KDE com kernel Epanechnikov (ótimo teórico, mas menos suave)
kde(auto3, price, kernel=epanechnikov)

# KDE de mpg (comparar com distribuição normal esperada)
kde(auto3, mpg)
kde(auto3, weight)

