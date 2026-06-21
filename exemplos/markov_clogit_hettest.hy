# Markov Switching, Conditional Logit/Poisson e Testes de Heteroskedasticidade
# markov, clogit, cpoisson, gqtest, bphet

# ══════════════════════════════════════════════════════════════════════════════
# 1. MARKOV SWITCHING AR (Hamilton 1989)
# ══════════════════════════════════════════════════════════════════════════════
# Modelo de mudança de regime com probabilidades de transição estocásticas
# Cada regime tem sua própria média, variância e dinâmica AR
#
# Modelo: y_t = μ_{s_t} + Σ φ_{j,s_t} · y_{t-j} + σ_{s_t} · ε_t
#   onde s_t ∈ {1, ..., k} segue uma cadeia de Markov com matriz de transição P
#
# k=: número de regimes (padrão: 2 — expansão/recessão)
# p=: ordem AR dentro de cada regime (padrão: 1)
#
# Parâmetros estimados por regime:
#   μ_j     : intercepto (média incondicional do regime j)
#   φ_j     : coeficiente(s) AR
#   σ²_j    : variância do erro
#   P_{ij}  : prob de transição de regime i para j
#
# Saída:
#   Tabela de parâmetros por regime
#   Matriz de transição P (k×k)
#   Durações esperadas em cada regime: E[D_j] = 1/(1 - P_{jj})
#   AIC e BIC

load "pib.csv" as pib
# Variável: pib_growth — taxa de crescimento trimestral (%)

# Modelo base: 2 regimes, AR(1)
let m_ms = markov(pib, pib_growth, k=2, p=1)
print(m_ms)
# Regime 1: expansão (μ alto, σ baixo, alta persistência)
# Regime 2: recessão (μ baixo ou negativo, σ alto)
# E[D_1] = duração esperada em expansão; E[D_2] = em recessão

# Modelo com 3 regimes (expansão forte, fraca, recessão)
let m_ms3 = markov(pib, pib_growth, k=3, p=1)
print(m_ms3)

# Comparar por AIC/BIC (quanto maior p, mais lags AR no modelo)
# ic() não suporta markov diretamente — comparar AIC do print()

# Extrair probabilidades suavizadas por regime
# regime1: P(s_t=1 | y_1..y_T) — prob do regime 1 (suavizada via forward-backward)
# regime2: P(s_t=2 | y_1..y_T) — prob do regime 2
# regime: regime mais provável em cada t (1-indexed, argmax)

predict pib prob_expansao = m_ms, regime1
predict pib prob_recessao = m_ms, regime2
predict pib regime_hat    = m_ms, regime     # 1=expansão, 2=recessão

summarize(pib, pib_growth, prob_expansao, prob_recessao, regime_hat)

# Períodos com P(recessão) > 0.5
# (use predict + filter posterior para identificar recessões)

# Modelo AR(2) — permite dinâmica mais rica
let m_ms_ar2 = markov(pib, pib_growth, k=2, p=2)
print(m_ms_ar2)
predict pib prob_rec_ar2 = m_ms_ar2, regime2
summarize(pib, prob_recessao, prob_rec_ar2)

# Aplicação financeira: volatilidade de retornos
load "retornos.csv" as ret
# Variável: ret_ibov — retornos diários do Ibovespa

# Markov Switching ARCH(0) — captura mudanças de regime de volatilidade
let m_ms_ret = markov(ret, ret_ibov, k=2, p=0)
print(m_ms_ret)
# Regime 1: baixa volatilidade (σ² pequeno)
# Regime 2: alta volatilidade (σ² grande)
predict ret vol_baixa = m_ms_ret, regime1
predict ret vol_alta  = m_ms_ret, regime2

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONDITIONAL LOGIT — FE Logit (Chamberlain 1980)
# ══════════════════════════════════════════════════════════════════════════════
# clogit(y ~ x1 + x2, df, group="id")
#
# Estima logit com efeitos fixos individuais SEM os estimar diretamente
# Condiciona na soma de y por grupo: P(y | Σy_i = m_i)
# Elimina o "incidental parameters problem" (Neyman-Scott 1948)
#
# Diferença de FE linear:
#   FE linear → within transformation: X̃ = X - X̄_i
#   FE logit  → condiciona na soma (Chamberlain's trick)
#   FE probit → NÃO consistente! — usar clogit ou mundlak RE
#
# Grupos sem variação em y (y=0 todo período ou y=1 todo período) são excluídos
# → quem nunca mudou não contribui para a verossimilhança
#
# Interpretação: efeitos parciais "within-group"
# Sem intercepto (absorbed by FE) → sem efeito marginal médio direto
# Para OR: exp(β̂) — razão de chances dentro do grupo

load "painel_discreto.csv" as panel
# Variáveis:
#   syndicato : 0/1 — filiação a sindicato (outcome)
#   lsalario  : log salário
#   horas     : horas trabalhadas
#   ind       : indicador de mudança de indústria
#   id        : identificador individual

let m_clogit = clogit(syndicato ~ lsalario + horas + ind, panel, group="id")
print(m_clogit)
# Odds ratios: exp(coef)
# Grupos excluídos: n_obs total vs n_obs contribuindo

# Comparar com RE logit (sem condicional)
let m_re_logit = logit(syndicato ~ lsalario + horas + ind, panel)
print(m_re_logit)

# esttab compara clogit vs RE logit (escala dos coeficientes pode diferir)
esttab(m_clogit, m_re_logit)
# Se β_clogit >> β_RE logit → heterogeidade individual distorcia o RE

# ── Discrete Choice: XTLOGIT com FE ──────────────────────────────────────────
# Aplicação em economia do trabalho: participação no mercado de trabalho
load "labor.csv" as labor
# lfp: labor force participation (0/1); wage, age, children, id, year

let m_lfp = clogit(lfp ~ wage + children, labor, group="id")
print(m_lfp)

# ══════════════════════════════════════════════════════════════════════════════
# 3. CONDITIONAL POISSON — FE Poisson / PPML (Silva & Tenreyro 2006)
# ══════════════════════════════════════════════════════════════════════════════
# cpoisson(y ~ x1 + x2, df, group="id")
# Aliás: ppml (Pseudo-Poisson Maximum Likelihood)
#
# Estimador PPML: consistente mesmo sem y ~ Poisson, desde que E[y|x,c] = exp(c+Xβ)
# Vantagem: consistente com heterogeidade não observada e zeros inflados
# Amplamente usado para equações de gravidade de comércio (exportações)
#
# Diferença de clogit:
#   clogit: y binário (0/1)
#   cpoisson: y contagem não-negativa (0,1,2,...)
#
# Condiciona na soma de y por grupo → elimina c_i
# Grupos com Σy=0 são excluídos (sem informação)

load "comercio.csv" as comercio
# Variáveis:
#   exportacoes : fluxo de exportações (não-negativo, com zeros)
#   pib_orig, pib_dest : PIB dos países de origem e destino
#   distancia   : distância geográfica
#   par         : par de países (group)

# PPML para equação de gravidade com FE bilateral
let m_ppml = cpoisson(exportacoes ~ pib_orig + pib_dest + distancia,
                      comercio, group="par")
print(m_ppml)

# Interpretação: exp(β) = efeito multiplicativo sobre exportações
# β(log_pib_orig) ≈ 1 → elasticidade unitária (padrão na teoria)

# Comparar com OLS em log (log-linearização)
generate comercio log_exp = log(exportacoes + 1)  # +1 para tratar zeros
let m_ols_grav = ols(log_exp ~ pib_orig + pib_dest + distancia, comercio)

esttab(m_ppml, m_ols_grav)
# OLS em log: descarta zeros e tem viés de Jensen se σ² correlaciona com regressores
# PPML: correto mesmo com zeros e heteroscedasticidade

# ══════════════════════════════════════════════════════════════════════════════
# 4. TESTES DE HETEROSKEDASTICIDADE — Complementos ao White e Breusch-Godfrey
# ══════════════════════════════════════════════════════════════════════════════

load "auto.csv" as auto

let m_ols = ols(price ~ mpg + weight + C(foreign), auto, cov=nonrobust)

# ── White test (já disponível) ───────────────────────────────────────────────
# H0: homocedasticidade
# Regride u² em X e X² — detecta heterocedasticidade de forma geral
white(m_ols)

# ── Breusch-Pagan het test ─────────────────────────────────────────────────
# H0: homocedasticidade
# LM = n·R² da regressão de u² em X
# Mais potente que White quando heterocedasticidade é linear em X
# bphet ≠ bptest: bptest é para efeitos aleatórios em painel; bphet é para OLS
bphet(m_ols)

# Sequência de diagnóstico de heteroskedasticidade:
#  1. white(model)  → teste geral (quadrático em X)
#  2. bphet(model)  → teste linear em X
#  3. gqtest(model) → teste baseado em subamostras ordenadas

# ── Goldfeld-Quandt test ─────────────────────────────────────────────────────
# H0: homocedasticidade (σ²₁ = σ²₂)
# Divide resíduos em dois grupos (descarta split do meio) e compara variâncias
# split=0.2: descarta 20% do meio (observações mais difíceis de classificar)
# Mais potente quando heterocedasticidade é monotônica com um regressor
# (ex: preços aumentam com renda — variância pode crescer com renda)
gqtest(m_ols, split=0.2)

# Com split maior (mais conservador):
gqtest(m_ols, split=0.3)

# Se todos os 3 testes rejeitam H0:
#   → usar SE robustos (HC1 ou HC3) ou WLS
let m_hc3 = ols(price ~ mpg + weight + C(foreign), auto, cov=HC3)
esttab(m_ols, m_hc3)
# Comparar SE: SE(HC3) > SE(OLS) quando há heterocedasticidade

# ── Bateria completa de diagnóstico pós-OLS ───────────────────────────────────
let m_ts = ols(pib_growth ~ inflation + juros, pib, cov=nonrobust)

# 1. Normalidade dos resíduos (importante para testes t exatos em n pequeno)
jb(m_ts)

# 2. Autocorrelação serial
bgodfrey(m_ts, lags=4)
ljungbox(m_ts, lags=10)

# 3. Heteroskedasticidade
white(m_ts)
bphet(m_ts)
gqtest(m_ts)

# 4. Forma funcional
reset(m_ts, power=3)

# 5. Multicolinearidade
vif(m_ts)

# 6. Observações influentes
influence(m_ts)
