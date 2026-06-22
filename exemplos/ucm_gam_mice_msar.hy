# UCM, GAM, MICE e Markov-AR
# ucm, gam, mice, msauto

# ══════════════════════════════════════════════════════════════════════════════
# 1. UCM — Unobserved Components Model (Harvey 1989)
# ══════════════════════════════════════════════════════════════════════════════
# Decompõe a série em componentes estruturais estimados via filtro de Kalman:
#   y_t = μ_t + γ_t + ε_t
#   μ_t : nível/tendência (trend)
#   γ_t : sazonalidade (opcional)
#   ε_t : irregular
#
# Especificações de nível (level=):
#   local_level      : μ_{t+1} = μ_t + η_t              — passeio aleatório
#   local_linear     : μ_{t+1} = μ_t + ν_t + η_t        — tendência estocástica (padrão)
#                      ν_{t+1} = ν_t + ζ_t
#   smooth_trend     : igual ao local_linear mas sem perturbação no nível (η=0)
#                      produz tendência mais suave
#   random_walk      : μ_{t+1} = μ_t + η_t, sem ruído de observação
#
# Especificações sazonais (seasonal=):
#   none             : sem componente sazonal (padrão)
#   deterministic    : sazonalidade fixa (dummies sazonais estáveis)
#   stochastic       : sazonalidade variante no tempo (mais flexível)
#   period=N         : comprimento do ciclo sazonal
#
# Vantagens sobre STL/decompose:
#   - Otimização MLE dos parâmetros de variância (σ²_η, σ²_ζ, σ²_ε, σ²_γ)
#   - Fornece AIC/BIC para comparação de especificações
#   - Incerteza via filtro de Kalman (propaga erros de previsão)
#   - Permite previsões fora da amostra com intervalo de confiança

load "pib_trim.csv" as pib
# Variável: pib_real — PIB real trimestral dessazonalizado (índice)

# ── UCM básico: local linear trend ───────────────────────────────────────────
let m_ucm = ucm(pib, pib_real, level=local_linear)
print(m_ucm)
# AIC/BIC: comparar com especificações alternativas
# Parâmetros: σ²_η (variância do nível), σ²_ζ (variância da tendência), σ²_ε

# ── UCM com sazonalidade estocástica ─────────────────────────────────────────
load "vendas_mensais.csv" as vendas
# Variável: receita — série mensal com tendência e sazonalidade

let m_ucm_saz = ucm(vendas, receita, level=local_linear, seasonal=stochastic, period=12)
print(m_ucm_saz)

# UCM com sazonalidade determinística (mais restrito)
let m_ucm_det = ucm(vendas, receita, level=local_linear, seasonal=deterministic, period=12)
print(m_ucm_det)

# Comparar AIC: estocástico vs determinístico
# AIC menor → melhor especificação sazonal para os dados

# UCM de tendência suave (σ²_η=0 → sem ruído no nível, só na inclinação)
let m_ucm_smooth = ucm(vendas, receita, level=smooth_trend, seasonal=stochastic, period=12)
print(m_ucm_smooth)

# ── Extrair componentes ───────────────────────────────────────────────────────
predict pib nivel_ucm   = m_ucm, "level"
predict pib tend_ucm    = m_ucm, "trend"
predict pib irr_ucm     = m_ucm, "residuals"

predict vendas nivel_s  = m_ucm_saz, "level"
predict vendas sazon_s  = m_ucm_saz, "seasonal"
predict vendas irr_s    = m_ucm_saz, "residuals"

summarize(vendas, receita, nivel_s, sazon_s, irr_s)

# ── UCM para dados diários com ciclo semanal ──────────────────────────────────
load "energia_diaria.csv" as energia
let m_ucm_dia = ucm(energia, consumo_mwh, level=local_linear,
                    seasonal=stochastic, period=7)
print(m_ucm_dia)
predict energia ciclo_sem = m_ucm_dia, "seasonal"

# ── UCM como filtro HP (caso limite) ─────────────────────────────────────────
# Hodrick-Prescott: caso especial com σ²_ε=0, σ²_η=0, σ²_ζ/σ²_ε = λ
# UCM smooth_trend → estimador MLE que generaliza HP com λ otimizado
let m_hp_ucm = ucm(pib, pib_real, level=smooth_trend)
print(m_hp_ucm)
predict pib tendencia_hp = m_hp_ucm, "level"
predict pib ciclo_hp     = m_hp_ucm, "residuals"

# Compara filtro HP (λ=1600) com UCM smooth_trend (λ estimado por MLE)
summarize(pib, pib_real, tendencia_hp, nivel_ucm)

# ══════════════════════════════════════════════════════════════════════════════
# 2. GAM — Generalized Additive Model (P-Splines)
# ══════════════════════════════════════════════════════════════════════════════
# GAM generaliza GLM permitindo relações não-lineares via splines:
#   g(E[Y]) = β₀ + β₁x₁ + ... + f(x_s)
# onde f(x_s) é uma função suave estimada via B-splines penalizados
#
# Parâmetros:
#   smooth="var"      : variável a suavizar (B-spline cúbico)
#   smooth=["v1","v2"]: múltiplas variáveis suavizadas
#   spline_df=10      : graus de liberdade dos splines por variável (default: 10)
#   alpha=0.1         : parâmetro de penalização λ (maior → mais suave)
#   degree=3          : grau do spline (3 = cúbico, padrão)
#   family=gaussian   : família da distribuição (igual ao GLM)
#   link=identity     : função de ligação (igual ao GLM)
#
# Quando usar GAM vs OLS/GLM:
#   OLS/GLM: quando a relação é linear (ou log-linear, logística, etc.)
#   GAM: quando suspeita de não-linearidade mas sem forma funcional clara
#   GAM > LOWESS: GAM trata múltiplos regressores simultaneamente
#   GAM > polinômio: GAM é mais estável fora da amostra
#
# EDF (Effective Degrees of Freedom): mede a complexidade da curva suave
#   EDF ≈ 1  → relação linear (spline degenerado)
#   EDF ≈ df → curva muito flexível (possível overfitting)
#   EDF intermediário → não-linearidade real
#
# GCV Score (Generalized Cross-Validation): critério para selecionar alpha
#   alpha menor → GCV menor → mas risco de overfitting
#   Regra prática: variar alpha=[0.01, 0.1, 1, 10] e escolher menor GCV

load "auto.csv" as auto
# Variáveis: price, mpg, weight, length, foreign

# ── GAM com um smooth term ───────────────────────────────────────────────────
# Pergunta: a relação entre price e mpg é realmente linear?
let m_gam = gam(price ~ weight, auto, smooth="mpg", spline_df=8, alpha=0.1)
print(m_gam)
# EDF próximo de 2+ → relação curvilínea
# EDF ≈ 1 → relação essencialmente linear

# Comparar com OLS linear
let m_ols = ols(price ~ mpg + weight, auto, cov=nonrobust)
print(m_ols)

# GCV: mede qualidade de ajuste fora da amostra
# Se GAM tem GCV muito menor → não-linearidade é real e relevante

# ── GAM com família Poisson (contagem) ───────────────────────────────────────
load "acidentes.csv" as acid
# Variáveis: contagem (# acidentes), velocidade (km/h), volume (veículos/hora)
# Relação entre velocidade e acidentes provavelmente não-linear

let m_gam_pois = gam(contagem ~ volume, acid,
                     smooth="velocidade", spline_df=10, alpha=0.1,
                     family=poisson, link=log)
print(m_gam_pois)
# Smooth de velocidade: captura ponto de inflexão (risco cresce não-linearmente)

# ── GAM com múltiplos smooth terms ───────────────────────────────────────────
load "saude.csv" as saude
# Variáveis: custo, idade, renda, anos_doenca, sinistros_prev

let m_gam_multi = gam(custo ~ sinistros_prev, saude,
                      smooth=["idade", "renda"], spline_df=8, alpha=0.1,
                      family=gamma, link=log)
print(m_gam_multi)
# Smooth de idade: risco cresce com curvatura após certa idade
# Smooth de renda: possível relação em U (baixa renda = risco, alta renda = cobertura)

# ── Seleção de alpha via GCV ──────────────────────────────────────────────────
# Variar alpha e comparar GCV (menor = melhor)
let m01  = gam(price ~ weight, auto, smooth="mpg", spline_df=8, alpha=0.01)
let m1   = gam(price ~ weight, auto, smooth="mpg", spline_df=8, alpha=0.1)
let m10  = gam(price ~ weight, auto, smooth="mpg", spline_df=8, alpha=1.0)
let m100 = gam(price ~ weight, auto, smooth="mpg", spline_df=8, alpha=10.0)
print(m01)   # GCV pode ser maior por overfitting
print(m1)    # provavelmente ótimo
print(m10)
print(m100)  # curva mais suave, possivelmente linear

# ══════════════════════════════════════════════════════════════════════════════
# 3. MICE — Multiple Imputation by Chained Equations
# ══════════════════════════════════════════════════════════════════════════════
# Substitui valores ausentes (NaN) por m imputações plausíveis
# Cada imputação é gerada por uma cadeia de regressões condicionais:
#   X₁|X₂,...,Xₚ ~ Modelo₁(X₂,...,Xₚ)
#   X₂|X₁,X₃,...,Xₚ ~ Modelo₂(X₁,X₃,...,Xₚ)
#   ...
# Após iter iterações, a distribuição condicional converge
#
# Resultado: m datasets completos (sem NaN)
# Análise: estimar modelo em cada dataset, pooling via Regras de Rubin:
#   β̄ = (1/m) Σ β̂_i
#   Var(β̄) = W̄ + (1 + 1/m)·B
#   onde W̄ = variância within (média das variâncias amostrais)
#         B = variância between (variância dos β̂_i)
#
# Parâmetros:
#   vars=["x1","x2"]  : variáveis a imputar (com NaN)
#   m=5               : número de imputações (mínimo: 5; recomendado: 10-20)
#   iter=10           : iterações MICE por imputação
#
# IMPORTANTE: mice() NUNCA transforma dados silenciosamente.
# Os NaN nos dados são preservados no df original; mice cria datasets completos.
#
# Quando usar MICE vs complete case analysis:
#   Complete case: válido APENAS se dados são MCAR (Missing Completely At Random)
#   MICE: válido para MAR (Missing At Random) — missing depende de X obs, não Y

load "renda_emprego.csv" as emp
# Variáveis:
#   salario    : renda salarial (com ~15% ausente — MCAR?)
#   educacao   : anos de escolaridade (completo)
#   experiencia: anos de experiência (com ~8% ausente)
#   setor      : setor de emprego (completo)

# Diagnóstico: quanto está ausente?
summarize(emp, salario, educacao, experiencia)
# Se p-missing grande → MICE necessário

# Imputar 10 datasets com 15 iterações
mice(emp, vars=["salario","experiencia"], m=10, iter=15)
# Saída: resumo do MICE (n_obs, n_vars, m imputações, iter)
# Os datasets imputados são armazenados em memória — use com OLS abaixo

# Sintaxe alternativa (vars como argumentos posicionais):
mice(emp, salario, experiencia, m=10, iter=15)

# ── Uso prático: pooling manual (estimativas por imputação) ──────────────────
# No Hayashi DSL, ainda não há pooling automático.
# Estratégia: usar apenas os casos completos para regressão,
# ou usar MICE para inspecionar o impacto dos ausentes na distribuição.
#
# Passo prático:
# 1. mice() → print resumo (quantos obs por variável imputados)
# 2. Estimar OLS nos dados completos (NaN excluídos automaticamente pelo Greeners)
# 3. Comparar coeficientes: se diferirem muito → MAR ou MNAR, não MCAR

let m_cc = ols(salario ~ educacao + experiencia, emp)
print(m_cc)
# m_cc usa apenas os casos completos (complete case)
# Se m_cc e m_mice (pooled) diferirem → ausentes não são MCAR

# ══════════════════════════════════════════════════════════════════════════════
# 4. MARKOV AUTOREGRESSION (Hamilton 1989) — MS-AR(k, p)
# ══════════════════════════════════════════════════════════════════════════════
# Modelo AR com mudança de regime:
#   y_t = μ_{s_t} + Σ φ_{j,s_t}·y_{t-j} + σ_{s_t}·ε_t
#   s_t ∈ {1,...,k} : cadeia de Markov com matriz P_{ij} = P(s_t=j|s_{t-1}=i)
#
# Diferença de markov() (MarkovSwitching):
#   markov(): mudança na média e variância do processo — sem dinâmica AR por regime
#   msauto(): parâmetros AR completos por regime (φ_{j,s_t})
#             → mais rico: cada regime tem sua própria dinâmica temporal
#
# Parâmetros:
#   k= : número de regimes (padrão: 2)
#   p= : ordem AR (padrão: 1)
#
# Estimação: EM algorithm com filtro de Hamilton (forward-backward smoothing)
# Saída:
#   Parâmetros por regime: μ_j, φ_{j,1..p}, σ_j
#   Matriz de transição P (k×k)
#   Durações esperadas: E[D_j] = 1/(1-P_{jj})
#   Probabilidades suavizadas: P(s_t=j | y_1..y_T) para cada t
#   AIC, BIC

load "pib.csv" as pib2

# ── MS-AR(2,1): 2 regimes, AR(1) ──────────────────────────────────────────
let m_msar = msauto(pib2, pib_growth, k=2, p=1)
print(m_msar)
# Regime 1 (expansão): φ_1 ≈ +0.5 (persistência positiva), σ₁ pequeno
# Regime 2 (recessão): φ_1 ≈ -0.2 (reversão mais rápida), σ₂ maior

# Extrair probabilidades suavizadas
predict pib2 prob_exp    = m_msar, "regime1" # P(s_t=1|Y) — expansão
predict pib2 prob_rec    = m_msar, "regime2" # P(s_t=2|Y) — recessão
predict pib2 regime_hat  = m_msar, "regime" # regime mais provável (1 ou 2)

summarize(pib2, pib_growth, prob_exp, prob_rec, regime_hat)

# ── MS-AR(2,2): dinâmica AR(2) por regime ─────────────────────────────────
let m_msar2 = msauto(pib2, pib_growth, k=2, p=2)
print(m_msar2)
# φ_{1,1} e φ_{1,2}: coeficientes AR por lag em cada regime
# Comparar AIC com AR(1): vale a pena o AR(2)?

# ── 3 regimes ─────────────────────────────────────────────────────────────────
let m_msar3 = msauto(pib2, pib_growth, k=3, p=1)
print(m_msar3)
# Regime 1: expansão forte  (φ alto, σ pequeno)
# Regime 2: expansão fraca  (φ médio)
# Regime 3: recessão        (φ baixo ou negativo, σ alto)
predict pib2 prob_r3 = m_msar3, "regime3"

# ── Comparação msauto vs markov ───────────────────────────────────────────────
# markov() = MarkovSwitching: muda apenas a média (intercept) e σ por regime
# msauto() = MarkovAutoregression: muda os coeficientes AR completos por regime
# → AIC(msauto) < AIC(markov) quando a dinâmica AR difere entre regimes
let m_ms_simples = markov(pib2, pib_growth, k=2, p=1)
print(m_ms_simples)
print(m_msar)
# Comparar AIC/BIC: qual modelo melhor descreve os dados?

# ── Aplicação financeira: volatility regimes ──────────────────────────────────
load "retornos.csv" as ret2
# Variável: ret_ibov — retornos diários do Ibovespa

let m_vol = msauto(ret2, ret_ibov, k=2, p=1)
print(m_vol)
# Regime 1: baixa volatilidade (σ₁ pequeno, persistência baixa)
# Regime 2: alta volatilidade  (σ₂ grande, clustering de volatilidade)

predict ret2 high_vol = m_vol, "regime2"
summarize(ret2, ret_ibov, high_vol)
# Período 2008, 2020: P(regime 2) alto → crises identificadas automaticamente
