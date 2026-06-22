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

input pib
pib_real
100.0
101.5
103.2
102.8
104.5
106.0
107.8
107.2
109.0
110.5
112.2
111.8
113.5
115.0
116.8
116.2
118.0
119.5
121.2
120.8
122.5
124.0
125.8
125.2
127.0
128.5
130.2
129.8
131.5
133.0
end
# Variável: pib_real — PIB real trimestral dessazonalizado (índice)

# ── UCM básico: local linear trend ───────────────────────────────────────────
let m_ucm = ucm(pib, pib_real, level=local_linear)
print(m_ucm)
# AIC/BIC: comparar com especificações alternativas
# Parâmetros: σ²_η (variância do nível), σ²_ζ (variância da tendência), σ²_ε

# ── UCM com sazonalidade estocástica ─────────────────────────────────────────
input vendas
receita
120.5
135.2
128.8
142.1
155.3
148.7
160.2
175.8
168.4
150.1
138.9
185.6
125.3
140.8
133.5
147.2
160.8
153.4
165.9
181.2
173.8
155.7
143.5
190.2
130.1
145.5
138.2
152.0
165.5
158.1
end
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
input energia
consumo_mwh
450.2
430.5
465.8
480.1
470.3
420.6
410.8
455.1
435.4
470.7
485.0
475.2
425.5
415.7
460.0
440.3
475.6
490.9
480.1
430.4
420.6
465.9
445.2
480.5
495.8
485.0
435.3
425.5
470.8
450.1
end
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

load "https://www.stata-press.com/data/r9/auto.dta" as auto
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
input acid
contagem velocidade volume
3.0 50.0 800.0
5.0 70.0 900.0
2.0 40.0 600.0
8.0 90.0 1000.0
1.0 30.0 500.0
6.0 80.0 950.0
4.0 60.0 850.0
10.0 110.0 1100.0
7.0 85.0 980.0
2.0 45.0 650.0
9.0 100.0 1050.0
3.0 55.0 750.0
12.0 120.0 1200.0
5.0 65.0 880.0
1.0 35.0 550.0
8.0 95.0 1020.0
4.0 58.0 820.0
11.0 115.0 1150.0
6.0 75.0 920.0
2.0 42.0 620.0
7.0 88.0 990.0
3.0 52.0 780.0
9.0 105.0 1080.0
5.0 68.0 890.0
13.0 125.0 1250.0
end
# Variáveis: contagem (# acidentes), velocidade (km/h), volume (veículos/hora)
# Relação entre velocidade e acidentes provavelmente não-linear

let m_gam_pois = gam(contagem ~ volume, acid,
                     smooth="velocidade", spline_df=10, alpha=0.1,
                     family=poisson, link=log)
print(m_gam_pois)
# Smooth de velocidade: captura ponto de inflexão (risco cresce não-linearmente)

# ── GAM com múltiplos smooth terms ───────────────────────────────────────────
input saude
custo idade renda anos_doenca sinistros_prev
1500.0 25.0 3500.0 0.0 0.0
3200.0 45.0 4200.0 5.0 2.0
800.0 30.0 2800.0 1.0 0.0
5500.0 60.0 5000.0 10.0 4.0
1200.0 28.0 3200.0 0.0 1.0
4800.0 55.0 4800.0 8.0 3.0
2000.0 35.0 3800.0 2.0 1.0
6200.0 65.0 5500.0 12.0 5.0
1800.0 32.0 3600.0 1.0 0.0
4200.0 50.0 4500.0 7.0 3.0
900.0 27.0 3000.0 0.0 0.0
5000.0 58.0 4900.0 9.0 4.0
2500.0 40.0 4000.0 3.0 1.0
7000.0 70.0 6000.0 15.0 6.0
1100.0 29.0 3100.0 0.0 0.0
3800.0 48.0 4400.0 6.0 2.0
1600.0 33.0 3400.0 1.0 1.0
5800.0 62.0 5200.0 11.0 5.0
2200.0 38.0 3900.0 2.0 1.0
4500.0 52.0 4600.0 7.0 3.0
end
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

input emp
salario educacao experiencia
2500.0 12.0 5.0
. 16.0 10.0
3200.0 14.0 8.0
4500.0 18.0 .
1800.0 10.0 3.0
. 12.0 6.0
3800.0 16.0 12.0
2200.0 11.0 4.0
5000.0 20.0 15.0
. 14.0 7.0
2800.0 13.0 .
4000.0 17.0 11.0
1500.0 9.0 2.0
3500.0 15.0 9.0
. 12.0 5.0
2600.0 13.0 6.0
4200.0 18.0 13.0
1900.0 10.0 .
3000.0 14.0 7.0
5500.0 20.0 18.0
end
# Variáveis:
#   salario    : renda salarial (com ~15% ausente)
#   educacao   : anos de escolaridade (completo)
#   experiencia: anos de experiência (com ~8% ausente)

# Diagnóstico: quanto está ausente?
summarize(emp, salario, educacao, experiencia)
# Se p-missing grande → MICE necessário

# Imputar 10 datasets com 15 iterações (sintaxe posicional)
mice(emp, salario, experiencia, m=10, iter=15)
# Saída: resumo do MICE (n_obs, n_vars, m imputações, iter)
# Os datasets imputados são armazenados em memória

# ── Uso prático: pooling manual (estimativas por imputação) ──────────────────
# No Hayashi DSL, ainda não há pooling automático.
# Estratégia: usar apenas os casos completos para regressão,
# ou usar MICE para inspecionar o impacto dos ausentes na distribuição.
#
# Passo prático:
# 1. mice() → print resumo (quantos obs por variável imputados)
# 2. Estimar OLS nos dados completos (NaN excluídos automaticamente pelo Greeners)
# 3. Comparar coeficientes: se diferirem muito → MAR ou MNAR, não MCAR

# OLS com NaN requer dropna() prévio — dados com missing não são passados diretamente
# let m_cc = ols(salario ~ educacao + experiencia, emp)
# print(m_cc)

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

input pib2
pib_growth
3.2
2.8
-0.5
-1.2
0.8
2.5
3.5
3.8
2.0
-0.8
-1.5
0.5
2.2
3.0
3.6
4.0
2.5
0.3
-0.3
-1.0
0.8
2.8
3.2
3.5
2.0
-0.5
-1.2
0.3
1.8
3.0
end

# ── MS-AR(2,1): 2 regimes, AR(1) ──────────────────────────────────────────
let m_msar = msauto(pib2, pib_growth, k=2, p=1)
print(m_msar)
# Regime 1 (expansão): φ_1 ≈ +0.5 (persistência positiva), σ₁ pequeno
# Regime 2 (recessão): φ_1 ≈ -0.2 (reversão mais rápida), σ₂ maior

# predict de regimes gera n-p linhas (MS-AR dropa p obs iniciais)
# predict pib2 prob_exp    = m_msar, "regime1"
# predict pib2 prob_rec    = m_msar, "regime2"
# predict pib2 regime_hat  = m_msar, "regime"

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
# predict pib2 prob_r3 = m_msar3, "regime3"  # dimension mismatch (n-p)

# ── Comparação msauto vs markov ───────────────────────────────────────────────
# markov() = MarkovSwitching: muda apenas a média (intercept) e σ por regime
# msauto() = MarkovAutoregression: muda os coeficientes AR completos por regime
# → AIC(msauto) < AIC(markov) quando a dinâmica AR difere entre regimes
let m_ms_simples = markov(pib2, pib_growth, k=2, p=1)
print(m_ms_simples)
print(m_msar)
# Comparar AIC/BIC: qual modelo melhor descreve os dados?

# ── Aplicação financeira: volatility regimes ──────────────────────────────────
input ret2
ret_ibov
0.012
-0.008
0.005
0.015
-0.020
0.003
-0.002
0.018
-0.025
0.010
0.008
-0.005
0.002
-0.015
0.020
-0.003
0.012
-0.030
0.025
-0.010
0.005
-0.018
0.008
0.015
-0.012
0.003
-0.035
0.022
-0.008
0.010
end
# Variável: ret_ibov — retornos diários do Ibovespa

let m_vol = msauto(ret2, ret_ibov, k=2, p=1)
print(m_vol)
# Regime 1: baixa volatilidade (σ₁ pequeno, persistência baixa)
# Regime 2: alta volatilidade  (σ₂ grande, clustering de volatilidade)

# predict ret2 high_vol = m_vol, "regime2"  # dimension mismatch (n-p)
# summarize(ret2, ret_ibov, high_vol)
# Período 2008, 2020: P(regime 2) alto → crises identificadas automaticamente
