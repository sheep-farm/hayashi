# ETS (Holt-Winters), Panel Threshold e Plots Diagnósticos
# ets, ses, hwes, forecast, pthresh, acfplot, pacfplot, qqplot, corrplot

# ══════════════════════════════════════════════════════════════════════════════
# 1. ETS — Exponential Smoothing / Holt-Winters
# ══════════════════════════════════════════════════════════════════════════════
# Modelos ETS(Error, Trend, Seasonal):
#   Erro:    A(ditivo) — resíduos independentes do nível
#            M(ultiplicativo) — resíduos escalam com o nível (implementado internamente)
#   Tendência: N(ão) | A(ditiva) | A_d(amortecida) | M(ultiplicativa)
#   Sazonal:   N(ão) | A(ditiva) | M(ultiplicativa) + período
#
# Parâmetros:
#   trend=    : "add" | "mul" | "none"        (padrão: "add")
#   seasonal= : "add" | "mul" | "none"        (padrão: "add")
#   period=   : comprimento do ciclo sazonal  (padrão: 12)
#   damped=   : true → tendência amortecida φ ∈ (0,1) (padrão: false)
#
# Aliases convenientes:
#   ses(df, var)          → ETS(A,N,N): suavização simples sem tendência/sazonal
#   hwes(df, var, ...)    → ETS(A,A,A): Holt-Winters aditivo completo
#   ets(df, var, ...)     → especificação explícita
#
# Parâmetros estimados (grid search SSE):
#   α: suavização do nível     (0.05 a 0.95)
#   β: suavização da tendência (0.05 a 0.95, se trend ≠ none)
#   γ: suavização sazonal      (0.05 a 0.95, se seasonal ≠ none)
#   φ: fator de amortecimento  (0.80 a 0.99, se damped=true)
#
# Seleção de modelo via AIC/BIC:
#   AIC = n·log(SSE/n) + 2k    onde k = nº parâmetros
#   Menor AIC/BIC → melhor ajuste penalizado pela complexidade

input v
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
end
generate v t = _n

# ── SES — Simple Exponential Smoothing (ETS A,N,N) ───────────────────────────
# Sem tendência e sem sazonalidade — bom para séries estacionárias com ruído
# Previsão: ŷ_{t+h|t} = l_t  (nível constante para todos os horizontes)
let m_ses = ses(v, receita)
print(m_ses)
# α perto de 1 → reativo a mudanças recentes (memória curta)
# α perto de 0 → suavização pesada (memória longa)

predict v fitted_ses  = m_ses, "fitted"
predict v resid_ses   = m_ses, "residuals"
predict v level_ses   = m_ses, "level"

lineplot(v, t, receita,     title="Receita — série original")
lineplot(v, t, fitted_ses,  title="Receita — SES (ŷ)")
histogram(v, resid_ses, bins=15, title="Resíduos SES")

# ── Holt (ETS A,A,N) — com tendência ─────────────────────────────────────────
# ŷ_{t+h|t} = l_t + h·b_t
let m_holt = ets(v, receita, trend=add, seasonal=none)
print(m_holt)

predict v trend_holt = m_holt, "trend"
lineplot(v, t, trend_holt, title="Receita — tendência Holt")

# ── Holt-Winters aditivo (ETS A,A,A) ─────────────────────────────────────────
# Padrão sazonal constante (mesma amplitude ao longo do tempo)
# Equações:
#   l_t = α(y_t - s_{t-m}) + (1-α)(l_{t-1} + b_{t-1})
#   b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}
#   s_t = γ(y_t - l_{t-1} - b_{t-1}) + (1-γ)s_{t-m}
# ŷ_{t+h} = l_t + h·b_t + s_{t+h-m(k+1)}
let m_hw = hwes(v, receita, period=12)
print(m_hw)

predict v fitted_hw   = m_hw, "fitted"
predict v trend_hw    = m_hw, "trend"
predict v seasonal_hw = m_hw, "seasonal"
predict v resid_hw    = m_hw, "residuals"

lineplot(v, t, receita,     title="Receita — observado")
lineplot(v, t, fitted_hw,   title="Receita — Holt-Winters ŷ")
lineplot(v, t, trend_hw,    title="Receita — tendência HW")
lineplot(v, t, seasonal_hw, title="Receita — sazonalidade HW")
histogram(v, resid_hw, bins=15, title="Resíduos Holt-Winters")
kdensity(v, resid_hw, title="Resíduos HW — KDE")

# ── Holt-Winters multiplicativo ────────────────────────────────────────────────
# Para séries onde a amplitude sazonal cresce com o nível (ex: vendas de varejo)
# seasonal=mul → s_t escala o nível (razão, não diferença)
let m_hwm = ets(v, receita, trend=add, seasonal=mul, period=12)
print(m_hwm)
# Comparar AIC: se AIC(mul) < AIC(add) → sazonalidade multiplicativa melhor

# ── Holt-Winters com tendência amortecida ────────────────────────────────────
# damped=true → b_t decai geometricamente: ŷ_{t+h} = l_t + φ_h·b_t
# Útil para evitar extrapolação linear em horizontes longos
let m_hwd = ets(v, receita, trend=add, seasonal=add, period=12, damped=true)
print(m_hwd)
# φ estimado ≈ 0.9 → amortecimento moderado
# φ ≈ 1 → equivale ao não-amortecido

# ── Comparação de modelos via AIC ────────────────────────────────────────────
# Ver AIC no print() de cada modelo; menor = melhor
# Regra geral: hwes > holt > ses (mais complexo, melhor ajuste)
# Mas AIC penaliza complexidade → pode preferir ses se sazonalidade for fraca

# ── Previsão fora da amostra com forecast() ──────────────────────────────────
# forecast(model, steps=12, alpha=0.05)
# Retorna vetor de previsões + imprime tabela com IC (σ cresce com √h)
# forecast(m_hw, steps=12, alpha=0.05)
# Coluna lower/upper: IC Gaussiano ingênuo (σ dos resíduos × √h)
# Para intervalos exatos, usar bootstrap posterior

# Armazenar previsões como lista
# let previsoes = forecast(m_hw, steps=6)
# previsoes[0] ... previsoes[5] = h=1..6

# ── Exemplo: dados de energia elétrica ───────────────────────────────────────
input en
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
generate en t_e = _n

# Dado que há ciclo semanal e anual, MSTL é melhor para decomposição
# Mas ETS (com period=7) ainda pode ser útil para forecast de curto prazo
let m_hw7 = hwes(en, consumo_mwh, period=7)
print(m_hw7)
# forecast(m_hw7, steps=14)    # 2 semanas à frente

# ══════════════════════════════════════════════════════════════════════════════
# 2. PANEL THRESHOLD — Hansen (1999) Threshold Regression
# ══════════════════════════════════════════════════════════════════════════════
# Modelo: y_{it} = α₁ · x_{it} · 1(q_{it} ≤ γ) + α₂ · x_{it} · 1(q_{it} > γ) + c_i + ε_{it}
#
# O parâmetro γ (limiar) é estimado por grid search:
#   γ̂ = argmin SSR(γ) — minimiza soma dos quadrados dos resíduos
#
# Trimming: valores extremos de q (15% de cada lado) são descartados como candidatos
# → evita regimes com pouquíssimas observações
#
# Diferença de modelos de mudança de regime:
#   Markov Switching: transição probabilística (regime é latente)
#   Panel Threshold:  transição determinística baseada em q_{it} observado
#                     (q pode ser uma variável econômica: renda, dívida, etc.)
#
# Sintaxe:
#   pthresh(y ~ x1 + x2, df, q=threshold_var, id=entity_col)
#   q= : variável que determina o regime (ex: dívida/PIB, nível de renda)
#   id= : coluna que identifica as entidades (necessário para FE within)
#
# Quando usar:
#   - Hipótese de não-linearidade na relação entre y e x
#   - Efeito de x muda quando q cruza um limiar desconhecido
#   - Ex: efeito da dívida sobre crescimento muda quando dívida > 90% do PIB?

input pan
crescimento investimento abertura divida id
4.5 22.1 35.2 0.45 1.0
3.8 21.5 36.8 0.52 1.0
4.2 23.0 34.5 0.48 1.0
3.1 20.8 37.1 0.61 1.0
2.5 19.5 38.5 0.75 1.0
5.1 24.2 33.8 0.38 2.0
4.6 23.8 35.1 0.42 2.0
4.9 24.5 34.2 0.40 2.0
3.5 22.0 36.5 0.58 2.0
2.8 20.5 37.8 0.70 2.0
3.2 21.0 40.2 0.65 3.0
3.9 22.5 38.5 0.55 3.0
4.1 23.2 37.8 0.50 3.0
2.6 19.8 41.5 0.80 3.0
1.8 18.5 42.8 0.92 3.0
5.5 25.0 32.5 0.35 4.0
5.0 24.0 33.2 0.39 4.0
5.3 24.8 32.8 0.37 4.0
4.0 22.8 35.5 0.55 4.0
3.2 21.2 37.0 0.68 4.0
2.0 18.0 43.5 0.95 5.0
2.5 19.2 42.0 0.88 5.0
3.0 20.5 40.5 0.78 5.0
3.8 21.8 38.2 0.62 5.0
4.3 23.0 36.5 0.50 5.0
end
# Variáveis:
#   crescimento : taxa de crescimento do PIB (y)
#   investimento: taxa de investimento/PIB
#   abertura    : abertura comercial (exog.)
#   divida      : dívida pública/PIB (variável threshold)
#   id          : identificador do país

# Modelo threshold: efeito do investimento muda com nível de dívida?
# H₀: β_investimento é constante
# H₁: β_investimento = α₁ se divida ≤ γ, α₂ se divida > γ
let m_thresh = pthresh(crescimento ~ investimento + abertura, pan,
                       q=divida, id=id)
print(m_thresh)
# Output:
#   Threshold estimado γ̂: ponto onde a relação muda
#   Regime 1 (abaixo do limiar): coef. de investimento = α₁
#   Regime 2 (acima do limiar): coef. de investimento = α₂
#   R² combinado
#   RMSPE no período de busca

# Interpretação econômica:
#   Se γ̂ ≈ 0.90 (90%):
#     α₁ > 0 e α₂ < α₁ → dívida alta reduz multiplicador do investimento
#     Ou: α₁ > 0 e α₂ ≈ 0 → investimento ineficaz quando dívida alta
#   Se γ̂ alto e regimes quase iguais → evidência fraca de threshold

# ── Outro exemplo: efeito de crédito bancário no crescimento ─────────────────
input cred
crescimento credito capital_regulatorio inadimplencia banco
3.5 15.2 12.0 0.03 1.0
3.8 16.1 11.5 0.02 1.0
3.2 14.8 12.3 0.04 1.0
2.8 13.5 11.8 0.06 1.0
2.5 12.8 12.5 0.08 1.0
4.2 18.5 13.0 0.02 2.0
4.5 19.2 12.8 0.01 2.0
3.9 17.8 13.2 0.03 2.0
3.2 16.0 12.5 0.05 2.0
2.8 14.5 13.5 0.07 2.0
2.0 11.5 10.5 0.09 3.0
2.3 12.2 10.8 0.08 3.0
1.8 10.8 10.2 0.11 3.0
1.5 10.0 11.0 0.13 3.0
1.2 9.5 10.5 0.15 3.0
5.0 20.5 14.0 0.01 4.0
4.8 20.0 13.8 0.02 4.0
4.5 19.5 14.2 0.02 4.0
3.8 18.0 13.5 0.04 4.0
3.5 17.0 14.0 0.05 4.0
1.0 8.5 9.5 0.18 5.0
1.5 9.8 9.8 0.15 5.0
0.8 8.0 9.2 0.20 5.0
1.8 10.5 10.0 0.12 5.0
2.2 11.8 10.5 0.10 5.0
end
# crescimento ~ credito, q=inadimplencia, id=banco
let m_cred = pthresh(crescimento ~ credito + capital_regulatorio, cred,
                     q=inadimplencia, id=banco)
print(m_cred)
# γ̂: nível de inadimplência onde o efeito do crédito muda de sinal?
# Acima do limiar: expansão de crédito pode ser contracionária

# ══════════════════════════════════════════════════════════════════════════════
# 3. ACF / PACF — Autocorrelação e Autocorrelação Parcial
# ══════════════════════════════════════════════════════════════════════════════
# acfplot(df, var, lags=20, width=50, title="")
# pacfplot(df, var, lags=20, width=50, title="")
# Alias: corrgram (para acf)
#
# Interpretação:
#   ACF(k):  correlação entre y_t e y_{t-k} (sem remover efeitos intermediários)
#   PACF(k): correlação entre y_t e y_{t-k} APÓS remover efeitos de y_{t-1}..y_{t-k+1}
#
# PACF usa o algoritmo de Durbin-Levinson recursivo:
#   φ_{kk} = [acf_k - Σ_{j=1}^{k-1} φ_{k-1,j} · acf_{k-j}] /
#             [1      - Σ_{j=1}^{k-1} φ_{k-1,j} · acf_j]
#
# Banda de confiança: ±1.96/√n (distribuição assintótica sob H₀: ρ_k = 0)
# ── (representação: ── dentro  █ fora da banda)
#
# Diagnóstico ARMA:
#   AR(p): ACF decai gradualmente, PACF corta após lag p
#   MA(q): ACF corta após lag q, PACF decai gradualmente
#   ARMA(p,q): ambas decaem (mistura) — usar AIC para selecionar

input mac
pib inflation juros
100.0 3.5 10.5
101.2 3.8 10.8
102.5 3.2 10.2
103.1 4.0 11.0
104.8 3.6 10.6
105.5 4.2 11.2
106.2 3.9 10.9
107.0 3.4 10.4
108.5 4.5 11.5
109.2 3.7 10.7
110.0 4.1 11.1
111.5 3.3 10.3
112.0 3.8 10.8
113.2 4.3 11.3
114.5 3.5 10.5
115.0 4.0 11.0
116.8 3.6 10.6
117.5 4.4 11.4
118.2 3.2 10.2
119.0 3.9 10.9
end
generate mac t2 = _n

# ACF da série original de PIB
acfplot(mac, pib, lags=20, title="ACF — PIB trimestral")
# Se ACF decai muito lentamente → série não-estacionária → diferenciar

# Usando ARIMA com d=1 para diferenciar internamente (evita NaN manual)
# Diagnóstico:
#   ACF corta em q=1, PACF decai → MA(1) → ARIMA(0,1,1)
#   PACF corta em p=2, ACF decai → AR(2) → ARIMA(2,1,0)
#   Ambas decaem → ARIMA(1,1,1) ou selecionar por AIC

# Depois de estimar ARIMA, verificar resíduos:
let m_ar = arima(mac, pib, p=2, d=1, q=0)
print(m_ar)
# ljungbox(m_ar, lags=10)    # H₀: resíduos i.i.d.

# ACF de resíduos ETS
predict v resid_hwv = m_hw, "residuals"
acfplot(v, resid_hwv, lags=20, title="ACF Resíduos Holt-Winters")
# Bom modelo ETS: resíduos sem autocorrelação em nenhum lag

# ══════════════════════════════════════════════════════════════════════════════
# 4. QQ-PLOT — diagnóstico de normalidade
# ══════════════════════════════════════════════════════════════════════════════
# qqplot(df, var, width=50, height=20, title="")
# Alias: qnorm, pnorm
#
# Plota quantis empíricos ordenados vs quantis teóricos de N(0,1)
# → dados padronizados: (x - x̄) / σ
# Quantis teóricos: p_i = (i - 3/8) / (n + 1/4) — Blom (1958)
# Invnorm via aproximação racional de Abramowitz & Stegun
#
# Leitura:
#   Pontos ao longo da diagonal y=x → Normal
#   Caudas pesadas: pontos abaixo da diagonal (esquerda) e acima (direita)
#   Assimetria positiva: cauda direita acima da diagonal
#   Assimetria negativa: cauda esquerda abaixo

load "https://www.stata-press.com/data/r9/auto.dta" as auto

let m_a = ols(price ~ mpg + weight + C(foreign), auto)
predict auto resid_a = m_a, "residuals"

qqplot(auto, resid_a, title="QQ-plot — Resíduos OLS (price)")
# Se caudas se afastam da diagonal → violação de normalidade
# Confirmar com testes formais:
jb(m_a)
adtest(auto, resid_a)
lilliefors(auto, resid_a)

# QQ da série bruta vs transformada
qqplot(auto, price,      title="QQ-plot — price (nível)")
generate auto log_price = log(price)
qqplot(auto, log_price,  title="QQ-plot — log(price)")
# log(price) deve ser mais próximo da Normal

# QQ de resíduos de séries temporais
# qqplot/jb on ARIMA residuals — predict not supported for ARIMA models
# qqplot(mac, resid_ar, title="QQ-plot — Resíduos AR(2) do ΔPIB")
# jb(m_ar)

# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRPLOT — matriz de correlação como heatmap
# ══════════════════════════════════════════════════════════════════════════════
# corrplot(df, var1, var2, ...) — sem limite de variáveis
# Alias: corr_heatmap, pwcorr_plot
#
# Para cada par (i,j), exibe r_{ij} com bloco de escala visual:
#   ████ |r| ≥ 0.9   (muito alta)
#   ▓▓▓▓ |r| ≥ 0.7   (alta)
#   ▒▒▒▒ |r| ≥ 0.5   (moderada)
#   ░░░░ |r| ≥ 0.3   (fraca)
#        |r| < 0.3   (negligenciável)
# Sinal: + (positivo) ou - (negativo) antes do bloco
#
# Útil para: detectar multicolinearidade, correlações espúrias,
# selecionar variáveis antes de PCA, identificar confounders

corrplot(auto, price, mpg, weight)
# Esperado: price-weight positivo alto; price-mpg negativo; mpg-weight negativo

# Painel macro: correlações contemporâneas
corrplot(mac, pib, inflation, juros)
# Relação Phillips: pib-inflation? Taylor: inflation-juros?

# Seleção de variáveis antes de OLS
corrplot(auto, price, mpg, weight, length, displacement)
# Se displacement e weight ≈ 1.0 → multicolinearidade grave → remover um

# Diagnóstico pós-estimação: regressores vs resíduos
corrplot(auto, resid_a, weight, mpg)
# Se resíduo correlaciona com regressor → forma funcional errada ou variável omitida

# ── Sequência visual completa de análise exploratória ────────────────────────

# 1. Distribuição de cada variável
histogram(auto, price,  bins=15)
histogram(auto, mpg,    bins=12)
histogram(auto, weight, bins=12)

# 2. QQ para testar normalidade
qqplot(auto, price, title="QQ — price")

# 3. Relações bivariadas
scatter(auto, weight, price,   title="price vs weight")
scatter(auto, mpg,    price,   title="price vs mpg")

# 4. Matriz de correlação (visão global)
corrplot(auto, price, mpg, weight, length)

# 5. Série temporal: ACF antes de modelar
acfplot(mac, pib,    lags=16, title="ACF PIB — estacionariedade?")
pacfplot(mac, pib,  lags=16, title="PACF PIB — ordem AR?")

# 6. Ajustar ETS e verificar resíduos
predict v resid2 = m_hw, "residuals"
acfplot(v,  resid2,    lags=20, title="ACF Resíduos HW")
qqplot(v,   resid2,    title="QQ Resíduos HW")
histogram(v, resid2,   bins=15, title="Distribuição Resíduos HW")
