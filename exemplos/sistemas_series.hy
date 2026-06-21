# SUR, Rolling OLS, Recursive OLS e critérios de informação (ic)
# Estimação multiquação, coeficientes time-varying e seleção de modelos

# ══════════════════════════════════════════════════════════════════════════════
# 1. SUR — Seemingly Unrelated Regressions (Zellner, 1962)
# ══════════════════════════════════════════════════════════════════════════════
# Quando: múltiplas equações com mesmas unidades, erros correlacionados
# Ganho vs OLS equação-a-equação: SE mais eficientes quando Corr(ε₁, ε₂) ≠ 0
# Sintaxe: sur(df, y1 ~ x1 + x2, y2 ~ x3 + x4, ...)
# Cada equação pode ter regressores distintos (diferente do FE/RE)

load "empresa.csv" as emp

# Variáveis: log_investimento, log_valor, log_capital, log_estoque
# Duas equações de demanda: investimento e estoque determinados conjuntamente

let m_sur = sur(emp,
                log_investimento ~ log_valor + log_capital,
                log_estoque      ~ log_valor + log_capital)
print(m_sur)

# Exibe: matriz de correlação cruzada Σ (diagnóstico de ganho de SUR vs OLS)
# Se Σ off-diagonal ≈ 0 → SUR ≈ OLS (sem ganho)
# Se Σ off-diagonal grande → SUR tem SE menores (eficiência de Zellner)

# Comparar com OLS separado para ver ganho de eficiência
let m_ols1 = ols(log_investimento ~ log_valor + log_capital, emp)
let m_ols2 = ols(log_estoque      ~ log_valor + log_capital, emp)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Rolling OLS — janela deslizante
# ══════════════════════════════════════════════════════════════════════════════
# Cada ponto t usa observações [t-window+1, t]
# Útil para: coeficientes time-varying, testes de estabilidade paramétrica
# Requer: dados ordenados temporalmente

load "macro.csv" as macro

# Variáveis: pib_growth, inflation, interest_rate

# Rolling com janela de 20 períodos (ex: 20 trimestres = 5 anos)
let m_rolling = rolling(pib_growth ~ inflation + interest_rate, macro, window=20)
print(m_rolling)

# Extrair resíduos (one-step-ahead prediction errors)
predict macro resid_roll = m_rolling, residuals
summarize(macro, pib_growth, resid_roll)

# Inspecionar estabilidade: se coeficientes variam muito ao longo do tempo
# → evidência de quebra estrutural → confirmar com recursive OLS

# ══════════════════════════════════════════════════════════════════════════════
# 3. Recursive OLS — janela crescente (CUSUM)
# ══════════════════════════════════════════════════════════════════════════════
# Usa t=1..t em cada passo (janela expande)
# CUSUM: se processo é estável → CUSUM flutua em torno de 0
# Quebra estrutural → CUSUM ultrapassa bandas de confiança

let m_rec = recursive(pib_growth ~ inflation + interest_rate, macro)
print(m_rec)

# CUSUM e CUSUM-of-Squares como colunas para visualização externa
predict macro cusum_vals  = m_rec, cusum
predict macro cusum_sq    = m_rec, cusum_sq
predict macro resid_rec   = m_rec, residuals

# Se CUSUM cruzar bandas ±0.948·√T → rejeita estabilidade a 5%
# (bandas de Brown-Durbin-Evans)
summarize(macro, cusum_vals, cusum_sq)

# ══════════════════════════════════════════════════════════════════════════════
# 4. ic() — tabela de critérios de informação (AIC/BIC)
# ══════════════════════════════════════════════════════════════════════════════
# Compara modelos com mesmo outcome mas especificações diferentes
# AIC penaliza k com 2k; BIC penaliza com k·ln(n) (mais conservador)
# Δ_AIC < 2: suporte substancial para o modelo alternativo
# Pesos de Akaike: prob relativa de cada modelo ser o "verdadeiro"

load "https://www.stata-press.com/data/r9/auto.dta" as auto
load "cancer.csv" as cancer

# Comparar especificações de OLS
let m1 = ols(price ~ mpg, auto)
let m2 = ols(price ~ mpg + weight, auto)
let m3 = ols(price ~ mpg + weight + length + C(foreign), auto)

ic(m1, m2, m3)
# Saída: tabela ordenada por AIC com ΔAIC e pesos de Akaike
# Melhor AIC e BIC identificados explicitamente

# Comparar modelos de contagem
load "saude.csv" as saude
let m_pois  = poisson(visitas ~ renda + doenca, saude)
let m_nb    = negbin(visitas  ~ renda + doenca, saude)
let m_zip   = zip(visitas     ~ renda + doenca, saude)
let m_zinb  = zinb(visitas    ~ renda + doenca, saude)

ic(m_pois, m_nb, m_zip, m_zinb)
# Aqui BIC penaliza zip/zinb pelo parâmetro extra da equação de inflação
