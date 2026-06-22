# SUR, Rolling OLS, Recursive OLS e critérios de informação (ic)
# Estimação multiquação, coeficientes time-varying e seleção de modelos

# ══════════════════════════════════════════════════════════════════════════════
# 1. SUR — Seemingly Unrelated Regressions (Zellner, 1962)
# ══════════════════════════════════════════════════════════════════════════════
# Quando: múltiplas equações com mesmas unidades, erros correlacionados
# Ganho vs OLS equação-a-equação: SE mais eficientes quando Corr(ε₁, ε₂) ≠ 0
# Sintaxe: sur(df, y1 ~ x1 + x2, y2 ~ x3 + x4, ...)
# Cada equação pode ter regressores distintos (diferente do FE/RE)

input emp
log_investimento log_valor log_capital log_estoque
5.20 6.10 4.80 3.90
5.35 5.95 5.10 4.25
4.90 6.40 4.60 3.70
5.80 6.20 5.30 4.50
5.15 5.80 4.90 3.85
6.00 6.70 5.50 4.70
5.40 6.30 4.75 4.10
5.70 5.90 5.20 4.40
4.85 6.50 4.65 3.60
6.10 6.80 5.60 4.80
5.25 6.00 4.85 3.95
5.60 6.45 5.15 4.30
5.05 5.85 4.70 3.75
5.90 6.60 5.40 4.60
5.30 6.15 4.95 4.05
5.75 6.35 5.25 4.45
4.95 6.55 4.55 3.65
6.05 6.75 5.45 4.75
5.50 6.05 5.00 4.15
5.85 6.90 5.35 4.55
end

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

input macro
pib_growth inflation interest_rate
3.20 2.10 4.50
2.80 2.50 4.75
3.50 1.90 4.25
1.50 3.20 5.00
0.80 3.80 5.50
1.20 3.50 5.25
2.50 2.80 4.80
3.80 2.00 4.30
4.10 1.80 4.00
3.00 2.40 4.60
2.20 3.00 5.10
1.80 3.40 5.30
3.60 2.20 4.40
4.00 1.70 3.90
2.70 2.60 4.70
1.00 3.60 5.40
0.50 4.00 5.80
1.50 3.30 5.20
2.90 2.30 4.50
3.40 2.10 4.20
3.70 1.95 4.10
2.10 2.90 5.00
1.60 3.10 5.15
3.30 2.25 4.45
4.20 1.65 3.85
2.40 2.70 4.65
0.90 3.70 5.60
1.70 3.20 5.10
3.10 2.35 4.55
3.90 1.85 4.15
2.60 2.55 4.70
1.30 3.45 5.35
3.45 2.15 4.35
4.05 1.75 3.95
2.85 2.45 4.60
1.10 3.55 5.45
0.70 3.90 5.70
1.90 3.15 5.05
3.15 2.30 4.50
3.55 2.05 4.25
end

# Variáveis: pib_growth, inflation, interest_rate

# Rolling com janela de 20 períodos (ex: 20 trimestres = 5 anos)
let m_rolling = rolling(pib_growth ~ inflation + interest_rate, macro, window=20)
print(m_rolling)

# Extrair resíduos (one-step-ahead prediction errors)
predict macro resid_roll = m_rolling, "residuals"
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
predict macro cusum_vals  = m_rec, "cusum"
predict macro cusum_sq    = m_rec, "cusum_sq"
predict macro resid_rec   = m_rec, "residuals"

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
input cancer
studytime died drug
1.0 1.0 1.0
2.0 0.0 1.0
3.0 1.0 2.0
4.0 0.0 2.0
5.0 1.0 3.0
6.0 0.0 3.0
7.0 1.0 1.0
8.0 0.0 2.0
end

# Comparar especificações de OLS
let m1 = ols(price ~ mpg, auto)
let m2 = ols(price ~ mpg + weight, auto)
let m3 = ols(price ~ mpg + weight + length + C(foreign), auto)

ic(m1, m2, m3)
# Saída: tabela ordenada por AIC com ΔAIC e pesos de Akaike
# Melhor AIC e BIC identificados explicitamente

# Comparar modelos de contagem
input saude
visitas renda doenca
3.0 2500.0 1.0
0.0 4500.0 0.0
5.0 1800.0 2.0
1.0 3200.0 0.0
7.0 1500.0 3.0
2.0 3800.0 1.0
0.0 5200.0 0.0
4.0 2100.0 2.0
6.0 1200.0 3.0
1.0 4000.0 0.0
3.0 2800.0 1.0
8.0 1100.0 3.0
2.0 3500.0 1.0
0.0 4800.0 0.0
5.0 1600.0 2.0
1.0 3900.0 0.0
4.0 2200.0 2.0
9.0 900.0 3.0
2.0 3300.0 1.0
0.0 5000.0 0.0
3.0 2600.0 1.0
6.0 1400.0 2.0
1.0 4200.0 0.0
7.0 1300.0 3.0
0.0 4600.0 0.0
end
let m_pois  = poisson(visitas ~ renda + doenca, saude)
let m_nb    = negbin(visitas  ~ renda + doenca, saude)
let m_zip   = zip(visitas     ~ renda + doenca, saude)
let m_zinb  = zinb(visitas    ~ renda + doenca, saude)

ic(m_pois, m_nb, m_zip, m_zinb)
# Aqui BIC penaliza zip/zinb pelo parâmetro extra da equação de inflação
