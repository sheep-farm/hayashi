# SVAR, 3SLS, DFM e Diagnósticos Menores
# svar, sirf, sfevd, threesl, dfm, adtest, lilliefors, omnibus, harveycollier

# ══════════════════════════════════════════════════════════════════════════════
# 1. SVAR — Structural VAR (identificação de choques)
# ══════════════════════════════════════════════════════════════════════════════
# VAR reduzido: y_t = c + A_1·y_{t-1} + ... + A_p·y_{t-p} + u_t
# SVAR impõe estrutura sobre u_t: A·u_t = B·e_t  (e_t = choques estruturais)
#
# O problema de identificação:
#   VAR reduzido estima Σ_u (k×k), mas há k(k-1)/2 restrições necessárias
#   para separar k choques ortogonais e_t a partir de u_t correlacionados
#
# Identificação disponível:
#   id=cholesky  : recursiva — A=I, B=L (Cholesky de Σ_u)
#                  Restrição: ordenação das variáveis importa
#                  Variável 1 → afeta todas; Variável k → afetada por todas
#   id=longrun   : Blanchard-Quah (1989) — restrições de longo prazo
#                  Um choque tem efeito permanente zero sobre uma variável
#                  Clássico: choque de demanda não afeta PIB no longo prazo
#
# Após SVAR:
#   sirf(model, steps=20)  : IRF estrutural (resposta de j ao choque de i)
#   sfevd(model, steps=20) : FEVD estrutural (decomposição da variância do erro)

input macro
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
120.5 3.7 10.7
121.2 4.1 11.1
122.0 3.4 10.4
123.5 4.3 11.3
124.0 3.6 10.6
125.8 3.8 10.8
126.5 4.0 11.0
127.2 3.5 10.5
128.0 4.2 11.2
129.5 3.9 10.9
end
# Variáveis: pib, inflation, juros — séries trimestrais
# Ordenação teórica: pib → inflation → juros
#   (PIB não responde imediatamente a choque de juros no mesmo trimestre)
#   (inflação responde a PIB, mas não a juros no mesmo trimestre)
#   (juros responde a tudo no mesmo trimestre — banco central reativo)

# ── SVAR com identificação recursiva (Cholesky) ──────────────────────────────
let m_svar = svar(macro, pib, inflation, juros, lags=2, id=cholesky)
print(m_svar)
# Matriz A: identidade (restrição recursiva)
# Matriz B: Cholesky lower triangular de Σ_u
#   b11 = choque de oferta (PIB)
#   b21, b22 = choque de demanda (inflation) — contém componente de PIB
#   b31, b32, b33 = choque de política monetária (juros)

# IRF estrutural: como cada variável responde a cada choque?
sirf(m_svar, steps=20)
# Leitura:
#   Coluna 1 (choque de PIB):       PIB sobe, inflation sobe com lag, juros sobe
#   Coluna 2 (choque de demanda):   PIB sobe pouco, inflation sobe, juros sobe
#   Coluna 3 (choque de juros/MP):  PIB cai (puzzle?), inflation cai

# FEVD estrutural: quanto de cada variância é explicado por cada choque?
sfevd(m_svar, steps=20)
# Leitura no horizonte h=20:
#   Fração da variância de PIB explicada por: choque 1 + 2 + 3 = 100%

# ── SVAR Blanchard-Quah: long-run restrictions ───────────────────────────────
# Hipótese: choque de demanda (2ª) tem efeito zero sobre PIB no longo prazo
# Identificação via multiplicador de longo prazo C(1) = (I - A_1 - ... - A_p)^{-1}
input bd
dlpib desemprego
0.008 5.2
0.012 5.0
-0.003 5.5
0.015 4.8
0.005 5.1
-0.008 5.8
0.010 5.3
0.018 4.9
-0.002 5.4
0.007 5.2
0.013 5.0
-0.005 5.6
0.009 5.1
0.016 4.7
0.003 5.3
-0.006 5.7
0.011 5.0
0.014 4.8
-0.001 5.5
0.006 5.2
end
# Variáveis: dlpib (Δ log PIB), desemprego

let m_bq = svar(bd, dlpib, desemprego, lags=2, id=longrun)
print(m_bq)
sirf(m_bq, steps=40)
# Interpretação Blanchard-Quah:
#   Choque 1 (oferta): impacto permanente sobre PIB
#   Choque 2 (demanda): impacto temporário sobre PIB (zero no longo prazo)
#   → separa ciclo de negócios (demanda) vs mudanças estruturais (oferta)

# ── Comparação SVAR vs VAR reduzido ──────────────────────────────────────────
let m_var = var(macro, pib, inflation, juros, lags=2)
irf(m_var, steps=20)   # IRF reduced-form (não identificado estruturalmente)
sirf(m_svar, steps=20) # IRF estrutural (identificado via Cholesky)
# Diferença: reduced-form IRF é mistura de choques; structural IRF isola cada um

# ══════════════════════════════════════════════════════════════════════════════
# 2. 3SLS — Three Stage Least Squares (sistema de equações simultâneas)
# ══════════════════════════════════════════════════════════════════════════════
# Problema: variáveis endógenas como regressores criam correlação entre X e ε
# Solução: 3SLS combina 2SLS (cada equação) com SUR (covariância entre erros)
#
# Estágios:
#   1º: projeta X endógenas no espaço de Z (reduzida) → X̂ = Z(Z'Z)⁻¹Z'X
#   2º: 2SLS equação a equação usando X̂
#   3º: estima Σ (covariância dos erros), aplica GLS entre equações
#       → recupera eficiência da correlação entre os choques das equações
#
# Quando usar:
#   SUR:  sem endogeneidade, mas erros correlacionados entre equações
#   2SLS: endogeneidade, uma equação por vez
#   3SLS: endogeneidade E correlação entre equações → mais eficiente que 2SLS
#
# Sintaxe:
#   threesl(df, y1~x_end+z_exo, y2~x_end+z_exo, instruments=["z1","z2"])
#   instruments=: lista de TODAS as variáveis exógenas (union dos Z de cada eq)
#
# A matrix X de cada equação inclui tanto endógenas quanto exógenas
# Os instrumentos são as exógenas que ficam fora da equação correspondente

input od
q p renda custo tempo
50.0 10.0 3000.0 5.0 1.0
52.0 10.5 3100.0 5.2 2.0
48.0 9.8 2900.0 4.8 3.0
55.0 11.0 3200.0 5.5 4.0
53.0 10.8 3150.0 5.3 5.0
47.0 9.5 2850.0 4.6 6.0
56.0 11.2 3300.0 5.6 7.0
51.0 10.2 3050.0 5.1 8.0
54.0 10.9 3250.0 5.4 9.0
49.0 9.9 2950.0 4.9 10.0
57.0 11.5 3400.0 5.8 11.0
50.0 10.1 3000.0 5.0 12.0
58.0 11.8 3500.0 6.0 13.0
52.0 10.4 3100.0 5.2 14.0
46.0 9.3 2800.0 4.5 15.0
59.0 12.0 3550.0 6.1 16.0
53.0 10.6 3150.0 5.3 17.0
55.0 11.1 3200.0 5.5 18.0
48.0 9.7 2900.0 4.7 19.0
60.0 12.2 3600.0 6.2 20.0
end
# Variáveis:
#   q      : quantidade transacionada (endógena em ambas)
#   p      : preço (endógeno em ambas)
#   renda  : renda dos consumidores (exógena — desloca demanda)
#   custo  : custo de produção (exógeno — desloca oferta)
#   tempo  : tendência temporal (exógena — ambas)
#
# Sistema de oferta e demanda:
#   Demanda: q = α₀ + α₁·p + α₂·renda + ε₁   (p endógeno; renda exógeno)
#   Oferta:  q = β₀ + β₁·p + β₂·custo + ε₂   (p endógeno; custo exógeno)
#
# Identificação: demanda identificada por (custo ∉ demanda)
#               oferta identificada por (renda ∉ oferta)
# Instrumentos globais: renda, custo, tempo

let m_3sls = threesl(od,
    q ~ p + renda,    # equação de demanda
    q ~ p + custo,    # equação de oferta
    instruments=["renda", "custo", "tempo"])
print(m_3sls)
# Eq demanda: α₁ = elasticidade-preço da demanda (negativo)
#             α₂ = efeito renda (positivo)
# Eq oferta:  β₁ = elasticidade-preço da oferta (positivo)
#             β₂ = efeito custo (negativo)
# Σ: correlação dos erros entre equações (estrutura do sistema)

# ── Exemplo 2: modelo IS-LM como sistema simultâneo ──────────────────────────
input islm
y r g m pi
100.0 5.0 20.0 50.0 2.0
102.0 4.8 21.0 52.0 2.1
98.0 5.5 19.0 48.0 1.8
105.0 4.5 22.0 55.0 2.3
103.0 4.7 21.5 53.0 2.2
97.0 5.8 18.5 47.0 1.7
106.0 4.3 23.0 56.0 2.4
101.0 5.1 20.5 51.0 2.0
104.0 4.6 22.0 54.0 2.2
99.0 5.3 19.5 49.0 1.9
107.0 4.2 23.5 57.0 2.5
100.0 5.2 20.0 50.0 2.0
108.0 4.0 24.0 58.0 2.6
102.0 4.9 21.0 52.0 2.1
96.0 5.9 18.0 46.0 1.6
109.0 3.8 24.5 59.0 2.7
103.0 4.7 21.5 53.0 2.1
105.0 4.4 22.5 55.0 2.3
98.0 5.4 19.0 48.0 1.8
110.0 3.6 25.0 60.0 2.8
end
# Variáveis:
#   y      : produto (endógeno)
#   r      : taxa de juros (endógena)
#   g      : gasto governo (exógeno — IS)
#   m      : oferta de moeda (exógeno — LM)
#   pi     : inflação esperada (exógeno — ambas)
#
# IS: y = a₀ + a₁·r + a₂·g + ε_IS
# LM: y = b₀ + b₁·r + b₂·m + ε_LM

let m_islm = threesl(islm,
    y ~ r + g,
    y ~ r + m,
    instruments=["g", "m", "pi"])
print(m_islm)

# ══════════════════════════════════════════════════════════════════════════════
# 3. DFM — Dynamic Factor Model
# ══════════════════════════════════════════════════════════════════════════════
# Assume que k variáveis observadas são dirigidas por r < k fatores latentes:
#   y_t = Λ·f_t + ε_t        (equação de observação: y_t = k×1, Λ = k×r)
#   f_t = A₁f_{t-1} + ... + ε_f  (equação de transição: VAR(p) nos fatores)
#
# Estimação: two-step (PCA para inicializar → Kalman smoother → refinamento)
# Útil para: índices coincidentes/antecedentes, nowcasting, stress tests
#
# Parâmetros:
#   factors=2 : número de fatores latentes (deve ser < n_variáveis)
#   order=1   : ordem AR dos fatores (VAR(p) na equação de transição)
#
# predict fN : extrai o N-ésimo fator latente suavizado (1-indexed)
# factor_loadings: Λ — quais variáveis carregam em quais fatores

input mp
pib consumo investimento exportacoes producao_ind
100.0 65.0 18.0 12.0 95.0
101.5 66.2 18.5 12.3 96.2
103.0 67.5 19.2 12.8 97.5
102.0 66.8 18.8 12.5 96.8
104.5 68.5 19.8 13.2 98.5
106.0 70.0 20.5 13.8 100.0
105.0 69.2 20.0 13.5 99.2
107.5 71.0 21.0 14.2 101.0
109.0 72.5 21.8 14.5 102.5
108.0 71.8 21.2 14.0 101.8
110.5 73.2 22.5 15.0 103.5
112.0 74.5 23.0 15.5 105.0
111.0 73.8 22.5 15.2 104.2
113.5 75.5 23.8 15.8 106.0
115.0 76.8 24.2 16.2 107.5
114.0 76.0 23.8 15.8 106.8
116.5 77.5 25.0 16.5 108.5
118.0 79.0 25.5 17.0 110.0
117.0 78.2 25.0 16.8 109.2
119.5 80.0 26.0 17.5 111.0
end
# Variáveis: pib, consumo, investimento, exportacoes, producao_ind
# (5 indicadores reais da atividade econômica)

# DFM com 1 fator latente (índice coincidente da atividade)
let m_dfm1 = dfm(mp, pib, consumo, investimento, exportacoes, producao_ind,
                 factors=1, order=1)
print(m_dfm1)
# Factor loadings Λ: quanto cada série carrega no fator comum
# Fator 1: provavelmente ciclo geral de atividade econômica

predict mp atividade = m_dfm1, "f1" # índice de atividade latente
summarize(mp, pib, atividade)

# DFM com 2 fatores
let m_dfm2 = dfm(mp, pib, consumo, investimento, exportacoes, producao_ind,
                 factors=2, order=1)
print(m_dfm2)
predict mp fator1 = m_dfm2, "f1" # fator doméstico (consumo + PIB)
predict mp fator2 = m_dfm2, "f2" # fator externo (exportações)
summarize(mp, fator1, fator2)

# DFM para nowcasting: fator latente como variável auxiliar em regressão
let m_now = ols(pib ~ atividade, mp)
print(m_now)

# ── Aplicação: monitoring de risco financeiro ─────────────────────────────────
input spr
cds_brazil cds_mexico cds_argentina embi vix
150.0 120.0 450.0 280.0 18.0
155.0 125.0 460.0 290.0 19.5
145.0 118.0 440.0 275.0 17.0
160.0 130.0 480.0 300.0 21.0
170.0 135.0 500.0 320.0 23.5
165.0 132.0 490.0 310.0 22.0
148.0 122.0 455.0 285.0 18.5
175.0 140.0 520.0 340.0 25.0
180.0 145.0 540.0 350.0 27.0
155.0 128.0 465.0 295.0 20.0
142.0 115.0 430.0 270.0 16.5
185.0 148.0 550.0 360.0 28.5
160.0 130.0 475.0 305.0 21.5
190.0 150.0 560.0 370.0 30.0
170.0 138.0 510.0 330.0 24.0
145.0 120.0 445.0 278.0 17.5
195.0 155.0 580.0 385.0 32.0
175.0 142.0 525.0 345.0 26.0
150.0 125.0 458.0 288.0 19.0
200.0 160.0 600.0 400.0 35.0
end
# Variáveis: cds_brazil, cds_mexico, cds_argentina, embi, vix
# (5 medidas de risco — fator latente = "risk-off" global)

let m_risco = dfm(spr, cds_brazil, cds_mexico, cds_argentina, embi, vix,
                  factors=1, order=1)
print(m_risco)
predict spr risco_global = m_risco, "f1"
summarize(spr, risco_global, vix, cds_brazil)
# Se loadings do VIX > 0 e CDS > 0 → fator comum = aversão ao risco global

# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNÓSTICOS MENORES — Normalidade e forma funcional
# ══════════════════════════════════════════════════════════════════════════════

load "https://www.stata-press.com/data/r9/auto.dta" as auto

let m_ols = ols(price ~ mpg + weight + C(foreign), auto, cov=nonrobust)

# ── Anderson-Darling (AD) — teste de normalidade ──────────────────────────────
# Mais potente que KS nas caudas — detecta afastamento nas caudas da distribuição
# H₀: dados provêm de distribuição normal
# A²* > valor crítico → rejeitar H₀
#
# Quando usar: dados brutos (não resíduos) onde suspeita de não-normalidade
# → para resíduos de OLS, usar jb(), omnibus() ou lilliefors()

predict auto resid_ols = m_ols, "residuals"
adtest(auto, resid_ols)    # AD nos resíduos
adtest(auto, price)        # AD no nível — obviamente não-normal

# ── Lilliefors — KS com parâmetros estimados ──────────────────────────────────
# Kolmogorov-Smirnov adaptado para quando μ e σ são desconhecidos (estimados)
# KS clássico requer distribuição completamente especificada → Lilliefors corrige
# H₀: distribuição normal com média e variância amostrais
#
# Quando usar: antes de testes paramétricos em pequenas amostras
# → mais conservador que AD para n grande; preferível a KS quando parâmetros são estimados

lilliefors(auto, resid_ols)
lilliefors(auto, price)

# ── Omnibus (D'Agostino-Pearson) — normalidade dos resíduos ──────────────────
# K² = Z₁² + Z₂² ~ χ²(2)
# Z₁: componente de assimetria (skewness test de D'Agostino)
# Z₂: componente de curtose   (kurtosis test de Anscombe-Glynn)
# Mais potente que JB para n médio; requer n ≥ 20
#
# Diferença de JB:
#   JB: usa momentos brutos — menos estável para n pequeno
#   Omnibus: usa estatísticas de ordem — mais estável
#   JB: assintoticamente χ²(2), Omnibus: correção de tamanho finito

omnibus(m_ols)   # H₀: resíduos normalmente distribuídos
jb(m_ols)        # para comparar — devem concordar em n grande

# Bateria completa de normalidade:
# adtest  → detecta afastamento nas caudas
# lillie  → KS com parâmetros estimados
# jb      → assimetria + curtose (clássico, n grande)
# omnibus → assimetria + curtose (correção n finito)
# Se todos rejeitam: erros claramente não-normais → usar bootstrap ou HC3

# ── Harvey-Collier — teste de linearidade ────────────────────────────────────
# Computa resíduos recursivos: e_t* = y_t - ŷ_t|t-1 (previsão fora da amostra)
# t-test: H₀: média dos e_t* = 0 (modelo linear bem especificado)
# Se H₀ é rejeitado: o modelo linear sistematicamente subestima/superestima
# → forma funcional errada (quadrática, log, etc.)
#
# Diferença de RESET:
#   RESET: testa se ŷ² e ŷ³ teriam poder explicativo adicional
#   Harvey-Collier: testa via resíduos recursivos — mais robusto para outliers
#   Ambos testam linearidade mas com abordagens diferentes

harveycollier(m_ols)    # H₀: relação entre price e (mpg, weight, foreign) é linear

# Se HC rejeita → tentar transformações:
generate auto log_price  = log(price)
generate auto log_weight = log(weight)
let m_log = ols(log_price ~ mpg + log_weight + C(foreign), auto, cov=nonrobust)
harveycollier(m_log)    # se não rejeita → log-log melhor especificado

# ── Sequência recomendada de diagnóstico pós-OLS ──────────────────────────────
# 1. vif(m_ols)             → multicolinearidade entre regressores
# 2. condnum(m_ols)         → multicolinearidade global (κ da matriz X)
# 3. white(m_ols)           → heteroskedasticidade geral
# 4. bphet(m_ols)           → heteroskedasticidade linear em X
# 5. bgodfrey(m_ols, lags=4) → autocorrelação serial
# 6. reset(m_ols, power=3)  → forma funcional
# 7. harveycollier(m_ols)   → linearidade via resíduos recursivos
# 8. jb(m_ols)              → normalidade dos resíduos
# 9. omnibus(m_ols)         → normalidade (correção n finito)
# 10. influence(m_ols)      → observações influentes

reset(m_ols, power=3)
harveycollier(m_ols)
# white(m_ols)  # Singular matrix com C(foreign) — poucas obs para cross-terms
bphet(m_ols)
bgodfrey(m_ols, lags=4)
jb(m_ols)
omnibus(m_ols)
vif(m_ols)
condnum(m_ols)
influence(m_ols)
