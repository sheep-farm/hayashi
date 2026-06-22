# Séries Temporais: VARMA, Decomposição Sazonal, MSTL
# Testes de Proporção e Múltiplos Testes

# ══════════════════════════════════════════════════════════════════════════════
# 1. VARMA(p, q) — Vetor ARMA multivariado
# ══════════════════════════════════════════════════════════════════════════════
# Extensão do VAR com componentes MA(q): incorpora choques passados na dinâmica
# Modelo: y_t = c + A_1·y_{t-1} + ... + A_p·y_{t-p}
#                 + M_1·ε_{t-1} + ... + M_q·ε_{t-q} + ε_t
#
# Parâmetros:
#   p= : lags AR (padrão: 1)
#   q= : lags MA (padrão: 1)
#
# Diferença de VAR:
#   VAR(p)    = caso especial VARMA(p,0) — sem MA
#   VARMA(p,q): captura padrões de autocovariância que exigem q > 0 MA
#   MA captura persistência de choques além da janela AR — útil em macroeconomia
#
# Identificação: VAR(p) é sempre identificado; VARMA pode ter problemas de
#   cancelamento de raízes (AR/MA) — preferir VAR se n for pequeno
#
# AIC e BIC para comparar com VAR:
#   AIC(VARMA) < AIC(VAR) → componente MA acrescenta poder explicativo

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
end
# Variáveis: pib, inflation, juros — séries trimestrais

# ── VARMA(1,1) base ──────────────────────────────────────────────────────────
let m_varma11 = varma(macro, pib, inflation, juros, p=1, q=1)
print(m_varma11)
# Matrizes A_1 (AR) e M_1 (MA) de dimensão 3×3
# σ_u: matriz de covariância dos choques

# ── VARMA(2,1) — mais lags AR ─────────────────────────────────────────────────
let m_varma21 = varma(macro, pib, inflation, juros, p=2, q=1)
print(m_varma21)
# Comparar AIC vs VARMA(1,1)

# ── Univariado: VARMA degenerado em ARMA ──────────────────────────────────────
# Se k=1 variável → equivale a ARMA(p,q)
# varma requer >= 2 variáveis; para univariado usar arima()
let m_arma = arima(macro, pib, p=1, d=0, q=1)
print(m_arma)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DECOMPOSIÇÃO SAZONAL CLÁSSICA
# ══════════════════════════════════════════════════════════════════════════════
# Decomposição aditiva: y_t = T_t + S_t + R_t
# Decomposição multiplicativa: y_t = T_t × S_t × R_t
#   Use multiplicativa quando variância sazonal cresce com o nível (ex: vendas)
#   Use aditiva quando amplitude sazonal é constante (ex: temperaturas)
#
# Componentes:
#   trend    : tendência de longo prazo (média móvel centrada)
#   seasonal : padrão sazonal estável ao longo dos anos
#   residual : componente irregular (ruído, outliers, ciclos curtos)
#
# period=: comprimento do ciclo sazonal
#   period=12 → mensal (12 meses por ano)
#   period=4  → trimestral (4 trimestres)
#   period=52 → semanal (52 semanas por ano)
#   period=7  → diário com ciclo semanal

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
# Variável: receita — vendas mensais de um varejista

# Decomposição aditiva
let m_dec_add = decompose(vendas, receita, period=12, model=additive)
print(m_dec_add)

# Decomposição multiplicativa (quando a sazonalidade escala com o nível)
let m_dec_mul = decompose(vendas, receita, period=12, model=multiplicative)
print(m_dec_mul)

# Extrair componentes como colunas
predict vendas trend_lin    = m_dec_add, "trend"
predict vendas seasonal_lin = m_dec_add, "seasonal"
predict vendas resid_lin    = m_dec_add, "residual"

summarize(vendas, receita, trend_lin, seasonal_lin, resid_lin)
# Se max(abs(resid_lin)) >> max(abs(seasonal_lin)) → outliers na série

# ── STL — Seasonal and Trend decomposition using LOESS ───────────────────────
# Mais robusto que decomposição clássica:
#   - Lida com sazonalidade que muda ao longo do tempo
#   - Robusto a outliers (iterações de downweighting)
#
# sw= : janela LOESS sazonal (padrão: 7 — deve ser ímpar ≥ 7)
# tw= : janela LOESS tendência (padrão: 0 = auto-seleção)
#   sw maior → sazonalidade mais suave, menos adaptativa
#   tw menor → tendência mais flexível (mais reativa a mudanças)

let m_stl = stl(vendas, receita, period=12, sw=7)
print(m_stl)

predict vendas trend_stl    = m_stl, "trend"
predict vendas seasonal_stl = m_stl, "seasonal"
predict vendas resid_stl    = m_stl, "residual"

# Comparar resíduos STL vs clássica
summarize(vendas, resid_lin, resid_stl)
# STL tende a ter resíduos menores quando sazonalidade não é estável

# ── Remoção de sazonalidade para modelagem ────────────────────────────────────
# Estratégia: dessazonalizar antes de estimar VAR ou ARIMA
generate vendas receita_sa = receita - seasonal_stl   # série dessazonalizada

let m_arima_sa = arima(vendas, receita_sa, p=1, d=1, q=1)
print(m_arima_sa)

# ══════════════════════════════════════════════════════════════════════════════
# 3. MSTL — Múltiplas Sazonalidades
# ══════════════════════════════════════════════════════════════════════════════
# STL iterado para séries com mais de um ciclo sazonal
# Exemplos clássicos:
#   - Dados diários de energia: ciclo semanal (7) + ciclo anual (365)
#   - Dados horários de tráfego: ciclo diário (24) + semanal (168)
#   - Dados diários de varejo: semanal (7) + mensal (30) + anual (365)
#
# O MSTL decompõe iterativamente:
#   1. Estima S_1 (período curto) via STL nos resíduos do trend
#   2. Estima S_2 (período longo) via STL nos resíduos de (y - S_1 - trend)
#   3. Refina trend com todos os componentes sazonais removidos
#
# periods=: lista de períodos em ordem crescente
# predict seasonal   → componente do primeiro período
# predict seasonal1  → idem (explícito)
# predict seasonal2  → componente do segundo período

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
# Variável: consumo_mwh — consumo de energia elétrica em MWh (dados diários)

# MSTL com ciclo semanal (7) — anual (365) requer dados com >= 365 observações
let m_mstl = mstl(energia, consumo_mwh, periods=[7])
print(m_mstl)

# Extrair componentes
predict energia trend_e     = m_mstl, "trend"
predict energia sazon_sem   = m_mstl, "seasonal1" # ciclo semanal (dom < seg...)
predict energia resid_e     = m_mstl, "resid"

summarize(energia, consumo_mwh, trend_e, sazon_sem, resid_e)
# trend_e: crescimento tendencial do consumo
# sazon_sem: padrão de final de semana (consumo cai no domingo)
# resid_e: choques aleatórios (feriados, temperaturas extremas)

# Dessazonalização (remover ciclo semanal)
generate energia consumo_sa = consumo_mwh - sazon_sem

# Dados horários de tráfego — requer ~500+ linhas para periods=[24, 168]
# Exemplo comentado por necessitar de arquivo externo (trafego_hora.csv)
# load "trafego_hora.csv" as trafego
# let m_mstl_traf = mstl(trafego, volume_veiculos, periods=[24, 168])
# print(m_mstl_traf)
# predict trafego sazon_diario  = m_mstl_traf, "seasonal1"
# predict trafego sazon_semanal = m_mstl_traf, "seasonal2"
# summarize(trafego, volume_veiculos, sazon_diario, sazon_semanal)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TESTES DE PROPORÇÃO
# ══════════════════════════════════════════════════════════════════════════════
# proptest(count, n, mu=0.5)    — z-test 1 amostra: H₀: p = mu
# proptest2(c1,n1,c2,n2)        — z-test 2 amostras: H₀: p₁ = p₂
# propci(count, n, alpha=0.05)  — IC Wilson para proporção
# chisq2x2(a,b,c,d)             — qui-quadrado em tabela 2×2

# ── 1 amostra ────────────────────────────────────────────────────────────────
# Exemplo: taxa de aprovação de crédito — H₀: p = 0.60 (benchmark do setor)
# Observado: 73 aprovações em 120 solicitações → p̂ = 0.608
proptest(73, 120, mu=0.60)
# Se p > 0.05 → não rejeitar H₀; taxa ≈ benchmark do setor

# Exemplo: proporção de inadimplência — H₀: p = 0.05
proptest(8, 200, mu=0.05)
# p̂ = 4% — menor que 5%; é estatisticamente menor?

# ── IC para uma proporção ─────────────────────────────────────────────────────
# Wilson Score: preferível ao Wald (p̂ ± z·√(p̂q̂/n)) para n pequeno
propci(73, 120, alpha=0.05)    # IC 95%
propci(8, 200, alpha=0.10)     # IC 90% — mais largo

# ── 2 amostras ───────────────────────────────────────────────────────────────
# H₀: taxa de conversão do grupo A = grupo B (teste A/B)
# Grupo A (controle): 45/300 converteram
# Grupo B (tratamento): 67/290 converteram
proptest2(45, 300, 67, 290)
# Se p < 0.05 → diferença significativa → grupo B superior

# Exemplo: inadimplência entre regiões
# Norte: 12/150 inadimplentes; Sul: 5/200 inadimplentes
proptest2(12, 150, 5, 200)

# ── Qui-quadrado 2×2 ─────────────────────────────────────────────────────────
# Tabela:         | Fumante | Não Fumante |
#     Câncer      |   84    |     84      |
#     Sem câncer  |   84    |   280       |
#
# H₀: fumo e câncer são independentes
chisq2x2(84, 84, 84, 280)
# RR ≠ OR; qui-quadrado testa associação sem quantificá-la

# Exemplo 2: inadimplência × renda
#          | Alta renda | Baixa renda |
# Paga     |   120      |     80      |
# Não paga |    30      |     70      |
chisq2x2(120, 80, 30, 70)

# ══════════════════════════════════════════════════════════════════════════════
# 5. MÚLTIPLOS TESTES — Correção da taxa de erro tipo I
# ══════════════════════════════════════════════════════════════════════════════
# Problema: ao testar k hipóteses com α=5%, a probabilidade de ao menos
#   um falso positivo é 1 - (1-0.05)^k → cresce rapidamente com k
#
# Métodos:
#   bonferroni      : p_adj = min(k·p, 1)      — mais conservador; FWER
#   sidak           : p_adj = 1-(1-p)^k        — levemente menos conservador; FWER
#   holm            : sequencial Bonferroni     — uniformemente mais potente que Bonferroni
#   bh              : Benjamini-Hochberg        — controla FDR (falso discovery rate)
#   by              : Benjamini-Yekutieli       — FDR com dependência arbitrária
#
# Quando usar:
#   FWER (bonferroni/holm): quando qualquer falso positivo é grave (ensaios clínicos)
#   FDR  (bh/by):            quando há muitas hipóteses e tolera-se alguns FP (genômica)
#
# Input: lista de p-values de testes independentes (ou correlacionados para by)

# ── Exemplo 1: testes de coeficientes em múltiplas regressões ────────────────
input painel
retorno tamanho alavancagem liquidez beta bm_ratio
0.12 8.5 0.35 1.20 0.95 0.80
0.08 7.2 0.55 0.85 1.30 0.65
0.15 9.1 0.28 1.50 0.70 1.10
-0.03 6.8 0.72 0.60 1.50 0.45
0.10 8.0 0.40 1.10 1.05 0.75
0.05 7.5 0.60 0.75 1.25 0.55
0.18 9.5 0.22 1.65 0.65 1.20
-0.02 6.5 0.78 0.50 1.55 0.40
0.09 7.8 0.45 1.00 1.10 0.70
0.14 8.8 0.32 1.35 0.80 0.95
0.06 7.0 0.58 0.80 1.35 0.60
0.11 8.3 0.38 1.15 0.90 0.85
-0.01 6.6 0.75 0.55 1.45 0.42
0.16 9.3 0.25 1.55 0.68 1.15
0.07 7.4 0.52 0.90 1.20 0.62
0.13 8.6 0.30 1.30 0.85 0.90
0.04 7.1 0.62 0.70 1.40 0.50
0.17 9.4 0.20 1.60 0.60 1.18
0.02 6.9 0.68 0.65 1.42 0.48
0.10 8.1 0.42 1.05 1.00 0.78
0.08 7.6 0.50 0.88 1.15 0.68
0.15 9.0 0.27 1.45 0.75 1.05
-0.04 6.4 0.80 0.48 1.58 0.38
0.11 8.2 0.36 1.18 0.92 0.82
0.06 7.3 0.56 0.82 1.28 0.58
end

# Estima 5 modelos separados (uma variável de interesse por vez)
let m1 = ols(retorno ~ tamanho,      painel)
let m2 = ols(retorno ~ alavancagem,  painel)
let m3 = ols(retorno ~ liquidez,     painel)
let m4 = ols(retorno ~ beta,         painel)
let m5 = ols(retorno ~ bm_ratio,     painel)

# p-values dos coeficientes de interesse (primeiro regressor de cada modelo)
# (extraído manualmente do print() de cada modelo)
let pvals = [0.031, 0.048, 0.012, 0.072, 0.044]

# Sem correção: 3 de 5 significativos a 5%
# Com correção Bonferroni (mais conservador):
multipletests(pvals, method=bonferroni, alpha=0.05)

# Holm é uniformemente mais potente que Bonferroni:
multipletests(pvals, method=holm, alpha=0.05)

# Benjamini-Hochberg controla FDR (menos conservador):
multipletests(pvals, method=bh, alpha=0.05)

# ── Exemplo 2: testes de heteroskedasticidade em múltiplos modelos ───────────
# Ao rodar White test em 10 especificações diferentes, sem correção esperamos
# 0.5 falsos positivos; mas a interpretação viesada tende a selecionar sig.
let pvals_het = [0.04, 0.08, 0.002, 0.21, 0.06, 0.15, 0.03, 0.09, 0.11, 0.01]
multipletests(pvals_het, method=bh, alpha=0.10)
# FDR=10%: tolerar até 10% de falsos positivos entre os rejeitados

# ── Exemplo 3: comparações de médias entre grupos ─────────────────────────────
# ANOVA rejeitou H₀ global → testar pares com correção
# 4 grupos: 6 comparações par-a-par (4 escolhe 2)
# p-values dos 6 testes t par-a-par:
let pvals_pares = [0.002, 0.048, 0.321, 0.019, 0.084, 0.412]
multipletests(pvals_pares, method=holm, alpha=0.05)
multipletests(pvals_pares, method=bonferroni, alpha=0.05)
# Holm rejeita mais; Bonferroni é mais conservador
