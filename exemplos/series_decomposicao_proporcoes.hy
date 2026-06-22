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

load "macro_tri.csv" as macro
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
let m_arma = varma(macro, pib, p=1, q=1)
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

load "vendas_mensais.csv" as vendas
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

load "energia_diaria.csv" as energia
# Variável: consumo_mwh — consumo de energia elétrica em MWh (dados diários)

# MSTL com dois ciclos: semanal (7) e anual (365)
let m_mstl = mstl(energia, consumo_mwh, periods=[7, 365])
print(m_mstl)

# Extrair componentes
predict energia trend_e     = m_mstl, "trend"
predict energia sazon_sem   = m_mstl, "seasonal1" # ciclo semanal (dom < seg...)
predict energia sazon_anual = m_mstl, "seasonal2" # ciclo anual (jan > jul no BR)
predict energia resid_e     = m_mstl, "resid"

summarize(energia, consumo_mwh, trend_e, sazon_sem, sazon_anual, resid_e)
# trend_e: crescimento tendencial do consumo
# sazon_sem: padrão de final de semana (consumo cai no domingo)
# sazon_anual: sazonalidade climática (inverno vs verão)
# resid_e: choques aleatórios (feriados, temperaturas extremas)

# Dessazonalização total (remover ambos os ciclos)
generate energia consumo_sa = consumo_mwh - sazon_sem - sazon_anual

# Dados horários de tráfego (ex: Waze, pedágio)
load "trafego_hora.csv" as trafego
# Variável: volume_veiculos — contagem por hora

# MSTL: ciclo diário (24h) + ciclo semanal (168h = 24*7)
let m_mstl_traf = mstl(trafego, volume_veiculos, periods=[24, 168])
print(m_mstl_traf)
predict trafego sazon_diario  = m_mstl_traf, "seasonal1"
predict trafego sazon_semanal = m_mstl_traf, "seasonal2"
summarize(trafego, volume_veiculos, sazon_diario, sazon_semanal)

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
load "painel.csv" as painel

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
