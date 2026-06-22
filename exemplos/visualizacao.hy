# Visualização ASCII no Terminal
# histogram, scatter, lineplot, boxplot, kdensity, residplot

# ══════════════════════════════════════════════════════════════════════════════
# 1. HISTOGRAMA — distribuição de uma variável
# ══════════════════════════════════════════════════════════════════════════════
# histogram(df, var, bins=20, width=50, title="")
#
# Divide a variável em `bins` faixas iguais e exibe barras ██ proporcionais
# à frequência de cada faixa.  Abaixo do plot: min, max, média e desvio padrão.
#
# bins=:  número de barras  (padrão 20; reduza para amostras pequenas)
# width=: largura máxima da barra mais longa em caracteres (padrão 50)
# title=: título opcional (padrão: "Histograma — <var>")

load "auto.csv" as auto

histogram(auto, price)
# output: 20 barras, width=50

histogram(auto, price, bins=10, width=40, title="Distribuição de Preços")

histogram(auto, mpg, bins=15, width=60)
histogram(auto, weight, bins=20)

# ── Comparação com estatísticas descritivas ───────────────────────────────────
summarize(auto, price, mpg, weight)
# Se histograma mostra cauda longa direita: mean >> mediana → skewness positiva
# → confirmar com jb() ou adtest()

# ── Após transformação ────────────────────────────────────────────────────────
generate auto log_price = log(price)
histogram(auto, log_price, bins=15, title="log(price) — mais simétrico?")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SCATTER — gráfico de dispersão bivariado
# ══════════════════════════════════════════════════════════════════════════════
# scatter(df, x, y, width=60, height=20, title="")
#
# Plota pontos (·) numa grade w×h.  Eixos mostram mín/máx.
# Útil para detectar: padrão linear, curvatura, heteroskedasticidade visual,
# outliers, clusters.
#
# width= / height=: dimensões da grade em caracteres

scatter(auto, weight, price)
# Esperado: correlação positiva — carros pesados costumam ser mais caros

scatter(auto, mpg, price, width=70, height=25)
# Esperado: correlação negativa — mais econômicos tendem a ser mais baratos

scatter(auto, mpg, weight, title="Peso vs Eficiência")
# Trade-off: carro pesado consome mais → relação negativa

# ── Correlação entre regressores ──────────────────────────────────────────────
scatter(auto, weight, mpg)
# Detectar multicolinearidade visual antes de vif()

# ── Resíduos vs regressor ─────────────────────────────────────────────────────
let m = ols(price ~ weight + mpg, auto)
predict auto e = m, "residuals"
scatter(auto, weight, e, title="Resíduos vs Weight")
# Padrão em leque → heteroskedasticidade (confirmar com white() ou bphet())
# Curvatura → forma funcional errada (confirmar com reset() ou harveycollier())

# ══════════════════════════════════════════════════════════════════════════════
# 3. LINEPLOT — série temporal ou variável ordenada
# ══════════════════════════════════════════════════════════════════════════════
# lineplot(df, x, y, width=60, height=20, title="")
# Alias: tsplot, twoway
#
# Ordena os pares (x, y) pelo eixo x, marca pontos com ● e conecta com ─.
# Ideal para séries temporais onde x = índice de tempo ou ano.

load "macro_tri.csv" as macro
# Variáveis: t (índice 1..T), pib, inflation, juros

lineplot(macro, t, pib, title="PIB trimestral")
lineplot(macro, t, inflation, width=70, height=18, title="Inflação")
lineplot(macro, t, juros, title="Taxa de Juros — Selic")

# ── Componentes da decomposição ───────────────────────────────────────────────
load "vendas_mensais.csv" as vendas
generate vendas t = _n                          # índice temporal 1..N

let m_stl = stl(vendas, receita, period=12, sw=7)
predict vendas trend_v    = m_stl, "trend"
predict vendas seasonal_v = m_stl, "seasonal"
predict vendas resid_v    = m_stl, "residual"

lineplot(vendas, t, receita,    title="Vendas — série original")
lineplot(vendas, t, trend_v,    title="Vendas — tendência STL")
lineplot(vendas, t, seasonal_v, title="Vendas — componente sazonal")
lineplot(vendas, t, resid_v,    title="Vendas — resíduos STL")

# ── Fator latente DFM ────────────────────────────────────────────────────────
load "macro_painel.csv" as mp
generate mp t = _n
let m_dfm = dfm(mp, pib, consumo, investimento, exportacoes, producao_ind,
                factors=1, order=1)
predict mp atividade = m_dfm, "f1"

lineplot(mp, t, atividade, title="Índice de Atividade Econômica (DFM)")
lineplot(mp, t, pib, title="PIB observado")
# Comparar: atividade ≈ PIB suavizado + co-movimentos dos outros indicadores

# ══════════════════════════════════════════════════════════════════════════════
# 4. BOXPLOT — distribuição com quartis e outliers
# ══════════════════════════════════════════════════════════════════════════════
# boxplot(df, var, width=60, title="")
#
# Representação esquemática:
#
#   ─────├▐│▌┤──────   ○ ○ ○
#       Q1 │ Q3
#        median
#   ├┤ = whiskers (Q1 - 1.5·IQR, Q3 + 1.5·IQR)
#   ○  = outliers além dos whiskers
#
# Abaixo: tabela com min, Q1, median, mean, Q3, max, IQR, lista de outliers

boxplot(auto, price)
# Esperado: cauda direita (preços de luxo = outliers altos)

boxplot(auto, mpg, width=70)
boxplot(auto, weight)

# ── Comparar distribuições entre grupos ──────────────────────────────────────
# Por ora, filter + boxplot por subconjunto
# (futuro: boxplot(df, var, by=grupo))

load "salarios.csv" as sal
# Variáveis: salario, educacao (1=fundamental, 2=medio, 3=superior, 4=pos)

boxplot(sal, salario, title="Distribuição de Salários — Geral")

# Grupo por nível de educação (filter ainda não implementado — usar predict/generate)
# → quando filter estiver disponível:
# filter sal educ1 = educacao == 1
# boxplot(educ1, salario, title="Salários — Fundamental")

# ══════════════════════════════════════════════════════════════════════════════
# 5. KDENSITY — estimativa de densidade por kernel
# ══════════════════════════════════════════════════════════════════════════════
# kdensity(df, var, bw=auto, kernel=gaussian, width=60, height=20, title="")
#
# Kernels disponíveis:
#   gaussian (padrão) — suave, adequado para distribuições unimodais
#   epanechnikov      — ótimo em MSE (mínimo EQM assintótico), padrão do Stata
#   triangular        — intermediário
#   uniform           — rectangular, menos suave
#
# bw=: bandwidth (largura de banda)
#   omitir → regra de Silverman: h = 0.9·min(σ, IQR/1.34)·n^{-1/5}
#   bw=0.5 → valor fixo (sobresuavizado ou subsuavizado conforme os dados)
#
# Útil para: detectar bimodalidade, assimetria, distribuição aproximada

kdensity(auto, price)
kdensity(auto, price, kernel=epanechnikov)
kdensity(auto, price, bw=1000, title="price — bandwidth fixo = 1000")

kdensity(auto, mpg, width=70, height=22)

# ── Comparar kdensity vs histograma ──────────────────────────────────────────
histogram(auto, price, bins=15, title="price — histograma")
kdensity(auto, price, title="price — KDE gaussian")
# Histograma: discreta (depende de bins)
# KDE: contínua (depende de bandwidth)
# Ambos devem mostrar a mesma forma geral

# ── Diagnóstico de normalidade visual ────────────────────────────────────────
let m_ols2 = ols(price ~ mpg + weight, auto)
predict auto resid2 = m_ols2, "residuals"
kdensity(auto, resid2, title="Resíduos OLS — KDE (verificar normalidade)")
# Uma curva aproximadamente em sino → consistente com normalidade
# Caudas pesadas ou assimetria → violação → usar HC3 ou bootstrap

# ══════════════════════════════════════════════════════════════════════════════
# 6. RESIDPLOT — resíduos vs valores ajustados
# ══════════════════════════════════════════════════════════════════════════════
# residplot(model, width=60, height=20)
# Alias: rvfplot
#
# Atalho conveniente: plota e vs ŷ sem precisar extrair colunas manualmente
# Equivale a: predict df e = m, residuals; scatter(df, yhat, e)
#
# Padrões a observar:
#   Dispersão aleatória sem padrão   → homocedasticidade, boa especificação
#   Leque crescente (heterosced.)    → usar white()/bphet() + SE robusto
#   Curvatura sistemática            → forma funcional errada (reset/harveycollier)
#   Outliers extremos                → verificar com influence()

let m3 = ols(price ~ weight + mpg + C(foreign), auto, cov=nonrobust)

residplot(m3)
residplot(m3, width=70, height=25)

# ── Sequência visual completa pós-OLS ────────────────────────────────────────
# 1. residplot  → padrão geral dos resíduos
# 2. kdensity   → formato da distribuição dos resíduos
# 3. boxplot    → outliers nos resíduos
# 4. histogram  → distribuição da variável dependente
# 5. scatter    → cada regressor vs y (para detectar curvatura)
predict auto resid3  = m3, "residuals"
predict auto yhat3   = m3, "xb"

residplot(m3, title="Resíduos vs Ŷ — modelo base")
kdensity(auto, resid3,  title="Resíduos — KDE")
boxplot(auto, resid3,   title="Resíduos — Boxplot")
histogram(auto, resid3, bins=15, title="Resíduos — Histograma")
scatter(auto, weight, price,   title="price vs weight")
scatter(auto, mpg,    price,   title="price vs mpg")

# Complementar com testes formais:
white(m3)
bphet(m3)
reset(m3, power=3)
harveycollier(m3)
jb(m3)

# ══════════════════════════════════════════════════════════════════════════════
# 7. COMBINAÇÕES ÚTEIS
# ══════════════════════════════════════════════════════════════════════════════

# ── Inspecionar série antes de modelar ───────────────────────────────────────
load "vendas_mensais.csv" as v2
generate v2 t2 = _n

histogram(v2, receita, bins=15, title="Receita — distribuição")
boxplot(v2, receita, title="Receita — quartis e outliers")
kdensity(v2, receita, title="Receita — KDE")
lineplot(v2, t2, receita, title="Receita — evolução temporal")

# Se lineplot mostra tendência → diferenciar ou decompor antes de ARIMA
# Se boxplot mostra outliers   → investigar antes de estimar

# ── Markov Switching: probabilidades de regime ────────────────────────────────
load "pib.csv" as pib2
generate pib2 t3 = _n

let m_ms = msauto(pib2, pib_growth, k=2, p=1)
predict pib2 prob_rec = m_ms, "regime2"

lineplot(pib2, t3, pib_growth,  title="PIB — crescimento trimestral")
lineplot(pib2, t3, prob_rec,    title="P(recessão) — Markov AR(1)")
# Picos em prob_rec correspondem a períodos de recessão

# ── UCM: componente de tendência ──────────────────────────────────────────────
load "ibov.csv" as ibov
generate ibov t4 = _n

let m_uc = ucm(ibov, nivel, level=local_linear, seasonal=stochastic, period=12)
predict ibov nivel_uc = m_uc, "level"
predict ibov trend_uc = m_uc, "trend"

lineplot(ibov, t4, nivel,    title="Ibovespa — observado")
lineplot(ibov, t4, nivel_uc, title="Ibovespa — nível UCM")
lineplot(ibov, t4, trend_uc, title="Ibovespa — tendência UCM")
