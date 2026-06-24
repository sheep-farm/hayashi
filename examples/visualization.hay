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

load "https://www.stata-press.com/data/r9/auto.dta" as auto

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

input macro
t pib inflation juros
1.0 100.5 3.5 10.5
2.0 101.2 4.0 10.2
3.0 99.8 2.8 11.0
4.0 102.0 5.1 9.8
5.0 100.0 3.2 10.8
6.0 103.5 2.5 9.2
7.0 98.5 4.8 11.5
8.0 101.8 3.0 10.0
9.0 99.2 4.2 11.2
10.0 104.0 2.2 8.8
11.0 100.8 3.8 10.3
12.0 97.5 5.5 12.0
13.0 102.5 3.1 9.5
14.0 99.5 4.6 11.3
15.0 103.0 2.6 9.0
16.0 101.0 3.4 10.1
17.0 98.0 5.0 11.8
18.0 104.5 2.1 8.5
19.0 100.2 3.9 10.6
20.0 102.8 2.9 9.3
end
# Variáveis: t (índice 1..T), pib, inflation, juros

lineplot(macro, t, pib, title="PIB trimestral")
lineplot(macro, t, inflation, width=70, height=18, title="Inflação")
lineplot(macro, t, juros, title="Taxa de Juros — Selic")

# ── Componentes da decomposição ───────────────────────────────────────────────
input vendas
receita
120.0
135.0
150.0
128.0
142.0
160.0
155.0
145.0
138.0
165.0
170.0
190.0
125.0
140.0
158.0
132.0
148.0
168.0
162.0
152.0
143.0
172.0
178.0
198.0
end
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
input mp
pib consumo investimento exportacoes producao_ind
100.5 65.2 22.3 18.5 95.0
101.2 66.0 23.1 19.0 96.5
99.8 64.5 21.8 17.8 94.0
102.0 67.1 24.0 19.5 97.0
100.0 65.0 22.0 18.2 94.5
103.5 68.5 25.2 20.0 98.5
98.5 63.8 20.5 17.0 92.0
101.8 66.8 23.5 19.2 96.0
99.2 64.0 21.2 17.5 93.5
104.0 69.0 26.0 20.5 99.0
100.8 65.5 22.8 18.8 95.5
97.5 63.0 19.8 16.5 91.0
102.5 67.5 24.5 19.8 97.5
99.5 64.2 21.5 17.6 93.8
103.0 68.0 25.0 20.2 98.0
101.0 65.8 23.0 18.9 95.8
98.0 63.5 20.0 16.8 91.5
104.5 69.5 26.5 21.0 99.5
100.2 65.3 22.5 18.6 95.2
102.8 67.8 24.8 19.9 97.8
end
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

input sal
salario educacao
1500.0 1.0
1800.0 1.0
2200.0 2.0
2500.0 2.0
3000.0 2.0
3500.0 3.0
4200.0 3.0
4800.0 3.0
5500.0 3.0
6000.0 4.0
6500.0 4.0
7200.0 4.0
8000.0 4.0
1200.0 1.0
2000.0 1.0
2800.0 2.0
3200.0 2.0
4000.0 3.0
5000.0 3.0
7500.0 4.0
end
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
# white(m3)   # singular com C(foreign) — usar bphet
bphet(m3)
reset(m3, power=3)
harveycollier(m3)
jb(m3)

# ══════════════════════════════════════════════════════════════════════════════
# 7. COMBINAÇÕES ÚTEIS
# ══════════════════════════════════════════════════════════════════════════════

# ── Inspecionar série antes de modelar ───────────────────────────────────────
input v2
receita
120.0
135.0
150.0
128.0
142.0
160.0
155.0
145.0
138.0
165.0
170.0
190.0
125.0
140.0
158.0
132.0
148.0
168.0
162.0
152.0
143.0
172.0
178.0
198.0
end
generate v2 t2 = _n

histogram(v2, receita, bins=15, title="Receita — distribuição")
boxplot(v2, receita, title="Receita — quartis e outliers")
kdensity(v2, receita, title="Receita — KDE")
lineplot(v2, t2, receita, title="Receita — evolução temporal")

# Se lineplot mostra tendência → diferenciar ou decompor antes de ARIMA
# Se boxplot mostra outliers   → investigar antes de estimar

# ── Markov Switching: probabilidades de regime ────────────────────────────────
input pib2
pib_growth
2.1
1.8
3.2
0.5
2.5
1.2
3.0
0.8
2.8
1.5
3.5
0.3
2.0
1.0
2.7
1.6
3.3
0.6
2.3
1.9
end
generate pib2 t3 = _n

let m_ms = msauto(pib2, pib_growth, k=2, p=1)
# predict pib2 prob_rec = m_ms, "regime2"  # dimension mismatch: msauto(p=1) produz n-1 valores

lineplot(pib2, t3, pib_growth,  title="PIB — crescimento trimestral")
# lineplot(pib2, t3, prob_rec,    title="P(recessão) — Markov AR(1)")
# Picos em prob_rec correspondem a períodos de recessão

# ── UCM: componente de tendência ──────────────────────────────────────────────
input ibov
nivel
50000.0
51200.0
49800.0
52500.0
53000.0
51500.0
54000.0
52800.0
55000.0
53500.0
56000.0
54500.0
57000.0
55500.0
58000.0
56500.0
59000.0
57500.0
60000.0
58500.0
61000.0
59500.0
62000.0
60500.0
end
generate ibov t4 = _n

let m_uc = ucm(ibov, nivel, level=local_linear, seasonal=stochastic, period=12)
predict ibov nivel_uc = m_uc, "level"
predict ibov trend_uc = m_uc, "trend"

lineplot(ibov, t4, nivel,    title="Ibovespa — observado")
lineplot(ibov, t4, nivel_uc, title="Ibovespa — nível UCM")
lineplot(ibov, t4, trend_uc, title="Ibovespa — tendência UCM")
