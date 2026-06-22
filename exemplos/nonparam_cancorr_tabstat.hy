# Testes Não-Paramétricos, Correlação Canônica, Estatísticas por Grupo
# spearman, ranksum, kruskal, signrank, bitest, cancorr, summarize_w, tabstat, xtsum

# ══════════════════════════════════════════════════════════════════════════════
# 1. TESTES NÃO-PARAMÉTRICOS
# ══════════════════════════════════════════════════════════════════════════════
# Quando usar métodos não-paramétricos:
#   - n pequeno → assimetria importa mais que em n grande
#   - Outliers extremos que violam normalidade dos resíduos
#   - Dados de ranking/escala ordinal (Likert, percentis)
#   - Verificação de robustez de resultados paramétricos
#
# Comparação com equivalentes paramétricos:
#   spearman  ←→  correlate/pwcorr (Pearson)
#   ranksum   ←→  ttest (2 amostras)
#   kruskal   ←→  oneway ANOVA
#   signrank  ←→  ttest (1 amostra ou pares)
#   bitest    ←→  proptest (binomial)

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# ── Spearman ρ — correlação de rank ─────────────────────────────────────────
# H₀: ρₛ = 0  (variáveis independentes em ranking)
# ρₛ = 1 − 6Σd²ᵢ/(n(n²−1))   para dados sem empates
# Com empates: Pearson sobre os ranks (implementação mais geral)
#
# Diferença de Pearson:
#   Pearson mede associação LINEAR
#   Spearman mede associação MONÓTONA (linear ou não)
#   Exemplo: y = x² em x ∈ [0,∞) → Pearson < 1, Spearman ≈ 1
#
# Estatística de teste: t = ρₛ√((n−2)/(1−ρₛ²)) ~ t(n−2)

spearman(auto, price, weight)
# Esperado: alta correlação positiva (carros pesados custam mais)
# Se ρₛ ≈ r(Pearson) → relação monotônica ≈ linear (pouco a ganhar com Spearman)
# Se ρₛ >> r(Pearson) → curvatura ou outliers importantes

spearman(auto, mpg, price)
# Esperado: correlação negativa (carros eficientes tendem a ser mais baratos)

spearman(auto, mpg, weight)
# Esperado: alta correlação negativa (trade-off peso × eficiência)

# Comparar Spearman vs Pearson (para verificar linearidade)
correlate(auto, price, weight)     # Pearson
spearman(auto, price, weight)      # Spearman
# Se muito diferentes → outliers ou não-linearidade

# ── ranksum — Mann-Whitney U / Wilcoxon Rank-Sum ─────────────────────────────
# H₀: distribuição de y idêntica nos dois grupos (localização igual)
# Equivale a: P(X₁ > X₂) = 0.5
#
# U₁ = Σᵢ Σⱼ 1(x₁ᵢ > x₂ⱼ) — número de pares com x₁ > x₂
# U₂ = n₁·n₂ − U₁
# W = W₁ (soma dos ranks do grupo 1) — equivalente por transformação
#
# Aproximação normal para n > 10:
#   z = (U − μᵤ) / σᵤ   onde μᵤ = n₁n₂/2,  σᵤ = √(n₁n₂(n₁+n₂+1)/12)
#
# Correto: by= deve ter exatamente 2 valores únicos
# (para k > 2 grupos: usar kruskal)

load "https://www.stata-press.com/data/r8/nlswork.dta" as nls
# Variáveis: ln_wage, union (0/1), race, age, tenure, ...
let nls = dropna(nls, ln_wage, union, tenure, age)

ranksum(nls, ln_wage, by=union)
# H₀: distribuição de ln_wage igual entre sindicalizados e não-sindicalizados
# Se p < 0.05 → salários diferentes entre os grupos (sentido não especificado)
# Complementar com ttest para verificar direção

# Teste paramétrico equivalente para comparar:
ttest(nls, ln_wage, by=union)
# Resultado semelhante → se amostra grande e distribuição razoável

# Exemplo 2: preço de carros por fabricante de origem
# (foreign = 0 doméstico, 1 importado)
ranksum(auto, price, by=foreign)
# H₀: preços iguais entre domésticos e importados

# ── kruskal — Kruskal-Wallis (≥ 2 grupos) ────────────────────────────────────
# Generalização do ranksum para k grupos
# H statistic: H = 12/(N(N+1)) · Σₖ nₖ(R̄ₖ − R̄)²   onde R̄ = (N+1)/2
# H ~ χ²(k−1) sob H₀
#
# H₀: mesma distribuição em todos os grupos
# Rejeitando H₀: pelo menos um grupo tem localização diferente
# Post-hoc: pares via ranksum com correção de múltiplos testes

load "painel.csv" as pan

# Kruskal entre setores econômicos
kruskal(pan, lucro, by=empresa)
# Empresas com lucros estatisticamente diferentes?

let auto_clean = dropna(auto, rep78)
kruskal(auto_clean, price, by=rep78)
# rep78: rating de reparação (1-5) — preços diferem por qualidade?
# Se H₀ rejeitado → algum nível de qualidade tem preços diferentes

# Post-hoc: comparar pares com ranksum + correção Holm
ranksum(auto, price, by=foreign)
# Para múltiplos grupos: seria necessário múltiplas chamadas ranksum
# + multipletests para controlar FWER

# ── signrank — Wilcoxon Signed-Rank (1 amostra / pares) ──────────────────────
# H₀: mediana(x) = mu0
# Descarta observações com x = mu0
# Ranqueia |x − mu0|, soma dos ranks positivos W+
# W = min(W+, W−)
# Aproximação normal: z = (W − μ_W) / σ_W
#
# Mais potente que sign test pois considera magnitude das diferenças
# Mais robusto que t-test pois não assume normalidade
# Exige que as diferenças sejam simétricas em torno da mediana

# H₀: mediana do preço = 5000 (hipótese arbitrária)
signrank(auto, price, mu0=5000)

# Para dados pareados: create a diferença e testar contra 0
# generate nls diff_wage = ln_wage_t2 - ln_wage_t1
# signrank(nls, diff_wage, mu0=0)

# Verificação pré-salários: mediana de ln_wage = ln(5.0)?
signrank(nls, ln_wage, mu0=1.609)    # ln(5.0) ≈ 1.609

# ── bitest — Binomial / Sign Test ────────────────────────────────────────────
# Modo 1: bitest(count, n, mu=p0)   — teste de proporção via normal approx
# Modo 2: bitest(df, var, mu=mu0)   — sign test (conta + vs -)
#
# Sign test: H₀: P(X > mu0) = 0.5
# Menos potente que signrank mas não assume simetria das diferenças
# Útil quando só a direção da mudança é confiável, não a magnitude

# Modo 1: 35 de 100 trabalhadores aderiram ao plano
bitest(35, 100, mu=0.4)
# H₀: p = 0.40 (40% de adesão)

# Modo 2: sign test sobre ln_wage
bitest(nls, ln_wage, mu=1.609)    # H₀: mediana = e^1.609 ≈ 5

# ══════════════════════════════════════════════════════════════════════════════
# 2. CORRELAÇÃO CANÔNICA (CCA)
# ══════════════════════════════════════════════════════════════════════════════
# Encontra combinações lineares de dois conjuntos de variáveis (X e Y) que
# maximizam a correlação entre elas
#
# Modelo: u = Xα,  v = Yβ,  max corr(u, v)
# Estimação: problema de autovalores de Sxx⁻¹·Sxy·Syy⁻¹·Syx
# Saída:
#   CC1, CC2, ..., CC_min(p,q): correlações canônicas decrescentes
#   x_weights: α (combinação linear de X para cada CC)
#   y_weights: β (combinação linear de Y para cada CC)
#   Wilks' Lambda: Λ = Π(1 − CCᵢ²) — significância conjunta
#   F approx: teste de H₀: não há correlação canônica
#
# Aplicações em economia:
#   - Crescimento (PIB, consumo, investimento) ↔ comércio (exportações, importações)
#   - Macro financeiro (juros, câmbio, risco) ↔ atividade (produção, emprego, crédito)
#   - Características da firma (tamanho, alavancagem) ↔ retornos e risco

input mac
pib consumo investimento juros cambio cds
100.5 65.2 22.3 10.5 3.80 150.0
101.2 66.0 23.1 10.2 3.75 145.0
99.8 64.5 21.8 11.0 3.90 160.0
102.0 67.1 24.0 9.8 3.65 135.0
100.0 65.0 22.0 10.8 3.85 155.0
103.5 68.5 25.2 9.2 3.50 120.0
98.5 63.8 20.5 11.5 4.00 170.0
101.8 66.8 23.5 10.0 3.70 140.0
99.2 64.0 21.2 11.2 3.95 165.0
104.0 69.0 26.0 8.8 3.40 110.0
100.8 65.5 22.8 10.3 3.78 148.0
97.5 63.0 19.8 12.0 4.10 180.0
102.5 67.5 24.5 9.5 3.60 130.0
99.5 64.2 21.5 11.3 3.92 162.0
103.0 68.0 25.0 9.0 3.45 115.0
101.0 65.8 23.0 10.1 3.72 142.0
98.0 63.5 20.0 11.8 4.05 175.0
104.5 69.5 26.5 8.5 3.35 105.0
100.2 65.3 22.5 10.6 3.82 152.0
102.8 67.8 24.8 9.3 3.55 125.0
end
# Variáveis:
#   real_vars: pib, consumo, investimento
#   financial_vars: juros, cambio, cds

# CCA: variáveis reais ↔ variáveis financeiras
cancorr(mac, xvars=["pib", "consumo", "investimento"],
             yvars=["juros", "cambio", "cds"])
# CC1: primeira correlação canônica (maior)
# Se CC1 alto (> 0.7) → forte associação entre bloco real e financeiro
# Wilks' Lambda próximo de 0 → alta correlação canônica
# p < 0.05 → rejeita H₀ de independência entre os blocos

# Painel de firmas: características ↔ métricas de risco
input firms
tamanho alavancagem liquidez beta volatilidade prob_default
8.5 0.45 1.20 1.10 0.25 0.03
9.2 0.35 1.50 0.90 0.20 0.02
7.8 0.55 0.90 1.30 0.30 0.05
10.0 0.30 1.80 0.75 0.15 0.01
8.0 0.50 1.00 1.25 0.28 0.04
9.5 0.40 1.60 0.85 0.18 0.02
7.5 0.60 0.80 1.40 0.35 0.06
8.8 0.42 1.30 1.05 0.22 0.03
9.8 0.32 1.70 0.80 0.16 0.01
7.2 0.65 0.70 1.50 0.38 0.07
8.3 0.48 1.10 1.15 0.26 0.04
9.0 0.38 1.40 0.95 0.21 0.02
7.0 0.70 0.60 1.60 0.40 0.08
10.5 0.25 2.00 0.70 0.12 0.01
8.6 0.44 1.25 1.08 0.24 0.03
9.3 0.36 1.55 0.88 0.19 0.02
7.6 0.58 0.85 1.35 0.32 0.05
8.1 0.52 1.05 1.20 0.27 0.04
9.6 0.33 1.65 0.82 0.17 0.01
7.3 0.62 0.75 1.45 0.36 0.06
end
# Características: tamanho, alavancagem, liquidez
# Risco: beta, volatilidade, prob_default

cancorr(firms, xvars=["tamanho", "alavancagem", "liquidez"],
               yvars=["beta", "volatilidade", "prob_default"])
# CC1: qual dimensão das características mais associada ao risco?
# x_weights: peso de cada característica na CC1
# y_weights: peso de cada medida de risco na CC1

# Abordagem com todas as variáveis positionals (sem xvars=):
# cancorr(mac, pib, consumo, yvars=["juros","cambio"])

# ══════════════════════════════════════════════════════════════════════════════
# 3. ESTATÍSTICAS PONDERADAS
# ══════════════════════════════════════════════════════════════════════════════
# summarize_w(df, var, weight=wvar, mu0=mu, alpha=0.05)
#
# Usa DescrStatsW com pesos de confiabilidade (reliability weights):
#   Var(X̄) = σ²/Σwᵢ   (pesos como repetições fracionais)
# t-test: t = (X̄ − μ₀) / SE(X̄)  com df = n − 1
#
# Quando usar pesos:
#   Dados de pesquisa amostral (survey weights)
#   Regressão WLS: usar resíduos ponderados
#   Dados com heteroskedasticidade conhecida (pesos = 1/σᵢ²)

input pnad
renda peso_amostral anos_estudo
1500.0 1.2 8.0
2800.0 0.9 12.0
900.0 1.5 5.0
3500.0 0.8 15.0
1200.0 1.3 7.0
4200.0 0.7 16.0
800.0 1.6 4.0
2200.0 1.0 11.0
1800.0 1.1 9.0
5000.0 0.6 18.0
1000.0 1.4 6.0
3000.0 0.85 13.0
700.0 1.7 3.0
2500.0 0.95 12.0
1600.0 1.2 8.0
3800.0 0.75 15.0
1100.0 1.35 6.0
4500.0 0.65 17.0
2000.0 1.05 10.0
600.0 1.8 2.0
end
# Variáveis: renda, peso_amostral, anos_estudo

# Estatísticas ponderadas da renda
summarize_w(pnad, renda, weight=peso_amostral)
# Média ponderada (representa a população, não a amostra)
# t-test H₀: média = 0 (automático, mu0=0 por padrão)

# Com teste específico
summarize_w(pnad, renda, weight=peso_amostral, mu0=2500, alpha=0.05)
# H₀: renda média da população = R$ 2500
# IC 95% da média populacional

summarize_w(pnad, anos_estudo, weight=peso_amostral)
# Escolaridade média ponderada

# Comparar com não-ponderada
summarize(pnad, renda)
# Se bem diferente → sub/sobrerepresentação na amostra

# ══════════════════════════════════════════════════════════════════════════════
# 4. TABSTAT — tabela de estatísticas flexível
# ══════════════════════════════════════════════════════════════════════════════
# tabstat(df, var1, var2, ..., stats=[mean,sd,n,p25,p75,min,max,sum,iqr,cv], by=grupo)
#
# Estatísticas disponíveis:
#   mean/avg : média
#   sd/std   : desvio padrão
#   var      : variância
#   n/count  : número de observações
#   sum      : soma
#   min, max : extremos
#   p25, p50/median, p75, p10, p90 : percentis
#   iqr      : amplitude interquartil (P75 - P25)
#   range    : amplitude (max - min)
#   cv       : coeficiente de variação (σ/|μ|)

load "https://www.stata-press.com/data/r9/auto.dta" as auto2

# Estatísticas básicas de múltiplas variáveis
tabstat(auto2, price, mpg, weight, stats=["mean", "sd", "n", "min", "max"])

# Com medidas robustas
tabstat(auto2, price, mpg, stats=["p25", "median", "p75", "iqr"])

# Por grupo
tabstat(auto2, price, mpg, weight, stats=["mean", "sd", "n"], by=foreign)
# Comparar domésticos (0) vs importados (1) em todas as métricas simultaneamente

# Análise de painel: variação por ano
load "painel.csv" as pan2
tabstat(pan2, lucro, alavancagem, tamanho, stats=["mean", "sd", "n", "p25", "p75"], by=ano)
# Evolução anual das estatísticas

# Coeficiente de variação: heterogeneidade relativa
tabstat(auto2, price, mpg, weight, stats=["mean", "cv"])
# CV = σ/|μ| — maior CV → mais heterogênea relativamente

# ══════════════════════════════════════════════════════════════════════════════
# 5. XTSUM — decomposição within/between
# ══════════════════════════════════════════════════════════════════════════════
# Decompõe a variância de x em três componentes:
#
#   overall  = variação total (obs. × tempo combinados)
#   between  = variação entre entidades (médias por entidade)
#              DP(between) = DP das médias de grupo x̄ᵢ
#   within   = variação dentro de cada entidade ao longo do tempo
#              x_{it} − x̄ᵢ + x̄ (re-centrado na média overall)
#              DP(within) = DP dos desvios dentro do grupo
#
# Por que isso importa?
#   FE (within): identifica β a partir da variação within
#   BE (between): identifica β a partir da variação between
#   RE: combina ambas (eficiente se efeito aleatório ⊥ X)
#
#   Se DP(within) >> DP(between): X varia mais ao longo do tempo
#     → FE tem boa identificação
#   Se DP(between) >> DP(within): X varia mais entre firmas/países
#     → FE tem identificação fraca (pouca variação temporal para explorar)
#     → considerar BE ou entre-variação como instrumento

load "https://www.stata-press.com/data/r8/nlswork.dta" as nls2
# Painel: id (pessoa), year, ln_wage, union, tenure, age, ...
let nls2 = dropna(nls2, ln_wage, union, tenure, age)

xtsum(nls2, ln_wage, union, tenure, age, id=idcode)
# ln_wage: DP(within) > 0 → há variação temporal de salários
# union: DP(within) > 0 → pessoas mudam de status sindical
#         se DP(within) = 0 → status fixo → FE não pode identificar efeito
# age: DP(within) = DP(between) ≈ mesmo — age cresce exatamente 1/ano
#       → quase colinear com efeito fixo de tempo

# Macro painel: países ao longo do tempo
input mp2
pib divida abertura country
100.5 45.0 30.2 1.0
101.2 46.5 31.0 1.0
102.0 47.0 32.1 1.0
103.5 48.2 33.5 1.0
98.5 55.0 25.0 2.0
99.2 56.3 26.2 2.0
100.0 57.1 27.5 2.0
101.0 58.0 28.0 2.0
95.0 65.0 20.0 3.0
96.5 66.2 21.5 3.0
97.0 67.0 22.0 3.0
98.0 68.5 23.0 3.0
105.0 40.0 35.0 4.0
106.2 41.5 36.5 4.0
107.5 42.0 37.0 4.0
108.0 43.0 38.2 4.0
110.0 35.0 40.0 5.0
111.5 36.0 41.5 5.0
112.0 37.2 42.0 5.0
113.5 38.0 43.0 5.0
end
xtsum(mp2, pib, divida, abertura, id=country)
# divida: se DP(between) >> DP(within) → países diferem em nível
#          mas não mudam muito ao longo do tempo
#          → FE absorve o sinal; precisaria de variação temporal
# abertura: se DP(within) alto → liberalizações aconteceram no período

# ── Sequência diagnóstica de painel antes de estimar ─────────────────────────
# 1. xtsum → verificar se há variação within suficiente para FE
# 2. tabstat + by=year → ver evolução temporal
# 3. corrplot → correlações entre regressores (multicolinearidade)
# 4. spearman → robustez das correlações com outliers

xtsum(nls2, ln_wage, union, tenure, age, id=idcode)
tabstat(nls2, ln_wage, union, tenure, stats=["mean", "sd", "n"], by=year)
corrplot(nls2, ln_wage, union, tenure, age)
spearman(nls2, ln_wage, tenure)
spearman(nls2, ln_wage, age)

# ── Sequência de análise não-paramétrica ──────────────────────────────────────
# Quando paramétrico não é confiável:
# 1. histogram + qqplot → ver distribuição
# 2. summarize → verificar outliers (max/min, skewness via mean vs mediana)
# 3. spearman → correlação robusta
# 4. ranksum  → comparação de grupos
# 5. signrank → comparação pareada ou 1-sample

histogram(auto2, price, bins=15, title="price — verificar outliers")
qqplot(auto2, price, title="QQ price — normalidade?")
spearman(auto2, price, weight)
ranksum(auto2, price, by=foreign)
# Se price muito não-normal e n pequeno → ranksum é mais confiável que ttest
