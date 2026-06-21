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

load "auto.csv" as auto

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

load "nlswork.csv" as nls
# Variáveis: ln_wage, union (0/1), race, age, tenure, ...

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
kruskal(pan, retorno, by=setor)
# Setores com retornos estatisticamente diferentes?

kruskal(auto, price, by=rep78)
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

load "macro_tri.csv" as mac
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
load "firms.csv" as firms
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

load "pnad.csv" as pnad
# Variáveis: renda, peso_amostral, uf, anos_estudo, ...

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

load "auto.csv" as auto2

# Estatísticas básicas de múltiplas variáveis
tabstat(auto2, price, mpg, weight, stats=[mean, sd, n, min, max])

# Com medidas robustas
tabstat(auto2, price, mpg, stats=[p25, median, p75, iqr])

# Por grupo
tabstat(auto2, price, mpg, weight, stats=[mean, sd, n], by=foreign)
# Comparar domésticos (0) vs importados (1) em todas as métricas simultaneamente

# Análise de painel: variação por ano
load "painel.csv" as pan2
tabstat(pan2, retorno, alavancagem, liquidez, stats=[mean, sd, n, p25, p75], by=ano)
# Evolução anual das estatísticas

# Coeficiente de variação: heterogeneidade relativa
tabstat(auto2, price, mpg, weight, stats=[mean, cv])
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

load "nlswork.csv" as nls2
# Painel: id (pessoa), year, ln_wage, union, tenure, age, ...

xtsum(nls2, ln_wage, union, tenure, age, id=id)
# ln_wage: DP(within) > 0 → há variação temporal de salários
# union: DP(within) > 0 → pessoas mudam de status sindical
#         se DP(within) = 0 → status fixo → FE não pode identificar efeito
# age: DP(within) = DP(between) ≈ mesmo — age cresce exatamente 1/ano
#       → quase colinear com efeito fixo de tempo

# Macro painel: países ao longo do tempo
load "macro_painel.csv" as mp2
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

xtsum(nls2, ln_wage, union, tenure, age, id=id)
tabstat(nls2, ln_wage, union, tenure, stats=[mean, sd, n], by=year)
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
