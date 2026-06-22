# Análise Multivariada e Inferência Bootstrap
# pca, factor, manova, bootse

# ══════════════════════════════════════════════════════════════════════════════
# 1. PCA — Análise de Componentes Principais
# ══════════════════════════════════════════════════════════════════════════════
# PCA decompõe a variância total de p variáveis em componentes ortogonais
# Variáveis padronizadas automaticamente (equivalente a PCA da matriz de correlação)
# Útil para:
#   - Redução de dimensionalidade antes de regressão (multicolinearidade)
#   - Construção de índices latentes (riqueza, desenvolvimento, risco)
#   - Análise de portfólio: extração de fatores de risco
#   - Visualização de dados de alta dimensão (biplot)
#
# n=k: primeiros k componentes (padrão: todos)
# Escolha de k: scree plot (cotovelo) ou critério Kaiser (eigenvalue > 1)

input macro
pib_growth inflation unemployment interest_rate trade_balance
2.1 3.5 5.2 4.5 -1.2
1.8 4.0 5.8 5.0 -0.8
3.2 2.8 4.5 3.8 -1.5
0.5 5.1 6.3 6.0 -0.3
2.5 3.2 5.0 4.2 -1.0
1.2 4.5 6.0 5.5 -0.5
3.0 2.5 4.2 3.5 -1.8
0.8 4.8 6.5 5.8 -0.2
2.8 3.0 4.8 4.0 -1.3
1.5 4.2 5.5 5.2 -0.6
3.5 2.2 4.0 3.2 -2.0
0.3 5.5 7.0 6.5 0.1
2.0 3.8 5.3 4.8 -0.9
1.0 4.6 6.2 5.6 -0.4
2.7 3.1 4.9 4.1 -1.1
1.6 4.3 5.7 5.3 -0.7
3.3 2.6 4.3 3.6 -1.6
0.6 5.0 6.8 6.2 -0.1
2.3 3.4 5.1 4.4 -1.0
1.9 4.1 5.6 5.1 -0.8
end
# Variáveis: pib_growth, inflation, unemployment, interest_rate, trade_balance

# PCA completo (todos os componentes)
let m_pca = pca(macro, pib_growth, inflation, unemployment, interest_rate, trade_balance)
print(m_pca)
# Output:
#   Tabela de variância explicada (% e acumulada por componente)
#   Tabela de loadings: contribuição de cada variável por componente

# PCA reduzido: apenas 2 componentes
let m_pca2 = pca(macro, pib_growth, inflation, unemployment, interest_rate, trade_balance, n=2)
print(m_pca2)

# Extrair escores dos componentes como novas colunas
predict macro pc1_scores = m_pca2, "pc1" # projeção no 1º componente
predict macro pc2_scores = m_pca2, "pc2" # projeção no 2º componente
summarize(macro, pc1_scores, pc2_scores)

# Usar escores como regressores (elimina multicolinearidade)
# PC1 pode capturar fator cíclico geral; PC2 pode capturar pressão inflacionária
input bonds
spread1 spread2 spread3 duration yield
1.2 0.8 1.5 3.5 4.2
1.5 1.0 1.8 4.0 4.5
0.9 0.6 1.2 3.0 3.8
1.8 1.2 2.1 4.5 5.0
1.1 0.7 1.4 3.2 4.0
1.6 1.1 1.9 4.2 4.7
0.8 0.5 1.1 2.8 3.5
2.0 1.4 2.3 5.0 5.3
1.3 0.9 1.6 3.6 4.3
1.7 1.2 2.0 4.3 4.8
0.7 0.4 1.0 2.5 3.2
2.2 1.5 2.5 5.2 5.5
1.0 0.7 1.3 3.1 3.9
1.4 1.0 1.7 3.8 4.4
1.9 1.3 2.2 4.7 5.1
0.6 0.3 0.9 2.3 3.0
2.1 1.5 2.4 5.1 5.4
1.2 0.8 1.5 3.4 4.1
1.6 1.1 1.9 4.1 4.6
0.9 0.6 1.2 2.9 3.6
end
let m_pca_bonds = pca(bonds, spread1, spread2, spread3, duration, n=2)
predict bonds factor1 = m_pca_bonds, "pc1"
predict bonds factor2 = m_pca_bonds, "pc2"

let m_bond_ols = ols(yield ~ factor1 + factor2, bonds)
print(m_bond_ols)

# ── PCA para detecção de multicolinearidade ──────────────────────────────────
# Se eigenvalue próximo de 0 → combinação linear quase exata entre variáveis
# Condition number = sqrt(max_eigenvalue / min_eigenvalue)
# CN > 30 → multicolinearidade grave

load "https://www.stata-press.com/data/r9/auto.dta" as auto
let m_pca_auto = pca(auto, price, mpg, weight, length)
print(m_pca_auto)
# Eigenvalue próximo de 0 → variáveis colineares

# ══════════════════════════════════════════════════════════════════════════════
# 2. ANÁLISE FATORIAL
# ══════════════════════════════════════════════════════════════════════════════
# Factor Analysis assume que variáveis observadas são manifestações de fatores latentes
# Modelo: X_j = Σ_f λ_jf F_f + ε_j
#   λ_jf = carga fatorial (loading): influência do fator f na variável j
#   F_f  = fator latente (não observado)
#   ε_j  = unicidade (uniqueness): variância específica de j
#
# Comunalidade (communality) = variância de j explicada pelos fatores
# Unicidade (uniqueness) = 1 - comunalidade
#
# Diferença de PCA:
#   PCA: rotação rígida que maximiza variância total explicada
#   FA:  estrutura de covariância específica com fatores latentes
#   Use PCA para redução de dimensão; FA para descobrir estrutura latente
#
# rotation=none    (padrão): cargas brutas — mais difícil de interpretar
# rotation=varimax : rotação ortogonal — cada variável carrega em um fator
#                   (solução mais interpretável; indicado para construção de índices)

input pesq
q1 q2 q3 q4 q5 q6 q7 q8 q9 q10
4.0 3.0 5.0 2.0 4.0 3.0 5.0 2.0 4.0 3.0
5.0 4.0 4.0 3.0 5.0 4.0 4.0 3.0 5.0 4.0
3.0 2.0 3.0 4.0 3.0 2.0 3.0 4.0 3.0 2.0
4.0 4.0 5.0 1.0 4.0 4.0 5.0 1.0 4.0 4.0
2.0 3.0 2.0 5.0 2.0 3.0 2.0 5.0 2.0 3.0
5.0 5.0 4.0 2.0 5.0 5.0 4.0 2.0 5.0 5.0
3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0
4.0 2.0 5.0 1.0 4.0 2.0 5.0 1.0 4.0 2.0
1.0 1.0 2.0 5.0 1.0 1.0 2.0 5.0 1.0 1.0
5.0 4.0 5.0 2.0 5.0 4.0 5.0 2.0 5.0 4.0
3.0 3.0 4.0 3.0 3.0 3.0 4.0 3.0 3.0 3.0
2.0 2.0 1.0 4.0 2.0 2.0 1.0 4.0 2.0 2.0
4.0 5.0 4.0 1.0 4.0 5.0 4.0 1.0 4.0 5.0
5.0 3.0 5.0 2.0 5.0 3.0 5.0 2.0 5.0 3.0
3.0 4.0 3.0 3.0 3.0 4.0 3.0 3.0 3.0 4.0
1.0 2.0 2.0 5.0 1.0 2.0 2.0 5.0 1.0 2.0
4.0 4.0 4.0 2.0 4.0 4.0 4.0 2.0 4.0 4.0
2.0 1.0 3.0 4.0 2.0 1.0 3.0 4.0 2.0 1.0
5.0 5.0 5.0 1.0 5.0 5.0 5.0 1.0 5.0 5.0
3.0 2.0 4.0 3.0 3.0 2.0 4.0 3.0 3.0 2.0
end
# Variáveis: q1..q10 — questões de uma escala Likert (bem-estar subjetivo)
# Hipótese: 3 fatores latentes (físico, emocional, social)

# Sem rotação
let m_factor = factor(pesq, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, n=3)
print(m_factor)

# Com rotação Varimax (cargas mais interpretáveis)
let m_factor_vx = factor(pesq, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
                          n=3, rotation=varimax)
print(m_factor_vx)
# Interpretar: cada grupo de variáveis com carga alta no mesmo fator → dimensão latente

# Escolha do número de fatores:
# 1. Critério Kaiser: manter fatores com eigenvalue > 1
# 2. Scree plot: cotovelo na curva de eigenvalues
# 3. Percentual de variância explicada: ≥ 60% da variância total

# Exemplo: PCA do mercado financeiro
input ret
ret_x1 ret_x2 ret_x3 ret_x4 ret_x5
0.05 0.03 0.04 0.02 0.06
-0.02 -0.01 -0.03 0.01 -0.02
0.08 0.06 0.07 0.04 0.09
-0.01 0.02 -0.01 0.03 0.00
0.03 0.04 0.02 0.05 0.04
-0.04 -0.03 -0.05 -0.01 -0.03
0.06 0.05 0.06 0.03 0.07
0.01 0.00 0.01 0.02 0.01
-0.03 -0.02 -0.04 0.00 -0.02
0.07 0.06 0.08 0.04 0.08
0.02 0.01 0.03 0.01 0.02
-0.05 -0.04 -0.06 -0.02 -0.04
0.04 0.03 0.05 0.02 0.05
0.00 0.01 -0.01 0.03 0.01
0.09 0.07 0.08 0.05 0.10
-0.02 -0.01 -0.02 0.01 -0.01
0.05 0.04 0.06 0.03 0.06
-0.01 0.00 -0.02 0.02 0.00
0.03 0.02 0.04 0.01 0.03
0.06 0.05 0.07 0.04 0.07
end
# Variáveis: ret_x1..ret_x5 — retornos de 5 ações

let m_fator_fin = factor(ret, ret_x1, ret_x2, ret_x3, ret_x4, ret_x5,
                          n=3, rotation=varimax)
print(m_fator_fin)
# F1 pode ser fator de mercado (beta)
# F2 pode ser fator tamanho (SMB)
# F3 pode ser fator valor (HML)

# ══════════════════════════════════════════════════════════════════════════════
# 3. MANOVA — Análise de Variância Multivariada (one-way)
# ══════════════════════════════════════════════════════════════════════════════
# Generalização do ANOVA para múltiplas variáveis outcome
# H0: μ_1 = μ_2 = ... = μ_k (vetores de média iguais entre grupos)
# Estatísticas de teste:
#   Wilks' Λ      : razão de determinantes ≈ variância não explicada / total
#   Pillai's trace: mais robusto a outliers; preferível com n pequeno
#   Roy's root    : sensível quando efeito concentrado em um componente
#   Hotelling-Lawley: mais potente quando erros normais
# Ao rejeitar H0: usar ANOVA univariado + ajuste Bonferroni para identificar quais outcomes diferem

load "painel.csv" as painel
# Variáveis: lucro, alavancagem, tamanho, empresa (grupo numérico)

# MANOVA: lucro e alavancagem diferem entre empresas?
manova(painel, lucro, alavancagem, by="empresa")
# H0: [E[lucro], E[alavancagem]] igual em todas as empresas

# MANOVA com 3 outcomes
manova(painel, lucro, alavancagem, tamanho, by="empresa")

# Sequência recomendada:
#  1. MANOVA (H0 global) → se p < 0.05, pelo menos um outcome difere
#  2. ANOVA separado para cada outcome → identificar quais
#  3. Comparações par-a-par com ttest + correção Bonferroni

load "https://www.stata-press.com/data/r9/auto.dta" as auto2
manova(auto2, price, mpg, weight, by="foreign")
# H0: distribuição conjunta de (price, mpg, weight) é igual entre nacional e importado

# Separar para confirmar qual diferiu:
anova(auto2, price,  by="foreign")
anova(auto2, mpg,    by="foreign")
anova(auto2, weight, by="foreign")

# ══════════════════════════════════════════════════════════════════════════════
# 4. BOOTSTRAP SE — Inferência por reamostração (pares)
# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap pairs: reamostral linhas (y_i, x_i) com reposição n vezes
# Produz distribuição empírica dos estimadores β̂
# Útil quando:
#   - Erros claramente não-normais e n pequeno
#   - Não quer assumir estrutura de heteroscedasticidade (HC1/HC3)
#   - Quer IC percentil sem hipóteses de normalidade
#
# n=: número de replicações (mínimo 999, recomendado 4999)
# alpha=: nível do IC (padrão: 0.05 → IC 95%)
#
# Saída: β̂, SE original, SE bootstrap, IC percentil inferior/superior

load "https://www.stata-press.com/data/r9/auto.dta" as auto3

let m_ols = ols(price ~ mpg + weight + C(foreign), auto3, cov=nonrobust)
print(m_ols)

# Bootstrap com 1000 replicações (rápido para inspeção)
bootse(m_ols, n=1000)

# Bootstrap com 4999 replicações (publicável)
bootse(m_ols, n=4999)

# Comparar SE:
# SE orig ≈ SE boot → erros aproximadamente normais e bem comportados
# SE boot > SE orig → assimetria ou caudas pesadas nos erros
# SE boot < SE orig → overfitting nos SE originais (raro mas possível com outliers)

# Bootstrap para modelo com regressores suspeitos
load "painel.csv" as painel2
let m_painel_ols = ols(lucro ~ alavancagem + tamanho, painel2, cov=nonrobust)

# Com poucos grupos ou suspeitando de não-normalidade: usar bootstrap
bootse(m_painel_ols, n=4999, alpha=0.05)

# Para modelos com erros robustos: comparar HC3 vs bootstrap
let m_hc3 = ols(lucro ~ alavancagem + tamanho, painel2, cov=HC3)
print(m_hc3)
bootse(m_painel_ols, n=4999)
# Se SE(HC3) ≈ SE(bootstrap) → ambas as abordagens consistentes
