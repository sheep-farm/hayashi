# RLM (Regressão Robusta) e GEE (Equações de Estimação Generalizadas)
# Ferramentas de robustez a outliers e correlação intra-cluster

load "https://www.stata-press.com/data/r9/auto.dta" as auto

# ══════════════════════════════════════════════════════════════════════════════
# 1. REGRESSÃO ROBUSTA (RLM — M-Estimadores)
# ══════════════════════════════════════════════════════════════════════════════
# Alternativa ao OLS quando há outliers nas observações
# Algoritmo: IRLS com pesos w(ε) = ψ(ε)/ε dependentes da norma escolhida
#
# norm=huber    — Huber (c=1.345): down-pesa grandes erros linearmente
#                 Padrão. Boa relação eficiência/robustez.
# norm=tukey    — Tukey bisquare (c=4.685): ignora totalmente outliers extremos
#                 Mais robusto que Huber mas pode ter múltiplos mínimos locais
# norm=andrews  — Andrew's wave: similar ao Tukey
# norm=hampel   — Hampel three-part: controle fino via 3 constantes

# OLS como referência
let m_ols = ols(price ~ mpg + weight + C(foreign), auto, cov=HC3)
print(m_ols)

# Huber (padrão — mais eficiente)
# Usa foreign diretamente (já é 0/1) para que predict funcione após a estimação
let m_huber = rlm(price ~ mpg + weight + foreign, auto)
print(m_huber)

# Tukey bisquare (mais robusto a outliers extremos)
let m_tukey = rlm(price ~ mpg + weight + C(foreign), auto, norm=tukey)
print(m_tukey)

# Comparação: coeficientes devem ser próximos se não houver outliers influentes;
# divergência indica que outliers afetam substancialmente o OLS
esttab(m_ols, m_huber, m_tukey)

# Predição pós-RLM
predict auto yhat_rlm = m_huber, "xb"
correlate(auto, price, yhat_rlm)


# ══════════════════════════════════════════════════════════════════════════════
# 2. GEE (Generalized Estimating Equations)
# ══════════════════════════════════════════════════════════════════════════════
# Para dados em painel/cluster com outcome não-Gaussian
# Foco em efeitos marginais populacionais (population-averaged), não sujeito-específicos
# SE robustos (sandwich) por padrão
#
# family:  gaussian (padrão), binomial, poisson
# corr:    independence | exchangeable | ar1 | unstructured
#           independence → ignora correlação intra-cluster (SE robusto ainda válido)
#           exchangeable → correlação homogênea: Cor(y_it, y_is) = ρ (compound symmetry)
#           ar1          → Cor(y_it, y_is) = ρ^|t-s| (decai com distância temporal)
#           unstructured → estima R livremente (requer T > k)

load "painel.csv" as painel

# Outcome contínuo (Gaussian + exchangeable — alternativa ao RE)
let m_gee_gauss = gee(lucro ~ alavancagem + tamanho, painel,
                      id="empresa", family=gaussian, corr=exchangeable)
print(m_gee_gauss)

# Outcome binário — ex: lucro > mediana como indicador
generate painel lucro_pos = (lucro > 0)
let m_gee_bin = gee(lucro_pos ~ alavancagem + tamanho, painel,
                    id="empresa", family=binomial, corr=exchangeable)
print(m_gee_bin)

# Comparação GEE Gaussian vs RE (ambos estimam efeitos populacionais)
let m_re = re(lucro ~ alavancagem + tamanho, painel, id=empresa, time=ano)
esttab(m_re, m_gee_gauss)

# Predição pós-GEE
predict painel yhat_gee = m_gee_gauss, "xb"
