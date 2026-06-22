# Diagnósticos para dados em painel
# Testes de especificação após estimação FE/RE

load "painel.csv" as painel
# Variáveis: lucro, alavancagem, tamanho, empresa (id), ano (time)

# ══════════════════════════════════════════════════════════════════════════════
# Sequência de decisão recomendada:
#
#  1. bptest   → pooled OLS vs RE?        (H0: pooled OLS adequado)
#  2. mundlak  → RE consistente?          (H0: sem correlação X com efeitos)
#  3. hausman  → FE vs RE?                (H0: RE consistente)
#  4. wooldridge → correlação serial?     (H0: sem correlação serial)
#  5. pesaran  → dependência CS?          (H0: sem dependência cross-seccional)
#  6. abtest   → instrumentos GMM ok?     (m1 deve rejeitar, m2 não deve)
# ══════════════════════════════════════════════════════════════════════════════

# Passo 1: pooled OLS adequado?
bptest(painel, lucro ~ alavancagem + tamanho, id="empresa")
# p < 0.05 → usar RE ou FE; p ≥ 0.05 → pooled OLS é suficiente

# Passo 2: RE é consistente? (via Mundlak)
# Adiciona médias individuais X̄_i ao modelo RE e testa γ = 0
mundlak(painel, lucro ~ alavancagem + tamanho, id="empresa")
# p < 0.05 → RE inconsistente → usar FE ou Hausman para confirmar

# Passo 3: confirmação FE vs RE via Hausman
let m_fe = fe(lucro ~ alavancagem + tamanho, painel, id=empresa, time=ano)
let m_re = re(lucro ~ alavancagem + tamanho, painel, id=empresa, time=ano)
hausman(m_fe, m_re)
# χ²(k) grande e p < 0.05 → FE é necessário

# Passo 4: correlação serial nos erros idiossincráticos?
wooldridge(painel, lucro ~ alavancagem + tamanho, id="empresa", time="ano")
# p < 0.05 → correlação serial → usar SE robustos clustered ou PCSE
# Se FE escolhido: re-estimar com cov=HC1 ou usar pcse()

# Passo 5: dependência cross-seccional?
# Relevante quando N grande e T moderado (painel largo)
pesaran(painel, lucro ~ alavancagem + tamanho, id="empresa", time="ano")
# p < 0.05 → choques comuns → usar PCSE ou Driscoll-Kraay SE

# Passo 6: se estimação AB/GMM — validação de instrumentos
abtest(painel, lucro ~ alavancagem + tamanho, id="empresa", time="ano")
# m1 DEVE rejeitar (esperado; FD induz AR(1))
# m2 NÃO deve rejeitar (valida instrumentos defasados y_{t-2})

# ══════════════════════════════════════════════════════════════════════════════
# Beta Regression — outcome em (0,1) estritamente
# ══════════════════════════════════════════════════════════════════════════════
# Exemplos: market share, proporção de dívida, taxa de aprovação
# Distribui sobre Beta(μ·φ, (1-μ)·φ); estima μ e φ (precisão)
# Não aceita exatamente 0 ou 1 — transformar antes se necessário

load "proporcoes.csv" as prop
# Variáveis: market_share (outcome), renda, preco, pub (regressores)

# Link logit (padrão): η = logit(μ) = log(μ/(1-μ))
let m_beta = betareg(market_share ~ renda + preco + pub, prop)
print(m_beta)

# Link probit: η = Φ⁻¹(μ)
let m_beta_probit = betareg(market_share ~ renda + preco + pub, prop, link=probit)
print(m_beta_probit)

# Comparar com logit binário (inadequado mas referência)
let m_logit = logit(market_share ~ renda + preco + pub, prop)

ic(m_beta, m_beta_probit)

# Predição
predict prop mu_hat = m_beta, "pr" # E[y|x] ∈ (0,1)
summarize(prop, market_share, mu_hat)

# esttab suporta betareg diretamente
esttab(m_beta, m_beta_probit)
