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

input prop
market_share renda preco pub
0.15 3.2 10.5 2.0
0.25 4.1 8.2 3.5
0.10 2.8 12.0 1.0
0.35 4.5 7.5 4.0
0.20 3.5 9.8 2.5
0.30 4.3 8.0 3.8
0.12 3.0 11.5 1.5
0.40 4.8 6.5 4.5
0.18 3.3 10.0 2.2
0.28 4.0 8.5 3.2
0.08 2.5 13.0 0.8
0.45 5.0 6.0 5.0
0.22 3.6 9.5 2.8
0.32 4.2 7.8 3.6
0.14 2.9 11.0 1.2
0.38 4.6 7.0 4.2
0.16 3.1 10.2 1.8
0.42 4.9 6.2 4.8
0.24 3.8 9.0 3.0
0.05 2.3 14.0 0.5
end
# Variáveis: market_share (outcome), renda, preco, pub (regressores)

# Link logit (padrão): η = logit(μ) = log(μ/(1-μ))
let m_beta = betareg(market_share ~ renda + preco + pub, prop)
print(m_beta)

# Link probit: η = Φ⁻¹(μ)
let m_beta_probit = betareg(market_share ~ renda + preco + pub, prop, link=probit)
print(m_beta_probit)

# Comparar com logit binário (inadequado mas referência)
let m_logit = logit(market_share ~ renda + preco + pub, prop)

# ic(m_beta, m_beta_probit)  # ic() não suporta betareg ainda

# Predição
predict prop mu_hat = m_beta, "pr" # E[y|x] ∈ (0,1)
summarize(prop, market_share, mu_hat)

# esttab suporta betareg diretamente
esttab(m_beta, m_beta_probit)
