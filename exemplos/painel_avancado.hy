# Métodos de painel avançados: AB, SysGMM, FE-IV, PCSE, XTGLS, weak_iv
# Datasets:
#   abdata.dta — Arellano & Bond (1991): 140 firmas UK, 1976–1984
#     n (log emprego), w (log salário real), k (log capital), ys (log produto setor)
#   painel.csv — sintético: 10 empresas × 5 anos

# ── 1. Arellano-Bond (AB) ──────────────────────────────────────────────────────
load "https://www.stata-press.com/data/r17/abdata.dta" as ab

summarize(ab, n, w, k, ys)

# Equação de emprego dinâmica:
#   Δnᵢₜ = β₁ Δnᵢ,ₜ₋₁ + β₂ Δwᵢₜ + β₃ Δkᵢₜ + γ Δysᵢₜ + Δεᵢₜ
# Instrumentos: níveis defasados 2+ períodos (GMM-style)
let m_ab = ab(n ~ w + k + ys, ab, id=id, time=year)
print(m_ab)

# Testes de especificação:
# AR(1) esperado negativo, AR(2) deve ser NÃO rejeitado
# Sargan/Hansen: validade dos instrumentos

# ── 2. System GMM ─────────────────────────────────────────────────────────────
# Adiciona equações em nível (instrumentos: diferenças defasadas)
# Mais eficiente que AB quando série muito persistente
# sysgmm: desabilitado temporariamente — matriz de instrumentos singular com abdata
# let m_sys = sysgmm(n ~ w + k + ys, ab, id=id, time=year)
# print(m_sys)
# esttab(m_ab, m_sys)

# ── 3. PCSE (Beck-Katz 1995) ──────────────────────────────────────────────────
# Para painéis N-dominantes: N grande, T pequeno
# Erros padrão corrigidos para correlação cross-seccional e heterocedasticidade
load "painel.csv" as df

let m_pcse = pcse(lucro ~ alavancagem + tamanho, df, id=empresa, time=ano)
print(m_pcse)

# ── 4. Panel GLS (Parks 1967) ─────────────────────────────────────────────────
# Estima Σ̂ inter-painel e faz GLS em dois passos
# hetero: Σ̂ diagonal (heteroscedasticidade entre painéis)
# corr:   Σ̂ completa (+ correlação cross-seccional)
let m_gls_h = xtgls(lucro ~ alavancagem + tamanho, df, id=empresa, time=ano, panels=hetero)
let m_gls_c = xtgls(lucro ~ alavancagem + tamanho, df, id=empresa, time=ano, panels=corr)
print(m_gls_h)
print(m_gls_c)

# ── 5. FE-IV (Fixed Effects 2SLS) ─────────────────────────────────────────────
# Painel com variável endógena e instrumento
# Estrutural: lucro ~ alavancagem + tamanho (alavancagem endógena?)
# Instrumentos: ~ tamanho + setor_idx (setor como instrumento excluído)
# Nota: setor deve estar no df como numérico
# let m_feiv = feiv(lucro ~ alavancagem + tamanho,
#                   ~ tamanho + setor_idx,
#                   df, id=empresa)
# print(m_feiv)

# ── 6. Teste de Instrumentos Fracos (Stock & Yogo 2005) ───────────────────────
# weak_iv(fórmula_estrutural, fórmula_instrumentos, df)
# fórmula_estrutural: variável dependente ~ endógenas + exógenas incluídas
# fórmula_instrumentos: ~ todas_exógenas_incluídas + instrumentos_excluídos
#
# Exemplo com dados cross-section (dados.csv): salario ~ educacao + exp
# Instrumento excluído: educacao_pai (anos de estudo do pai)
# load "dados_iv.csv" as dff
# weak_iv(salario ~ educacao + experiencia,
#         ~ experiencia + educ_pai + educ_mae, dff)

# Com abdata: testar if ys é instrumento fraco para k
weak_iv(n ~ k + w, ~ w + ys, ab)
