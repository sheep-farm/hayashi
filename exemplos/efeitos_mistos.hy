# MixedLM — Modelos Lineares de Efeitos Mistos
# Para dados agrupados/hierárquicos com efeitos fixos e aleatórios
# Diferença de RE (Random Effects): MixedLM maximiza verossimilhança diretamente (ML/REML)

# ══════════════════════════════════════════════════════════════════════════════
# Terminologia
# ══════════════════════════════════════════════════════════════════════════════
# Efeitos fixos (β): coeficientes comuns a todos os grupos (como RE/FE)
# Efeitos aleatórios (u_i): desvios grupo-específicos ~N(0, D)
# Modelo: y_ij = X_ij β + Z_ij u_i + ε_ij
#
# id=    : coluna com identificador do grupo (escola, indivíduo, empresa)
# re=[]  : variáveis com slopes aleatórios além do intercept
#          se omitido → apenas intercept aleatório (modelo mais simples)

# ══════════════════════════════════════════════════════════════════════════════
# Exemplo 1: dados de alunos em escolas (dois níveis)
# ══════════════════════════════════════════════════════════════════════════════
load "alunos.csv" as alunos

# Variáveis:
#   nota    : nota do aluno (outcome)
#   ses     : status socioeconômico (contínuo)
#   genero  : 0/1
#   escola  : id da escola (nível 2)

# ── Intercept aleatório por escola ────────────────────────────────────────────
# As escolas têm níveis baseline de nota diferentes (u_i ~ N(0, σ_u²))
# Efeitos fixos de ses e genero são comuns a todas
let m_ri = mixed(nota ~ ses + genero, alunos, id="escola")
print(m_ri)

# ── Intercept + slope aleatório de ses ────────────────────────────────────────
# O efeito de ses TAMBÉM varia entre escolas
# u_i = [u_{0i}, u_{1i}]' ~ N(0, D) com D 2×2
let m_rs = mixed(nota ~ ses + genero, alunos, id="escola", re=["ses"])
print(m_rs)

# Comparação com RE clássico (GLS)
let m_re = re(nota ~ ses + genero, alunos, id=escola, time=ano)
esttab(m_re, m_ri, m_rs)

# ── Predição: Xβ (parte fixa, sem efeitos aleatórios) ─────────────────────────
predict alunos yhat_fixed = m_ri, xb
correlate(alunos, nota, yhat_fixed)

# ══════════════════════════════════════════════════════════════════════════════
# Exemplo 2: medidas repetidas por indivíduo (longitudinal)
# ══════════════════════════════════════════════════════════════════════════════
load "painel.csv" as painel

# Intercept aleatório → captura heterogeneidade individual não observada
let m_long = mixed(lucro ~ alavancagem + tamanho, painel, id="empresa")
print(m_long)

# WLS como complemento: peso por número de observações por grupo
#   (útil quando grupos têm tamanhos muito diferentes)
load "painel.csv" as painel2
generate painel2 n_obs = 1
let m_wls = wls(lucro ~ alavancagem + tamanho, painel2, weights="n_obs")
print(m_wls)
