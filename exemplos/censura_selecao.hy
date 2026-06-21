# Tobit e Heckman Two-Step
# Dataset: mroz.dta — Mroz (1987), labor supply de mulheres casadas (n=753)
# Variáveis:
#   inlf    — in labor force (0/1)
#   hours   — horas trabalhadas (0 se inlf=0 → censura em zero)
#   lwage   — log-salário (missing se inlf=0 → seleção amostral)
#   educ    — anos de educação
#   exper   — anos de experiência
#   expersq — exper²
#   fatheduc, motheduc — educação dos pais (exclusões para Heckman)
#   nwifeinc — renda não-laboral da mulher
#   kidslt6  — filhos < 6 anos
#   kidsge6  — filhos 6–18 anos

load "https://www.stata-press.com/data/r17/mroz.dta" as df

summarize(df, hours, lwage, educ, exper, expersq)
tabulate(df, inlf)

# ── Tobit ──────────────────────────────────────────────────────────────────────
# Variável dependente: hours (censurada em 0 para mulheres fora do mercado)
# ll=0 → limite inferior de censura em zero (padrão)
# A seleção de não participar é informativa apenas via mecanismo de censura
let m_tobit = tobit(hours ~ nwifeinc + educ + exper + expersq + kidslt6 + kidsge6,
                    df, ll=0)
print(m_tobit)

# ── Heckman Two-Step ───────────────────────────────────────────────────────────
# Trata a ausência de wage como seleção amostral (não MCAR)
# Equação de resultado: lwage ~ educ + exper + expersq  (apenas para inlf=1)
# Equação de seleção:  inlf ~ educ + exper + expersq + nwifeinc + kidslt6 + kidsge6
#                       + fatheduc + motheduc  ← exclusões que identificam o modelo
#
# Etapa 1: Probit em inlf sobre TODAS as variáveis → Razão de Mills Inversa (λ)
# Etapa 2: OLS de lwage ~ X + λ  nos selecionados (inlf=1)
# SE corrigidos pela fórmula analítica de Heckman (1979)
let m_heck = heckman(lwage  ~ educ + exper + expersq,
                     inlf   ~ educ + exper + expersq + nwifeinc + kidslt6 + kidsge6 +
                               fatheduc + motheduc,
                     df)
print(m_heck)

# δ̂ significativo → viés de seleção amostral presente
# δ̂ ≈ 0 → OLS simples seria adequado (sem seleção)

# ── OLS ingênuo como contrafactual ────────────────────────────────────────────
# lwage é missing (NaN) para inlf=0 → dropna seleciona apenas as participantes
let df_sel = dropna(df, lwage)
let m_ols  = ols(lwage ~ educ + exper + expersq, df_sel, cov=HC3)
print(m_ols)

esttab(m_ols, m_heck)
