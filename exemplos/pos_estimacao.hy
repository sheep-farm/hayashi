# Pós-estimação: margins e esttab
# Datasets: auto.dta e cancer.dta

load "https://www.stata-press.com/data/r9/auto.dta"  as auto
load "https://www.stata-press.com/data/r8/cancer.dta" as cancer

# ── MARGINS ───────────────────────────────────────────────────────────────
# Efeitos marginais médios (AME) após logit e probit

let m_logit  = logit(died  ~ age + C(drug), cancer)
let m_probit = probit(died ~ age + C(drug), cancer)

print(m_logit)
print(m_probit)

# AME: variação média em P(died=1) para uma unidade em cada preditor
margins(m_logit)
margins(m_probit)

# ── ESTTAB ────────────────────────────────────────────────────────────────
# Tabela comparativa: OLS e modelos binários lado a lado

let m_ols    = ols(price ~ mpg + weight + C(foreign), auto, cov=HC3)
let m_ols2   = ols(price ~ mpg + weight, auto, cov=HC3)

# comparação entre especificações OLS
esttab(m_ols2, m_ols)

# comparação logit vs probit
esttab(m_logit, m_probit)

# exporta tabela LaTeX pronta para artigo
esttab(m_logit, m_probit, fmt=latex, path="/tmp/logit_probit.tex")

# exporta OLS
esttab(m_ols2, m_ols, fmt=latex, path="/tmp/ols_specs.tex")
