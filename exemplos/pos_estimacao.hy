# Pós-estimação: predict, margins, esttab
# Datasets: auto.dta, cancer.dta, mroz.dta

load "https://www.stata-press.com/data/r9/auto.dta"   as auto
load "https://www.stata-press.com/data/r8/cancer.dta" as cancer

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT — geração de colunas pós-estimação
# Sintaxe: predict df nova_col = modelo, kind
# ══════════════════════════════════════════════════════════════════════════════

# ── OLS ────────────────────────────────────────────────────────────────────────
let m_ols = ols(price ~ mpg + weight, auto)
predict auto yhat = m_ols, "xb" # valores ajustados (Xβ)
predict auto ehat = m_ols, "residuals" # resíduos
summarize(auto, price, yhat, ehat)
correlate(auto, yhat, ehat)            # deve ser ≈ 0

# ── Logit / Probit → probabilidade predita ─────────────────────────────────────
let m_logit  = logit(died ~ age + C(drug), cancer)
predict cancer pr_logit  = m_logit, "pr" # P(died=1 | X)

let m_probit = probit(died ~ age + C(drug), cancer)
predict cancer pr_probit = m_probit, "pr"

# Comparar predições dos dois modelos
correlate(cancer, pr_logit, pr_probit)     # deve ser muito alto

# ── Poisson → contagens preditas ───────────────────────────────────────────────
load "dados.csv" as dados
let m_pois = poisson(educacao ~ experiencia + idade, dados)
predict dados mu_hat = m_pois, "count" # E[y|x] = exp(Xβ)
predict dados xb_hat = m_pois, "xb" # preditor linear Xβ (log da média)

# ── Ordered Logit ──────────────────────────────────────────────────────────────
let auto2 = dropna(auto, rep78)
let m_olog = ologit(rep78 ~ mpg + weight, auto2)
predict auto2 yhat_cat = m_olog, "yhat" # categoria predita (argmax)
predict auto2 pr5      = m_olog, "pr5" # P(rep78 = 5) — categoria mais alta
predict auto2 pr1      = m_olog, "pr1" # P(rep78 = 1) — categoria mais baixa

# ── Cox PH → log-hazard ratio e hazard ratio ──────────────────────────────────
# predict cancer loghr = m_cox, loghr   # log-hazard ratio relativo ao baseline
# predict cancer hr    = m_cox, hr      # hazard ratio (exp(log-hazard))

# ══════════════════════════════════════════════════════════════════════════════
# MARGINS — efeitos marginais médios (AME)
# Suporta: logit, probit, poisson, negbin, ologit, oprobit
# ══════════════════════════════════════════════════════════════════════════════

margins(m_logit)
margins(m_probit)
margins(m_pois)

let m_nb = negbin(educacao ~ experiencia + idade, dados)
margins(m_nb)

margins(m_olog)   # tabela AME × categoria

# ══════════════════════════════════════════════════════════════════════════════
# ESTTAB — tabela de coeficientes
# ══════════════════════════════════════════════════════════════════════════════

let m_ols2 = ols(price ~ mpg + weight + C(foreign), auto, cov=HC3)
esttab(m_ols, m_ols2)

esttab(m_logit, m_probit)

# LaTeX pronto para artigo
esttab(m_logit, m_probit, fmt=latex, path="/tmp/logit_probit.tex")
esttab(m_ols, m_ols2,     fmt=latex, path="/tmp/ols_specs.tex")
