# Dataset: cancer.dta — Stata Press R8 (n=48)
# Fonte: https://www.stata-press.com/data/r8/cancer.dta
# Variáveis:
#   studytime : semanas de observação
#   died      : 1 = óbito, 0 = censurado
#   drug      : 1 = placebo, 2 = droga 1, 3 = droga 2
#   age       : idade do paciente

load "https://www.stata-press.com/data/r8/cancer.dta" as cancer

summarize(cancer)
correlate(cancer, studytime, age, drug)

# OLS: tempo de sobrevivência em função de idade e tratamento
let m_ols = ols(studytime ~ age + C(drug), cancer, cov=HC3)
print(m_ols)

test(m_ols, bp)
test(m_ols, dw)

# Logit: probabilidade de óbito
let m_logit = logit(died ~ age + C(drug), cancer)
print(m_logit)

predict cancer pr_obito = m_logit, pr
summarize(cancer, died, pr_obito)

export(m_ols,   latex, "/tmp/cancer_ols.tex")
export(m_logit, txt,   "/tmp/cancer_logit.txt")
