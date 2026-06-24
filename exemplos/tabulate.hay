# Dataset: cancer.dta — Stata Press R8 (n=48)
# Fonte: https://www.stata-press.com/data/r8/cancer.dta

load "https://www.stata-press.com/data/r8/cancer.dta" as cancer

# Frequência simples: tipo de droga
tabulate(cancer, drug)

# Frequência simples: status de óbito
tabulate(cancer, died)

# Tabela cruzada: droga × óbito
tabulate(cancer, drug, died)

# Tabela cruzada com teste chi2
tabulate(cancer, drug, died, chi2=true)
