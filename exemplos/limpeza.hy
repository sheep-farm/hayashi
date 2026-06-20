# Limpeza de dados: dropna, rename, replace
# Dataset: cancer.dta — Stata Press R8 (n=48)

load "https://www.stata-press.com/data/r8/cancer.dta" as df

summarize(df)

# rename: nomes mais descritivos
let df = rename(df, studytime, semanas)
let df = rename(df, died,      obito)
let df = rename(df, drug,      tratamento)
let df = rename(df, age,       idade)

summarize(df)

# replace: recodifica tratamento (1=placebo, 2=droga1, 3=droga2)
# cria indicador binário para drogas ativas (tratamento > 1)
generate df droga_ativa = tratamento > 1
tabulate(df, droga_ativa)

# replace: corrige valores extremos de idade (censura em 70)
replace df idade = 70 if idade > 70
summarize(df, idade)

# dropna: remove observações com missing em semanas ou idade
let df = dropna(df, semanas, idade)
count df
