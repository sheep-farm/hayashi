// Cattaneo (2010) — efeito de fumar na gravidez sobre peso ao nascer
// Reproduz: t-test de diferença de médias + OLS simples

load "https://github.com/Daniel-Uhr/data/raw/main/cattaneo2.dta" as df

// mbsmoke é string no .dta — comparação direta funciona
generate df Y = bweight
generate df Treated = (mbsmoke == "smoker")

// Médias condicionais
scalar mu1 = mean(df, Y, if = Treated == 1)
scalar mu0 = mean(df, Y, if = Treated == 0)
display mu1
display mu0
display mu1 - mu0

// Teste-t de duas amostras (Welch)
ttest(df, Y, by=Treated)

// Regressão simples — coef. de Treated = diferença de médias
ols(Y ~ Treated, df)

ols(Y ~ Treated + medu, df)