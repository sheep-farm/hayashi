// Reproduz o exercício de Uhr OLS

load "https://github.com/Daniel-Uhr/data/raw/main/cattaneo2.dta" as df

ols(bweight ~ mbsmoke, df)