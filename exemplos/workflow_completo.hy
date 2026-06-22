// Workflow completo — demonstra features desta sessão
// Dataset: Cattaneo (2010) — efeito de fumar sobre peso ao nascer

set_seed(2024)

load "https://github.com/Daniel-Uhr/data/raw/main/cattaneo2.dta" as df

// ── Preparação ──────────────────────────────────────────────
// mbsmoke no .dta é byte: 1=smoker, 0=nonsmoker
generate df Y = bweight
generate df Treated = mbsmoke
generate df lnY = log(Y)

// Winsorizar para robustez
winsor(df, Y, p=0.01, gen=Y_w)

// Estatísticas descritivas
summarize(df, Y, Treated, medu, mage)

// Médias condicionais com if=
scalar mu1 = mean(df, Y, if = Treated == 1)
scalar mu0 = mean(df, Y, if = Treated == 0)
display mu1 - mu0

// ── Teste-t ─────────────────────────────────────────────────
ttest(df, Y, by=Treated)

// ── Regressões progressivas ─────────────────────────────────
let m1 = reg(Y ~ Treated, df)
let m2 = reg(Y ~ Treated + medu, df)
let m3 = reg(Y ~ Treated + medu + mage, df)

// Tabela comparativa
esttab(m1, m2, m3)

// ── Testes pós-estimação ────────────────────────────────────
test(m3, white)
test(m3, Treated, medu)
test(m3, "Treated = 0.5")

// ── Visualização de coeficientes ─────────────────────────────
coefplot(m3)

// ── Cluster-robust SEs ──────────────────────────────────────
let m4 = reg(Y ~ Treated + medu + mage, df, cov=robust)
print(m4)

// ── nlcom — razão de coeficientes ───────────────────────────
nlcom(m3, Treated / medu)

// ── Critérios de informação ─────────────────────────────────
estat(m1, m2, m3)

// ── Predição e diagnóstico ──────────────────────────────────
predict df yhat = m3
predict df resid = m3, residuals

// ── Bootstrap ───────────────────────────────────────────────
bootstrap(ols, Y ~ Treated + medu, df, n=200)

// ── Exportar tabela LaTeX ───────────────────────────────────
// esttab(m1, m2, m3, fmt=latex, path="cattaneo_table.tex")
