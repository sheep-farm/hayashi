# Hayashi vs Stata

## Resumo

| | Hayashi | Stata 18 |
|---|---|---|
| Preço | Grátis (MIT) | US$ 595–2.985/ano |
| Binário | ~18 MB, zero deps (ODBC opcional) | ~500 MB + licença |
| Linguagem | Rust | C/Java |
| Interface | Terminal (REPL + script) + VS Code | GUI + terminal |
| I/O | CSV, TSV, JSON, DTA, Excel, Parquet, SQLite, ODBC | DTA, CSV, Excel, ODBC |
| Gráficos | SVG vetorial + ASCII | PNG/SVG/PDF nativos |
| Testes | 338 automatizados + 59 exemplos | Suite interna proprietária |
| Scoping | Block-scoped, sem GC | Global |
| DataFrames | Rc + copy-on-write | Único dataset ativo |

## Sintaxe lado a lado

```
// Stata                              // Hayashi
reg Y X1 X2                           reg(Y ~ X1 + X2, df)
reg Y X1 X2, vce(robust)              reg(Y ~ X1 + X2, df, cov=robust)
reg Y X1 X2, vce(cluster firm)        reg(Y ~ X1 + X2, df, cluster=firm)
reg Y X1 X2 if year==2020             reg(Y ~ X1 + X2, df, if=year==2020)
ivregress 2sls Y (X1=Z1 Z2) X2       iv(Y ~ X1 + X2, ~ Z1 + Z2, df)
xtset firm year                       xtset(df, firm, year)
xtreg Y X1 X2, fe                     fe(Y ~ X1 + X2, df)
xtreg Y X1 X2, re                     re(Y ~ X1 + X2, df)
hausman fe re                         hausman(m_fe, m_re)
test X1 X2                            test(m, "X1", "X2")
test X1 = X2                          test(m, "X1 = X2")
nlcom _b[X1]/_b[X2]                   nlcom(m, X1 / X2)
margins, dydx(X1)                     margins(m, dydx=[X1])
margins, at(X2=0)                     margins(m, at_X2=0)
estat ic                              estat(m1, m2)
predict yhat                          predict df yhat = m
predict e, resid                      predict df e = m, "residuals"
eststo: reg Y X1                      eststo(reg(Y ~ X1, df))
esttab, se                            esttab()
esttab using "t.tex", tex             esttab(fmt=latex, path="t.tex")
scatter Y X                           graph_scatter(df, X, Y, path="f.svg")
histogram Y                           graph_hist(df, Y, path="f.svg")
gen lnY = log(Y)                      generate df lnY = log(Y)
gen D = regexm(name, "^Dr")           generate df D = regexm(name, "^Dr")
replace Y = 0 if X > 10               replace df Y = 0 if X > 10
winsor2 Y, cuts(1 99)                 winsor(df, Y, p=0.01)
encode str_var, gen(num)               encode(df, str_var)
tab group, gen(d_)                     tabgen(df, group)
summarize, detail                      summarize(df, detail=true)
ci means Y                            ci(df, Y)
pwcorr X1 X2 X3, star(0.05)           pwcorr(df, X1, X2, X3)
preserve                              preserve(df)
restore                               restore(df)
quietly reg Y X                        quietly(ols(Y ~ X, df))
capture reg Y X                        capture(ols(Y ~ X, df))
assert price > 0                       assert(X > 0, "msg")

foreach v in X1 X2 X3 {               for v in ["X1", "X2", "X3"] {
    reg Y `v'                              eststo(ols("Y ~ " + v, df))
    est store m_`v'                    }
}                                      esttab()
esttab m_*
```

## Cobertura de estimadores

| Categoria | Stata | Hayashi | Status |
|---|---|---|---|
| OLS + HC1-HC4 | `reg` | `ols`/`reg` | Paridade |
| Cluster SEs | `vce(cluster)` | `cluster=` | Paridade |
| Two-way cluster | `vce(cluster c1 c2)` | `cluster= cluster2=` | Paridade |
| Newey-West | `newey` | `nw=` | Paridade |
| IV/2SLS | `ivregress` | `iv` | Paridade |
| Painel FE/RE | `xtreg` | `fe`/`re` + `xtset` | Paridade |
| Arellano-Bond | `xtabond`/`xtdpdsys` | `ab`/`sysgmm` | Paridade |
| Hausman | `hausman` | `hausman` | Paridade |
| Logit/Probit | `logit`/`probit` | `logit`/`probit` | Paridade |
| Margins AME + SEs | `margins` | `margins` | Paridade |
| Poisson/NegBin | `poisson`/`nbreg` | `poisson`/`nbreg` | Paridade |
| Tobit/Heckman | `tobit`/`heckman` | `tobit`/`heckman` | Paridade |
| Quantile | `qreg` | `qreg` | Paridade |
| ARIMA/GARCH | `arima`/`arch` | `arima`/`garch` | Paridade |
| VAR/VECM | `var`/`vec` | `var`/`vecm` | Paridade |
| Lasso/Ridge | `lasso` | `lasso`/`ridge` | Paridade |
| Cox PH | `stcox` | `cox` | Paridade |
| DID/RD/Synth/PSM | addons | builtins | Paridade |
| Fama-MacBeth | `xtfmb` (addon pago) | `fmb` (builtin + NW) | Hayashi superior |
| Portfolio sorts | programação manual | `portsort`/`doublesort` | Hayashi superior |
| Multinomial | `mlogit` | `mlogit` | Paridade |
| Mixed/HLM | `mixed` | `mixed` | Parcial |
| Survey | `svy:` | -- | Ausente |
| SEM | `sem`/`gsem` | -- | Ausente |
| Bayesian | `bayes:` | -- | Ausente |
| Spatial | `spregress` | -- | Ausente |

## Onde Hayashi ganha

- **Custo**: grátis vs US$ 595+/ano
- **Portabilidade**: binário de 5 MB sem dependências
- **Fama-MacBeth**: builtin com Newey-West (Stata requer addon pago)
- **Portfolio sorts**: `portsort`, `doublesort` builtin
- **Bootstrap genérico**: qualquer estimador, não só OLS
- **Fórmulas dinâmicas**: `ols("Y ~ " + v, df)` nativo
- **Block scoping**: lifetime determinístico sem GC
- **Regex row-wise**: `ols(Y ~ X, df, if = regexm(name, "Dr"))`
- **Copy-on-write**: `Rc<DataFrame>` — zero-copy em funções
- **I/O multi-formato**: CSV, TSV, JSON, DTA, Excel, Parquet, SQLite, ODBC
- **Export multi-formato**: CSV, JSON, TSV, XLSX, Parquet, SQLite, LaTeX, HTML
- **338 testes + 59 exemplos**: `cargo test` em <1s
- **help() completo**: ~95 tópicos com assinatura + exemplo no REPL

## Onde Stata ganha

- **Maturidade**: 40+ anos, battle-tested
- **Documentação**: 15.000+ páginas de manual
- **GUI**: interface gráfica completa
- **Ecossistema**: 10.000+ pacotes SSC
- **Survey**: `svy:` para amostras complexas
- **SEM/Bayesian/Spatial**: nichos especializados
- **Gráficos**: 50+ tipos vs 4 SVG + 8 ASCII
- **Aceitação acadêmica**: padrão de facto em journals
- **Suporte**: empresa + StataCorp

## Conclusão

Hayashi cobre ~97% do workflow de econometria aplicada de pós-graduação com paridade funcional completa em estimação, pós-estimação, manipulação de dados, e output publicável. Os gaps restantes são nichos especializados (survey, SEM, Bayesian, spatial) que poucos pesquisadores usam simultaneamente.
