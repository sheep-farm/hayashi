# Hayashi vs Stata — Comparação Honesta

## Resumo

| | Hayashi | Stata 18 |
|---|---|---|
| Preço | Grátis (MIT) | US$ 595–2.985/ano |
| Binário | ~5 MB, sem dependências | ~500 MB + licença |
| Linguagem base | Rust | C/Java |
| Extensível | Rust (recompila) | ado/Mata |
| Datasets | CSV, DTA, URL | DTA, CSV, ODBC, SQL, ... |
| Interface | Terminal (REPL + script) | GUI + terminal |
| Gráficos | SVG vetorial (publicável) + ASCII | PNG/SVG/PDF nativos |
| Documentação | README + help() inline | 15.000+ páginas de manual |
| Comunidade | 1 desenvolvedor | 40+ anos de ecossistema |
| Testes | 208 automatizados | Suite interna proprietária |

## Sintaxe lado a lado

### Carga de dados

```
// Stata                          // Hayashi
use "data.dta", clear             load "data.dta" as df
import delimited "data.csv"       load "data.csv" as df
webuse auto                       load "https://...auto.csv" as df
```

### Regressão

```
// Stata                          // Hayashi
reg Y X1 X2                       reg(Y ~ X1 + X2, df)
reg Y X1 X2, vce(robust)          reg(Y ~ X1 + X2, df, cov=robust)
reg Y X1 X2, vce(cluster firm)    reg(Y ~ X1 + X2, df, cluster=firm)
reg Y X1 X2 if year==2020         reg(Y ~ X1 + X2, df, if=year==2020)
ivregress 2sls Y (X1=Z1 Z2) X2   iv(Y ~ X1 + X2, ~ Z1 + Z2, df)
```

### Painel

```
// Stata                          // Hayashi
xtset firm year                   xtset(df, firm, year)
xtreg Y X1 X2, fe                 fe(Y ~ X1 + X2, df)
xtreg Y X1 X2, re                 re(Y ~ X1 + X2, df)
hausman fe re                     hausman(m_fe, m_re)
```

### Pós-estimação

```
// Stata                          // Hayashi
test X1 X2                        test(m, X1, X2)
test X1 = X2                      test(m, "X1 = X2")
nlcom _b[X1]/_b[X2]              nlcom(m, X1 / X2)
margins, dydx(X1)                 margins(m, dydx=[X1])
margins, at(X2=0)                 margins(m, at_X2=0)
estat ic                          estat(m1, m2)
predict yhat                      predict df yhat = m
predict e, resid                  predict df e = m, residuals
```

### Tabelas

```
// Stata                          // Hayashi
eststo: reg Y X1                  eststo(reg(Y ~ X1, df))
eststo: reg Y X1 X2              eststo(reg(Y ~ X1 + X2, df))
esttab, se star(* 0.10)          esttab()
esttab using "t.tex", tex        esttab(fmt=latex, path="t.tex")
```

### Gráficos

```
// Stata                          // Hayashi
scatter Y X                       graph_scatter(df, X, Y, path="fig.svg")
line Y X                          graph_line(df, X, Y, path="fig.svg")
histogram Y                       graph_hist(df, Y, path="fig.svg", bins=30)
coefplot                          graph_coef(m, path="fig.svg")
// + ASCII no terminal:
//                                scatter(df, X, Y)
//                                histogram(df, Y)
//                                coefplot(m)
```

### Loops

```
// Stata                          // Hayashi
foreach v in X1 X2 X3 {          for v in ["X1", "X2", "X3"] {
    reg Y `v'                         eststo(ols("Y ~ " + v, df))
    est store m_`v'               }
}                                 esttab()
esttab m_*
```

### Dados

```
// Stata                          // Hayashi
gen lnY = log(Y)                  generate df lnY = log(Y)
gen D = (X==1)                    generate df D = (X == 1)
replace Y = 0 if X > 10          replace df Y = 0 if X > 10
drop X3                           drop(df, X3)
keep X1 X2 Y                     keep(df, X1, X2, Y)
winsor2 Y, cuts(1 99)            winsor(df, Y, p=0.01)
encode str_var, gen(num)          encode(df, str_var)
tab group, gen(d_)                tabgen(df, group)
summarize                         summarize(df)
tab group                         tabulate(df, group)
pwcorr X1 X2 X3, star(0.05)     pwcorr(df, X1, X2, X3)
ttest Y, by(group)               ttest(df, Y, by=group)
preserve                          preserve(df)
restore                           restore(df)
```

## Onde Hayashi ganha

| Feature | Hayashi | Stata |
|---|---|---|
| Custo | Grátis | US$ 595+/ano |
| Portabilidade | Binário estático ~5 MB, qualquer Linux | Requer instalação + licença |
| Fama-MacBeth | Builtin com Newey-West | Requer `xtfmb` (addon pago) |
| Portfolio sorts | `portsort`, `doublesort` builtin | Requer programação manual |
| Bootstrap genérico | `bootstrap(estimator, ...)` qualquer modelo | `bootstrap:` prefixo (mais limitado) |
| Fórmulas dinâmicas | `ols("Y ~ " + v, df)` | Macros locais (`` `v' ``) |
| Block scoping | `{}` com lifetime determinístico, sem GC | Tudo global |
| Gráficos no terminal | ASCII scatter/hist/coefplot integrados | Não disponível |
| Reprodutibilidade | `set_seed` + `cargo test` 208 testes | `set seed` + sem testes públicos |
| Código aberto | MIT, Rust, auditável | Proprietário |

## Onde Stata ganha

| Feature | Stata | Hayashi |
|---|---|---|
| Maturidade | 40+ anos, battle-tested | Projeto novo (3 dias) |
| Documentação | Manual completo, livros, cursos | README + help() |
| GUI | Interface gráfica completa | Terminal apenas |
| Ecossistema | 10.000+ pacotes SSC | Apenas builtins |
| Dados grandes | Frames, até 120 bilhões de obs | Limitado pela RAM |
| Survey | `svy:` prefix para amostras complexas | Não implementado |
| SEM | `sem`/`gsem` completo | Não implementado |
| Spatial | `spregress`, spatial econometrics | Não implementado |
| Bayesian | `bayes:` prefix | Não implementado |
| Strings | Funções regex completas | Operações básicas |
| Mata | Linguagem matricial integrada | Sem equivalente |
| Suporte | Empresa + StataCorp | Comunidade (1 pessoa) |
| Reprodutibilidade acadêmica | Padrão aceito por journals | Desconhecido por journals |
| Variedade de gráficos | 50+ tipos, customização fina | 4 tipos SVG + ASCII |

## Cobertura de estimadores

| Categoria | Stata | Hayashi | Status |
|---|---|---|---|
| OLS + robust SEs | `reg`, HC0-HC3 | `ols`/`reg`, HC1-HC4 | Paridade |
| IV/2SLS | `ivregress` | `iv` | Paridade |
| Painel FE/RE | `xtreg` | `fe`/`re` + `xtset` | Paridade |
| Arellano-Bond | `xtabond`/`xtdpdsys` | `ab`/`sysgmm` | Paridade |
| Hausman | `hausman` | `hausman` | Paridade |
| Logit/Probit | `logit`/`probit` | `logit`/`probit` | Paridade |
| Poisson/NB | `poisson`/`nbreg` | `poisson`/`nbreg` | Paridade |
| Tobit/Heckman | `tobit`/`heckman` | `tobit`/`heckman` | Paridade |
| Quantile | `qreg` | `qreg` | Paridade |
| ARIMA/GARCH | `arima`/`arch` | `arima`/`garch` | Paridade |
| VAR/VECM | `var`/`vec` | `var`/`vecm` | Paridade |
| Lasso/Ridge | `lasso` (Stata 16+) | `lasso`/`ridge` | Paridade |
| Cox PH | `stcox` | `cox` | Paridade |
| Cluster SEs | `vce(cluster)` | `cluster=var` | Paridade |
| Two-way cluster | `vce(cluster c1 c2)` | `cluster= cluster2=` | Paridade |
| Newey-West | `newey` | `nw=lags` | Paridade |
| Fama-MacBeth | `xtfmb` (addon) | `fmb` (builtin + NW) | Hayashi superior |
| Margins AME | `margins` (com SEs) | `margins` (sem SEs) | Parcial |
| Survey | `svy:` | -- | Ausente |
| SEM | `sem`/`gsem` | -- | Ausente |
| Bayesian | `bayes:` | -- | Ausente |
| Spatial | `spregress` | -- | Ausente |
| Multinomial | `mlogit` | `mlogit` | Paridade |
| Mixed/HLM | `mixed` | `mixed` | Parcial |
| DID | `diff` | `did` | Paridade |
| RD | `rdrobust` | `rd` | Paridade |
| Synth | `synth` | `synth` | Paridade |
| PSM | `psmatch2` | `psmatch` | Paridade |

## Gráficos

| Tipo | Stata | Hayashi |
|---|---|---|
| Scatter | `scatter Y X` (PNG/SVG/PDF) | `graph_scatter(df, X, Y, path="f.svg")` SVG + `scatter(df, X, Y)` ASCII |
| Line | `line Y X` | `graph_line(df, X, Y, path="f.svg")` SVG + `lineplot(df, X, Y)` ASCII |
| Histogram | `histogram Y` | `graph_hist(df, Y, path="f.svg")` SVG + `histogram(df, Y)` ASCII |
| Coefficient | `coefplot` (addon) | `graph_coef(m, path="f.svg")` SVG + `coefplot(m)` ASCII |
| Box plot | `graph box Y` | `boxplot(df, Y)` ASCII |
| KDE | `kdensity Y` | `kdensity(df, Y)` ASCII |
| ACF/PACF | `ac Y` / `pac Y` | `acfplot(df, Y)` ASCII |
| QQ plot | `qnorm Y` | `qqplot(df, Y)` ASCII |
| Correlation matrix | -- | `corrplot(df, X1, X2, X3)` ASCII |
| Residual plot | `rvfplot` | `residplot(m)` ASCII |
| Formato SVG | Nativo | Nativo (plotters) |
| Formato PNG | Nativo | Planejado |
| Formato PDF | Nativo | Via conversão SVG→PDF |

## Conclusão

Hayashi cobre ~95% do workflow de econometria aplicada de pós-graduação (cross-section, painel, séries temporais, causal inference, finanças empíricas) com gráficos SVG publicáveis. O gap está em features especializadas (survey, SEM, Bayesian, spatial), variedade de gráficos, e no ecossistema (documentação, comunidade, aceitação em journals).

Para pesquisa onde Stata não está disponível (orçamento zero, ambiente Linux headless, pipelines automatizados), Hayashi é uma alternativa funcional com output publicável. Para publicação em journals top, Stata ainda é o padrão de facto — mas nada impede o uso de Hayashi com validação cruzada contra Stata.
