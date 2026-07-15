# Roadmap

Itens pendentes que exigem implementação de matemática nova no Greeners
(não são simples wrappers — exigem algoritmos que ainda não existem na
biblioteca subjacente).

## Pós-estimação — implementação pendente no Greeners

### 1. IV / 2SLS — testes de validação de instrumentos

Um economista que roda `iv()` precisa responder a três perguntas
fundamentais. Hoje nenhuma delas está disponível:

- **First-stage F-statistic** (`estat firststage`): relevância dos
  instrumentos. Regra de dedo: F > 10 indica instrumentos fortes
  (Stock-Yogo). Hoje o `IvResult` não reporta nem o F do primeiro
  estágio, nem Cragg-Donald nem Stock-Yogo critical values.

- **Sargan / Hansen J** (`estat overid`): exogeneidade dos instrumentos
  quando há sobreidentificação (mais instrumentos que endógenas). O
  `GmmResult` já tem `j_stat` e `j_p_value`, mas não há comando
  pós-estimação para acessá-lo via `iv()`. Para 2SLS exatamente
  identificado, o teste não se aplica.

- **Teste de endogeneidade** (`estat endog`): Durbin-Wu-Hausman.
  Compara β_IV com β_OLS para testar se a variável suspeita é
  realmente endógena. Se não rejeita H0 (exogeneidade), OLS é
  consistente e preferível (mais eficiente).

**Onde implementar**: `Greeners/src/iv.rs` — adicionar campos ao
`IvResult` ou criar funções de teste separadas.

### 2. Logit / Probit — avaliação de classificação

Modelos binários precisam de métricas de qualidade de previsão, não
apenas coeficientes e efeitos marginais:

- **`estat classification`**: tabela de classificação com
  sensibilidade, especificidade, taxa de correção global. Usa
  threshold padrão 0.5 (configurável).

- **ROC / AUC** (`lroc`): curva ROC e área sob a curva. AUC > 0.5
  indica poder discriminatório; AUC = 0.5 é equivalente a aleatório.

- **Hosmer-Lemeshow** (`estat gof`): goodness-of-fit para modelos
  binários. Divide a amostra em decilas de probabilidade prevista e
  compara observado vs esperado via χ².

**Onde implementar**: `Greeners/src/discrete.rs` — adicionar funções
ao `BinaryModelResult` ou criar struct separada de diagnósticos.

### 3. `linktest` — detecção de erro de especificação

Stata's `linktest` para modelos binários: reestima o modelo usando
ŷ e ŷ² como únicos regressores. Se ŷ² for significativo, há erro
de especificação (link function inadequada ou forma funcional
incorreta).

**Onde implementar**: `Greeners/src/discrete.rs` — função que
extrai fitted values do modelo binário, constrói ŷ², reestima, e
reporta o p-value do coeficiente de ŷ².

---

## Concluído

- ~~`lrtest(m1, m2)`~~ — implementado em `Greeners/src/model_selection.rs`
  como `ModelSelection::lr_test()`. Exposto em Hayashi como `lrtest(m_restricted,
  m_unrestricted)`. Suporta OLS, logit/probit, Poisson, NegBin, Tobit, Ordered,
  Mixed, ZI, GLM, GARCH, ARIMA.

- ~~IV: first-stage F~~ — já existia em Hayashi como `weak_iv(endog_formula,
  instrument_formula, df)`. Computa F do primeiro estágio por variável
  endógena, Cragg-Donald Wald F, e critical values de Stock-Yogo (2005).

- ~~IV: Sargan/Hansen J~~ — implementado em `Greeners/src/iv.rs` como
  `IV::sargan_test()`. Exposto em Hayashi como `estat_overid(endog_formula,
  instrument_formula, df)`. Testa exogeneidade dos instrumentos quando
  sobreidentificado (L > K).

- ~~IV: endogeneity test (DWH)~~ — implementado em `Greeners/src/iv.rs` como
  `IV::endogeneity_test()`. Exposto em Hayashi como `estat_endog(endog_formula,
  instrument_formula, df)`. Testa se regressores são exógenos via regressão
  augmentada (Hausman).

---

## Priorização sugerida

1. **IV: first-stage F, Sargan/J, endogeneity** — sem isso, IV é
   inutilizável para pesquisa séria. Prioridade máxima.

2. **Logit: classification + ROC + Hosmer-Lemeshow** — avaliação de
   qualidade de previsão binária. Importante para aplicações
   aplicadas (credit scoring, choice models).

3. **`linktest`** — útil mas mais nicho. Pode ficar por último.
