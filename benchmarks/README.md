# Hayashi Benchmarks

Benchmarks honestos do Hayashi/Greeners contra R e Python para estimadores
econométricos comuns.

## Objetivo

Medir tempo de execução de forma reproducível, mostrando tanto as vitórias
quanto as derrotas do Hayashi. Nenhum cherry-picking.

## Estimadores cobertos

- `ols` — Ordinary Least Squares
- `logit` — Logit binário
- `arima` — AR(1) via `arima(df, y, p=1, d=0, q=0)`
- `garch` — GARCH(1,1)
- `panel` — Fixed-effects panel (`plm`/`linearmodels`)

## Concorrentes

- **R:** `lm`, `glm`, `arima`, `rugarch`, `plm`
- **Python:** `statsmodels`, `linearmodels`, `arch`
- **Hayashi:** `ols`, `logit`, `arima`, `garch`, `fe`

## Metodologia

1. Cada estimador roda sobre datasets sintéticos de tamanhos crescentes.
2. Cada combinação (estimador × linguagem × tamanho) roda várias vezes.
3. Descarta-se a primeira execução (warmup) quando aplicável.
4. Reporta-se média e desvio-padrão do tempo de parede (wall-clock).
5. Datasets e scripts são versionados; resultados brutos ficam em
   `results/` e não são commitados.

## Uso

```bash
cd benchmarks
./run.sh
```

Ou, com controle fino:

```bash
python scripts/run.py --estimator ols --sizes 1000,10000,100000 --reps 10
```

## Interpretação honesta / caveats

- Hayashi pode perder em datasets pequenos devido ao tempo de compilação/
  carregamento do binário.
- Hayashi tende a ganhar em datasets grandes e loops repetidos graças ao
  Rust/LLVM.
- Os concorrentes calculam mais coisas por padrão (matriz de covariância,
  testes, influence). Este benchmark mede o tempo do comando padrão, não de
  uma implementação minimamente equivalente.
- `statsmodels` em particular faz muito trabalho extra no `fit()` padrão,
  por isso pode parecer mais lento do que realmente é para uma tarefa
  equivalente.
- R e Python possuem ecossistemas maduros; este benchmark mede velocidade
  bruta de estimação, não produtividade geral.

## Resultados

Resultados são escritos em `results/<estimator>_YYYYMMDD_HHMMSS.json`.
