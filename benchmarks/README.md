# Hayashi Benchmarks

Benchmarks honestos do Hayashi/Greeners contra R e Python para estimadores
econométricos comuns.

## Objetivo

Medir tempo de execução e memória de forma reproducível, mostrando tanto as
vitórias quanto as derrotas do Hayashi. Nenhum cherry-picking.

## Estimadores cobertos (inicialmente)

- OLS (mínimos quadrados ordinários)
- Logit binário
- ARIMA(1,1,1)
- GARCH(1,1)
- Fixed-effects panel

## Concorrentes

- **R:** `lm`, `glm`, `forecast`, `fGarch`, `plm`
- **Python:** `statsmodels`, `linearmodels`, `arch`
- **Hayashi:** `reg`/`ols`, `logit`, `arima`, `garch`, `fe`

## Metodologia

1. Cada estimador roda sobre datasets sintéticos de tamanhos crescentes.
2. Cada combinação (estimador × linguagem × tamanho) roda várias vezes.
3. Descarta-se a primeira execução (warmup) quando aplicável.
4. Reporta-se média, desvio-padrão e pico de memória.
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

## Interpretação honesta

- Hayashi pode perder em datasets pequenos devido ao tempo de compilação/
  carregamento do binário.
- Hayashi tende a ganhar em datasets grandes e loops repetidos graças ao
  Rust/LLVM.
- R e Python possuem ecossistemas maduros; este benchmark mede velocidade
  bruta, não produtividade geral.

## Resultados

Resultados são escritos em `results/benchmark_YYYYMMDD_HHMMSS.json` e podem
ser plotados com `python scripts/plot.py`.
