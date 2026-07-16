# Hayashi Benchmark Results — Baseline

Este é um snapshot de referência gerado em uma máquina específica. Os
arquivos `.json` e `.png` em `results/` continuam ignorados pelo git; este
README é o relatório commitado.

## Máquina

| Característica | Valor |
|---|---|
| SO | Arch Linux (`7.1.3-arch1-2`, `x86_64`) |
| CPU | 11th Gen Intel Core i5-11320H @ 3.20GHz |
| Núcleos / threads | 4 cores / 8 threads |
| RAM | 7.5 GiB |
| Swap | 11 GiB |

## Versões

| Ferramenta | Versão |
|---|---|
| Hayashi | 0.2.10-dev |
| Python | 3.14.6 |
| R | 4.6.1 |
| statsmodels | 0.14.6 |
| linearmodels | 7.0 |
| arch | 8.0.0 |
| numpy | 2.5.1 |
| pandas | 3.0.3 |
| plm (R) | 2.6.7 |
| rugarch (R) | 1.5.5 |
| jsonlite (R) | 2.0.0 |

## Metodologia

- Datasets sintéticos gerados por `benchmarks/datasets/generate.py`.
- Tamanhos testados: `n = 1.000` e `n = 10.000`.
- Repetições: 5, com a primeira descartada como warmup.
- Tempo medido pelo orquestrador em Python (`time.perf_counter`).
- Memória medida por polling de `/proc/<pid>/status` (`VmRSS`) do processo
  principal e filhos.
- Hayashi compilado em release (`target/release/hay`).

## Resultados

| Estimator | n | Language | Mean (s) | Std (s) | Memory (MB) |
|---|---|---|---:|---:|---:|
| arima | 1000 | hayashi | 0.0244 | 0.0028 | 0.2 |
| arima | 1000 | python | 2.0715 | 0.3487 | 151.9 |
| arima | 1000 | r | 0.3676 | 0.0440 | 90.4 |
| arima | 10000 | hayashi | 0.0289 | 0.0127 | 0.2 |
| arima | 10000 | python | 2.8595 | 0.0577 | 165.1 |
| arima | 10000 | r | 0.3998 | 0.0105 | 98.2 |
| garch | 1000 | hayashi | 0.0488 | 0.0020 | 7.8 |
| garch | 1000 | python | 1.8695 | 0.0519 | 169.1 |
| garch | 1000 | r | 2.1719 | 0.1039 | 267.9 |
| garch | 10000 | hayashi | 0.2201 | 0.0230 | 8.4 |
| garch | 10000 | python | 2.5667 | 0.2406 | 171.1 |
| garch | 10000 | r | 5.8569 | 0.1809 | 281.1 |
| logit | 1000 | hayashi | 0.0316 | 0.0159 | 0.2 |
| logit | 1000 | python | 2.1653 | 0.4729 | 162.2 |
| logit | 1000 | r | 0.4716 | 0.0348 | 93.5 |
| logit | 10000 | hayashi | 0.2667 | 0.0979 | 12.4 |
| logit | 10000 | python | 1.7971 | 0.1753 | 163.9 |
| logit | 10000 | r | 0.4931 | 0.0701 | 133.7 |
| ols | 1000 | hayashi | 0.0570 | 0.0062 | 8.1 |
| ols | 1000 | python | 2.7471 | 0.1800 | 161.9 |
| ols | 1000 | r | 0.4592 | 0.0721 | 81.7 |
| ols | 10000 | hayashi | 0.1387 | 0.0576 | 12.7 |
| ols | 10000 | python | 2.4008 | 0.3755 | 163.2 |
| ols | 10000 | r | 0.3024 | 0.0221 | 90.9 |
| panel | 1000 | hayashi | 0.0244 | 0.0022 | 0.1 |
| panel | 1000 | python | 1.9700 | 0.0583 | 150.5 |
| panel | 1000 | r | 0.9907 | 0.0296 | 126.7 |
| panel | 10000 | hayashi | 0.0552 | 0.0121 | 10.9 |
| panel | 10000 | python | 2.1127 | 0.0776 | 154.1 |
| panel | 10000 | r | 1.4863 | 0.1682 | 158.1 |

## Speedup Hayashi vs concorrentes

| Estimator | n | vs Python | vs R |
|---|---|---:|---:|
| arima | 1000 | 85.0x | 15.1x |
| arima | 10000 | 99.1x | 13.9x |
| garch | 1000 | 38.3x | 44.5x |
| garch | 10000 | 11.7x | 26.6x |
| logit | 1000 | 68.6x | 14.9x |
| logit | 10000 | 6.7x | 1.8x |
| ols | 1000 | 48.2x | 8.1x |
| ols | 10000 | 17.3x | 2.2x |
| panel | 1000 | 80.7x | 40.6x |
| panel | 10000 | 38.3x | 26.9x |

## Interpretação / caveats

- Hayashi é consistentemente mais rápido e usa menos memória nesta máquina.
- Diferenças de até 100x aparecem em estimadores com otimização MLE
  (arima, garch), onde Hayashi parece convergir rapidamente.
- Python e R calculam mais estatísticas por padrão (covariância robusta,
  diagnósticos, influence). Este benchmark mede o comando padrão, não uma
  implementação minimamente equivalente.
- A memória do Hayashi varia de 0.1 MB a 12.7 MB, enquanto R/Python
  frequentemente alocam 80–280 MB. Parte se deve ao carregamento de
  bibliotecas numéricas no processo filho.
- Resultados podem mudar em outras máquinas, SOs, flags de compilação ou
  versões de dependências.
