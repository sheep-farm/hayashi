# doublesort_simulated

Double portfolio sort validation on simulated finance data.

- DGP: `ret = 0.02 - 0.01*size + 0.015*bm + N(0, 0.05)` for 1000 firms.
- Hayashi: `doublesort(df, ret, size, bm, n1=5, n2=5)`.
- R reference: `dplyr::ntile` then `group_by`/`summarise`.
- Python reference: `pandas.qcut` then `groupby` mean.
- Output: mean return of the small-size / high-BM (1,5) cell.
