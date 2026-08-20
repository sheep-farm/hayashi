# manova_simulated

MANOVA validation on simulated three-group bivariate data.

- DGP: three groups with distinct `y1` and `y2` means.
- Hayashi: `manova(df, y1, y2, by="group")`.
- R reference: `manova(cbind(y1, y2) ~ group)`.
- Python reference: `statsmodels.multivariate.manova.MANOVA.from_formula`.
- Output: Pillai, Wilks, Hotelling-Lawley and Roy test statistics.
