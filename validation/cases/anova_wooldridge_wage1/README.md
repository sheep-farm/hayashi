# One-way ANOVA of wage by education in Wooldridge wage1

Compares `anova(df, wage, by=educ)` against R and SciPy reference implementations.

Quantities compared: ss_between, ss_within, ss_total, df_between, df_within, ms_between, ms_within, f_stat, p_value, n_groups, n_obs.
