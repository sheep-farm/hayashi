# sv_simulated

Stochastic volatility validation on simulated Taylor (1986) SV data.

- DGP: Taylor-style log-volatility process.
- Hayashi: `sv(df, y, iter=100)`.
- References: `stochvol::svsample` (R) and a PyMC NUTS SV model (Python).
- Output: `variable,coef,std_err` CSV with the posterior mean of latent log-volatility.
- Status: blocked because Hayashi's `sv` log-volatility mean is offset from R and PyMC on this DGP.
