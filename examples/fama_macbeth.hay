// Fama-MacBeth (1973) — cross-sectional asset pricing
// Roda OLS por período, reporta média dos β̂_t com FM standard errors

input df
ret beta size firm year
0.05 0.8 10 1 2020
0.08 1.2 12 2 2020
0.03 0.5 8 3 2020
0.12 1.5 15 4 2020
0.06 0.9 11 1 2021
0.10 1.3 13 2 2021
0.04 0.6 9 3 2021
0.14 1.6 16 4 2021
0.07 1.0 10 1 2022
0.11 1.4 14 2 2022
0.02 0.3 7 3 2022
0.15 1.7 17 4 2022
end

// FM básico
fmb(ret ~ beta + size, df, time=year)

// FM com Newey-West (1 lag) para autocorrelação temporal nos β̂_t
fmb(ret ~ beta + size, df, time=year, nw=1)

// Portfolio sorts por beta
portsort(df, ret, beta, n=3)
