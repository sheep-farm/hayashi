// Double sort — Fama-French style (size × book-to-market)
input df
ret size bm
0.05 100 0.3
0.08 200 0.5
0.03 50 0.2
0.12 300 0.8
0.04 80 0.4
0.10 250 0.7
0.02 40 0.1
0.14 350 0.9
0.06 120 0.3
0.11 280 0.6
0.01 30 0.1
0.15 400 1.0
0.07 150 0.5
0.09 220 0.6
0.13 320 0.8
0.03 60 0.2
0.11 270 0.7
0.06 130 0.4
0.14 340 0.9
0.02 45 0.15
end

// Univariate sorts
portsort(df, ret, size, n=5)
portsort(df, ret, bm, n=5)

// Bivariate sort: 3×3
doublesort(df, ret, size, bm, n1=3, n2=3)
