// Gráficos SVG — output publicável para LaTeX
input df
Y X1 X2 X3
10 2 5 3
12 3 3 5
8 1 7 2
15 5 2 8
11 2 6 4
14 4 4 6
9 1 8 1
13 4 3 7
7 1 9 2
16 6 1 9
end

// Scatter: Y vs X1
graph_scatter(df, X1, Y, path="scatter.svg", title="Y vs X1")

// Line: Y ao longo de X1 (ordenado)
graph_line(df, X1, Y, path="line.svg", title="Y por X1")

// Histogram: distribuição de Y
graph_hist(df, Y, path="hist.svg", bins=5, title="Distribuição de Y")

// Coefficient plot com IC 95%
let m = ols(Y ~ X1 + X2 + X3, df)
graph_coef(m, path="coefplot.svg", title="OLS Estimates")

// ASCII também funciona (terminal)
coefplot(m)
