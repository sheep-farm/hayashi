input df
X
0.5
1.0
2.0
3.0
4.0
5.0
6.0
7.0
8.0
9.0
end

// Novas funções math
generate df T = tan(X)
generate df AS = asin(X * 0.1)
generate df S = sign(X - 5)
generate df M = mod(X, 3)
generate df F = factorial(X)
generate df C = cond(X > 5, 1, 0)
generate df MX = max(X, 5)

// Random
generate df U = uniform()
generate df N = rnormal()

// Summarize detail
summarize(df, X, detail=true)

list(df, n=5)
