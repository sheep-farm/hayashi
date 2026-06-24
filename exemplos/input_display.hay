// Exemplo: input inline, display, scalar e list
// Reproduz o exercício 2.1 de Hayashi (2000), cap. 1

// Dados inline — sem arquivo CSV
input df
Y X u
6 2 7
8 2 9
1 2 3
4 1 3
5 1 4
7 1 6
3 1 2
9 0 9
2 0 2
8 0 8
4 0 4
end

// Inspecionar as primeiras obs
list(df)

// Médias condicionais
scalar mu1 = mean(df, Y, if = X == 1)
scalar mu0 = mean(df, Y, if = X == 0)
scalar mu2 = mean(df, Y, if = X >= 2)

display mu1
display mu0
display mu1 - mu0
display mu2 

// Média do resíduo deve ser ~0
scalar eu = mean(df, u)
display eu

// Regressão simples
let m = ols(Y ~ X, df)
print(m)
