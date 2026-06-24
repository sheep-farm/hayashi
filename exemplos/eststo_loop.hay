// eststo + loop: acumula modelos e tabela automática
input df
Y X1 X2 X3
10 2 3 5
12 3 4 6
8 1 2 4
15 5 6 8
11 2 5 7
14 4 3 6
9 1 4 5
13 4 5 7
end

// Loop acumulando modelos
for spec in ["X1", "X1 + X2", "X1 + X2 + X3"] {
    let m = ols("Y ~ " + spec, df)
    eststo(m)
}

// Tabela com todos os modelos acumulados
esttab()
estclear()
