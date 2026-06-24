input df
Y X firm
10 2 1
12 3 1
8 1 1
15 5 2
11 2 2
14 4 2
9 1 3
13 4 3
end

let m1 = reg(Y ~ X, df)
let m2 = reg(Y ~ X, df, cov=robust)
let m3 = reg(Y ~ X, df, cov=HC3)
let m4 = reg(Y ~ X, df, cluster=firm)
esttab(m1, m2, m3, m4)
