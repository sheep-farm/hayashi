input df
Y X1 X2 X3
6 2 1 3
8 2 0 5
1 2 1 2
4 1 1 3
5 1 0 4
7 1 1 6
3 1 0 2
9 0 1 9
2 0 0 2
8 0 1 8
4 0 0 4
end

// correlate sem stars
correlate(df, Y, X1, X2, X3)

// pwcorr com stars
pwcorr(df, Y, X1, X2, X3)
