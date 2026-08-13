set.seed(42)

x <- rnorm(200, mean = 5, sd = 2)
write.csv(data.frame(x = x), "validation/cases/jb_simulated/data/data.csv", row.names = FALSE)
