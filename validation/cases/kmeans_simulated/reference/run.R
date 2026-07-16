# Reference implementation in R for K-Means clustering.

library(jsonlite)

# Simulate data following the same DGP as Hayashi.
set.seed(42)
n_per_cluster <- 100
n <- n_per_cluster * 3

# Cluster 1: center at (0, 0)
cluster1_x <- rnorm(n_per_cluster, 0, 0.5)
cluster1_y <- rnorm(n_per_cluster, 0, 0.5)

# Cluster 2: center at (5, 5)
cluster2_x <- rnorm(n_per_cluster, 5, 0.5)
cluster2_y <- rnorm(n_per_cluster, 5, 0.5)

# Cluster 3: center at (10, 0)
cluster3_x <- rnorm(n_per_cluster, 10, 0.5)
cluster3_y <- rnorm(n_per_cluster, 0, 0.5)

# Combine all clusters
x <- c(cluster1_x, cluster2_x, cluster3_x)
y <- c(cluster1_y, cluster2_y, cluster3_y)

# Create DataFrame
df <- data.frame(x = x, y = y)

# Ensure the data directory exists.
data_dir <- "validation/cases/kmeans_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(df, file.path(data_dir, "kmeans.csv"), row.names = FALSE)

# Run K-Means with k-means++ initialization (R's default is Hartigan-Wong)
# Use nstart=25 for stability
set.seed(42)
km_result <- kmeans(df, centers = 3, nstart = 25, iter.max = 100)

# Extract results compatible with Hayashi output.
inertia <- km_result$tot.withinss
between_ss <- km_result$betweenss
total_ss <- km_result$totss
pct_explained <- 100 * (1 - inertia / total_ss)

# Order centroids by the first coordinate so comparison is deterministic.
ordered_centers <- km_result$centers[order(km_result$centers[, 1]), , drop = FALSE]
cluster_centers_list <- as.vector(t(ordered_centers))

result <- list(
  inertia = inertia,
  between_ss = between_ss,
  total_ss = total_ss,
  pct_explained = pct_explained,
  cluster_centers = cluster_centers_list,
  withinss = as.list(km_result$withinss)
)

out_dir <- "validation/cases/kmeans_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
