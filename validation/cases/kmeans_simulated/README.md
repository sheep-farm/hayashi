# K-Means clustering on simulated data

This validation case implements K-Means clustering (MacQueen 1967) on simulated 2D data with 3 Gaussian clusters.

## Dataset

- **Name:** `simulated_kmeans`
- **Source:** Simulated 2D Gaussian clusters
- **Licence:** MIT
- **Size:** 300 observations (100 per cluster)

## Reference implementation

- **R:** `stats::kmeans()` with k-means++ initialization
- **Python:** `sklearn.cluster.KMeans` with k-means++ initialization

## Compared quantities

- cluster_centers (final centroid coordinates)
- cluster_assignments (cluster labels for each observation)
- withinss (within-cluster sum of squares)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| cluster_centers | 1e-4 | K-means++ should converge to similar centroids |
| cluster_assignments | 0 | Exact match expected for deterministic initialization |
| withinss | 1e-4 | Should match exactly given same centroids |

## Notes

K-Means uses k-means++ initialization for better convergence. The simulated data has 3 well-separated Gaussian clusters, making the clustering problem straightforward and the results deterministic across implementations.
