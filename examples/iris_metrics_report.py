from IW.datasets import load_iris_dataset
from IW.metrics.covariance import feature_covariance
from IW.metrics.separability import separability_silhouette

ds = load_iris_dataset()

cov = feature_covariance(ds.X)
sil = separability_silhouette(ds.X, ds.y)

print("Dataset:", ds.name)
print("Covariance shape:", cov.shape)
print("Silhouette score:", sil)
