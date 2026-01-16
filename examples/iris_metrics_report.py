from IW.datasets import load_iris_dataset
from IW.metrics.covariance import feature_covariance
from IW.metrics.separability import separability_silhouette
from IW.metrics.mle import intrinsic_dimension_mle
from IW.metrics.label_noise import estimate_label_noise_cleanlab

def main():
    ds = load_iris_dataset()

    cov = feature_covariance(ds.X)
    sil = separability_silhouette(ds.X, ds.y)
    mle_dim = intrinsic_dimension_mle(ds.X, k=10)

    noise = estimate_label_noise_cleanlab(ds.X, ds.y, cv=5, method="self_confidence", n_jobs=1)

    print("Dataset:", ds.name)
    print("Covariance shape:", cov.shape)
    print("Silhouette score:", sil)
    print("MLE intrinsic dimension (k=10):", mle_dim)
    print("Estimated label-issue fraction:", noise.issue_fraction)
    print("Top suspected label-issue indices:", noise.issue_indices[:10])

if __name__ == "__main__":
    main()
