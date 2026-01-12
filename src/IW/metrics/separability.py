import numpy as np
from sklearn.metrics import silhouette_score

def separability_silhouette(X: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> float:
    """
    Average silhouette score across samples.
    Higher is better (near 1).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same n_samples")
    return float(silhouette_score(X, y, metric=metric))
