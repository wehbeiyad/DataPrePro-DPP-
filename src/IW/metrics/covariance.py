import numpy as np

def feature_covariance(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    """
    Covariance matrix of features (columns of X).
    Returns shape (n_features, n_features).
    ddof=1 gives the unbiased sample covariance (default).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    return np.cov(X, rowvar=False, ddof=ddof)
