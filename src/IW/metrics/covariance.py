import numpy as np

def feature_covariance(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    return np.cov(X, rowvar=False, ddof=ddof)
