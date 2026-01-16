"""
MLE Intrinsic Dimension Estimator (Levina–Bickel), with optional MacKay–Ghahramani correction.

This implementation is designed to be robust on real datasets that may contain
duplicate / identical samples (which can create zero k-NN distances and break
the log-ratio in the estimator). To handle that, we request extra neighbors and
then, for each sample, select the first k strictly-positive distances.

Conceptually adapted from:
- Levina, E. & Bickel, P. (2005): Maximum likelihood estimation of intrinsic dimension
- MacKay & Ghahramani correction (commonly used in practice)
- Code style inspired by GeoMLE:
  https://github.com/stat-ml/GeoMLE/blob/master/geomle/mle.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors


ArrayLike = Union[np.ndarray, "object"]  # allow torch.Tensor without importing torch


def _to_numpy(X: ArrayLike) -> np.ndarray:
    """Convert input to a numpy array (supports torch tensors without importing torch)."""
    # Torch Tensor support (duck-typing)
    if hasattr(X, "detach") and hasattr(X, "cpu") and hasattr(X, "numpy"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {X.shape}.")
    return X


def _knn_distances_positive(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
    extra_neighbors: int = 20,
) -> np.ndarray:
    """
    Compute distances to k nearest neighbors for each point, excluding self-distance,
    ensuring distances are strictly positive.

    Robust to duplicate points:
      - query (k + extra_neighbors + 1) neighbors (capped at n_samples),
      - drop self-distance,
      - per-sample, keep the first k distances > 0.

    Returns
    -------
    dist_pos : np.ndarray
        Shape (n_samples, k), strictly positive distances.
    """
    n = X.shape[0]
    if n <= k:
        raise ValueError(f"Need n_samples > k. Got n_samples={n}, k={k}.")

    # +1 for self-distance, +extra to survive duplicates/zeros
    n_query = min(n, k + extra_neighbors + 1)

    nn = NearestNeighbors(n_neighbors=n_query, metric=metric, n_jobs=n_jobs)
    nn.fit(X)
    dist, _ = nn.kneighbors(X, return_distance=True)

    dist = dist[:, 1:]  # drop self-distance

    out = np.empty((n, k), dtype=dist.dtype)
    for i in range(n):
        pos = dist[i][dist[i] > 0]
        if pos.shape[0] < k:
            raise ValueError(
                f"Not enough positive neighbor distances for sample {i}: "
                f"needed {k}, got {pos.shape[0]}. "
                "This can happen if there are many identical samples."
            )
        out[i] = pos[:k]
    return out


def intrinsic_dimension_mle_per_sample(dist_pos: np.ndarray, k: int) -> np.ndarray:
    """
    Levina–Bickel MLE intrinsic dimension per sample from k-NN distances.

    Parameters
    ----------
    dist_pos : np.ndarray
        Shape (n_samples, k), strictly positive distances excluding self-distance.
    k : int
        Number of neighbors used by the estimator (must be >= 3).

    Returns
    -------
    mle_per_sample : np.ndarray
        Shape (n_samples,).
    """
    if k < 3:
        raise ValueError("k must be >= 3 for MLE intrinsic dimension.")
    if dist_pos.ndim != 2 or dist_pos.shape[1] < k:
        raise ValueError(f"dist_pos must have shape (n_samples, >=k). Got {dist_pos.shape}")

    d = dist_pos[:, :k]

    # m_i = 1 / ( (1/(k-2)) * sum_{j=1}^{k-1} log(T_k / T_j) )
    Tk = d[:, k - 1 : k]      # shape (n, 1)
    Tj = d[:, 0 : k - 1]      # shape (n, k-1)
    logs = np.log(Tk / Tj)    # shape (n, k-1)

    denom = logs.sum(axis=1) / (k - 2)
    mle = 1.0 / denom
    return mle


def intrinsic_dimension_mle(
    X: ArrayLike,
    k: int = 10,
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
    extra_neighbors: int = 20,
    return_details: bool = False,
    mackay_correction: bool = True,
) -> Union[float, Tuple[float, float, np.ndarray]]:
    """
    Estimate intrinsic dimension using Levina–Bickel MLE.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    k : int
        Number of neighbors (must be >= 3).
    metric : str
        Distance metric for sklearn NearestNeighbors.
    n_jobs : int or None
        Parallelism for sklearn (None uses default behavior).
    extra_neighbors : int
        Request extra neighbors so we can skip zero distances caused by duplicates.
    return_details : bool
        If True, returns (mle_mean, mle_corrected, mle_per_sample).
    mackay_correction : bool
        If True, returns a corrected estimate:
            mle_corrected = 1 / mean(1 / mle_i)
        (often referred to as MacKay–Ghahramani correction in practice)

    Returns
    -------
    If return_details is False:
        mle_mean : float

    If return_details is True:
        (mle_mean, mle_corrected, mle_per_sample)
    """
    Xn = _to_numpy(X)
    dist_pos = _knn_distances_positive(
        Xn, k=k, metric=metric, n_jobs=n_jobs, extra_neighbors=extra_neighbors
    )

    mle_per = intrinsic_dimension_mle_per_sample(dist_pos, k=k)
    mle_mean = float(np.mean(mle_per))

    if not return_details:
        return mle_mean

    inv_mle_per = 1.0 / mle_per
    mle_corrected = float(1.0 / np.mean(inv_mle_per)) if mackay_correction else float("nan")

    return mle_mean, mle_corrected, mle_per


def intrinsic_dimension_mle_interval(
    X: ArrayLike,
    k1: int = 10,
    k2: int = 20,
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
    extra_neighbors: int = 20,
    mackay_correction: bool = True,
) -> Dict[str, List[float]]:
    """
    Estimate intrinsic dimension for a range of k values [k1, k2].

    Returns a dict with keys:
      - "k": list of k values
      - "mle": list of mean MLE estimates
      - "mle_corrected": list of corrected estimates
    """
    if k2 < k1:
        raise ValueError("k2 must be >= k1.")
    if k1 < 3:
        raise ValueError("k1 must be >= 3.")

    Xn = _to_numpy(X)

    out_k: List[int] = []
    out_mle: List[float] = []
    out_corr: List[float] = []

    for k in range(k1, k2 + 1):
        mle_mean, mle_corr, _ = intrinsic_dimension_mle(
            Xn,
            k=k,
            metric=metric,
            n_jobs=n_jobs,
            extra_neighbors=extra_neighbors,
            return_details=True,
            mackay_correction=mackay_correction,
        )
        out_k.append(k)
        out_mle.append(mle_mean)
        out_corr.append(mle_corr)

    return {"k": out_k, "mle": out_mle, "mle_corrected": out_corr}
