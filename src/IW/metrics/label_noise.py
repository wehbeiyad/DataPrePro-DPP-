from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
except ImportError as e:
    find_label_issues = None
    get_label_quality_scores = None

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class LabelNoiseReport:
    """Summary of label-noise / label-issue estimation."""
    issue_indices: np.ndarray           # indices suspected as label issues (ranked)
    issue_mask: np.ndarray              # boolean mask of length n_samples
    issue_fraction: float               # len(issue_indices) / n_samples
    label_quality_scores: np.ndarray    # per-sample score in [0, 1], lower = more suspicious


def estimate_label_noise_cleanlab(
    X: Union[np.ndarray, "object"],
    y: Union[np.ndarray, "object"],
    *,
    clf: Optional[object] = None,
    cv: int = 5,
    seed: int = 42,
    method: str = "self_confidence",
    n_jobs: Optional[int] = 1,
) -> LabelNoiseReport:
    """
    Estimate label issues / noise using cleanlab.

    Strategy:
      1) get out-of-sample predicted probabilities via cross-validation,
      2) compute label quality scores,
      3) identify indices of suspected label issues.

    Notes:
      - cleanlab expects pred_probs to be out-of-sample (CV) for best results.
      - For small clean datasets like Iris, you may get 0 or very few issues.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
    y : array-like (n_samples,)
    clf : sklearn-like classifier with predict_proba
        Default: StandardScaler + LogisticRegression.
    cv : int
        Number of folds for StratifiedKFold.
    seed : int
        Random seed used by CV splitter.
    method : str
        cleanlab label-quality method (e.g., "self_confidence", "normalized_margin",
        "confidence_weighted_entropy"). :contentReference[oaicite:1]{index=1}
    n_jobs : int or None
        Parallelism for cross_val_predict where supported.

    Returns
    -------
    LabelNoiseReport
    """
    if find_label_issues is None or get_label_quality_scores is None:
        raise ImportError(
            "cleanlab is not installed. Install it via: pip install cleanlab"
        )

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    # Default classifier: simple, strong baseline for tabular data
    if clf is None:
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, n_jobs=n_jobs)),
            ]
        )

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    # Out-of-sample predicted probabilities
    pred_probs = cross_val_predict(
        clf, X, y,
        cv=splitter,
        method="predict_proba",
        n_jobs=n_jobs,
    )

    # Label quality scores: lower means label more likely wrong. :contentReference[oaicite:2]{index=2}
    label_quality_scores = get_label_quality_scores(
        labels=y,
        pred_probs=pred_probs,
        method=method,
        adjust_pred_probs=False,
    )

    # Find label issues (ranked indices). :contentReference[oaicite:3]{index=3}
    issue_indices = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by=method,
        n_jobs=1,            # <- IMPORTANT: disable multiprocessing
        verbose=False,
    )

    issue_indices = np.asarray(issue_indices, dtype=int)
    issue_mask = np.zeros(len(y), dtype=bool)
    issue_mask[issue_indices] = True
    issue_fraction = float(issue_mask.mean())

    return LabelNoiseReport(
        issue_indices=issue_indices,
        issue_mask=issue_mask,
        issue_fraction=issue_fraction,
        label_quality_scores=label_quality_scores,
    )
