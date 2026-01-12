# src/qmetrics/datasets/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    label_names: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")
        object.__setattr__(self, "X", X)

        if self.y is not None:
            y = np.asarray(self.y)
            if y.ndim != 1:
                raise ValueError(f"y must be 1D (n_samples,), got shape {y.shape}")
            if y.shape[0] != X.shape[0]:
                raise ValueError(f"X and y must have same n_samples, got {X.shape[0]} and {y.shape[0]}")
            object.__setattr__(self, "y", y)

        if self.feature_names and len(self.feature_names) != X.shape[1]:
            raise ValueError("feature_names length must match n_features")

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> Optional[int]:
        if self.y is None:
            return None
        return int(np.unique(self.y).size)
