# src/qmetrics/datasets/iris.py
from __future__ import annotations
from sklearn.datasets import load_iris
from .base import Dataset

def load_iris_dataset() -> Dataset:
    bunch = load_iris(as_frame=False)
    X = bunch["data"]
    y = bunch["target"]
    feature_names = list(bunch["feature_names"])
    label_names = list(bunch["target_names"])

    return Dataset(
        name="iris",
        X=X,
        y=y,
        feature_names=feature_names,
        label_names=label_names,
        meta={"source": "sklearn.datasets.load_iris", "n_samples": X.shape[0]},
    )
