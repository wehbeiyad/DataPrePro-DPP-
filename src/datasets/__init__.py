# src/qmetrics/datasets/__init__.py
from .iris import load_iris_dataset
from .base import Dataset

__all__ = ["Dataset", "load_iris_dataset"]

