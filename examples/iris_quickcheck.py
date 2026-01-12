# examples/iris_quickcheck.py
from src.datasets import load_iris_dataset
import numpy as np

ds = load_iris_dataset()

print("Dataset:", ds.name)
print("X shape:", ds.X.shape)
print("y shape:", None if ds.y is None else ds.y.shape)
print("n_samples:", ds.n_samples)
print("n_features:", ds.n_features)
print("n_classes:", ds.n_classes)
print("feature_names:", ds.feature_names)
print("label_names:", ds.label_names)

# sanity: class counts
unique, counts = np.unique(ds.y, return_counts=True)
print("class_counts:", dict(zip(unique.tolist(), counts.tolist())))
