"""NumPy-only inference for the deployed Dense network.

The trained Keras model is exported by src/export_model.py with BatchNorm and
the StandardScaler folded into the Dense weights, so a prediction is just
matrix multiplications + ReLU + softmax. This module imports NumPy only, which
keeps the Vercel function small and fast to start.
"""
from pathlib import Path
from typing import List

import numpy as np


class HARModel:
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            n = int(data["n_layers"])
            self.layers = [(data[f"W{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32)) for i in range(n)]
            self.classes: List[str] = [str(c) for c in data["classes"]]
            self.features: List[str] = [str(f) for f in data["features"]]
        self.n_features = self.layers[0][0].shape[0]
        if self.n_features != len(self.features):
            raise ValueError("Model file is inconsistent: feature names do not match the first layer")

    def predict_proba(self, x) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        if h.ndim == 1:
            h = h[None, :]
        if h.ndim != 2 or h.shape[1] != self.n_features:
            raise ValueError(f"expected shape (n, {self.n_features}), got {tuple(h.shape)}")
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)                 # ReLU (BatchNorm already folded into w, b)
        h = h - h.max(axis=1, keepdims=True)           # numerically stable softmax
        e = np.exp(h)
        p = e / e.sum(axis=1, keepdims=True)
        return np.nan_to_num(p, nan=1.0 / len(self.classes))

    def predict(self, x):
        p = self.predict_proba(x)
        return [self.classes[i] for i in p.argmax(1)], p.max(1)
