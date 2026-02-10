"""
Common utilities for NELP probekit.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, make_scorer


class _IndexList(list[int]):
    """List with numpy-like tolist() for backward compatibility."""

    def tolist(self) -> list[int]:
        return list(self)


def classification_scorer() -> Any:
    """Return a scorer for binary classification correctness."""
    return make_scorer(
        lambda y_true, y_pred: accuracy_score(
            (np.asarray(y_true) > 0.5).astype(np.int32),
            (np.asarray(y_pred) > 0.5).astype(np.int32),
        )
    )


def validate_training_data(
    activations: NDArray[np.float64],
    labels: NDArray[np.float64 | np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64 | np.int64], int, int]:
    """
    Validate and prepare training data.

    Returns:
        (activations, labels, n_samples, n_features)
    """
    activations = np.asarray(activations)
    labels = np.asarray(labels)

    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got shape {activations.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if activations.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch: {activations.shape[0]} samples but {labels.shape[0]} labels")

    n_samples, n_features = activations.shape
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for cross-validated probe fitting, got {n_samples}")

    return activations, labels, n_samples, n_features
