"""
Probe result container.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


class _IndexList(list[int]):
    """List with numpy-like tolist() for backward compatibility."""

    def tolist(self) -> list[int]:
        return list(self)


@dataclass
class ProbeResult:
    """Results from training a NELP probe."""

    accuracy: float
    accuracy_std: float = 0.0
    sparse_neurons: NDArray[np.int64] | list[int] = field(default_factory=list)
    coefficients: NDArray[np.float64] | list[float] = field(default_factory=list)
    intercept: float = 0.0
    n_features: int = 0
    n_samples: int = 0
    # Backward-compatible aliases expected by older callers/tests.
    weights: list[tuple[int, float]] | None = None
    layer: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sparse_neurons_arr = np.asarray(self.sparse_neurons, dtype=np.int64)
        self.sparse_neurons = _IndexList(int(i) for i in sparse_neurons_arr.tolist())
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)

        if self.weights is not None:
            self.weights = [(int(i), float(w)) for i, w in self.weights]
            if self.coefficients.size == 0 and self.weights:
                inferred_size = max(i for i, _ in self.weights) + 1
                size = max(self.n_features, inferred_size)
                coeffs = np.zeros(size, dtype=np.float64)
                for idx, weight in self.weights:
                    coeffs[idx] = weight
                self.coefficients = coeffs
            if len(self.sparse_neurons) == 0 and self.weights:
                self.sparse_neurons = _IndexList(int(i) for i, _ in self.weights)
        else:
            self.weights = self.get_weighted_neurons(min_weight=0.01)

        if self.n_features == 0 and self.coefficients.size > 0:
            self.n_features = int(self.coefficients.size)

    def get_weighted_neurons(self, min_weight: float = 0.01) -> list[tuple[int, float]]:
        """Return (neuron_idx, coefficient) pairs sorted by magnitude."""
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        mask = np.abs(coefficients) > min_weight
        indices = np.where(mask)[0]
        weights = coefficients[mask]

        # Sort by absolute magnitude, descending
        order = np.argsort(np.abs(weights))[::-1]
        return [(int(indices[i]), float(weights[i])) for i in order]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        sparse_neurons = list(self.sparse_neurons)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        return {
            "accuracy": self.accuracy,
            "accuracy_std": self.accuracy_std,
            "sparse_neurons": sparse_neurons,
            "coefficients": coefficients.tolist(),
            "intercept": self.intercept,
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "weights": self.weights,
            "layer": self.layer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbeResult":
        """Create from dictionary."""
        return cls(
            accuracy=data["accuracy"],
            accuracy_std=data.get("accuracy_std", 0.0),
            sparse_neurons=data.get("sparse_neurons", []),
            coefficients=data.get("coefficients", []),
            intercept=data.get("intercept", 0.0),
            n_features=data.get("n_features", 0),
            n_samples=data.get("n_samples", 0),
            weights=data.get("weights"),
            layer=data.get("layer"),
            metadata=data.get("metadata", {}),
        )
