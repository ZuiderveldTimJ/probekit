"""
Core Probe Model (V2).

This module defines the `LinearProbe` class, which is a mechanism-agnostic
container for linear models. It separates the "what" (weights, bias) from the
"how" (fitters).
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class NormalizationStats:
    """Statistics for input normalization."""
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizationStats":
        return cls(
            mean=np.array(data["mean"]),
            std=np.array(data["std"]),
            count=data["count"],
        )


class LinearProbe:
    """
    A generic linear probe model: y = w^T * (x - mu) / sigma + b.

    Attributes:
        weights (NDArray): The direction vector (shape: [d_in] or [d_out, d_in]).
        bias (NDArray | float): The bias term.
        normalization (NormalizationStats | None): Input normalization stats.
        metadata (dict): Arbitrary metadata (hyperparams, training stats, etc.).
    """

    def __init__(
        self,
        weights: NDArray[np.float32],
        bias: NDArray[np.float32] | float = 0.0,
        normalization: NormalizationStats | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.weights = weights
        self.bias = bias
        self.normalization = normalization
        self.metadata = metadata or {}

    @property
    def direction(self) -> NDArray[np.float32]:
        """Return the steering direction (weights)."""
        # If weights are [d_out, d_in] and d_out=1, flatten them.
        if self.weights.ndim == 2 and self.weights.shape[0] == 1:
            return self.weights.flatten()
        return self.weights

    def predict_score(self, x: NDArray[np.float32] | torch.Tensor) -> NDArray[np.float32]:
        """
        Compute raw scores (logits): s(x) = w^T * norm(x) + b.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # 1. Normalize
        if self.normalization:
            # Avoid division by zero
            std = self.normalization.std
            std = np.where(std == 0, 1.0, std)
            x = (x - self.normalization.mean) / std

        # 2. Linear projection
        # If weights are 1D, use dot product
        if self.weights.ndim == 1:
            scores = (x @ self.weights + self.bias).astype(np.float32)
        else:
            # x: [N, D], weights: [K, D] -> [N, K]
            scores = (x @ self.weights.T + self.bias).astype(np.float32)

        return scores

    def project(self, x: NDArray[np.float32] | torch.Tensor) -> NDArray[np.float32]:
        """
        Project x onto the probe direction (without bias).
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if self.normalization:
            std = self.normalization.std
            std = np.where(std == 0, 1.0, std)
            x = (x - self.normalization.mean) / std

        if self.weights.ndim == 1:
            return cast(NDArray[np.float32], (x @ self.weights).astype(np.float32))
        else:
            return cast(NDArray[np.float32], (x @ self.weights.T).astype(np.float32))

    def predict(self, x: NDArray[np.float32] | torch.Tensor, threshold: float = 0.0) -> NDArray[np.int32]:
        """
        Predict binary classes based on threshold.
        """
        scores = self.predict_score(x)
        return (scores > threshold).astype(np.int32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize probe to dictionary."""
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias if isinstance(self.bias, float) else self.bias.tolist(),
            "normalization": self.normalization.to_dict() if self.normalization else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearProbe":
        """Deserialize probe from dictionary."""
        norm_data = data.get("normalization")
        norm = NormalizationStats.from_dict(norm_data) if norm_data else None

        return cls(
            weights=np.array(data["weights"], dtype=np.float32),
            bias=np.array(data["bias"], dtype=np.float32) if isinstance(data["bias"], list) else data["bias"],
            normalization=norm,
            metadata=data.get("metadata", {}),
        )
