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

from probekit.utils.normalization import safe_std_numpy, safe_std_torch


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

    def _predict_score_torch(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.to(dtype=torch.float32)

        if self.normalization:
            mean = torch.as_tensor(self.normalization.mean, device=x_t.device, dtype=x_t.dtype)
            std = torch.as_tensor(self.normalization.std, device=x_t.device, dtype=x_t.dtype)
            x_t = (x_t - mean) / safe_std_torch(std)

        weights = torch.as_tensor(self.weights, device=x_t.device, dtype=x_t.dtype)

        if self.weights.ndim == 1:
            bias = torch.as_tensor(self.bias, device=x_t.device, dtype=x_t.dtype)
            return x_t @ weights + bias

        bias = torch.as_tensor(self.bias, device=x_t.device, dtype=x_t.dtype)
        return x_t @ weights.T + bias

    def predict_score(self, x: NDArray[np.float32] | torch.Tensor) -> NDArray[np.float32] | torch.Tensor:
        """
        Compute raw scores (logits): s(x) = w^T * norm(x) + b.
        """
        if isinstance(x, torch.Tensor):
            return self._predict_score_torch(x)

        # 1. Normalize
        if self.normalization:
            std = safe_std_numpy(self.normalization.std)
            x = (x - self.normalization.mean) / std

        # 2. Linear projection
        # If weights are 1D, use dot product
        if self.weights.ndim == 1:
            scores = (x @ self.weights + self.bias).astype(np.float32)
        else:
            # x: [N, D], weights: [K, D] -> [N, K]
            scores = (x @ self.weights.T + self.bias).astype(np.float32)

        return scores

    def project(self, x: NDArray[np.float32] | torch.Tensor) -> NDArray[np.float32] | torch.Tensor:
        """
        Project x onto the probe direction (without bias).
        """
        if isinstance(x, torch.Tensor):
            x_t = x.to(dtype=torch.float32)
            if self.normalization:
                mean = torch.as_tensor(self.normalization.mean, device=x_t.device, dtype=x_t.dtype)
                std = torch.as_tensor(self.normalization.std, device=x_t.device, dtype=x_t.dtype)
                x_t = (x_t - mean) / safe_std_torch(std)

            weights = torch.as_tensor(self.weights, device=x_t.device, dtype=x_t.dtype)
            if self.weights.ndim == 1:
                return x_t @ weights
            return x_t @ weights.T

        x_np = x
        if self.normalization:
            std_np = safe_std_numpy(self.normalization.std)
            x_np = cast(NDArray[np.float32], (x_np - self.normalization.mean) / std_np)

        if self.weights.ndim == 1:
            return cast(NDArray[np.float32], (x_np @ self.weights).astype(np.float32))
        else:
            return cast(NDArray[np.float32], (x_np @ self.weights.T).astype(np.float32))

    def predict(
        self, x: NDArray[np.float32] | torch.Tensor, threshold: float = 0.0
    ) -> NDArray[np.int32] | torch.Tensor:
        """
        Predict binary classes based on threshold.
        """
        scores = self.predict_score(x)
        if isinstance(scores, torch.Tensor):
            return (scores > threshold).to(torch.int32)
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
