from collections.abc import Iterator
from typing import Any, Union, cast, overload

import numpy as np
import torch
from numpy.typing import NDArray

from .probe import LinearProbe, NormalizationStats


class ProbeCollection:
    """
    A collection of linear probes stored as batched tensors.
    Provides utility methods for batch operations and analysis.
    """

    def __init__(
        self,
        weights: torch.Tensor | NDArray[Any] | list[LinearProbe],
        biases: torch.Tensor | NDArray[Any] | None = None,
        normalizations: list[NormalizationStats | None] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ):
        if isinstance(weights, list) and len(weights) > 0 and isinstance(weights[0], LinearProbe):
            # Backwards compatibility: initialize from a list of LinearProbe objects
            self.weights = np.stack([p.weights for p in weights])
            self.biases = np.array([p.bias for p in weights], dtype=np.float32)
            self.normalizations = [p.normalization for p in weights]
            self.metadatas = [p.metadata for p in weights]
        else:
            if isinstance(weights, torch.Tensor):
                self.weights = weights.detach().cpu().numpy()
            else:
                self.weights = np.array(weights)

            if biases is not None:
                if isinstance(biases, torch.Tensor):
                    self.biases = biases.detach().cpu().numpy()
                else:
                    self.biases = np.array(biases)
            else:
                self.biases = np.zeros(self.weights.shape[0], dtype=np.float32)

            self.normalizations = normalizations or [None] * self.weights.shape[0]
            self.metadatas = metadatas or [{}] * self.weights.shape[0]

        # Ensure correct dtype
        self.weights = self.weights.astype(np.float32)
        self.biases = self.biases.astype(np.float32)

    def __len__(self) -> int:
        return self.weights.shape[0]

    @overload
    def __getitem__(self, index: int) -> LinearProbe:
        ...

    @overload
    def __getitem__(self, index: slice) -> "ProbeCollection":
        ...

    def __getitem__(self, index: int | slice) -> Union[LinearProbe, "ProbeCollection"]:
        if isinstance(index, slice):
            return ProbeCollection(
                weights=self.weights[index],
                biases=self.biases[index],
                normalizations=self.normalizations[index],
                metadatas=self.metadatas[index],
            )
        return LinearProbe(
            weights=self.weights[index],
            bias=float(self.biases[index]),
            normalization=self.normalizations[index],
            metadata=self.metadatas[index],
        )

    def __iter__(self) -> Iterator[LinearProbe]:
        for i in range(len(self)):
            yield self[i]

    def predict_score(self, x: NDArray[np.float32] | torch.Tensor) -> NDArray[np.float32]:
        """
        Predict scores for all probes.
        Supports x as [N, D] (same data for every probe) or [B, N, D] (per-probe data).
        Returns [B, N] scores.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        batch_size = len(self)

        if x.ndim not in (2, 3):
            raise ValueError(f"Expected x with 2 or 3 dims, got {x.ndim}.")
        if x.ndim == 3 and x.shape[0] != batch_size:
            raise ValueError(f"Expected x batch dimension {batch_size}, got {x.shape[0]}.")

        d = self.weights.shape[1]

        if x.ndim == 2:
            x_batch = np.broadcast_to(x[np.newaxis], (batch_size, x.shape[0], d))
        else:
            x_batch = x

        # Apply normalization vectorized
        if any(norm is not None for norm in self.normalizations):
            mu = np.zeros((batch_size, d), dtype=np.float32)
            std = np.ones((batch_size, d), dtype=np.float32)

            for i, norm in enumerate(self.normalizations):
                if norm is not None:
                    mu[i] = norm.mean
                    std[i] = norm.std

            std = np.where(std == 0, 1.0, std)
            x_normed = (x_batch - mu[:, np.newaxis, :]) / std[:, np.newaxis, :]
        else:
            x_normed = x_batch

        # Vectorized score: [B, N, D] @ [B, D, 1] -> [B, N]
        scores = np.squeeze(x_normed @ self.weights[:, :, np.newaxis], axis=-1)
        scores = (scores + self.biases[:, np.newaxis]).astype(np.float32)
        return cast(NDArray[np.float32], scores)

    def predict(self, x: NDArray[np.float32] | torch.Tensor, threshold: float = 0.0) -> NDArray[np.int32]:
        """
        Predict binary classes for all probes.
        Supports x as [N, D] (same data for every probe) or [B, N, D] (per-probe data).
        Returns [B, N] predictions.
        """
        scores = self.predict_score(x)
        return (scores > threshold).astype(np.int32)

    def to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns stacked weight tensor [B, D] and bias tensor [B].
        """
        return torch.from_numpy(self.weights).float(), torch.from_numpy(self.biases).float()

    def best_layer(self, metric: str = "val_accuracy") -> tuple[int, LinearProbe]:
        """
        Returns (layer_index, probe) for the probe with the highest value of the given metric.
        Raises KeyError if metric is missing in any probe's metadata.
        """
        best_idx = -1
        best_val = -float("inf")

        for i, meta in enumerate(self.metadatas):
            if metric not in meta:
                continue
            val = meta[metric]
            if val > best_val:
                best_val = val
                best_idx = i

        if best_idx == -1:
            raise ValueError(f"No probes found with metric '{metric}' or collection is empty.")

        return best_idx, self[best_idx]
