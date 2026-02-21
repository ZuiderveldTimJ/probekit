from collections.abc import Iterator
from typing import Any, Union, cast, overload

import numpy as np
import torch
from numpy.typing import NDArray

from probekit.utils.normalization import safe_std_numpy, safe_std_torch

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
        self._weights_torch: torch.Tensor | None = None
        self._biases_torch: torch.Tensor | None = None

        if isinstance(weights, list) and len(weights) > 0 and isinstance(weights[0], LinearProbe):
            # Backwards compatibility: initialize from a list of LinearProbe objects
            self.weights = np.stack([p.weights for p in weights])
            self.biases = np.array([p.bias for p in weights], dtype=np.float32)
            self.normalizations = [p.normalization for p in weights]
            self.metadatas = [p.metadata for p in weights]
        else:
            if isinstance(weights, torch.Tensor):
                self._weights_torch = weights.detach().to(dtype=torch.float32)
                self.weights = self._weights_torch.cpu().numpy()
            else:
                self.weights = np.array(weights)

            if biases is not None:
                if isinstance(biases, torch.Tensor):
                    self._biases_torch = biases.detach().to(dtype=torch.float32)
                    self.biases = self._biases_torch.cpu().numpy()
                else:
                    self.biases = np.array(biases)
                    if self._weights_torch is not None:
                        self._biases_torch = torch.as_tensor(
                            self.biases, device=self._weights_torch.device, dtype=torch.float32
                        )
            else:
                self.biases = np.zeros(self.weights.shape[0], dtype=np.float32)
                if self._weights_torch is not None:
                    self._biases_torch = torch.zeros(self.weights.shape[0], device=self._weights_torch.device)

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

    def _predict_score_numpy(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        batch_size = len(self)
        d = self.weights.shape[1]

        if x.ndim not in (2, 3):
            raise ValueError(f"Expected x with 2 or 3 dims, got {x.ndim}.")
        if x.ndim == 3 and x.shape[0] != batch_size:
            raise ValueError(f"Expected x batch dimension {batch_size}, got {x.shape[0]}.")

        if x.ndim == 2:
            x_batch = np.broadcast_to(x[np.newaxis], (batch_size, x.shape[0], d))
        else:
            x_batch = x

        if any(norm is not None for norm in self.normalizations):
            mu = np.zeros((batch_size, d), dtype=np.float32)
            std = np.ones((batch_size, d), dtype=np.float32)

            for i, norm in enumerate(self.normalizations):
                if norm is not None:
                    mu[i] = norm.mean
                    std[i] = norm.std

            std = safe_std_numpy(std)
            x_normed = (x_batch - mu[:, np.newaxis, :]) / std[:, np.newaxis, :]
        else:
            x_normed = x_batch

        scores = np.squeeze(x_normed @ self.weights[:, :, np.newaxis], axis=-1)
        scores = (scores + self.biases[:, np.newaxis]).astype(np.float32)
        return cast(NDArray[np.float32], scores)

    def _predict_score_torch(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.to(dtype=torch.float32)
        batch_size = len(self)
        weights, biases = self.to_tensor(device=x_t.device)
        d = weights.shape[1]

        if x_t.ndim not in (2, 3):
            raise ValueError(f"Expected x with 2 or 3 dims, got {x_t.ndim}.")
        if x_t.ndim == 3 and x_t.shape[0] != batch_size:
            raise ValueError(f"Expected x batch dimension {batch_size}, got {x_t.shape[0]}.")

        if x_t.ndim == 2:
            x_batch = x_t.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            x_batch = x_t

        if any(norm is not None for norm in self.normalizations):
            mu = torch.zeros((batch_size, d), dtype=x_batch.dtype, device=x_batch.device)
            std = torch.ones((batch_size, d), dtype=x_batch.dtype, device=x_batch.device)
            for i, norm in enumerate(self.normalizations):
                if norm is not None:
                    mu[i] = torch.as_tensor(norm.mean, dtype=x_batch.dtype, device=x_batch.device)
                    std[i] = torch.as_tensor(norm.std, dtype=x_batch.dtype, device=x_batch.device)

            x_normed = (x_batch - mu.unsqueeze(1)) / safe_std_torch(std).unsqueeze(1)
        else:
            x_normed = x_batch

        scores = torch.bmm(x_normed, weights.unsqueeze(-1)).squeeze(-1)
        return scores + biases.unsqueeze(1)

    def predict_score(self, x: NDArray[np.float32] | torch.Tensor) -> NDArray[np.float32] | torch.Tensor:
        """
        Predict scores for all probes.
        Supports x as [N, D] (same data for every probe) or [B, N, D] (per-probe data).
        Returns [B, N] scores.
        """
        if isinstance(x, torch.Tensor):
            return self._predict_score_torch(x)
        return self._predict_score_numpy(x)

    def predict(
        self, x: NDArray[np.float32] | torch.Tensor, threshold: float = 0.0
    ) -> NDArray[np.int32] | torch.Tensor:
        """
        Predict binary classes for all probes.
        Supports x as [N, D] (same data for every probe) or [B, N, D] (per-probe data).
        Returns [B, N] predictions.
        """
        scores = self.predict_score(x)
        if isinstance(scores, torch.Tensor):
            return (scores > threshold).to(torch.int32)
        return (scores > threshold).astype(np.int32)

    def to_tensor(self, device: str | torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns stacked weight tensor [B, D] and bias tensor [B].
        """
        if self._weights_torch is not None and self._biases_torch is not None:
            if device is None:
                return self._weights_torch, self._biases_torch
            return self._weights_torch.to(device), self._biases_torch.to(device)

        weights = torch.from_numpy(self.weights).float()
        biases = torch.from_numpy(self.biases).float()
        if device is None:
            return weights, biases
        return weights.to(device), biases.to(device)

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
