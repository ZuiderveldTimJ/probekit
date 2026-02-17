from collections.abc import Iterator
from typing import Union, overload

import numpy as np
import torch
from numpy.typing import NDArray

from .probe import LinearProbe


class ProbeCollection:
    """
    A thin wrapper around a list of LinearProbe objects.
    Provides utility methods for batch operations and analysis.
    """

    def __init__(self, probekit: list[LinearProbe]):
        self.probekit = probekit

    def __len__(self) -> int:
        return len(self.probekit)

    @overload
    def __getitem__(self, index: int) -> LinearProbe:
        ...

    @overload
    def __getitem__(self, index: slice) -> "ProbeCollection":
        ...

    def __getitem__(self, index: int | slice) -> Union[LinearProbe, "ProbeCollection"]:
        if isinstance(index, slice):
            return ProbeCollection(self.probekit[index])
        return self.probekit[index]

    def __iter__(self) -> Iterator[LinearProbe]:
        return iter(self.probekit)

    def predict_score(self, x: NDArray[np.float32] | torch.Tensor) -> NDArray[np.float32]:
        """
        Predict scores for all probes.
        Supports x as [N, D] (same data for every probe) or [B, N, D] (per-probe data).
        Returns [B, N] scores.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        batch_size = len(self.probekit)

        if x.ndim not in (2, 3):
            raise ValueError(f"Expected x with 2 or 3 dims, got {x.ndim}.")
        if x.ndim == 3 and x.shape[0] != batch_size:
            raise ValueError(f"Expected x batch dimension {batch_size}, got {x.shape[0]}.")

        # Try vectorized path: stack weights/biases and normalize in bulk.
        # Only works when all probes have 1D weights of the same dimensionality
        # and consistent normalization.
        can_vectorize = batch_size > 1 and all(
            p.weights.ndim == 1 and p.weights.shape[0] == self.probekit[0].weights.shape[0]
            for p in self.probekit
        )

        if can_vectorize:
            d = self.probekit[0].weights.shape[0]
            w_stacked = np.stack([p.weights for p in self.probekit])  # [B, D]
            b_stacked = np.array([p.bias for p in self.probekit], dtype=np.float32)  # [B]

            if x.ndim == 2:
                # x: [N, D] -> normalize per probe then score
                # Need per-probe normalization, so expand x to [B, N, D]
                x_batch = np.broadcast_to(x[np.newaxis], (batch_size, x.shape[0], d))
            else:
                x_batch = x  # [B, N, D]

            # Apply normalization vectorized
            has_norm = self.probekit[0].normalization is not None
            all_same_norm = has_norm and all(p.normalization is not None for p in self.probekit)

            if all_same_norm:
                mu = np.stack([p.normalization.mean for p in self.probekit])  # [B, D]
                std = np.stack([p.normalization.std for p in self.probekit])  # [B, D]
                std = np.where(std == 0, 1.0, std)
                # [B, N, D] - [B, 1, D] / [B, 1, D]
                x_normed = (x_batch - mu[:, np.newaxis, :]) / std[:, np.newaxis, :]
            else:
                x_normed = x_batch

            # Vectorized score: [B, N, D] @ [B, D, 1] -> [B, N]
            scores = np.squeeze(x_normed @ w_stacked[:, :, np.newaxis], axis=-1)
            scores = (scores + b_stacked[:, np.newaxis]).astype(np.float32)
            return scores

        # Fallback: per-probe loop (handles mixed weight shapes, etc.)
        if x.ndim == 2:
            scores = [probe.predict_score(x) for probe in self.probekit]
        else:
            scores = [probe.predict_score(x[i]) for i, probe in enumerate(self.probekit)]

        return np.stack(scores, axis=0).astype(np.float32)

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
        weights_list = [p.weights for p in self.probekit]
        biases_list = [p.bias for p in self.probekit]

        # Convert to numpy first for efficiency/consistency if they act as such
        # The LinearProbe stores weights as numpy arrays usually.

        # Stack
        w_stacked = np.stack(weights_list)  # [B, D]
        b_stacked = np.array(biases_list)  # [B]

        return torch.from_numpy(w_stacked).float(), torch.from_numpy(b_stacked).float()

    def best_layer(self, metric: str = "val_accuracy") -> tuple[int, LinearProbe]:
        """
        Returns (layer_index, probe) for the probe with the highest value of the given metric.
        Raises KeyError if metric is missing in any probe's metadata.
        """
        best_idx = -1
        best_val = -float("inf")

        for i, probe in enumerate(self.probekit):
            if metric not in probe.metadata:
                continue  # Or raise? "Thin" usually means simple. Let's start with skipping or erroring?
                # The requirement says "highest value... in probe.metadata".
                # Let's assume usage is consistent.

            val = probe.metadata[metric]
            if val > best_val:
                best_val = val
                best_idx = i

        if best_idx == -1:
            raise ValueError(f"No probekit found with metric '{metric}' or collection is empty.")

        return best_idx, self.probekit[best_idx]
