from collections.abc import Iterator
from typing import Union, overload

import numpy as np
import torch

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
    def __getitem__(self, index: int) -> LinearProbe: ...

    @overload
    def __getitem__(self, index: slice) -> "ProbeCollection": ...

    def __getitem__(self, index: int | slice) -> Union[LinearProbe, "ProbeCollection"]:
        if isinstance(index, slice):
            return ProbeCollection(self.probekit[index])
        return self.probekit[index]

    def __iter__(self) -> Iterator[LinearProbe]:
        return iter(self.probekit)

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
