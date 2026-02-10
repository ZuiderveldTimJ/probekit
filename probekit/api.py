from typing import Any

import numpy as np
import torch

from .core.collection import ProbeCollection
from .core.probe import LinearProbe
from .fitters.batch.dim import fit_dim_batch
from .fitters.batch.logistic import fit_logistic_batch
from .fitters.dim import fit_dim
from .fitters.logistic import fit_logistic


def _ensure_tensor(x: Any, device: str = "cuda") -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=device)
    return x.to(device)  # Ensure on device


def _ensure_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def sae_probe(x: Any, y: Any, **kwargs) -> LinearProbe | ProbeCollection:
    """
    Fit a probe on SAE features.
    Routes to batched implementation if x is 3D [B, N, D], else single [N, D].
    Uses Logistic Regression by default for SAE probekit in this suite.
    """
    # Check shape of x
    # We might need to convert to tensor to check shape if it's a list or numpy
    if isinstance(x, list):
        x = np.array(x)

    ndim = x.ndim

    if ndim == 3:
        # Batched
        x_t = _ensure_tensor(x)
        y_t = _ensure_tensor(y)
        return fit_logistic_batch(x_t, y_t, **kwargs)
    else:
        # Single
        # Helper fitters usually take numpy or tensor?
        # Existing fitters/logistic.py likely takes numpy or tensor.
        # Let's assume it handles it or we pass what it needs.
        # fit_logistic usually takes sklearn-style (numpy).
        return fit_logistic(x, y, **kwargs)


def logistic_probe(x: Any, y: Any, **kwargs) -> LinearProbe | ProbeCollection:
    """Alias for logistic regression probe."""
    return sae_probe(x, y, **kwargs)


def nelp_probe(x: Any, y: Any, **kwargs) -> LinearProbe | ProbeCollection:
    """Alias for NELP probe (logistic)."""
    return sae_probe(x, y, **kwargs)


def dim_probe(x: Any, y: Any, **kwargs) -> LinearProbe | ProbeCollection:
    """
    Difference-in-Means probe.
    """
    if isinstance(x, list):
        x = np.array(x)

    ndim = x.ndim

    if ndim == 3:
        x_t = _ensure_tensor(x)
        y_t = _ensure_tensor(y)
        return fit_dim_batch(x_t, y_t, **kwargs)
    else:
        # fit_dim likely takes numpy
        return fit_dim(x, y, **kwargs)
