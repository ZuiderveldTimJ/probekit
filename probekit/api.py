from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .core.collection import ProbeCollection
from .core.probe import LinearProbe
from .fitters.batch.dim import fit_dim_batch
from .fitters.batch.elastic import fit_elastic_net_batch
from .fitters.batch.logistic import fit_logistic_batch
from .fitters.dim import fit_dim
from .fitters.elastic import fit_elastic_net
from .fitters.logistic import fit_logistic


def _ensure_tensor(x: Any, device: str = "cuda") -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=device)
    return x.to(device)  # Ensure on device


def _ensure_numpy(x: Any) -> NDArray[Any]:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def sae_probe(x: Any, y: Any, **kwargs: Any) -> LinearProbe | ProbeCollection:
    """
    Fit a sparse probe on SAE features using ElasticNet (L1-heavy).

    SAE features are typically high-dimensional and sparse, so L1
    regularization is used to select the most informative features.
    Uses ElasticNet with high L1 ratios by default.
    """
    if isinstance(x, list):
        x = np.array(x)

    ndim = x.ndim

    if ndim == 3:
        x_t = _ensure_tensor(x)
        y_t = _ensure_tensor(y)
        # Default to high sparsity for SAE features
        kwargs.setdefault("l1_ratio", 0.9)
        return fit_elastic_net_batch(x_t, y_t, **kwargs)
    else:
        # Default L1-heavy ratios for SAE sparsity
        kwargs.setdefault("l1_ratios", [0.5, 0.7, 0.9, 0.95, 1.0])
        return fit_elastic_net(x, y, **kwargs)


def logistic_probe(x: Any, y: Any, **kwargs: Any) -> LinearProbe | ProbeCollection:
    """
    Fit a standard L2-regularized Logistic Regression probe (dense weights).

    This is the baseline probe — no sparsity, all features contribute.
    """
    if isinstance(x, list):
        x = np.array(x)

    ndim = x.ndim

    if ndim == 3:
        x_t = _ensure_tensor(x)
        y_t = _ensure_tensor(y)
        return fit_logistic_batch(x_t, y_t, **kwargs)
    else:
        return fit_logistic(x, y, **kwargs)


def nelp_probe(x: Any, y: Any, **kwargs: Any) -> LinearProbe | ProbeCollection:
    """
    Fit a NELP (Non-linear Embedding Linear Probe) using ElasticNet.

    Uses balanced L1/L2 regularization — less sparse than sae_probe
    but still encourages feature selection on dense activations.
    """
    if isinstance(x, list):
        x = np.array(x)

    ndim = x.ndim

    if ndim == 3:
        x_t = _ensure_tensor(x)
        y_t = _ensure_tensor(y)
        kwargs.setdefault("l1_ratio", 0.5)
        return fit_elastic_net_batch(x_t, y_t, **kwargs)
    else:
        kwargs.setdefault("l1_ratios", [0.1, 0.5, 0.7, 0.9])
        return fit_elastic_net(x, y, **kwargs)


def dim_probe(x: Any, y: Any, **kwargs: Any) -> LinearProbe | ProbeCollection:
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
