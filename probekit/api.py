from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .core.collection import ProbeCollection
from .core.probe import LinearProbe
from .fitters.batch.dim import fit_dim_batch
from .fitters.batch.logistic import fit_logistic_batch
from .fitters.dim import fit_dim
from .fitters.elastic import fit_elastic_net
from .fitters.logistic import fit_logistic


def _ensure_tensor(x: Any, device: str = "cuda") -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=device)
    return x.to(device)


def _ensure_numpy(x: Any) -> NDArray[Any]:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _fit_elastic_net_batch_search(x_t: torch.Tensor, y_t: torch.Tensor, **kwargs: Any) -> ProbeCollection:
    from .fitters.batch.path import fit_elastic_net_path

    l1_ratios = kwargs.pop("l1_ratios", [0.5])
    if "l1_ratio" in kwargs:
        l1_ratios = [kwargs.pop("l1_ratio")]

    best_probes: list[LinearProbe | None] = [None] * x_t.shape[0]
    best_accs = [-1.0] * x_t.shape[0]

    for l1_ratio in l1_ratios:
        col = fit_elastic_net_path(x_t, y_t, l1_ratio=l1_ratio, **kwargs)
        if isinstance(col, list):
            col = col[-1]  # fallback if select="all" was passed inexplicably

        for i in range(x_t.shape[0]):
            p = col[i]
            acc = p.metadata.get("val_accuracy", -1.0)
            if best_probes[i] is None or acc > best_accs[i]:
                best_accs[i] = acc
                best_probes[i] = p

    from typing import cast

    return ProbeCollection(cast(list[LinearProbe], best_probes))


def sae_probe(x: Any, y: Any, **kwargs: Any) -> LinearProbe | ProbeCollection:
    """
    Fit a sparse probe on SAE features using ElasticNet (L1-heavy).
    """
    if isinstance(x, list):
        x = np.array(x)

    ndim = x.ndim

    if ndim == 3:
        x_t = _ensure_tensor(x)
        y_t = _ensure_tensor(y)
        kwargs.setdefault("l1_ratios", [0.5, 0.7, 0.9, 0.95, 1.0])
        return _fit_elastic_net_batch_search(x_t, y_t, **kwargs)
    else:
        # Default L1-heavy ratios for SAE sparsity
        kwargs.setdefault("l1_ratios", [0.5, 0.7, 0.9, 0.95, 1.0])
        return fit_elastic_net(x, y, **kwargs)


def logistic_probe(x: Any, y: Any, **kwargs: Any) -> LinearProbe | ProbeCollection:
    """
    Fit a standard L2-regularized Logistic Regression probe (dense weights).
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
    """
    if isinstance(x, list):
        x = np.array(x)

    ndim = x.ndim

    if ndim == 3:
        x_t = _ensure_tensor(x)
        y_t = _ensure_tensor(y)
        kwargs.setdefault("l1_ratios", [0.1, 0.5, 0.7, 0.9])
        return _fit_elastic_net_batch_search(x_t, y_t, **kwargs)
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
        return fit_dim(x, y, **kwargs)
