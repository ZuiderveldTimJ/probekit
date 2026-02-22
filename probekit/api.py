import os
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from .core.collection import ProbeCollection
from .core.probe import LinearProbe
from .fitters.batch.dim import fit_dim_batch
from .fitters.batch.logistic import fit_logistic_batch
from .fitters.batch.path import fit_elastic_net_path
from .fitters.dim import fit_dim
from .fitters.elastic import fit_elastic_net
from .fitters.logistic import fit_logistic

Backend = Literal["auto", "torch", "sklearn"]
ResolvedBackend = Literal["torch", "sklearn"]


def _ensure_tensor(x: Any, device: str | torch.device | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    if device is None:
        return torch.as_tensor(x)
    return torch.as_tensor(x, device=device)


def _ensure_numpy(x: Any) -> NDArray[Any]:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _resolve_backend(x: Any, backend: Backend) -> ResolvedBackend:
    if backend not in {"auto", "torch", "sklearn"}:
        raise ValueError(f"Invalid backend {backend!r}. Expected 'auto', 'torch', or 'sklearn'.")
    if backend != "auto":
        return backend
    if isinstance(x, torch.Tensor):
        return "torch"
    if isinstance(x, np.ndarray) and x.ndim == 3:
        return "torch"
    return "sklearn"


def _resolve_torch_device(x: Any, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if isinstance(x, torch.Tensor):
        return x.device

    env_device = os.getenv("PROBEKIT_TORCH_DEVICE")
    if env_device:
        return torch.device(env_device)

    try:
        default_device = torch.get_default_device()
        if str(default_device) != "cpu":
            return torch.device(default_device)
    except AttributeError:
        pass

    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _torch_kwargs(kwargs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    torch_kwargs = dict(kwargs)
    if "val_x" in torch_kwargs and torch_kwargs["val_x"] is not None:
        torch_kwargs["val_x"] = _ensure_tensor(torch_kwargs["val_x"], device=device)
    if "val_y" in torch_kwargs and torch_kwargs["val_y"] is not None:
        torch_kwargs["val_y"] = _ensure_tensor(torch_kwargs["val_y"], device=device)
    return torch_kwargs


def _sklearn_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    sklearn_kwargs = dict(kwargs)
    sklearn_kwargs.pop("val_x", None)
    sklearn_kwargs.pop("val_y", None)
    return sklearn_kwargs


def fit_sparse_probe_batch(x_t: torch.Tensor, y_t: torch.Tensor, **kwargs: Any) -> ProbeCollection:
    l1_ratios = kwargs.pop("l1_ratios", [0.5])
    if "l1_ratio" in kwargs:
        l1_ratios = [kwargs.pop("l1_ratio")]

    best_probes: list[LinearProbe | None] = [None] * x_t.shape[0]
    best_accs = [-1.0] * x_t.shape[0]

    for l1_ratio in l1_ratios:
        col = fit_elastic_net_path(x_t, y_t, l1_ratio=l1_ratio, **kwargs)
        if isinstance(col, list):
            col = col[-1]

        for i in range(x_t.shape[0]):
            p = col[i]
            acc = p.metadata.get("val_accuracy", -1.0)
            if best_probes[i] is None or acc > best_accs[i]:
                best_accs[i] = acc
                best_probes[i] = p

    return ProbeCollection(cast(list[LinearProbe], best_probes))


def _fit_logistic_sklearn_batch(x: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> ProbeCollection:
    probes: list[LinearProbe] = []
    for i in range(x.shape[0]):
        y_i = y if y.ndim == 1 else y[i]
        probes.append(fit_logistic(x[i], y_i, **kwargs))
    return ProbeCollection(probes)


def _fit_elastic_sklearn_batch(x: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> ProbeCollection:
    probes: list[LinearProbe] = []
    for i in range(x.shape[0]):
        y_i = y if y.ndim == 1 else y[i]
        probes.append(fit_elastic_net(x[i], y_i, **kwargs))
    return ProbeCollection(probes)


def _fit_dim_sklearn_batch(x: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> ProbeCollection:
    probes: list[LinearProbe] = []
    for i in range(x.shape[0]):
        y_i = y if y.ndim == 1 else y[i]
        probes.append(fit_dim(x[i], y_i, **kwargs))
    return ProbeCollection(probes)


def sae_probe(
    x: Any,
    y: Any,
    backend: Backend = "auto",
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> LinearProbe | ProbeCollection:
    """
    Fit a sparse probe on SAE features using ElasticNet (L1-heavy).
    """
    if isinstance(x, list):
        x = np.array(x)

    resolved_backend = _resolve_backend(x, backend)
    if resolved_backend == "torch":
        target_device = _resolve_torch_device(x, device)
        x_t = _ensure_tensor(x, device=target_device)
        y_t = _ensure_tensor(y, device=x_t.device)
        torch_kwargs = _torch_kwargs(kwargs, x_t.device)
        torch_kwargs.setdefault("l1_ratios", [0.5, 0.7, 0.9, 0.95, 1.0])

        if x_t.ndim == 2:
            col = fit_sparse_probe_batch(x_t.unsqueeze(0), y_t, **torch_kwargs)
            return col[0]
        if x_t.ndim == 3:
            return fit_sparse_probe_batch(x_t, y_t, **torch_kwargs)
        raise ValueError(f"Expected 2D or 3D input for torch backend, got {x_t.ndim}D.")

    x_np = _ensure_numpy(x)
    y_np = _ensure_numpy(y)
    sklearn_kwargs = _sklearn_kwargs(kwargs)
    sklearn_kwargs.setdefault("l1_ratios", [0.5, 0.7, 0.9, 0.95, 1.0])

    if x_np.ndim == 2:
        return fit_elastic_net(x_np, y_np, **sklearn_kwargs)
    if x_np.ndim == 3:
        return _fit_elastic_sklearn_batch(x_np, y_np, **sklearn_kwargs)
    raise ValueError(f"Expected 2D or 3D input, got {x_np.ndim}D.")


def logistic_probe(
    x: Any,
    y: Any,
    backend: Backend = "auto",
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> LinearProbe | ProbeCollection:
    """
    Fit a standard L2-regularized Logistic Regression probe (dense weights).
    """
    if isinstance(x, list):
        x = np.array(x)

    resolved_backend = _resolve_backend(x, backend)
    if resolved_backend == "torch":
        target_device = _resolve_torch_device(x, device)
        x_t = _ensure_tensor(x, device=target_device)
        y_t = _ensure_tensor(y, device=x_t.device)
        torch_kwargs = _torch_kwargs(kwargs, x_t.device)

        if x_t.ndim == 2:
            col = fit_logistic_batch(x_t.unsqueeze(0), y_t, **torch_kwargs)
            return col[0]
        if x_t.ndim == 3:
            return fit_logistic_batch(x_t, y_t, **torch_kwargs)
        raise ValueError(f"Expected 2D or 3D input for torch backend, got {x_t.ndim}D.")

    x_np = _ensure_numpy(x)
    y_np = _ensure_numpy(y)
    sklearn_kwargs = _sklearn_kwargs(kwargs)

    if x_np.ndim == 2:
        return fit_logistic(x_np, y_np, **sklearn_kwargs)
    if x_np.ndim == 3:
        return _fit_logistic_sklearn_batch(x_np, y_np, **sklearn_kwargs)
    raise ValueError(f"Expected 2D or 3D input, got {x_np.ndim}D.")


def nelp_probe(
    x: Any,
    y: Any,
    backend: Backend = "auto",
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> LinearProbe | ProbeCollection:
    """
    Fit a NELP (Non-linear Embedding Linear Probe) using ElasticNet.
    """
    if isinstance(x, list):
        x = np.array(x)

    resolved_backend = _resolve_backend(x, backend)
    if resolved_backend == "torch":
        target_device = _resolve_torch_device(x, device)
        x_t = _ensure_tensor(x, device=target_device)
        y_t = _ensure_tensor(y, device=x_t.device)
        torch_kwargs = _torch_kwargs(kwargs, x_t.device)
        torch_kwargs.setdefault("l1_ratios", [0.1, 0.5, 0.7, 0.9])

        if x_t.ndim == 2:
            col = fit_sparse_probe_batch(x_t.unsqueeze(0), y_t, **torch_kwargs)
            return col[0]
        if x_t.ndim == 3:
            return fit_sparse_probe_batch(x_t, y_t, **torch_kwargs)
        raise ValueError(f"Expected 2D or 3D input for torch backend, got {x_t.ndim}D.")

    x_np = _ensure_numpy(x)
    y_np = _ensure_numpy(y)
    sklearn_kwargs = _sklearn_kwargs(kwargs)
    sklearn_kwargs.setdefault("l1_ratios", [0.1, 0.5, 0.7, 0.9])

    if x_np.ndim == 2:
        return fit_elastic_net(x_np, y_np, **sklearn_kwargs)
    if x_np.ndim == 3:
        return _fit_elastic_sklearn_batch(x_np, y_np, **sklearn_kwargs)
    raise ValueError(f"Expected 2D or 3D input, got {x_np.ndim}D.")


def dim_probe(
    x: Any,
    y: Any,
    backend: Backend = "auto",
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> LinearProbe | ProbeCollection:
    """
    Difference-in-Means probe.
    """
    if isinstance(x, list):
        x = np.array(x)

    resolved_backend = _resolve_backend(x, backend)
    if resolved_backend == "torch":
        target_device = _resolve_torch_device(x, device)
        x_t = _ensure_tensor(x, device=target_device)
        y_t = _ensure_tensor(y, device=x_t.device)

        if x_t.ndim == 2:
            col = fit_dim_batch(x_t.unsqueeze(0), y_t, **kwargs)
            return col[0]
        if x_t.ndim == 3:
            return fit_dim_batch(x_t, y_t, **kwargs)
        raise ValueError(f"Expected 2D or 3D input for torch backend, got {x_t.ndim}D.")

    x_np = _ensure_numpy(x)
    y_np = _ensure_numpy(y)

    if x_np.ndim == 2:
        return fit_dim(x_np, y_np, **kwargs)
    if x_np.ndim == 3:
        return _fit_dim_sklearn_batch(x_np, y_np, **kwargs)
    raise ValueError(f"Expected 2D or 3D input, got {x_np.ndim}D.")
