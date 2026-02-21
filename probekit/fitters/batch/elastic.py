from typing import Any

import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection
from probekit.core.probe import NormalizationStats

from .normalize import fit_normalization


def soft_threshold(x: Tensor, lambd: Tensor | float) -> Tensor:
    """Soft-thresholding operator: sign(x) * max(|x| - lambda, 0)."""
    return torch.sign(x) * torch.clamp(torch.abs(x) - lambd, min=0.0)


def _prepare_labels(y: Tensor, n_batch: int, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if y.ndim == 1:
        if y.shape[0] != n:
            raise ValueError(f"Expected y shape [{n}] for 1D labels, got {tuple(y.shape)}.")
        return y.unsqueeze(0).expand(n_batch, n).to(device=device, dtype=dtype)

    if y.ndim == 2:
        if y.shape == (n_batch, n):
            return y.to(device=device, dtype=dtype)
        if y.shape == (1, n):
            return y.expand(n_batch, n).to(device=device, dtype=dtype)
        raise ValueError(f"Expected y shape [{n_batch}, {n}] or [1, {n}], got {tuple(y.shape)}.")

    raise ValueError(f"Expected y with 1 or 2 dims, got {y.ndim}.")


def _prepare_val_x(val_x: Tensor, n_batch: int, d: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if val_x.ndim == 2:
        if val_x.shape[1] != d:
            raise ValueError(f"Expected val_x feature dim {d}, got {val_x.shape[1]}.")
        if n_batch != 1:
            raise ValueError("2D val_x is only valid when fitting a single batched probe (batch size 1).")
        return val_x.unsqueeze(0).to(device=device, dtype=dtype)

    if val_x.ndim == 3:
        if val_x.shape[0] != n_batch or val_x.shape[2] != d:
            raise ValueError(f"Expected val_x shape [{n_batch}, N, {d}], got {tuple(val_x.shape)}.")
        return val_x.to(device=device, dtype=dtype)

    raise ValueError(f"Expected val_x with 2 or 3 dims, got {val_x.ndim}.")


def _estimate_lipschitz_constant(x: Tensor, n_power_iters: int = 10) -> Tensor:
    """
    Estimate logistic-gradient Lipschitz constants per batch element.
    L <= 0.25 * lambda_max(X^T X) / n
    """
    n_batch, n, d = x.shape
    v = torch.ones((n_batch, d), device=x.device, dtype=x.dtype)
    v = v / torch.norm(v, dim=1, keepdim=True).clamp_min(1e-12)

    for _ in range(n_power_iters):
        xv = torch.bmm(x, v.unsqueeze(-1)).squeeze(-1)
        xtxv = torch.bmm(x.transpose(1, 2), xv.unsqueeze(-1)).squeeze(-1)
        v = xtxv / torch.norm(xtxv, dim=1, keepdim=True).clamp_min(1e-12)

    xv = torch.bmm(x, v.unsqueeze(-1)).squeeze(-1)
    lambda_max = torch.sum(xv * xv, dim=1).clamp_min(1e-12)
    return 0.25 * lambda_max / n


def fit_elastic_net_batch(
    x: Tensor,
    y: Tensor,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 500,
    tol: float = 1e-6,
    normalize: bool = True,
    val_x: Tensor | None = None,
    val_y: Tensor | None = None,
    w_init: Tensor | None = None,
    b_init: Tensor | None = None,
    positive: bool = False,
) -> ProbeCollection:
    """
    Fit batched elastic-net logistic probes with FISTA acceleration.

    x: [B, N, D]
    y: [N] or [B, N]
    """
    if x.ndim != 3:
        raise ValueError(f"x must be 3D [B, N, D], got {tuple(x.shape)}.")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}.")
    if not (0.0 <= l1_ratio <= 1.0):
        raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")

    x = x.to(dtype=torch.float32)
    device = x.device
    n_batch, n, d = x.shape
    y_float = _prepare_labels(y, n_batch=n_batch, n=n, device=device, dtype=x.dtype)

    if normalize:
        mu, sigma = fit_normalization(x)
        x_norm = (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    else:
        x_norm = x
        mu = torch.zeros((n_batch, d), device=device, dtype=x.dtype)
        sigma = torch.ones((n_batch, d), device=device, dtype=x.dtype)

    l1_strength = alpha * l1_ratio
    l2_strength = alpha * (1.0 - l1_ratio)

    if w_init is not None:
        if w_init.shape != (n_batch, d):
            raise ValueError(f"Expected w_init shape {(n_batch, d)}, got {tuple(w_init.shape)}.")
        w = w_init.detach().to(device=device, dtype=x.dtype, copy=True)
    else:
        w = torch.zeros((n_batch, d), device=device, dtype=x.dtype)

    if b_init is not None:
        if b_init.shape != (n_batch,):
            raise ValueError(f"Expected b_init shape {(n_batch,)}, got {tuple(b_init.shape)}.")
        b = b_init.detach().to(device=device, dtype=x.dtype, copy=True)
    else:
        b = torch.zeros((n_batch,), device=device, dtype=x.dtype)

    z_w = w.clone()
    z_b = b.clone()
    t_k = torch.ones((n_batch,), device=device, dtype=x.dtype)

    lipschitz_l = _estimate_lipschitz_constant(x_norm)
    step_size_scalar = 1.0 / (lipschitz_l + l2_strength + 1e-6)
    step_size = step_size_scalar.unsqueeze(1)

    converged = torch.zeros(n_batch, dtype=torch.bool, device=device)
    update_inf = torch.full((n_batch,), float("inf"), device=device, dtype=x.dtype)
    n_iter = max_iter

    for i in range(max_iter):
        logits = torch.bmm(x_norm, z_w.unsqueeze(-1)).squeeze(-1) + z_b.unsqueeze(1)
        probs = torch.sigmoid(logits)
        err = probs - y_float

        grad_w = torch.bmm(x_norm.transpose(1, 2), err.unsqueeze(-1)).squeeze(-1) / n
        grad_w += l2_strength * z_w
        grad_b = err.mean(dim=1)

        w_candidate = z_w - step_size * grad_w
        b_next = z_b - step_size_scalar * grad_b
        w_next = soft_threshold(w_candidate, step_size * l1_strength)
        if positive:
            w_next = torch.clamp(w_next, min=0.0)

        delta_w = w_next - w
        delta_b = torch.abs(b_next - b)
        update_inf = torch.maximum(torch.amax(torch.abs(delta_w), dim=1), delta_b)
        converged = converged | (update_inf < tol)

        t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t_k * t_k))
        momentum = ((t_k - 1.0) / t_next).unsqueeze(1)

        z_w = w_next + momentum * delta_w
        z_b = b_next + ((t_k - 1.0) / t_next) * (b_next - b)

        w = w_next
        b = b_next
        t_k = t_next
        n_iter = i + 1

        if torch.all(update_inf < tol):
            break

    val_accs = None
    if val_x is not None and val_y is not None:
        val_x_t = _prepare_val_x(val_x, n_batch=n_batch, d=d, device=device, dtype=x.dtype)
        if normalize:
            val_x_t = (val_x_t - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        val_y_t = _prepare_labels(val_y, n_batch=n_batch, n=val_x_t.shape[1], device=device, dtype=x.dtype)

        logits_val = torch.bmm(val_x_t, w.unsqueeze(-1)).squeeze(-1) + b.unsqueeze(1)
        preds_val = (logits_val > 0).to(dtype=val_y_t.dtype)
        val_accs = (preds_val == val_y_t).to(dtype=torch.float32).mean(dim=1).cpu().numpy()

    mu_cpu = mu.cpu().numpy()
    sigma_cpu = sigma.cpu().numpy()
    normalizations = [
        NormalizationStats(mean=mu_cpu[i], std=sigma_cpu[i], count=n) if normalize else None for i in range(n_batch)
    ]

    metadatas: list[dict[str, Any]] = []
    converged_cpu = converged.detach().cpu().numpy()
    update_cpu = update_inf.detach().cpu().numpy()
    lipschitz_cpu = lipschitz_l.detach().cpu().numpy()
    for i in range(n_batch):
        meta: dict[str, Any] = {
            "fit_method": "elastic_batch_fista",
            "iterations": n_iter,
            "converged": bool(converged_cpu[i]),
            "final_update_inf": float(update_cpu[i]),
            "lipschitz_l": float(lipschitz_cpu[i]),
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "positive": positive,
        }
        if val_accs is not None:
            meta["val_accuracy"] = float(val_accs[i])
        metadatas.append(meta)

    return ProbeCollection(
        weights=w,
        biases=b,
        normalizations=normalizations,
        metadatas=metadatas,
    )
