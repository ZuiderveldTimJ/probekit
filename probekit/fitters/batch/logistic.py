from typing import Any

import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection
from probekit.core.probe import NormalizationStats

from .normalize import fit_normalization


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


def fit_logistic_batch(
    x: Tensor,
    y: Tensor,
    c_param: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    normalize: bool = True,
    val_x: Tensor | None = None,
    val_y: Tensor | None = None,
) -> ProbeCollection:
    """
    Fit batched L2-regularized logistic regression with damped IRLS/Newton updates.

    x: [B, N, D]
    y: [N] or [B, N]
    """
    if x.ndim != 3:
        raise ValueError(f"x must be 3D [B, N, D], got {tuple(x.shape)}")
    if c_param <= 0:
        raise ValueError(f"c_param must be > 0, got {c_param}.")

    x = x.to(dtype=torch.float32)
    device = x.device
    n_batch, n, d = x.shape

    y_float = _prepare_labels(y, n_batch=n_batch, n=n, device=device, dtype=x.dtype)

    if normalize:
        mu, sigma = fit_normalization(x)
        x_norm = (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    else:
        mu = torch.zeros((n_batch, d), device=device, dtype=x.dtype)
        sigma = torch.ones((n_batch, d), device=device, dtype=x.dtype)
        x_norm = x

    x_aug = torch.cat([x_norm, torch.ones((n_batch, n, 1), device=device, dtype=x.dtype)], dim=2)
    d_aug = d + 1
    # Match sklearn's objective scaling where the data term is averaged over samples.
    lambda_reg = 1.0 / (c_param * n)
    reg_diag = torch.cat(
        [
            torch.full((n_batch, d), fill_value=lambda_reg, device=device, dtype=x.dtype),
            torch.zeros((n_batch, 1), device=device, dtype=x.dtype),
        ],
        dim=1,
    )

    theta = torch.zeros((n_batch, d_aug), device=device, dtype=x.dtype)
    class_prob = y_float.mean(dim=1).clamp(1e-4, 1.0 - 1e-4)
    theta[:, d] = torch.log(class_prob / (1.0 - class_prob))

    eye = torch.eye(d_aug, device=device, dtype=x.dtype).unsqueeze(0).expand(n_batch, -1, -1)
    converged = torch.zeros(n_batch, dtype=torch.bool, device=device)
    step_inf = torch.full((n_batch,), float("inf"), device=device, dtype=x.dtype)
    n_iter = max_iter

    for i in range(max_iter):
        logits = torch.bmm(x_aug, theta.unsqueeze(-1)).squeeze(-1)
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        err = probs - y_float

        grad = torch.bmm(x_aug.transpose(1, 2), err.unsqueeze(-1)).squeeze(-1) / n
        grad = grad + reg_diag * theta

        curvature = (probs * (1.0 - probs)).clamp_min(1e-6)
        x_weighted = x_aug * torch.sqrt(curvature / n).unsqueeze(-1)
        hessian = torch.bmm(x_weighted.transpose(1, 2), x_weighted) + torch.diag_embed(reg_diag)
        hessian = hessian + 1e-6 * eye

        delta, info = torch.linalg.solve_ex(hessian, grad.unsqueeze(-1), check_errors=False)
        if torch.any(info != 0):
            hessian = hessian + 1e-3 * eye
            delta = torch.linalg.solve(hessian, grad.unsqueeze(-1))

        delta = delta.squeeze(-1)
        theta = theta - delta
        step_inf = torch.amax(torch.abs(delta), dim=1)
        converged = converged | (step_inf < tol)
        n_iter = i + 1

        if torch.all(step_inf < tol):
            break

    weights = theta[:, :d]
    bias = theta[:, d]

    val_accs = None
    if val_x is not None and val_y is not None:
        val_x_t = _prepare_val_x(val_x, n_batch=n_batch, d=d, device=device, dtype=x.dtype)
        if normalize:
            val_x_t = (val_x_t - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        val_y_t = _prepare_labels(val_y, n_batch=n_batch, n=val_x_t.shape[1], device=device, dtype=x.dtype)

        logits_val = torch.bmm(val_x_t, weights.unsqueeze(-1)).squeeze(-1) + bias.unsqueeze(1)
        preds_val = (logits_val > 0).to(dtype=val_y_t.dtype)
        val_accs = (preds_val == val_y_t).to(dtype=torch.float32).mean(dim=1).cpu().numpy()

    mu_cpu = mu.cpu().numpy()
    sigma_cpu = sigma.cpu().numpy()

    normalizations = [
        NormalizationStats(mean=mu_cpu[i], std=sigma_cpu[i], count=n) if normalize else None for i in range(n_batch)
    ]

    metadatas: list[dict[str, Any]] = []
    step_cpu = step_inf.detach().cpu().numpy()
    converged_cpu = converged.detach().cpu().numpy()
    for i in range(n_batch):
        meta: dict[str, Any] = {
            "fit_method": "logistic_batch_irls",
            "iterations": n_iter,
            "converged": bool(converged_cpu[i]),
            "final_step_inf": float(step_cpu[i]),
            "C": c_param,
        }
        if val_accs is not None:
            meta["val_accuracy"] = float(val_accs[i])
        metadatas.append(meta)

    return ProbeCollection(
        weights=weights,
        biases=bias,
        normalizations=normalizations,
        metadatas=metadatas,
    )
