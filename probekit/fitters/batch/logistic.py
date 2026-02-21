from typing import Any, Literal

import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection
from probekit.core.probe import NormalizationStats

from .normalize import fit_normalization

NewtonSolver = Literal["auto", "newton-cg", "newton-cholesky"]


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


def _hessian_vector_product(
    x_aug: Tensor,
    x_aug_t: Tensor,
    curvature: Tensor,
    reg_diag: Tensor,
    v: Tensor,
    n: int,
    damping: Tensor,
) -> Tensor:
    xv = torch.bmm(x_aug, v.unsqueeze(-1)).squeeze(-1)
    weighted_xv = (curvature * xv) / n
    hv = torch.bmm(x_aug_t, weighted_xv.unsqueeze(-1)).squeeze(-1)
    hv = hv + reg_diag * v + damping.unsqueeze(1) * v
    return hv


def _batched_conjugate_gradient(
    x_aug: Tensor,
    x_aug_t: Tensor,
    curvature: Tensor,
    reg_diag: Tensor,
    rhs: Tensor,
    n: int,
    damping: Tensor,
    max_iter: int,
    tol: float,
) -> tuple[Tensor, Tensor]:
    """
    Solve A x = rhs with batched conjugate gradient using Hessian-vector products.
    """
    x_sol = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()

    rs_old = torch.sum(r * r, dim=1)
    residual_norm = torch.sqrt(rs_old.clamp_min(0.0))

    for _ in range(max_iter):
        if torch.all(residual_norm <= tol):
            break

        ap = _hessian_vector_product(x_aug, x_aug_t, curvature, reg_diag, p, n=n, damping=damping)
        denom = torch.sum(p * ap, dim=1)
        safe_denom = torch.where(denom > 1e-12, denom, torch.full_like(denom, 1e-12))

        alpha = rs_old / safe_denom
        x_sol = x_sol + alpha.unsqueeze(1) * p
        r = r - alpha.unsqueeze(1) * ap

        rs_new = torch.sum(r * r, dim=1)
        residual_norm = torch.sqrt(rs_new.clamp_min(0.0))

        beta = rs_new / rs_old.clamp_min(1e-12)
        p = r + beta.unsqueeze(1) * p
        rs_old = rs_new

    return x_sol, residual_norm


def _solve_with_damped_cg(
    x_aug: Tensor,
    x_aug_t: Tensor,
    curvature: Tensor,
    reg_diag: Tensor,
    grad: Tensor,
    grad_norm: Tensor,
    n: int,
    cg_max_iter: int,
    cg_tol: float,
    damping_init: float,
    max_damping_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    damping = torch.full((grad.shape[0],), fill_value=damping_init, device=grad.device, dtype=grad.dtype)

    best_delta = torch.zeros_like(grad)
    best_residual = torch.full_like(grad_norm, float("inf"))
    best_damping = damping.clone()
    best_finite = torch.zeros_like(grad_norm, dtype=torch.bool)
    target = cg_tol * (1.0 + grad_norm)

    for _ in range(max_damping_steps):
        trial_delta, trial_residual = _batched_conjugate_gradient(
            x_aug=x_aug,
            x_aug_t=x_aug_t,
            curvature=curvature,
            reg_diag=reg_diag,
            rhs=grad,
            n=n,
            damping=damping,
            max_iter=cg_max_iter,
            tol=cg_tol,
        )

        finite = torch.isfinite(trial_delta).all(dim=1) & torch.isfinite(trial_residual)
        better = finite & (trial_residual < best_residual)
        best_delta = torch.where(better.unsqueeze(1), trial_delta, best_delta)
        best_residual = torch.where(better, trial_residual, best_residual)
        best_damping = torch.where(better, damping, best_damping)
        best_finite = best_finite | finite

        converged = finite & (trial_residual <= target)
        if torch.all(converged):
            return trial_delta, trial_residual, damping

        damping = torch.where(converged, damping, damping * 10.0)

    if not torch.all(best_finite):
        raise RuntimeError("Newton-CG failed to produce finite updates for one or more batch elements.")

    return best_delta, best_residual, best_damping


def fit_logistic_batch(
    x: Tensor,
    y: Tensor,
    c_param: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    normalize: bool = True,
    val_x: Tensor | None = None,
    val_y: Tensor | None = None,
    solver: NewtonSolver = "auto",
    cg_max_iter: int | None = None,
    cg_tol: float = 1e-4,
    damping_init: float = 1e-6,
    max_damping_steps: int = 6,
    max_dense_hessian_mb: float = 64.0,
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
    if solver not in {"auto", "newton-cg", "newton-cholesky"}:
        raise ValueError(f"Invalid solver {solver!r}. Expected 'auto', 'newton-cg', or 'newton-cholesky'.")
    if cg_tol <= 0:
        raise ValueError(f"cg_tol must be > 0, got {cg_tol}.")
    if damping_init <= 0:
        raise ValueError(f"damping_init must be > 0, got {damping_init}.")
    if max_damping_steps < 1:
        raise ValueError(f"max_damping_steps must be >= 1, got {max_damping_steps}.")
    if max_dense_hessian_mb <= 0:
        raise ValueError(f"max_dense_hessian_mb must be > 0, got {max_dense_hessian_mb}.")

    x = x.to(dtype=torch.float32)
    device = x.device
    n_batch, n, d = x.shape
    d_aug = d + 1

    y_float = _prepare_labels(y, n_batch=n_batch, n=n, device=device, dtype=x.dtype)

    if normalize:
        mu, sigma = fit_normalization(x)
        x_norm = (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    else:
        mu = torch.zeros((n_batch, d), device=device, dtype=x.dtype)
        sigma = torch.ones((n_batch, d), device=device, dtype=x.dtype)
        x_norm = x

    x_aug = torch.cat([x_norm, torch.ones((n_batch, n, 1), device=device, dtype=x.dtype)], dim=2)
    x_aug_t = x_aug.transpose(1, 2)

    # Match sklearn's objective scaling where the data term is averaged over samples.
    lambda_reg = 1.0 / (c_param * n)
    reg_diag = torch.cat(
        [
            torch.full((n_batch, d), fill_value=lambda_reg, device=device, dtype=x.dtype),
            torch.zeros((n_batch, 1), device=device, dtype=x.dtype),
        ],
        dim=1,
    )

    dense_hessian_est_mb = n_batch * d_aug * d_aug * x.element_size() / (1024**2)
    if solver == "auto":
        newton_solver = "newton-cholesky" if dense_hessian_est_mb <= max_dense_hessian_mb else "newton-cg"
    elif solver == "newton-cholesky":
        newton_solver = "newton-cholesky"
    else:
        newton_solver = "newton-cg"

    if cg_max_iter is None:
        cg_max_iter = min(256, max(16, 2 * d_aug))
    if cg_max_iter < 1:
        raise ValueError(f"cg_max_iter must be >= 1, got {cg_max_iter}.")

    theta = torch.zeros((n_batch, d_aug), device=device, dtype=x.dtype)
    class_prob = y_float.mean(dim=1).clamp(1e-4, 1.0 - 1e-4)
    theta[:, d] = torch.log(class_prob / (1.0 - class_prob))

    eye = torch.eye(d_aug, device=device, dtype=x.dtype).unsqueeze(0).expand(n_batch, -1, -1)
    converged = torch.zeros(n_batch, dtype=torch.bool, device=device)
    step_inf = torch.full((n_batch,), float("inf"), device=device, dtype=x.dtype)
    final_damping = torch.full((n_batch,), fill_value=damping_init, device=device, dtype=x.dtype)
    final_cg_residual = torch.full((n_batch,), float("nan"), device=device, dtype=x.dtype)
    used_cg_fallback = False
    n_iter = max_iter

    for i in range(max_iter):
        logits = torch.bmm(x_aug, theta.unsqueeze(-1)).squeeze(-1)
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        err = probs - y_float

        grad = torch.bmm(x_aug_t, err.unsqueeze(-1)).squeeze(-1) / n
        grad = grad + reg_diag * theta

        curvature = (probs * (1.0 - probs)).clamp_min(1e-8)
        grad_norm = torch.linalg.vector_norm(grad, dim=1)

        if newton_solver == "newton-cholesky":
            x_weighted = x_aug * torch.sqrt(curvature / n).unsqueeze(-1)
            hessian_base = torch.bmm(x_weighted.transpose(1, 2), x_weighted) + torch.diag_embed(reg_diag)

            dense_damping = damping_init
            delta = None
            for _ in range(max_damping_steps):
                hessian = hessian_base + dense_damping * eye
                delta_try, info = torch.linalg.solve_ex(hessian, grad.unsqueeze(-1), check_errors=False)
                delta_try = delta_try.squeeze(-1)
                if torch.all(info == 0) and torch.isfinite(delta_try).all():
                    delta = delta_try
                    final_damping.fill_(dense_damping)
                    break
                dense_damping *= 10.0

            if delta is None:
                used_cg_fallback = True
                delta, cg_residual, cg_damping = _solve_with_damped_cg(
                    x_aug=x_aug,
                    x_aug_t=x_aug_t,
                    curvature=curvature,
                    reg_diag=reg_diag,
                    grad=grad,
                    grad_norm=grad_norm,
                    n=n,
                    cg_max_iter=cg_max_iter,
                    cg_tol=cg_tol,
                    damping_init=damping_init,
                    max_damping_steps=max_damping_steps,
                )
                final_cg_residual = cg_residual
                final_damping = cg_damping
        else:
            delta, cg_residual, cg_damping = _solve_with_damped_cg(
                x_aug=x_aug,
                x_aug_t=x_aug_t,
                curvature=curvature,
                reg_diag=reg_diag,
                grad=grad,
                grad_norm=grad_norm,
                n=n,
                cg_max_iter=cg_max_iter,
                cg_tol=cg_tol,
                damping_init=damping_init,
                max_damping_steps=max_damping_steps,
            )
            final_cg_residual = cg_residual
            final_damping = cg_damping

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
    damping_cpu = final_damping.detach().cpu().numpy()
    cg_residual_cpu = final_cg_residual.detach().cpu().numpy()
    for i in range(n_batch):
        solver_name = newton_solver
        if used_cg_fallback and newton_solver == "newton-cholesky":
            solver_name = f"{solver_name}+cg-fallback"
        meta: dict[str, Any] = {
            "fit_method": "logistic_batch_irls",
            "newton_solver": solver_name,
            "iterations": n_iter,
            "converged": bool(converged_cpu[i]),
            "final_step_inf": float(step_cpu[i]),
            "final_damping": float(damping_cpu[i]),
            "C": c_param,
            "dense_hessian_est_mb": float(dense_hessian_est_mb),
        }
        if torch.isfinite(final_cg_residual[i]):
            meta["final_cg_residual"] = float(cg_residual_cpu[i])
        if val_accs is not None:
            meta["val_accuracy"] = float(val_accs[i])
        metadatas.append(meta)

    return ProbeCollection(
        weights=weights,
        biases=bias,
        normalizations=normalizations,
        metadatas=metadatas,
    )
