import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection
from probekit.core.probe import LinearProbe, NormalizationStats

from .normalize import fit_normalization


def soft_threshold(x: Tensor, lambd: Tensor | float) -> Tensor:
    """
    Soft thresholding operator: sign(x) * max(|x| - lambda, 0)
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - lambd, min=0.0)


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
) -> ProbeCollection:
    """
    x: [b, n, d]
    y: [n] or [b, n]
    alpha: overall regularization strength
    l1_ratio: mix between L1 (1.0) and L2 (0.0)
    w_init: [b, d] optional initial weights
    b_init: [b] optional initial bias
    Returns: list of b LinearProbe objects

    Algorithm: Proximal Gradient Descent (ISTA/FISTA)
    """
    device = x.device
    n_batch, n, d = x.shape

    if y.ndim == 1:
        y = y.unsqueeze(0).expand(n_batch, n)

    # Normalization
    if normalize:
        mu, sigma = fit_normalization(x)  # [b, d]
        # x: [b, n, d]
        x_norm = (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        if val_x is not None:
            val_x_norm = (val_x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    else:
        x_norm = x
        mu = torch.zeros(n_batch, d, device=device)
        sigma = torch.ones(n_batch, d, device=device)
        if val_x is not None:
            val_x_norm = val_x

    # Regularization parameters
    # L1 strength = alpha * l1_ratio
    # L2 strength = alpha * (1 - l1_ratio)
    l1_strength = alpha * l1_ratio
    l2_strength = alpha * (1.0 - l1_ratio)

    # Initialize parameters
    if w_init is not None:
        if w_init.shape != (n_batch, d):
            # Handle shape mismatch if needed or assume caller handles it
            # Warm start usually implies same shape
            pass
        w = w_init.clone().to(device=device, dtype=x.dtype)
    else:
        w = torch.zeros(n_batch, d, device=device, dtype=x.dtype)

    if b_init is not None:
        b = b_init.clone().to(device=device, dtype=x.dtype)
    else:
        b = torch.zeros(n_batch, device=device, dtype=x.dtype)

    # Current step size estimation
    # Lipschitz constant lipschitz_l <= ||x||^2 / 4n (for logistic)
    # We can compute max eigenvalue of x^T x per batch? Or just use backtracking / heuristic.
    # Or simpler: fixed step size = 1 / (lipschitz_l_max + l2_strength)
    # Heuristic: lipschitz_l approx max(norm(row)^2) ?
    # For logistic regression: H <= 0.25 * x^T x.
    # Let's approximate lipschitz_l conservatively.
    # lipschitz_l = 0.25 * (max eigenvalue of x^T x)
    # Using 1/lipschitz_l might be slow if lipschitz_l is loose.
    # Let's use simple backtracking or just fixed small step if stable?
    # The prompt suggested: "step_size = ... 1/(lipschitz_l + l2_strength) ... lipschitz_l = ||x||^2 / (4n)"

    # Compute lipschitz_l per batch
    # ||x||^2 is Frobenius norm squared? No, spectral norm squared (max singular value squared).
    # Upper bound for spectral norm is Frobenius norm.
    # ||x||_f_sq = sum(x**2)
    # This is a safe upper bound.
    # lipschitz_l <= ||x||_f_sq / (4n) ?
    # Actually lipschitz_l for logistic loss is 0.25 * lambda_max(x^T x).
    # lambda_max(x^T x) <= Trace(x^T x) = ||x||_f_sq.
    # So lipschitz_l <= 0.25 * ||x||_f_sq / n (if 1/n factor in loss, which we used in gradient?)
    # Wait, gradient is usually 1/n * x^T (y-p). Yes.

    x_frob_sq = torch.sum(x_norm**2, dim=(1, 2))  # [n_batch]
    lipschitz_l = x_frob_sq / (4.0 * n)
    step_size = 1.0 / (lipschitz_l + l2_strength + 1e-6)  # [n_batch]
    step_size = step_size.unsqueeze(1)  # [n_batch, 1] for broadcasting to w

    # Optimization Loop
    for _i in range(max_iter):
        w_prev = w.clone()

        # 1. Forward
        # [b, n, d] @ [b, d, 1] -> [b, n]
        logits = torch.bmm(x_norm, w.unsqueeze(-1)).squeeze(-1) + b.unsqueeze(1)
        p = torch.sigmoid(logits)

        # 2. Gradients
        # Loss = -1/n * sum (y log p + ...) + Reg
        # grad_w = 1/n * x^T (p - y) + l2 * w
        # grad_b = 1/n * sum (p - y)

        err = p - y  # [b, n]
        grad_w = torch.bmm(x_norm.transpose(1, 2), err.unsqueeze(-1)).squeeze(-1) / n  # [b, d]
        grad_w += l2_strength * w

        grad_b = err.mean(dim=1)  # [b]

        # 3. Update
        # w_candidate = w - step * grad_w
        w_candidate = w - step_size * grad_w
        b = b - step_size.squeeze(1) * grad_b

        # 4. Proximal operator (Soft Thresholding) for L1
        # threshold = step * l1_strength
        threshold = step_size * l1_strength
        w = soft_threshold(w_candidate, threshold)

        # Convergence check
        change = torch.max(torch.abs(w - w_prev))
        if change < tol:
            break

    # Pack results
    probekit = []

    # Validation accuracy
    val_accs = None
    if val_x is not None and val_y is not None:
        if val_y.ndim == 1:
            val_y = val_y.unsqueeze(0).expand(n_batch, -1)
        logits_val = torch.bmm(val_x_norm, w.unsqueeze(-1)).squeeze(-1) + b.unsqueeze(1)
        preds_val = (logits_val > 0).float()
        accs = (preds_val == val_y).float().mean(dim=1)
        val_accs = accs.cpu().numpy()

    mu_cpu = mu.cpu().numpy()
    sigma_cpu = sigma.cpu().numpy()
    weights_cpu = w.cpu().numpy()
    bias_cpu = b.cpu().numpy()

    for i in range(n_batch):
        meta = {"fit_method": "elastic_batch", "iterations": i}
        if val_accs is not None:
            meta["val_accuracy"] = val_accs[i]

        probekit.append(
            LinearProbe(
                weights=weights_cpu[i],
                bias=bias_cpu[i].item(),
                normalization=NormalizationStats(mean=mu_cpu[i], std=sigma_cpu[i], count=n) if normalize else None,
                metadata=meta,
            )
        )

    return ProbeCollection(probekit)
