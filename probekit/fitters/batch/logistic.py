import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection
from probekit.core.probe import LinearProbe, NormalizationStats

from .normalize import fit_normalization


def fit_logistic_batch(
    x: Tensor,
    y: Tensor,
    c_param: float = 1.0,
    max_iter: int = 15,
    tol: float = 1e-6,
    normalize: bool = True,
    val_x: Tensor | None = None,
    val_y: Tensor | None = None,
) -> ProbeCollection:
    """
    x: [b, n, d]
    y: [n] or [b, n]
    C: inverse regularization strength (higher = less regularization)
    Returns: ProbeCollection
    """
    device = x.device
    if x.ndim != 3:
        raise ValueError(f"x must be 3D [b, n, d], got {x.shape}")

    n_batch, n, d = x.shape

    # Broadcasting y
    if y.ndim == 1:
        y = y.unsqueeze(0).expand(n_batch, n)

    # Normalization
    if normalize:
        mu, sigma = fit_normalization(x)
        # Apply normalization: (x - mu) / sigma
        # x is [b, n, d], mu, sigma are [b, d]
        x_norm = (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)

        # Validation normalization
        if val_x is not None:
            val_x_norm = (val_x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    else:
        x_norm = x
        if val_x is not None:
            val_x_norm = val_x
        # Dummies
        mu = torch.zeros(n_batch, d, device=device)
        sigma = torch.ones(n_batch, d, device=device)

    # Add bias term to x: concate 1s -> [b, n, d+1]
    ones = torch.ones(n_batch, n, 1, device=device, dtype=x.dtype)
    x_aug = torch.cat([x_norm, ones], dim=2)  # [b, n, d+1]

    # Initialize weights: [b, d+1]
    w = torch.zeros(n_batch, d + 1, device=device, dtype=x.dtype)

    # Regularization lambda = 1/c_param
    # We apply it to weights but NOT bias (last dim)
    # L2 penalty = 0.5 * lambda * ||w||^2
    # Gradient = lambda * w
    # Hessian = lambda * I
    lambda_reg = 1.0 / c_param

    # Identity matrix for regularization [b, d+1, d+1]
    # Diag is 1 for weights, 0 for bias
    i_reg = torch.eye(d + 1, device=device).unsqueeze(0).expand(n_batch, -1, -1).clone()
    i_reg[:, d, d] = 0.0  # No regularization on bias

    # Optimization Loop (IRLS)
    for _i in range(max_iter):
        w_prev = w.clone()

        # 1. Predictions
        # logits = x @ w: [b, n, d+1] @ [b, d+1, 1] -> [b, n, 1]
        logits = torch.bmm(x_aug, w.unsqueeze(-1)).squeeze(-1)  # [b, n]
        p = torch.sigmoid(logits)

        # 2. Weights r = p * (1-p)
        # Clamp for stability
        p_clamped = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        r = p_clamped * (1.0 - p_clamped)  # [B, N]

        # 3. Working response z
        # z = Xw + (y - p) / r
        # effectively: update target for weighted least squares
        # z = logits + (y - p_clamped) / r # [B, N]

        # 4. Weighted Least Squares: (x^T R x + lambda i) w = x^T R z
        # We need to construct h = x^T R x + lambda i per batch

        # R is diagonal [b, n, n]. Too big to construct explicitly?
        # x^T R x = sum_n r_n * x_n * x_n^T
        # Efficient computation:
        # Scale x by sqrt(r): x_scaled = x * sqrt(r)
        # Then x^T R x = x_scaled^T x_scaled

        sqrt_r = torch.sqrt(r).unsqueeze(-1)  # [b, n, 1]
        x_scaled = x_aug * sqrt_r  # [b, n, d+1]

        h = torch.bmm(x_scaled.transpose(1, 2), x_scaled)  # [b, d+1, d+1]

        # Add regularization
        h = h + lambda_reg * i_reg

        # RHS: g = X^T R z = X^T (r * z)
        # r * z = r * (logits + (y-p)/r) = r*logits + y - p
        # Actually standard IRLS formulation often simpler:
        # Update step: w_new = w + (H)^-1 * grad
        # grad = X^T (y - p) - lambda * w
        # Hessian is H.
        # Newton step: delta = H^-1 * grad
        # w_new = w + delta
        # Let's use this update form, it's numerically cleaner?
        # Or proper solve: (X^T R X + lambda I) w_new = X^T R z
        # = X^T (R * (Xw + (y-p)/r)) = X^T R X w + X^T (y-p)
        # So H w_new = H w + X^T (y - p) - lambda * I * w
        # => H (w_new - w) = X^T (y - p) - lambda * w[reg]
        # => delta = H^-1 * (gradient - reg_grad)

        # Gradient of LogLikelihood(w) = X^T (y - p)
        # Gradient of Penalty = lambda * w (0 for bias)
        # Total Gradient = X^T (y - p) - lambda * w_reg

        residual = y - p  # [b, n]
        grad_data = torch.bmm(x_aug.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)  # [b, d+1]

        w_reg = w.clone()
        w_reg[:, d] = 0.0  # Don't penalize bias
        grad_total = grad_data - lambda_reg * w_reg

        # Solve h * delta = grad_total
        # Use linalg.solve (or cholesky_solve if h is PD, which it should be)
        # Add small jitter to diagonal for stability?
        # h is [b, d+1, d+1]
        h_damped = h + 1e-6 * torch.eye(d + 1, device=device).unsqueeze(0)

        try:
            delta = torch.linalg.solve(h_damped, grad_total)
        except RuntimeError:
            # Fallback or break?
            # Could use lstsq
            delta = torch.linalg.lstsq(h_damped, grad_total).solution

        w = w + delta  # Step size 1.0 (Newton)

        # Check convergence
        change = torch.max(torch.abs(w - w_prev))
        if change < tol:
            break

    # Pack results
    # w: [b, d+1] -> split to weights [b, d] and bias [b]
    weights = w[:, :d]
    bias = w[:, d]

    probekit = []

    # Validation accuracy calculation if needed
    val_accs = None
    if val_x is not None and val_y is not None:
        # val_x is [b, n_val, d], val_y is [b, n_val] maybe?
        # Or val_y is [n_val] broadcast?
        if val_y.ndim == 1:
            val_y = val_y.unsqueeze(0).expand(n_batch, -1)

        logits_val = torch.bmm(val_x_norm, weights.unsqueeze(-1)).squeeze(-1) + bias.unsqueeze(1)
        preds_val = (logits_val > 0).float()
        accs = (preds_val == val_y).float().mean(dim=1)  # [b]
        val_accs = accs.cpu().numpy()

    mu_cpu = mu.cpu().numpy()
    sigma_cpu = sigma.cpu().numpy()
    weights_cpu = weights.cpu().numpy()
    bias_cpu = bias.cpu().numpy()

    for i in range(n_batch):
        meta = {"fit_method": "logistic_batch", "iterations": i}
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
