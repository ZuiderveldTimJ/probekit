from typing import Any

import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection
from probekit.core.probe import NormalizationStats

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
        x_norm = (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)

        if val_x is not None:
            val_x_norm = (val_x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    else:
        x_norm = x
        if val_x is not None:
            val_x_norm = val_x
        mu = torch.zeros(n_batch, d, device=device)
        sigma = torch.ones(n_batch, d, device=device)

    # Add bias term to x: cat 1s -> [b, n, d+1]
    ones = torch.ones(n_batch, n, 1, device=device, dtype=x.dtype)
    x_aug = torch.cat([x_norm, ones], dim=2)  # [b, n, d+1]

    # Initialize weights: [b, d+1]
    w = torch.zeros(n_batch, d + 1, device=device, dtype=x.dtype, requires_grad=True)

    # Regularization lambda = 1/c_param
    lambda_reg = 1.0 / c_param

    # Objective logic using Adam avoids materializing [B, D+1, D+1] Hessians
    # PyTorch's LBFGS has a known bug causing FPE on CUDA for some batched losses.
    # Adam converges efficiently without second-order [D, D] memory allocation.
    optimizer = torch.optim.Adam([w], lr=0.1)

    # Convert y to float for BCE with logits
    y_float = y.float()

    # Run optimizer for multiple iterations (Adam takes more steps than LBFGS but each is cheaper)
    adam_iters = max(max_iter, 100)
    for _ in range(adam_iters):
        optimizer.zero_grad()
        # [b, n, d+1] @ [b, d+1, 1] -> [b, n]
        logits = torch.bmm(x_aug, w.unsqueeze(-1)).squeeze(-1)

        # BCE loss mean over sequence length [b]
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_float, reduction="none").mean(dim=1)

        # L2 penalty: only penalize weights, not bias (last dim)
        # Sum over features: [b]
        l2_loss = 0.5 * lambda_reg * torch.sum(w[:, :d] ** 2, dim=1)

        # Summing across the batch for independent gradients
        loss = (bce_loss + l2_loss).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    with torch.no_grad():
        # w: [b, d+1] -> split to weights [b, d] and bias [b]
        weights = w[:, :d]
        bias = w[:, d]

        # Validation accuracy calculation
        val_accs = None
        if val_x is not None and val_y is not None:
            if val_y.ndim == 1:
                val_y = val_y.unsqueeze(0).expand(n_batch, -1)

            logits_val = torch.bmm(val_x_norm, weights.unsqueeze(-1)).squeeze(-1) + bias.unsqueeze(1)
            preds_val = (logits_val > 0).float()
            val_accs = (preds_val == val_y).float().mean(dim=1).cpu().numpy()

        mu_cpu = mu.cpu().numpy()
        sigma_cpu = sigma.cpu().numpy()
        weights_cpu = weights.cpu().numpy()
        bias_cpu = bias.cpu().numpy()

        normalizations = [
            NormalizationStats(mean=mu_cpu[i], std=sigma_cpu[i], count=n) if normalize else None for i in range(n_batch)
        ]

        metadatas = []
        for i in range(n_batch):
            meta: dict[str, Any] = {"fit_method": "logistic_batch_lbfgs"}
            if val_accs is not None:
                meta["val_accuracy"] = float(val_accs[i])
            metadatas.append(meta)

        return ProbeCollection(
            weights=weights_cpu,
            biases=bias_cpu,
            normalizations=normalizations,
            metadatas=metadatas,
        )
