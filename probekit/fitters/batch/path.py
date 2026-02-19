import numpy as np
import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection

from .elastic import fit_elastic_net_batch
from .normalize import fit_normalization


def fit_elastic_net_path(
    x: Tensor,
    y: Tensor,
    alphas: list[float] | None = None,
    n_alphas: int = 20,
    l1_ratio: float = 0.5,
    normalize: bool = True,
    val_x: Tensor | None = None,
    val_y: Tensor | None = None,
    select: str = "best_val",  # 'best_val' or 'all'
) -> ProbeCollection | list[ProbeCollection]:
    """
    Fits elastic net probekit across a path of alpha values with warm starting.
    """
    n_batch, n, _ = x.shape
    device = x.device

    if y.ndim == 1:
        y = y.unsqueeze(0).expand(n_batch, n)

    # Generate alphas if needed
    if alphas is None:
        # alpha_max = max |X^T (y - y_mean)| / (N * l1_ratio)
        # We need a rough alpha_max.
        # For logistic regression, gradient at 0 is X^T (y - 0.5).
        # max component of this gradient sets the scale.

        # Center y
        y_float = y.float()
        y_mean = y_float.mean(dim=1, keepdim=True)
        # Normalized x if normalize=True
        if normalize:
            mu, sigma = fit_normalization(x)
            x_eff = (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        else:
            x_eff = x

        # score = |x^T (y - y_mean)|
        residual = y - y_mean
        grad = torch.bmm(x_eff.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)  # [b, d]
        grad_abs = torch.abs(grad)
        # Defer .item() to after division to minimize GPU-CPU sync points
        max_grad = grad_abs.max()

        alpha_max = (max_grad / (n * (l1_ratio if l1_ratio > 1e-3 else 0.001))).item()
        # Log space down to alpha_max * 1e-3
        alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-3), n_alphas).tolist()

    # Sort alphas high to low
    alphas = sorted(alphas, reverse=True)

    path_results = []  # List of ProbeCollections

    # Warm start weights
    # Note: fit_elastic_net_batch currently doesn't accept init_weights.
    # We need to modify fit_elastic_net_batch to accept `init_w` and `init_b`?
    # Or we can just call it and it restarts.
    # BUT the prompt says "warm starting each solve".
    # So I definitely need to add `w_init` / `b_init` support to `fit_elastic_net_batch`.
    # I should update `batch_elastic.py` first.
    # OR, I can inline the logic or pass a hidden kwargs?
    # Let's modify `fit_elastic_net_batch` to accept optional initialization.

    # For now, let's assume I can update `batch_elastic.py` in the next step or
    # pass it if it supports it (it doesn't yet).
    # I'll update this file to assume `w_init` kwarg exists, and then I'll update `batch_elastic.py`.

    w_init = None
    b_init = None

    for alpha in alphas:
        # Fit
        # We need to access the raw weights to pass as init for next.
        # Collection stores probekit, so we can extract.

        col = fit_elastic_net_batch(
            x,
            y,
            alpha=alpha,
            l1_ratio=l1_ratio,
            normalize=normalize,
            val_x=val_x,
            val_y=val_y,
            max_iter=100 if w_init is not None else 500,  # Faster if warm
            # We need to pass w_init.
            # I will add **kwargs to fit_elastic_net_batch or update signature.
            w_init=w_init,
            b_init=b_init,
        )
        path_results.append(col)

        # Extract weights for warm-start of next alpha
        w_init, b_init = col.to_tensor()
        # Only transfer if needed (to_tensor returns CPU tensors)
        if w_init.device != device:
            w_init = w_init.to(device)
            b_init = b_init.to(device)

    if select == "all":
        return path_results

    if select == "best_val":
        # Pick best alpha per batch element
        if val_x is None:
            # Fallback to last (lowest alpha) or raise?
            # "returns list of b LinearProbe objects"
            return path_results[-1]

        # We need to collect val_accuracy for each probe in the path
        # path_results is [n_alphas] of [b] probekit.
        # We want to pivot to [b] of best probe.

        best_probes = []
        for i in range(n_batch):
            best_p = None
            best_acc = -1.0

            for _col_idx, col in enumerate(path_results):
                p = col[i]
                acc = p.metadata.get("val_accuracy", -1.0)
                if acc > best_acc:
                    best_acc = acc
                    best_p = p

            # If no val acc found (e.g. -1 for all), pick last (least regularized)
            if best_p is None and best_acc == -1.0:
                best_p = path_results[-1][i]
            elif best_p is None:  # Should not happen if loop runs
                best_p = path_results[-1][i]

            best_probes.append(best_p)

        return ProbeCollection(best_probes)

    return path_results[-1]  # Fallback
