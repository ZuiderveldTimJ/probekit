import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection
from probekit.core.probe import NormalizationStats

from .normalize import fit_normalization


def fit_dim_batch(x: Tensor, y: Tensor, normalize: bool = True) -> ProbeCollection:
    """
    x: [b, n, d]
    y: [n] or [b, n] (binary labels 0/1)
    Returns: ProbeCollection
    """
    device = x.device
    if x.ndim != 3:
        raise ValueError(f"x must be 3D [b, n, d], got {x.shape}")

    n_batch, n, d = x.shape

    # Handle y broadcasting
    if y.ndim == 1:
        # [n] -> [b, n]
        y = y.unsqueeze(0).expand(n_batch, n)

    if y.shape != (n_batch, n):
        raise ValueError(f"y shape {y.shape} mismatch with x {x.shape}")

    # 1. Compute means efficiently using einsum
    sum_1 = torch.einsum("bnd,bn->bd", x, y)
    count_1 = y.sum(dim=1, keepdim=True)
    mean_1 = sum_1 / count_1.clamp(min=1.0)

    y_inv = 1.0 - y
    sum_0 = torch.einsum("bnd,bn->bd", x, y_inv)
    count_0 = y_inv.sum(dim=1, keepdim=True)
    mean_0 = sum_0 / count_0.clamp(min=1.0)

    # 2. Weights
    w = mean_1 - mean_0  # [b, d]

    # 3. Bias and Normalization
    if normalize:
        mu, sigma = fit_normalization(x)  # [b, d]
    else:
        mu = torch.zeros(n_batch, d, device=device)
        sigma = torch.ones(n_batch, d, device=device)

    # Vectorized bias computation
    # Normalize all batches at once: [b, n, d]
    x_norm = (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    # Scores: [b, n, d] @ [b, d, 1] -> [b, n]
    scores = torch.bmm(x_norm, w.unsqueeze(-1)).squeeze(-1)
    # Median per batch: [b]
    biases = -torch.median(scores, dim=1).values

    # Transfer to CPU once for all batches
    w_cpu = w.cpu().numpy()
    mu_cpu = mu.cpu().numpy()
    sigma_cpu = sigma.cpu().numpy()
    biases_cpu = biases.cpu().numpy()

    normalizations = [
        NormalizationStats(mean=mu_cpu[i], std=sigma_cpu[i], count=n) if normalize else None for i in range(n_batch)
    ]
    metadatas = [{"fit_method": "dim_batch"} for _ in range(n_batch)]

    return ProbeCollection(
        weights=w_cpu,
        biases=biases_cpu,
        normalizations=normalizations,
        metadatas=metadatas,
    )
