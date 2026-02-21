from torch import Tensor

from probekit.utils.normalization import safe_std_torch


def fit_normalization(x: Tensor) -> tuple[Tensor, Tensor]:
    """
    x: [b, n, d] or [n, d]
    Returns (mu, sigma) where:
      - mu: [b, d] or [d] (mean per feature, per batch element)
      - sigma: [b, d] or [d] (std per feature, per batch element)
    Near-zero sigma values are replaced with 1.0 to avoid division by zero.
    Stats are computed along the n (samples) axis only.
    """
    if x.ndim == 3:
        # [b, n, d] -> reduce over dim 1 (n)
        mu = x.mean(dim=1)
        sigma = x.std(dim=1, unbiased=True)
    elif x.ndim == 2:
        # [n, d] -> reduce over dim 0 (n)
        mu = x.mean(dim=0)
        sigma = x.std(dim=0, unbiased=True)
    else:
        raise ValueError(f"Expected 2D or 3D input, got {x.shape}")

    # Replace near-zero sigma to avoid division by zero.
    sigma = safe_std_torch(sigma)

    return mu, sigma


def apply_normalization(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Subtract mu, divide by sigma. Handles broadcasting for batched case."""
    # If x is 3D [b, n, d] and mu/sigma are [b, d], we need to unsqueeze mu/sigma to [b, 1, d]
    if x.ndim == 3 and mu.ndim == 2:
        return (x - mu.unsqueeze(1)) / sigma.unsqueeze(1)

    # If x is 2D [n, d] and mu/sigma are [d], direct broadcast works
    if x.ndim == 2 and mu.ndim == 1:
        return (x - mu) / sigma

    # Validation fallback (though explicit shapes above cover expected cases)
    # If dimensions match exactly (e.g. single sample normalized by single stats?)
    # But typically mu is reduced dim.

    return (x - mu) / sigma
