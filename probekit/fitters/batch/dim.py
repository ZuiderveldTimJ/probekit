import torch
from torch import Tensor

from probekit.core.collection import ProbeCollection
from probekit.core.probe import LinearProbe, NormalizationStats

from .normalize import fit_normalization


def fit_dim_batch(x: Tensor, y: Tensor, normalize: bool = True) -> ProbeCollection:
    """
    x: [b, n, d]
    y: [n] or [b, n] (binary labels 0/1)
    Returns: ProbeCollection of b LinearProbe objects

    Algorithm:
    1. Compute per-class means: mean of x where y==1, mean where y==0, per batch element
    2. w = class1_mean - class0_mean
    3. Normalize x, compute bias via median threshold on normalized scores
    4. Store normalization stats (mu, sigma) on each probe
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

    # 1. Compute means
    # We can use masked summation
    # y is 0/1.
    # sum_1 = (X * y.unsqueeze(-1)).sum(dim=1)
    # count_1 = y.sum(dim=1).unsqueeze(-1)

    y_expanded = y.unsqueeze(-1) # [b, n, 1]

    sum_1 = (x * y_expanded).sum(dim=1) # [b, d]
    count_1 = y_expanded.sum(dim=1) # [b, 1]
    mean_1 = sum_1 / count_1.clamp(min=1.0)

    sum_0 = (x * (1 - y_expanded)).sum(dim=1)
    count_0 = (1 - y_expanded).sum(dim=1)
    mean_0 = sum_0 / count_0.clamp(min=1.0)

    # 2. Weights
    w = mean_1 - mean_0 # [b, d]

    # 3. Bias and Normalization
    probekit = []

    # If normalize=True, we compute stats and store them.
    # The bias calculation generally happens on the *normalized* scores if normalization is utilized at inference time.
    # The spec says: "Normalize x, compute bias via median threshold on normalized scores"

    if normalize:
        mu, sigma = fit_normalization(x) # [b, d]
    else:
        mu = torch.zeros(n_batch, d, device=device)
        sigma = torch.ones(n_batch, d, device=device)

    for i in range(n_batch):
        # Extract for this batch
        wi = w[i] # [D]
        mui = mu[i]
        sigmai = sigma[i]

        # Calculate scores to find bias
        # s = (x - mu)/sigma @ w
        # We want median score to be 0 (or balanced threshold)
        # Actually, standard DiM often sets bias = -(mean1 + mean0)/2 @ w  (midpoint)
        # But the spec says "median threshold".

        xi = x[i] # [n, d]
        # Normalize
        xi_norm = (xi - mui) / sigmai
        scores = xi_norm @ wi

        # Bias = -median(scores)
        # So that median(scores + b) = 0
        b = -torch.median(scores)

        probe = LinearProbe(
            weights=wi.cpu().numpy(),
            bias=b.item(),
            normalization=NormalizationStats(
                mean=mui.cpu().numpy(),
                std=sigmai.cpu().numpy(),
                count=n
            ) if normalize else None,
            metadata={"fit_method": "dim_batch"}
        )
        probekit.append(probe)

    return ProbeCollection(probekit)
