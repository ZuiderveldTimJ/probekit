"""
Difference in Means (DiM) Fitter.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probekit.core.probe import LinearProbe, NormalizationStats


def fit_dim(
    x: NDArray[Any],
    y: NDArray[Any],
    normalize: bool = False,  # Generally DiM is done on raw or whitened data
    use_calibration: bool = True,  # Fit a 1D logistic regression on the projection
    random_state: int = 42,
) -> LinearProbe:
    """
    Fit a Difference in Means probe.

    weight = mean(pos) - mean(neg)

    Args:
        x: Activations [n_samples, n_features]
        y: Labels [n_samples]
        normalize: Whether to standardize input first
        use_calibration: If True, fits a scalar s and b to: s * (w.x) + b
        random_state: Seed

    Returns:
        LinearProbe
    """
    # 1. Normalization
    norm_stats = None
    if normalize:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        norm_stats = NormalizationStats(
            mean=scaler.mean_,
            std=scaler.scale_,
            count=scaler.n_samples_seen_,
        )

    # 2. Compute Means
    pos_mask = y == 1
    neg_mask = y == 0

    mu_pos = x[pos_mask].mean(axis=0)
    mu_neg = x[neg_mask].mean(axis=0)

    direction = mu_pos - mu_neg

    # 3. Calibration (Optional)
    scale = 1.0
    bias = 0.0

    if use_calibration:
        # Project data onto direction
        proj = x @ direction

        # Fit 1D Logistic Regression
        lr = LogisticRegression(random_state=random_state)
        lr.fit(proj.reshape(-1, 1), y)

        scale = lr.coef_[0][0]
        bias = lr.intercept_[0]

    final_weights = scale * direction
    final_bias = bias

    # 4. Construct Probe
    return LinearProbe(
        weights=final_weights,
        bias=final_bias,
        normalization=norm_stats,
        metadata={
            "solver": "DifferenceInMeans",
            "calibrated": use_calibration,
            "raw_mu_diff_norm": np.linalg.norm(direction),
        },
    )
