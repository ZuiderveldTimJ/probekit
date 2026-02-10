"""
ElasticNet Fitter (Sparse Probes).
"""


import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from probekit.core.probe import LinearProbe, NormalizationStats


def fit_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    l1_ratios: list[float] | float | None = None,
    alphas: list[float] | None = None,
    cv_folds: int = 5,
    normalize: bool = True,
    max_features: int | None = None,  # For SAE feature selection
    positive: bool = False,
    random_state: int = 42,
    max_iter: int = 1000,
    **kwargs,
) -> LinearProbe:
    """
    Fit an ElasticNet probe (sparse).

    Args:
        x: Activations [n_samples, n_features]
        y: Labels [n_samples]
        l1_ratios: Mix of L1/L2 (1.0 = Lasso)
        alphas: Regularization strengths (None = auto)
        cv_folds: Number of CV folds
        normalize: Whether to standardize input features
        max_features: If set, use ANOVA to select top-k features first (useful for SAEs)
        positive: Force positive coefficients
        random_state: Seed
        max_iter: Max iterations

    Returns:
        LinearProbe: The fitted probe.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    n_features_total = x.shape[1]
    selected_indices = None

    # 1. Feature Selection (Optional, for SAEs)
    if max_features and n_features_total > max_features:
        f_scores, _ = f_classif(x, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        selected_indices = np.argsort(f_scores)[-max_features:]
        selected_indices.sort()
        x_train = x[:, selected_indices]
    else:
        x_train = x

    # 2. Normalization (on subset)
    norm_stats = None
    if normalize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        # We need global stats for LinearProbe
        # Initialize with identity transform (mean=0, std=1)
        # So (x - 0) / 1 = x. Since w=0 for unselected, this is fine.
        global_mean = np.zeros(n_features_total, dtype=np.float64)
        global_std = np.ones(n_features_total, dtype=np.float64)

        if selected_indices is not None:
            # Check dimension of scaler stats
            # scaler.mean_ is (n_selected,)
            if scaler.mean_ is not None:
                global_mean[selected_indices] = scaler.mean_
            if scaler.scale_ is not None:
                global_std[selected_indices] = scaler.scale_
        else:
            # If no selection, x_train is x
             if scaler.mean_ is not None:
                global_mean[:] = scaler.mean_
             if scaler.scale_ is not None:
                global_std[:] = scaler.scale_

        norm_stats = NormalizationStats(
            mean=global_mean,
            std=global_std,
            count=scaler.n_samples_seen_,
        )

    # 3. Fit Model
    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv_folds,
        random_state=random_state,
        max_iter=max_iter,
        positive=positive,
        n_jobs=-1,
        **kwargs,
    )

    model.fit(x_train, y)

    # 4. Extract Weights & Remap if Selection was used
    local_weights = model.coef_
    # ElasticNet is a regressor on 0/1 labels.
    # Scores are in [0, 1] range.
    # We shift bias by -0.5 so that 0.0 becomes the decision boundary.
    bias = model.intercept_ - 0.5

    if selected_indices is not None:
        global_weights = np.zeros(n_features_total, dtype=np.float32)
        global_weights[selected_indices] = local_weights
    else:
        global_weights = local_weights

    # 5. Construct Probe
    return LinearProbe(
        weights=global_weights,
        bias=bias,
        normalization=norm_stats,
        metadata={
            "solver": "ElasticNetCV",
            "alpha": model.alpha_,
            "l1_ratio": model.l1_ratio_,
            "mse_path": model.mse_path_.tolist() if hasattr(model, "mse_path_") else None,
            "selected_features": selected_indices.tolist() if selected_indices is not None else None,
        },
    )
