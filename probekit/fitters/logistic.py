"""
Standard Logistic Regression Fitter.
"""


import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from probekit.core.probe import LinearProbe, NormalizationStats


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    c_param: float = 1.0,
    cv_folds: int | None = 5,
    normalize: bool = True,
    random_state: int = 42,
    max_iter: int = 1000,
    **kwargs,
) -> LinearProbe:
    """
    Fit a standard L2-regularized Logistic Regression probe.

    Args:
        x: Activations [n_samples, n_features]
        y: Labels [n_samples]
        c_param: Inverse regularization strength (smaller = stronger reg)
        cv_folds: Number of CV folds (if None, use fixed c_param)
        normalize: Whether to standardize input features
        random_state: Seed for reproducibility
        max_iter: Max iterations for solver

    Returns:
        LinearProbe: The fitted probe.
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

    # 2. Fit Model
    if cv_folds:
        model = LogisticRegressionCV(
            cv=cv_folds,
            random_state=random_state,
            max_iter=max_iter,
            scoring="accuracy",
            n_jobs=-1,
            **kwargs,
        )
    else:
        model = LogisticRegression(
            C=c_param,
            random_state=random_state,
            max_iter=max_iter,
            n_jobs=-1,
            **kwargs,
        )

    model.fit(x, y)

    # 3. Extract Weights
    # sklearn returns [n_classes, n_features] for binary too (usually 1 row)
    weights = model.coef_.flatten()
    bias = model.intercept_[0]

    # 4. Construct Probe
    return LinearProbe(
        weights=weights,
        bias=bias,
        normalization=norm_stats,
        metadata={
            "solver": "LogisticRegressionCV" if cv_folds else "LogisticRegression",
            "C": model.C_[0] if cv_folds else c_param,
            "cv_accuracy": model.scores_[1].mean() if cv_folds else None,  # Approximate
            "classes": model.classes_.tolist(),
            "n_iter": model.n_iter_.tolist(),
        },
    )
