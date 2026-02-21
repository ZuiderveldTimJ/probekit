"""Utilities for numerically safe normalization statistics."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


def safe_std_numpy(std: NDArray[Any], eps: float = 1e-12) -> NDArray[Any]:
    """Replace near-zero standard deviations with 1.0."""
    return np.where(np.abs(std) <= eps, np.ones_like(std), std)


def safe_std_torch(std: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Replace near-zero standard deviations with 1.0."""
    return torch.where(torch.abs(std) <= eps, torch.ones_like(std), std)
