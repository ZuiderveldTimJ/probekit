from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from probekit.core.probe import LinearProbe
from probekit.steering.builder import build_steering_vector, build_steering_vectors


@pytest.fixture
def mock_sae() -> MagicMock:
    sae = MagicMock()
    # w_dec [n_features, hidden_dim]
    sae.W_dec.data = torch.randn(100, 32, device="cuda" if torch.cuda.is_available() else "cpu")
    return sae


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_steering_vector_linear_probe(mock_sae: MagicMock) -> None:
    weights = np.random.randn(100).astype(np.float32)
    probe = LinearProbe(weights, bias=0.0)

    res = build_steering_vector(probe, mock_sae, layer=10)
    assert res["layer"] == 10
    assert res["vector"].shape == (32,)
    assert res["n_features"] == 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_steering_vector_legacy_dict(mock_sae: MagicMock) -> None:
    probe_dict = {"features": [0, 1, 2], "weights": [1.0, -0.5, 2.0]}
    res = build_steering_vector(probe_dict, mock_sae, layer=10)
    assert res["n_features"] == 3
    assert res["vector"].shape == (32,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_steering_vectors_batched(mock_sae: MagicMock) -> None:
    probekit = [
        LinearProbe(np.random.randn(100).astype(np.float32), 0.0),
        LinearProbe(np.random.randn(100).astype(np.float32), 0.0),
    ]
    layers = [10, 11]
    results = build_steering_vectors(probekit, mock_sae, layers)
    assert len(results) == 2
    assert results[0]["layer"] == 10
    assert results[1]["layer"] == 11
