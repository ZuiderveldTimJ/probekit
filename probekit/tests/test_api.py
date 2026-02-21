from typing import Any, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from probekit.api import _resolve_torch_device, dim_probe, logistic_probe, nelp_probe, sae_probe
from probekit.core.collection import ProbeCollection
from probekit.core.probe import LinearProbe


@pytest.fixture
def data_2d() -> tuple[NDArray[Any], NDArray[Any]]:
    x = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=(10,)).astype(np.int32)
    return x, y


@pytest.fixture
def data_3d() -> tuple[torch.Tensor, torch.Tensor]:
    # [B, N, D]
    x = torch.randn(2, 10, 5)
    y = torch.randint(0, 2, size=(2, 10)).float()
    return x, y


def test_api_routing_2d(data_2d: tuple[NDArray[Any], NDArray[Any]]) -> None:
    x, y = data_2d
    # Test sae_probe (logistic)
    probe = sae_probe(x, y)
    assert isinstance(probe, LinearProbe)

    # Test dim_probe
    probe_dim = dim_probe(x, y)
    assert isinstance(probe_dim, LinearProbe)


def test_api_routing_3d(data_3d: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = data_3d
    # Test sae_probe (elastic_batch path)
    collection = sae_probe(x, y)
    assert isinstance(collection, ProbeCollection)
    assert len(collection) == 2

    # Test dim_probe (dim_batch)
    collection_dim = dim_probe(x, y)
    assert isinstance(collection_dim, ProbeCollection)
    assert len(collection_dim) == 2


def test_aliases(data_2d: tuple[NDArray[Any], NDArray[Any]]) -> None:
    x, y = data_2d
    p1 = sae_probe(x, y)
    p2 = logistic_probe(x, y)
    p3 = nelp_probe(x, y)
    assert isinstance(p1, LinearProbe)
    assert isinstance(p2, LinearProbe)
    assert isinstance(p3, LinearProbe)


def test_torch_backend_for_2d_tensor() -> None:
    x = torch.randn(16, 6, dtype=torch.float32)
    y = torch.randint(0, 2, (16,), dtype=torch.float32)

    probe = logistic_probe(x, y, backend="torch", max_iter=20)
    assert isinstance(probe, LinearProbe)
    assert probe.metadata["fit_method"] == "logistic_batch_irls"


def test_sklearn_backend_for_3d_numpy() -> None:
    x = np.random.randn(2, 20, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=(2, 20)).astype(np.int32)

    probes = logistic_probe(x, y, backend="sklearn", cv_folds=None)
    assert isinstance(probes, ProbeCollection)
    assert len(probes) == 2
    assert probes[0].metadata["solver"] == "LogisticRegression"


def test_invalid_backend_raises(data_2d: tuple[NDArray[Any], NDArray[Any]]) -> None:
    x, y = data_2d
    with pytest.raises(ValueError):
        sae_probe(x, y, backend=cast(Any, "not-a-backend"))


def test_resolve_torch_device_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROBEKIT_TORCH_DEVICE", "cpu")
    resolved = _resolve_torch_device(np.zeros((2, 2), dtype=np.float32), None)
    assert str(resolved) == "cpu"
