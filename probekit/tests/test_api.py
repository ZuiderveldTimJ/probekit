import numpy as np
import pytest
import torch

from probekit.api import dim_probe, logistic_probe, nelp_probe, sae_probe
from probekit.core.collection import ProbeCollection
from probekit.core.probe import LinearProbe


@pytest.fixture
def data_2d():
    x = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=(10,)).astype(np.int32)
    return x, y


@pytest.fixture
def data_3d():
    # [B, N, D]
    x = torch.randn(2, 10, 5)
    y = torch.randint(0, 2, size=(2, 10))
    return x, y


def test_api_routing_2d(data_2d):
    x, y = data_2d
    # Test sae_probe (logistic)
    probe = sae_probe(x, y)
    assert isinstance(probe, LinearProbe)

    # Test dim_probe
    probe_dim = dim_probe(x, y)
    assert isinstance(probe_dim, LinearProbe)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_api_routing_3d(data_3d):
    x, y = data_3d
    # Test sae_probe (logistic_batch)
    collection = sae_probe(x, y)
    assert isinstance(collection, ProbeCollection)
    assert len(collection.probekit) == 2

    # Test dim_probe (dim_batch)
    collection_dim = dim_probe(x, y)
    assert isinstance(collection_dim, ProbeCollection)
    assert len(collection_dim.probekit) == 2


def test_aliases(data_2d):
    x, y = data_2d
    p1 = sae_probe(x, y)
    p2 = logistic_probe(x, y)
    p3 = nelp_probe(x, y)
    assert isinstance(p1, LinearProbe)
    assert isinstance(p2, LinearProbe)
    assert isinstance(p3, LinearProbe)
