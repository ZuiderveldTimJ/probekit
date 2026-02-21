import pytest
import torch

from probekit.core.collection import ProbeCollection
from probekit.fitters.batch.dim import fit_dim_batch
from probekit.fitters.batch.elastic import fit_elastic_net_batch
from probekit.fitters.batch.logistic import fit_logistic_batch
from probekit.fitters.batch.normalize import fit_normalization


@pytest.fixture
def batch_data() -> tuple[torch.Tensor, torch.Tensor]:
    b, n, d = 2, 20, 10
    x = torch.randn(b, n, d, device="cuda" if torch.cuda.is_available() else "cpu")
    y = torch.randint(0, 2, (b, n), device=x.device).float()
    return x, y


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_logistic_batch(batch_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = batch_data
    collection = fit_logistic_batch(x, y, max_iter=5)
    assert isinstance(collection, ProbeCollection)
    assert len(collection) == x.shape[0]

    # Verify weights shape
    w, b = collection.to_tensor()
    assert w.shape == (x.shape[0], x.shape[2])
    assert b.shape == (x.shape[0],)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_dim_batch(batch_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = batch_data
    collection = fit_dim_batch(x, y)
    assert isinstance(collection, ProbeCollection)
    assert len(collection) == x.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_elastic_net_batch(batch_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = batch_data
    collection = fit_elastic_net_batch(x, y, max_iter=5)
    assert isinstance(collection, ProbeCollection)
    assert len(collection) == x.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_elastic_net_batch_positive(batch_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = batch_data
    collection = fit_elastic_net_batch(x, y, max_iter=5, positive=True)
    weights, _ = collection.to_tensor()
    assert torch.all(weights >= -1e-7)


def test_fit_normalization_handles_zero_std() -> None:
    x = torch.ones(2, 5, 3)
    _mu, sigma = fit_normalization(x)
    assert torch.allclose(sigma, torch.ones_like(sigma))
