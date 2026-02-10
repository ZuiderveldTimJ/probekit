from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from probekit.core.collection import ProbeCollection
from probekit.core.probe import LinearProbe, NormalizationStats
from probekit.fitters.dim import fit_dim
from probekit.fitters.elastic import fit_elastic_net
from probekit.fitters.logistic import fit_logistic


@pytest.fixture
def sample_data() -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    x = np.random.randn(n_samples, n_features).astype(np.float32)

    # Linear signal
    weights = np.zeros(n_features)
    weights[:5] = [1.0, -0.5, 0.8, -0.3, 0.6]
    logits = x @ weights
    y = (logits > 0).astype(np.int32)
    return x, y


class TestLinearProbe:
    def test_init_and_properties(self) -> None:
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        bias = 0.5
        probe = LinearProbe(weights, bias)

        assert np.allclose(probe.weights, weights)
        assert probe.bias == 0.5
        assert np.allclose(probe.direction, weights)

    def test_predict_score(self) -> None:
        # x = [1, 1, 1] -> 1*1 + 2*1 + 3*1 + 0.5 = 6.5
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        bias = 0.5
        probe = LinearProbe(weights, bias)

        x = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        scores = probe.predict_score(x)
        assert np.allclose(scores, [6.5])

    def test_predict_threshold(self) -> None:
        weights = np.array([1.0], dtype=np.float32)
        bias = 0.0
        probe = LinearProbe(weights, bias)

        x = np.array([[-1.0], [1.0]], dtype=np.float32)
        preds = probe.predict(x, threshold=0.0)
        assert np.allclose(preds.flatten(), [0, 1])

    def test_normalization(self) -> None:
        weights = np.array([1.0], dtype=np.float32)
        # norm: mean=10, std=2. x=12 -> (12-10)/2 = 1. 1*1 = 1.
        norm = NormalizationStats(mean=np.array([10.0]), std=np.array([2.0]), count=100)
        probe = LinearProbe(weights, bias=0.0, normalization=norm)

        x = np.array([[12.0]], dtype=np.float32)
        scores = probe.predict_score(x)
        assert np.allclose(scores, [1.0])

    def test_serialization(self) -> None:
        weights = np.array([1.0, 2.0], dtype=np.float32)
        probe = LinearProbe(weights, bias=0.5)
        data = probe.to_dict()
        probe2 = LinearProbe.from_dict(data)

        assert np.allclose(probe.weights, probe2.weights)
        assert probe.bias == probe2.bias


class TestFitters:
    def test_fit_logistic(self, sample_data: tuple[NDArray[Any], NDArray[Any]]) -> None:
        x, y = sample_data
        probe = fit_logistic(x, y, c_param=1.0)
        assert isinstance(probe, LinearProbe)
        assert probe.weights.shape == (50,)
        # Should learn something
        acc = (probe.predict(x) == y).mean()
        assert acc > 0.8

    def test_fit_elastic(self, sample_data: tuple[NDArray[Any], NDArray[Any]]) -> None:
        x, y = sample_data
        probe = fit_elastic_net(x, y, l1_ratios=[0.5])
        assert isinstance(probe, LinearProbe)
        assert probe.weights.shape == (50,)
        # Weights should be somewhat sparse or at least learned
        assert np.any(probe.weights != 0)

    def test_fit_dim(self, sample_data: tuple[NDArray[Any], NDArray[Any]]) -> None:
        x, y = sample_data
        probe = fit_dim(x, y)
        assert isinstance(probe, LinearProbe)
        assert probe.weights.shape == (50,)


class TestProbeCollection:
    def test_predict_with_shared_input(self) -> None:
        probe_a = LinearProbe(weights=np.array([1.0, 0.0], dtype=np.float32), bias=0.0)
        probe_b = LinearProbe(weights=np.array([0.0, 1.0], dtype=np.float32), bias=0.0)
        collection = ProbeCollection([probe_a, probe_b])

        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
        scores = collection.predict_score(x)
        preds = collection.predict(x)

        assert scores.shape == (2, 2)
        assert preds.shape == (2, 2)
        assert np.array_equal(preds[0], np.array([1, 0], dtype=np.int32))
        assert np.array_equal(preds[1], np.array([0, 1], dtype=np.int32))

    def test_predict_with_batched_input(self) -> None:
        probe_a = LinearProbe(weights=np.array([1.0, 0.0], dtype=np.float32), bias=0.0)
        probe_b = LinearProbe(weights=np.array([0.0, 1.0], dtype=np.float32), bias=0.0)
        collection = ProbeCollection([probe_a, probe_b])

        x = np.array(
            [
                [[1.0, -1.0], [-1.0, 1.0]],
                [[-1.0, 1.0], [1.0, -1.0]],
            ],
            dtype=np.float32,
        )
        preds = collection.predict(x)
        assert preds.shape == (2, 2)
        assert np.array_equal(preds[0], np.array([1, 0], dtype=np.int32))
        assert np.array_equal(preds[1], np.array([1, 0], dtype=np.int32))
