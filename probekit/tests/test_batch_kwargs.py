import torch

from probekit.api import fit_sparse_probe_batch, sae_probe


def test_fit_sparse_probe_batch_kwargs() -> None:
    x = torch.randn(2, 8, 16)
    y = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float32)

    # Test that passing positive=True and max_iter=10 works
    col = fit_sparse_probe_batch(
        x,
        y,
        l1_ratios=[0.5],
        positive=True,
        max_iter=10,
        normalize=True,
    )
    assert len(col) == 2

    weights, _ = col.to_tensor(device="cpu")
    # if positive=True, all weights should be >= 0
    assert torch.all(weights >= 0.0)


def test_sae_probe_kwargs_passing() -> None:
    x = torch.randn(1, 8, 16)
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)

    # Sae probe should pass kwargs down without crashing
    probe = sae_probe(
        x,
        y,
        positive=True,
        max_iter=10,
        normalize=True,
    )

    assert probe is not None
    weights = torch.tensor(probe.weights)
    assert torch.all(weights >= 0.0)
