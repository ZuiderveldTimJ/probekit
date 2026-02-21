import numpy as np
import pytest
import torch

from probekit.fitters.batch.elastic import fit_elastic_net_batch
from probekit.fitters.batch.logistic import fit_logistic_batch
from probekit.fitters.elastic import fit_elastic_net
from probekit.fitters.logistic import fit_logistic


def _synthetic_classification_batch(
    batch_size: int, n_samples: int, n_features: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch_size, n_samples, n_features, device=device)
    w_true = torch.randn(batch_size, n_features, device=device)
    b_true = torch.randn(batch_size, device=device)
    logits = torch.bmm(x, w_true.unsqueeze(-1)).squeeze(-1) + b_true.unsqueeze(1)
    probs = torch.sigmoid(logits)
    y = torch.bernoulli(probs)
    return x, y


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_logistic_batch_matches_cpu_accuracy() -> None:
    x, y = _synthetic_classification_batch(batch_size=3, n_samples=256, n_features=24, device="cuda")

    batch_col = fit_logistic_batch(x, y, c_param=1.0, max_iter=80, tol=1e-6)
    batch_preds = batch_col.predict(x)
    assert isinstance(batch_preds, torch.Tensor)
    batch_acc = (batch_preds == y.to(dtype=torch.int32)).to(dtype=torch.float32).mean(dim=1).cpu().numpy()

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy().astype(np.int32)
    cpu_acc: list[float] = []
    for i in range(x_np.shape[0]):
        probe = fit_logistic(x_np[i], y_np[i], c_param=1.0, cv_folds=None, max_iter=3000)
        cpu_acc.append(float(np.mean(probe.predict(x_np[i]) == y_np[i])))

    cpu_acc_arr = np.array(cpu_acc)
    assert np.max(np.abs(batch_acc - cpu_acc_arr)) < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_elastic_batch_matches_cpu_accuracy() -> None:
    x, y = _synthetic_classification_batch(batch_size=2, n_samples=256, n_features=18, device="cuda")

    batch_col = fit_elastic_net_batch(x, y, alpha=0.05, l1_ratio=0.8, max_iter=400, tol=1e-6)
    batch_preds = batch_col.predict(x)
    assert isinstance(batch_preds, torch.Tensor)
    batch_acc = (batch_preds == y.to(dtype=torch.int32)).to(dtype=torch.float32).mean(dim=1).cpu().numpy()

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy().astype(np.int32)
    cpu_acc: list[float] = []
    for i in range(x_np.shape[0]):
        probe = fit_elastic_net(
            x_np[i],
            y_np[i],
            l1_ratios=[0.8],
            alphas=[0.05],
            cv_folds=3,
            max_iter=5000,
        )
        cpu_acc.append(float(np.mean(probe.predict(x_np[i]) == y_np[i])))

    cpu_acc_arr = np.array(cpu_acc)
    assert np.max(np.abs(batch_acc - cpu_acc_arr)) < 0.08
