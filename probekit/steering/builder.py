"""Utilities for building steering vectors from probes and SAE decoders."""

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sae_lens import SAE

from probekit.core.probe import LinearProbe
from probekit.utils.log import get_logger

logger = get_logger(__name__)

ProbeInput = LinearProbe | dict[str, Any] | NDArray[Any] | list[Any] | torch.Tensor


def build_steering_vector(
    probe: ProbeInput,
    sae: SAE,
    layer: int,
    output_dir: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build a single steering vector from probe coefficients and SAE decoder directions."""
    w_dec = sae.W_dec.data.to(dtype=torch.float32)
    device = w_dec.device
    steering_vector = torch.zeros(w_dec.shape[1], device=device, dtype=torch.float32)
    n_features_used = 0

    if isinstance(probe, LinearProbe):
        coeffs = torch.tensor(probe.direction, device=device, dtype=torch.float32)
        if len(coeffs) != w_dec.shape[0]:
            raise ValueError(f"LinearProbe direction shape {len(coeffs)} != SAE features {w_dec.shape[0]}")
        steering_vector = torch.einsum("f,fh->h", coeffs, w_dec)
        n_features_used = int(torch.count_nonzero(coeffs).item())

    elif isinstance(probe, dict):
        weights_raw = probe.get("weights")
        if weights_raw is None:
            raise ValueError("Probe dictionary must contain 'weights'.")

        features = probe.get("features", probe.get("feature_indices"))
        if features is None:
            raise ValueError("Probe dictionary must contain 'features' or 'feature_indices'.")
        if not isinstance(features, Sequence):
            raise ValueError("Probe 'features' must be a sequence of indices.")
        if not isinstance(weights_raw, Sequence):
            raise ValueError("Probe 'weights' must be a sequence.")

        feature_ids = [int(feat) for feat in features]
        weights = [float(weight) for weight in weights_raw]

        if len(feature_ids) != len(weights):
            raise ValueError(f"Length mismatch: {len(feature_ids)} features vs {len(weights)} weights.")

        for feat_id, weight in zip(feature_ids, weights, strict=True):
            if feat_id >= w_dec.shape[0]:
                raise ValueError(f"Feature index {feat_id} out of bounds for SAE with {w_dec.shape[0]} features.")
            steering_vector += weight * w_dec[feat_id].float()
        n_features_used = len(feature_ids)

    elif isinstance(probe, np.ndarray | list | torch.Tensor):
        coeffs = probe if isinstance(probe, torch.Tensor) else torch.tensor(probe, device=device, dtype=torch.float32)
        coeffs = coeffs.to(device=device, dtype=torch.float32)
        if len(coeffs) != w_dec.shape[0]:
            raise ValueError(f"Input vector shape {len(coeffs)} != SAE features {w_dec.shape[0]}")
        steering_vector = torch.einsum("f,fh->h", coeffs, w_dec)
        n_features_used = len(coeffs)

    else:
        raise TypeError(f"Unsupported probe type: {type(probe)}")

    norm = steering_vector.norm()
    steering_vector_normed = steering_vector / norm if norm > 0 else steering_vector
    if norm <= 0:
        logger.warning("Steering vector has zero norm.")

    data = {
        "vector": steering_vector.cpu(),
        "vector_normed": steering_vector_normed.cpu(),
        "norm": norm.item(),
        "n_features": n_features_used,
        "layer": layer,
    }

    if output_dir is not None:
        output_path = Path(output_dir) / "steering_vector.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        logger.info("Saved steering vector to %s", output_path)

    return data


def build_steering_vectors(
    probekit: Sequence[LinearProbe],
    sae_model: SAE,
    layers: Sequence[int],
) -> list[dict[str, Any]]:
    """Build steering vectors for probes and matching layer labels."""
    if len(probekit) != len(layers):
        raise ValueError(f"Number of probekit ({len(probekit)}) must match number of layers ({len(layers)})")

    return [build_steering_vector(probe, sae_model, layer) for probe, layer in zip(probekit, layers, strict=True)]
