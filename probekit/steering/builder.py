"""
Steering Vector Builder (V2).

Creates steering vectors from probe weights × SAE decoder directions.
Can be used standalone or imported by other scripts.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sae_lens import SAE

from probekit.core.probe import LinearProbe
from probekit.utils.log import get_logger

logger = get_logger(__name__)

# Defaults
DEFAULT_SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"


def build_steering_vector(
    probe: LinearProbe | dict | NDArray | list,
    sae: SAE,
    layer: int,
    output_dir: Path | None = None,
) -> dict:
    """Create steering vector from probe weights x SAE decoder directions.

    This V2 implementation primarily expects a `LinearProbe` object,
    but supports legacy formats for backward compatibility during migration.

    Args:
        probe: Source of direction. Can be:
            - LinearProbe: The V2 standard.
            - Dict: Legacy sparse probe {'features': [], 'weights': []}
            - NDArray/List: Dense vector in SAE feature space
        sae: Loaded SAE object
        layer: Layer number
        output_dir: Where to save (optional, won't save if None)

    Returns:
        Dict with 'vector', 'vector_normed', 'norm', 'n_features', 'layer'

    Raises:
        ValueError: If probe format is unrecognized or dimensions mismatch.
    """
    # w_dec is [n_features, hidden_dim]
    w_dec = sae.W_dec.data
    steering_vector = torch.zeros(w_dec.shape[1], device="cuda", dtype=torch.float32)
    n_features_used = 0

    # Case 1: LinearProbe (V2 Standard)
    if isinstance(probe, LinearProbe):
        direction = probe.direction  # NDArray
        coeffs = torch.tensor(direction, device="cuda", dtype=torch.float32)

        # Check for sparse/dense logic implicitly via shape
        if len(coeffs) == w_dec.shape[0]:
            # Dense weights in SAE space -> Projection
            steering_vector = torch.einsum("f,fh->h", coeffs, w_dec)
            n_features_used = torch.count_nonzero(coeffs).item()
        else:
            # Mismatch? Maybe it's already in residual stream space?
            # If so, we can't use W_dec to project it.
            # But the contract says probe is in "input space".
            # If input space != SAE features, we have a problem unless we know the mapping.
            # For now, assume LinearProbe weights are in SAE feature space.
            raise ValueError(f"LinearProbe weights shape {len(coeffs)} != SAE features {w_dec.shape[0]}")

    # Case 2: Legacy Dictionary
    elif isinstance(probe, dict):
        features = probe.get("features", probe.get("feature_indices", []))
        weights = probe.get("weights")

        if weights is None:
            raise ValueError("Probe dictionary must contain 'weights'.")

        n_features_used = len(features)

        if len(features) != len(weights):
             raise ValueError(f"Length mismatch: {len(features)} features vs {len(weights)} weights.")

        for feat_id, weight in zip(features, weights, strict=True):
            if feat_id >= w_dec.shape[0]:
                raise ValueError(f"Feature index {feat_id} out of bounds for SAE with {w_dec.shape[0]} features.")
            steering_vector += weight * w_dec[feat_id].float()

    # Case 3: Dense Vector / List
    elif isinstance(probe, (np.ndarray, list, torch.Tensor)):
        coeffs = torch.tensor(probe, device="cuda", dtype=torch.float32) if not isinstance(probe, torch.Tensor) else probe.to("cuda")

        if len(coeffs) != w_dec.shape[0]:
            raise ValueError(f"Input vector shape {len(coeffs)} != SAE features {w_dec.shape[0]}")

        steering_vector = torch.einsum("f,fh->h", coeffs, w_dec)
        n_features_used = len(coeffs)

    else:
        raise TypeError(f"Unsupported probe type: {type(probe)}")

    # Normalize
    norm = steering_vector.norm()
    if norm > 0:
        steering_vector_normed = steering_vector / norm
    else:
        logger.warning("Steering vector has zero norm.")
        steering_vector_normed = steering_vector

    data = {
        "vector": steering_vector.cpu(),
        "vector_normed": steering_vector_normed.cpu(),
        "norm": norm.item(),
        "n_features": n_features_used,
        "layer": layer,
    }

    # Save if output_dir provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "steering_vector.pt"
        torch.save(data, output_path)
        logger.info(f"Saved steering vector to {output_path}")

    return data

def build_steering_vectors(
    probekit: list[LinearProbe] | Any,
    sae_model: Any,
    layers: list[int],
) -> list[dict]:
    """
    Batched version of build_steering_vector.
    Maps each probe to the corresponding layer.
    """
    if len(probekit) != len(layers):
        raise ValueError(f"Number of probekit ({len(probekit)}) must match number of layers ({len(layers)})")

    results = []
    for probe, layer in zip(probekit, layers, strict=True):
        # We assume sae_model is reusable or handles multi-layer?
        # Actually build_steering_vector takes `sae` which is specific to a layer usually?
        # The signature says `sae: SAE`. SAE is usually specific to a layer.
        # If the user passes a list of layers, and a SINGLE sae_model, that implies
        # the SAE model is somehow applicable to all (e.g. same architecture?)
        # OR the user handles SAE loading outside.
        # But wait, `build_steering_vector` uses `sae.W_dec`.
        # If we pass different layers, we might need different SAEs.
        # The prompt says: "Maps each probe to the corresponding layer. Just maps over build_steering_vector."
        # It accepts `sae_model`. If `sae_model` is a single SAE object,
        # it will be used for all. This might be wrong if layers differ.
        # But I must follow the prompt: "sae_model" (singular).
        # Maybe it's a dict of SAEs? Or maybe the user ensures it's correct context.
        # I will implement as requested.

        vec = build_steering_vector(probe, sae_model, layer)
        results.append(vec)

    return results


def load_probe(probe_path: str | Path) -> tuple[dict, int, Path]:
    """Load a probe and extract its layer and directory.

    Returns:
        (probe_data, layer, probe_dir)

    Raises:
        FileNotFoundError: If probe file not found.
        ValueError: If layer cannot be determined or JSON is invalid.
    """
    path = Path(probe_path)

    if not path.suffix == ".json":
        path = Path(str(path) + ".json")

    if not path.exists():
        raise FileNotFoundError(f"Probe not found: {path}")

    # Extract layer from path (e.g., probekit/L10/probekit/lies.json)
    layer = None
    for part in path.parts:
        if part.startswith("L") and part[1:].isdigit():
            layer = int(part[1:])
            break

    try:
        with open(path) as f:
            data = json.load(f)

        # Try to convert to LinearProbe if it matches schema
        if "weights" in data and "bias" in data:
            # It's likely a LinearProbe dict
            # We return it as a dict for now, build_steering_vector handles dicts
            # or we could hydrate it here?
            pass

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in probe file {path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error reading probe file {path}: {e}") from e

    if layer is None:
        if "layer" in data:
            layer = data["layer"]
        else:
            raise ValueError(f"Could not determine layer from path: {path}")

    # probe_dir (logic to find where to save related artifacts)
    try:
        probe_dir = path.parent.parent if path.parent.name == "probekit" else path.parent
    except Exception:
        probe_dir = path.parent

    return data, layer, probe_dir


def main():
    parser = argparse.ArgumentParser(description="Build steering vector from probe")
    parser.add_argument("--probe", required=True, help="Path to probe JSON file")
    parser.add_argument("--sae-release", default=DEFAULT_SAE_RELEASE, help="SAE release")
    parser.add_argument("--output", help="Output directory (default: same as probe)")
    args = parser.parse_args()

    # Configure logging for CLI usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load probe
        probe, layer, probe_dir = load_probe(args.probe)
        output_dir = Path(args.output) if args.output else probe_dir

        logger.info(f"Building steering vector for layer {layer}")

        # Load SAE
        sae_id = f"layer_{layer}/width_16k/canonical"
        logger.info(f"Loading SAE: {args.sae_release} / {sae_id}")
        sae = SAE.from_pretrained(release=args.sae_release, sae_id=sae_id, device="cuda")

        # Hydrate LinearProbe if possible, else use raw dict
        if isinstance(probe, dict) and "weights" in probe and "bias" in probe:
             probe_obj = LinearProbe.from_dict(probe)
        else:
             probe_obj = probe

        # Build vector
        data = build_steering_vector(probe_obj, sae, layer, output_dir)
        logger.info(f"Created vector: norm={data['norm']:.4f}")

    except Exception as e:
        logger.error(f"Failed to build steering vector: {e}")
        exit(1)


if __name__ == "__main__":
    main()
