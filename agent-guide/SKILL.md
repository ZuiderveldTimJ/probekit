# Probekit Usage Guide for AI Agents

This file is a plain operational guide, not a Codex skill definition.

## Goal

Use `probekit` to train linear probes on activations, evaluate probe behavior, and optionally build steering vectors.

## Core Imports

```python
from probekit import sae_probe, logistic_probe, dim_probe
from probekit import build_steering_vector, build_steering_vectors
```

## Input Conventions

- Single probe:
  - `x`: shape `[N, D]`
  - `y`: shape `[N]`
- Batched probes:
  - `x`: shape `[B, N, D]`
  - `y`: shape `[B, N]` (or broadcast-compatible)

Routing behavior:
- `sae_probe` and `logistic_probe` use logistic fitting.
- `dim_probe` uses difference-in-means fitting.
- 3D `x` routes to batch fitters and returns `ProbeCollection`.
- 2D `x` routes to single fitters and returns `LinearProbe`.

## Minimal Workflow

1. Prepare activations and binary labels.
2. Train a probe:

```python
probe = sae_probe(x, y)
```

3. Score or predict:

```python
scores = probe.predict_score(x)
pred = probe.predict(x, threshold=0.0)
```

4. If batched, convert collection to tensors:

```python
weights, biases = probes.to_tensor()
```

## Steering Workflow

Use when a trained probe direction should be projected through an SAE decoder.

```python
vec = build_steering_vector(probe, sae, layer=10)
```

Returns a dict containing:
- `vector`
- `vector_normed`
- `norm`
- `n_features`
- `layer`

## Practical Guardrails

- Keep dtype numeric and consistent (`float32` preferred for activations).
- Validate shapes before fitting; most runtime errors come from dimension mismatch.
- Treat `y` as binary labels for current probe fitters.
- For reproducibility, set random seeds in your calling code.
- For large jobs, use batched inputs (`[B, N, D]`) to leverage GPU batch fitters.

## Common Tasks

- Train SAE/logistic probe quickly:
  - call `sae_probe(x, y)`
- Train DiM probe for interpretable class separation:
  - call `dim_probe(x, y)`
- Select best probe from a `ProbeCollection`:
  - use `best_layer(metric="val_accuracy")`
- Build one steering vector:
  - `build_steering_vector(probe, sae, layer=<int>)`
- Build many steering vectors:
  - `build_steering_vectors(probes, sae_model, layers=[...])`

## Repo Pointers

- High-level API: `probekit/api.py`
- Probe model: `probekit/core/probe.py`
- Probe collections: `probekit/core/collection.py`
- Steering builder: `probekit/steering/builder.py`
