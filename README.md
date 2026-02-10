# Probes

A lightweight, modular library for training linear probes and steering vectors on neural network activations.

## Core Design (V2)

This library separates **Semantics** (the probe model) from **Fitting** (how it's learned).

### 1. The Models: `LinearProbe` and `ProbeCollection`
- **`LinearProbe`** (`probes.core.probe`): A container for a single probe (+ normalization stats).
- **`ProbeCollection`** (`probes.core.collection`): A container for a **batch** of probes.
    - `to_tensor()`: Stacks weights into `[B, D]` and biases into `[B]`.
    - `best_layer(metric)`: Finds the probe with the best validation accuracy.

### 2. The Fitters
Functional solvers in `probes.fitters` take training data and return a `LinearProbe` (or `ProbeCollection`).

- `fit_logistic`: Standard L2-regularized Logistic Regression.
- `fit_elastic_net`: ElasticNet (L1 + L2), useful for sparse features (SAEs, Neurons).
- `fit_dim`: Difference-in-Means (Class 1 Mean - Class 0 Mean).

#### Batched GPU Fitters
Optimized PyTorch implementations in `probes.fitters.batch` handle 3D inputs `[B, N, D]` efficiently on GPU:
- `fit_logistic_batch`: Batched IRLS solver.
- `fit_dim_batch`: Vectorized DiM with median thresholding.
- `fit_elastic_net_path`: Efficiently fits a regularization path (multiple alphas) using warm-starting.

## Quick Start

The high-level API automatically routes based on the input dimensions:

```python
from probes import sae_probe, dim_probe

# 1. Single Probe (X: [N, D], y: [N])
probe = sae_probe(X_2d, y_1d)

# 2. Batched Probes (X: [B, N, D], y: [B, N] or [N])
# Automatically uses GPU fitters and returns a ProbeCollection
probes = sae_probe(X_3d, y)
weights, biases = probes.to_tensor() # [B, D], [B]
```

## Steering Vectors

You can build steering vectors for individual probes or entire collections:

```python
from probes import build_steering_vector, build_steering_vectors

# Single
vec = build_steering_vector(probe, sae_model, layer=10)

# Batched (Maps layers to probes)
vecs = build_steering_vectors(probe_collection, sae_model, layers=[8, 9, 10])
```

## Structure

- `probes/core/`: `LinearProbe` and `ProbeCollection` definitions.
- `probes/fitters/`:
    - `logistic.py`, `elastic.py`, `dim.py`: Single-probe (CPU/sklearn) fitters.
    - `batch/`: Optimized GPU-batched fitters (IRLS, ISTA, DiM).
- `probes/api.py`: High-level aliases and dimension routing.
- `probes/steering/`: Tools for building steering vectors.
