"""
NELP Probes Package.

This package contains tools for training and using linear probekit on neural network activations.
"""

# Legacy imports removed (missing directory)
# Legacy imports removed
from probekit.api import dim_probe, logistic_probe, nelp_probe, sae_probe

# V2 Exports
from probekit.core.probe import LinearProbe
from probekit.fitters.dim import fit_dim
from probekit.fitters.elastic import fit_elastic_net
from probekit.fitters.logistic import fit_logistic
from probekit.steering import build_steering_vector, build_steering_vectors, load_probe
from probekit.utils.result import ProbeResult

__all__ = [
    "LinearProbe",
    "ProbeResult",
    "build_steering_vector",
    "build_steering_vectors",
    "dim_probe",
    "fit_dim",
    "fit_elastic_net",
    "fit_logistic",
    "load_probe",
    "logistic_probe",
    "nelp_probe",
    "sae_probe",
]
