"""Top-level exports for probekit."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from probekit.api import dim_probe, logistic_probe, nelp_probe, sae_probe
from probekit.core.probe import LinearProbe
from probekit.fitters.dim import fit_dim
from probekit.fitters.elastic import fit_elastic_net
from probekit.fitters.logistic import fit_logistic
from probekit.utils.result import ProbeResult

try:
    from probekit._version import version as __version__
except ImportError:
    try:
        __version__ = version("probekit")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "LinearProbe",
    "ProbeResult",
    "__version__",
    "build_steering_vector",
    "build_steering_vectors",
    "dim_probe",
    "fit_dim",
    "fit_elastic_net",
    "fit_logistic",
    "logistic_probe",
    "nelp_probe",
    "sae_probe",
]


def __getattr__(name: str) -> Any:
    """Lazy-load steering helpers to avoid importing heavy optional deps on import probekit."""
    if name in {"build_steering_vector", "build_steering_vectors"}:
        steering = import_module("probekit.steering")
        return getattr(steering, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
