# spyx/__init__.py

import jax
import jax.numpy as jnp

from . import axn, data, experimental, fn, nir, nn, optimize, quant
from ._version import __version__

__all__ = [
    "jax",
    "jnp",
    "axn",
    "data",
    "experimental",
    "fn",
    "nir",
    "nn",
    "optimize",
    "quant",
    "__version__",
]
