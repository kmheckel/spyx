# spyx/__init__.py

import jax
import jax.numpy as jnp

from . import axn, data, experimental, fn, nir, nn, quant, ssm
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
    "quant",
    "ssm",
    "__version__",
]
