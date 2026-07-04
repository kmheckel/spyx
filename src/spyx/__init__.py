# spyx/__init__.py

import jax
import jax.numpy as jnp

from . import (
    axn,
    bench,
    data,
    experimental,
    fn,
    nir,
    nn,
    optimize,
    phasor,
    quant,
    ssm,
)
from ._version import __version__

__all__ = [
    "jax",
    "jnp",
    "axn",
    "bench",
    "data",
    "experimental",
    "fn",
    "nir",
    "nn",
    "optimize",
    "phasor",
    "quant",
    "ssm",
    "__version__",
]
