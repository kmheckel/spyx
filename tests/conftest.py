"""Shared pytest fixtures and configuration for the Spyx test suite.

Keeps JAX on CPU during tests (deterministic, no accelerator required) and
provides a seeded ``nnx.Rngs`` fixture so individual tests don't each
re-seed by hand.
"""

from __future__ import annotations

import os

# Force CPU + deterministic behaviour before JAX is imported anywhere.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax  # noqa: E402
import pytest  # noqa: E402
from flax import nnx  # noqa: E402


@pytest.fixture
def rngs() -> nnx.Rngs:
    """A freshly seeded ``nnx.Rngs`` for constructing modules deterministically."""
    return nnx.Rngs(0)


@pytest.fixture
def key() -> jax.Array:
    """A deterministic PRNG key for tests that need raw ``jax.random`` draws."""
    return jax.random.PRNGKey(0)
