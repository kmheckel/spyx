"""Tests for the reset-preserving parallel LIF (FPT scan).

The neuron keeps the *exact* hard reset of :class:`spyx.nn.LIF` while replacing
the sequential time loop with a fixed-point-threshold (FPT) associative scan.
Two properties are checked:

* **Exactness in the limit** -- with ``K >= Time`` iterations the FPT
  ``.parallel`` path reproduces the sequential hard-reset spike train *exactly*
  (0 mismatched spikes), across random inputs / beta / threshold.
* **Convergence in K** -- the mismatch decreases with ``K``; a small ``K`` (=3)
  is near-exact in the short-cascade (sparse / low-beta) regime FPT targets and
  strictly better than ``K = 1``.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from spyx import nn
from spyx.experimental.parallel_reset import ParallelResetLIF


def _sequential_spikes(model, x):
    """Reference: run the neuron's own hard-reset ``__call__`` step by step."""
    outputs, _ = nn.run(model, x)
    return outputs


def _mismatch_fraction(a, b):
    return float(jnp.mean(jnp.abs(a - b)))


def test_call_matches_nn_lif_exactly():
    """The sequential ``__call__`` is byte-for-byte :class:`spyx.nn.LIF`."""
    hidden = (16,)
    ref = nn.LIF(hidden, beta=0.8, threshold=1.0, rngs=nnx.Rngs(0))
    prl = ParallelResetLIF(hidden, beta=0.8, threshold=1.0, rngs=nnx.Rngs(0))

    x = jax.random.normal(jax.random.PRNGKey(1), (32, 4, 16)) * 1.5
    ref_spikes = _sequential_spikes(ref, x)
    prl_spikes = _sequential_spikes(prl, x)
    assert jnp.array_equal(ref_spikes, prl_spikes)


@pytest.mark.parametrize("seed", range(6))
def test_parallel_exact_at_K_equals_T(seed):
    """``.parallel(x, K=T)`` reproduces the sequential spike train exactly.

    The FPT correctness wavefront advances one timestep per iteration, so
    ``K = Time`` is exact regardless of ``beta`` / ``threshold`` / input scale.
    """
    T = 24
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    beta = float(jax.random.uniform(keys[0], (), minval=0.1, maxval=0.99))
    threshold = float(jax.random.uniform(keys[1], (), minval=0.3, maxval=1.5))
    scale = float(jax.random.uniform(keys[2], (), minval=0.5, maxval=2.0))
    hidden = (16,)

    model = ParallelResetLIF(hidden, beta=beta, threshold=threshold, rngs=nnx.Rngs(0))
    x = jax.random.normal(keys[3], (T, 8, 16)) * scale

    seq = _sequential_spikes(model, x)
    par = model.parallel(x, K=T)
    # Exact: zero mismatched spike decisions.
    assert jnp.array_equal(seq, par)


def test_parallel_K3_tight_in_sparse_regime():
    """K=3 is near-exact (>99% of spikes) in the short-cascade regime.

    Sparse pulse drive with a fast leak (``beta=0.4``) keeps reset cascades
    short, which is where the FPT few-iteration approximation is meant to hold.
    """
    hidden = (32,)
    model = ParallelResetLIF(hidden, beta=0.4, threshold=1.0, rngs=nnx.Rngs(0))

    worst = 0.0
    for seed in range(20):
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        pulses = (jax.random.uniform(k1, (48, 8, 32)) < 0.12).astype(jnp.float32) * 1.7
        x = pulses + jax.random.normal(k2, (48, 8, 32)) * 0.1
        seq = _sequential_spikes(model, x)
        par = model.parallel(x, K=3)
        worst = max(worst, _mismatch_fraction(seq, par))
    # >99% of spike decisions agree at K=3 in this regime.
    assert worst < 1e-2


def test_parallel_converges_with_K():
    """Mismatch decreases with K; K=3 is far tighter than K=1.

    Uses a moderate regime (``beta=0.5``, sparse drive) where the FPT iteration
    converges cleanly, then checks the whole K-vs-error curve is monotone and
    that K=3 already meets a tight tolerance.
    """
    hidden = (32,)
    model = ParallelResetLIF(hidden, beta=0.5, threshold=1.0, rngs=nnx.Rngs(0))

    errs = {K: 0.0 for K in (1, 2, 3, 5)}
    n = 20
    for seed in range(n):
        k1, k2 = jax.random.split(jax.random.PRNGKey(100 + seed))
        pulses = (jax.random.uniform(k1, (48, 8, 32)) < 0.12).astype(jnp.float32) * 1.7
        x = pulses + jax.random.normal(k2, (48, 8, 32)) * 0.1
        seq = _sequential_spikes(model, x)
        for K in errs:
            errs[K] += _mismatch_fraction(seq, model.parallel(x, K=K)) / n

    # Monotone non-increasing error in K.
    assert errs[1] >= errs[2] >= errs[3] >= errs[5]
    # K=3 is strictly better than K=1 and already tight.
    assert errs[3] < errs[1]
    assert errs[3] < 1e-3


def test_parallel_drops_into_sequential_contract():
    """The neuron satisfies the ``(x, state) -> (out, new_state)`` contract."""
    hidden = (8,)
    model = ParallelResetLIF(hidden, beta=0.7, rngs=nnx.Rngs(0))
    state = model.initial_state(4)
    x = jnp.ones((4, 8))
    spikes, new_state = model(x, state)
    assert spikes.shape == (4, 8)
    assert new_state.shape == state.shape
    assert isinstance(model, nn.StatefulLayer)
