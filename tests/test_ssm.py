"""Tests for spyx.ssm."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from spyx import ssm

# ---------------------------------------------------------------------------
# associative-scan correctness
# ---------------------------------------------------------------------------


def test_diagonal_scan_matches_sequential_reference():
    """Parallel associative scan must match the naive lax.scan reference."""
    T, B, D = 32, 4, 16
    key = jax.random.PRNGKey(0)
    k_lam, k_Bu = jax.random.split(key)
    lam = (jax.random.uniform(k_lam, (D,)) - 0.5 + 1j * jax.random.uniform(k_lam, (D,)))
    lam = (lam * 0.9 / (jnp.abs(lam) + 1e-6)).astype(jnp.complex64)  # keep |λ| < 1
    Bu = jax.random.normal(k_Bu, (T, B, D), dtype=jnp.complex64)

    parallel = ssm._diagonal_scan(lam, Bu)
    sequential = ssm._diagonal_scan_reference(lam, Bu)

    assert jnp.allclose(parallel, sequential, atol=1e-4)


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------


def test_lru_forward_shape_and_dtype():
    rngs = nnx.Rngs(0)
    layer = ssm.LRU(d_model=8, d_state=16, rngs=rngs)
    u = jnp.ones((10, 3, 8))
    y = layer(u)
    assert y.shape == (10, 3, 8)
    assert y.dtype == jnp.float32


def test_lru_rejects_wrong_rank():
    rngs = nnx.Rngs(0)
    layer = ssm.LRU(d_model=4, d_state=8, rngs=rngs)
    with pytest.raises(ValueError, match=r"\[T, B, d_model\]"):
        layer(jnp.ones((10, 4)))


def test_lru_eigenvalues_are_stable():
    """The radial parameterisation must yield |λ| ≤ 1 at init."""
    rngs = nnx.Rngs(0)
    layer = ssm.LRU(d_model=4, d_state=32, rngs=rngs)
    lam, _, _ = layer._complex_matrices()
    assert jnp.all(jnp.abs(lam) <= 1.0 + 1e-5)


def test_lru_impulse_response_decays_for_short_sequence():
    """Feed a unit impulse; the magnitude of state should not explode."""
    rngs = nnx.Rngs(0)
    layer = ssm.LRU(d_model=1, d_state=8, rngs=rngs)
    T = 64
    u = jnp.zeros((T, 1, 1)).at[0, 0, 0].set(1.0)
    y = layer(u)
    assert jnp.all(jnp.isfinite(y))
    # Magnitude at t=T-1 should be less than at t=0 under stable dynamics.
    assert jnp.abs(y[-1]).sum() < jnp.abs(y[0]).sum() * 2.0


def test_lru_trains_on_copy_task():
    """Short copy task: y_t should match u_t after a few gradient steps."""
    rngs = nnx.Rngs(0)
    layer = ssm.LRU(d_model=4, d_state=8, rngs=rngs)
    optimizer = nnx.Optimizer(layer, optax.adam(5e-3), wrt=nnx.Param)

    T, B = 16, 8
    u = jax.random.normal(jax.random.PRNGKey(0), (T, B, 4))
    target = u  # identity copy

    @nnx.jit
    def step(model, optimizer, u, target):
        def loss_fn(m):
            return jnp.mean((m(u) - target) ** 2)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    initial = float(step(layer, optimizer, u, target))
    for _ in range(80):
        final = float(step(layer, optimizer, u, target))
    assert final < initial * 0.7, f"Loss did not decrease enough: {initial:.3f} -> {final:.3f}"


# ---------------------------------------------------------------------------
# S5Diag
# ---------------------------------------------------------------------------


def test_s5diag_forward_shape():
    rngs = nnx.Rngs(0)
    layer = ssm.S5Diag(d_model=8, d_state=16, rngs=rngs)
    y = layer(jnp.ones((12, 2, 8)))
    assert y.shape == (12, 2, 8)


def test_s5diag_hippo_init_is_stable():
    """HiPPO-LegS + positive dt gives |exp(A·dt)| < 1 for all states."""
    rngs = nnx.Rngs(0)
    layer = ssm.S5Diag(d_model=4, d_state=32, rngs=rngs)
    lam, _, _ = layer._complex_matrices()
    assert jnp.all(jnp.abs(lam) < 1.0)


# ---------------------------------------------------------------------------
# composition with the rest of Spyx
# ---------------------------------------------------------------------------


def test_ssm_composes_with_sequential_and_lif():
    """Hybrid SSM + spiking stack runs through spyx.nn.Sequential."""
    import spyx.nn as snn

    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(4, 8, use_bias=False, rngs=rngs),
        snn.LIF((8,), rngs=rngs),
    )
    ssm_layer = ssm.LRU(d_model=8, d_state=8, rngs=rngs)
    readout = nnx.Linear(8, 3, use_bias=False, rngs=rngs)

    # Forward pipeline: Spyx Sequential processes per-timestep, SSM processes the
    # whole sequence, readout collapses to logits.
    T, B = 6, 2
    u = jnp.ones((T, B, 4))
    spikes, _ = snn.run(model, u)
    y = ssm_layer(spikes)
    logits = readout(y.sum(axis=0))
    assert logits.shape == (B, 3)
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# quant integration
# ---------------------------------------------------------------------------


def _qwix_installed() -> bool:
    try:
        import qwix  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _qwix_installed(), reason="qwix not installed")
def test_ssm_can_be_quantized_with_spyx_quant_linear_rules():
    """spyx.quant should happily quantize the Linear layers around an SSM."""
    import spyx

    rngs = nnx.Rngs(0)
    # Quant applies to Linear modules; the SSM's own B/C projections are raw
    # params, not nnx.Linear, so they stay fp32 — matching the issue #39 plan
    # of keeping the state transition in full precision.
    model = nnx.Sequential(
        nnx.Linear(4, 8, use_bias=False, rngs=rngs),
        ssm.LRU(d_model=8, d_state=8, rngs=rngs),
        nnx.Linear(8, 3, use_bias=False, rngs=rngs),
    )
    T, B = 4, 2
    sample = jnp.ones((T, B, 4))
    # Wrapper needed because nnx.Sequential doesn't take our (u,) signature.

    class Wrapper(nnx.Module):
        def __init__(self, m):
            self.m = m

        def __call__(self, u):
            return self.m(u)

    wrapped = Wrapper(model)
    qwrapped = spyx.quant.quantize(wrapped, sample)
    y = qwrapped(sample)
    assert y.shape == (T, B, 3)
