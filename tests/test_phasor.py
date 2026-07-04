"""Tests for spyx.phasor."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from spyx import phasor


def test_real_to_phasor_round_trip_via_phase():
    """Encoding [0, 1) should produce phases [0, π).

    We exclude the right endpoint because e^{iπ} sits exactly on the branch
    cut and `jnp.angle` returns -π there (mathematically correct, just hostile
    to allclose).
    """
    x = jnp.linspace(0.0, 1.0, 6, endpoint=False)
    z = phasor.real_to_phasor(x)
    assert z.dtype == jnp.complex64
    assert jnp.allclose(jnp.abs(z), 1.0, atol=1e-6)
    assert jnp.allclose(phasor.phase_of(z), jnp.pi * x, atol=1e-6)


def test_phasor_linear_complex_dtype_and_shape():
    rngs = nnx.Rngs(0)
    layer = phasor.PhasorLinear(in_features=4, out_features=8, rngs=rngs)
    z = phasor.real_to_phasor(jnp.ones((2, 4)))
    out = layer(z)
    assert out.shape == (2, 8)
    assert out.dtype == jnp.complex64


def test_phasor_linear_rejects_real_input():
    rngs = nnx.Rngs(0)
    layer = phasor.PhasorLinear(in_features=4, out_features=2, rngs=rngs)
    with pytest.raises(TypeError, match="complex"):
        layer(jnp.ones((1, 4)))


def test_phasor_activation_unit_magnitude():
    layer = phasor.PhasorActivation()
    z = jnp.array([1 + 1j, 0.1 + 0.0j, -2.0 + 0.5j], dtype=jnp.complex64)
    out = layer(z)
    assert jnp.allclose(jnp.abs(out), 1.0, atol=1e-5)


def test_phasor_mlp_forward_shape_and_finite_grads():
    """A tiny PhasorMLP should produce real logits and finite gradients on a synthetic task."""
    rngs = nnx.Rngs(0)
    model = phasor.PhasorMLP(
        in_features=8, hidden_features=16, out_features=3, depth=2, rngs=rngs
    )
    x = jax.random.uniform(jax.random.PRNGKey(1), (4, 8))
    targets = jnp.array([0, 1, 2, 0])

    logits = model(x)
    assert logits.shape == (4, 3)
    assert logits.dtype == jnp.float32

    def loss_fn(m):
        return optax.softmax_cross_entropy_with_integer_labels(m(x), targets).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    assert jnp.isfinite(loss)
    # Recursively check that every param-leaf in `grads` is finite.
    flat_grads = jax.tree_util.tree_leaves(grads)
    assert flat_grads, "Expected at least one parameter gradient."
    for g in flat_grads:
        assert jnp.all(jnp.isfinite(g))


def test_phasor_mlp_training_reduces_loss():
    """A linearly-separable toy task should be learnable after the real/imag split.

    Before the real/imag parameter split, a stock optax loop couldn't
    converge on phasor weights because JAX's conjugate-Wirtinger gradient
    fights optax's real-arithmetic assumption. With ``kernel_re`` +
    ``kernel_im`` stored separately the fix is structural and this test
    serves as the regression guard.
    """
    rngs = nnx.Rngs(0)
    model = phasor.PhasorMLP(
        in_features=4, hidden_features=16, out_features=2, depth=2, rngs=rngs
    )
    optimizer = nnx.Optimizer(model, optax.adam(5e-3), wrt=nnx.Param)

    x = jax.random.uniform(jax.random.PRNGKey(2), (64, 4))
    y = (x[:, 0] > 0.5).astype(jnp.int32)

    @nnx.jit
    def step(model, optimizer, x, y):
        def loss_fn(m):
            return optax.softmax_cross_entropy_with_integer_labels(m(x), y).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    initial = float(step(model, optimizer, x, y))
    for _ in range(200):
        final = float(step(model, optimizer, x, y))
    assert jnp.isfinite(final), final
    assert final < initial * 0.6, (
        f"Loss did not drop enough: {initial:.3f} -> {final:.3f}"
    )


# ---------------------------------------------------------------------------
# spike <-> phase conversion
# ---------------------------------------------------------------------------


def test_phase_to_spikes_emits_one_spike_per_neuron():
    theta = jnp.array([-jnp.pi + 1e-3, -1.0, 0.0, 1.0, jnp.pi - 1e-3])
    spikes = phasor.phase_to_spikes(theta, T=8)
    assert spikes.shape == (8, 5)
    # Each neuron should fire exactly once.
    assert jnp.all(jnp.sum(spikes, axis=0) == 1)


def test_phase_to_spikes_then_spikes_to_phase_round_trip_within_quantisation_error():
    theta = jnp.linspace(-jnp.pi + 1e-3, jnp.pi - 1e-3, 32)
    T = 64
    spikes = phasor.phase_to_spikes(theta, T)
    recovered = phasor.spikes_to_phase(spikes, T)
    # Quantisation step is 2π/T; allow up to one bin of error.
    tol = 2.0 * jnp.pi / T
    assert jnp.max(jnp.abs(recovered - theta)) < tol


def test_spikes_to_phase_handles_silent_neurons():
    spikes = jnp.zeros((8, 3), dtype=jnp.float32)
    out = phasor.spikes_to_phase(spikes)
    assert out.shape == (3,)
    # Convention: silent neurons map to phase 0 (start of cycle).
    assert jnp.allclose(out, -jnp.pi)


def test_spiking_phasor_wrapper_runs_end_to_end():
    rngs = nnx.Rngs(0)
    layer = phasor.PhasorLinear(in_features=4, out_features=6, rngs=rngs)
    sp = phasor.SpikingPhasor(layer, period_T=16)
    # Build an input spike train: random phases -> spikes.
    theta_in = jax.random.uniform(
        jax.random.PRNGKey(0), (3, 4), minval=-jnp.pi, maxval=jnp.pi
    )
    spikes_in = phasor.phase_to_spikes(theta_in, T=16)  # [16, 3, 4]
    spikes_out = sp(spikes_in)
    assert spikes_out.shape == (16, 3, 6)
    # Output is also one-spike-per-cycle.
    assert jnp.all(jnp.sum(spikes_out, axis=0) == 1)
