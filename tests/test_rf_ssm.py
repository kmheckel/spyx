"""Tests for spyx.experimental.rf_ssm.RFSSM — Resonate-and-Fire spiking SSM.

RFSSM is the S5/HiPPO-initialised, PRF-decoupled-reset sibling of
spyx.phasor.ResonateFire. The reset is folded onto the *imaginary* axis as a
state-independent additive drive, so the complex membrane recurrence
``z_t = a * z_{t-1} + (x_t + i b)`` stays linear and its ``.parallel``
associative scan is *exactly* equivalent to the sequential ``__call__`` — even
with the reset engaged. These tests pin that scan-exactness, the neuron
contract, gradient flow, and pole stability of both init modes.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import spyx.nn as snn
from spyx.experimental.rf_ssm import RFSSM, ResonateFireSSM


def _seq_membrane_trace(neuron, x):
    """Complex membrane trace z_t from scanning __call__ in Python.

    :x: real input [Time, Batch, hidden].
    :return: complex [Time, Batch, hidden].
    """
    z = neuron.initial_state(x.shape[1])
    states = []
    for t in range(x.shape[0]):
        _, z = neuron(x[t], z)
        states.append(z)
    return jnp.stack(states, axis=0)


def test_contract_shapes_dtypes_and_run():
    """__call__ shapes/dtypes, complex initial_state, and drop-in spyx.nn.run."""
    hidden, batch, T = 5, 3, 7
    neuron = RFSSM(hidden_shape=(hidden,), rngs=nnx.Rngs(0))

    z0 = neuron.initial_state(batch)
    assert z0.shape == (batch, hidden)
    assert z0.dtype == jnp.complex64
    assert jnp.all(z0 == 0)

    x_t = jax.random.normal(jax.random.PRNGKey(1), (batch, hidden))
    spikes, z1 = neuron(x_t, z0)
    assert spikes.shape == (batch, hidden)
    assert z1.shape == (batch, hidden)
    assert z1.dtype == jnp.complex64
    assert not jnp.iscomplexobj(spikes)

    x = jax.random.normal(jax.random.PRNGKey(2), (T, batch, hidden))
    outputs, final_state = snn.run(neuron, x)
    assert outputs.shape == (T, batch, hidden)
    assert final_state.shape == (batch, hidden)
    assert final_state.dtype == jnp.complex64


def test_alias_is_rfssm():
    """ResonateFireSSM is an alias for RFSSM."""
    assert ResonateFireSSM is RFSSM


def _max_mismatch(neuron, x):
    """Return (spike_mismatch, real_trace_mismatch) between seq and parallel."""
    seq_spikes, _ = snn.run(neuron, x)
    par_spikes = neuron.parallel(x)
    spike_gap = float(jnp.max(jnp.abs(par_spikes - seq_spikes)))

    seq_trace = _seq_membrane_trace(neuron, x)
    a = neuron.a
    drive = neuron._drive(x)
    A = jnp.broadcast_to(a, drive.shape)

    def op(ei, ej):
        ai, bi = ei
        aj, bj = ej
        return aj * ai, aj * bi + bj

    _, par_trace = jax.lax.associative_scan(op, (A, drive), axis=0)
    trace_gap = float(jnp.max(jnp.abs(jnp.real(par_trace) - jnp.real(seq_trace))))
    return spike_gap, trace_gap


def test_parallel_equals_sequential_reset_free():
    """CRITICAL: parallel == sequential scan with the reset off (b == 0)."""
    neuron = RFSSM(hidden_shape=(6,), reset_init=0.0, threshold=0.5, rngs=nnx.Rngs(7))
    x = 0.6 * jax.random.normal(jax.random.PRNGKey(11), (25, 4, 6))
    spike_gap, trace_gap = _max_mismatch(neuron, x)
    assert spike_gap <= 1e-5, spike_gap
    assert trace_gap <= 1e-4, trace_gap


def test_parallel_equals_sequential_with_decoupled_reset():
    """CRITICAL: the decoupled reset keeps the scan exact (b != 0)."""
    neuron = RFSSM(hidden_shape=(6,), reset_init=0.75, threshold=0.5, rngs=nnx.Rngs(7))
    # sanity: the reset is actually engaged.
    assert float(jnp.max(jnp.abs(neuron.reset[...]))) > 0.0
    x = 0.6 * jax.random.normal(jax.random.PRNGKey(11), (25, 4, 6))
    spike_gap, trace_gap = _max_mismatch(neuron, x)
    assert spike_gap <= 1e-5, spike_gap
    assert trace_gap <= 1e-4, trace_gap


def test_parallel_equals_sequential_lru_init():
    """Scan-exactness holds for the LRU pole init too."""
    neuron = RFSSM(
        hidden_shape=(8,),
        pole_init="lru",
        reset_init=0.3,
        threshold=0.4,
        rngs=nnx.Rngs(3),
    )
    x = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (30, 2, 8))
    spike_gap, trace_gap = _max_mismatch(neuron, x)
    assert spike_gap <= 1e-5, spike_gap
    assert trace_gap <= 1e-4, trace_gap


def test_hippo_pole_init_sanity():
    """HiPPO init: poles inside the unit disk, omega spans the LegS spectrum."""
    hidden = 16
    neuron = RFSSM(hidden_shape=(hidden,), pole_init="hippo", rngs=nnx.Rngs(0))
    a = neuron.a
    assert a.dtype == jnp.complex64
    # Stable: every pole strictly inside (or on) the unit circle.
    assert jnp.all(jnp.abs(a) <= 1.0 + 1e-6)
    assert jnp.all(neuron.decay >= 0.0)
    assert jnp.all(neuron.step > 0.0)
    # HiPPO-LegS continuous real part is -1/2 everywhere at init.
    np.testing.assert_allclose(np.asarray(neuron.decay), 0.5, atol=1e-5)
    # omega_n = pi * n spans a wide band (units resonate at distinct frequencies).
    omega = np.asarray(neuron.omega[...])
    assert np.isclose(omega[0], 0.0, atol=1e-5)
    assert np.isclose(omega[-1], np.pi * (hidden - 1), atol=1e-4)


def test_lru_pole_init_stability():
    """LRU init: poles inside the unit disk within the requested magnitude band."""
    neuron = RFSSM(
        hidden_shape=(64,),
        pole_init="lru",
        r_min=0.4,
        r_max=0.99,
        rngs=nnx.Rngs(123),
    )
    a = neuron.a
    mag = jnp.abs(a)
    assert jnp.all(mag <= 1.0 + 1e-6)
    assert jnp.all(neuron.decay >= 0.0)
    # With dt == 1 the pole magnitude should sit inside the sampled band.
    assert jnp.all(mag >= 0.4 - 1e-2)
    assert jnp.all(mag <= 0.99 + 1e-2)


def test_gradients_flow():
    """Gradients flow through raw_lambda, omega, log_dt, reset, and upstream W."""
    in_features, hidden, batch, T = 4, 5, 2, 12

    class Net(nnx.Module):
        def __init__(self, *, rngs):
            self.linear = nnx.Linear(in_features, hidden, rngs=rngs)
            self.neuron = RFSSM(hidden_shape=(hidden,), reset_init=0.5, rngs=rngs)

        def __call__(self, x):
            cur = jax.vmap(self.linear)(x)
            return self.neuron.parallel(cur)

    model = Net(rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(3), (T, batch, in_features))

    grads = nnx.grad(lambda m: jnp.sum(m(x)))(model)
    checks = {
        "raw_lambda": grads.neuron.raw_lambda[...],
        "omega": grads.neuron.omega[...],
        "log_dt": grads.neuron.log_dt[...],
        "reset": grads.neuron.reset[...],
        "kernel": grads.linear.kernel[...],
    }
    for name, g in checks.items():
        assert g is not None, f"missing grad for {name}"
        assert jnp.all(jnp.isfinite(g)), f"non-finite grad for {name}"
        assert jnp.any(g != 0), f"zero grad for {name}"


def test_invalid_pole_init_raises():
    """An unknown pole_init is rejected with a clear error."""
    try:
        RFSSM(hidden_shape=(4,), pole_init="nope", rngs=nnx.Rngs(0))
    except ValueError as e:
        assert "pole_init" in str(e)
    else:
        raise AssertionError("expected ValueError for bad pole_init")
