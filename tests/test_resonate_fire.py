"""Tests for spyx.phasor.ResonateFire, the resonate-and-fire spiking neuron.

ResonateFire is the complex/oscillatory sibling of spyx.nn.PSU_LIF: a reset-free
complex linear recurrence ``z_t = a * z_{t-1} + x_t`` with an oscillator pole
``a = exp(dt(-lambda + i*omega))`` that is exactly parallelizable via
``jax.lax.associative_scan``. These tests pin the neuron contract, the critical
sequential/parallel equivalence, gradient flow, oscillation sanity, and pole
stability.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import spyx.nn as snn
from spyx.experimental import ResonateFire


def _membrane_trace(neuron, x):
    """Collect the complex membrane trace z_t by scanning __call__ in Python.

    :x: real input with shape [Time, Batch, hidden].
    :return: complex array [Time, Batch, hidden] of membrane states.
    """
    z = neuron.initial_state(x.shape[1])
    states = []
    for t in range(x.shape[0]):
        _, z = neuron(x[t], z)
        states.append(z)
    return jnp.stack(states, axis=0)


def test_contract_shapes_dtypes_and_run():
    """__call__ shapes/dtypes, complex initial_state, and spyx.nn.run."""
    hidden = 5
    batch = 3
    T = 7
    neuron = ResonateFire(hidden_shape=(hidden,), rngs=nnx.Rngs(0))

    z0 = neuron.initial_state(batch)
    assert z0.shape == (batch, hidden)
    assert z0.dtype == jnp.complex64
    assert jnp.all(z0 == 0)

    x_t = jax.random.normal(jax.random.PRNGKey(1), (batch, hidden))
    spikes, z1 = neuron(x_t, z0)
    assert spikes.shape == (batch, hidden)
    assert z1.shape == (batch, hidden)
    assert z1.dtype == jnp.complex64
    # spikes are real (surrogate heaviside on Re(z)).
    assert not jnp.iscomplexobj(spikes)

    x = jax.random.normal(jax.random.PRNGKey(2), (T, batch, hidden))
    outputs, final_state = snn.run(neuron, x)
    assert outputs.shape == (T, batch, hidden)
    assert final_state.shape == (batch, hidden)
    assert final_state.dtype == jnp.complex64


def test_parallel_equals_sequential():
    """CRITICAL: parallel(x) spikes == sequential scan of __call__ over x."""
    hidden = 6
    batch = 4
    T = 25
    neuron = ResonateFire(hidden_shape=(hidden,), threshold=0.5, rngs=nnx.Rngs(7))

    x = 0.6 * jax.random.normal(jax.random.PRNGKey(11), (T, batch, hidden))

    seq_spikes, _ = snn.run(neuron, x)
    par_spikes = neuron.parallel(x)

    assert par_spikes.shape == seq_spikes.shape
    # The underlying complex recurrence is identical; the two spike paths must
    # agree exactly (both are 0/1 surrogate-heaviside outputs).
    np.testing.assert_allclose(
        np.asarray(par_spikes), np.asarray(seq_spikes), atol=1e-5
    )

    # And the real membrane traces underlying both paths must match tightly.
    seq_trace = _membrane_trace(neuron, x)
    a = neuron.a
    A = jnp.broadcast_to(a, x.astype(jnp.complex64).shape)

    def op(ei, ej):
        ai, bi = ei
        aj, bj = ej
        return aj * ai, aj * bi + bj

    _, par_trace = jax.lax.associative_scan(op, (A, x.astype(jnp.complex64)), axis=0)
    np.testing.assert_allclose(
        np.asarray(jnp.real(par_trace)),
        np.asarray(jnp.real(seq_trace)),
        atol=1e-4,
        rtol=1e-4,
    )


def test_gradients_flow():
    """Gradients flow through lambda, omega, and an upstream nnx.Linear."""
    in_features = 4
    hidden = 5
    batch = 2
    T = 12

    class Net(nnx.Module):
        def __init__(self, *, rngs):
            self.linear = nnx.Linear(in_features, hidden, rngs=rngs)
            self.neuron = ResonateFire(hidden_shape=(hidden,), rngs=rngs)

        def __call__(self, x):
            # x: [T, B, in] -> per-timestep linear -> parallel resonate-and-fire.
            cur = jax.vmap(self.linear)(x)
            return self.neuron.parallel(cur)

    model = Net(rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(3), (T, batch, in_features))

    def loss_fn(m):
        return jnp.sum(m(x))

    grads = nnx.grad(loss_fn)(model)

    g_lambda = grads.neuron.raw_lambda[...]
    g_omega = grads.neuron.omega[...]
    g_kernel = grads.linear.kernel[...]

    for name, g in (("raw_lambda", g_lambda), ("omega", g_omega), ("kernel", g_kernel)):
        assert g is not None, f"missing grad for {name}"
        assert jnp.all(jnp.isfinite(g)), f"non-finite grad for {name}"
        assert jnp.any(g != 0), f"zero grad for {name}"


def test_oscillation_sanity():
    """A single impulse yields a decaying oscillation in the membrane."""
    T = 40
    batch = 1
    hidden = 1
    neuron = ResonateFire(
        hidden_shape=(hidden,),
        lambda_init=0.05,
        omega_init=1.0,
        dt=1.0,
        threshold=10.0,  # keep spiking irrelevant; probe the subthreshold ringing
        rngs=nnx.Rngs(0),
    )

    x = jnp.zeros((T, batch, hidden))
    x = x.at[0, 0, 0].set(1.0)  # impulse then silence

    trace = _membrane_trace(neuron, x)[:, 0, 0]  # [T] complex
    re = jnp.real(trace)
    mag = jnp.abs(trace)

    # Oscillation: Re(z) changes sign as the phasor rotates at ~omega.
    sign_changes = jnp.sum(jnp.abs(jnp.diff(jnp.sign(re))) > 0)
    assert int(sign_changes) >= 2

    # Decay: the magnitude peaks early and is smaller by the end.
    peak = jnp.max(mag)
    assert float(mag[-1]) < float(peak)
    # Monotone-ish envelope decay: |z| after the impulse is bounded by |a|^t.
    assert float(mag[-1]) < 0.5 * float(peak)


def test_pole_stability():
    """|a| <= 1 for the initialised (random) parameters."""
    neuron = ResonateFire(hidden_shape=(64,), rngs=nnx.Rngs(123))
    a = neuron.a
    assert a.dtype == jnp.complex64
    assert jnp.all(jnp.abs(a) <= 1.0 + 1e-6)
    # softplus decay is strictly non-negative, so poles are inside the unit disk.
    assert jnp.all(neuron.decay >= 0.0)
