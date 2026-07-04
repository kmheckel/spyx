import jax
import jax.numpy as jnp
from flax import nnx

from spyx import nn
from spyx.experimental import PSU_LIF


def test_contract_shapes_and_run():
    """__call__ / initial_state shapes and use inside spyx.nn.run."""
    rngs = nnx.Rngs(0)
    hidden_shape = (10,)
    model = PSU_LIF(hidden_shape, rngs=rngs)

    x = jnp.ones((5, 10))  # batch of 5
    V = model.initial_state(5)
    assert V.shape == (5, 10)

    spikes, V_next = model(x, V)
    assert spikes.shape == (5, 10)
    assert V_next.shape == (5, 10)

    # Drop-in for spyx.nn.run (time-major scan of __call__).
    T, B = 7, 5
    xt = jnp.ones((T, B, 10))
    outs, final_state = nn.run(model, xt)
    assert outs.shape == (T, B, 10)
    assert final_state.shape == (B, 10)


def test_run_inside_sequential():
    """PSU_LIF works as a stateful layer in Sequential + run."""
    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nnx.Linear(8, 4, rngs=rngs),
        PSU_LIF((4,), rngs=rngs),
    )
    T, B = 6, 3
    x = jnp.ones((T, B, 8))
    outs, final_state = nn.run(model, x)
    assert outs.shape == (T, B, 4)
    assert len(final_state) == 2
    assert final_state[1].shape == (B, 4)


def _sequential_spikes(model, x):
    """Reference: scan __call__ over time-major x, collect spikes."""
    spikes, _ = nn.run(model, x)
    return spikes


def test_parallel_equals_sequential_per_unit_beta():
    """CRITICAL: parallel(x) spikes == sequential scan of __call__ (per-unit beta)."""
    rngs = nnx.Rngs(1)
    hidden_shape = (12,)
    model = PSU_LIF(hidden_shape, rngs=rngs)

    T, B = 20, 4
    key = jax.random.key(0)
    x = jax.random.normal(key, (T, B, 12))

    par = model.parallel(x)
    seq = _sequential_spikes(model, x)

    assert par.shape == (T, B, 12)
    assert jnp.allclose(par, seq, atol=1e-5)


def test_parallel_equals_sequential_scalar_beta():
    """Equivalence also holds with a scalar beta."""
    rngs = nnx.Rngs(2)
    model = PSU_LIF((8,), beta=0.9, rngs=rngs)

    # scalar beta stored as a 0-d param
    assert model.beta[...].shape == ()

    T, B = 15, 3
    x = jax.random.normal(jax.random.key(3), (T, B, 8))

    par = model.parallel(x)
    seq = _sequential_spikes(model, x)
    assert jnp.allclose(par, seq, atol=1e-5)


def test_parallel_membrane_matches_recurrence():
    """The associative-scan membrane matches an explicit python recurrence."""
    rngs = nnx.Rngs(4)
    model = PSU_LIF((5,), beta=0.75, rngs=rngs)

    T, B = 10, 2
    x = jax.random.normal(jax.random.key(5), (T, B, 5))
    beta = jnp.clip(model.beta[...], 0, 1)

    V = jnp.zeros((B, 5))
    ref = []
    for t in range(T):
        V = beta * V + x[t]
        ref.append(V)
    ref = jnp.stack(ref)

    # Recover the parallel membrane trace via the scan directly.
    A = jnp.broadcast_to(beta, x.shape)
    _, V_par = jax.lax.associative_scan(nn._leaky_associative_op, (A, x), axis=0)
    assert jnp.allclose(V_par, ref, atol=1e-5)


def test_gradients_finite_and_nonzero():
    """Loss on parallel(x) has finite, nonzero grads w.r.t. beta and a Linear."""
    rngs = nnx.Rngs(6)
    linear = nnx.Linear(6, 9, rngs=rngs)
    neuron = PSU_LIF((9,), rngs=rngs)

    T, B = 12, 4
    x = jax.random.normal(jax.random.key(7), (T, B, 6))

    def loss_fn(linear, neuron):
        # Apply the Linear per timestep, then the parallel spiking scan.
        proj = jax.vmap(linear)(x)  # (T, B, 9)
        spikes = neuron.parallel(proj)
        return jnp.sum(spikes)

    grads = nnx.grad(loss_fn, argnums=(0, 1))(linear, neuron)
    lin_grads, neuron_grads = grads

    beta_g = neuron_grads["beta"][...]
    w_g = lin_grads["kernel"][...]

    assert jnp.all(jnp.isfinite(beta_g))
    assert jnp.all(jnp.isfinite(w_g))
    assert jnp.any(beta_g != 0)
    assert jnp.any(w_g != 0)


if __name__ == "__main__":
    test_contract_shapes_and_run()
    test_run_inside_sequential()
    test_parallel_equals_sequential_per_unit_beta()
    test_parallel_equals_sequential_scalar_beta()
    test_parallel_membrane_matches_recurrence()
    test_gradients_finite_and_nonzero()
    print("Tests passed!")
