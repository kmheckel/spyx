import jax
import jax.numpy as jnp
from flax import nnx

from spyx.experimental import SPSN, StochasticAssociativeLIF


def test_spsn_runs_finite_shape():
    """SPSN.__call__ produces finite spikes/membrane of the input shape."""
    rngs = nnx.Rngs(0)
    B, T, C = 3, 7, 5
    model = SPSN((C,), rngs=rngs)

    # beta init must not go negative (truncated_normal(0.25) + 0.5).
    assert jnp.all(model.beta[:] >= 0)

    key = jax.random.key(0)
    x = jax.random.normal(jax.random.key(1), (B, T, C))

    spikes, V = model(key, x)

    assert spikes.shape == (B, T, C)
    assert V.shape == (B, T, C)
    assert jnp.all(jnp.isfinite(spikes))
    assert jnp.all(jnp.isfinite(V))


def test_stochastic_associative_lif_runs_finite_shape():
    """StochasticAssociativeLIF.__call__ produces finite output of input shape."""
    rngs = nnx.Rngs(0)
    B, T, C = 4, 6, 8
    model = StochasticAssociativeLIF((C,), rngs=rngs)

    assert jnp.all(model.beta[:] >= 0)

    key = jax.random.key(2)
    x = jax.random.normal(jax.random.key(3), (B, T, C))

    spikes, V = model(key, x)

    assert spikes.shape == (B, T, C)
    assert V.shape == (B, T, C)
    assert jnp.all(jnp.isfinite(spikes))
    assert jnp.all(jnp.isfinite(V))


def test_spsn_membrane_matches_recurrence():
    """SPSN FFT membrane equals the explicit leaky recurrence V_t = beta*V_{t-1}+(1-beta)*x_t."""
    rngs = nnx.Rngs(5)
    B, T, C = 2, 9, 4
    model = SPSN((C,), rngs=rngs)

    x = jax.random.normal(jax.random.key(6), (B, T, C))
    _, V = model(jax.random.key(7), x)

    beta = jnp.clip(model.beta[:], 0, 1)
    ref = jnp.zeros((B, C))
    trace = []
    for t in range(T):
        ref = beta * ref + (1 - beta) * x[:, t]
        trace.append(ref)
    ref = jnp.stack(trace, axis=1)

    assert jnp.allclose(V, ref, atol=1e-4)


if __name__ == "__main__":
    test_spsn_runs_finite_shape()
    test_stochastic_associative_lif_runs_finite_shape()
    test_spsn_membrane_matches_recurrence()
    print("Tests passed!")
