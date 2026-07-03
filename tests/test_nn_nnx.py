import jax.numpy as jnp
from flax import nnx

from spyx import nn


def test_lif():
    rngs = nnx.Rngs(0)
    hidden_shape = (10,)
    model = nn.LIF(hidden_shape, rngs=rngs)

    x = jnp.ones((5, 10))  # batch of 5
    V = model.initial_state(5)

    spikes, V_next = model(x, V)

    assert spikes.shape == (5, 10)
    assert V_next.shape == (5, 10)
    assert jnp.any(
        spikes == 0
    )  # Initially no spikes if threshold is 1 and dynamic V starts at 0


def test_rlif():
    rngs = nnx.Rngs(0)
    hidden_shape = (10,)
    model = nn.RLIF(hidden_shape, rngs=rngs)

    x = jnp.ones((5, 10))
    V = model.initial_state(5)

    spikes, V_next = model(x, V)

    assert spikes.shape == (5, 10)
    assert V_next.shape == (5, 10)


def test_activity_reg():
    spikes = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
    model = nn.ActivityRegularization(hidden_shape=(2,), batch_size=2)

    out = model(spikes)
    assert jnp.array_equal(out, spikes)
    assert jnp.array_equal(model.spike_count[...], spikes)

    model(spikes)
    assert jnp.array_equal(model.spike_count[...], spikes * 2)

    model.reset(batch_size=2)
    assert jnp.array_equal(model.spike_count[...], jnp.zeros((2, 2), dtype=jnp.float32))


def test_activity_reg_under_jit():
    spikes = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
    model = nn.ActivityRegularization(hidden_shape=(2,), batch_size=2)

    @nnx.jit
    def step(m, s):
        return m(s)

    out = step(model, spikes)
    assert jnp.array_equal(out, spikes)
    assert jnp.array_equal(model.spike_count[...], spikes)


def test_if():
    model = nn.IF((10,))
    x = jnp.ones((5, 10)) * 2.0
    V = model.initial_state(5)
    # First step: V starts at 0, no spike, V accumulates the input.
    spikes, V_next = model(x, V)
    assert spikes.shape == (5, 10)
    assert V_next.shape == (5, 10)
    assert jnp.all(spikes == 0)
    assert jnp.allclose(V_next, 2.0)
    # Second step: V (2.0) exceeds threshold 1, every neuron fires.
    spikes2, _ = model(x, V_next)
    assert jnp.all(spikes2 == 1)


def test_alif():
    rngs = nnx.Rngs(0)
    model = nn.ALIF((10,), rngs=rngs)
    x = jnp.ones((5, 10))
    state = model.initial_state(5)
    spikes, new_state = model(x, state)
    assert spikes.shape == (5, 10)
    # ALIF state shape doubles to track adaptive threshold.
    assert new_state.shape == state.shape


def test_cubalif():
    rngs = nnx.Rngs(0)
    model = nn.CuBaLIF((10,), rngs=rngs)
    x = jnp.ones((5, 10))
    state = model.initial_state(5)
    spikes, new_state = model(x, state)
    assert spikes.shape == (5, 10)
    # CuBaLIF state stacks V and I along the last axis.
    assert new_state.shape == (5, 20)


def test_rif():
    rngs = nnx.Rngs(0)
    model = nn.RIF((10,), rngs=rngs)
    x = jnp.ones((5, 10))
    V = model.initial_state(5)
    spikes, V_next = model(x, V)
    assert spikes.shape == (5, 10)
    assert V_next.shape == (5, 10)


def test_rcubalif():
    rngs = nnx.Rngs(0)
    model = nn.RCuBaLIF((10,), rngs=rngs)
    x = jnp.ones((5, 10))
    state = model.initial_state(5)
    spikes, new_state = model(x, state)
    assert spikes.shape == (5, 10)
    assert new_state.shape == (5, 20)


def test_li():
    rngs = nnx.Rngs(0)
    model = nn.LI((10,), rngs=rngs)
    x = jnp.ones((5, 10))
    V = model.initial_state(5)
    out, V_next = model(x, V)
    assert out.shape == (5, 10)
    assert V_next.shape == (5, 10)
    # LI is non-spiking; output equals new voltage trace.
    assert jnp.allclose(out, V_next)


def test_sumpool():
    # NHWC layout: pool with int window/stride; channel_axis is excluded from pooling.
    pool = nn.SumPool(window_shape=2, strides=2, padding="VALID", channel_axis=-1)
    x = jnp.ones((1, 4, 4, 3))
    y = pool(x)
    assert y.shape == (1, 2, 2, 3)
    assert jnp.all(y == 4.0)


def test_run_helper_time_major():
    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nnx.Linear(8, 4, rngs=rngs),
        nn.LIF((4,), rngs=rngs),
    )
    T, B = 6, 3
    x = jnp.ones((T, B, 8))
    outs, final_state = nn.run(model, x)
    assert outs.shape == (T, B, 4)
    # final_state corresponds to one entry per layer in the Sequential.
    assert len(final_state) == 2
    assert final_state[1].shape == (B, 4)


def test_sequential_run():
    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nnx.Linear(10, 20, rngs=rngs),
        nn.LIF((20,), rngs=rngs),
        nnx.Linear(20, 10, rngs=rngs),
        nn.LIF((10,), rngs=rngs),
    )

    x = jnp.ones((32, 5, 10))  # [T, B, C]

    outputs, final_state = nn.run(model, x)

    assert outputs.shape == (32, 5, 10)
    assert len(final_state) == 4
    assert final_state[1].shape == (5, 20)
    assert final_state[3].shape == (5, 10)


if __name__ == "__main__":
    test_lif()
    test_rlif()
    test_activity_reg()
    test_activity_reg_under_jit()
    test_sequential_run()
    print("Tests passed!")


def test_run_rejects_stateless_module_with_clear_error():
    """run() on a plain stateless module must raise a TypeError, not pass None."""
    import pytest

    rngs = nnx.Rngs(0)
    stateless = nnx.Linear(4, 4, rngs=rngs)
    with pytest.raises(TypeError, match="initial_state"):
        nn.run(stateless, jnp.ones((3, 2, 4)))
