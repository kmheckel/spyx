import jax
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

    count = model.initial_state(batch_size=2)
    assert jnp.array_equal(count, jnp.zeros((2, 2), dtype=jnp.float32))

    out, count = model(spikes, count)
    assert jnp.array_equal(out, spikes)
    assert jnp.array_equal(count, spikes)

    out, count = model(spikes, count)
    assert jnp.array_equal(count, spikes * 2)


def test_activity_reg_under_jit():
    spikes = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
    model = nn.ActivityRegularization(hidden_shape=(2,), batch_size=2)

    @nnx.jit
    def step(m, s, c):
        return m(s, c)

    count = model.initial_state(batch_size=2)
    out, count = step(model, spikes, count)
    assert jnp.array_equal(out, spikes)
    assert jnp.array_equal(count, spikes)


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


def test_if_accepts_and_ignores_rngs():
    """IF must be drop-in interchangeable with parametric neurons.

    Swapping ``LIF((n,), rngs=rngs)`` -> ``IF((n,), rngs=rngs)`` used to raise
    TypeError because IF's constructor rejected ``rngs=``. IF is parameterless,
    so it accepts and ignores it.
    """
    rngs = nnx.Rngs(0)
    model = nn.IF((10,), rngs=rngs)  # must not raise
    x = jnp.ones((5, 10)) * 2.0
    V = model.initial_state(5)
    spikes, V_next = model(x, V)
    assert spikes.shape == (5, 10)
    assert V_next.shape == (5, 10)

    # Behaviour must be identical to constructing IF without rngs.
    plain = nn.IF((10,))
    s2, v2 = plain(x, plain.initial_state(5))
    assert jnp.array_equal(spikes, s2)
    assert jnp.array_equal(V_next, v2)


def test_cubalif_single_subtractive_reset():
    """CuBaLIF must apply the subtractive reset exactly once per spike.

    Regression for a double-reset bug where the reset was subtracted both from
    the raw membrane (``V = V - reset``) and again from the integrated membrane
    (``V = beta * V + current_I - reset``). Here we drive a neuron above
    threshold with fixed alpha/beta and compare against a hand-computed
    single-reset reference. A re-introduced double reset shifts V_final by one
    threshold and fails this test.
    """
    rngs = nnx.Rngs(0)
    alpha, beta, threshold = 0.9, 0.8, 1.0
    model = nn.CuBaLIF((1,), alpha=alpha, beta=beta, threshold=threshold, rngs=rngs)

    # State layout is [V | current_I] concatenated on the last axis.
    V0, I0, x = 2.0, 0.5, 0.3  # V0 > threshold => the neuron spikes
    VI = jnp.array([[V0, I0]])
    x_in = jnp.array([[x]])

    spikes, VI_next = model(x_in, VI)
    V_next, I_next = jnp.split(VI_next, 2, -1)

    assert float(spikes[0, 0]) == 1.0  # sanity: it fired

    # Single-reset reference (what CuBaLIF should compute):
    V_after_reset = V0 - 1.0 * threshold
    I_ref = alpha * I0 + x
    V_ref = beta * V_after_reset + I_ref

    assert jnp.allclose(I_next, I_ref)
    assert jnp.allclose(V_next, V_ref)

    # And it must NOT match the old double-reset value (V_ref - threshold).
    assert not jnp.allclose(V_next, V_ref - threshold)


def test_cubalif_matches_scratch_current_lif_over_sequence():
    """CuBaLIF scanned over time matches a from-scratch single-reset CuBa-LIF.

    A second, independent guard against a re-introduced double reset: run a
    short sequence and compare against a reference loop implementing a single
    subtractive reset step by step.
    """
    rngs = nnx.Rngs(0)
    alpha, beta, threshold = 0.7, 0.85, 1.0
    C = 4
    model = nn.CuBaLIF((C,), alpha=alpha, beta=beta, threshold=threshold, rngs=rngs)
    spike_fn = model.spike

    T, B = 8, 3
    x = jax.random.normal(jax.random.key(1), (T, B, C)) * 3.0

    outputs, _ = nn.run(model, x)

    # Reference: single-reset current-based LIF, hand-rolled.
    V = jnp.zeros((B, C))
    current_I = jnp.zeros((B, C))
    ref_spikes = []
    for t in range(T):
        s = spike_fn(V - threshold)
        V = V - s * threshold
        current_I = alpha * current_I + x[t]
        V = beta * V + current_I
        ref_spikes.append(s)
    ref = jnp.stack(ref_spikes, axis=0)

    assert jnp.allclose(outputs, ref)


def test_stateful_layer_protocol():
    """Spyx neurons satisfy the runtime-checkable StatefulLayer contract."""
    rngs = nnx.Rngs(0)
    assert isinstance(nn.LIF((4,), rngs=rngs), nn.StatefulLayer)
    assert isinstance(nn.IF((4,)), nn.StatefulLayer)
    assert isinstance(nn.CuBaLIF((4,), rngs=rngs), nn.StatefulLayer)
    # A plain stateless layer does NOT satisfy it (no initial_state).
    assert not isinstance(nnx.Linear(4, 4, rngs=rngs), nn.StatefulLayer)


def test_run_batch_major_matches_transpose_run_transpose():
    """run(..., batch_major=True) == transpose -> run -> transpose."""
    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nnx.Linear(8, 4, rngs=rngs),
        nn.LIF((4,), rngs=rngs),
    )
    T, B = 6, 3
    x_bt = jax.random.normal(jax.random.key(2), (B, T, 8))

    outs_bm, state_bm = nn.run(model, x_bt, batch_major=True)
    assert outs_bm.shape == (B, T, 4)

    # Manual equivalent: transpose to time-major, run, transpose back.
    x_tb = jnp.swapaxes(x_bt, 0, 1)
    outs_tm, state_tm = nn.run(model, x_tb)
    assert jnp.allclose(outs_bm, jnp.swapaxes(outs_tm, 0, 1))
    # Final state is batch-leading in both cases and should be identical.
    assert jnp.allclose(state_bm[1], state_tm[1])


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


def test_flatten_collapses_non_batch_dims():
    """Flatten maps (B, ...) -> (B, prod(...)) and runs stateless in Sequential.

    Regression for spyx.nir referencing the non-existent ``nnx.Flatten``.
    """
    f = nn.Flatten()
    x = jnp.ones((4, 2, 3, 3))
    assert f(x).shape == (4, 18)

    # Stateless slot inside Sequential (no initial_state on Flatten).
    seq = nn.Sequential(nn.Flatten(), nnx.Linear(18, 5, rngs=nnx.Rngs(0)))
    out, _ = seq(x, seq.initial_state(4))
    assert out.shape == (4, 5)


def test_default_decay_init_no_negatives():
    """Default decay-param init must lie in [0, 1] with NO negatives.

    Regression for the Haiku->NNX port bug where the init produced values in
    [-0.75, 1.25], leaving ~30% of decay params negative; those get clipped to
    0 in __call__ and freeze there (clip zeroes the gradient).
    """
    rngs = nnx.Rngs(0)
    hidden_shape = (4096,)

    lif = nn.LIF(hidden_shape, rngs=rngs)
    alif = nn.ALIF(hidden_shape, rngs=rngs)
    cuba = nn.CuBaLIF(hidden_shape, rngs=rngs)
    psu = nn.PSU_LIF(hidden_shape, rngs=rngs)

    decay_params = [
        lif.beta[...],
        alif.beta[...],
        alif.gamma[...],
        cuba.alpha[...],
        cuba.beta[...],
        psu.beta[...],
    ]
    for p in decay_params:
        assert jnp.all(p >= 0.0), "decay init produced negative values"
        assert jnp.all(p <= 1.0), "decay init produced values above 1"
        # sanity: distribution actually populated (not all identical / degenerate)
        assert p.shape == hidden_shape


def test_activity_reg_accumulates_under_run():
    """Sequential(LIF, ActivityRegularization) through nn.run accumulates spikes.

    The final carried spike count must equal the per-neuron sum of the emitted
    spikes over the time axis.
    """
    rngs = nnx.Rngs(0)
    C = 6
    model = nn.Sequential(
        nn.LIF((C,), rngs=rngs),
        nn.ActivityRegularization((C,)),
    )

    T, B = 20, 4
    x = jax.random.normal(jax.random.key(0), (T, B, C)) * 2.0

    outputs, final_state = nn.run(model, x)
    assert outputs.shape == (T, B, C)

    # ActivityRegularization is the second (index 1) layer of the Sequential.
    accumulated = final_state[1]
    assert accumulated.shape == (B, C)

    expected = jnp.sum(outputs, axis=0)
    assert jnp.array_equal(accumulated, expected)
