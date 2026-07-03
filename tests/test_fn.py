import jax.numpy as jnp
import pytest

from spyx import fn


def test_integral_accuracy():
    acc = fn.integral_accuracy(time_axis=1)
    # Two batch items, T=3 timesteps, 2 classes; class-0 wins on item 0, class-1 on item 1.
    traces = jnp.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    targets = jnp.array([0, 1])
    score, preds = acc(traces, targets)
    assert score == 1.0
    assert jnp.array_equal(preds, jnp.array([0, 1]))

    targets_wrong = jnp.array([1, 0])
    score_wrong, _ = acc(traces, targets_wrong)
    assert score_wrong == 0.0


def test_integral_crossentropy_finite_and_decreases_for_correct_label():
    loss = fn.integral_crossentropy(smoothing=0.0, time_axis=1)
    correct = jnp.array([[[10.0, 0.0]] * 4])
    wrong = jnp.array([[[0.0, 10.0]] * 4])
    targets = jnp.array([0])
    l_correct = loss(correct, targets)
    l_wrong = loss(wrong, targets)
    assert jnp.isfinite(l_correct)
    assert jnp.isfinite(l_wrong)
    assert l_correct < l_wrong


def test_mse_spikerate_zero_when_matched():
    mse = fn.mse_spikerate(sparsity=0.5, smoothing=0.0, time_axis=1)
    # T=4, sparsity=0.5 => target rate scaled by sparsity * T = 2.
    # Build traces whose mean over time is 0.5 for the correct class only.
    # mean(traces, axis=1) shape -> [B, C]; we want logits == [[2., 0.]] after the loss
    # multiplies labels by sparsity*T (= 2). To do that, mean over time must be [0.5, 0.0].
    # But the loss does jnp.mean(traces, axis=1) which gives the trace mean, not 2*trace mean.
    # So set traces such that mean = [0.5, 0.0] but labels * sparsity * t = [2, 0]. That mismatches.
    # Instead: use sparsity=1.0 so labels * 1 * T = [T, 0] = [4, 0], and pick traces with mean
    # equal to [4, 0] across all timesteps => set traces all 4s and 0s.
    mse = fn.mse_spikerate(sparsity=1.0, smoothing=0.0, time_axis=1)
    traces = jnp.tile(jnp.array([[4.0, 0.0]]), (1, 4, 1))
    targets = jnp.array([0])
    assert mse(traces, targets) == 0.0


def test_silence_reg_penalises_low_spikers():
    reg = fn.silence_reg(min_spikes=1.0)
    # Single layer, batch=4, neurons=2: first neuron is always silent, second always spikes.
    spikes = [jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])]
    loss = reg(spikes)
    # silent neuron: max(0, 1 - 0)^2 = 1; active neuron: max(0, 1 - 1)^2 = 0; sum = 1.
    assert jnp.isclose(loss, 1.0)

    # With min_spikes=0 nothing is penalised.
    reg_zero = fn.silence_reg(min_spikes=0.0)
    assert reg_zero(spikes) == 0.0


def test_sparsity_reg_penalises_high_spikers():
    reg = fn.sparsity_reg(max_spikes=0.1)
    # Layer with mean spike rate 1 -> well above threshold.
    spikes = [jnp.ones((4, 8))]
    high_loss = reg(spikes)
    assert high_loss > 0
    # Layer with mean spike rate 0 -> no penalty.
    silent = [jnp.zeros((4, 8))]
    assert reg(silent) == 0.0


# ---------------------------------------------------------------------------
# Shape checks (issue #25)
# ---------------------------------------------------------------------------


def test_integral_accuracy_rejects_rank_1_traces():
    acc = fn.integral_accuracy(time_axis=1)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        acc(jnp.zeros((4,)), jnp.zeros((4,), dtype=jnp.int32))


def test_integral_accuracy_rejects_target_batch_mismatch():
    acc = fn.integral_accuracy(time_axis=1)
    # traces: (B=3, T=4, C=5); expected targets: (B=3,)
    traces = jnp.zeros((3, 4, 5))
    with pytest.raises(ValueError, match="does not match"):
        acc(traces, jnp.zeros((7,), dtype=jnp.int32))


def test_integral_crossentropy_rejects_target_batch_mismatch():
    loss = fn.integral_crossentropy(smoothing=0.0, time_axis=1)
    traces = jnp.zeros((3, 4, 5))
    with pytest.raises(ValueError, match="does not match"):
        loss(traces, jnp.zeros((7,), dtype=jnp.int32))


def test_integral_crossentropy_rejects_bad_time_axis():
    loss = fn.integral_crossentropy(time_axis=99)
    with pytest.raises(ValueError, match="out of range"):
        loss(jnp.zeros((2, 4, 5)), jnp.zeros((2,), dtype=jnp.int32))


def test_shape_check_accepts_valid_shapes():
    """Shape check must NOT fire on the canonical (B, T, C) layout."""
    acc = fn.integral_accuracy(time_axis=1)
    loss = fn.integral_crossentropy(smoothing=0.0, time_axis=1)
    mse = fn.mse_spikerate(sparsity=0.25, smoothing=0.0, time_axis=1)
    traces = jnp.zeros((3, 4, 5))
    targets = jnp.zeros((3,), dtype=jnp.int32)
    # None of these should raise; return values are not asserted here.
    acc(traces, targets)
    loss(traces, targets)
    mse(traces, targets)
