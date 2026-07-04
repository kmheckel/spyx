"""Tests for spyx.quant.

These tests import qwix lazily; if qwix isn't installed they only exercise the
graceful-degradation path. Install qwix with
``uv pip install "git+https://github.com/google/qwix"`` to run the full suite.
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

import spyx
import spyx.nn as snn


def _qwix_installed() -> bool:
    try:
        import qwix  # noqa: F401

        return True
    except ImportError:
        return False


needs_qwix = pytest.mark.skipif(not _qwix_installed(), reason="qwix not installed")


def test_available_flag_matches_import():
    assert spyx.quant.available() == _qwix_installed()


def test_quantize_without_qwix_raises_helpful_error():
    if _qwix_installed():
        pytest.skip("qwix is installed; this branch only runs without it")
    with pytest.raises(ImportError, match="qwix"):
        spyx.quant.quantize(object(), None)


@needs_qwix
def test_quantize_smoke():
    """quantize() returns a runnable nnx.Module that produces the expected shape."""
    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(8, 16, use_bias=False, rngs=rngs),
        snn.LIF((16,), rngs=rngs),
        nnx.Linear(16, 4, use_bias=False, rngs=rngs),
        snn.LI((4,), rngs=rngs),
    )
    B = 2
    sample_x = jnp.ones((B, 8))
    sample_state = model.initial_state(B)
    qmodel = spyx.quant.quantize(model, sample_x, sample_state)
    out, _ = qmodel(sample_x, sample_state)
    assert out.shape == (B, 4)


@needs_qwix
def test_quantize_supports_qat_training_loop():
    """A quantized model trains via the standard nnx.Optimizer + value_and_grad."""
    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(4, 8, use_bias=False, rngs=rngs),
        snn.LIF((8,), rngs=rngs),
        nnx.Linear(8, 3, use_bias=False, rngs=rngs),
        snn.LI((3,), rngs=rngs),
    )
    B = 2
    sample_x = jnp.ones((B, 4))
    sample_state = model.initial_state(B)
    qmodel = spyx.quant.quantize(model, sample_x, sample_state, mode="qat")

    optimizer = nnx.Optimizer(qmodel, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, x, target):
        def loss_fn(m):
            out, _ = m(x, sample_state)
            return jnp.mean((out - target) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    target = jnp.ones((B, 3))
    losses = [float(step(qmodel, optimizer, sample_x, target)) for _ in range(3)]
    assert all(jnp.isfinite(jnp.array(losses))), losses


@needs_qwix
def test_linear_only_rules_target_matmul_and_conv_ops():
    """linear_only_rules() selects dense/conv work by *op name*, not class name.

    qwix matches ``module_path`` against the NNX attribute path (which never
    contains the class name), so the rule matches every module and narrows to
    the matmul / conv ops. Elementwise neuron updates use none of these ops and
    stay in fp32.
    """
    rules = spyx.quant.linear_only_rules(weight_qtype="int8", act_qtype="int8")
    assert len(rules) == 1
    ops = set(rules[0].op_names)
    assert {"dot_general", "conv_general_dilated"} <= ops


@needs_qwix
def test_weights_only_rules_disables_act_qtype():
    rules = spyx.quant.weights_only_rules(weight_qtype="int8")
    assert len(rules) == 1
    assert rules[0].weight_qtype == "int8"
    assert rules[0].act_qtype is None
    assert "dot_general" in rules[0].op_names


@needs_qwix
def test_bitnet_ternary_rules_use_int2_weights():
    rules = spyx.quant.bitnet_ternary_rules()
    assert len(rules) == 1
    assert rules[0].weight_qtype == "int2"
    # BitNet rules quantize activations too; defaults to int8.
    assert rules[0].act_qtype == "int8"
    assert "dot_general" in rules[0].op_names


@needs_qwix
@pytest.mark.parametrize(
    "make_rules",
    [
        lambda: spyx.quant.linear_only_rules("int8", "int8"),
        lambda: spyx.quant.bitnet_ternary_rules(),
        lambda: spyx.quant.weights_only_rules("int8"),
    ],
)
def test_quantization_actually_changes_outputs(make_rules):
    """Regression: the built-in rules must actually quantize, not no-op.

    The rules previously used ``module_path=r".*Linear.*"``, which qwix's
    ``re.fullmatch`` against NNX attribute paths never matched — so
    ``quantize()`` returned a model with fp32-identical outputs
    (``max|fp - q| == 0``). Guard against that regression by requiring the
    quantized forward pass to differ measurably from full precision.
    """
    rngs = nnx.Rngs(0)

    class Net(nnx.Module):
        def __init__(self, *, rngs):
            self.core = snn.Sequential(
                nnx.Linear(16, 32, use_bias=False, rngs=rngs),
                snn.LIF((32,), rngs=rngs),
                nnx.Linear(32, 4, use_bias=False, rngs=rngs),
                snn.LI((4,), rngs=rngs),
            )

        def __call__(self, x_TBC):
            traces, _ = snn.run(self.core, x_TBC)
            return traces.sum(0)

    model = Net(rngs=rngs)
    T, B = 16, 8
    # Drive over time (dense input) so spikes accumulate and signal reaches the
    # readout; a single-step call leaves the neurons silent and hides quant.
    sample = jax.random.uniform(jax.random.PRNGKey(0), (T, B, 16)) * 2.0
    qmodel = spyx.quant.quantize(model, sample, rules=make_rules())
    fp_out = model(sample)
    q_out = qmodel(sample)
    assert float(jnp.max(jnp.abs(fp_out - q_out))) > 1e-4


@needs_qwix
def test_bitnet_ternary_rules_quantize_a_real_snn():
    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(8, 16, use_bias=False, rngs=rngs),
        snn.LIF((16,), rngs=rngs),
        nnx.Linear(16, 4, use_bias=False, rngs=rngs),
        snn.LI((4,), rngs=rngs),
    )
    sample_x = jnp.ones((2, 8))
    sample_state = model.initial_state(2)
    qmodel = spyx.quant.quantize(
        model, sample_x, sample_state, rules=spyx.quant.bitnet_ternary_rules()
    )
    out, _ = qmodel(sample_x, sample_state)
    assert out.shape == (2, 4)
    assert jnp.all(jnp.isfinite(out))


def _dequant_kernel(qmodel, path):
    """Reconstruct a quantized Linear kernel (``qvalue * scale``) from state.

    ``path`` is the tuple of attribute keys leading to the ``nnx.Linear`` whose
    kernel we want (e.g. ``("lin",)``). Returns the dequantized fp32 weight.
    """
    state = nnx.state(qmodel)
    node = state
    for key in path:
        node = node[key]
    array = node["kernel"]["array"]
    qvalue = jnp.asarray(array["qvalue"].value, dtype=jnp.float32)
    scale = jnp.asarray(array["scale"].value, dtype=jnp.float32)
    return qvalue * scale


# --- binary-activation-aware quantization -----------------------------------


def test_binary_activation_error_is_zero_for_spikes():
    """A genuine {0,1} spike train round-trips through int quant with zero error.

    No qwix required: this is the pure-math core of the losslessness claim.
    """
    spikes = jnp.array(
        [[0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0]], dtype=jnp.float32
    )
    for qtype in ("int8", "int4", "int2"):
        assert spyx.quant.binary_activation_error(spikes, weight_qtype=qtype) == 0.0


def test_binary_activation_error_flags_non_binary_input():
    """Graded (non-spike) activations do lose precision; the check catches it."""
    graded = jnp.array([[0.0, 0.37, 1.0, 0.63]], dtype=jnp.float32)
    assert spyx.quant.binary_activation_error(graded, weight_qtype="int8") > 0.0


def test_binary_activation_error_rejects_non_integer_qtype():
    with pytest.raises(ValueError, match="integer qtypes"):
        spyx.quant.binary_activation_error(jnp.array([0.0, 1.0]), weight_qtype="fp8")


@needs_qwix
def test_spiking_feedforward_rules_are_weight_only_and_skip_einsum():
    """The recipe quantizes feedforward weights only and never touches einsum.

    Weight-only (``act_qtype is None``) keeps the binary spikes untouched -
    lossless on the activation side - and excluding ``einsum`` keeps the
    recurrent / SSM state transition in fp32 (Q-S5, arXiv:2406.09477).
    """
    rules = spyx.quant.spiking_feedforward_rules(weight_qtype="int8")
    assert len(rules) == 1
    rule = rules[0]
    assert rule.weight_qtype == "int8"
    assert rule.act_qtype is None  # activations (spikes) left alone
    ops = set(rule.op_names)
    assert {"dot_general", "conv_general_dilated"} <= ops
    assert "einsum" not in ops  # recurrent path stays fp32


@needs_qwix
@pytest.mark.parametrize("weight_qtype", ["int8", "int4"])
def test_feedforward_quant_is_lossless_on_binary_activations(weight_qtype):
    """Quantized output == fp matmul with the SAME quantized weights on spikes.

    Demonstrates the headline property: because the input is exactly {0,1},
    weight-only quantization introduces ZERO activation-side error. The only
    difference from full precision is the weight rounding, and reconstructing
    ``dequant(quant(W)) @ spikes`` reproduces the model output bit-for-bit.
    """
    rngs = nnx.Rngs(0)
    lin = nnx.Linear(12, 7, use_bias=False, rngs=rngs)

    B = 4
    spikes = (jax.random.uniform(jax.random.PRNGKey(1), (B, 12)) > 0.5).astype(
        jnp.float32
    )
    assert spyx.quant.binary_activation_error(spikes) == 0.0  # input really is {0,1}

    qmodel = spyx.quant.quantize(
        lin,
        spikes,
        rules=spyx.quant.spiking_feedforward_rules(weight_qtype),
        mode="ptq",
    )
    q_out = qmodel(spikes)

    # fp reference computed with the exact same (dequantized) quantized weights.
    Wq = _dequant_kernel(qmodel, ())  # kernel lives at top level for a bare Linear
    ref = spikes @ Wq

    # Exactly equal: no activation was quantized, and spikes need no rounding.
    assert jnp.array_equal(q_out, ref)


@needs_qwix
def test_feedforward_quant_is_not_a_silent_noop():
    """Guard against the historical ``.*Linear.*`` no-op: weights must change.

    The stored kernel must be genuine int8 (not fp32 passthrough), and the
    quantized output must differ measurably from the full-precision matmul.
    """
    rngs = nnx.Rngs(0)
    lin = nnx.Linear(12, 7, use_bias=False, rngs=rngs)
    W = jnp.asarray(lin.kernel.value, dtype=jnp.float32)

    spikes = (jax.random.uniform(jax.random.PRNGKey(2), (4, 12)) > 0.5).astype(
        jnp.float32
    )
    qmodel = spyx.quant.quantize(
        lin, spikes, rules=spyx.quant.spiking_feedforward_rules("int8"), mode="ptq"
    )

    # The kernel is actually stored as int8, with a per-channel fp32 scale.
    state = nnx.state(qmodel)
    array = state["kernel"]["array"]
    assert array["qvalue"].value.dtype == jnp.int8
    assert array["scale"].value.dtype == jnp.float32

    # And the quantized weights genuinely differ from fp32 (rounding happened).
    Wq = _dequant_kernel(qmodel, ())
    assert float(jnp.max(jnp.abs(Wq - W))) > 0.0
    # ...hence the spike output differs from the fp reference.
    assert float(jnp.max(jnp.abs(qmodel(spikes) - spikes @ W))) > 1e-6


@needs_qwix
def test_spiking_feedforward_leaves_recurrent_einsum_in_fp32():
    """A recurrent einsum weight stays fp32 while the feedforward kernel is int8."""

    class Recurrent(nnx.Module):
        def __init__(self, *, rngs):
            self.lin = nnx.Linear(6, 6, use_bias=False, rngs=rngs)
            self.A = nnx.Param(jax.random.normal(rngs.params(), (6, 6)))

        def __call__(self, x):
            h = self.lin(x)  # dot_general -> feedforward, quantized
            return jnp.einsum("ij,bj->bi", self.A.value, h)  # recurrent -> fp32

    rngs = nnx.Rngs(0)
    model = Recurrent(rngs=rngs)
    spikes = (jax.random.uniform(jax.random.PRNGKey(3), (2, 6)) > 0.5).astype(
        jnp.float32
    )
    qmodel = spyx.quant.quantize(
        model, spikes, rules=spyx.quant.spiking_feedforward_rules("int8"), mode="ptq"
    )

    state = nnx.state(qmodel)
    # Feedforward kernel became int8...
    assert state["lin"]["kernel"]["array"]["qvalue"].value.dtype == jnp.int8
    # ...but the recurrent einsum weight is untouched fp32 (no qvalue leaf).
    assert state["A"].value.dtype == jnp.float32
    assert not hasattr(state["A"].value, "qvalue")


@needs_qwix
def test_unknown_mode_raises():
    rngs = nnx.Rngs(0)
    model = snn.Sequential(nnx.Linear(2, 2, rngs=rngs))
    with pytest.raises(ValueError, match="Unknown quantization mode"):
        spyx.quant.quantize(model, jnp.zeros((1, 2)), mode="banana")
