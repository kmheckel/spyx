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


@needs_qwix
def test_unknown_mode_raises():
    rngs = nnx.Rngs(0)
    model = snn.Sequential(nnx.Linear(2, 2, rngs=rngs))
    with pytest.raises(ValueError, match="Unknown quantization mode"):
        spyx.quant.quantize(model, jnp.zeros((1, 2)), mode="banana")
