"""Tests for spyx.quant.

These tests import qwix lazily; if qwix isn't installed they only exercise the
graceful-degradation path. Install qwix with
``uv pip install "git+https://github.com/google/qwix"`` to run the full suite.
"""

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
def test_linear_only_rules_skips_neuron_modules():
    """linear_only_rules() should match Linear/Conv module paths only."""
    rules = spyx.quant.linear_only_rules(weight_qtype="int8", act_qtype="int8")
    paths = [r.module_path for r in rules]
    assert any("Linear" in p for p in paths)
    assert any("Conv" in p for p in paths)
    # Spiking neurons should NOT be matched by the default rules.
    assert not any("LIF" in p or "ALIF" in p for p in paths)


@needs_qwix
def test_weights_only_rules_disables_act_qtype():
    rules = spyx.quant.weights_only_rules(weight_qtype="int8")
    assert len(rules) == 1
    assert rules[0].weight_qtype == "int8"
    assert rules[0].act_qtype is None


@needs_qwix
def test_bitnet_ternary_rules_use_int2_weights():
    rules = spyx.quant.bitnet_ternary_rules()
    assert len(rules) == 2
    assert {r.weight_qtype for r in rules} == {"int2"}
    # All BitNet rules should have act_qtype set; defaults to int8.
    assert {r.act_qtype for r in rules} == {"int8"}


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
