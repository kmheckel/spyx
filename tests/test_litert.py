"""Tests for spyx.experimental.litert (spyx -> LiteRT/TFLite step export).

The conversion path needs TensorFlow; every test that touches it is gated with
``pytest.importorskip("tensorflow")`` so the suite skips cleanly in the default
CI environment (which has no TF). The signature-only test needs just JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from spyx import nn
from spyx.experimental import litert


def _build_model():
    rngs = nnx.Rngs(0)
    return nn.Sequential(
        nnx.Linear(8, 16, rngs=rngs),
        nn.LIF((16,), rngs=rngs),
        nnx.Linear(16, 4, rngs=rngs),
        nn.LI((4,), rngs=rngs),
    )


def test_step_signature_no_tf():
    """step_signature traces the flat state layout without needing TensorFlow."""
    model = _build_model()
    sig = litert.step_signature(model, (8,), batch=1)

    assert sig.input_shape == (1, 8)
    # One state tensor per stateful layer (LIF -> (1,16), LI -> (1,4)).
    assert sig.num_state == 2
    assert sig.state_shapes == [(1, 16), (1, 4)]
    assert sig.output_shape == (1, 4)

    seeded = sig.seed_state()
    assert len(seeded) == 2
    assert seeded[0].shape == (1, 16)
    assert all(np.count_nonzero(s) == 0 for s in seeded)


def test_step_signature_requires_stateful_model():
    """A bare stateless module has no initial_state and must be rejected."""
    with pytest.raises(TypeError):
        litert.step_signature(nnx.Linear(8, 4, rngs=nnx.Rngs(0)), (8,))


def test_to_litert_matches_jax_step():
    """The exported .tflite step reproduces the JAX single-timestep output."""
    tf = pytest.importorskip("tensorflow")

    model = _build_model()
    sig = litert.step_signature(model, (8,), batch=1)

    # Reference JAX single step on a concrete input + zero state.
    rng = np.random.default_rng(0)
    x_t = jnp.asarray(rng.standard_normal((1, 8)), dtype=jnp.float32)
    state = model.initial_state(1)
    out_jax, new_state_jax = model(x_t, state)
    new_state_flat_jax = [
        np.asarray(s) for s in jax.tree_util.tree_leaves(new_state_jax)
    ]

    tflite_bytes = litert.to_litert(model, (8,), batch=1)
    assert isinstance(tflite_bytes, (bytes, bytearray))
    assert len(tflite_bytes) > 0

    interpreter = tf.lite.Interpreter(model_content=bytes(tflite_bytes))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Feed x_t and the (zero) flat state in the same order to_litert declared.
    flat_inputs = [np.asarray(x_t, dtype=np.float32)] + sig.seed_state(dtype=np.float32)
    assert len(input_details) == len(flat_inputs)
    for det, val in zip(input_details, flat_inputs, strict=True):
        interpreter.set_tensor(det["index"], val)
    interpreter.invoke()

    tflite_outputs = [interpreter.get_tensor(det["index"]) for det in output_details]

    # TFLite output ordering is not guaranteed to match the tuple order, so
    # match each JAX reference array to a tflite output by shape + value.
    jax_refs = [np.asarray(out_jax)] + new_state_flat_jax
    remaining = list(tflite_outputs)
    for ref in jax_refs:
        match = None
        for cand in remaining:
            if cand.shape == ref.shape and np.allclose(cand, ref, atol=1e-4, rtol=1e-4):
                match = cand
                break
        assert match is not None, (
            f"no tflite output matched JAX ref of shape {ref.shape}"
        )
        remaining.remove(match)
