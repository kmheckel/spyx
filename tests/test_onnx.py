"""Tests for spyx.experimental.onnx (spyx -> ONNX step / full-sequence export).

Conversion goes through jax2onnx (direct jaxpr -> ONNX, no TensorFlow); running
the exported graph needs onnxruntime. Every test that touches conversion is
gated with ``pytest.importorskip`` so the suite skips cleanly in the default CI
environment (which has neither). The signature-only tests need just JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from spyx import nn
from spyx.experimental import onnx


def _build_model():
    rngs = nnx.Rngs(0)
    return nn.Sequential(
        nnx.Linear(8, 16, use_bias=False, rngs=rngs),
        nn.LIF((16,), rngs=rngs),
        nnx.Linear(16, 4, use_bias=False, rngs=rngs),
        nn.LI((4,), rngs=rngs),
    )


def _match_outputs(onnx_outputs, jax_refs, atol=1e-4, rtol=1e-4):
    """Match each JAX reference array to an ONNX output by shape + value.

    Output ordering across the conversion is preserved by name, but we match by
    value to stay robust; every reference must find a distinct ONNX output.
    """
    remaining = list(onnx_outputs)
    for ref in jax_refs:
        match = None
        for cand in remaining:
            if cand.shape == ref.shape and np.allclose(cand, ref, atol=atol, rtol=rtol):
                match = cand
                break
        assert match is not None, f"no ONNX output matched JAX ref of shape {ref.shape}"
        remaining.remove(match)


def test_step_signature_no_conversion_deps():
    """step_signature traces the flat state layout without any conversion deps."""
    model = _build_model()
    sig = onnx.step_signature(model, (8,), batch=1)

    assert sig.input_shape == (1, 8)
    # One state tensor per stateful layer (LIF -> (1,16), LI -> (1,4)).
    assert sig.num_state == 2
    assert sig.state_shapes == [(1, 16), (1, 4)]
    assert sig.output_shape == (1, 4)
    assert sig.input_names == ["x", "state_0", "state_1"]
    assert sig.output_names == ["out", "new_state_0", "new_state_1"]
    assert sig.sequence_length is None

    seeded = sig.seed_state()
    assert len(seeded) == 2
    assert seeded[0].shape == (1, 16)
    assert all(np.count_nonzero(s) == 0 for s in seeded)


def test_sequence_signature_no_conversion_deps():
    """A full-sequence signature carries a leading time axis on input/output."""
    model = _build_model()
    sig = onnx.step_signature(model, (8,), batch=1, sequence_length=5)

    assert sig.input_shape == (5, 1, 8)
    assert sig.output_shape == (5, 1, 4)
    assert sig.num_state == 2
    assert sig.output_names == ["out", "final_state_0", "final_state_1"]
    assert sig.sequence_length == 5


def test_step_signature_requires_stateful_model():
    """A bare stateless module has no initial_state and must be rejected."""
    with pytest.raises(TypeError):
        onnx.step_signature(nnx.Linear(8, 4, rngs=nnx.Rngs(0)), (8,))


def test_to_onnx_step_matches_jax():
    """The exported per-timestep ONNX graph reproduces the JAX single step."""
    pytest.importorskip("jax2onnx")
    ort = pytest.importorskip("onnxruntime")

    B = 3
    model = _build_model()
    sig = onnx.step_signature(model, (8,), batch=B)

    rng = np.random.default_rng(0)
    x_t = jnp.asarray(rng.standard_normal((B, 8)), dtype=jnp.float32)
    # Nonzero initial state so the exported state threading is actually exercised.
    state_flat = [
        jnp.asarray(rng.standard_normal(shape) * 0.7, dtype=jnp.float32)
        for shape in sig.state_shapes
    ]
    state = jax.tree_util.tree_unflatten(sig.state_treedef, state_flat)
    out_jax, new_state_jax = model(x_t, state)
    new_state_flat_jax = [
        np.asarray(s) for s in jax.tree_util.tree_leaves(new_state_jax)
    ]

    onnx_bytes = onnx.to_onnx(model, (8,), batch=B)
    assert isinstance(onnx_bytes, (bytes, bytearray))
    assert len(onnx_bytes) > 0

    sess = ort.InferenceSession(bytes(onnx_bytes), providers=["CPUExecutionProvider"])
    feeds = {"x": np.asarray(x_t, dtype=np.float32)}
    for name, val in zip(sig.input_names[1:], state_flat, strict=True):
        feeds[name] = np.asarray(val, dtype=np.float32)
    onnx_outputs = sess.run(None, feeds)

    jax_refs = [np.asarray(out_jax)] + new_state_flat_jax
    _match_outputs(onnx_outputs, jax_refs)


def test_to_onnx_full_sequence_matches_run():
    """The full-sequence ONNX graph is a native Loop and matches spyx.nn.run.

    jax2onnx's scan plugin lowers the ``jax.lax.scan`` driving ``spyx.nn.run``
    directly to a native ONNX ``Loop``; the whole temporal loop therefore lives
    inside a single self-contained graph.
    """
    onnx_pkg = pytest.importorskip("onnx")
    pytest.importorskip("jax2onnx")
    ort = pytest.importorskip("onnxruntime")

    T, B = 5, 3
    model = _build_model()
    sig = onnx.step_signature(model, (8,), batch=B, sequence_length=T)

    rng = np.random.default_rng(1)
    x_seq = jnp.asarray(rng.standard_normal((T, B, 8)) * 1.5, dtype=jnp.float32)
    # Nonzero initial state.
    state_flat = [
        jnp.asarray(rng.standard_normal(shape) * 0.7, dtype=jnp.float32)
        for shape in sig.state_shapes
    ]
    state = jax.tree_util.tree_unflatten(sig.state_treedef, state_flat)
    out_seq_jax, final_state_jax = nn.run(model, x_seq, state)
    final_flat_jax = [np.asarray(s) for s in jax.tree_util.tree_leaves(final_state_jax)]

    onnx_bytes = onnx.to_onnx(model, (8,), batch=B, sequence_length=T)
    assert isinstance(onnx_bytes, (bytes, bytearray))
    assert len(onnx_bytes) > 0

    # The temporal loop must lower to a native ONNX Loop (or Scan) op.
    model_proto = onnx_pkg.load_from_string(bytes(onnx_bytes))
    op_types = {node.op_type for node in model_proto.graph.node}
    assert op_types & {"Loop", "Scan"}, f"no native Loop/Scan op; got {op_types}"

    sess = ort.InferenceSession(bytes(onnx_bytes), providers=["CPUExecutionProvider"])
    feeds = {"x": np.asarray(x_seq, dtype=np.float32)}
    for name, val in zip(sig.input_names[1:], state_flat, strict=True):
        feeds[name] = np.asarray(val, dtype=np.float32)
    onnx_outputs = sess.run(None, feeds)

    jax_refs = [np.asarray(out_seq_jax)] + final_flat_jax
    _match_outputs(onnx_outputs, jax_refs)
