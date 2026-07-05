"""Export a spiking model to ONNX — single-timestep step, or a full temporal loop.

.. warning::
   **Experimental — unstable API.** May change without a deprecation cycle.

A spyx neuron (or a :class:`spyx.nn.Sequential` of them) implements *one*
timestep of the temporal loop::

    (x_t, state) -> (out, new_state)

:func:`spyx.nn.run` scans this over the time axis with ``jax.lax.scan``. There
are two useful things to hand a general runtime (ONNX Runtime, ONNX Runtime
Mobile on a phone, a browser, an embedded target):

* **Per-timestep** (``sequence_length=None``, the default). Export the single
  feed-forward step above; the application runs the temporal loop, calling the
  ONNX graph once per timestep and threading the neuron state (membrane
  potentials, adaptive thresholds, …) itself. ONNX speaks *flat tensor* I/O,
  not pytrees, so the exported signature is the flattened state::

      step(x_t, state_0, state_1, ...) -> (out, new_state_0, new_state_1, ...)

* **Full-sequence** (``sequence_length=T``). Export :func:`spyx.nn.run` over
  ``T`` timesteps so the *whole* temporal loop lives inside the ONNX graph as a
  native ``Loop`` op::

      run(x_seq, state_0, ...) -> (out_seq, final_state_0, ...)

  with ``x_seq`` shaped ``(T, batch, *input_shape)`` and ``out_seq`` shaped
  ``(T, batch, *out)``. jax2onnx's scan plugin lowers the ``jax.lax.scan``
  driving :func:`spyx.nn.run` straight to an ONNX ``Loop``, so no host-side
  temporal loop is needed at all — a real advantage over runtimes that lack a
  clean scan primitive.

:func:`step_signature` returns a :class:`ONNXStepSignature` describing the flat
layout (order, shapes and dtypes of every state tensor, plus the pytree
structure needed to reassemble it) so callers know how to seed state (zeros of
the given shapes) and thread ``new_state_i`` back into the next call. It needs
only JAX, never the conversion stack.

The conversion is a **direct jaxpr -> ONNX lowering** via `jax2onnx
<https://pypi.org/project/jax2onnx/>`_: ``jax2onnx.to_onnx`` traces the pure
JAX function and emits an ``onnx.ModelProto`` — no TensorFlow, no jax2tf, no
TFLite, no tf2onnx. Its scan plugin maps ``jax.lax.scan`` to a native ONNX
``Loop``, which is what makes the full-sequence export a single self-contained
graph.

``jax2onnx`` (and ``onnx``) are imported **lazily** inside the functions, so
``import spyx.experimental.onnx`` works without them installed. Install the
conversion dependencies with::

    pip install jax2onnx onnx onnxruntime

Inference only needs ``onnxruntime`` (or ONNX Runtime Mobile on-device), not the
conversion stack. Only the forward Heaviside spike is exported; the surrogate
gradient is training-only and irrelevant to inference.

Example::

    import jax.numpy as jnp
    from flax import nnx
    from spyx import nn
    from spyx.experimental import onnx

    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nnx.Linear(8, 16, rngs=rngs),
        nn.LIF((16,), rngs=rngs),
        nnx.Linear(16, 4, rngs=rngs),
        nn.LI((4,), rngs=rngs),
    )

    onnx_bytes = onnx.to_onnx(model, (8,), batch=1)  # per-timestep step
    with open("step.onnx", "wb") as f:
        f.write(onnx_bytes)

    # Or the whole temporal loop in one graph (native ONNX Loop):
    seq_bytes = onnx.to_onnx(model, (8,), batch=1, sequence_length=100)

    sig = onnx.step_signature(model, (8,), batch=1)
    # sig.state_shapes -> [(1, 16), (1, 4)] : seed each with zeros on-device.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .. import nn as _nn

_DEFAULT_OPSET = 21


@dataclass
class ONNXStepSignature:
    """Flat tensor layout of an exported step (or full-sequence) function.

    The per-timestep export has the signature ``step(x_t, *state_flat) ->
    (out, *new_state_flat)``; the full-sequence export has
    ``run(x_seq, *state_flat) -> (out_seq, *final_state_flat)`` where ``x_seq``
    carries a leading time axis. This dataclass records everything a caller
    needs to drive either: how to seed the state (zeros of ``state_shapes`` /
    ``state_dtypes``), the order state tensors appear as inputs and outputs, and
    the pytree structure to reassemble the flat state back into the model's
    native (possibly nested / ``None``-holed) state tree.

    :input_shape: Shape of the input tensor (including batch, and, for the
        full-sequence export, a leading time axis).
    :input_dtype: NumPy dtype of the input.
    :state_shapes: Shape of each flattened state tensor, in call order.
    :state_dtypes: NumPy dtype of each flattened state tensor, in call order.
    :output_shape: Shape of the primary output tensor.
    :output_dtype: NumPy dtype of the primary output tensor.
    :input_names: ONNX graph input names, in call order (``x`` first, then each
        flat state tensor).
    :output_names: ONNX graph output names, in call order (primary output first,
        then each flat new-/final-state tensor).
    :sequence_length: ``None`` for the per-timestep export; ``T`` for the
        full-sequence export.
    :state_treedef: The pytree structure of the model's native state, so the
        flat ``state_i`` tensors can be reassembled with
        ``jax.tree_util.tree_unflatten(state_treedef, state_flat)``.
    """

    input_shape: tuple[int, ...]
    input_dtype: np.dtype
    state_shapes: list[tuple[int, ...]] = field(default_factory=list)
    state_dtypes: list[np.dtype] = field(default_factory=list)
    output_shape: tuple[int, ...] = ()
    output_dtype: np.dtype = None  # ty: ignore[invalid-assignment]
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    sequence_length: Any = None
    state_treedef: Any = None

    @property
    def num_state(self) -> int:
        """Number of flat state tensors threaded through the step."""
        return len(self.state_shapes)

    def seed_state(self, dtype=None) -> list[np.ndarray]:
        """Return a fresh zero-initialized flat state (one array per tensor).

        :dtype: Override dtype for every state tensor; defaults to each
            tensor's recorded ``state_dtypes`` entry.
        """
        return [
            np.zeros(shape, dtype=dtype if dtype is not None else dt)
            for shape, dt in zip(self.state_shapes, self.state_dtypes, strict=True)
        ]


def _require_stateful(model) -> None:
    if not hasattr(model, "initial_state"):
        raise TypeError(
            "spyx.experimental.onnx: model has no `initial_state` method. "
            "to_onnx exports a stateful (x_t, state) -> (out, new_state) step "
            "(or a full spyx.nn.run over it); wrap stateless layers in "
            "spyx.nn.Sequential with at least one stateful neuron."
        )


def _build_fn(model, batch, input_shape, dtype, sequence_length):
    """Build the pure flat-I/O export fn and its signature for ``model``.

    Returns ``(fn, sig)``. When ``sequence_length`` is ``None`` this is the
    per-timestep step ``step(x_t, *state_flat) -> (out, *new_state_flat)``;
    otherwise it is the full-sequence ``run(x_seq, *state_flat) ->
    (out_seq, *final_state_flat)`` with a leading time axis of length ``T``.
    """
    _require_stateful(model)

    graphdef, params = nnx.split(model)

    state = model.initial_state(batch)
    state_flat, state_treedef = jax.tree_util.tree_flatten(state)

    np_dtype = np.dtype(jnp.dtype(dtype))
    n_state = len(state_flat)

    if sequence_length is None:
        x_shape = (batch, *tuple(input_shape))

        def fn(x_t, *state_flat_args):
            st = jax.tree_util.tree_unflatten(state_treedef, list(state_flat_args))
            m = nnx.merge(graphdef, params)
            out, new_state = m(x_t, st)
            new_state_flat = jax.tree_util.tree_flatten(new_state)[0]
            return (out, *new_state_flat)
    else:
        x_shape = (int(sequence_length), batch, *tuple(input_shape))

        def fn(x_seq, *state_flat_args):
            st = jax.tree_util.tree_unflatten(state_treedef, list(state_flat_args))
            m = nnx.merge(graphdef, params)
            out_seq, final_state = _nn.run(m, x_seq, st)
            final_state_flat = jax.tree_util.tree_flatten(final_state)[0]
            return (out_seq, *final_state_flat)

    # Trace shapes/dtypes with JAX alone (no conversion deps) so the signature
    # can be filled even when jax2onnx/onnx are absent.
    x_spec = jax.ShapeDtypeStruct(x_shape, dtype)
    state_specs = [
        jax.ShapeDtypeStruct(np.shape(s), jnp.asarray(s).dtype) for s in state_flat
    ]
    out_structs = jax.eval_shape(fn, x_spec, *state_specs)
    out_struct = out_structs[0]

    input_names = ["x"] + [f"state_{i}" for i in range(n_state)]
    out_prefix = "final_state" if sequence_length is not None else "new_state"
    output_names = ["out"] + [f"{out_prefix}_{i}" for i in range(n_state)]

    sig = ONNXStepSignature(
        input_shape=x_shape,
        input_dtype=np_dtype,
        state_shapes=[tuple(np.shape(s)) for s in state_flat],
        state_dtypes=[np.dtype(jnp.asarray(s).dtype) for s in state_flat],
        output_shape=tuple(out_struct.shape),
        output_dtype=np.dtype(out_struct.dtype),
        input_names=input_names,
        output_names=output_names,
        sequence_length=sequence_length,
        state_treedef=state_treedef,
    )
    return fn, sig


def step_signature(
    model, input_shape, *, batch=1, dtype=jnp.float32, sequence_length=None
) -> ONNXStepSignature:
    """Describe the flat tensor I/O of ``model``'s exported step.

    Does **not** require jax2onnx/onnx — it only traces shapes/dtypes with JAX,
    so callers can plan state seeding/threading without running a conversion.
    See :class:`ONNXStepSignature`.

    :model: A spyx neuron or :class:`spyx.nn.Sequential` implementing
        ``(x_t, state) -> (out, new_state)`` and exposing ``initial_state``.
    :input_shape: Per-timestep input feature shape, *excluding* batch and time
        (e.g. ``(8,)`` for a length-8 input vector).
    :batch: Batch dimension of the exported step. Defaults to 1.
    :dtype: Input/compute dtype. Defaults to ``jnp.float32``.
    :sequence_length: ``None`` (default) describes the per-timestep step;
        an integer ``T`` describes the full-sequence export (leading time axis).
    """
    _, sig = _build_fn(model, batch, input_shape, dtype, sequence_length)
    return sig


def to_onnx(
    model,
    input_shape,
    *,
    batch=1,
    dtype=jnp.float32,
    opset=None,
    sequence_length=None,
) -> bytes:
    """Export a spiking model to ONNX and return the serialized ``ModelProto``.

    With ``sequence_length=None`` (default) this exports the single feed-forward
    step ``(x_t, state) -> (out, new_state)`` — no temporal scan — whose flat
    ONNX signature is ``step(x_t, *state_flat) -> (out, *new_state_flat)``. The
    application runs the temporal loop, calling the graph once per timestep and
    threading ``new_state_i`` back in as ``state_i``. Pair with
    :func:`step_signature` to learn the flat state layout and to seed zeros.

    With an integer ``sequence_length=T`` this exports :func:`spyx.nn.run` over
    ``T`` timesteps, so the ONNX graph contains the whole temporal loop as a
    native ``Loop`` (jax2onnx's scan plugin lowers the ``jax.lax.scan`` to it);
    the signature becomes ``run(x_seq, *state_flat) -> (out_seq,
    *final_state_flat)`` with a leading time axis of length ``T`` on ``x_seq``
    and ``out_seq``.

    Conversion is a direct jaxpr -> ONNX lowering via ``jax2onnx.to_onnx`` — no
    TensorFlow. Only the forward Heaviside spike is exported; the surrogate
    gradient is training-only and irrelevant to inference.

    Requires ``jax2onnx`` and ``onnx`` (``pip install jax2onnx onnx
    onnxruntime``); they are imported lazily here so importing this module does
    not need them. Inference only needs ``onnxruntime`` (or ONNX Runtime Mobile
    on a phone), not the conversion stack.

    :model: A spyx neuron or :class:`spyx.nn.Sequential` implementing
        ``(x_t, state) -> (out, new_state)`` and exposing ``initial_state``.
    :input_shape: Per-timestep input feature shape, *excluding* batch and time
        (e.g. ``(8,)``).
    :batch: Batch dimension of the exported graph. Defaults to 1.
    :dtype: Input/compute dtype. Defaults to ``jnp.float32``.
    :opset: ONNX opset version to target. ``None`` defaults to ``21`` (recent
        enough for the native ``Loop`` used by the full-sequence export).
    :sequence_length: ``None`` exports the per-timestep step; an integer ``T``
        exports the full ``spyx.nn.run`` over ``T`` timesteps.
    :return: The serialized ONNX ``ModelProto`` as ``bytes``.
    """
    try:
        import jax2onnx  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "spyx.experimental.onnx.to_onnx requires jax2onnx + onnx for the "
            "direct jaxpr -> ONNX conversion. Install them with "
            "`pip install jax2onnx onnx onnxruntime`. (Inference only needs "
            "onnxruntime, not the conversion stack.)"
        ) from exc

    fn, sig = _build_fn(model, batch, input_shape, dtype, sequence_length)

    opset = _DEFAULT_OPSET if opset is None else int(opset)

    # jax2onnx traces the pure JAX fn and emits an onnx.ModelProto directly. Its
    # scan plugin maps the jax.lax.scan driving spyx.nn.run to a native ONNX
    # Loop, so the full-sequence export is a single self-contained graph.
    inputs = [
        jax.ShapeDtypeStruct(sig.input_shape, dtype),
    ]
    inputs += [
        jax.ShapeDtypeStruct(shape, np.dtype(dt))
        for shape, dt in zip(sig.state_shapes, sig.state_dtypes, strict=True)
    ]

    suffix = f"seq{sequence_length}" if sequence_length is not None else "step"
    model_proto = jax2onnx.to_onnx(
        fn,
        inputs,
        model_name=f"spyx_{suffix}",
        opset=opset,
        input_names=sig.input_names,
    )

    return model_proto.SerializeToString()
