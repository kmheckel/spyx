"""Export a spiking model's single-timestep step function to LiteRT (TFLite).

.. warning::
   **Experimental — unstable API.** May change without a deprecation cycle.

A spyx neuron (or a :class:`spyx.nn.Sequential` of them) implements *one*
timestep of the temporal loop::

    (x_t, state) -> (out, new_state)

:func:`spyx.nn.run` scans this over the time axis on the host with
``jax.lax.scan``. On a phone you instead want the temporal loop to live in the
application code, calling a tiny compiled kernel once per timestep and threading
the neuron state (membrane potentials, adaptive thresholds, …) yourself. That
kernel is exactly the feed-forward step above — no scan, and, for inference, no
surrogate gradient: only the forward Heaviside spike matters.

:func:`to_litert` converts that single step to a `LiteRT <https://ai.google.dev/edge/litert>`_
(``.tflite``) flatbuffer via :mod:`jax.experimental.jax2tf` and
:class:`tf.lite.TFLiteConverter`. TFLite speaks flat tensor I/O, not pytrees, so
the exported signature is the *flattened* state::

    step(x_t, state_0, state_1, ...) -> (out, new_state_0, new_state_1, ...)

:func:`step_signature` returns a :class:`LiteRTStepSignature` describing that
flat layout — the order, shapes and dtypes of every state tensor plus the
pytree structure needed to reassemble it — so on-device code knows how to seed
state (zeros of the given shapes) and thread ``new_state_i`` back into the next
call.

``tensorflow`` and ``jax2tf`` are imported **lazily** inside the functions, so
``import spyx.experimental.litert`` works without TensorFlow installed. Install
the conversion dependency with ``pip install "tensorflow>=2.16"``. Phone-side
inference needs only the much lighter ``tflite-runtime`` / ``ai-edge-litert``,
not full TensorFlow — **provided** the model converts under ``TFLITE_BUILTINS``.
Conversion enables the ``SELECT_TF_OPS`` fallback for robustness; if a model
actually lands Flex ops, the resulting ``.tflite`` needs the TF Flex delegate at
runtime. Simple feed-forward neuron steps (Linear/LIF/LI) stay within builtins.

Example::

    import jax.numpy as jnp
    from flax import nnx
    from spyx import nn
    from spyx.experimental import litert

    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nnx.Linear(8, 16, rngs=rngs),
        nn.LIF((16,), rngs=rngs),
        nnx.Linear(16, 4, rngs=rngs),
        nn.LI((4,), rngs=rngs),
    )

    tflite_bytes = litert.to_litert(model, (8,), batch=1)
    with open("step.tflite", "wb") as f:
        f.write(tflite_bytes)

    sig = litert.step_signature(model, (8,), batch=1)
    # sig.state_shapes -> [(1, 16), (1, 4)] : seed each with zeros on-device.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class LiteRTStepSignature:
    """Flat tensor layout of an exported single-timestep step function.

    The exported ``.tflite`` step has the signature ``step(x_t, *state_flat) ->
    (out, *new_state_flat)``. This dataclass records everything a caller needs
    to drive it on-device: how to seed the state (zeros of ``state_shapes`` /
    ``state_dtypes``), the order state tensors appear as inputs and outputs, and
    the pytree structure to reassemble the flat state back into the model's
    native (possibly nested / ``None``-holed) state tree if desired.

    :input_shape: Shape of the per-timestep input ``x_t`` (including batch).
    :input_dtype: NumPy dtype of ``x_t``.
    :state_shapes: Shape of each flattened state tensor, in call order.
    :state_dtypes: NumPy dtype of each flattened state tensor, in call order.
    :output_shape: Shape of the step's ``out`` tensor.
    :output_dtype: NumPy dtype of the step's ``out`` tensor.
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


def _build_step(model, batch, input_shape, dtype):
    """Build the pure flat-I/O step fn and its signature for ``model``.

    Returns ``(step, sig)`` where ``step(x_t, *state_flat) ->
    (out, *new_state_flat)`` is a pure jittable function over flat tensors and
    ``sig`` is the :class:`LiteRTStepSignature` describing its I/O.
    """
    if not hasattr(model, "initial_state"):
        raise TypeError(
            "spyx.experimental.litert: model has no `initial_state` method. "
            "to_litert exports a stateful (x_t, state) -> (out, new_state) "
            "step; wrap stateless layers in spyx.nn.Sequential with at least "
            "one stateful neuron."
        )

    graphdef, params = nnx.split(model)

    state = model.initial_state(batch)
    state_flat, state_treedef = jax.tree_util.tree_flatten(state)

    x_shape = (batch, *tuple(input_shape))
    np_dtype = np.dtype(jnp.dtype(dtype))

    def step(x_t, *state_flat_args):
        st = jax.tree_util.tree_unflatten(state_treedef, list(state_flat_args))
        m = nnx.merge(graphdef, params)
        out, new_state = m(x_t, st)
        new_state_flat = jax.tree_util.tree_flatten(new_state)[0]
        return (out, *new_state_flat)

    # Trace shapes/dtypes without running conversion so we can fill the
    # signature even when TensorFlow is absent.
    x_spec = jax.ShapeDtypeStruct(x_shape, dtype)
    state_specs = [
        jax.ShapeDtypeStruct(np.shape(s), jnp.asarray(s).dtype) for s in state_flat
    ]
    out_shapes = jax.eval_shape(step, x_spec, *state_specs)
    out_struct = out_shapes[0]

    sig = LiteRTStepSignature(
        input_shape=x_shape,
        input_dtype=np_dtype,
        state_shapes=[tuple(np.shape(s)) for s in state_flat],
        state_dtypes=[np.dtype(jnp.asarray(s).dtype) for s in state_flat],
        output_shape=tuple(out_struct.shape),
        output_dtype=np.dtype(out_struct.dtype),
        state_treedef=state_treedef,
    )
    return step, sig


def step_signature(
    model, input_shape, *, batch=1, dtype=jnp.float32
) -> LiteRTStepSignature:
    """Describe the flat tensor I/O of ``model``'s single-timestep step.

    Does **not** require TensorFlow — it only traces shapes/dtypes with JAX, so
    callers can plan on-device state seeding/threading without running a
    conversion. See :class:`LiteRTStepSignature`.

    :model: A spyx neuron or :class:`spyx.nn.Sequential` implementing
        ``(x_t, state) -> (out, new_state)`` and exposing ``initial_state``.
    :input_shape: Per-timestep input feature shape, *excluding* batch
        (e.g. ``(8,)`` for a length-8 input vector).
    :batch: Batch dimension of the exported step. Defaults to 1 (phone use).
    :dtype: Input/compute dtype. Defaults to ``jnp.float32``.
    """
    _, sig = _build_step(model, batch, input_shape, dtype)
    return sig


def to_litert(model, input_shape, *, batch=1, dtype=jnp.float32) -> bytes:
    """Export one timestep of a spiking model to a LiteRT (TFLite) flatbuffer.

    Converts the feed-forward step ``(x_t, state) -> (out, new_state)`` — no
    temporal scan — to a ``.tflite`` model whose flat signature is
    ``step(x_t, *state_flat) -> (out, *new_state_flat)``. The phone application
    runs the temporal loop, calling this kernel once per timestep and threading
    ``new_state_i`` back in as ``state_i``. Pair with :func:`step_signature` to
    learn the flat state layout (order/shapes/dtypes) and to seed zeros.

    Only the forward Heaviside spike is exported; the surrogate gradient is
    training-only and irrelevant to on-device inference.

    Requires TensorFlow (``pip install "tensorflow>=2.16"``); it is imported lazily
    here so importing this module does not need TF.

    :model: A spyx neuron or :class:`spyx.nn.Sequential` implementing
        ``(x_t, state) -> (out, new_state)`` and exposing ``initial_state``.
    :input_shape: Per-timestep input feature shape, *excluding* batch
        (e.g. ``(8,)``).
    :batch: Batch dimension of the exported step. Defaults to 1.
    :dtype: Input/compute dtype. Defaults to ``jnp.float32``.
    :return: The serialized ``.tflite`` flatbuffer as ``bytes``.
    """
    try:
        import tensorflow as tf  # noqa: PLC0415  # ty: ignore[unresolved-import]
        from jax.experimental import jax2tf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "spyx.experimental.litert.to_litert requires TensorFlow for the "
            "jax2tf -> TFLite conversion. Install it with "
            '`pip install "tensorflow>=2.16"`. (On-device inference only needs the '
            "lighter tflite-runtime / ai-edge-litert, not full TensorFlow.)"
        ) from exc

    step, sig = _build_step(model, batch, input_shape, dtype)

    # jax2tf lowers the pure step to a tf.function; wrap with a concrete
    # signature over flat float tensors so TFLite sees fixed shapes. jax2tf
    # emits XLA ops by default (its `enable_xla=True` default), which the
    # SELECT_TF_OPS fallback below covers during TFLite conversion.
    tf_step = tf.function(
        jax2tf.convert(step),
        autograph=False,
    )

    input_specs = [tf.TensorSpec(sig.input_shape, tf.as_dtype(sig.input_dtype))]
    input_specs += [
        tf.TensorSpec(shape, tf.as_dtype(dt))
        for shape, dt in zip(sig.state_shapes, sig.state_dtypes, strict=True)
    ]
    concrete = tf_step.get_concrete_function(*input_specs)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], tf_step)
    # jax2tf with enable_xla emits a few ops (e.g. XlaCallModule / select-style
    # ops from the Heaviside) that the builtin TFLite op set may not cover;
    # allow the SELECT_TF_OPS fallback so conversion is robust across models.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    return converter.convert()
