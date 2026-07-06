from collections.abc import Sequence
from typing import Any, Optional, Protocol, Union, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx

from .axn import superspike

# Module-level singleton for default activation to avoid B008
_DEFAULT_ACTIVATION = superspike()


@runtime_checkable
class StatefulLayer(Protocol):
    """The contract every Spyx neuron/stateful layer follows.

    This is a *documentation aid*, not an enforced base class — Spyx neurons
    are plain :class:`flax.nnx.Module` subclasses and do **not** inherit from
    this Protocol. It captures, in one place, the two-method contract that
    :func:`run`, :class:`Sequential`, and :mod:`spyx.nir` rely on:

    * ``initial_state(batch_size)`` returns a fresh zero state for a batch of
      the given size (the leading axis is the batch dimension).
    * ``__call__(x, state)`` advances one timestep, returning
      ``(out, new_state)`` where ``new_state`` has the same structure as
      ``state`` so it can be threaded through :func:`jax.lax.scan`.

    Because it is ``@runtime_checkable``, ``isinstance(layer, StatefulLayer)``
    checks for the *presence* of these methods (not their signatures), which is
    handy in tests. New neurons should match this shape so they drop straight
    into :class:`Sequential` and :func:`run`.
    """

    def initial_state(self, batch_size: int) -> Any: ...

    def __call__(self, x: Any, state: Any) -> tuple[Any, Any]: ...


class ALIF(nnx.Module):
    """
    Adaptive LIF Neuron based on the model used in LSNNs:

    Bellec, G., Salaj, D., Subramoney, A., Legenstein, R. & Maass, Maass, W.
    Long short- term memory and learning-to-learn in networks of spiking neurons.
    32nd Conference on Neural Information Processing Systems (2018).

    """

    def __init__(
        self,
        hidden_shape,
        beta=None,
        gamma=None,
        threshold=1,
        activation=None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        :hidden_shape: Hidden layer shape.
        :beta: Membrane decay/inverse time constant.
        :gamma: Threshold adaptation constant.
        :threshold: Neuron firing threshold.
        :activation: spyx.axn.Axon object determining forward function and surrogate gradient function.
        """
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

        if gamma is None:
            self.gamma = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.gamma = nnx.Param(jnp.full((), gamma))

    def __call__(self, x, VT):
        """
        :x: Tensor from previous layer.
        :VT: Neuron state vector.
        """
        V, T = jnp.split(VT, 2, -1)

        beta = jnp.clip(self.beta[...], 0, 1)
        gamma = jnp.clip(self.gamma[...], 0, 1)

        # calculate whether spike is generated, and update membrane potential
        thresh = self.threshold + T
        spikes = self.spike(V - thresh)  # T is the dynamic threshold adaptation
        V = beta * V + x - spikes * thresh
        T = gamma * T + (1 - gamma) * spikes

        VT = jnp.concatenate([V, T], axis=-1)
        return spikes, VT

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2 * s for s in self.hidden_shape))


class LI(nnx.Module):
    """
    Leaky-Integrate (Non-spiking) neuron model.
    """

    def __init__(self, layer_shape, beta=None, *, rngs: nnx.Rngs):
        """
        :layer_shape: Shape of the layer.
        :beta: Decay rate on membrane potential (voltage).
        """
        self.layer_shape = layer_shape
        if beta is None:
            self.beta = nnx.Param(jnp.full(layer_shape, 0.8))
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, Vin):
        """
        :x: Input tensor from previous layer.
        :Vin: Neuron state tensor.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        Vout = beta * Vin + x
        return Vout, Vout

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.layer_shape)


class IF(nnx.Module):
    """
    Integrate and Fire neuron model.
    """

    def __init__(self, hidden_shape, threshold=1, activation=None, *, rngs=None):
        """
        :hidden_shape: Shape of the layer.
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.activation function.
        :rngs: Accepted and ignored — IF is parameterless, but taking ``rngs``
            keeps it drop-in interchangeable with the parametric neurons
            (``LIF``, ``CuBaLIF``, ...) that require it.
        """
        del rngs  # parameterless; accepted only for a uniform constructor
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

    def __call__(self, x, V):
        """
        :x: Vector coming from previous layer.
        :V: Neuron state tensor.
        """
        spikes = self.spike(V - self.threshold)
        V = V + x - spikes * self.threshold
        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)


class LIF(nnx.Module):
    """
    Leaky Integrate and Fire neuron model.
    """

    def __init__(
        self,
        hidden_shape: tuple,
        beta=None,
        threshold=1.0,
        activation=None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        :hidden_shape: Shape of the layer.
        :beta: decay rate.
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.axn.Axon object.
        """
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, V):
        """
        :x: input vector coming from previous layer.
        :V: neuron state tensor.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        spikes = self.spike(V - self.threshold)
        V = beta * V + x - spikes * self.threshold
        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)


def _leaky_associative_op(element_i, element_j):
    """Associative combine for a first-order linear leaky recurrence.

    Each element is a pair ``(A, b)`` standing for the affine map
    ``V -> A * V + b``. Composing two such maps (apply ``i`` then ``j``)
    is itself affine, ``V -> (A_j A_i) V + (A_j b_i + b_j)``, so this
    operator is associative and usable with ``jax.lax.associative_scan``.
    Adapted from the parallel-scan formulation in the S5 SSM paper.
    """
    A_i, b_i = element_i
    A_j, b_j = element_j
    return A_j * A_i, A_j * b_i + b_j


class PSU_LIF(nnx.Module):
    r"""Parallel Spiking Unit LIF: a reset-free leaky integrate-and-fire neuron.

    .. note::
       **Experimental.** Its supported entry point is
       :class:`spyx.experimental.PSU_LIF`; the API may change without a
       deprecation cycle. It is defined here for locality with the other neurons.

    A standard :class:`LIF` subtracts a reset ``spikes * threshold`` from the
    membrane every step, which couples each timestep to the (nonlinear) spike
    of the previous step and forces a strictly sequential ``O(T)`` scan.
    Dropping the reset turns the membrane into a pure linear leaky integrator,

    .. math::
        V_t = \beta \, V_{t-1} + x_t ,

    which is a first-order *associative* recurrence and can therefore be
    evaluated with :func:`jax.lax.associative_scan` in ``O(\log T)`` parallel
    depth on an accelerator. Spikes are a pointwise surrogate threshold applied
    to the whole membrane trace, :math:`s_t = \sigma(V_t - \text{threshold})`.

    Removing the reset is a deliberate accuracy/parallelism trade-off: the
    neuron never depresses after firing, so it can fire on consecutive steps
    while a well-tuned integration window keeps activity bounded. In exchange
    the sequence can be scored in logarithmic instead of linear depth.

    Two execution modes are provided and are numerically identical:

    * :meth:`__call__` -- one reset-free timestep ``(x, V) -> (spikes, V)``
      with ``V = beta * V + x``; a drop-in for :func:`spyx.nn.run`,
      :class:`Sequential`, and NIR, exactly like :class:`LIF`.
    * :meth:`parallel` -- the whole time-major sequence at once via an
      associative scan over the leak, ``O(\log T)`` depth.

    Because both modes use the *same* clipped ``beta`` and the *same* surrogate,
    and :meth:`__call__` integrates the input *before* spiking, scanning
    :meth:`__call__` over ``x`` reproduces :meth:`parallel` exactly.
    """

    def __init__(
        self,
        hidden_shape: tuple,
        beta=None,
        threshold=1.0,
        activation=None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        :hidden_shape: Shape of the layer.
        :beta: decay rate. Scalar if provided, else learnable per-unit init.
        :threshold: firing threshold. Defaults to 1.
        :activation: spyx.axn.Axon object determining the surrogate spike.
        """
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, V):
        """One reset-free timestep.

        :x: input vector coming from previous layer.
        :V: neuron state tensor.

        Integrates the input into the membrane (``V = beta * V + x``, no
        reset), then emits a surrogate spike on the updated membrane so that
        scanning this method matches :meth:`parallel` exactly.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        V = beta * V + x
        spikes = self.spike(V - self.threshold)
        return spikes, V

    def parallel(self, x):
        r"""Score a whole time-major sequence with an associative scan.

        :x: input with shape ``[Time, Batch, ...]``.
        :return: spikes with shape ``[Time, Batch, ...]``.

        Computes the full membrane trace ``V_t = beta * V_{t-1} + x_t`` (with
        ``V_{-1} = 0``) via :func:`jax.lax.associative_scan` over the time axis
        in ``O(\log T)`` depth, then applies the surrogate spike pointwise.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        # Broadcast the (scalar or per-unit) leak to every (Time, Batch, ...)
        # element so the linear-recurrence coefficient A_t == beta everywhere.
        A = jnp.broadcast_to(beta, x.shape)
        _, V = jax.lax.associative_scan(_leaky_associative_op, (A, x), axis=0)
        return self.spike(V - self.threshold)

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)


class CuBaLIF(nnx.Module):
    def __init__(
        self,
        hidden_shape,
        alpha=None,
        beta=None,
        threshold=1,
        activation=None,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        if alpha is None:
            self.alpha = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.alpha = nnx.Param(jnp.full((), alpha))

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, VI):
        V, current_I = jnp.split(VI, 2, -1)

        alpha = jnp.clip(self.alpha[...], 0, 1)
        beta = jnp.clip(self.beta[...], 0, 1)

        # calculate whether spike is generated, and update membrane potential.
        # The subtractive reset is applied exactly ONCE, before the leaky
        # integration of the synaptic current (mirroring RCuBaLIF); applying
        # `- reset` a second time on the integrated membrane would double-count
        # the reset.
        spikes = self.spike(V - self.threshold)
        V = V - spikes * self.threshold
        current_I = alpha * current_I + x
        V = beta * V + current_I

        VI = jnp.concatenate([V, current_I], axis=-1)
        return spikes, VI

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2 * v for v in self.hidden_shape))


class RIF(nnx.Module):
    """
    Recurrent Integrate and Fire neuron model.
    """

    def __init__(self, hidden_shape, threshold=1, activation=None, *, rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        # recurrent weight matrix
        self.recurrent_w = nnx.Param(
            nnx.initializers.truncated_normal()(
                rngs.params(), self.hidden_shape + self.hidden_shape
            )
        )

    def __call__(self, x, V):
        """
        :x: Vector coming from previous layer.
        :V: Neuron state tensor.
        """
        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V - self.threshold)
        feedback = spikes @ self.recurrent_w[...]
        V = V + x + feedback - spikes * self.threshold

        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)


class RLIF(nnx.Module):
    """
    Recurrent LIF Neuron.
    """

    def __init__(
        self, hidden_shape, beta=None, threshold=1, activation=None, *, rngs: nnx.Rngs
    ):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        # recurrent weight matrix
        self.recurrent_w = nnx.Param(
            nnx.initializers.truncated_normal()(
                rngs.params(), self.hidden_shape + self.hidden_shape
            )
        )

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, V):
        """
        :x: The input data/latent vector from another layer.
        :V: The state tensor.
        """
        beta = jnp.clip(self.beta[...], 0, 1)

        spikes = self.spike(V - self.threshold)
        feedback = spikes @ self.recurrent_w[...]
        V = beta * V + x + feedback - spikes * self.threshold

        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)


class RCuBaLIF(nnx.Module):
    def __init__(
        self,
        hidden_shape,
        alpha=None,
        beta=None,
        threshold=1,
        activation=None,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        # recurrent weight matrix
        self.recurrent_w = nnx.Param(
            nnx.initializers.truncated_normal()(
                rngs.params(), self.hidden_shape + self.hidden_shape
            )
        )

        if alpha is None:
            self.alpha = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.alpha = nnx.Param(jnp.full((), alpha))

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, VI):
        V, current_I = jnp.split(VI, 2, -1)

        alpha = jnp.clip(self.alpha[...], 0, 1)
        beta = jnp.clip(self.beta[...], 0, 1)

        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V - self.threshold)
        V = V - spikes * self.threshold
        feedback = spikes @ self.recurrent_w[...]
        current_I = alpha * current_I + x + feedback
        V = beta * V + current_I

        VI = jnp.concatenate([V, current_I], axis=-1)
        return spikes, VI

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2 * v for v in self.hidden_shape))


class ActivityRegularization(nnx.Module):
    """
    Track the cumulative number of spikes emitted per neuron per batch.

    The running spike count is threaded through :func:`spyx.nn.run` (and
    :class:`Sequential`) as part of the scan carry, exactly like a neuron's
    membrane state: :meth:`initial_state` seeds a zero buffer and each
    :meth:`__call__` returns the incoming spikes unchanged plus the updated
    count. The final accumulated count comes back as this layer's entry in the
    ``final_state`` returned by ``run``, and can be fed to
    ``spyx.fn.silence_reg`` / ``spyx.fn.sparsity_reg`` for activity penalties.

    Threading the count through the carry (rather than mutating an
    ``nnx.Variable`` in place) is what lets it accumulate inside the raw
    ``jax.lax.scan`` used by :func:`spyx.nn.run`, where in-place variable
    mutation raises ``TraceContextError``.
    """

    def __init__(self, hidden_shape, batch_size=1, dtype=jnp.float32):
        """
        :hidden_shape: Per-neuron shape of the layer being regularized.
        :batch_size: Leading batch dimension of the spike-count buffer.
        :dtype: Storage dtype for the spike-count buffer.
        """
        self.hidden_shape = (
            tuple(hidden_shape)
            if not isinstance(hidden_shape, int)
            else (hidden_shape,)
        )
        self.dtype = dtype

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape, dtype=self.dtype)

    def __call__(self, spikes, spike_count):
        """
        :spikes: Spikes emitted by the previous layer at this timestep.
        :spike_count: Running per-neuron spike count carried through the scan.
        :return: ``(spikes, spike_count + spikes)`` -- the spikes pass through
            unchanged while the count accumulates.
        """
        return spikes, spike_count + spikes.astype(self.dtype)


def PopulationCode(num_classes):
    def _pop_code(x):
        return jnp.sum(jnp.reshape(x, (-1, num_classes)), axis=-1)

    return jax.jit(_pop_code)


def _infer_shape(
    x: jax.Array,
    size: Union[int, Sequence[int]],
    channel_axis: Optional[int] = -1,
) -> tuple[int, ...]:
    """Infer shape for pooling window or strides."""
    if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
        raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
    if channel_axis and channel_axis < 0:
        channel_axis = x.ndim + channel_axis

    if isinstance(size, int):
        return (1,) + tuple(size if d != channel_axis else 1 for d in range(1, x.ndim))
    elif len(size) < x.ndim:
        # Distribute the window over the spatial axes — every axis except the
        # batch axis (0) and the channel axis — respecting channel_axis so that
        # channels-last (B, H, W, C) pools H/W, not the channels.
        spatial = [d for d in range(x.ndim) if d != 0 and d != channel_axis]
        if len(size) == len(spatial):
            window = [1] * x.ndim
            for ax, sz in zip(spatial, size, strict=True):
                window[ax] = sz
            return tuple(window)
        # Fallback: treat leading extra dims as batch (channels-first layout).
        return (1,) * (x.ndim - len(size)) + tuple(size)
    else:
        assert x.ndim == len(size)
        return tuple(size)


def sum_pool(
    value: jax.Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    channel_axis: Optional[int] = -1,
) -> jax.Array:
    """Sum pool."""
    if padding not in ("SAME", "VALID"):
        raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

    window_shape = _infer_shape(value, window_shape, channel_axis)
    strides = _infer_shape(value, strides, channel_axis)

    return jax.lax.reduce_window(
        value, 0.0, jax.lax.add, window_shape, strides, padding
    )


class SumPool(nnx.Module):
    """Sum pool."""

    def __init__(
        self,
        window_shape: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]],
        padding: str,
        channel_axis: Optional[int] = -1,
    ):
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding
        self.channel_axis = channel_axis

    def __call__(self, value: jax.Array) -> jax.Array:
        return sum_pool(
            value, self.window_shape, self.strides, self.padding, self.channel_axis
        )


class Sequential(nnx.Sequential):
    """
    A Sequential container that supports passing state through its layers.
    """

    def initial_state(self, batch_size):
        return [
            layer.initial_state(batch_size) if hasattr(layer, "initial_state") else None  # ty: ignore[call-non-callable]  # untyped module list
            for layer in self.layers
        ]

    def __call__(self, x, state):
        new_state = []
        for layer, s in zip(self.layers, state, strict=True):
            if s is not None:
                x, s_new = layer(x, s)
                new_state.append(s_new)
            else:
                x = layer(x)
                new_state.append(None)
        return x, new_state


class Flatten(nnx.Module):
    """Flatten every non-batch dimension of a per-timestep input.

    Stateless: maps ``x`` of shape ``(B, ...)`` to ``(B, prod(...))``. It has no
    ``initial_state``, so :class:`Sequential` runs it in stateless mode. Used by
    :mod:`spyx.nir` to represent NIR ``Flatten`` nodes; ``flax.nnx`` has no
    built-in flatten layer.
    """

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)


def run(model, x, state=None, *, batch_major=False):
    """
    Execute a model over a sequence of inputs using jax.lax.scan.

    :model: A stateful Flax NNX Module, typically :class:`Sequential` or a
        Spyx neuron following the :class:`StatefulLayer` contract. It must
        either take ``(x_t, state) -> (out, next_state)`` or expose an
        ``initial_state(batch_size)`` method (or both). Plain stateless modules
        like ``nnx.Linear`` don't fit the contract — wrap them in a
        :class:`Sequential` with at least one stateful layer, or use
        ``jax.vmap`` if you just want to apply the module per timestep.
    :x: Input data. By default this is **time-major** ``[Time, Batch, ...]``
        (``jax.lax.scan`` walks the leading axis). Pass ``batch_major=True`` if
        your data is ``[Batch, Time, ...]`` instead.
    :state: Initial state for the model. If None, ``model.initial_state`` is
        consulted; if the model has no ``initial_state`` and no state is
        supplied explicitly, a clear error is raised.
    :batch_major: When ``True``, ``x`` is treated as ``[Batch, Time, ...]``:
        it is transposed to time-major internally for the scan and the outputs
        are transposed back to ``[Batch, Time, ...]``. Default ``False``
        preserves the historical time-major behaviour.
    :return: ``(outputs, final_state)``. ``outputs`` is time-major
        ``[Time, Batch, ...]`` by default, or ``[Batch, Time, ...]`` when
        ``batch_major=True``.

    .. note::
        **Mind the time axis when computing losses.** ``run`` is time-major by
        default (time on axis 0), whereas the :mod:`spyx.fn` losses/metrics
        default to ``time_axis=1`` (batch-major, ``[Batch, Time, Classes]``).
        Feeding time-major ``run`` outputs straight into an ``fn`` loss reduces
        over the *batch* axis instead of *time* — silently wrong, and
        undetectable when ``Time == Batch``. Pick one of:

        * call ``run(..., batch_major=True)`` so outputs are ``[Batch, Time,
          ...]`` and line up with the ``fn`` default, or
        * keep time-major and pass ``time_axis=0`` to the ``spyx.fn`` factory.
    """

    if batch_major:
        # [Batch, Time, ...] -> [Time, Batch, ...] for the scan.
        x = jnp.swapaxes(x, 0, 1)

    if state is None:
        if not hasattr(model, "initial_state"):
            raise TypeError(
                "spyx.nn.run: the given model has no `initial_state` method "
                "and no explicit `state=` was provided. run() scans a stateful "
                "(x, state) -> (out, new_state) module; wrap stateless layers "
                "in spyx.nn.Sequential or use jax.vmap over the time axis."
            )
        batch_size = x.shape[1]
        state = model.initial_state(batch_size)

    def scan_fn(carry, x_t):
        out, next_state = model(x_t, carry)
        return next_state, out

    final_state, outputs = jax.lax.scan(scan_fn, state, x)

    if batch_major:
        # [Time, Batch, ...] -> [Batch, Time, ...] to match the input layout.
        outputs = jnp.swapaxes(outputs, 0, 1)

    return outputs, final_state
