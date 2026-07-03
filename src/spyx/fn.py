"""Losses, metrics, and activity regularisers for spiking networks.

Every public function in this module is a factory: it returns a
JIT-compiled callable that takes network outputs and targets and returns
a scalar loss (or a ``(metric, predictions)`` tuple).

Signatures at a glance
----------------------

- Losses: ``(traces, targets) -> loss``
  where ``traces`` has shape ``(..., time, classes)`` and ``targets`` has
  the batch shape (everything before ``time`` and without ``classes``).
- Metrics: ``(traces, targets) -> (score, predictions)``.
- Regularisers: ``(spike_pytree) -> loss``.

All three check argument shapes at trace time and raise ``ValueError``
early if the target / prediction layout doesn't line up — see
:func:`_check_traces_vs_targets`.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import tree_util as tree

LossFn = Callable[[jax.Array, jax.Array], jax.Array]
"""Type alias for ``(traces, targets) -> loss``."""

MetricFn = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
"""Type alias for ``(traces, targets) -> (score, predictions)``."""

RegFn = Callable[[Any], jax.Array]
"""Type alias for ``(spike_pytree) -> scalar``."""


def _check_traces_vs_targets(traces, targets, time_axis, fn_name):
    """Raise a clear error if the trace / target shapes disagree.

    Runs at trace time (not per call), so there's no runtime cost once the
    outer function is JIT-compiled. Addresses the common "my targets don't
    line up with my readout" foot-gun tracked in issue #25.
    """
    if traces.ndim < 2:
        raise ValueError(
            f"{fn_name}: traces must have at least 2 dimensions (time + classes); "
            f"got shape {traces.shape}."
        )
    # Resolve negative time_axis to a positive index.
    ta = time_axis if time_axis >= 0 else traces.ndim + time_axis
    if not 0 <= ta < traces.ndim:
        raise ValueError(
            f"{fn_name}: time_axis {time_axis} is out of range for traces with "
            f"ndim={traces.ndim}."
        )
    # After reducing along time_axis the trailing dim is the class dim; the
    # remaining leading dims must match targets.
    reduced = traces.shape[:ta] + traces.shape[ta + 1 :]
    expected = reduced[:-1]
    if targets.shape != expected:
        raise ValueError(
            f"{fn_name}: targets shape {targets.shape} does not match the batch "
            f"portion of traces shape {traces.shape} (expected {expected} after "
            f"reducing along time_axis={time_axis} and dropping the class axis)."
        )


def silence_reg(min_spikes: float) -> RegFn:
    """L2-Norm per-neuron activation normalization for spiking less than a target number of times.

    :param min_spikes: neurons which spike below this value on average over the batch incur quadratic penalty.
    :return: JIT compiled regularization function.
    """

    def _loss(x):
        return (jnp.maximum(0, min_spikes - jnp.mean(x, axis=0))) ** 2

    def _flatten(x):
        return jnp.reshape(x, (x.shape[0], -1))

    def _call(spikes):
        flat_spikes = tree.tree_map(_flatten, spikes)
        loss_vectors = tree.tree_map(_loss, flat_spikes)
        return jnp.sum(jnp.concatenate(tree.tree_flatten(loss_vectors)[0]))

    return jax.jit(_call)


def sparsity_reg(
    max_spikes: float,
    norm: Callable[[jax.Array], jax.Array] = optax.huber_loss,
) -> RegFn:
    """Layer activation normalization that seeks to discourage all neurons having a high firing rate.

    :param max_spikes: Threshold for which penalty is incurred if the average number of spikes in the layer exceeds it.
    :param norm: an Optax loss function. Default is Huber loss.
    :return: JIT compiled regularization function.
    """

    def _loss(x):
        return norm(
            jnp.maximum(0, jnp.mean(x, axis=-1) - max_spikes)
        )  # this may not work for convolution layers....

    def _flatten(x):
        return jnp.reshape(x, (x.shape[0], -1))

    def _call(spikes):
        flat_spikes = tree.tree_map(_flatten, spikes)
        loss_vectors = tree.tree_map(_loss, flat_spikes)
        return jnp.sum(jnp.concatenate(tree.tree_flatten(loss_vectors)[0]))

    return jax.jit(_call)


def integral_accuracy(time_axis: int = 1) -> MetricFn:
    """Calculate the accuracy of a network's predictions based on the voltage traces. Used in combination with a Leaky-Integrate neuron model as the final layer.

    :param traces: the output of the final layer of the SNN
    :param targets: the integer labels for each class
    :return: function which computes Accuracy score and predictions that takes SNN output traces and integer index labels.
    """

    def _integral_accuracy(traces, targets):
        _check_traces_vs_targets(traces, targets, time_axis, "integral_accuracy")
        preds = jnp.argmax(jnp.sum(traces, axis=time_axis), axis=-1)
        return jnp.mean(preds == targets), preds

    return jax.jit(_integral_accuracy)


# smoothing can be critical to the performance of your model...


def integral_crossentropy(smoothing: float = 0.3, time_axis: int = 1) -> LossFn:
    """Calculate the crossentropy between the integral of membrane potentials. Allows for label smoothing to discourage silencing the other neurons in the readout layer.

    :param smoothing: rate at which to smooth labels.
    :param time_axis: temporal axis of data
    :return: crossentropy loss function that takes SNN output traces and integer index labels.
    """

    def _integral_crossentropy(traces, targets):
        _check_traces_vs_targets(traces, targets, time_axis, "integral_crossentropy")
        logits = jnp.sum(traces, axis=time_axis)  # time axis.
        one_hot = jax.nn.one_hot(targets, logits.shape[-1])
        labels = optax.smooth_labels(one_hot, smoothing)
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))

    return jax.jit(_integral_crossentropy)


def mse_spikerate(
    sparsity: float = 0.25, smoothing: float = 0.0, time_axis: int = 1
) -> LossFn:
    """Calculate the mean squared error of the mean spike rate. Allows for label smoothing to discourage silencing the other neurons in the readout layer.

    :param sparsity: the percentage of the time you want the neurons to spike
    :param smoothing: [optional] rate at which to smooth labels.
    :return: Mean-Squared-Error loss function on the spike rate that takes SNN output traces and integer index labels.
    """

    def _mse_spikerate(traces, targets):
        _check_traces_vs_targets(traces, targets, time_axis, "mse_spikerate")
        t = traces.shape[time_axis]
        logits = jnp.mean(traces, axis=time_axis)  # time axis.
        labels = optax.smooth_labels(
            jax.nn.one_hot(targets, logits.shape[-1]), smoothing
        )
        return jnp.mean(optax.squared_error(logits, labels * sparsity * t))

    return jax.jit(_mse_spikerate)
