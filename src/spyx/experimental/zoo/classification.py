"""Classification recipe: a surrogate-trained recurrent SNN on temporal spikes.

**Experimental.** Part of :mod:`spyx.experimental.zoo`; the API may change
without a deprecation cycle.

A small recurrent spiking network (``RSNN``: a recurrent LIF hidden layer)
classifies SHD-shaped synthetic spike trains of shape ``[T=16, B, inputs=8]``
into ``classes=4``. Each class activates a distinct pair of input channels at a
higher firing rate, so the temporal signal is learnable. Training is by
surrogate gradient: the recurrent layer uses an :mod:`spyx.axn` arctangent
surrogate, sequences run through :func:`spyx.nn.run`, and the readout is scored
with :func:`spyx.fn.integral_crossentropy`.

Synthetic data only, no downloads; a handful of steps runs in well under a
second on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ... import axn, fn, optimize
from ... import nn as snn

NAME = "classification-rsnn"
APPLICATION = "classification"
METHOD = "surrogate"
ARCHITECTURE = "RSNN"
DESCRIBE = (
    "Recurrent spiking network (recurrent LIF) classifying SHD-shaped synthetic "
    "spike trains into 4 classes, trained by surrogate gradient via spyx.axn / "
    "spyx.optimize."
)

TIME = 16
INPUTS = 8
HIDDEN = 24
CLASSES = 4


def build(rngs: nnx.Rngs) -> nnx.Module:
    """Build the RSNN as a :class:`spyx.nn.Sequential`.

    ``inputs -> Linear -> RLIF (recurrent, arctan surrogate) -> Linear -> LI``.
    The leaky-integrator readout accumulates evidence over time for
    :func:`spyx.fn.integral_crossentropy`.
    """
    return snn.Sequential(
        nnx.Linear(INPUTS, HIDDEN, rngs=rngs),
        snn.RLIF((HIDDEN,), activation=axn.arctan(), rngs=rngs),
        nnx.Linear(HIDDEN, CLASSES, rngs=rngs),
        snn.LI((CLASSES,), rngs=rngs),
    )


def synthetic_batch(key: jax.Array | None = None, batch_size: int = 32) -> tuple:
    """Generate class-conditioned synthetic spike trains.

    Each label ``c`` drives input channels ``2c`` and ``2c+1`` (mod ``INPUTS``)
    at an elevated firing rate over a low background rate.

    :return: ``(spikes, labels)`` with ``spikes`` of shape ``[T, B, INPUTS]``
        (float 0/1) and integer ``labels`` of shape ``[B]``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    k_label, k_spike = jax.random.split(key)
    labels = jax.random.randint(k_label, (batch_size,), 0, CLASSES)

    channels = jnp.stack([(labels * 2) % INPUTS, (labels * 2 + 1) % INPUTS], axis=-1)
    active = jax.nn.one_hot(channels, INPUTS).sum(axis=1)  # (B, INPUTS)
    rate = 0.1 + 0.6 * active[None]  # (1, B, INPUTS) broadcast over time
    draws = jax.random.uniform(k_spike, (TIME, batch_size, INPUTS))
    spikes = (draws < rate).astype(jnp.float32)
    return spikes, labels


# Loss closure over the SNN readout; scored across the (time-major) trace.
_loss_fn = fn.integral_crossentropy(time_axis=0)


def loss(model: nnx.Module, spikes: jax.Array, labels: jax.Array) -> jax.Array:
    """Integral cross-entropy of the SNN readout on a synthetic batch."""
    traces, _ = snn.run(model, spikes)
    return _loss_fn(traces, labels)


def demo(steps: int = 40, *, seed: int = 0, learning_rate: float = 2e-3) -> list[float]:
    """Train a handful of surrogate-gradient steps and return the loss history.

    :steps: number of gradient steps.
    :seed: PRNG seed for model init and data.
    :learning_rate: Adam learning rate.
    :return: list of per-step training losses (should decrease).
    """
    import optax  # lazy import; optax is a core dependency

    key = jax.random.PRNGKey(seed)
    k_model, k_data = jax.random.split(key)
    model = build(nnx.Rngs(k_model))
    spikes, labels = synthetic_batch(k_data)

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    step = optimize.make_train_step(loss)

    history: list[float] = []
    for _ in range(steps):
        history.append(float(step(model, optimizer, spikes, labels)))
    return history
