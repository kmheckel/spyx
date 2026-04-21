"""Phasor and Spiking Phasor networks for Spyx.

Implements the deep phasor architecture of Bybee, Frady & Sommer (2022,
arXiv 2106.11908) on top of Flax NNX, taking advantage of JAX's native complex
dtype so the complex-valued forward and backward passes are handled by the
autodiff engine without manual real/imag splitting.

The two halves of a phasor pipeline:

* **Continuous (training-time)**: complex-valued layers with phases on the unit
  circle. ``PhasorLinear`` does ``z_out = W @ z_in + b`` with ``W: complex64``;
  ``PhasorActivation`` projects back onto the unit circle, mimicking the
  threshold function of the Frady/Sommer attractor model.

* **Spiking (inference-time)**: each phase is mapped to a single spike inside a
  cycle of length ``T``. The companion helpers ``phase_to_spikes`` and
  ``spikes_to_phase`` make it possible to run the same trained weights on a
  spiking substrate via :class:`SpikingPhasor`.

This module is intentionally minimal and targets the pattern documented in
``docs/examples/phasor/phasor_intro.ipynb`` (issue #38).

.. note::
    Parameters that enter a complex-valued forward pass are stored as
    separate ``kernel_re`` + ``kernel_im`` ``float32`` tensors and assembled
    on each call (see :class:`PhasorLinear`). This sidesteps the JAX
    Wirtinger-conjugate-gradient surprise that bit the first iteration of
    this module, and lets you train phasor networks with a stock
    ``optax.adam`` + ``nnx.Optimizer`` loop.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# encoder / decoder helpers
# ---------------------------------------------------------------------------


def real_to_phasor(x: jax.Array, scale: float = jnp.pi) -> jax.Array:
    """Encode real-valued inputs as unit-magnitude phasors.

    Maps each scalar ``x`` to ``e^{i * scale * x}``. With the default
    ``scale = π`` and inputs in ``[0, 1]`` this fills the upper half-circle,
    which keeps the encoding monotonic in ``x`` without aliasing.

    :x: real array of any shape.
    :scale: phase scaling. ``π`` is the natural choice for inputs in [0, 1].
    :return: complex64 array, same shape as ``x``.
    """
    return jnp.exp(1j * scale * x).astype(jnp.complex64)


def phasor_to_real(z: jax.Array) -> jax.Array:
    """Decode phasors to real values via the real component (cos of phase).

    Convenient when feeding a downstream real-valued readout / loss.
    """
    return jnp.real(z)


def phase_of(z: jax.Array) -> jax.Array:
    """Return the phase angle of ``z`` in ``(-π, π]``."""
    return jnp.angle(z)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


def _complex_glorot(key: jax.Array, in_features: int, out_features: int) -> jax.Array:
    """Glorot / Xavier initialiser for a complex-valued matrix.

    Uses the standard fan-in scaling but splits the variance equally across the
    real and imaginary parts (so ``Var(|w|) = 1/in_features``).
    """
    scale = jnp.sqrt(1.0 / (2.0 * in_features))
    k_r, k_i = jax.random.split(key)
    real = jax.random.normal(k_r, (in_features, out_features)) * scale
    imag = jax.random.normal(k_i, (in_features, out_features)) * scale
    return (real + 1j * imag).astype(jnp.complex64)


class PhasorLinear(nnx.Module):
    """Complex-valued dense layer with real/imag parameter storage.

    ``z_out = z_in @ kernel + bias`` where ``kernel = kernel_re + i·kernel_im``
    is reconstructed on each forward pass from two ``float32`` parameters.

    Why not store ``kernel`` as a single ``complex64`` ``nnx.Param``?
    JAX returns the *conjugate* Wirtinger derivative when you take
    ``jax.grad`` of a real-valued loss with respect to a complex parameter.
    Optax is real-arithmetic only and does not unwind the conjugation, which
    caused vanilla ``optax.adam`` steps to drift sideways on the imaginary
    axis in the first iteration of this module. Splitting storage into
    ``kernel_re`` + ``kernel_im`` sidesteps the whole issue: the gradients
    optax sees are always real, and the complex structure shows up only in
    the forward pass. This matches the pattern used by the TF reference in
    ``wilkieolin/phasor_networks``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        complex_kernel = _complex_glorot(rngs.params(), in_features, out_features)
        self.kernel_re = nnx.Param(jnp.real(complex_kernel).astype(jnp.float32))
        self.kernel_im = nnx.Param(jnp.imag(complex_kernel).astype(jnp.float32))
        self.use_bias = use_bias
        if use_bias:
            self.bias_re = nnx.Param(jnp.zeros((out_features,), dtype=jnp.float32))
            self.bias_im = nnx.Param(jnp.zeros((out_features,), dtype=jnp.float32))
        else:
            self.bias_re = None
            self.bias_im = None

    @property
    def kernel(self) -> jax.Array:
        """Complex kernel reconstructed from the real/imag storage."""
        return (self.kernel_re[...] + 1j * self.kernel_im[...]).astype(jnp.complex64)

    @property
    def bias(self) -> jax.Array | None:
        """Complex bias reconstructed from the real/imag storage (or None)."""
        if not self.use_bias:
            return None
        return (self.bias_re[...] + 1j * self.bias_im[...]).astype(jnp.complex64)

    def __call__(self, z: jax.Array) -> jax.Array:
        if z.dtype not in (jnp.complex64, jnp.complex128):
            raise TypeError(
                f"PhasorLinear expects a complex input; got dtype {z.dtype}. "
                "Use spyx.phasor.real_to_phasor(x) on the input first."
            )
        out = z @ self.kernel
        if self.use_bias:
            out = out + self.bias
        return out


class PhasorActivation(nnx.Module):
    """Project complex activations back onto the unit circle.

    This is the "threshold" function of the TPAM attractor model: it discards
    the magnitude and keeps only the phase. ``eps`` prevents division-by-zero
    when an activation collapses to ``0 + 0j`` (rare but possible during early
    training).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, z: jax.Array) -> jax.Array:
        magnitude = jnp.abs(z) + self.eps
        return z / magnitude.astype(z.dtype)


class PhasorReadout(nnx.Module):
    """Map complex hidden states to real-valued logits.

    Implementation: take the real part of a final ``PhasorLinear``. Equivalent
    to projecting each output phasor onto the cosine basis. Works as a drop-in
    replacement for the final ``nnx.Linear`` of a classifier.
    """

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.proj = PhasorLinear(in_features, out_features, use_bias=False, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        return jnp.real(self.proj(z))


class PhasorMLP(nnx.Module):
    """A small phasor MLP: encode -> N x (PhasorLinear -> PhasorActivation) -> readout.

    Convenience constructor for the most common phasor topology.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        if depth < 1:
            raise ValueError("PhasorMLP requires depth >= 1.")
        layers: list[nnx.Module] = [
            PhasorLinear(in_features, hidden_features, rngs=rngs),
            PhasorActivation(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    PhasorLinear(hidden_features, hidden_features, rngs=rngs),
                    PhasorActivation(),
                ]
            )
        # nnx requires container attributes that hold sub-modules to be wrapped
        # so that the pytree machinery can walk through them.
        self.layers = nnx.data(layers)
        self.readout = PhasorReadout(hidden_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        z = real_to_phasor(x) if x.dtype not in (jnp.complex64, jnp.complex128) else x
        for layer in self.layers:
            z = layer(z)
        return self.readout(z)


# ---------------------------------------------------------------------------
# spike <-> phase conversion
# ---------------------------------------------------------------------------


def phase_to_spikes(theta: jax.Array, T: int) -> jax.Array:
    """Convert phases to single-spike-per-cycle spike trains.

    A neuron with phase ``θ ∈ (-π, π]`` fires at timestep ``round((θ + π) /
    (2π) * T)`` within a cycle of ``T`` ticks. The returned tensor has the time
    axis prepended.

    :theta: real array of shape ``(...)``.
    :T: int, number of ticks per cycle.
    :return: float32 array of shape ``(T, ...)``, exactly one ``1.`` per
        ``(time, neuron)`` slice along the time axis.
    """
    if T <= 0:
        raise ValueError(f"T must be positive; got {T}.")
    spike_idx = jnp.floor(((theta + jnp.pi) / (2.0 * jnp.pi)) * T).astype(jnp.int32)
    spike_idx = jnp.clip(spike_idx, 0, T - 1)
    one_hot = jax.nn.one_hot(spike_idx, T, dtype=jnp.float32)  # [..., T]
    return jnp.moveaxis(one_hot, -1, 0)  # [T, ...]


def spikes_to_phase(spike_train: jax.Array, T: Optional[int] = None) -> jax.Array:
    """Recover phases from a spike train (inverse of :func:`phase_to_spikes`).

    For each unit, computes the spike-time centroid weighted by the spike
    train, then maps it back to a phase in ``(-π, π]``. If a unit emits no
    spikes the centroid is undefined; we return ``0`` in that case.

    :spike_train: shape ``(T, ...)``.
    :T: cycle length; defaults to ``spike_train.shape[0]``.
    :return: real array of shape ``(...)``.
    """
    if T is None:
        T = spike_train.shape[0]
    times = jnp.arange(T, dtype=spike_train.dtype)
    times = times.reshape((T,) + (1,) * (spike_train.ndim - 1))
    total = jnp.sum(spike_train, axis=0)
    weighted = jnp.sum(times * spike_train, axis=0)
    centroid = jnp.where(total > 0, weighted / jnp.where(total > 0, total, 1.0), 0.0)
    return ((centroid / T) * 2.0 * jnp.pi) - jnp.pi


class SpikingPhasor(nnx.Module):
    """Spiking inference wrapper around a single :class:`PhasorLinear`.

    The forward pass:

      1. Takes a batched spike train ``[T, B, in_features]``.
      2. Recovers per-unit phases via :func:`spikes_to_phase`.
      3. Multiplies the resulting unit-magnitude phasors through ``PhasorLinear``.
      4. Applies :class:`PhasorActivation` to renormalise to the unit circle.
      5. Re-emits a spike train ``[T, B, out_features]`` via :func:`phase_to_spikes`.

    This makes a phasor layer drop-in compatible with ``spyx.nn.Sequential``
    + ``spyx.nn.run`` for spike-domain evaluation. For training, use
    ``PhasorLinear`` directly on the complex domain (much faster) and only
    convert to ``SpikingPhasor`` at deployment.
    """

    def __init__(self, phasor_layer: PhasorLinear, period_T: int):
        if period_T <= 0:
            raise ValueError(f"period_T must be positive; got {period_T}.")
        self.layer = phasor_layer
        self.activation = PhasorActivation()
        self.T = period_T

    def __call__(self, spike_train: jax.Array) -> jax.Array:
        theta = spikes_to_phase(spike_train, self.T)
        z_in = jnp.exp(1j * theta).astype(jnp.complex64)
        z_out = self.activation(self.layer(z_in))
        return phase_to_spikes(jnp.angle(z_out), self.T)


__all__ = [
    "PhasorLinear",
    "PhasorActivation",
    "PhasorReadout",
    "PhasorMLP",
    "SpikingPhasor",
    "real_to_phasor",
    "phasor_to_real",
    "phase_of",
    "phase_to_spikes",
    "spikes_to_phase",
]
