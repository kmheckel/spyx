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

from .axn import superspike

# Module-level singleton for the default surrogate activation (avoids B008).
_DEFAULT_ACTIVATION = superspike()

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
        return (self.bias_re[...] + 1j * self.bias_im[...]).astype(jnp.complex64)  # ty: ignore[not-subscriptable]  # guarded by use_bias

    def __call__(self, z: jax.Array) -> jax.Array:
        if z.dtype not in (jnp.complex64, jnp.complex128):
            raise TypeError(
                f"PhasorLinear expects a complex input; got dtype {z.dtype}. "
                "Use spyx.phasor.real_to_phasor(x) on the input first."
            )
        out = z @ self.kernel
        if self.use_bias:
            out = out + self.bias  # ty: ignore[unsupported-operator]  # guarded by use_bias
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


# ---------------------------------------------------------------------------
# resonate-and-fire spiking neuron
# ---------------------------------------------------------------------------


def _resonate_associative_op(element_i, element_j):
    """Associative combine for a first-order *complex* linear recurrence.

    Each element is a pair ``(a, b)`` standing for the affine map
    ``z -> a * z + b`` on the complex plane. Composing two such maps (apply
    ``i`` then ``j``) is again affine, ``z -> (a_j a_i) z + (a_j b_i + b_j)``,
    so this operator is associative and usable with
    :func:`jax.lax.associative_scan`. It is the complex twin of
    :func:`spyx.nn._leaky_associative_op`; ``a`` here is the complex oscillator
    pole ``exp(dt(-lambda + i*omega))`` instead of a real leak.
    """
    a_i, b_i = element_i
    a_j, b_j = element_j
    return a_j * a_i, a_j * b_i + b_j


def _inverse_softplus(y: jax.Array) -> jax.Array:
    """Inverse of ``softplus``: return ``x`` such that ``softplus(x) == y``.

    Used to initialise the raw decay parameter so that ``softplus(raw)`` equals
    a user-requested positive decay ``lambda``. Requires ``y > 0``.
    """
    return jnp.log(jnp.expm1(y))


class ResonateFire(nnx.Module):
    r"""Resonate-and-fire neuron: the complex/oscillatory sibling of ``PSU_LIF``.

    .. note::
       **Experimental.** Its supported entry point is
       :class:`spyx.experimental.ResonateFire`; the API may change without a
       deprecation cycle. It is defined here for locality with the phasor layers.


    A resonate-and-fire neuron carries a **complex** membrane that behaves as a
    damped harmonic oscillator. Written reset-free, its subthreshold dynamics
    are a *complex linear recurrence*

    .. math::
        z_t = a \, z_{t-1} + x_t , \qquad a = e^{\,\mathrm{dt}\,(-\lambda + i\,\omega)} ,

    with per-unit decay :math:`\lambda \ge 0` and angular frequency
    :math:`\omega`. The real input current ``x_t`` is injected into the *real*
    part of the membrane. Because there is no reset, the recurrence stays
    linear, so exactly like :class:`spyx.nn.PSU_LIF` it can be evaluated with
    :func:`jax.lax.associative_scan` in :math:`O(\log T)` parallel depth -- only
    now the scan runs over a *complex* pole ``a`` instead of a real leak.

    Spikes are emitted by a pointwise surrogate threshold on the real part of
    the oscillator, :math:`s_t = \sigma(\Re(z_t) - \text{threshold})`. The rule
    is reset-free so the linear recurrence -- and therefore the parallel scan --
    is preserved.

    Stability: the pole magnitude is ``|a| = exp(-dt * lambda)``. Storing the
    decay through a ``softplus`` keeps :math:`\lambda \ge 0`, hence
    :math:`|a| \le 1` and the oscillation never grows.

    Parameters that enter the complex pole (``lambda``, ``omega``) are stored as
    **real** ``float32`` ``nnx.Param`` tensors, mirroring :class:`PhasorLinear`:
    the complex structure appears only in the forward pass, so a stock
    ``optax`` + ``jax.grad`` loop over a real loss trains them without the
    Wirtinger-conjugate surprise.

    Two execution modes are provided and are numerically identical:

    * :meth:`__call__` -- one reset-free timestep ``(x, z) -> (spikes, z)`` with
      ``z = a * z + x``; a drop-in for :func:`spyx.nn.run` / :class:`Sequential`.
    * :meth:`parallel` -- the whole time-major sequence at once via an
      associative scan over the complex pole, :math:`O(\log T)` depth.

    Because both modes use the *same* pole and surrogate and integrate the input
    *before* spiking, scanning :meth:`__call__` over ``x`` reproduces
    :meth:`parallel` exactly.
    """

    def __init__(
        self,
        hidden_shape: tuple,
        lambda_init=None,
        omega_init=None,
        threshold: float = 1.0,
        dt: float = 1.0,
        activation=None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        :hidden_shape: Per-unit shape of the layer.
        :lambda_init: Membrane decay ``>= 0``. Scalar constant if provided, else
            a learnable per-unit initialisation. Stored through ``softplus`` so
            the effective decay is always non-negative.
        :omega_init: Angular frequency of the oscillator. Scalar constant if
            provided, else a learnable per-unit initialisation.
        :threshold: Real firing threshold on ``Re(z)``. Defaults to 1.
        :dt: Integration timestep entering the pole ``exp(dt(-lambda+i*omega))``.
        :activation: :class:`spyx.axn.Axon` surrogate spike; defaults to
            ``superspike``.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive; got {dt}.")
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.dt = dt
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        # Raw decay parameter; effective lambda = softplus(raw) >= 0 so |a| <= 1.
        if lambda_init is None:
            # Small positive decays: softplus(N(0.5, 0.25)) ~ light damping.
            raw = (
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
            self.raw_lambda = nnx.Param(raw.astype(jnp.float32))
        else:
            self.raw_lambda = nnx.Param(
                _inverse_softplus(jnp.full((), float(lambda_init))).astype(jnp.float32)
            )

        if omega_init is None:
            # Spread frequencies around ~1 rad/step so units resonate distinctly.
            omega = (
                nnx.initializers.truncated_normal(stddev=0.5)(
                    rngs.params(), self.hidden_shape
                )
                + 1.0
            )
            self.omega = nnx.Param(omega.astype(jnp.float32))
        else:
            self.omega = nnx.Param(jnp.full((), float(omega_init)))

    @property
    def decay(self) -> jax.Array:
        """Effective non-negative decay ``lambda = softplus(raw_lambda)``."""
        return jax.nn.softplus(self.raw_lambda[...])

    @property
    def a(self) -> jax.Array:
        """Complex oscillator pole ``a = exp(dt(-lambda + i*omega))``.

        The magnitude ``|a| = exp(-dt * lambda) <= 1`` guarantees stability.
        """
        exponent = self.dt * (-self.decay + 1j * self.omega[...])
        return jnp.exp(exponent).astype(jnp.complex64)

    def __call__(self, x, z):
        """One reset-free timestep.

        :x: real input current from the previous layer, broadcastable to ``z``.
        :z: complex64 membrane state.

        Injects ``x`` into the real part of the membrane and advances the
        complex recurrence ``z = a * z + x`` (no reset), then emits a surrogate
        spike on ``Re(z)`` so that scanning this method matches :meth:`parallel`.
        """
        a = self.a
        z = a * z + x.astype(z.dtype)
        spikes = self.spike(jnp.real(z) - self.threshold)
        return spikes, z

    def parallel(self, x):
        r"""Score a whole time-major sequence with an associative scan.

        :x: real input with shape ``[Time, Batch, ...]``.
        :return: spikes with shape ``[Time, Batch, ...]``.

        Computes the full complex membrane trace ``z_t = a * z_{t-1} + x_t``
        (with ``z_{-1} = 0``) via :func:`jax.lax.associative_scan` over the time
        axis in :math:`O(\log T)` depth, then applies the surrogate spike
        pointwise on ``Re(z)``.
        """
        a = self.a
        xc = x.astype(jnp.complex64)
        # Broadcast the (scalar or per-unit) complex pole to every element so the
        # linear-recurrence coefficient a_t == a everywhere along the time axis.
        A = jnp.broadcast_to(a, xc.shape)
        _, z = jax.lax.associative_scan(_resonate_associative_op, (A, xc), axis=0)
        return self.spike(jnp.real(z) - self.threshold)

    def initial_state(self, batch_size):
        """Return complex64 zeros of shape ``(batch_size,) + hidden_shape``."""
        return jnp.zeros((batch_size,) + tuple(self.hidden_shape), dtype=jnp.complex64)


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
