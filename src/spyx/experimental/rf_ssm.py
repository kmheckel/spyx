"""S5-RF: Resonate-and-Fire as a scaled spiking SSM (experimental).

This module unifies two pieces that already live in Spyx:

* :class:`spyx.phasor.ResonateFire` — a reset-free complex-pole spiking neuron
  whose subthreshold membrane is the linear recurrence
  ``z_t = a z_{t-1} + x_t`` with oscillator pole
  ``a = exp(dt (-lambda + i omega))``. Because the recurrence is linear it is
  exactly parallelisable with :func:`jax.lax.associative_scan`. What it lacks
  is a principled long-range pole *initialisation* and any notion of a reset.

* :class:`spyx.ssm.S5Diag` — a diagonal S4D/S5 layer initialised from the
  HiPPO-LegS eigenvalues ``lambda_n = -1/2 + i pi n`` with a per-unit learnable
  log-step. That init is what lets a diagonal SSM represent long-range
  dependencies out of the box, but ``S5Diag`` is a linear readout, not a
  spiking neuron.

:class:`RFSSM` is the resonate-and-fire neuron given the S5 treatment. It keeps
the ``(x, state) -> (out, new_state)`` neuron contract (so it drops into
:class:`spyx.nn.Sequential` / :func:`spyx.nn.run`) and the ``.parallel(x)``
associative-scan path of ``ResonateFire``/``PSU_LIF``, but:

1. **S5/HiPPO pole init** (S5-RF, arXiv:2504.00719). The complex poles are
   initialised from HiPPO-LegS eigenvalues (or, optionally, the LRU radial
   sampler :func:`spyx.ssm._init_lru_eigenvalues`) with a per-unit learnable
   log-step ``log_dt``, rather than the ad-hoc ``lambda~0.5, omega~1`` defaults
   that plain ``ResonateFire`` ships with. This places the oscillators on the
   HiPPO spectrum so the neuron starts life able to integrate over long
   horizons.

2. **Decoupled reset** (PRF, arXiv:2410.03530). A resonate-and-fire neuron
   would ordinarily subtract a reset from the state each time it spikes, which
   makes the recurrence *state-dependent* and *nonlinear* — destroying the
   parallel scan. Following the parallel-resonate-and-fire recipe we *decouple*
   the reset onto the **imaginary axis**: the spike is read from ``Re(z)`` while
   the reset is a differentiable additive correction ``+ i b`` injected into the
   input drive. Because the correction does not depend on the state ``z``, the
   recurrence stays the linear, associative
   ``z_t = a z_{t-1} + (x_t + i b)`` — sequentially and in the scan alike — so
   the O(log T) parallel path remains *exactly* equivalent to its sequential
   reference for any reset strength.

Both execution modes therefore remain numerically identical, exactly as for
``ResonateFire``/``PSU_LIF``: scanning :meth:`__call__` over ``x`` reproduces
:meth:`parallel`.

.. note::
   **Experimental.** Import via ``spyx.experimental`` once wired; the API may
   change without a deprecation cycle. Tests and studies import the concrete
   module path :mod:`spyx.experimental.rf_ssm`.

References
----------
* S5-RF: "Scaling Up Resonate-and-Fire Networks" (arXiv:2504.00719).
* PRF: "Parallel Resonate and Fire" / decoupled reset (arXiv:2410.03530).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..axn import superspike
from ..phasor import _inverse_softplus, _resonate_associative_op
from ..ssm import _hippo_legs_diagonal, _init_lru_eigenvalues

# Module-level singleton for the default surrogate activation (avoids B008).
_DEFAULT_ACTIVATION = superspike()


class RFSSM(nnx.Module):
    r"""Resonate-and-Fire spiking SSM (S5-RF) with a PRF-style decoupled reset.

    A complex-pole spiking neuron whose subthreshold membrane obeys the linear
    recurrence

    .. math::
        z_t = a \, z_{t-1} + x_t + i\,b , \qquad
        a = e^{\,\Delta_n (-\lambda_n + i\,\omega_n)} ,

    where the input current ``x_t`` drives the *real* axis, ``b`` is the
    :class:`PRF <https://arxiv.org/abs/2410.03530>` decoupled reset injected on
    the *imaginary* axis, and the per-unit pole ``(\lambda_n, \omega_n)`` plus
    log-step ``\Delta_n`` are initialised from the S5/HiPPO-LegS spectrum
    (:class:`S5-RF <https://arxiv.org/abs/2504.00719>`). Spikes are a pointwise
    surrogate threshold on the real part, :math:`s_t = \sigma(\Re(z_t) - \vartheta)`.

    The reset is *decoupled*: it is orthogonal to the ``Re(z)`` readout and,
    crucially, independent of the state ``z``. The recurrence therefore stays
    linear and associative, so — exactly like :class:`spyx.experimental.PSU_LIF`
    and :class:`spyx.phasor.ResonateFire` — it evaluates in :math:`O(\log T)`
    depth via :func:`jax.lax.associative_scan`, and scanning :meth:`__call__`
    over time reproduces :meth:`parallel` to numerical tolerance for any reset.

    Stability: ``|a_n| = exp(-\Delta_n \lambda_n)`` with ``\lambda_n =
    softplus(raw) \ge 0`` and ``\Delta_n = exp(log\_dt) > 0``, so every pole lies
    inside the unit disk and the oscillation never grows.

    Parameters entering the complex pole are stored as **real** ``float32``
    ``nnx.Param`` tensors (``raw_lambda``, ``omega``, ``log_dt``, ``reset``);
    the complex structure appears only in the forward pass, so a stock
    ``optax`` + ``jax.grad`` loop over a real loss trains them without the
    Wirtinger-conjugate surprise (mirroring :class:`spyx.phasor.PhasorLinear`).
    """

    def __init__(
        self,
        hidden_shape: tuple,
        *,
        pole_init: str = "hippo",
        reset_init: float = 0.0,
        threshold: float = 1.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        r_min: float = 0.4,
        r_max: float = 0.999,
        activation=None,
        rngs: nnx.Rngs,
    ):
        """
        :hidden_shape: Per-unit shape of the layer (e.g. ``(H,)``). The complex
            poles are initialised over ``prod(hidden_shape)`` units and reshaped.
        :pole_init: ``"hippo"`` (HiPPO-LegS ``-1/2 + i pi n``, the S5-RF default)
            or ``"lru"`` (radial LRU sampler, :func:`spyx.ssm._init_lru_eigenvalues`).
        :reset_init: initial PRF decoupled-reset strength ``b`` (per-unit,
            learnable). ``0.0`` recovers a HiPPO-initialised reset-free
            ``ResonateFire``; a non-zero value engages the imaginary-axis reset.
        :threshold: real firing threshold on ``Re(z)``. Defaults to 1.
        :dt_min, dt_max: log-uniform range for the per-unit log-step ``log_dt``
            (HiPPO init only), matching :class:`spyx.ssm.S5Diag`.
        :r_min, r_max: pole-magnitude range for the ``"lru"`` init.
        :activation: :class:`spyx.axn.Axon` surrogate spike; defaults to
            ``superspike``.
        """
        if pole_init not in ("hippo", "lru"):
            raise ValueError(f"pole_init must be 'hippo' or 'lru'; got {pole_init!r}.")
        self.hidden_shape = tuple(hidden_shape)
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        n_units = int(math.prod(self.hidden_shape)) if self.hidden_shape else 1

        if pole_init == "hippo":
            # HiPPO-LegS diagonal eigenvalues lambda_n = -1/2 + i pi n. The real
            # part (constant -1/2) becomes the decay; the imaginary part becomes
            # the angular frequency. A per-unit learnable log-step scales the
            # continuous eigenvalues, exactly as spyx.ssm.S5Diag does.
            legs = _hippo_legs_diagonal(n_units)
            decay0 = -legs.real  # = 0.5 everywhere
            omega0 = legs.imag  # = pi * n
            log_dt0 = jax.random.uniform(
                rngs.params(),
                (n_units,),
                minval=math.log(dt_min),
                maxval=math.log(dt_max),
            )
        else:  # "lru"
            # Radial LRU sampler: |lambda|^2 uniform in [r_min^2, r_max^2].
            # It returns (nu, theta) with lambda = exp(-exp(nu) + i theta); the
            # discrete magnitude exp(-exp(nu)) equals our |a| = exp(-dt*decay)
            # with dt == 1, so decay = exp(nu) and omega = theta.
            nu, theta = _init_lru_eigenvalues(rngs.params(), n_units, r_min, r_max)
            decay0 = jnp.exp(nu)
            omega0 = theta
            log_dt0 = jnp.zeros((n_units,))  # dt == 1

        shape = self.hidden_shape if self.hidden_shape else (1,)
        # Store decay through inverse-softplus so effective lambda = softplus(raw)
        # >= 0 for any raw the optimiser reaches, hence |a| <= 1 always.
        self.raw_lambda = nnx.Param(
            _inverse_softplus(decay0).astype(jnp.float32).reshape(shape)
        )
        self.omega = nnx.Param(omega0.astype(jnp.float32).reshape(shape))
        self.log_dt = nnx.Param(log_dt0.astype(jnp.float32).reshape(shape))
        self.reset = nnx.Param(jnp.full(shape, float(reset_init), dtype=jnp.float32))

    @property
    def decay(self) -> jax.Array:
        """Effective non-negative decay ``lambda = softplus(raw_lambda)``."""
        return jax.nn.softplus(self.raw_lambda[...])

    @property
    def step(self) -> jax.Array:
        """Effective positive per-unit log-step ``dt = exp(log_dt)``."""
        return jnp.exp(self.log_dt[...])

    @property
    def a(self) -> jax.Array:
        """Complex oscillator pole ``a = exp(dt (-lambda + i omega))``.

        ``|a| = exp(-dt * lambda) <= 1`` guarantees the pole is inside the unit
        disk (stable).
        """
        exponent = self.step * (-self.decay + 1j * self.omega[...])
        return jnp.exp(exponent).astype(jnp.complex64)

    def _drive(self, x: jax.Array) -> jax.Array:
        """Complex input drive: current on the real axis, reset on the imaginary.

        ``drive = x + i b`` with ``b = reset``. State-independent, so the
        recurrence ``z = a z + drive`` stays linear (scan-exact).
        """
        return x.astype(jnp.complex64) + 1j * self.reset[...].astype(jnp.complex64)

    def __call__(self, x, z):
        """One reset-free-scan-compatible timestep.

        :x: real input current from the previous layer, broadcastable to ``z``.
        :z: complex64 membrane state.

        Advances ``z = a z + (x + i b)`` (the PRF decoupled reset lives on the
        imaginary axis, so the recurrence stays linear) and emits a surrogate
        spike on ``Re(z)``, so scanning this method matches :meth:`parallel`.
        """
        z = self.a * z + self._drive(x)
        spikes = self.spike(jnp.real(z) - self.threshold)
        return spikes, z

    def parallel(self, x):
        r"""Score a whole time-major sequence with an associative scan.

        :x: real input with shape ``[Time, Batch, ...]``.
        :return: spikes with shape ``[Time, Batch, ...]``.

        Computes the complex membrane trace ``z_t = a z_{t-1} + (x_t + i b)``
        (with ``z_{-1} = 0``) via :func:`jax.lax.associative_scan` over the time
        axis in :math:`O(\log T)` depth, then applies the surrogate spike
        pointwise on ``Re(z)``. Uses the same complex pole and drive as
        :meth:`__call__`, so the two paths are numerically identical.
        """
        a = self.a
        drive = self._drive(x)
        # Broadcast pole and drive to the full [Time, Batch, ...] shape so the
        # recurrence coefficient a_t == a everywhere along the time axis.
        A = jnp.broadcast_to(a, drive.shape)
        B = jnp.broadcast_to(drive, drive.shape)
        _, z = jax.lax.associative_scan(_resonate_associative_op, (A, B), axis=0)
        return self.spike(jnp.real(z) - self.threshold)

    def initial_state(self, batch_size):
        """Return complex64 zeros of shape ``(batch_size,) + hidden_shape``."""
        return jnp.zeros((batch_size,) + self.hidden_shape, dtype=jnp.complex64)


# Alias: the neuron is equivalently the "resonate-and-fire SSM".
ResonateFireSSM = RFSSM


__all__ = ["RFSSM", "ResonateFireSSM"]
