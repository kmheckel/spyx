"""State-space models (SSMs) for Spyx.

A first-pass implementation of diagonal complex-valued SSMs targeting the
Flax NNX module system. This module focuses on the linear recurrence:

    x_k = λ ⊙ x_{k-1} + B u_k
    y_k = Re(C x_k) + D u_k

where ``λ`` is a diagonal complex decay. The recurrence is run with
:func:`jax.lax.associative_scan` for O(log T) parallel depth on accelerators.

Two layer classes are provided:

* :class:`LRU` — Linear Recurrent Unit (Orvieto et al., 2023,
  arXiv 2303.06349). Stability-preserving radial/angular parameterisation;
  no HiPPO required.
* :class:`S5Diag` — a diagonal S4D / S5-style layer that initialises from the
  HiPPO-LegS eigenvalues so the layer can represent long-range dependencies
  out of the box.

Both compose with :class:`spyx.nn.Sequential` and can be quantized via
:mod:`spyx.quant` (see the BitNet helper for ternary SSM weights). A tiny
worked example lives in ``scripts/ssm_demo.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Associative-scan primitive
# ---------------------------------------------------------------------------


def _lru_binary_op(a: tuple[jax.Array, jax.Array], b: tuple[jax.Array, jax.Array]):
    """Combine two linear-recurrence steps.

    Each element is ``(λ, x)`` representing the transformation
    ``new_state = λ * prev_state + x``. Composing left-to-right yields
    ``(λ_b · λ_a, λ_b · x_a + x_b)``, which is the standard associative rule
    for diagonal linear recurrences and matches the LRU / S5 papers.
    """
    lam_a, x_a = a
    lam_b, x_b = b
    return lam_b * lam_a, lam_b * x_a + x_b


def _diagonal_scan(lam: jax.Array, Bu: jax.Array) -> jax.Array:
    """Run ``x_k = λ · x_{k-1} + Bu_k`` in parallel along axis 0.

    :lam: complex64 state vector, shape ``(d_state,)``. Broadcast across the
        leading time and batch dimensions of ``Bu``.
    :Bu: complex64, shape ``(T, B, d_state)``.
    :return: complex64, shape ``(T, B, d_state)``, the state trajectory.
    """
    lam_bcast = jnp.broadcast_to(lam[None, None, :], Bu.shape)
    _, x_seq = jax.lax.associative_scan(_lru_binary_op, (lam_bcast, Bu), axis=0)
    return x_seq


def _diagonal_scan_reference(lam: jax.Array, Bu: jax.Array) -> jax.Array:
    """Sequential reference implementation for testing.

    Same recurrence as :func:`_diagonal_scan`, but via :func:`jax.lax.scan`.
    Both must agree within numerical tolerance.
    """

    def step(x, Bu_t):
        x_next = lam * x + Bu_t
        return x_next, x_next

    x0 = jnp.zeros(Bu.shape[1:], dtype=Bu.dtype)
    _, xs = jax.lax.scan(step, x0, Bu)
    return xs


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def _init_lru_eigenvalues(
    key: jax.Array, d_state: int, r_min: float, r_max: float
) -> tuple[jax.Array, jax.Array]:
    """Sample LRU eigenvalue parameters so |λ|² is uniform in ``[r_min², r_max²]``.

    Returns ``(ν, θ)`` such that ``λ = exp(-exp(ν) + i θ)``.
    """
    k_u, k_theta = jax.random.split(key)
    u1 = jax.random.uniform(k_u, (d_state,))
    r_sq = u1 * (r_max**2 - r_min**2) + r_min**2
    nu = jnp.log(-0.5 * jnp.log(r_sq + 1e-8))
    theta = jax.random.uniform(k_theta, (d_state,)) * 2.0 * jnp.pi
    return nu, theta


class LRU(nnx.Module):
    """Linear Recurrent Unit (Orvieto et al., 2023).

    ``d_model`` is the input/output channel count; ``d_state`` is the diagonal
    hidden-state size. The recurrence runs in fp32 arithmetic with complex64
    state; gradients flow through both the radial (``ν``) and angular (``θ``)
    parameterisations, so the stability constraint ``|λ| < 1`` is enforced by
    construction rather than clipping.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        *,
        r_min: float = 0.0,
        r_max: float = 1.0,
        use_skip: bool = True,
        rngs: nnx.Rngs,
    ):
        k_lam, k_B, k_C, k_D = jax.random.split(rngs.params(), 4)
        nu, theta = _init_lru_eigenvalues(k_lam, d_state, r_min, r_max)
        self.nu = nnx.Param(nu)
        self.theta = nnx.Param(theta)

        # "Gamma" normalisation on B keeps the state variance bounded at init.
        lam_mag = jnp.exp(-jnp.exp(nu))
        gamma = jnp.sqrt(jnp.clip(1.0 - lam_mag**2, 0.0, None))

        scale_B = 1.0 / jnp.sqrt(2.0 * d_model)
        kB_r, kB_i = jax.random.split(k_B)
        B_re = jax.random.normal(kB_r, (d_state, d_model)) * scale_B
        B_im = jax.random.normal(kB_i, (d_state, d_model)) * scale_B
        self.B_re = nnx.Param(B_re * gamma[:, None])
        self.B_im = nnx.Param(B_im * gamma[:, None])

        scale_C = 1.0 / jnp.sqrt(2.0 * d_state)
        kC_r, kC_i = jax.random.split(k_C)
        self.C_re = nnx.Param(jax.random.normal(kC_r, (d_model, d_state)) * scale_C)
        self.C_im = nnx.Param(jax.random.normal(kC_i, (d_model, d_state)) * scale_C)

        self.use_skip = use_skip
        self.D = (
            nnx.Param(jax.random.normal(k_D, (d_model,))) if use_skip else None
        )

        self.d_model = d_model
        self.d_state = d_state

    def _complex_matrices(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Assemble the complex (λ, B, C) tensors from the real parameters."""
        lam_mag = jnp.exp(-jnp.exp(self.nu[...]))
        lam = lam_mag * jnp.exp(1j * self.theta[...])
        B = (self.B_re[...] + 1j * self.B_im[...]).astype(jnp.complex64)
        C = (self.C_re[...] + 1j * self.C_im[...]).astype(jnp.complex64)
        return lam.astype(jnp.complex64), B, C

    def __call__(self, u: jax.Array) -> jax.Array:
        """Apply the SSM to a time-major input.

        :u: real array of shape ``(T, B, d_model)``.
        :return: real array of the same shape.
        """
        if u.ndim != 3:
            raise ValueError(
                f"LRU expects [T, B, d_model]; got shape {u.shape}."
            )
        lam, B, C = self._complex_matrices()
        u_c = u.astype(jnp.complex64)
        Bu = jnp.einsum("sn,tbn->tbs", B, u_c)
        x_seq = _diagonal_scan(lam, Bu)
        y = jnp.einsum("ms,tbs->tbm", C, x_seq).real
        if self.use_skip:
            y = y + self.D[...] * u
        return y


def _hippo_legs_diagonal(d_state: int) -> jax.Array:
    """Diagonal approximation of the HiPPO-LegS transition matrix.

    Returns complex eigenvalues ``λ_n = -½ + i·π·n`` for n = 0, 1, …, d_state-1,
    which is the canonical S4D / S5 diagonal init. In discrete time with step
    ``dt`` the state decays as ``exp(λ·dt)`` — stable for any ``dt > 0``.
    """
    n = jnp.arange(d_state, dtype=jnp.float32)
    return (-0.5 + 1j * jnp.pi * n).astype(jnp.complex64)


class S5Diag(nnx.Module):
    """Diagonal S4D / S5-style layer with HiPPO-LegS initialisation.

    Mechanically the same as :class:`LRU` but with (a) a continuous-time
    eigenvalue prior (HiPPO-LegS) and (b) a learnable log-step ``log_dt`` that
    controls the effective decay. This is the flavour that performs best on
    long-range tasks in the S4/S5 papers.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        *,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        use_skip: bool = True,
        rngs: nnx.Rngs,
    ):
        k_dt, k_B, k_C, k_D = jax.random.split(rngs.params(), 4)

        legs = _hippo_legs_diagonal(d_state)
        self.A_re = nnx.Param(legs.real)
        self.A_im = nnx.Param(legs.imag)

        log_dt = jax.random.uniform(
            k_dt, (d_state,), minval=jnp.log(dt_min), maxval=jnp.log(dt_max)
        )
        self.log_dt = nnx.Param(log_dt)

        scale_B = 1.0 / jnp.sqrt(2.0 * d_model)
        kB_r, kB_i = jax.random.split(k_B)
        self.B_re = nnx.Param(jax.random.normal(kB_r, (d_state, d_model)) * scale_B)
        self.B_im = nnx.Param(jax.random.normal(kB_i, (d_state, d_model)) * scale_B)

        scale_C = 1.0 / jnp.sqrt(2.0 * d_state)
        kC_r, kC_i = jax.random.split(k_C)
        self.C_re = nnx.Param(jax.random.normal(kC_r, (d_model, d_state)) * scale_C)
        self.C_im = nnx.Param(jax.random.normal(kC_i, (d_model, d_state)) * scale_C)

        self.use_skip = use_skip
        self.D = (
            nnx.Param(jax.random.normal(k_D, (d_model,))) if use_skip else None
        )

        self.d_model = d_model
        self.d_state = d_state

    def _complex_matrices(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Discretise (A, B) via zero-order hold; assemble complex C."""
        A_c = (self.A_re[...] + 1j * self.A_im[...]).astype(jnp.complex64)
        dt = jnp.exp(self.log_dt[...]).astype(A_c.dtype)
        lam = jnp.exp(A_c * dt)
        # ZOH discretisation: B_d = A^{-1} (e^{A dt} - I) B ≈ dt * B when A is small.
        B_c = (self.B_re[...] + 1j * self.B_im[...]).astype(jnp.complex64)
        C_c = (self.C_re[...] + 1j * self.C_im[...]).astype(jnp.complex64)
        B_discrete = ((lam - 1.0) / A_c)[:, None] * B_c
        return lam, B_discrete, C_c

    def __call__(self, u: jax.Array) -> jax.Array:
        if u.ndim != 3:
            raise ValueError(
                f"S5Diag expects [T, B, d_model]; got shape {u.shape}."
            )
        lam, B, C = self._complex_matrices()
        u_c = u.astype(jnp.complex64)
        Bu = jnp.einsum("sn,tbn->tbs", B, u_c)
        x_seq = _diagonal_scan(lam, Bu)
        y = jnp.einsum("ms,tbs->tbm", C, x_seq).real
        if self.use_skip:
            y = y + self.D[...] * u
        return y


__all__ = [
    "LRU",
    "S5Diag",
]
