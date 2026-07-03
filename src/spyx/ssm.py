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
        self.D = nnx.Param(jax.random.normal(k_D, (d_model,))) if use_skip else None

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
            raise ValueError(f"LRU expects [T, B, d_model]; got shape {u.shape}.")
        lam, B, C = self._complex_matrices()
        u_c = u.astype(jnp.complex64)
        Bu = jnp.einsum("sn,tbn->tbs", B, u_c)
        x_seq = _diagonal_scan(lam, Bu)
        y = jnp.einsum("ms,tbs->tbm", C, x_seq).real
        if self.use_skip:
            y = y + self.D[...] * u  # ty: ignore[not-subscriptable]  # guarded by use_skip
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
        self.D = nnx.Param(jax.random.normal(k_D, (d_model,))) if use_skip else None

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
            raise ValueError(f"S5Diag expects [T, B, d_model]; got shape {u.shape}.")
        lam, B, C = self._complex_matrices()
        u_c = u.astype(jnp.complex64)
        Bu = jnp.einsum("sn,tbn->tbs", B, u_c)
        x_seq = _diagonal_scan(lam, Bu)
        y = jnp.einsum("ms,tbs->tbm", C, x_seq).real
        if self.use_skip:
            y = y + self.D[...] * u  # ty: ignore[not-subscriptable]  # guarded by use_skip
        return y


# ---------------------------------------------------------------------------
# Selective SSM (Mamba)
# ---------------------------------------------------------------------------


def _selective_binary_op(a, b):
    """Associative op for per-timestep diagonal recurrences.

    Elements: ``(A, x)`` where ``A`` is the per-step transition scalar (can
    vary across the state dim but not the batch) and ``x`` is the accumulator.
    Composes as ``(A_b · A_a, A_b · x_a + x_b)``, the same rule as the fixed-λ
    case but now ``A`` is a per-timestep tensor.
    """
    A_a, x_a = a
    A_b, x_b = b
    return A_b * A_a, A_b * x_a + x_b


def _selective_scan(A_bar: jax.Array, Bu_bar: jax.Array) -> jax.Array:
    """Run Mamba's selective recurrence via associative scan.

    :A_bar: shape ``(T, B, d_inner, d_state)`` — per-step state transition.
    :Bu_bar: shape ``(T, B, d_inner, d_state)`` — per-step input drive.
    :return: shape ``(T, B, d_inner, d_state)``.
    """
    _, x_seq = jax.lax.associative_scan(_selective_binary_op, (A_bar, Bu_bar), axis=0)
    return x_seq


def _selective_scan_reference(A_bar: jax.Array, Bu_bar: jax.Array) -> jax.Array:
    """Sequential reference for :func:`_selective_scan`."""

    def step(x, ab):
        A_t, Bu_t = ab
        x_next = A_t * x + Bu_t
        return x_next, x_next

    x0 = jnp.zeros(Bu_bar.shape[1:], dtype=Bu_bar.dtype)
    _, xs = jax.lax.scan(step, x0, (A_bar, Bu_bar))
    return xs


class Mamba(nnx.Module):
    """Selective state-space layer (Gu & Dao, 2023) — the SSM core of a Mamba block.

    Implements the input-dependent ``(Δ, B, C)`` recurrence with a learned
    diagonal ``A`` matrix, running the selective scan via
    :func:`jax.lax.associative_scan` (O(log T) parallel depth). This is the
    portable pure-JAX fallback for the ``selective_scan_cuda`` op in the
    reference PyTorch implementation; it has the same semantics but lower
    throughput on long sequences compared to the custom CUDA kernel.

    Note: ``Mamba`` is the SSM subroutine. For the full block with the in-proj,
    depthwise conv, SiLU gate and out-proj, use :class:`MambaBlock`.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        *,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        rngs: nnx.Rngs,
    ):
        if dt_rank is None:
            # The published Mamba recipe uses ceil(d_inner / 16).
            dt_rank = max(1, (d_inner + 15) // 16)

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        k_xproj, k_dtproj, k_A, k_D = jax.random.split(rngs.params(), 4)

        # A tiny projection that extracts (Δ_rank, B, C) from the already-SSM
        # input. Δ is a low-rank scalar-per-channel signal; B, C are state-sized.
        self.x_proj = nnx.Linear(
            d_inner,
            dt_rank + 2 * d_state,
            use_bias=False,
            rngs=nnx.Rngs(0),
        )
        # Re-init the x_proj kernel so we don't accidentally share RNG state
        # with other layers (x_proj needs its own stream).
        self.x_proj.kernel = nnx.Param(
            jax.random.normal(k_xproj, self.x_proj.kernel[...].shape)
            * (1.0 / jnp.sqrt(d_inner))
        )

        # dt_proj maps the Δ_rank projection back to d_inner, with a bias
        # that's initialised so softplus(bias) ~ uniform(dt_min, dt_max).
        self.dt_proj = nnx.Linear(dt_rank, d_inner, rngs=nnx.Rngs(1))
        self.dt_proj.kernel = nnx.Param(
            jax.random.normal(k_dtproj, self.dt_proj.kernel[...].shape)
            * (dt_rank**-0.5)
        )
        # Match the published Mamba recipe: sample dt log-uniformly in
        # [dt_min, dt_max], then set bias = inverse_softplus(dt) so that
        # softplus(bias) ~ dt. The inverse-softplus identity
        # ``inv_softplus(x) = x + log1p(-exp(-x))`` takes a *positive* dt,
        # not log(dt) — feeding the log-space sample in directly (as the
        # first draft did) produced dt values ~1e-6 at init instead of
        # the intended 1e-3..1e-1 range, which damps the selective SSM.
        log_dt = jax.random.uniform(
            k_dtproj, (d_inner,), minval=jnp.log(dt_min), maxval=jnp.log(dt_max)
        )
        dt = jnp.exp(log_dt)
        inv_dt = dt + jnp.log1p(-jnp.exp(-dt))
        self.dt_proj.bias = nnx.Param(inv_dt)

        # A is a real-valued diagonal: A = -exp(A_log).
        A_init = jnp.tile(
            jnp.arange(1, d_state + 1, dtype=jnp.float32)[None, :], (d_inner, 1)
        )
        self.A_log = nnx.Param(jnp.log(A_init))

        # Skip-style D.
        self.D = nnx.Param(
            jnp.ones((d_inner,)) + 0.1 * jax.random.normal(k_D, (d_inner,))
        )

    def __call__(self, u: jax.Array) -> jax.Array:
        """Run the selective SSM.

        :u: real array ``(T, B, d_inner)``.
        :return: same shape.
        """
        if u.ndim != 3 or u.shape[-1] != self.d_inner:
            raise ValueError(
                f"Mamba expects [T, B, d_inner={self.d_inner}]; got {u.shape}."
            )
        T, B, _ = u.shape

        # x_proj(u) -> (dt_rank, d_state, d_state) split along the last axis.
        x_proj = self.x_proj(u)
        dt_rank = self.dt_rank
        d_state = self.d_state
        dt_lowrank, B_mat, C_mat = jnp.split(
            x_proj, (dt_rank, dt_rank + d_state), axis=-1
        )
        # dt: (T, B, d_inner) via dt_proj + softplus.
        dt = jax.nn.softplus(self.dt_proj(dt_lowrank))

        # A: (d_inner, d_state).
        A = -jnp.exp(self.A_log[...])

        # Discretise: A_bar = exp(dt ⊗ A), B_bar = dt ⊗ B.
        # dt: (T, B, d_inner); A: (d_inner, d_state); -> A_bar: (T, B, d_inner, d_state)
        A_bar = jnp.exp(jnp.einsum("tbd,ds->tbds", dt, A))
        # B_bar: (T, B, d_inner, d_state) via dt[..., None] * B_mat[..., None, :]
        B_bar = dt[..., None] * B_mat[..., None, :]
        # Input drive: (B_bar u) has shape (T, B, d_inner, d_state)
        Bu_bar = B_bar * u[..., None]

        # Selective scan.
        x_state = _selective_scan(A_bar, Bu_bar)  # (T, B, d_inner, d_state)

        # y = C · x + D · u
        y = jnp.einsum("tbs,tbds->tbd", C_mat, x_state)
        y = y + self.D[...] * u
        return y


class MambaBlock(nnx.Module):
    """Full Mamba block: in-proj → depthwise conv → SSM → gate → out-proj.

    Residual connection is left to the caller (usually composed alongside an
    ``RMSNorm`` inside a stack). The depthwise convolution uses
    ``flax.nnx.Conv`` with ``feature_group_count = d_inner`` to mimic the
    reference Mamba ``conv1d`` with ``groups = d_inner``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        d_inner = d_model * expand
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv

        self.in_proj = nnx.Linear(d_model, 2 * d_inner, use_bias=False, rngs=rngs)
        # Depthwise (groups == d_inner) causal 1D convolution.
        self.conv = nnx.Conv(
            in_features=d_inner,
            out_features=d_inner,
            kernel_size=(d_conv,),
            feature_group_count=d_inner,
            padding=((d_conv - 1, 0),),
            rngs=rngs,
        )
        self.ssm = Mamba(d_inner, d_state=d_state, rngs=rngs)
        self.out_proj = nnx.Linear(d_inner, d_model, use_bias=False, rngs=rngs)

    def __call__(self, u: jax.Array) -> jax.Array:
        """u: (T, B, d_model) → (T, B, d_model)."""
        if u.ndim != 3 or u.shape[-1] != self.d_model:
            raise ValueError(
                f"MambaBlock expects [T, B, d_model={self.d_model}]; got {u.shape}."
            )
        T, B, _ = u.shape

        # In-projection: split to (x, z_gate).
        x_z = self.in_proj(u)  # (T, B, 2*d_inner)
        x, z = jnp.split(x_z, 2, axis=-1)

        # Depthwise conv over time: nnx.Conv expects (B, T, C).
        x_BTC = jnp.transpose(x, (1, 0, 2))
        x_conv = self.conv(x_BTC)
        x = jnp.transpose(x_conv, (1, 0, 2))

        # SiLU + selective SSM.
        x = jax.nn.silu(x)
        y = self.ssm(x)
        # Gate with SiLU(z), then out-project.
        y = y * jax.nn.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# H-Net skeleton (chunked hierarchical SSM)
# ---------------------------------------------------------------------------


class ChunkedSSM(nnx.Module):
    """Hierarchical SSM stack — the structural skeleton of an H-Net.

    Splits the input sequence into fixed chunks of ``chunk_size`` timesteps,
    processes each chunk with an inner SSM (``inner``), pools the chunk to a
    single vector, runs the sequence of chunk-vectors through an outer SSM
    (``outer``), and up-samples the outer signal back into the chunk slots
    via a learnable affine blend. This captures the H-Net idea — hierarchical
    composition of SSMs at different temporal resolutions — without the
    dynamic-chunking and byte-level specifics of the full Hwang et al. 2024
    recipe, which are separate research pieces.

    ``inner`` and ``outer`` can be any module whose ``__call__`` takes
    ``(T, B, d_model)`` and returns the same shape — for example
    :class:`LRU`, :class:`S5Diag`, or :class:`MambaBlock`.

    :chunk_size: number of timesteps per chunk. The input length must be a
        multiple of this.
    :pool: ``"mean"`` or ``"last"`` (last-timestep pooling is closer to the
        H-Net's "segment-end" summary).
    """

    def __init__(
        self,
        inner: nnx.Module,
        outer: nnx.Module,
        *,
        chunk_size: int,
        pool: str = "mean",
    ):
        if pool not in ("mean", "last"):
            raise ValueError(f"pool must be 'mean' or 'last'; got {pool!r}.")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}.")
        self.inner = inner
        self.outer = outer
        self.chunk_size = chunk_size
        self.pool = pool

    def __call__(self, u: jax.Array) -> jax.Array:
        """u: (T, B, d_model) → (T, B, d_model), where T is divisible by chunk_size."""
        if u.ndim != 3:
            raise ValueError(f"ChunkedSSM expects 3D input; got {u.shape}.")
        T, B, D = u.shape
        if T % self.chunk_size != 0:
            raise ValueError(
                f"Sequence length {T} is not divisible by chunk_size {self.chunk_size}."
            )

        # Run the inner SSM over the full sequence first — cheap and keeps
        # per-timestep resolution.
        h = self.inner(u)  # (T, B, D)

        # Pool chunks -> (n_chunks, B, D).
        n_chunks = T // self.chunk_size
        reshaped = h.reshape(n_chunks, self.chunk_size, B, D)
        if self.pool == "mean":
            summaries = reshaped.mean(axis=1)
        else:  # "last"
            summaries = reshaped[:, -1]

        # Outer SSM on the summaries.
        summaries_out = self.outer(summaries)  # (n_chunks, B, D)

        # Broadcast each chunk summary back across its timesteps and add.
        broadcast = jnp.repeat(summaries_out, self.chunk_size, axis=0)  # (T, B, D)
        return h + broadcast


__all__ = [
    "LRU",
    "S5Diag",
    "Mamba",
    "MambaBlock",
    "ChunkedSSM",
]
