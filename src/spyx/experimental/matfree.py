r"""Matmul-free linear primitives — ternary (BitNet) and shift-add (DeepShift).

.. note::
   **Experimental / sketch.** Unstable API. These are the *native* (train-from-
   scratch, QAT) counterpart to the post-training :func:`spyx.quant.bitnet_ternary_rules`
   path: layers whose forward pass replaces the expensive multiplies of a dense
   matmul with cheap **accumulations** (ternary) or **bit-shifts** (power-of-two),
   so you can *build* multiplication-light architectures rather than convert them.

The multiply-free idea
----------------------
A dense layer ``y = x @ W`` costs ``in*out`` multiplies. Two ways to remove them:

* **Ternary** (BitNet b1.58; *Scalable MatMul-free LM*, Zhu et al. 2024). Constrain
  ``W`` to ``{-1, 0, +1}`` times a per-tensor scale ``β``. Then
  ``y = β · (Σ_{W=+1} x − Σ_{W=-1} x)`` — pure signed **accumulation**, plus one
  scale multiply per output.
* **Shift-add** (DeepShift, Elhoushi et al. 2021; *ShiftAddLLM*, You et al. 2024).
  Constrain ``W`` to signed powers of two ``±2^p``. Then ``W·x = ± (x << p)`` — a
  **bit-shift** and a sign, no multiply, on fixed-point hardware.

Both are trained with a straight-through estimator (STE): the forward uses the
quantised weight, the backward flows to a full-precision shadow weight.

The spiking synthesis
---------------------
Spyx's real leverage here: a **binary spike** activation (``s ∈ {0,1}``, from any
:mod:`spyx.nn` neuron) times a **ternary weight** (``∈ {-1,0,+1}``) is a *fully
add-only* operation — no multiplies anywhere in the layer. Pair these layers with
spiking neurons (or feed :func:`spyx.experimental.compress.pack_spikes` outputs) to
get networks that are matmul-free in both operands. See the roadmap in the module
for the matmul-free-LM block (ternary channel-mixer + an SSM / ternary-GRU token
mixer) that this is the substrate for.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "ste",
    "ternary_weights",
    "power_of_two_weights",
    "activation_quant",
    "TernaryLinear",
    "ShiftAddLinear",
    "TernaryMLP",
    "RMSNorm",
    "MLGRU",
    "MatMulFreeBlock",
]


def ste(x: jax.Array, x_q: jax.Array) -> jax.Array:
    """Straight-through estimator: forward is ``x_q``, backward is identity in ``x``.

    ``ste(x, quantize(x))`` evaluates to the quantised value but passes gradients to
    the full-precision ``x`` unchanged — the standard trick for training through a
    non-differentiable quantiser.
    """
    return x + jax.lax.stop_gradient(x_q - x)


def ternary_weights(w: jax.Array, eps: float = 1e-5) -> tuple[jax.Array, jax.Array]:
    """BitNet b1.58 absmean ternarisation. Returns ``(w_ternary, scale)``.

    ``scale = mean(|w|)``; ``w_ternary = round(clip(w/scale, -1, 1)) ∈ {-1, 0, +1}``.
    The reconstruction is ``scale · w_ternary``.
    """
    scale = jnp.mean(jnp.abs(w)) + eps
    w_ternary = jnp.round(jnp.clip(w / scale, -1.0, 1.0))
    return w_ternary, scale


def power_of_two_weights(
    w: jax.Array, min_exp: int = -8, max_exp: int = 0, eps: float = 1e-12
) -> jax.Array:
    """Round each weight to the nearest signed power of two ``±2^p`` (DeepShift).

    ``p`` is clamped to ``[min_exp, max_exp]``; the result multiplies as a bit-shift
    on fixed-point hardware. Near-zero weights round to the smallest magnitude.
    """
    sign = jnp.sign(w)
    exp = jnp.round(jnp.log2(jnp.abs(w) + eps))
    exp = jnp.clip(exp, min_exp, max_exp)
    return sign * (2.0**exp)


def activation_quant(x: jax.Array, bits: int = 8, eps: float = 1e-5) -> jax.Array:
    """Per-token absmax quantisation of activations to ``bits`` (BitNet's a8).

    Returns the dequantised value (STE-friendly); pass through :func:`ste` to train.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = qmax / (jnp.max(jnp.abs(x), axis=-1, keepdims=True) + eps)
    return jnp.round(jnp.clip(x * scale, -qmax, qmax)) / scale


class TernaryLinear(nnx.Module):
    """Dense layer with ternary ``{-1,0,+1}`` weights — a BitNet ``BitLinear``.

    Forward: ``y = β · (x_q @ W_ternary) + b``. The ``x_q @ W_ternary`` product is
    accumulation-only. With ``activation_bits`` set, activations are absmax-quantised
    first (BitNet b1.58 + a8). Trained via STE through a full-precision shadow weight.

    :param activation_bits: if set, quantise inputs to this many bits (e.g. 8).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        use_bias: bool = False,
        activation_bits: int | None = None,
    ):
        self.w = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (in_features, out_features))
        )
        self.bias = nnx.Param(jnp.zeros((out_features,))) if use_bias else None
        self.activation_bits = activation_bits

    def __call__(self, x):
        if self.activation_bits is not None:
            x = ste(x, activation_quant(x, self.activation_bits))
        w = self.w[...]
        w_ternary, scale = ternary_weights(w)
        w_q = ste(w, w_ternary)  # forward ternary, backward full-precision
        y = (x @ w_q) * scale  # x @ w_ternary is signed accumulation
        if self.bias is not None:
            y = y + self.bias[...]
        return y


class ShiftAddLinear(nnx.Module):
    """Dense layer with signed-power-of-two weights — DeepShift / ShiftAdd.

    Forward: ``y = x @ W_po2 + b`` with ``W_po2 = ±2^p``; each product is a shift on
    fixed-point hardware. Trained via STE through a full-precision shadow weight.

    :param min_exp/max_exp: clamp range for the exponents ``p``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        use_bias: bool = False,
        min_exp: int = -8,
        max_exp: int = 0,
    ):
        self.w = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (in_features, out_features))
        )
        self.bias = nnx.Param(jnp.zeros((out_features,))) if use_bias else None
        self.min_exp = min_exp
        self.max_exp = max_exp

    def __call__(self, x):
        w = self.w[...]
        w_q = ste(w, power_of_two_weights(w, self.min_exp, self.max_exp))
        y = x @ w_q
        if self.bias is not None:
            y = y + self.bias[...]
        return y


class TernaryMLP(nnx.Module):
    """A matmul-free channel mixer: two :class:`TernaryLinear` with a nonlinearity.

    The multiply-free counterpart of a Transformer/SSM feed-forward block; drop it in
    wherever a dense MLP sits. Pair with a matmul-free *token* mixer (an SSM from
    :mod:`spyx.ssm`, or a ternary GRU — see the module roadmap) for a matmul-free LM.
    """

    def __init__(
        self,
        features: int,
        hidden: int,
        *,
        rngs: nnx.Rngs,
        activation_bits: int | None = 8,
    ):
        self.up = TernaryLinear(
            features, hidden, rngs=rngs, activation_bits=activation_bits
        )
        self.down = TernaryLinear(
            hidden, features, rngs=rngs, activation_bits=activation_bits
        )

    def __call__(self, x):
        return self.down(jax.nn.gelu(self.up(x)))


class RMSNorm(nnx.Module):
    """Root-mean-square layer norm — a per-feature rescale, no matmul.

    The only non-accumulation op in a matmul-free block: an element-wise
    normalisation (O(D) work), negligible next to a dense layer's O(D²).
    """

    def __init__(self, dim: int, *, rngs: nnx.Rngs, eps: float = 1e-6):
        self.scale = nnx.Param(jnp.ones((dim,)))
        self.eps = eps

    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale[...]


class MLGRU(nnx.Module):
    r"""MatMul-free Linear GRU token mixer (Zhu et al., 2024).

    The multiply-free replacement for attention: instead of an ``O(T²)`` ``QKᵀ``
    matmul, tokens are mixed by a **causal element-wise linear recurrence**

    .. math::
        h_t = f_t \odot h_{t-1} + (1 - f_t) \odot c_t,\qquad y_t = W_o (g_t \odot h_t)

    where the gate/candidate projections (``f``, ``c``, ``g``) and the output ``o``
    are :class:`TernaryLinear` (accumulation-only) and everything else is
    element-wise. The recurrence is a first-order linear scan — parallelisable with
    ``jax.lax.associative_scan`` the same way :class:`spyx.experimental.PSU_LIF` is;
    here it uses ``jax.lax.scan`` for clarity.

    Input/output are batch-major ``[B, T, D]``; the mixing is strictly causal.
    """

    def __init__(self, dim: int, hidden: int, *, rngs: nnx.Rngs, activation_bits=8):
        self.f = TernaryLinear(dim, hidden, rngs=rngs, activation_bits=activation_bits)
        self.c = TernaryLinear(dim, hidden, rngs=rngs, activation_bits=activation_bits)
        self.g = TernaryLinear(dim, hidden, rngs=rngs, activation_bits=activation_bits)
        self.o = TernaryLinear(hidden, dim, rngs=rngs, activation_bits=activation_bits)

    def __call__(self, x):
        f = jax.nn.sigmoid(self.f(x))  # forget gate      [B, T, H]
        c = jax.nn.silu(self.c(x))  # candidate state   [B, T, H]
        g = jax.nn.sigmoid(self.g(x))  # output gate       [B, T, H]
        i = (1.0 - f) * c  # input contribution

        f_t = jnp.moveaxis(f, 1, 0)  # time-major for the scan [T, B, H]
        i_t = jnp.moveaxis(i, 1, 0)

        def recur(h, fi):
            ft, it = fi
            h = ft * h + it  # element-wise linear recurrence
            return h, h

        h0 = jnp.zeros((x.shape[0], f.shape[-1]), x.dtype)
        _, h_t = jax.lax.scan(recur, h0, (f_t, i_t))
        h = jnp.moveaxis(h_t, 0, 1)  # back to [B, T, H]
        return self.o(g * h)


class MatMulFreeBlock(nnx.Module):
    """A matmul-free transformer-style block: pre-norm, MLGRU mixer, ternary MLP.

    ``x = x + MLGRU(RMSNorm(x));  x = x + TernaryMLP(RMSNorm(x))``. Every dense
    operation is ternary (accumulation-only); the token mixing is an element-wise
    recurrence. Stack these for a matmul-free language model — swap it into
    ``research/new/ternary_llm`` in place of a Transformer block and read the
    efficiency off ``spyx.bench``.

    :param mlp_ratio: channel-mixer hidden width as a multiple of ``dim``.
    """

    def __init__(
        self,
        dim: int,
        *,
        rngs: nnx.Rngs,
        hidden: int | None = None,
        mlp_ratio: int = 4,
        activation_bits: int | None = 8,
    ):
        self.norm1 = RMSNorm(dim, rngs=rngs)
        self.mixer = MLGRU(
            dim, hidden or dim, rngs=rngs, activation_bits=activation_bits
        )
        self.norm2 = RMSNorm(dim, rngs=rngs)
        self.mlp = TernaryMLP(
            dim, dim * mlp_ratio, rngs=rngs, activation_bits=activation_bits
        )

    def __call__(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
