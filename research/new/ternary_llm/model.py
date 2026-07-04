"""A tiny GPT-style transformer built entirely from ``flax.nnx`` primitives.

Every learned projection - the attention Q/K/V/output maps, the MLP, and the LM
head - is an :class:`flax.nnx.Linear`, so its matmul lowers to a ``dot_general``
primitive. That is exactly the op :mod:`spyx.quant`'s rule builders match, which
lets us quantize the whole transformer (int8 or BitNet-ternary) with the *same*
:func:`spyx.quant.quantize` call used for spiking nets - no model changes.

Design note - attention without ``einsum`` / ``dot_general``
------------------------------------------------------------
``qwix`` (the backend behind :mod:`spyx.quant`) intercepts ``dot_general`` and
``einsum`` at trace time. The attention *score* and *value* products are
activation-by-activation matmuls with no weight to quantize; routing them through
those primitives makes qwix try to resolve a module path it doesn't have and
raises ``Current module is not known``. We therefore compute the score/value
contractions with an explicit broadcast-multiply-and-sum (``a * b).sum(axis)``,
which uses only elementwise ops and a reduction. This keeps the attention math in
fp32 (as it should be - only the learned *weights* are quantized) and sidesteps
the interception entirely. For the tiny context lengths here the O(T^2 * d)
materialization is negligible.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters for :class:`TinyGPT`.

    Defaults are a small-but-real configuration; :func:`run` in ``run.py`` shrinks
    them further under ``SMOKE=1`` so the 3-way comparison runs on CPU in a minute.
    """

    vocab_size: int = 128
    block_size: int = 64
    n_layer: int = 3
    n_head: int = 4
    d_model: int = 128
    d_ff: int | None = None  # defaults to 4 * d_model

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})."
            )
        return self.d_model // self.n_head

    @property
    def ff_dim(self) -> int:
        return self.d_ff if self.d_ff is not None else 4 * self.d_model


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention with all projections as ``nnx.Linear``."""

    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        d = cfg.d_model
        self.q_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)

    def _split_heads(self, x: jax.Array) -> jax.Array:
        # (B, T, d) -> (B, H, T, head_dim)
        b, t, _ = x.shape
        x = x.reshape(b, t, self.n_head, self.head_dim)
        return jnp.transpose(x, (0, 2, 1, 3))

    def __call__(self, x: jax.Array) -> jax.Array:
        b, t, _ = x.shape
        q = self._split_heads(self.q_proj(x))  # (B, H, T, hd)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Scores (B, H, Tq, Tk) via broadcast multiply + sum over head_dim.
        scores = (q[:, :, :, None, :] * k[:, :, None, :, :]).sum(-1)
        scores = scores / jnp.sqrt(jnp.asarray(self.head_dim, x.dtype))

        # Causal mask: query i may attend to key j <= i.
        idx = jnp.arange(t)
        causal = idx[:, None] >= idx[None, :]  # (T, T)
        scores = jnp.where(causal, scores, jnp.asarray(-1e9, x.dtype))
        attn = jax.nn.softmax(scores, axis=-1)  # (B, H, Tq, Tk)

        # Weighted sum of values via broadcast multiply + sum over Tk.
        out = (attn[:, :, :, :, None] * v[:, :, None, :, :]).sum(3)  # (B,H,Tq,hd)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, t, -1)
        return self.out_proj(out)


class MLP(nnx.Module):
    """Position-wise feed-forward block: Linear -> GELU -> Linear."""

    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.fc = nnx.Linear(cfg.d_model, cfg.ff_dim, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(cfg.ff_dim, cfg.d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.proj(jax.nn.gelu(self.fc(x)))


class Block(nnx.Module):
    """Pre-norm transformer block: x + attn(ln(x)); x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.attn = CausalSelfAttention(cfg, rngs=rngs)
        self.ln2 = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.mlp = MLP(cfg, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nnx.Module):
    """A minimal decoder-only transformer language model.

    ``__call__`` maps integer token ids ``(B, T)`` to next-token logits
    ``(B, T, vocab_size)``. Token and positional lookups use :class:`nnx.Embed`
    (a gather, not a matmul), so they stay fp32 under every :mod:`spyx.quant`
    rule; all the ``dot_general`` work lives in the Linear layers that quant
    targets.
    """

    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.wte = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.wpe = nnx.Embed(cfg.block_size, cfg.d_model, rngs=rngs)
        self.blocks = nnx.List([Block(cfg, rngs=rngs) for _ in range(cfg.n_layer)])
        self.ln_f = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.lm_head = nnx.Linear(
            cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs
        )

    def __call__(self, idx: jax.Array) -> jax.Array:
        _, t = idx.shape
        pos = jnp.arange(t)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def linear_kernels(model: nnx.Module) -> dict[str, jax.Array]:
    """Collect every ``nnx.Linear`` kernel in ``model`` keyed by its NNX path.

    Used by ``run.py`` to inspect the trained weights and prove the quantized
    variants really are ternary / int8 (few distinct levels), not a silent no-op.
    """
    kernels: dict[str, jax.Array] = {}
    for path, mod in nnx.iter_graph(model):
        if isinstance(mod, nnx.Linear):
            name = "/".join(str(p) for p in path) or "lm_head"
            kernels[name] = mod.kernel[...]
    return kernels
