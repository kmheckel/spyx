r"""Test-time-training fast-weight sequence layer — the hidden *state is a weight*.

A conventional recurrent layer carries a hidden *vector*; a **fast-weight** /
test-time-training (TTT) layer carries a small hidden *matrix* ``W_t`` and treats
each incoming token as a tiny supervised example, taking one online learning step
on ``W`` as the sequence streams. The token is projected by *slow* (ordinary,
backprop-trained) weights into a key ``k_t``, a value/target ``v_t`` and a query
``q_t``; the fast weights are updated toward mapping ``k_t -> v_t`` and then read
out with ``q_t``. This is the "fast-weight programmer" view of linear attention
(Schlag, Irie & Schmidhuber 2021) and the TTT-Linear layer of Sun et al. 2024,
where the hidden state is itself a linear model updated by self-supervised
gradient descent.

Two write rules are provided, and the distinction is the whole point of putting
this in a *parallel* spiking library:

* ``rule="hebb"`` — the purely additive outer-product write

  .. math::
      W_t = \lambda \, W_{t-1} + \eta \, v_t k_t^\top .

  The transition is a **scalar** (``lambda``), so this is a first-order *affine*
  recurrence in the matrix state — associative in exactly the sense
  :data:`spyx.nn._leaky_associative_op` exploits — and the whole sequence is an
  ``O(log T)`` :func:`jax.lax.associative_scan`, matrix-valued sibling of
  :class:`spyx.experimental.PSU_LIF` / :class:`spyx.experimental.SigmaDelta`.

* ``rule="delta"`` — the error-correcting *delta rule* (Schlag 2021, Sun 2024)

  .. math::
      W_t = \lambda \, W_{t-1}
            + \eta \, (v_t - W_{t-1} k_t)\, k_t^\top
          = W_{t-1}\,(\lambda I - \eta\, k_t k_t^\top) + \eta\, v_t k_t^\top .

  It writes the *residual* ``v_t - pred_t`` instead of ``v_t``, so the fast
  weights correct their current key->value mapping rather than just accumulate.
  It is still linear in ``W``, but the transition ``M_t = \lambda I -
  \eta k_t k_t^\top`` is now a **matrix**, not a scalar.

Honest parallelism boundary: matrix-affine maps are still associative
(``(M_i, B_i) ∘ (M_j, B_j) = (M_i M_j, B_i M_j + B_j)``), but each
``associative_scan`` combine then costs a ``key×key`` matrix–matrix product
``O(d_k^3)`` and stores ``T`` such matrices (``O(T d_k^2)`` memory) — so a *naive*
scan is not a win. The hardware-efficient parallel form of the delta rule is the
**chunked** WY / Householder algorithm of Yang et al. 2024 (DeltaNet), which is
out of scope here. Accordingly :meth:`TTTFastWeight.parallel` is implemented only
for ``rule="hebb"`` (scalar transition, genuinely cheap); for ``rule="delta"``
use the sequential :meth:`__call__` via :func:`spyx.nn.run`, and reach for chunked
DeltaNet if you need the parallel speed-up.

Both execution modes for ``rule="hebb"`` are numerically identical: scanning
:meth:`__call__` over time reproduces :meth:`parallel` exactly (same slow
projections, same ``eta``/``lambda``, read-out taken *after* the write in both).

References:

* Sun, Li, Dalal, Xu, Xu, Xie, et al., *Learning to (Learn at Test Time): RNNs
  with Expressive Hidden States*, 2024
  (`arXiv:2407.04620 <https://arxiv.org/abs/2407.04620>`_).
* Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight
  Programmers*, ICML 2021
  (`arXiv:2102.11174 <https://arxiv.org/abs/2102.11174>`_).
* Yang, Wang, Zhang, Shen & Kim, *Parallelizing Linear Transformers with the
  Delta Rule over Sequence Length*, NeurIPS 2024
  (`arXiv:2406.06484 <https://arxiv.org/abs/2406.06484>`_) — the chunked parallel
  form of ``rule="delta"``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ..nn import _leaky_associative_op

__all__ = ["TTTFastWeight"]


class TTTFastWeight(nnx.Module):
    r"""Fast-weight / test-time-training sequence layer (matrix hidden state).

    The hidden state carried through the sequence is a fast-weight matrix
    ``W_t`` of shape ``(batch, val_dim, key_dim)`` — *not* a learnable parameter,
    but a state updated online from the stream. The learnable (*slow*) parameters
    are the input projections ``W_k``, ``W_v``, ``W_q`` and the read-out
    projection ``W_o``; ``eta`` (the online learning rate) is a learnable scalar.
    Gradients from a downstream loss flow into all of these through the unrolled
    (or scanned) recurrence.

    Per step ``(x_t, W) -> (out_t, W_new)``:

    #. project ``k_t = W_k x_t``, ``v_t = W_v x_t``, ``q_t = W_q x_t``;
    #. write the fast weights with the selected ``rule`` (see below);
    #. read out ``y_t = W_new q_t`` and project ``out_t = W_o y_t``.

    Following the ``(x, state) -> (out, new_state)`` contract, it drops straight
    into :class:`spyx.nn.Sequential` and :func:`spyx.nn.run`.

    :in_features: input feature dimension.
    :out_features: output feature dimension.
    :key_dim: key/query dimension (fast-weight columns). Defaults to
        ``in_features``.
    :val_dim: value dimension (fast-weight rows). Defaults to ``out_features``.
    :eta: online learning rate of the fast-weight write. Learnable scalar
        (initialised to this value); used identically in both execution modes.
    :decay: fast-weight leak ``lambda`` in ``[0, 1]``. Fixed (not learned) to
        keep the recurrence's scalar transition stable; ``1.0`` is a pure
        (undecayed) associative memory.
    :rule: ``"hebb"`` for the additive outer-product write (scalar transition,
        ``.parallel`` associative-scan available) or ``"delta"`` for the
        error-correcting delta rule (matrix transition; sequential only here).

    Honest caveat: with ``decay=1.0`` and no key normalisation the additive rule
    is an unbounded associative memory — repeated keys accumulate — so long
    sequences may need ``decay < 1`` or unit-norm keys. The delta rule is
    self-stabilising (it subtracts its own prediction) but is not cheaply
    parallel here; that is the deliberate trade, mirroring reset-free vs. reset
    in the parallel LIF neurons.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key_dim: int | None = None,
        val_dim: int | None = None,
        eta: float = 0.5,
        decay: float = 1.0,
        rule: str = "delta",
        *,
        rngs: nnx.Rngs,
    ):
        if rule not in ("hebb", "delta"):
            raise ValueError(f"rule must be 'hebb' or 'delta', got {rule!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.key_dim = key_dim if key_dim is not None else in_features
        self.val_dim = val_dim if val_dim is not None else out_features
        self.decay = float(decay)
        self.rule = rule

        self.eta = nnx.Param(jnp.full((), float(eta)))
        self.W_k = nnx.Linear(in_features, self.key_dim, rngs=rngs)
        self.W_v = nnx.Linear(in_features, self.val_dim, rngs=rngs)
        self.W_q = nnx.Linear(in_features, self.key_dim, rngs=rngs)
        self.W_o = nnx.Linear(self.val_dim, out_features, rngs=rngs)

    def __call__(self, x, W):
        """One online step ``(x_t, W) -> (out_t, W_new)``.

        Projects ``x_t`` to ``(k, v, q)``, writes the fast weights with the
        selected ``rule``, then reads out with the query. Scanning this over time
        via :func:`spyx.nn.run` reproduces :meth:`parallel` exactly for
        ``rule="hebb"``.
        """
        k = self.W_k(x)  # (B, key_dim)
        v = self.W_v(x)  # (B, val_dim)
        q = self.W_q(x)  # (B, key_dim)
        eta = self.eta[...]

        if self.rule == "delta":
            pred = jnp.einsum("bvk,bk->bv", W, k)  # W_{t-1} k_t
            write = eta * jnp.einsum("bv,bk->bvk", v - pred, k)
        else:  # "hebb"
            write = eta * jnp.einsum("bv,bk->bvk", v, k)
        W_new = self.decay * W + write

        y = jnp.einsum("bvk,bk->bv", W_new, q)  # read with the query
        return self.W_o(y), W_new

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size, self.val_dim, self.key_dim))

    def parallel(self, x):
        r"""Score a whole time-major ``[T, B, in]`` sequence with an associative scan.

        Only defined for ``rule="hebb"``: the additive write has a **scalar**
        transition ``lambda``, so the fast-weight trace
        ``W_t = lambda W_{t-1} + eta v_t k_t^T`` (with ``W_{-1} = 0``) is a
        matrix-valued affine recurrence and folds with
        :func:`jax.lax.associative_scan` via
        :data:`spyx.nn._leaky_associative_op` in ``O(log T)`` depth — the same
        parallel primitive as :class:`spyx.experimental.PSU_LIF`, with a matrix
        increment in place of a scalar. Read-out ``y_t = W_t q_t`` is then a
        pointwise batched mat-vec.

        For ``rule="delta"`` the transition is a ``key×key`` matrix, so a naive
        scan is ``O(T d_k^3)`` work / ``O(T d_k^2)`` memory rather than a win;
        this raises, pointing you to sequential :func:`spyx.nn.run` (or the
        chunked DeltaNet algorithm, Yang et al. 2024).
        """
        if self.rule != "hebb":
            raise NotImplementedError(
                "parallel() is only available for rule='hebb' (scalar transition). "
                "The delta rule has a matrix transition; scan it sequentially with "
                "spyx.nn.run, or use the chunked DeltaNet form (arXiv:2406.06484)."
            )
        k = self.W_k(x)  # (T, B, key_dim)
        v = self.W_v(x)  # (T, B, val_dim)
        q = self.W_q(x)  # (T, B, key_dim)

        # Per-step matrix increment eta * outer(v_t, k_t): (T, B, val, key).
        b = self.eta[...] * jnp.einsum("tbv,tbk->tbvk", v, k)
        # Scalar transition lambda broadcast to the (T, B, 1, 1) leading shape so
        # the affine-map coefficient A_t == lambda for every element.
        A = jnp.broadcast_to(jnp.asarray(self.decay), b.shape[:2] + (1, 1))
        _, W = jax.lax.associative_scan(_leaky_associative_op, (A, b), axis=0)
        return self.W_o(jnp.einsum("tbvk,tbk->tbv", W, q))
