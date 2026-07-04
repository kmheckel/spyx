"""Raven Routing-Slot-Memory (RSM) block for Spyx.

A Flax NNX implementation of the *Routing Slot Memory* recurrence introduced by
Raven (Afzal, Bick, Xing, Cevher, Gu, 2026; "High-recall sequence modeling with
sparse memory routing"). Compressed-state recurrent models (a single SSM state
with uniform decay) struggle with *exact recall*: every new token perturbs the
whole state, so previously written associations interfere with each other.

Raven's fix is to partition the memory into ``M`` independent **slots** and use a
learned **sparse router** ``r_t`` to write only the selected slots, leaving the
rest untouched (shielded from interference). Writing slot ``m`` at step ``t``:

.. math::
    S_t = (1 - r_t) \\odot S_{t-1} + r_t \\odot ( D_t S_{t-1} A_t + U_t )

* ``S_t``: slot memory, shape ``(B, M, d_slot)``.
* ``r_t \\in [0, 1]^M``: the per-slot router (ideally sparse). Unselected slots
  (``r_t[m] ≈ 0``) pass through unchanged; selected slots decay and are written.
* ``U_t``: the write (a projection of the current input).

The router is "a Mixture-of-Experts for memory". Two reductions are worth
remembering (and are exercised by the tests):

* a **dense** router (``r_t`` all-ones) recovers a standard gated diagonal SSM,
* a one-hot *cyclic* router recovers sliding-window attention.

Faithful-but-tractable simplification (documented, see
:class:`RavenRSM`): the per-slot transition is made **diagonal** — the full
matrix sandwich ``D_t S_{t-1} A_t`` is replaced by a per-slot (per-dim) decay
``a ⊙ S_{t-1}``, so each slot is a gated diagonal recurrence. The full
matrix-sandwich form is deferred. Likewise the recurrence is run with a plain
:func:`jax.lax.scan` reference (honest baseline); because the per-step transition
is *input-dependent* through the router gate ``(1 - r_t)``, the recurrence is a
per-timestep diagonal linear recurrence and an associative / chunked
``associative_scan`` form is in principle possible (the Raven authors defer it to
a "Part 2"), but is not implemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ..axn import superspike

# Module-level singleton for the default surrogate spike (avoids B008 and
# matches the default used by spyx.nn.PSU_LIF / spyx.phasor.ResonateFire).
_DEFAULT_SPIKE = superspike()

__all__ = ["RavenRSM", "SlotRouter", "SpikingSlotMemory", "make_recall_batch"]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def _straight_through_topk(r: jax.Array, k: int) -> jax.Array:
    """Hard top-``k`` sparsification of ``r`` along the last axis (straight-through).

    The forward value keeps the (soft) gate on the ``k`` largest entries and sets
    every other entry to *exactly* ``0`` — so unselected slots are provably
    shielded. The backward pass sees the dense soft gate, so gradients still flow
    to every router weight (a standard straight-through estimator).

    :r: soft gates in ``[0, 1]``, shape ``(..., M)``.
    :k: number of slots to keep active per row. ``k >= M`` is a no-op (dense).
    """
    m = r.shape[-1]
    if k >= m:
        return r
    # Select the k largest entries by *index* (not by a value threshold) so that
    # boundary ties keep exactly k slots -- top_k breaks ties deterministically,
    # whereas a ``r >= kth`` threshold would keep every tied entry (> k slots).
    _, idx = jax.lax.top_k(r, k)  # (..., k) distinct indices
    mask = jnp.sum(jax.nn.one_hot(idx, m, dtype=r.dtype), axis=-2)  # (..., M)
    r_hard = r * mask
    # Forward == r_hard; gradient == d r  (STE).
    return r + jax.lax.stop_gradient(r_hard - r)


class SlotRouter(nnx.Module):
    """Learned per-slot write gate ``r_t = sigmoid(W_r u_t)``.

    A small, reusable submodule (the spiking Raven variant reuses it). Maps an
    input of shape ``(..., d_model)`` to per-slot gates of shape ``(..., M)`` in
    ``[0, 1]``. With ``hard_top_k`` set, the gate is additionally sparsified to
    the ``k`` most-active slots per row via a straight-through top-``k`` (forward
    is sparse, gradients stay dense); the default (``None``) is a soft gate.

    Design choice: a per-input ``sigmoid`` (independent per-slot Bernoulli
    logits) is used rather than a ``softmax`` so that *several* slots can be
    written at once (a multi-write MoE-for-memory), and so the dense all-ones
    reduction is reachable in the limit of large positive logits.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int,
        *,
        hard_top_k: int | None = None,
        rngs: nnx.Rngs,
    ):
        if hard_top_k is not None and hard_top_k < 1:
            raise ValueError(f"hard_top_k must be >= 1 or None; got {hard_top_k}.")
        self.proj = nnx.Linear(d_model, n_slots, rngs=rngs)
        self.n_slots = n_slots
        self.hard_top_k = hard_top_k

    def __call__(self, u: jax.Array) -> jax.Array:
        """u: ``(..., d_model)`` -> gates ``(..., M)`` in ``[0, 1]``."""
        r = jax.nn.sigmoid(self.proj(u))
        if self.hard_top_k is not None:
            r = _straight_through_topk(r, self.hard_top_k)
        return r


# ---------------------------------------------------------------------------
# Raven RSM block
# ---------------------------------------------------------------------------


class RavenRSM(nnx.Module):
    r"""Routing-Slot-Memory recurrent block (diagonal simplification).

    Sequence-in / sequence-out, matching the :mod:`spyx.ssm` interface:
    ``__call__(u: (T, B, d_model)) -> (T, B, d_model)``.

    Per step ``t`` the block computes, from ``u_t``:

    * a sparse write router ``r_t = SlotRouter(u_t) \in [0, 1]^{(B, M)}``,
    * the write ``U_t = reshape(W_u u_t) \in (B, M, d_slot)``,

    and updates the slot memory with the diagonal RSM recurrence

    .. math::
        S_t = (1 - r_t) \odot S_{t-1} + r_t \odot (a \odot S_{t-1} + U_t)

    where ``a = sigmoid(raw_decay) \in (0, 1)^{(M, d_slot)}`` is a **static,
    learnable per-slot / per-dim decay** (kept in ``(0, 1)`` for stability; an
    input-dependent / selective decay is a straightforward extension but is not
    used here so the dense reduction stays a clean gated diagonal SSM). The
    recurrence is evaluated with :func:`jax.lax.scan` over time.

    **Readout** (``y_t``): a query-gated read over slots. A learned query
    ``q_t = softmax(W_q u_t) \in (B, M)`` mixes the slots into a single read
    vector ``read_t = \sum_m q_t[m] S_t[m] \in (B, d_slot)``, which a linear map
    projects back to ``(B, d_model)``. This mirrors the routing idea on the read
    side: the query key selects which slot(s) to retrieve.

    Simplifications (deferred, per the module docstring): (1) the full
    matrix-sandwich transition ``D_t S_{t-1} A_t`` is replaced by the diagonal
    decay ``a``; (2) only a sequential ``lax.scan`` is provided — a chunked /
    associative-scan form is possible but deferred.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = 8,
        d_slot: int | None = None,
        *,
        hard_top_k: int | None = None,
        decay_init: float = 0.9,
        rngs: nnx.Rngs,
    ):
        if d_slot is None:
            d_slot = d_model
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1; got {n_slots}.")
        if d_slot < 1:
            raise ValueError(f"d_slot must be >= 1; got {d_slot}.")
        if not 0.0 < decay_init < 1.0:
            raise ValueError(f"decay_init must be in (0, 1); got {decay_init}.")

        self.d_model = d_model
        self.n_slots = n_slots
        self.d_slot = d_slot

        self.router = SlotRouter(d_model, n_slots, hard_top_k=hard_top_k, rngs=rngs)
        # Write projection: u_t -> (M * d_slot), reshaped to (M, d_slot).
        self.write = nnx.Linear(d_model, n_slots * d_slot, rngs=rngs)
        # Read side: query over slots + projection back to d_model.
        self.readout_query = nnx.Linear(d_model, n_slots, rngs=rngs)
        self.out_proj = nnx.Linear(d_slot, d_model, rngs=rngs)

        # Static learnable per-slot / per-dim decay, stored as a raw logit so
        # that a = sigmoid(raw_decay) stays in (0, 1). Init near ``decay_init``
        # (slow decay -> long memory) with a little jitter.
        logit = float(jnp.log(decay_init / (1.0 - decay_init)))
        noise = 0.01 * jax.random.normal(rngs.params(), (n_slots, d_slot))
        self.raw_decay = nnx.Param(jnp.full((n_slots, d_slot), logit) + noise)

    @property
    def decay(self) -> jax.Array:
        """Effective per-slot / per-dim decay ``a = sigmoid(raw_decay)`` in ``(0, 1)``."""
        return jax.nn.sigmoid(self.raw_decay[...])

    def initial_state(self, batch_size: int) -> jax.Array:
        """Return zero slot memory of shape ``(batch_size, M, d_slot)``."""
        return jnp.zeros((batch_size, self.n_slots, self.d_slot), dtype=jnp.float32)

    def _route(self, u_t: jax.Array) -> jax.Array:
        """Expose the router for reuse: ``u_t (..., d_model) -> r (..., M)``."""
        return self.router(u_t)

    def step(self, state: jax.Array, u_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One reset-free RSM timestep.

        :state: slot memory ``S_{t-1}``, shape ``(B, M, d_slot)``.
        :u_t: input ``(B, d_model)``.
        :return: ``(S_t, y_t)`` with ``y_t`` of shape ``(B, d_model)``.
        """
        r_t = self.router(u_t)  # (B, M)
        U_t = self.write(u_t).reshape(u_t.shape[0], self.n_slots, self.d_slot)
        a = self.decay[None]  # (1, M, d_slot)
        gated = a * state + U_t
        r_exp = r_t[..., None]  # (B, M, 1)
        s_new = (1.0 - r_exp) * state + r_exp * gated
        attn = jax.nn.softmax(self.readout_query(u_t), axis=-1)  # (B, M)
        read = jnp.einsum("bm,bmd->bd", attn, s_new)  # (B, d_slot)
        y_t = self.out_proj(read)
        return s_new, y_t

    def _run(self, u: jax.Array, r: jax.Array) -> jax.Array:
        """Core recurrence with a *precomputed* router ``r`` of shape ``(T, B, M)``.

        Factored out so tests (and the dense-router reduction) can force ``r``.
        """
        T, B, _ = u.shape
        U = self.write(u).reshape(T, B, self.n_slots, self.d_slot)
        attn = jax.nn.softmax(self.readout_query(u), axis=-1)  # (T, B, M)
        a = self.decay[None]  # (1, M, d_slot)

        def scan_step(state, inp):
            r_t, U_t, attn_t = inp
            r_exp = r_t[..., None]
            gated = a * state + U_t
            s_new = (1.0 - r_exp) * state + r_exp * gated
            read = jnp.einsum("bm,bmd->bd", attn_t, s_new)
            return s_new, read

        s0 = self.initial_state(B)
        _, read_seq = jax.lax.scan(scan_step, s0, (r, U, attn))  # (T, B, d_slot)
        return self.out_proj(read_seq)

    def __call__(self, u: jax.Array) -> jax.Array:
        """Apply the RSM block to a time-major input.

        :u: real array of shape ``(T, B, d_model)``.
        :return: real array of shape ``(T, B, d_model)``.
        """
        if u.ndim != 3 or u.shape[-1] != self.d_model:
            raise ValueError(
                f"RavenRSM expects [T, B, d_model={self.d_model}]; got {u.shape}."
            )
        r = self.router(u)  # (T, B, M)
        return self._run(u, r)


# ---------------------------------------------------------------------------
# Spiking Raven: routing-slot memory of reset-free spiking units
# ---------------------------------------------------------------------------


class SpikingSlotMemory(nnx.Module):
    r"""Spiking Routing-Slot Memory: a slot memory whose slots are *spiking* units.

    This is the spiking sibling of :class:`RavenRSM`. It keeps the two ideas that
    make Raven a high-recall memory -- a bank of ``M`` independent **slots** and
    the *same* sparse write **router** -- but replaces each slot's linear
    accumulator with the **reset-free spiking membrane** of
    :class:`spyx.nn.PSU_LIF`: a leaky integrator ``V \leftarrow \beta V + x`` that
    emits a surrogate spike ``s = \sigma(V - \text{threshold})``. The result is
    **dual sparsity** -- sparse in *time* (spikes) *and* sparse in *slots*
    (routing).

    The slot membrane ``V_t`` has shape ``(B, M, d_slot)``. Per step ``t``, from
    the input ``u_t``:

    * the write router ``r_t = SlotRouter(u_t) \in [0, 1]^{(B, M)}`` (the **exact**
      router type reused from :class:`RavenRSM` -- ``self.router`` is a
      :class:`SlotRouter`, not a fork), and
    * the write ``U_t = reshape(W_u u_t) \in (B, M, d_slot)``.

    The membrane is then advanced with the routed, reset-free spiking recurrence

    .. math::
        V_t = (1 - r_t) \odot V_{t-1} + r_t \odot (\beta \odot V_{t-1} + U_t),
        \qquad s_t = \sigma(V_t - \text{threshold}),

    where ``\beta = sigmoid(raw_beta) \in (0, 1)^{(M, d_slot)}`` is a static,
    learnable per-slot / per-dim leak. **Shielding:** where ``r_t[m] = 0`` the
    update collapses to ``V_t[m] = V_{t-1}[m]`` -- the slot's membrane (and hence
    its spike) is passed through byte-for-byte unchanged, shielded from
    interference exactly as in :class:`RavenRSM`. Where ``r_t[m] = 1`` the slot
    runs a plain :class:`spyx.nn.PSU_LIF` step ``V \leftarrow \beta V + U_t``.

    **Output** is the raw slot spike train of shape ``(T, B, M, d_slot)`` (no
    dense readout projection -- the block *is* a spiking memory; compose a linear
    head downstream if real-valued outputs are needed).

    Reset-freeness is deliberate: the membrane recurrence stays a first-order
    linear map per slot, so -- exactly as documented for :class:`spyx.nn.PSU_LIF`
    -- a chunked / :func:`jax.lax.associative_scan` parallel form is *possible*.
    Because the per-step transition here is *input-dependent* through the router
    gate ``(1 - r_t)``, the associative element is the affine map
    ``V \mapsto A_t V + b_t`` with ``A_t = (1 - r_t) + r_t \beta`` and
    ``b_t = r_t U_t``; only the sequential :func:`jax.lax.scan` reference is
    implemented here (an honest baseline), matching :class:`RavenRSM`.

    Reductions (exercised by the tests): a **dense** router (``r_t`` all-ones)
    turns every slot into an independent, always-written
    :class:`spyx.nn.PSU_LIF` -- i.e. a plain bank of spiking leaky integrators
    driven by ``U_t``; the routing is what makes it a *memory*.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = 8,
        d_slot: int | None = None,
        *,
        hard_top_k: int | None = None,
        beta_init: float = 0.9,
        threshold: float = 1.0,
        activation=None,
        rngs: nnx.Rngs,
    ):
        """
        :d_model: Input feature width.
        :n_slots: Number of independent memory slots ``M``.
        :d_slot: Per-slot membrane width (defaults to ``d_model``).
        :hard_top_k: If set, the router keeps only its ``k`` most-active slots per
            step (straight-through top-``k``); the default is a soft gate.
        :beta_init: Initial per-slot leak in ``(0, 1)`` (stored as a logit).
        :threshold: Firing threshold on the membrane.
        :activation: :class:`spyx.axn.Axon` surrogate spike; defaults to
            ``superspike`` (matching :class:`spyx.nn.PSU_LIF`).
        :rngs: NNX PRNG collection.
        """
        if d_slot is None:
            d_slot = d_model
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1; got {n_slots}.")
        if d_slot < 1:
            raise ValueError(f"d_slot must be >= 1; got {d_slot}.")
        if not 0.0 < beta_init < 1.0:
            raise ValueError(f"beta_init must be in (0, 1); got {beta_init}.")

        self.d_model = d_model
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_SPIKE

        # Reuse the *exact* router mechanism from RavenRSM (same SlotRouter class).
        self.router = SlotRouter(d_model, n_slots, hard_top_k=hard_top_k, rngs=rngs)
        # Write projection: u_t -> (M * d_slot), reshaped to (M, d_slot).
        self.write = nnx.Linear(d_model, n_slots * d_slot, rngs=rngs)

        # Static learnable per-slot / per-dim leak, stored as a raw logit so that
        # beta = sigmoid(raw_beta) stays in (0, 1). Init near ``beta_init`` (slow
        # leak -> long membrane memory) with a little jitter.
        logit = float(jnp.log(beta_init / (1.0 - beta_init)))
        noise = 0.01 * jax.random.normal(rngs.params(), (n_slots, d_slot))
        self.raw_beta = nnx.Param(jnp.full((n_slots, d_slot), logit) + noise)

    @property
    def beta(self) -> jax.Array:
        """Effective per-slot / per-dim leak ``beta = sigmoid(raw_beta)`` in ``(0, 1)``."""
        return jax.nn.sigmoid(self.raw_beta[...])

    def initial_state(self, batch_size: int) -> jax.Array:
        """Return zero slot membrane of shape ``(batch_size, M, d_slot)``."""
        return jnp.zeros((batch_size, self.n_slots, self.d_slot), dtype=jnp.float32)

    def _route(self, u_t: jax.Array) -> jax.Array:
        """Expose the reused router: ``u_t (..., d_model) -> r (..., M)``."""
        return self.router(u_t)

    def step(self, state: jax.Array, u_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One reset-free spiking-slot timestep.

        :state: slot membrane ``V_{t-1}``, shape ``(B, M, d_slot)``.
        :u_t: input ``(B, d_model)``.
        :return: ``(V_t, s_t)`` -- the new membrane and the slot spikes of shape
            ``(B, M, d_slot)``.
        """
        r_t = self.router(u_t)  # (B, M)
        U_t = self.write(u_t).reshape(u_t.shape[0], self.n_slots, self.d_slot)
        beta = self.beta[None]  # (1, M, d_slot)
        gated = beta * state + U_t
        r_exp = r_t[..., None]  # (B, M, 1)
        v_new = (1.0 - r_exp) * state + r_exp * gated
        spikes = self.spike(v_new - self.threshold)
        return v_new, spikes

    def _run(self, u: jax.Array, r: jax.Array) -> jax.Array:
        """Core recurrence with a *precomputed* router ``r`` of shape ``(T, B, M)``.

        Factored out so tests (and the dense-router reduction) can force ``r``.
        """
        T, B, _ = u.shape
        U = self.write(u).reshape(T, B, self.n_slots, self.d_slot)
        beta = self.beta[None]  # (1, M, d_slot)

        def scan_step(state, inp):
            r_t, U_t = inp
            r_exp = r_t[..., None]
            gated = beta * state + U_t
            v_new = (1.0 - r_exp) * state + r_exp * gated
            spikes = self.spike(v_new - self.threshold)
            return v_new, spikes

        v0 = self.initial_state(B)
        _, spikes = jax.lax.scan(scan_step, v0, (r, U))  # (T, B, M, d_slot)
        return spikes

    def __call__(self, u: jax.Array) -> jax.Array:
        """Apply the spiking slot memory to a time-major input.

        :u: real array of shape ``(T, B, d_model)``.
        :return: spike train of shape ``(T, B, M, d_slot)``.
        """
        if u.ndim != 3 or u.shape[-1] != self.d_model:
            raise ValueError(
                f"SpikingSlotMemory expects [T, B, d_model={self.d_model}]; "
                f"got {u.shape}."
            )
        r = self.router(u)  # (T, B, M)
        return self._run(u, r)


# ---------------------------------------------------------------------------
# Synthetic associative-recall task (MQAR-style)
# ---------------------------------------------------------------------------


def make_recall_batch(
    key: jax.Array,
    *,
    batch: int = 8,
    n_pairs: int = 3,
    n_keys: int = 8,
    n_values: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """Generate a multi-query associative-recall (MQAR-style) batch.

    Each example is a sequence of ``n_pairs`` ``(key, value)`` bindings followed
    by a single **query** token equal to one of the presented keys. The target
    is the value bound to the queried key — a task compressed-state SSMs fail at
    but slot-routed memories solve, because each binding can live in its own
    (interference-free) slot.

    Tokens are one-hot encoded into ``d_model = n_keys + n_values`` dims: key
    ``i`` -> ``e_i``; value ``j`` -> ``e_{n_keys + j}``. The query token reuses
    its key's encoding. Sequence length is ``T = 2 * n_pairs + 1``.

    :key: PRNG key.
    :batch: number of independent examples.
    :n_pairs: key/value bindings per example (distinct keys, sampled w/o repl.).
    :n_keys: key vocabulary size (must be ``>= n_pairs``).
    :n_values: value vocabulary size.
    :return: ``(u, target)`` where ``u`` is ``(T, B, d_model)`` float one-hots
        and ``target`` is ``(B,)`` int32 value ids for the query.
    """
    if n_keys < n_pairs:
        raise ValueError(f"n_keys ({n_keys}) must be >= n_pairs ({n_pairs}).")
    d_model = n_keys + n_values
    T = 2 * n_pairs + 1

    keys_out = jnp.zeros((T, batch, d_model), dtype=jnp.float32)
    targets = jnp.zeros((batch,), dtype=jnp.int32)

    for b in range(batch):
        key, k_perm, k_val, k_q = jax.random.split(key, 4)
        # Distinct keys for this example.
        key_ids = jax.random.permutation(k_perm, n_keys)[:n_pairs]
        value_ids = jax.random.randint(k_val, (n_pairs,), 0, n_values)

        for p in range(n_pairs):
            kid = int(key_ids[p])
            vid = int(value_ids[p])
            keys_out = keys_out.at[2 * p, b, kid].set(1.0)
            keys_out = keys_out.at[2 * p + 1, b, n_keys + vid].set(1.0)

        q = int(jax.random.randint(k_q, (), 0, n_pairs))
        qid = int(key_ids[q])
        keys_out = keys_out.at[T - 1, b, qid].set(1.0)
        targets = targets.at[b].set(int(value_ids[q]))

    return keys_out, targets
