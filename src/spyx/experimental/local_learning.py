r"""Local online three-factor (e-prop / OTTT-style) plasticity for spiking layers.

Backprop-through-time stores every activation and replays them *backwards* to
assign credit, which is neither online nor local. **Eligibility propagation**
(e-prop) replaces the backward replay with a per-synapse **eligibility trace**
that is maintained *forward in time* and combined with an instantaneous
top-down **learning signal** (a neuromodulator) to update weights as the
sequence streams past — a biologically-motivated *three-factor* rule (pre x
post x modulator). This module implements that rule as a self-contained plastic
spiking layer whose weights are updated **during** the forward pass, with no
stored activations and no BPTT.

The three factors
-----------------
For a plastic weight ``W[i, j]`` (presynaptic unit ``i`` -> postsynaptic unit
``j``), each step ``t``:

1. **Presynaptic trace** — a low-pass filter of the input spikes,

   .. math:: \bar p^t_i = \alpha\, \bar p^{t-1}_i + x^t_i .

2. **Postsynaptic pseudo-derivative** — the surrogate slope of the spike
   nonlinearity at the membrane, computed *in the forward pass* (this is what
   removes the need for a backward pass):

   .. math:: \psi^t_j = h'(V^t_j - v_\mathrm{th}),
      \qquad h'(u) = \frac{1}{(1 + k|u|)^2}

   (the SuperSpike slope; the same function ``spyx.axn.superspike(k)`` uses in
   its backward pass).

3. **Eligibility trace** — a decaying accumulation of the pre x post
   coincidence, exactly a leaky integrator of the outer product:

   .. math:: e^t_{ij} = \gamma\, e^{t-1}_{ij} + \bar p^t_i\, \psi^t_j .

The weight is then moved by the trace **gated by the modulator** ``m`` (the
third factor — a reward / error / learning signal delivered per sample, or per
postsynaptic unit), averaged over the batch:

.. math:: \Delta W^t_{ij} = \eta\; \big\langle m^t\, e^t_{ij}\big\rangle_\mathrm{batch},
   \qquad W^t = W^{t-1} + \Delta W^t .

This is the online three-factor form shared by e-prop and OTTT: OTTT tracks the
presynaptic activity :math:`\bar p` and multiplies it by an instantaneous
gradient/learning signal, which is precisely ``m`` here.

Relation to the TTT delta rule
------------------------------
A Test-Time-Training linear layer treats its weights as a fast state and updates
them by one SGD step of a self-supervised loss; for a linear reconstruction loss
that step *is* the delta rule, a rank-1 outer-product update
``W <- W - eta (W k - v) k^T``. That is the **same outer-product skeleton** as
the eligibility update above: post-factor :math:`\otimes` pre-factor, scaled by a
rate. e-prop generalises it with (a) an eligibility *memory* ``gamma`` (the TTT
step keeps no trace, i.e. ``gamma = 0``) and (b) an explicit third factor ``m``
(TTT folds the modulator into the reconstruction error). Setting ``gamma = 0``
and letting ``m`` carry the reconstruction error recovers a delta-rule / TTT
inner step, so this module is the spiking, three-factor generalisation of the
TTT delta rule.

Meta-learning the rule (the outer loop)
---------------------------------------
The rule's coefficients — ``eta`` (rate), ``gamma`` (eligibility decay),
``alpha`` (presynaptic-filter decay) and ``beta`` (membrane leak) — are plain
:class:`flax.nnx.Param` scalars, while the plastic weight ``W`` and the traces
live in the *carried state* (not a ``Param``). So an **outer loop** can
meta-learn *how the layer learns*:

* by **gradient** — :func:`flax.nnx.grad` through :meth:`apply_sequence`
  differentiates a downstream meta-loss w.r.t. the coefficients (the inner
  weight updates are ordinary differentiable ops), or
* **gradient-free** — because the coefficients are the only ``nnx.Param``\ s,
  they are exactly what ``spyx.experimental.hybrid`` / ``spyx.experimental.evolve``
  perturb (both split on ``nnx.Param``), so ES / CMA-ES can meta-learn the rule
  without differentiating through it.

This is the ES-meta-learned-plasticity substrate: the *inner* learning is local,
online and BPTT-free; only the *outer* optimisation of a handful of coefficients
uses SGD or ES.

Honest caveat: unlike :class:`spyx.experimental.PSU_LIF` /
:class:`spyx.experimental.SigmaDelta`, this rule is **inherently sequential** in
``O(T)`` — the membrane (and hence the pseudo-derivative and eligibility) depends
on the running plastic ``W``, so there is no associative-scan shortcut. The
eligibility trace *alone* is a linear leaky integrator (an associative scan) only
when the drive is held fixed; coupling it back into ``W`` breaks that, by design.

References
----------
* Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, *A solution to
  the learning dilemma for recurrent networks of spiking neurons*, Nature
  Communications 11, 3625 (2020),
  `doi:10.1038/s41467-020-17236-y
  <https://www.nature.com/articles/s41467-020-17236-y>`_ (e-prop).
* Xiao, Meng, Zhang, He & Lin, *Online Training Through Time for Spiking Neural
  Networks*, NeurIPS 2022 (`arXiv:2210.04195
  <https://arxiv.org/abs/2210.04195>`_) (OTTT; three-factor online form).
* Sun, Li, Dalal, Xu, Vikram, Zhang, Dubois, Chen, Wang, Koyejo, Hashimoto &
  Guestrin, *Learning to (Learn at Test Time): RNNs with Expressive Hidden
  States*, ICML 2025 (`arXiv:2407.04620 <https://arxiv.org/abs/2407.04620>`_)
  (TTT; the delta-rule inner update).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ..axn import superspike

__all__ = ["surrogate_deriv", "ThreeFactorLIF"]

# Default forward spike; its backward slope matches :func:`surrogate_deriv`.
_DEFAULT_ACTIVATION = superspike()


def surrogate_deriv(v: jax.Array, k: float = 25.0) -> jax.Array:
    """Forward-computed postsynaptic pseudo-derivative ``h'(v) = 1/(1+k|v|)^2``.

    This is the SuperSpike surrogate slope — identical to the backward function of
    :func:`spyx.axn.superspike` — but evaluated as an ordinary forward quantity so
    the eligibility trace can be built *without* a backward pass. ``v`` is the
    membrane relative to threshold; larger ``k`` narrows the window in which a
    synapse is eligible.
    """
    return 1.0 / (1.0 + k * jnp.abs(v)) ** 2


def _expand_mod(mod: jax.Array) -> jax.Array:
    """Broadcast a modulator to align with an ``(B, in, out)`` eligibility trace.

    Accepts a per-sample signal ``(B,)`` -> ``(B, 1, 1)`` (gates every synapse of
    a sample equally) or a per-postsynaptic-unit signal ``(B, out)`` ->
    ``(B, 1, out)`` (a distinct learning signal per output neuron, as in e-prop).
    """
    if mod.ndim == 1:
        return mod[:, None, None]
    return mod[:, None, :]


class ThreeFactorLIF(nnx.Module):
    r"""Plastic-synapse LIF trained online by a local three-factor rule.

    A leaky integrate-and-fire layer whose input weights ``W`` are **plastic**:
    they are carried in the layer state and nudged every timestep by
    ``ΔW = eta * <modulator * eligibility>`` (see the module docstring), so
    learning happens *inside* the forward-through-time pass with no BPTT and no
    stored activations.

    Contract. :meth:`__call__` follows the Spyx ``(x, state) -> (out, new_state)``
    shape, with ``x = (pre, mod)`` a two-tuple of the presynaptic input and the
    per-step modulator — so it scans directly under :func:`jax.lax.scan`. Because
    ``x`` is a pytree rather than a single array, drive it with :meth:`apply_sequence`
    (its analogue of :meth:`spyx.experimental.PSU_LIF.parallel`) rather than
    :func:`spyx.nn.run`.

    :in_features: presynaptic width (input units).
    :out_features: postsynaptic width (LIF units).
    :eta: plasticity rate ``eta`` — learnable ``nnx.Param`` scalar.
    :gamma: eligibility-trace decay ``gamma in [0, 1]`` — learnable ``nnx.Param``.
        ``gamma = 0`` keeps no trace and recovers a delta-rule / TTT inner step.
    :alpha: presynaptic-filter decay ``alpha in [0, 1]`` — learnable ``nnx.Param``.
    :beta: membrane leak ``beta in [0, 1]`` — learnable ``nnx.Param``.
    :threshold: fixed firing threshold ``v_th``.
    :k: fixed SuperSpike slope for the forward pseudo-derivative ``psi``.
    :activation: ``spyx.axn`` forward spike; defaults to ``superspike(k=25)`` so
        its surrogate matches :func:`surrogate_deriv`.

    The plastic ``W`` and the traces are **state**, not ``Param``\ s: the only
    ``nnx.Param``\ s are the four coefficients, which is exactly what lets an outer
    SGD/ES loop meta-learn the rule (see the module docstring).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        eta: float = 0.01,
        gamma: float = 0.9,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1.0,
        k: float = 25.0,
        activation=None,
        rngs: nnx.Rngs | None = None,
    ):
        del rngs  # W starts at zeros; accepted for a uniform constructor.
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = float(threshold)
        self.k = float(k)
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION
        # Rule coefficients — the meta-learnable surface (the ONLY nnx.Params).
        self.eta = nnx.Param(jnp.full((), eta))
        self.gamma = nnx.Param(jnp.full((), gamma))
        self.alpha = nnx.Param(jnp.full((), alpha))
        self.beta = nnx.Param(jnp.full((), beta))

    def initial_state(self, batch_size: int):
        """Fresh carry ``(V, pbar, e, W)`` for a batch (plastic ``W`` starts at 0).

        ``V`` membrane ``(B, out)``, ``pbar`` presynaptic trace ``(B, in)``, ``e``
        eligibility ``(B, in, out)``, ``W`` plastic weight ``(in, out)`` shared
        across the batch. To warm-start ``W`` (e.g. a meta-learned init), build the
        tuple yourself and pass it as ``state`` to :meth:`apply_sequence`.
        """
        V = jnp.zeros((batch_size, self.out_features))
        pbar = jnp.zeros((batch_size, self.in_features))
        e = jnp.zeros((batch_size, self.in_features, self.out_features))
        W = jnp.zeros((self.in_features, self.out_features))
        return V, pbar, e, W

    def __call__(self, x, state):
        r"""One online step ``((pre, mod), state) -> (spikes, new_state)``.

        Integrates the drive through the *current* plastic ``W``, emits a spike,
        forms the eligibility trace from the presynaptic trace and the forward
        pseudo-derivative, then applies the modulated weight update — all in the
        forward pass. ``pre`` is ``(B, in)``; ``mod`` is ``(B,)`` or ``(B, out)``.
        """
        pre, mod = x
        V, pbar, e, W = state
        gamma = jnp.clip(self.gamma[...], 0, 1)
        alpha = jnp.clip(self.alpha[...], 0, 1)
        beta = jnp.clip(self.beta[...], 0, 1)

        # Drive and membrane use the current (pre-update) plastic weight.
        drive = pre @ W  # (B, out)
        V = beta * V + drive
        v = V - self.threshold
        spikes = self.spike(v)  # forward Heaviside, surrogate backward
        psi = surrogate_deriv(v, self.k)  # (B, out) forward pseudo-derivative

        # Three factors: presynaptic trace, postsynaptic psi, decaying coincidence.
        pbar = alpha * pbar + pre  # (B, in)
        coincidence = pbar[:, :, None] * psi[:, None, :]  # (B, in, out) outer product
        e = gamma * e + coincidence

        # Modulated, batch-averaged update — applied DURING the forward pass.
        dW = self.eta[...] * jnp.mean(_expand_mod(mod) * e, axis=0)  # (in, out)
        W = W + dW
        return spikes, (V, pbar, e, W)

    def apply_sequence(self, pre, mod, state=None):
        r"""Stream a whole time-major sequence, learning ``W`` online as it goes.

        :pre: presynaptic input ``(T, B, in)``.
        :mod: per-step modulator ``(T, B)`` or ``(T, B, out)``.
        :state: optional initial carry (defaults to :meth:`initial_state`); pass
            a hand-built tuple to warm-start the plastic ``W``.
        :return: ``(spikes, final_state)`` with ``spikes`` ``(T, B, out)`` and
            ``final_state`` the ``(V, pbar, e, W)`` carry whose ``W`` is the weight
            *learned* by the local rule over the sequence.

        This is the driver analogue of :meth:`spyx.experimental.PSU_LIF.parallel`;
        it is an ``O(T)`` :func:`jax.lax.scan` (the rule is sequential by design).
        Differentiating a loss on its output w.r.t. the coefficients meta-learns
        the rule; evaluating it forward-only lets ES do the same.
        """
        if state is None:
            state = self.initial_state(pre.shape[1])

        def step(carry, xt):
            spikes, carry = self(xt, carry)
            return carry, spikes

        final_state, spikes = jax.lax.scan(step, state, (pre, mod))
        return spikes, final_state
