r"""Sigma-delta / graded-spike neuron — transmit only the *change* in activation.

A rate-coded spiking neuron fires on the *level* of its drive, so a slowly-varying
(temporally redundant) input still costs a steady stream of spikes. A **sigma-delta**
neuron instead integrates its input into a leaky membrane and emits the **quantized
change** of that membrane each step. When the membrane is stable the change rounds to
zero and *nothing is transmitted*, so redundant input is nearly free — the efficiency
mechanism behind Loihi-2's graded/sigma-delta mode.

The membrane is the same reset-free linear recurrence as
:class:`spyx.experimental.PSU_LIF`, ``V_t = beta V_{t-1} + x_t`` — so it is an
``O(log T)`` :func:`jax.lax.associative_scan` (the graded event is a pointwise
function of consecutive membrane values). The output is a **graded** event: a
signed, quantized real value (the change), mostly zero, not a binary spike. Its time
integral telescopes back to the membrane (``sum_t s_t = V_T``), so a downstream layer
recovers the integrated signal by accumulation — the standard sigma-delta dataflow.

References:

* O'Connor & Welling, *Sigma-Delta Quantized Networks*, ICLR 2017
  (`arXiv:1611.02024 <https://arxiv.org/abs/1611.02024>`_).
* Shrestha et al., efficient audio/video on Loihi 2 via resonate-and-fire +
  sigma-delta, ICASSP 2024 (`arXiv:2310.03251 <https://arxiv.org/abs/2310.03251>`_).

This is the reset-free *graded* sibling of the binary parallel neurons; its ``.parallel``
path and its per-step ``__call__`` are exactly equivalent, like ``PSU_LIF``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ..nn import _leaky_associative_op


def graded_quant(delta: jax.Array, step: float) -> jax.Array:
    """Quantize ``delta`` to the grid ``step`` with a straight-through gradient.

    Forward: ``round(delta / step) * step`` — the graded sigma-delta event, which is
    exactly zero whenever ``|delta| < step/2`` (the source of the sparsity). Backward:
    identity (straight-through), the standard estimator for a round nonlinearity.
    """

    @jax.custom_gradient
    def _q(d):
        out = jnp.round(d / step) * step
        return out, lambda g: g

    return _q(delta)


class SigmaDelta(nnx.Module):
    r"""Sigma-delta graded-spike neuron.

    Membrane ``V_t = clip(beta) V_{t-1} + x_t`` (reset-free, the same recurrence as
    :class:`spyx.experimental.PSU_LIF`); output ``s_t = graded_quant(V_t - V_{t-1})``.
    Follows the ``(x, state) -> (out, new_state)`` contract, so it drops into
    :class:`spyx.nn.Sequential` / :func:`spyx.nn.run`.

    :hidden_shape: per-neuron feature shape.
    :beta: membrane leak. Scalar constant if provided, else a learnable per-unit init
        (truncated-normal around 0.5, clipped to ``[0, 1]`` each forward), matching
        ``PSU_LIF``.
    :step: quantization grid of the graded event. Larger ``step`` → sparser, coarser
        events; smaller → denser, finer. Fixed (not learned) to keep the event grid
        stable.

    Honest caveat: this is the *feedforward* delta form (the quantization error is not
    fed back), so the reconstructed signal can drift by ``O(sqrt(T) * step)`` over long
    sequences. A closed-loop (error-feedback) variant is drift-free but sequential — a
    deliberate parallelism trade-off, as with the reset in ``PSU_LIF`` vs ``LIF``.
    """

    def __init__(
        self, hidden_shape: tuple, beta=None, step: float = 1.0, *, rngs: nnx.Rngs
    ):
        self.hidden_shape = hidden_shape
        self.step = float(step)
        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), hidden_shape
                )
                + 0.5
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, V):
        """One reset-free step ``(x, V) -> (graded_event, V_new)``.

        Scanning this over time via :func:`spyx.nn.run` reproduces :meth:`parallel`
        exactly (same clipped ``beta``, same grid).
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        V_new = beta * V + x
        s = graded_quant(V_new - V, self.step)
        return s, V_new

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(self.hidden_shape))

    def parallel(self, x):
        r"""Score a whole time-major ``[T, B, ...]`` sequence with an associative scan.

        Builds the membrane trace ``V`` (``V_{-1} = 0``) via
        :func:`jax.lax.associative_scan` in ``O(log T)`` depth, then emits the graded
        change ``graded_quant(V_t - V_{t-1})`` pointwise.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        A = jnp.broadcast_to(beta, x.shape)
        _, V = jax.lax.associative_scan(_leaky_associative_op, (A, x), axis=0)
        V_prev = jnp.concatenate([jnp.zeros_like(V[:1]), V[:-1]], axis=0)
        return graded_quant(V - V_prev, self.step)
