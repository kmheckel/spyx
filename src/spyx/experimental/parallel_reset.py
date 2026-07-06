r"""Reset-preserving parallel LIF via a fixed-point-threshold (FPT) scan.

.. note::
   **Experimental.** Unstable API; may change without a deprecation cycle. Its
   intended supported entry point is :class:`spyx.experimental.ParallelResetLIF`
   once wired into ``spyx.experimental.__init__``. Tests and studies import it
   from this concrete module path.

The parallelisation problem
---------------------------
A standard :class:`spyx.nn.LIF` subtracts a reset ``spikes * threshold`` from the
membrane every step,

.. math::
    V_{t+1} = \beta\, V_t + x_t - s_t\,\theta,\qquad s_t = H(V_t - \theta),

so each timestep depends on the *nonlinear* spike of the previous step. That
coupling forces the strictly sequential ``O(T)`` scan used by :func:`spyx.nn.run`.
:class:`spyx.nn.PSU_LIF` sidesteps this by **dropping the reset** entirely, buying
an ``O(\log T)`` associative scan at the cost of the reset dynamics.

``ParallelResetLIF`` is the **reset-preserving complement** to ``PSU_LIF``: it
keeps the *exact* hard reset of :class:`spyx.nn.LIF` (its :meth:`__call__` is
byte-for-byte the same recurrence) yet still parallelises the time loop.

The FPT fixed point
-------------------
Split the membrane into a reset-free part and an accumulated-reset part. Writing
the pre-input membrane the spike sees as :math:`V_t = U_t - R_t`,

.. math::
    U_t = \sum_{j<t}\beta^{\,t-1-j}\,x_j,\qquad
    R_t = \theta\sum_{j<t}\beta^{\,t-1-j}\,s_j,

both :math:`U` and :math:`R` are *first-order linear leaky recurrences* — each a
one-step-shifted :func:`jax.lax.associative_scan` over the same leak used by
``PSU_LIF`` (via :func:`spyx.nn._leaky_associative_op`), i.e. ``O(\log T)`` depth.
The only nonlinearity is the threshold that ties :math:`s` to :math:`R`. FPT
(Zhang et al., *Fixed-Point RNN*, arXiv:2506.12087) solves that coupling by a
fixed-point iteration:

0. compute the reset-free membrane :math:`U` once (an associative scan), and seed
   :math:`s^{(0)} = H(U - \theta)` (reset :math:`R = 0`);
k. given the current spike estimate, recompute the accumulated reset
   :math:`R^{(k)}` (another associative scan over ``s * threshold``) and
   re-threshold :math:`s^{(k)} = H(U - R^{(k)} - \theta)`.

Because :math:`R_t` depends only on spikes strictly *before* ``t``, iteration
``k`` makes the first ``k+1`` timesteps exact (a correctness *wavefront*
advancing one step per iteration), so ``K = T`` reproduces the sequential
hard-reset spike train **exactly**, while a small ``K`` (``~3``) reproduces it up
to short reset *cascades* — near-exact in the sparse / short-cascade activity
regime that trained SNNs occupy, and the fast-approximate mode by default. Each
iteration is ``O(\log T)`` parallel depth, so the whole solve is ``O(K \log T)``.

This is the same reset-preserving-parallel goal as the "Bullet Trains"
segment-scan neuron (arXiv:2603.13283); here it is expressed as an FPT iteration
that reuses Spyx's existing associative-scan machinery.

Two execution modes are provided:

* :meth:`__call__` -- one exact hard-reset timestep ``(x, V) -> (spikes, V)``,
  identical to :class:`spyx.nn.LIF`; a drop-in for :func:`spyx.nn.run`,
  :class:`spyx.nn.Sequential`, and NIR. This is the reference.
* :meth:`parallel` -- the whole time-major sequence at once via the FPT scan,
  ``O(K \log T)`` depth. Exact for ``K >= T``; fast-approximate for small ``K``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ..axn import superspike
from ..nn import _leaky_associative_op

# Module-level singleton for the default activation to avoid B008.
_DEFAULT_ACTIVATION = superspike()


def _presynaptic_leaky_scan(A: jax.Array, inputs: jax.Array) -> jax.Array:
    r"""Membrane a hard-reset neuron sees *before* integrating the current step.

    Runs the inclusive first-order leaky recurrence
    ``Q_t = A_t * Q_{t-1} + inputs_t`` with :func:`jax.lax.associative_scan`
    (``O(\log T)`` depth over ``axis=0``), then shifts it down one step along
    time so the result is ``pre_t = sum_{j<t} (prod) inputs_j`` with
    ``pre_0 = 0``. This matches the :class:`spyx.nn.LIF` timing, where the spike
    at step ``t`` is a function of the membrane accumulated from inputs strictly
    before ``t``.
    """
    _, Q = jax.lax.associative_scan(_leaky_associative_op, (A, inputs), axis=0)
    return jnp.concatenate([jnp.zeros_like(Q[:1]), Q[:-1]], axis=0)


class ParallelResetLIF(nnx.Module):
    r"""Hard-reset LIF whose time loop is parallelised with an FPT scan.

    The reset-preserving complement to :class:`spyx.nn.PSU_LIF`: same exact
    subtractive hard reset as :class:`spyx.nn.LIF`, but the sequential scan is
    replaced by a ``O(K \log T)`` fixed-point-threshold solve (see the module
    docstring for the derivation and the FPT / Bullet Trains references).

    :meth:`__call__` is numerically identical to :class:`spyx.nn.LIF` and is the
    reference; :meth:`parallel` scores a whole time-major sequence and converges
    to that reference exactly as ``K -> T``.
    """

    def __init__(
        self,
        hidden_shape: tuple,
        beta=None,
        threshold=1.0,
        activation=None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        :hidden_shape: Shape of the layer.
        :beta: decay rate. Scalar if provided, else learnable per-unit init.
        :threshold: firing threshold. Defaults to 1.
        :activation: spyx.axn.Axon object determining the surrogate spike.
        """
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.25)(
                    rngs.params(), self.hidden_shape
                )
                + 0.5
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, V):
        """One exact hard-reset timestep -- identical to :class:`spyx.nn.LIF`.

        :x: input vector coming from the previous layer.
        :V: neuron state tensor (pre-input membrane).

        Emits the surrogate spike on the incoming membrane, then applies the
        subtractive reset while integrating the input:
        ``V = beta * V + x - spikes * threshold``. This is the reference the
        :meth:`parallel` FPT solve reproduces.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        spikes = self.spike(V - self.threshold)
        V = beta * V + x - spikes * self.threshold
        return spikes, V

    def parallel(self, x, K: int = 3):
        r"""Score a whole time-major sequence with the FPT fixed-point scan.

        :x: input with shape ``[Time, Batch, ...]``.
        :K: number of fixed-point iterations. Iteration ``k`` makes the first
            ``k + 1`` timesteps exact, so ``K >= Time`` reproduces the
            sequential :class:`spyx.nn.LIF` spike train **exactly**; the default
            ``K = 3`` is the fast approximation, near-exact wherever reset
            cascades are short (sparse / low-``beta`` activity).
        :return: spikes with shape ``[Time, Batch, ...]``.

        Computes the reset-free pre-input membrane
        ``U_t = sum_{j<t} beta^{t-1-j} x_j`` once, then repeatedly subtracts the
        accumulated reset ``R_t = threshold * sum_{j<t} beta^{t-1-j} s_j`` (both
        one-step-shifted associative scans over the shared leak) and
        re-thresholds. Each iteration is ``O(\log T)`` parallel depth, so the
        solve is ``O(K \log T)``.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        # Broadcast the (scalar or per-unit) leak to every (Time, Batch, ...)
        # element so the linear-recurrence coefficient A_t == beta everywhere.
        A = jnp.broadcast_to(beta, x.shape)

        # (0) reset-free membrane and its spikes (accumulated reset R = 0).
        U = _presynaptic_leaky_scan(A, x)
        spikes = self.spike(U - self.threshold)

        # (k) subtract the reset implied by the current spikes, then re-threshold.
        for _ in range(K):
            R = _presynaptic_leaky_scan(A, spikes * self.threshold)
            spikes = self.spike(U - R - self.threshold)
        return spikes

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)
