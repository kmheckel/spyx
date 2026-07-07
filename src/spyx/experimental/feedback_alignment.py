r"""Backprop-free training via random feedback: Feedback Alignment / Direct FA.

Backpropagation forms the hidden-layer error by multiplying the downstream error
by the *transpose of the forward weights*, ``W^T``. That exact ``W^T`` -- the
"weight-transport problem" -- is what makes BP biologically implausible and
awkward on neuromorphic hardware. **Feedback Alignment** replaces ``W^T`` with a
*fixed random* matrix ``B``; remarkably, the forward weights then rotate during
training until they partially align with ``B``, so the random pseudo-gradient
acquires a positive projection on the true gradient and the network learns.

Two variants are provided, both as exact-forward / random-backward ops built with
:func:`jax.custom_vjp`, so the surrogate spike gradient (:mod:`spyx.axn`) still
handles the neuron nonlinearity and everything composes with :func:`spyx.nn.run`:

* **Feedback Alignment (FA)** -- *layer-local*. Each :class:`FALinear` keeps its
  own fixed random ``B`` (the shape of ``W^T``) and, in the backward pass, sends
  ``g @ B`` to the previous layer instead of ``g @ W^T``. The weight update
  ``dW = x^T g`` is unchanged. Because it is a drop-in stateless layer, it slots
  straight into :class:`spyx.nn.Sequential` and is differentiated by a plain
  ``jax.grad`` / :func:`spyx.nn.run` -- the ``custom_vjp`` silently swaps in the
  random feedback everywhere, including through the BPTT scan.

* **Direct Feedback Alignment (DFA)** -- *global*. The output error ``e`` is
  projected **directly** to every hidden layer through a fixed random ``B_l``
  (shape ``(n_out, hidden_l)``), skipping the layer-by-layer chain entirely.
  Implemented with the :func:`dfa_inject` identity op, whose backward discards
  the true downstream cotangent and substitutes ``e @ B_l`` at each hidden
  neuron's output; ordinary autodiff then carries that through the surrogate
  spike (applying ``f'``) and the preceding matmul to form the DFA weight update.
  :func:`dfa_gradient` orchestrates this over a feedforward
  :class:`spyx.nn.Sequential` and returns grads that drop into
  ``optimizer.update(model, grads)``.

References (verified):

* Lillicrap, Cownden, Tweed & Akerman, *Random synaptic feedback weights support
  error backpropagation for deep learning*, Nature Communications 7:13276 (2016),
  `<https://www.nature.com/articles/ncomms13276>`_.
* Nøkland, *Direct Feedback Alignment Provides Learning in Deep Neural Networks*,
  NeurIPS 2016, `arXiv:1609.01596 <https://arxiv.org/abs/1609.01596>`_.
* Han & Yoo et al., *Spike-Train Level Direct Feedback Alignment*, Frontiers in
  Neuroscience 14:143 (2020),
  `<https://www.frontiersin.org/articles/10.3389/fnins.2020.00143/full>`_ -- DFA
  applied to spiking networks / on-chip training.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "Feedback",
    "fa_dense",
    "FALinear",
    "dfa_inject",
    "dfa_gradient",
]


class Feedback(nnx.Variable):
    """A fixed random feedback matrix -- state that is *never* trained.

    Marking feedback matrices with their own ``nnx.Variable`` subclass keeps them
    out of ``nnx.split(model, nnx.Param, ...)`` and out of
    ``nnx.Optimizer(model, tx, wrt=nnx.Param)``, so they stay frozen at their
    random init while the ``nnx.Param`` weights learn.
    """


def _dense(x: jax.Array, weight: jax.Array) -> jax.Array:
    """``x @ weight`` over the trailing feature axis, preserving leading dims."""
    in_features = x.shape[-1]
    flat = x.reshape(-1, in_features)
    out = flat @ weight
    return out.reshape(*x.shape[:-1], weight.shape[-1])


@jax.custom_vjp
def fa_dense(x: jax.Array, weight: jax.Array, feedback: jax.Array) -> jax.Array:
    """``x @ weight`` with a random-feedback backward pass (Feedback Alignment).

    Forward numerics are exactly ``x @ weight`` (``x`` is ``(..., in)``, ``weight``
    is ``(in, out)``). The custom VJP leaves the weight gradient exact,
    ``dW = x^T g``, but replaces the input gradient ``g @ weight^T`` with
    ``g @ feedback``, where ``feedback`` is the *fixed random* matrix of shape
    ``(out, in)`` standing in for ``weight^T``. This is the sole difference from a
    plain dense layer, and it is what makes learning weight-transport-free.
    """
    return _dense(x, weight)


def _fa_dense_fwd(x, weight, feedback):
    return _dense(x, weight), (x, weight, feedback)


def _fa_dense_bwd(res, g):
    x, weight, feedback = res
    in_features = x.shape[-1]
    out_features = weight.shape[-1]
    flat_x = x.reshape(-1, in_features)
    flat_g = g.reshape(-1, out_features)
    dweight = flat_x.T @ flat_g
    # Random feedback replaces weight^T: dx = g @ feedback, feedback is (out, in).
    dx = (flat_g @ feedback).reshape(x.shape)
    return dx, dweight, jnp.zeros_like(feedback)


fa_dense.defvjp(_fa_dense_fwd, _fa_dense_bwd)


class FALinear(nnx.Module):
    r"""Linear layer trained by Feedback Alignment (fixed random backward weights).

    A drop-in replacement for ``nnx.Linear``: the forward pass is the identical
    affine ``x @ weight + bias``, but the backward pass propagates the error to the
    previous layer through a *fixed random* ``feedback`` matrix instead of
    ``weight^T`` (see :func:`fa_dense`). Being stateless, it slots straight into
    :class:`spyx.nn.Sequential` and is scored over time by :func:`spyx.nn.run`;
    a plain ``jax.grad`` through the model then yields the FA pseudo-gradient
    automatically, including through the BPTT scan.

    :in_features: input feature size.
    :out_features: output feature size.
    :use_bias: add a learnable bias (default ``True``).
    :feedback_stddev: standard deviation of the fixed random feedback init.
    :rngs: ``nnx.Rngs``; ``rngs.params()`` seeds the weight, ``rngs.params()`` the
        (frozen) feedback.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        feedback_stddev: float = 0.05,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (in_features, out_features))
        )
        self.bias = nnx.Param(jnp.zeros((out_features,))) if use_bias else None
        # Fixed random feedback standing in for weight^T; shape (out, in).
        self.feedback = Feedback(
            nnx.initializers.normal(feedback_stddev)(
                rngs.params(), (out_features, in_features)
            )
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        y = fa_dense(x, self.weight[...], self.feedback[...])
        if self.bias is not None:
            y = y + self.bias[...]
        return y


@jax.custom_vjp
def dfa_inject(h: jax.Array, error: jax.Array, feedback: jax.Array) -> jax.Array:
    """Identity forward; backward injects a direct random projection of ``error``.

    Forward: returns ``h`` unchanged, so wrapping a hidden neuron's spike output in
    ``dfa_inject`` does not perturb the forward pass at all. Backward: the true
    downstream cotangent is **discarded** and replaced by ``error @ feedback`` --
    the Direct-Feedback-Alignment signal, where ``error`` is the global output
    error ``(batch, n_out)`` and ``feedback`` is the fixed random ``(n_out,
    hidden)`` matrix. Ordinary autodiff then carries that cotangent back through
    the surrogate spike (applying ``f'``) and the preceding matmul, so each hidden
    layer's weight update depends only on the output error, never on the
    downstream weights. Used by :func:`dfa_gradient`.
    """
    del error, feedback
    return h


def _dfa_inject_fwd(h, error, feedback):
    return h, (error, feedback, jnp.shape(h))


def _dfa_inject_bwd(res, g):
    error, feedback, h_shape = res
    del g  # the true downstream cotangent is intentionally discarded (DFA)
    c = jnp.broadcast_to(error @ feedback, h_shape)
    return c, jnp.zeros_like(error), jnp.zeros_like(feedback)


dfa_inject.defvjp(_dfa_inject_fwd, _dfa_inject_bwd)


LossFn = Callable[[jax.Array, Any], jax.Array]
"""``(logits, targets) -> scalar`` loss, differentiable w.r.t. ``logits``."""


def dfa_gradient(
    model: nnx.Module,
    x: jax.Array,
    targets: Any,
    feedbacks: list[jax.Array],
    *,
    loss_fn: LossFn,
) -> nnx.State:
    r"""Direct-Feedback-Alignment gradients for a feedforward spiking stack.

    ``model`` is a :class:`spyx.nn.Sequential` whose layers alternate stateless
    transforms (``FALinear`` / ``nnx.Linear`` / ``Flatten``) and surrogate
    spiking neurons (any layer exposing ``initial_state``), ending in a stateless
    **readout**. It is evaluated as a single feedforward step (zero initial
    neuron state), so use a neuron that spikes on its *current* drive, e.g.
    :class:`spyx.nn.PSU_LIF` (``V = beta V + x`` then spike) rather than one that
    spikes on the pre-update membrane.

    The update is pure DFA: the output error ``e = d loss / d logits`` is projected
    to each hidden neuron through its fixed random ``feedbacks[l]`` (shape
    ``(n_out, hidden_l)``) via :func:`dfa_inject`; the surrogate spike supplies
    ``f'``; the readout is trained by the true error. Hidden layers are decoupled
    with :func:`jax.lax.stop_gradient`, so no error crosses between them -- each
    learns only from the direct projection. Returned grads match ``model``'s
    ``nnx.Param`` tree and drop into ``optimizer.update(model, grads)``.

    :model: the feedforward ``Sequential`` described above.
    :x: input batch ``(batch, in)``.
    :targets: labels forwarded to ``loss_fn``.
    :feedbacks: one fixed random ``(n_out, hidden_l)`` matrix per hidden neuron.
    :loss_fn: ``(logits, targets) -> scalar``.
    :return: an ``nnx.State`` of gradients over the model's ``nnx.Param`` tree.
    """
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    batch = x.shape[0]

    def _forward_logits(p):
        m = nnx.merge(graphdef, p, rest)
        h = x
        for layer in m.layers:
            if hasattr(layer, "initial_state"):
                h, _ = layer(h, layer.initial_state(batch))
            else:
                h = layer(h)
        return h

    logits = _forward_logits(params)
    error = jax.lax.stop_gradient(jax.grad(lambda z: loss_fn(z, targets))(logits))

    def objective(p):
        m = nnx.merge(graphdef, p, rest)
        h = x
        k = 0
        total = jnp.asarray(0.0, logits.dtype)
        for layer in m.layers:
            if hasattr(layer, "initial_state"):
                out, _ = layer(h, layer.initial_state(batch))
                # Seed cotangent error @ feedbacks[k] at this hidden output.
                total = total + jnp.sum(dfa_inject(out, error, feedbacks[k]))
                k = k + 1
                # Decouple layers: no error flows between hidden blocks (DFA).
                h = jax.lax.stop_gradient(out)
            else:
                h = layer(h)
        # After the loop ``h`` is the readout logits; train it by the true error.
        total = total + jnp.sum(h * error)
        return total

    return jax.grad(objective)(params)
