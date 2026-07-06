r"""Hybrid surrogate-gradient / evolutionary training for spiking networks.

.. note::
   **Experimental.** Unstable API — may change without a deprecation cycle.
   Import it as ``from spyx.experimental.hybrid import hybrid_gradient`` (the
   :mod:`spyx.experimental.hybrid` submodule is importable without touching the
   package ``__init__``).

The idea
--------
Surrogate-gradient descent through a spiking network is *cheap* but *biased*:
the true forward objective uses a hard Heaviside spike whose gradient is zero
almost everywhere, so we substitute a smooth surrogate (``spyx.axn``) in the
backward pass. The resulting direction descends a *related* landscape, not the
true one, and the mismatch is a systematic bias.

Evolutionary strategies (ES / NES) estimate the gradient of the **true**
(hard-spike, non-differentiable) objective from forward evaluations alone, with
no surrogate at all. Pure ES is unbiased but high-variance and slow to converge
in high dimensions.

``hybrid_gradient`` combines the two so that ES pays only for what the surrogate
gets *wrong*:

1. ``g_s  = ∇θ loss_surrogate(θ)`` — the cheap, biased bulk descent direction
   (one ``jax.grad`` through the surrogate spikes).
2. ``g_es`` — an **antithetic NES** estimate of the gradient of the true loss,
   drawn over the *full flattened parameter vector*::

       g_es = 1/(2 σ K) Σ_k [loss_true(θ + σ ε_k) − loss_true(θ − σ ε_k)] ε_k,
       ε_k ~ N(0, I).

3. **Global orthogonalisation** (over the whole flattened vector, not per-leaf).
   Let ``ĝ_s = g_s / (‖g_s‖ + eps)``. Project the ES estimate onto the subspace
   the surrogate does *not* already cover::

       g_orth = g_es − ⟨g_es, ĝ_s⟩ ĝ_s.

4. **Corrected gradient**: ``g = g_s + λ · g_orth``.

The surrogate supplies the bulk direction; ES supplies *only* the correction in
the subspace where the surrogate is blind (its bias). Orthogonalising avoids
double-counting directions the surrogate already handles. This is the exact
complement of **Guided-ES** (Maheswaranathan et al. 2019), which restricts the
ES *search* to the surrogate's subspace; here we restrict it to the orthogonal
complement and add it as an error-correction term.

Self-normalising ``λ``
----------------------
With a *raw* ``λ`` the correction magnitude ``λ·‖g_orth‖`` depends on the ES
smoothing ``σ`` and sample count ``K`` — when ``‖g_orth‖ ≫ ‖g_s‖`` (common with a
high-variance estimate) even a modest ``λ`` lets the correction swamp the bulk
direction and *hurt*. Passing ``normalize=True`` reinterprets ``λ`` as a
dimensionless **fraction of the surrogate step**: the correction is rescaled by
``λ · ‖g_s‖ / ‖g_orth‖`` so that ``‖applied correction‖ = λ · ‖g_s‖`` exactly,
regardless of the ES scale. ``λ = 0.2`` then means "nudge the surrogate step by at
most 20 % in the direction it is blind to," which transfers across regimes and
keeps the ES term from ever dominating.

Surrogate-steered Self-Guided ES (variance reduction)
-----------------------------------------------------
The orthogonal correction above lives in the high-dimensional complement, so it is
*high variance* — the reason the raw ``λ`` blows up. :func:`sges_gradient` takes
the dual view of **Self-Guided ES** (Liu et al., IJCAI 2020): instead of adding ES
only in the complement, it uses the surrogate direction as SGES's *guiding
subspace* and **stratifies** the sampling — the along-guide directional derivative
is measured *exactly* (one antithetic pair, no Monte-Carlo variance) and the ES
budget is spent on the orthogonal complement. The result is an unbiased estimate of
the true gradient with several-fold lower variance than isotropic ES at the same
budget. Where Guided ES concentrates ES *inside* the surrogate subspace, and the
orthogonal hybrid puts it *entirely outside*, SGES does both — cheap-and-exact
in-subspace, sampled in the complement.

All the linear algebra happens on the flat parameter vector via
:func:`jax.flatten_util.ravel_pytree`, and perturbations are applied by
``nnx.split`` → perturb-flat → ``nnx.merge``, so the machinery is agnostic to
the model's pytree structure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.flatten_util import ravel_pytree

LossFn = Callable[..., jax.Array]
"""``(model, *batch) -> scalar`` loss. Surrogate losses must be differentiable
through the ``spyx.axn`` surrogate spikes; true losses need only be evaluable."""


def _split_and_ravel(model: nnx.Module):
    """Split ``model`` into (graphdef, flat params, unravel, rest).

    ``rest`` holds every non-``Param`` piece of state, threaded through
    :func:`nnx.merge` untouched so perturbations only ever hit the trainable
    parameters. ``unravel`` maps a flat vector back to the ``Param`` ``nnx.State``
    pytree so the result drops straight into ``optimizer.update(model, grads)``.
    """
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    theta, unravel = ravel_pytree(params)
    return graphdef, theta, unravel, rest


def _es_flat(
    theta: jax.Array,
    true_loss_flat: Callable[[jax.Array], jax.Array],
    key: jax.Array,
    *,
    num_samples: int,
    sigma: float,
) -> jax.Array:
    """Antithetic NES estimate of ∇ true-loss over the flat vector ``theta``.

    Antithetic pairs share each ``ε_k`` (the ``+`` and ``−`` legs), which cancels
    the even-order terms of the Taylor expansion and halves the variance versus a
    one-sided estimate. For a locally quadratic loss this is *unbiased*.
    """
    eps = jax.random.normal(key, (num_samples,) + theta.shape, dtype=theta.dtype)
    l_plus = jax.vmap(true_loss_flat)(theta[None] + sigma * eps)
    l_minus = jax.vmap(true_loss_flat)(theta[None] - sigma * eps)
    coeff = (l_plus - l_minus) / (2.0 * sigma * num_samples)  # (K,)
    return coeff @ eps  # (D,)


def es_gradient(
    model: nnx.Module,
    loss_true: LossFn,
    key: jax.Array,
    *,
    batch: tuple[Any, ...] = (),
    num_samples: int = 8,
    sigma: float = 0.01,
) -> nnx.State:
    """Pure antithetic-NES gradient of the *true* loss as a param pytree.

    Gradient-free: only forward evaluations of ``loss_true`` are used, so the
    loss may be non-differentiable (hard Heaviside spikes, hard accuracy, …).
    Returned grads match ``model``'s ``Param`` structure and drop into
    ``optimizer.update(model, grads)``. This is the "pure ES" baseline arm; it is
    also the term :func:`hybrid_gradient` orthogonalises against the surrogate.

    :param model: the Spyx / Flax NNX module whose params are perturbed.
    :param loss_true: ``(model, *batch) -> scalar`` true objective.
    :param key: a ``jax.random.PRNGKey``; antithetic pairs share ``ε``.
    :param batch: extra positional args forwarded to the loss (e.g. ``(x, y)``).
    :param num_samples: number ``K`` of antithetic perturbation pairs.
    :param sigma: perturbation scale ``σ`` (smoothing radius of the estimate).
    :return: an ``nnx.State`` of gradients matching the model's ``Param`` pytree.
    """
    graphdef, theta, unravel, rest = _split_and_ravel(model)

    def true_loss_flat(flat):
        return loss_true(nnx.merge(graphdef, unravel(flat), rest), *batch)

    g_es = _es_flat(theta, true_loss_flat, key, num_samples=num_samples, sigma=sigma)
    return unravel(g_es)


def _sges_flat(
    theta: jax.Array,
    true_loss_flat: Callable[[jax.Array], jax.Array],
    guide: jax.Array,
    key: jax.Array,
    *,
    num_samples: int,
    sigma: float,
    eps: float,
):
    r"""Surrogate-steered Self-Guided-ES estimate of ∇ true-loss (variance-reduced).

    ``guide`` (typically the surrogate gradient) defines a 1-D guiding direction
    ``u = guide/‖guide‖``. Following Self-Guided ES (Liu et al., IJCAI 2020) — but
    with the guiding subspace supplied by an *external surrogate* rather than
    historical ES estimates — the antithetic sampling is STRATIFIED:

    * the along-guide directional derivative ``a = ⟨∇f, u⟩`` is captured *exactly*
      by a single antithetic pair (the loss is deterministic along a fixed
      direction, so this costs one pair and carries no Monte-Carlo variance);
    * the remaining ``K-1`` antithetic pairs are drawn from the ORTHOGONAL
      COMPLEMENT of ``u`` and estimate ``P_⊥ ∇f`` — the part of the gradient the
      guide is blind to.

    ``g = a·u + g_⊥`` is unbiased for ∇f (up to the usual O(σ²) smoothing). Because
    the surrogate is a biased estimate of the true gradient, the true gradient has
    most of its support along ``u``; spending zero sampling variance there and all
    ``K-1`` samples on the complement yields a lower-variance estimate than
    isotropic ES (:func:`_es_flat`) at the same budget.
    """
    u = guide / (jnp.linalg.norm(guide) + eps)  # unit guiding direction (D,)
    # Along-guide component: one exact antithetic pair (deterministic → zero var).
    a = (true_loss_flat(theta + sigma * u) - true_loss_flat(theta - sigma * u)) / (
        2.0 * sigma
    )
    # Orthogonal component: K-1 antithetic pairs sampled in the complement of u.
    k_orth = max(num_samples - 1, 1)
    xi = jax.random.normal(key, (k_orth,) + theta.shape, dtype=theta.dtype)
    eps_perp = xi - (xi @ u)[:, None] * u[None, :]  # project out the guide direction
    l_plus = jax.vmap(true_loss_flat)(theta[None] + sigma * eps_perp)
    l_minus = jax.vmap(true_loss_flat)(theta[None] - sigma * eps_perp)
    coeff = (l_plus - l_minus) / (2.0 * sigma * k_orth)
    g_orth = coeff @ eps_perp  # (D,) estimate of P_⊥ ∇f
    g_es = a * u + g_orth
    return g_es, a, u, g_orth


def sges_gradient(
    model: nnx.Module,
    loss_surrogate: LossFn,
    loss_true: LossFn,
    key: jax.Array,
    *,
    batch: tuple[Any, ...] = (),
    num_samples: int = 8,
    sigma: float = 0.01,
    lam: float = 1.0,
    eps: float = 1e-8,
    return_diagnostics: bool = False,
):
    r"""Surrogate-steered Self-Guided ES — a variance-reduced 0+1 gradient.

    The surrogate gradient ``g_s`` *steers* a Self-Guided-ES estimate ``g_es`` of
    the true (hard-spike) loss gradient: ``g_s`` picks the guiding direction, its
    along-guide component is measured exactly, and ES spends its whole budget on
    the orthogonal complement (see :func:`_sges_flat`). Returns
    ``g = (1-λ)·g_s + λ·g_es`` — ``λ=1`` descends on the variance-reduced true-
    gradient estimate (the surrogate only steers sampling), ``λ=0`` recovers pure
    surrogate descent.

    This is the "0+1" method in its variance-reduced form: the 1st-order surrogate
    steers where the 0th-order ES samples land, buying Self-Guided ES's variance
    reduction (Liu et al., IJCAI 2020) with a surrogate-defined guiding subspace.
    Unlike :func:`hybrid_gradient` (which adds ES *only* in the orthogonal
    complement), ES here also corrects the surrogate's magnitude/sign *along* its
    own direction, via the exact directional derivative ``a``.

    :param model: the Spyx / Flax NNX module to differentiate.
    :param loss_surrogate: differentiable ``(model, *batch) -> scalar`` (the guide).
    :param loss_true: ``(model, *batch) -> scalar`` true objective (forward-only).
    :param key: ``jax.random.PRNGKey`` for the orthogonal ES perturbations.
    :param batch: extra positional args forwarded to both losses.
    :param num_samples: total antithetic pairs ``K`` (1 along-guide, ``K-1`` orth).
    :param sigma: ES perturbation scale ``σ``.
    :param lam: blend ``g = (1-λ)g_s + λ g_es``; ``λ=1`` = pure SGES estimate.
    :param eps: numerical floor for normalisation.
    :param return_diagnostics: also return the diagnostics dict.
    :return: an ``nnx.State`` of grads, or ``(grads, diagnostics)``.
    """
    graphdef, theta, unravel, rest = _split_and_ravel(model)

    def surrogate_loss_flat(flat):
        return loss_surrogate(nnx.merge(graphdef, unravel(flat), rest), *batch)

    def true_loss_flat(flat):
        return loss_true(nnx.merge(graphdef, unravel(flat), rest), *batch)

    g_s = jax.grad(surrogate_loss_flat)(theta)
    g_es, a, u, g_orth = _sges_flat(
        theta, true_loss_flat, g_s, key, num_samples=num_samples, sigma=sigma, eps=eps
    )
    g = (1.0 - lam) * g_s + lam * g_es
    grads = unravel(g)
    if return_diagnostics:
        g_s_norm = jnp.linalg.norm(g_s)
        g_es_norm = jnp.linalg.norm(g_es)
        diagnostics = {
            "along_guide_deriv": a,  # exact ⟨∇f_true, û_s⟩ (0th-order, 1 pair)
            "surrogate_along_guide": jnp.dot(g_s, u),  # the surrogate's own magnitude
            "g_orth_norm": jnp.linalg.norm(g_orth),
            "g_s_norm": g_s_norm,
            "g_es_norm": g_es_norm,
            # cosine of the SGES true-gradient estimate with the surrogate.
            "cosine": jnp.dot(g_es, g_s) / (g_es_norm * g_s_norm + eps),
            "g_s": g_s,
            "g_es": g_es,
            "g_orth": g_orth,
        }
        return grads, diagnostics
    return grads


def make_sges_hybrid_train_step(
    loss_surrogate: LossFn,
    loss_true: LossFn,
    *,
    num_samples: int = 8,
    sigma: float = 0.01,
    lam: float = 1.0,
) -> Callable[..., jax.Array]:
    r"""Single-step updater using :func:`sges_gradient` (surrogate-steered SGES).

    ``step(model, optimizer, key, *batch) -> true_loss`` — mirrors
    :func:`make_hybrid_train_step` but builds the update with the variance-reduced
    Self-Guided-ES gradient. Returns ``loss_true`` at the pre-update parameters.
    """

    def step(model, optimizer, key, *batch):
        grads = sges_gradient(
            model,
            loss_surrogate,
            loss_true,
            key,
            batch=tuple(batch),
            num_samples=num_samples,
            sigma=sigma,
            lam=lam,
        )
        loss = loss_true(model, *batch)
        optimizer.update(model, grads)
        return loss

    return step


def _hybrid_flat(
    model: nnx.Module,
    loss_surrogate: LossFn,
    loss_true: LossFn,
    key: jax.Array,
    *,
    batch: tuple[Any, ...],
    num_samples: int,
    sigma: float,
    lam: float,
    eps: float,
    normalize: bool,
):
    """Core routine: returns ``(unravel, g_flat, diagnostics)`` on the flat vector.

    Split out so both :func:`hybrid_gradient` and :func:`hybrid_diagnostics` share
    exactly one implementation of the projection algebra.
    """
    graphdef, theta, unravel, rest = _split_and_ravel(model)

    def surrogate_loss_flat(flat):
        return loss_surrogate(nnx.merge(graphdef, unravel(flat), rest), *batch)

    def true_loss_flat(flat):
        return loss_true(nnx.merge(graphdef, unravel(flat), rest), *batch)

    # 1. Cheap biased bulk direction from the surrogate backward pass.
    g_s = jax.grad(surrogate_loss_flat)(theta)
    # 2. Antithetic NES estimate of the true-loss gradient.
    g_es = _es_flat(theta, true_loss_flat, key, num_samples=num_samples, sigma=sigma)
    # 3. Global orthogonalisation against the (normalised) surrogate direction.
    g_s_norm = jnp.linalg.norm(g_s)
    g_s_hat = g_s / (g_s_norm + eps)
    proj = jnp.dot(g_es, g_s_hat)
    g_orth = g_es - proj * g_s_hat
    # 4. Effective correction weight. In ``normalize`` mode ``lam`` is a
    #    dimensionless *fraction of the surrogate step*: the correction is rescaled
    #    so ``‖lam_eff · g_orth‖ = lam · ‖g_s‖`` regardless of the ES variance /
    #    smoothing scale. This is the self-normalising ``λ · ‖g_s‖ / ‖g_orth‖`` that
    #    keeps the high-variance ES term from ever dominating the bulk direction
    #    (the failure mode when ‖g_orth‖ ≫ ‖g_s‖ at a fixed raw ``lam``).
    g_orth_norm = jnp.linalg.norm(g_orth)
    lam_eff = jnp.where(
        normalize, lam * g_s_norm / (g_orth_norm + eps), jnp.asarray(lam, theta.dtype)
    )
    # 5. Corrected gradient: bulk surrogate + orthogonal ES correction.
    g = g_s + lam_eff * g_orth

    g_es_norm = jnp.linalg.norm(g_es)
    diagnostics = {
        # cosine ⟨g_es, ĝ_s⟩ / ‖g_es‖ : how aligned ES is with the surrogate.
        "cosine": jnp.dot(g_es, g_s) / (g_es_norm * g_s_norm + eps),
        "g_orth_norm": g_orth_norm,  # magnitude of the raw correction
        "g_s_norm": g_s_norm,
        "g_es_norm": g_es_norm,
        "proj": proj,  # ⟨g_es, ĝ_s⟩
        "lam_eff": lam_eff,  # weight actually applied to g_orth
        # ‖applied correction‖ / ‖g_s‖ — in normalize mode this equals ``lam``.
        "correction_fraction": lam_eff * g_orth_norm / (g_s_norm + eps),
        # Flat vectors, exposed for tests / research inspection.
        "g_s": g_s,
        "g_es": g_es,
        "g_orth": g_orth,
    }
    return unravel, g, diagnostics


def hybrid_gradient(
    model: nnx.Module,
    loss_surrogate: LossFn,
    loss_true: LossFn,
    key: jax.Array,
    *,
    batch: tuple[Any, ...] = (),
    num_samples: int = 8,
    sigma: float = 0.01,
    lam: float = 1.0,
    eps: float = 1e-8,
    normalize: bool = False,
    return_diagnostics: bool = False,
):
    r"""Surrogate gradient corrected by orthogonalised evolutionary strategies.

    Computes ``g = g_s + λ · g_orth`` (see the module docstring for the full
    derivation), where ``g_s`` is the surrogate gradient and ``g_orth`` is the
    antithetic-NES estimate of the *true* gradient with its surrogate-aligned
    component projected out. The returned grads match ``model``'s ``Param``
    pytree, so::

        grads = hybrid_gradient(model, loss_surrogate, loss_true, key, batch=(x, y))
        optimizer.update(model, grads)

    :param model: the Spyx / Flax NNX module to differentiate.
    :param loss_surrogate: differentiable ``(model, *batch) -> scalar`` (surrogate
        spikes). Supplies the cheap biased bulk direction ``g_s``.
    :param loss_true: ``(model, *batch) -> scalar`` true objective (may be
        non-differentiable / hard-spike); evaluated only in the forward pass.
    :param key: a ``jax.random.PRNGKey`` for the ES perturbations.
    :param batch: extra positional args forwarded to both losses (e.g. ``(x, y)``).
    :param num_samples: number ``K`` of antithetic perturbation pairs.
    :param sigma: ES perturbation scale ``σ``.
    :param lam: weight ``λ`` on the orthogonal ES correction. ``λ = 0`` recovers
        pure surrogate descent. With ``normalize=True`` it is a dimensionless
        *fraction of the surrogate step* rather than a raw scale.
    :param eps: numerical floor for the normalisation of ``g_s``.
    :param normalize: if ``True``, self-normalise the correction so its magnitude
        is exactly ``λ · ‖g_s‖`` — the ES term becomes a bounded fraction of the
        surrogate step, immune to the ES variance/``σ`` scaling that otherwise lets
        ``‖g_orth‖`` swamp ``‖g_s‖``. Recommended when tuning ``λ`` across regimes.
    :param return_diagnostics: if ``True`` also return the diagnostics dict from
        :func:`hybrid_diagnostics` (``cosine``, ``g_orth_norm``, ``lam_eff``,
        ``correction_fraction``, the flat vectors, …).
    :return: an ``nnx.State`` of grads, or ``(grads, diagnostics)`` if
        ``return_diagnostics``.
    """
    unravel, g, diagnostics = _hybrid_flat(
        model,
        loss_surrogate,
        loss_true,
        key,
        batch=batch,
        num_samples=num_samples,
        sigma=sigma,
        lam=lam,
        eps=eps,
        normalize=normalize,
    )
    grads = unravel(g)
    if return_diagnostics:
        return grads, diagnostics
    return grads


def hybrid_diagnostics(
    model: nnx.Module,
    loss_surrogate: LossFn,
    loss_true: LossFn,
    key: jax.Array,
    *,
    batch: tuple[Any, ...] = (),
    num_samples: int = 8,
    sigma: float = 0.01,
    lam: float = 1.0,
    eps: float = 1e-8,
    normalize: bool = False,
) -> dict[str, jax.Array]:
    r"""Diagnostics for a hybrid step *without* applying it.

    Returns a dict describing the correction the ES term contributes:

    - ``cosine`` — ``⟨g_es, ĝ_s⟩ / ‖g_es‖``: alignment of the ES estimate with the
      surrogate direction. Near ``±1`` means ES mostly re-derives the surrogate
      (little to correct); near ``0`` means ES points somewhere the surrogate is
      blind (the regime where hybrid should help).
    - ``g_orth_norm`` — ``‖g_orth‖``: magnitude of the (raw) orthogonal correction.
    - ``g_s_norm`` / ``g_es_norm`` — the two source magnitudes.
    - ``proj`` — the scalar projection ``⟨g_es, ĝ_s⟩``.
    - ``lam_eff`` — the weight actually applied to ``g_orth`` (equals ``lam`` unless
      ``normalize``, where it is ``lam · ‖g_s‖ / ‖g_orth‖``).
    - ``correction_fraction`` — ``‖lam_eff · g_orth‖ / ‖g_s‖`` (equals ``lam`` in
      ``normalize`` mode); how big the applied correction is next to the surrogate.
    - ``g_s`` / ``g_es`` / ``g_orth`` — the flat vectors themselves.

    Same signature as :func:`hybrid_gradient` (minus ``return_diagnostics``).
    """
    _, _, diagnostics = _hybrid_flat(
        model,
        loss_surrogate,
        loss_true,
        key,
        batch=batch,
        num_samples=num_samples,
        sigma=sigma,
        lam=lam,
        eps=eps,
        normalize=normalize,
    )
    return diagnostics


def make_hybrid_train_step(
    loss_surrogate: LossFn,
    loss_true: LossFn,
    *,
    num_samples: int = 8,
    sigma: float = 0.01,
    lam: float = 1.0,
    normalize: bool = False,
) -> Callable[..., jax.Array]:
    r"""Build a single-step hybrid updater.

    The returned callable has signature ``(model, optimizer, key, *batch) ->
    true_loss`` and mutates ``model`` / ``optimizer`` in place via NNX, mirroring
    :func:`spyx.optimize.make_train_step` but using :func:`hybrid_gradient` to
    build the update. The scalar returned is ``loss_true`` evaluated at the
    *pre-update* parameters (the objective the ES term actually targets).

    :param loss_surrogate: differentiable ``(model, *batch) -> scalar``.
    :param loss_true: ``(model, *batch) -> scalar`` true objective.
    :param num_samples: number ``K`` of antithetic perturbation pairs.
    :param sigma: ES perturbation scale ``σ``.
    :param lam: weight ``λ`` on the orthogonal ES correction.
    :param normalize: self-normalise the correction to ``λ · ‖g_s‖`` (see
        :func:`hybrid_gradient`).
    :return: ``step(model, optimizer, key, *batch) -> true_loss``.
    """

    def step(model, optimizer, key, *batch):
        grads = hybrid_gradient(
            model,
            loss_surrogate,
            loss_true,
            key,
            batch=tuple(batch),
            num_samples=num_samples,
            sigma=sigma,
            lam=lam,
            normalize=normalize,
        )
        loss = loss_true(model, *batch)
        optimizer.update(model, grads)
        return loss

    return step


__all__ = [
    "hybrid_gradient",
    "hybrid_diagnostics",
    "es_gradient",
    "sges_gradient",
    "make_hybrid_train_step",
    "make_sges_hybrid_train_step",
]
