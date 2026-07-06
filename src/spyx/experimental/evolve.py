"""Evolutionary :class:`~spyx.optimize.Solver`\\ s for :func:`spyx.optimize.compile_fit`.

.. note::
   **Experimental.** Unstable API. The CMA-ES solvers need ``evosax`` (the ``[evo]``
   extra); it is imported lazily so this module imports without it.

These plug gradient-free training into the *same* single-dispatch compiled loop as
backprop. Each is a ``Solver`` builder ``(loss_flat, params0) -> SolverImpl`` that
sets up ``ravel_pytree`` / the evosax strategy against the concrete parameters, so::

    from spyx.optimize import compile_fit
    from spyx.experimental import evolve

    model, hist = compile_fit(
        model, evolve.cmaes(population_size=128), loss_fn, (X, Y), epochs=20
    )

* :func:`es` — vanilla OpenAI-ES (isotropic antithetic NES) via Optax. Weakest;
  variance grows with dimension.
* :func:`cmaes` — CMA-ES ask/tell (adaptive full covariance + step size). Strong in
  low dimensions — e.g. a compressed / hypernetwork parameter space.
* :func:`primed_cmaes` — CMA-ES with a **surrogate-gradient candidate injected**
  each generation: the surrogate supplies a cheap high-quality population member,
  CMA-ES does the adaptation. The stable form of the "surrogate + evolution" hybrid
  (injection keeps CMA-ES's step-size bookkeeping consistent, unlike moving the mean
  directly).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..optimize import SolverImpl
from .hybrid import _es_flat

__all__ = ["es", "cmaes", "primed_cmaes"]


def es(tx: optax.GradientTransformation, *, num_samples: int = 32, sigma: float = 0.02):
    """OpenAI-ES solver: an antithetic-NES gradient estimate fed to an Optax update.

    :param tx: Optax transform applied to the ES gradient estimate.
    :param num_samples: population ``K`` (antithetic pairs).
    :param sigma: perturbation scale.
    """

    def build(loss_flat, params0):
        def init(key):
            return (params0, tx.init(params0))

        def step(state, batch, key):
            params, opt_state = state
            theta, unravel = ravel_pytree(params)
            g = _es_flat(
                theta,
                lambda t: loss_flat(unravel(t), batch),
                key,
                num_samples=num_samples,
                sigma=sigma,
            )
            grads = unravel(g)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss_flat(
                params, batch
            )

        return SolverImpl(init, step, lambda s: s[0])

    return build


def _mean(state: Any) -> jax.Array:
    """The current distribution mean across evosax versions."""
    for attr in ("mean", "best_solution", "best_member"):
        if hasattr(state, attr):
            return getattr(state, attr)
    raise AttributeError("evosax state has no mean/best_solution attribute")


def _cma_strategy(dim: int, population_size: int):
    try:
        from evosax.algorithms import CMA_ES  # evosax >= 0.2
    except ImportError:  # pragma: no cover - legacy path
        from evosax import CMA_ES  # ty: ignore[unresolved-import]
    return CMA_ES(population_size=population_size, solution=jnp.zeros((dim,)))


def cmaes(*, population_size: int = 128):
    """CMA-ES ask/tell solver (needs the ``[evo]`` extra).

    One generation per step: ask a population, evaluate each candidate's loss on the
    step's batch, tell. ``get_params`` returns the current mean. Scannable and
    JIT-compatible, so the whole evolution runs as one dispatch under
    :func:`~spyx.optimize.compile_fit`.
    """

    def build(loss_flat, params0):
        theta0, unravel = ravel_pytree(params0)
        strategy = _cma_strategy(theta0.size, population_size)
        es_params = strategy.default_params

        def init(key):
            return strategy.init(key, mean=theta0, params=es_params)

        def step(state, batch, key):
            ak, tk = jax.random.split(key)
            cands, state = strategy.ask(ak, state, es_params)
            fits = jax.vmap(lambda c: loss_flat(unravel(c), batch))(cands)
            state, _ = strategy.tell(tk, cands, fits, state, es_params)
            return state, jnp.min(fits)

        return SolverImpl(init, step, lambda s: unravel(_mean(s)))

    return build


def primed_cmaes(*, population_size: int = 128, prime_lr: float = 0.1):
    """CMA-ES with a surrogate-gradient candidate injected each generation.

    Each step computes the surrogate gradient at the current mean and injects
    ``mean - prime_lr · g_surrogate`` as a population member. If that point is good,
    CMA-ES's rank-based update moves toward it (accelerating early progress); if not,
    it is simply out-ranked and ignored — so injection never destabilises the
    strategy the way directly stepping the mean would. This is the ``0+1`` hybrid
    with an *adaptive* ES: the surrogate primes, CMA-ES adapts.

    :param prime_lr: step size for the injected surrogate candidate.
    """

    def build(loss_flat, params0):
        theta0, unravel = ravel_pytree(params0)
        strategy = _cma_strategy(theta0.size, population_size)
        es_params = strategy.default_params

        def init(key):
            return strategy.init(key, mean=theta0, params=es_params)

        def step(state, batch, key):
            ak, tk = jax.random.split(key)
            cands, state = strategy.ask(ak, state, es_params)
            mean = _mean(state)
            g_s = jax.grad(lambda t: loss_flat(unravel(t), batch))(mean)
            cands = cands.at[0].set(mean - prime_lr * g_s)  # inject the surrogate step
            fits = jax.vmap(lambda c: loss_flat(unravel(c), batch))(cands)
            state, _ = strategy.tell(tk, cands, fits, state, es_params)
            return state, jnp.min(fits)

        return SolverImpl(init, step, lambda s: unravel(_mean(s)))

    return build
