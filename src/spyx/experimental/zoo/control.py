"""Control recipe: an evolution-trained spiking policy on a toy point-mass.

**Experimental.** Part of :mod:`spyx.experimental.zoo`; the API may change
without a deprecation cycle.

A tiny leaky-integrate-and-fire MLP (``LIF-MLP``) acts as a closed-loop
controller for a 1-D point mass ``(x, v)`` with dynamics

    v <- v + dt * force
    x <- x + dt * v

where ``force`` is the policy's scalar readout at each step. The objective is
to drive ``x`` to the origin, so the (minimised) cost is the accumulated
``x**2`` over the rollout. The policy is trained by an evolution strategy
(OpenAI-ES via :mod:`evosax`) rather than backprop: parameters are flattened
to a single vector with :func:`jax.flatten_util.ravel_pytree`, the ES proposes
a population of perturbations, and each candidate is scored by rolling out the
closed loop. No gradients flow through the (non-differentiable) rollout.

Everything runs on synthetic initial conditions with no downloads and is sized
to finish well under a second on CPU.
"""

from __future__ import annotations

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx

from ... import nn as snn

NAME = "control-lif-es"
APPLICATION = "control"
METHOD = "evolutionary"
ARCHITECTURE = "LIF-MLP"
DESCRIBE = (
    "Spiking LIF-MLP policy for a 1-D point-mass reaching task, trained by "
    "OpenAI-ES (evosax) on flattened parameters — no backprop through the "
    "closed-loop rollout."
)

# Environment / rollout constants (kept tiny for a sub-second CPU smoke path).
OBS_DIM = 2  # [position, velocity]
HIDDEN = 8
ACT_DIM = 1  # scalar force
DT = 0.1
ROLLOUT_STEPS = 20


def build(rngs: nnx.Rngs) -> snn.Sequential:
    """Build the spiking policy as a :class:`spyx.nn.Sequential`.

    ``obs -> Linear -> LIF -> Linear -> LI`` where the leaky-integrator readout
    provides a continuous scalar force. Drops into :func:`spyx.nn.run` and the
    manual closed-loop scan in :func:`rollout_cost`.
    """
    return snn.Sequential(
        nnx.Linear(OBS_DIM, HIDDEN, rngs=rngs),
        snn.LIF((HIDDEN,), rngs=rngs),
        nnx.Linear(HIDDEN, ACT_DIM, rngs=rngs),
        snn.LI((ACT_DIM,), rngs=rngs),
    )


def synthetic_batch(key: jax.Array | None = None, batch_size: int = 16) -> tuple:
    """Sample a batch of initial ``(position, velocity)`` states.

    :return: a 1-tuple ``(init_states,)`` with ``init_states`` of shape
        ``(batch_size, 2)`` — matching the ``(batch,) + args`` convention used
        by the other recipes' ``loss`` callables.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    init_states = jax.random.uniform(
        key, (batch_size, OBS_DIM), minval=-1.0, maxval=1.0
    )
    return (init_states,)


def rollout_cost(model: snn.Sequential, init_states: jax.Array) -> jax.Array:
    """Roll the closed loop and return the mean accumulated ``x**2`` cost.

    :model: spiking policy from :func:`build`.
    :init_states: ``(batch, 2)`` initial ``(x, v)`` states.
    :return: scalar cost (lower is better) — the ES fitness to minimise.
    """
    batch = init_states.shape[0]
    neuron_state = model.initial_state(batch)

    def step(carry, _):
        env, nstate = carry
        action, nstate = model(env, nstate)
        force = action[..., 0]
        v = env[..., 1] + DT * force
        x = env[..., 0] + DT * v
        cost = x**2
        return (jnp.stack([x, v], axis=-1), nstate), cost

    (_, _), costs = jax.lax.scan(
        step, (init_states, neuron_state), None, length=ROLLOUT_STEPS
    )
    return jnp.mean(jnp.sum(costs, axis=0))


# ``loss`` in the Recipe sense: a finite scalar objective on a synthetic batch.
loss = rollout_cost


def demo(steps: int = 15, *, seed: int = 0, population_size: int = 24) -> list[float]:
    """Run a few OpenAI-ES generations and return the mean-fitness history.

    :steps: number of ES generations.
    :seed: PRNG seed for reproducibility.
    :population_size: ES population size per generation.
    :return: list of per-generation mean rollout costs (lower is better).
    """
    import optax  # lazy: optax is a core dep but imported here to keep import light
    from evosax.algorithms import Open_ES

    key = jax.random.PRNGKey(seed)
    key, k_model, k_batch = jax.random.split(key, 3)

    model = build(nnx.Rngs(k_model))
    graphdef, params = nnx.split(model, nnx.Param)
    flat, unravel = jax.flatten_util.ravel_pytree(params)

    (init_states,) = synthetic_batch(k_batch)

    def fitness(flat_vec):
        candidate = nnx.merge(graphdef, unravel(flat_vec))
        return rollout_cost(candidate, init_states)

    es = Open_ES(
        population_size=population_size,
        solution=jnp.zeros_like(flat),
        optimizer=optax.adam(0.1),
        std_schedule=optax.constant_schedule(0.3),
    )
    es_params = es.default_params
    state = es.init(key, flat, es_params)

    history: list[float] = []
    for _ in range(steps):
        key, k_ask, k_tell = jax.random.split(key, 3)
        population, state = es.ask(k_ask, state, es_params)
        fitnesses = jax.vmap(fitness)(population)
        state, _ = es.tell(k_tell, population, fitnesses, state, es_params)
        history.append(float(jnp.mean(fitnesses)))
    return history
