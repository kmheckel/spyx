"""Compressed-space CMA-ES on SHD — the adaptive-ES arm the earlier runs were missing.

Vanilla OpenAI-ES (isotropic, fixed sigma, no adaptation) only reached surrogate
*parity* in the 512-dim hypernetwork space. This runs the strategy the Spyx
neuroevolution work actually used — **CMA-ES** (adaptive full covariance + step
size, via evosax) — over the same synecdoche RandomProjection latent, to test
whether an adaptive ES *beats* the surrogate once the search is low-dimensional.

Ask/tell loop, JIT-scanned over generations (one dispatch) in the spyx style: each
generation asks a population, evaluates fitness on the eval batches (vmapped over
the population), and tells. Needs evosax + synecdoche on the path. Writes
``shd_cmaes_results.json``.
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import synecdoche as syn
from flax import nnx
from jax.flatten_util import ravel_pytree

import spyx.axn as axn
import spyx.data as data
import spyx.fn as fn
import spyx.nn as snn

try:
    from evosax.algorithms import CMA_ES
except ImportError:
    from evosax import CMA_ES  # legacy path

CHANNELS, N_CLASSES = 128, 20
SAMPLE_T = int(os.environ.get("SAMPLE_T", "128"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
BATCH = int(os.environ.get("BATCH", "256"))
EMB = int(os.environ.get("EMB", "128"))
POP = int(os.environ.get("POP", "128"))
GENERATIONS = int(os.environ.get("GENERATIONS", "150"))
EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", "4"))  # fitness eval budget / gen
TEST_BATCHES = int(os.environ.get("TEST_BATCHES", "8"))
SEED = int(os.environ.get("SEED", "0"))

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def _unpack(o):
    return jnp.unpackbits(jnp.asarray(o), axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)


def load_shd():
    dl = data.SHD_loader(batch_size=BATCH, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=0)
    tr_o, tr_l = dl.prestage("train")
    te_o, te_l = dl.prestage("test")
    ne = min(EVAL_BATCHES, tr_o.shape[0])
    nt = min(TEST_BATCHES, te_o.shape[0])
    ex = jnp.stack([_unpack(tr_o[i]) for i in range(ne)])
    ey = jnp.stack([jnp.asarray(tr_l[i]) for i in range(ne)])
    tx = jnp.stack([_unpack(te_o[i]) for i in range(nt)])
    ty = jnp.stack([jnp.asarray(te_l[i]) for i in range(nt)])
    return (ex, ey), (tx, ty)


class SpikingClassifier(nnx.Module):
    def __init__(self, *, rngs):
        self.net = snn.Sequential(
            nnx.Linear(CHANNELS, HIDDEN, use_bias=False, rngs=rngs),
            snn.LIF((HIDDEN,), activation=axn.superspike(), rngs=rngs),
            nnx.Linear(HIDDEN, N_CLASSES, use_bias=False, rngs=rngs),
            snn.LI((N_CLASSES,), rngs=rngs),
        )

    def __call__(self, x):
        return snn.run(self.net, x, batch_major=True)[0]


def build_hyper_flat(seed):
    model = SpikingClassifier(rngs=nnx.Rngs(seed))
    snn_gd, _p, snn_rest = nnx.split(model, nnx.Param, ...)
    hyper = syn.RandomProjection(nnx.state(model, nnx.Param), EMB, rngs=nnx.Rngs(seed + 100))
    hgd, hp, hrest = nnx.split(hyper, nnx.Param, ...)
    theta0, unravel = ravel_pytree(hp)

    def to_snn(theta):
        gen = nnx.merge(hgd, unravel(theta), hrest)()
        return nnx.merge(snn_gd, gen, snn_rest)

    return theta0, to_snn


def main():
    print(
        f"backend={jax.default_backend()}  CMA-ES  SHD T={SAMPLE_T} H={HIDDEN} emb={EMB} "
        f"pop={POP} gens={GENERATIONS} eval_batches={EVAL_BATCHES}",
        flush=True,
    )
    (ex, ey), test = load_shd()
    theta0, to_snn = build_hyper_flat(SEED)
    d = theta0.size
    print(f"search dim (hypernetwork latent): {d}", flush=True)

    def fitness(theta):  # mean loss over the eval batches (minimise)
        return jnp.mean(jax.vmap(lambda x, y: ce(to_snn(theta)(x), y))(ex, ey))

    prime_lr = float(os.environ.get("PRIME_LR", "0.1"))
    tx, ty = test

    std_init = float(os.environ.get("STD_INIT", "0.1"))  # MUST match param scale (~0.02)

    def run_cma(prime: bool):
        strategy = CMA_ES(population_size=POP, solution=jnp.zeros(d))
        # default std_init=1.0 is ~50x the hypernetwork param scale (0.02) -> the
        # initial population is all-garbage and CMA never gets a ranking signal.
        es_params = strategy.default_params.replace(std_init=std_init)
        key = jax.random.PRNGKey(SEED)
        state = strategy.init(key, mean=theta0, params=es_params)

        @jax.jit
        def evolve(state, key):
            def gen(carry, _):
                state, key = carry
                key, ak, tk = jax.random.split(key, 3)
                cands, state = strategy.ask(ak, state, es_params)
                if prime:  # inject a surrogate-gradient candidate (fixed-set fitness)
                    mean = state.mean if hasattr(state, "mean") else state.best_solution
                    g_s = jax.grad(fitness)(mean)
                    cands = cands.at[0].set(mean - prime_lr * g_s)
                fits = jax.vmap(fitness)(cands)
                state, _ = strategy.tell(tk, cands, fits, state, es_params)
                return (state, key), jnp.min(fits)

            (state, _k), hist = jax.lax.scan(gen, (state, key), None, length=GENERATIONS)
            return state, hist

        t0 = time.perf_counter()
        state, hist = evolve(state, key)
        jax.block_until_ready(hist)
        dt = time.perf_counter() - t0
        mean = state.mean if hasattr(state, "mean") else state.best_solution
        model = to_snn(mean)
        test_acc = float(jnp.mean(jax.vmap(lambda x, y: acc_fn(model(x), y)[0])(tx, ty)))
        # train accuracy of the MEAN on the fitness set — separates overfit
        # (train high, test low) from a bad/unconverged mean (train also low).
        train_acc = float(jnp.mean(jax.vmap(lambda x, y: acc_fn(model(x), y)[0])(ex, ey)))
        return test_acc, train_acc, [float(h) for h in np.asarray(hist)], dt

    results = {}
    for name, prime in [("cmaes", False), ("primed_cmaes", True)]:
        test_acc, train_acc, curve, dt = run_cma(prime)
        results[name] = {"test_acc": test_acc, "train_acc": train_acc, "fitness_curve": curve}
        print(
            f"  {name:<14} train_acc={train_acc * 100:.1f}%  test_acc={test_acc * 100:.1f}%  "
            f"fitness {curve[0]:.3f} -> {curve[-1]:.3f}  ({dt:.0f}s)",
            flush=True,
        )

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "shd_cmaes_results.json"), "w") as f:
        json.dump(
            {
                "config": {"T": SAMPLE_T, "emb": EMB, "pop": POP, "gens": GENERATIONS,
                           "search_dim": int(d), "eval_batches": EVAL_BATCHES,
                           "prime_lr": prime_lr},
                "arms": results,
            },
            f,
            indent=2,
        )
    print("wrote shd_cmaes_results.json", flush=True)


if __name__ == "__main__":
    main()
