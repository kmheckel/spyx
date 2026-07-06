"""Full-space CR-FM-NES on SHD — the scalable adaptive ES the thesis actually used.

Vanilla OpenAI-ES is too noisy and CMA-ES is O(d^2) (can't fit a 9556^2 covariance),
so both flail on the full parameter space. CR-FM-NES (Nomura & Ono, CEC 2022) uses a
rank-1 + diagonal covariance -> **linear O(d) time/space**, built for high dimensions
(the strategy in research/misc/nmnist_evo_crfmnes.ipynb / the Spyx thesis). This runs
it directly on the SNN's ~9556 weights — NO hypernetwork compression — to test whether
an adaptive ES designed for scale works where the others failed.

Both vanilla and surrogate-primed (inject `mean - lr*g_surrogate` as a candidate each
generation). Fixed-set fitness, JIT-scanned ask/tell. std_init is matched to the
parameter scale (the default 1.0 strangled CMA). Writes shd_crfmnes_results.json.
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.flatten_util import ravel_pytree

import spyx.axn as axn
import spyx.data as data
import spyx.fn as fn
import spyx.nn as snn

try:
    from evosax.algorithms import CR_FM_NES
except ImportError:  # pragma: no cover
    from evosax.algorithms.distribution_based import CR_FM_NES

CHANNELS, N_CLASSES = 128, 20
SAMPLE_T = int(os.environ.get("SAMPLE_T", "128"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
BATCH = int(os.environ.get("BATCH", "256"))
POP = int(os.environ.get("POP", "128"))
GENERATIONS = int(os.environ.get("GENERATIONS", "500"))
EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", "8"))
TEST_BATCHES = int(os.environ.get("TEST_BATCHES", "12"))
STD_INIT = float(os.environ.get("STD_INIT", "0.1"))
PRIME_LR = float(os.environ.get("PRIME_LR", "0.05"))
SEED = int(os.environ.get("SEED", "0"))

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def _unpack(o):
    return jnp.unpackbits(jnp.asarray(o), axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)


def load_shd():
    dl = data.SHD_loader(
        batch_size=BATCH, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=0
    )
    tr_o, tr_l = dl.prestage("train")
    te_o, te_l = dl.prestage("test")
    ne, nt = min(EVAL_BATCHES, tr_o.shape[0]), min(TEST_BATCHES, te_o.shape[0])
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


def build_full_flat(seed):
    model = SpikingClassifier(rngs=nnx.Rngs(seed))
    gd, p, rest = nnx.split(model, nnx.Param, ...)
    theta0, unravel = ravel_pytree(p)
    return theta0, lambda theta: nnx.merge(gd, unravel(theta), rest)


def main():
    print(
        f"backend={jax.default_backend()}  CR-FM-NES  FULL-SPACE  SHD T={SAMPLE_T} H={HIDDEN} "
        f"pop={POP} gens={GENERATIONS} std_init={STD_INIT}",
        flush=True,
    )
    (ex, ey), test = load_shd()
    theta0, to_snn = build_full_flat(SEED)
    d = theta0.size
    print(f"full parameter dim: {d}  (no hypernetwork)", flush=True)

    def loss(theta):
        return jnp.mean(jax.vmap(lambda x, y: ce(to_snn(theta)(x), y))(ex, ey))

    tx, ty = test

    def run(prime: bool):
        strat = CR_FM_NES(population_size=POP, solution=jnp.zeros(d))
        esp = strat.default_params.replace(std_init=STD_INIT)
        state = strat.init(jax.random.PRNGKey(SEED), mean=theta0, params=esp)

        @jax.jit
        def evolve(state, key):
            def gen(carry, _):
                state, key = carry
                key, ak, tk = jax.random.split(key, 3)
                cands, state = strat.ask(ak, state, esp)
                if prime:
                    mean = state.mean if hasattr(state, "mean") else state.best_solution
                    cands = cands.at[0].set(mean - PRIME_LR * jax.grad(loss)(mean))
                fits = jax.vmap(loss)(cands)
                state, _ = strat.tell(tk, cands, fits, state, esp)
                return (state, key), jnp.min(fits)

            (state, _k), hist = jax.lax.scan(
                gen, (state, key), None, length=GENERATIONS
            )
            return state, hist

        t0 = time.perf_counter()
        state, hist = evolve(state, jax.random.PRNGKey(SEED + 1))
        jax.block_until_ready(hist)
        dt = time.perf_counter() - t0
        mean = state.mean if hasattr(state, "mean") else state.best_solution
        m = to_snn(mean)
        test_acc = float(jnp.mean(jax.vmap(lambda x, y: acc_fn(m(x), y)[0])(tx, ty)))
        train_acc = float(jnp.mean(jax.vmap(lambda x, y: acc_fn(m(x), y)[0])(ex, ey)))
        return test_acc, train_acc, [float(h) for h in np.asarray(hist)], dt

    results = {}
    for name, prime in [("crfmnes", False), ("primed_crfmnes", True)]:
        test_acc, train_acc, curve, dt = run(prime)
        results[name] = {
            "test_acc": test_acc,
            "train_acc": train_acc,
            "fitness_curve": curve,
        }
        print(
            f"  {name:<16} train_acc={train_acc * 100:.1f}%  test_acc={test_acc * 100:.1f}%  "
            f"loss {curve[0]:.3f} -> {curve[-1]:.3f}  ({dt:.0f}s)",
            flush=True,
        )

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "shd_crfmnes_results.json"), "w") as f:
        json.dump(
            {
                "config": {
                    "T": SAMPLE_T,
                    "full_dim": int(d),
                    "pop": POP,
                    "gens": GENERATIONS,
                    "std_init": STD_INIT,
                    "eval_batches": EVAL_BATCHES,
                },
                "arms": results,
            },
            f,
            indent=2,
        )
    print("wrote shd_crfmnes_results.json", flush=True)


if __name__ == "__main__":
    main()
