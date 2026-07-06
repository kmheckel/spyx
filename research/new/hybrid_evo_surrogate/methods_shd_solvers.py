"""Compressed-space solver shootout on SHD, via spyx.optimize.compile_fit.

Dogfoods the Solver API: every arm is the *same* single-dispatch compiled loop with
a different optimiser, all over the 512-dim synecdoche hypernetwork latent —

    surrogate      : optax.adam            (backprop / 1st-order)
    es             : evolve.es             (OpenAI-ES, the weak one)
    cmaes          : evolve.cmaes          (adaptive covariance — thesis-grade)
    primed_cmaes   : evolve.primed_cmaes   (surrogate-injected CMA-ES, the 0+1)

CMA-ES is only viable because the search is low-dimensional (the hypernetwork);
full-space CMA-ES would need a 9556² covariance. Run on the iGPU with evosax +
synecdoche on the path. Writes ``shd_solvers_results.json``.
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import synecdoche as syn
from flax import nnx

import spyx.axn as axn
import spyx.data as data
import spyx.fn as fn
import spyx.nn as snn
from spyx.experimental import evolve
from spyx.optimize import compile_fit

CHANNELS, N_CLASSES = 128, 20
SAMPLE_T = int(os.environ.get("SAMPLE_T", "128"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
BATCH = int(os.environ.get("BATCH", "256"))
EMB = int(os.environ.get("EMB", "128"))
TRAIN_BATCHES = int(os.environ.get("TRAIN_BATCHES", "100"))
TEST_BATCHES = int(os.environ.get("TEST_BATCHES", "12"))
POP = int(os.environ.get("POP", "128"))
GRAD_EPOCHS = int(os.environ.get("GRAD_EPOCHS", "30"))  # surrogate / es
CMA_EPOCHS = int(os.environ.get("CMA_EPOCHS", "6"))  # gens = epochs * n_batches
K = int(os.environ.get("K", "256"))
SEED = int(os.environ.get("SEED", "0"))

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def _unpack(o):
    return jnp.unpackbits(jnp.asarray(o), axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)


def load_shd():
    dl = data.SHD_loader(batch_size=BATCH, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=0)
    tr_o, tr_l = dl.prestage("train")
    te_o, te_l = dl.prestage("test")
    ntr, nte = min(TRAIN_BATCHES, tr_o.shape[0]), min(TEST_BATCHES, te_o.shape[0])
    X = jnp.stack([_unpack(tr_o[i]) for i in range(ntr)])
    Y = jnp.stack([jnp.asarray(tr_l[i]) for i in range(ntr)])
    tX = jnp.stack([_unpack(te_o[i]) for i in range(nte)])
    tY = jnp.stack([jnp.asarray(te_l[i]) for i in range(nte)])
    return (X, Y), (tX, tY)


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


def make_hyper_task(seed):
    """Return (hyper_module, loss_fn(hyper,x,y), metric_fn(hyper,x,y)).

    The trainable model is the *hypernetwork*; loss/metric rebuild the SNN from the
    weights it generates. compile_fit then optimises the hypernetwork's latent.
    """
    model = SpikingClassifier(rngs=nnx.Rngs(seed))
    snn_gd, _p, snn_rest = nnx.split(model, nnx.Param, ...)
    hyper = syn.RandomProjection(nnx.state(model, nnx.Param), EMB, rngs=nnx.Rngs(seed + 100))

    def _snn(hyper):
        return nnx.merge(snn_gd, hyper(), snn_rest)

    def loss_fn(hyper, x, y):
        return ce(_snn(hyper)(x), y)

    def metric_fn(hyper, x, y):
        return acc_fn(_snn(hyper)(x), y)[0]

    return hyper, loss_fn, metric_fn


def main():
    print(
        f"backend={jax.default_backend()}  SHD-solvers  T={SAMPLE_T} H={HIDDEN} emb={EMB} "
        f"pop={POP} grad_ep={GRAD_EPOCHS} cma_ep={CMA_EPOCHS} K={K}",
        flush=True,
    )
    train, test = load_shd()
    hyper0, _, _ = make_hyper_task(SEED)
    print(f"train batches={train[0].shape[0]}  hyper latent dim={syn.param_count(hyper0)}\n", flush=True)

    arms = {
        "surrogate": (optax.adam(3e-3), GRAD_EPOCHS),
        "es(K={})".format(K): (evolve.es(optax.adam(3e-3), num_samples=K, sigma=0.02), GRAD_EPOCHS),
        "cmaes": (evolve.cmaes(population_size=POP), CMA_EPOCHS),
        "primed_cmaes": (evolve.primed_cmaes(population_size=POP, prime_lr=0.1), CMA_EPOCHS),
    }
    results = {}
    for name, (solver, epochs) in arms.items():
        hyper, loss_fn, metric_fn = make_hyper_task(SEED)
        t0 = time.perf_counter()
        trained, hist = compile_fit(
            hyper, solver, loss_fn, train,
            epochs=epochs, eval_data=test, metric_fn=metric_fn,
            key=jax.random.PRNGKey(SEED),
        )
        jax.block_until_ready(hist["eval_metric"])
        acc = float(hist["eval_metric"][-1])
        dt = time.perf_counter() - t0
        results[name] = {"test_acc": acc, "gens_or_epochs": epochs}
        print(
            f"  {name:<14} test_acc={acc * 100:5.1f}%  "
            f"metric {float(hist['eval_metric'][0]) * 100:.1f}%->{acc * 100:.1f}%  ({dt:.0f}s)",
            flush=True,
        )

    out = {"config": {"T": SAMPLE_T, "emb": EMB, "pop": POP, "K": K,
                      "latent_dim": int(syn.param_count(hyper0))}, "arms": results}
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "shd_solvers_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote shd_solvers_results.json", flush=True)


if __name__ == "__main__":
    main()
