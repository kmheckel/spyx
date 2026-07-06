"""Does compressing the weight space (a hypernetwork) rescue ES / SGES on SHD?

The full-space result (methods_shd.py) is stark: at ~9.5k parameters, ES and
surrogate-steered SGES collapse to near-chance because ES variance scales with
dimension. This script tests the fix we theorised: move the search into a
*hypernetwork's* small latent space (synecdoche), where the effective dimension —
and hence the ES variance — is far smaller.

For each of ES, SGES and (as a capacity check) the surrogate gradient, we compare:
  * FULL   — optimise the SNN's ~9.5k weights directly.
  * HYPER  — optimise a RandomProjection hypernetwork's few-hundred params; the
             SNN weights are generated from them each step.

Same task, model, budget; only the search space differs. Needs synecdoche
installed in the env (``uv pip install /home/kade/Code/synecdoche``) and SHD cached
in ./data. Writes ``shd_hyper_results.json``.
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
from spyx.experimental.hybrid import es_gradient, make_sges_hybrid_train_step, sges_gradient

CHANNELS = 128
SAMPLE_T = int(os.environ.get("SAMPLE_T", "48"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
N_CLASSES = 20
BATCH = int(os.environ.get("BATCH", "128"))
N_TRAIN_BATCHES = int(os.environ.get("TRAIN_BATCHES", "10"))
N_TEST_BATCHES = int(os.environ.get("TEST_BATCHES", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "12"))
K = int(os.environ.get("K", "24"))
SIGMA = float(os.environ.get("SIGMA", "0.02"))
LR = float(os.environ.get("LR", "3e-3"))
EMB = int(os.environ.get("EMB", "128"))  # hypernetwork embedding dim per layer
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def _unpack(obs_bTpC):
    dense = jnp.unpackbits(jnp.asarray(obs_bTpC), axis=1)
    return dense[:, :SAMPLE_T, :].astype(jnp.float32)


def load_shd():
    dl = data.SHD_loader(batch_size=BATCH, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=0)
    tr_obs, tr_lab = dl.prestage("train")
    te_obs, te_lab = dl.prestage("test")
    train = [(_unpack(tr_obs[i]), jnp.asarray(tr_lab[i])) for i in range(N_TRAIN_BATCHES)]
    test = [(_unpack(te_obs[i]), jnp.asarray(te_lab[i])) for i in range(N_TEST_BATCHES)]
    return train, test


class SpikingClassifier(nnx.Module):
    def __init__(self, *, rngs):
        self.net = snn.Sequential(
            nnx.Linear(CHANNELS, HIDDEN, use_bias=False, rngs=rngs),
            snn.LIF((HIDDEN,), activation=axn.superspike(), rngs=rngs),
            nnx.Linear(HIDDEN, N_CLASSES, use_bias=False, rngs=rngs),
            snn.LI((N_CLASSES,), rngs=rngs),
        )

    def __call__(self, x_BTC):
        traces, _ = snn.run(self.net, x_BTC, batch_major=True)
        return traces


def loss_fn(model, xb, yb):
    return ce(model(xb), yb)


def evaluate(model, test):
    accs = jnp.stack([acc_fn(model(x), y)[0] for x, y in test])
    return float(jnp.mean(accs))


# ---------------------------------------------------------------- full-space arms
def _fresh(seed):
    return SpikingClassifier(rngs=nnx.Rngs(seed))


def train_full(train, test, seed, kind):
    model = _fresh(seed)
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    key = jax.random.PRNGKey(seed)
    if kind == "surrogate":

        @nnx.jit
        def step(m, o, xb, yb):
            loss, g = nnx.value_and_grad(lambda mm: loss_fn(mm, xb, yb))(m)
            o.update(m, g)

        for _ in range(EPOCHS):
            for x, y in train:
                step(model, opt, x, y)
    else:
        sges_step = make_sges_hybrid_train_step(loss_fn, loss_fn, num_samples=K, sigma=SIGMA)
        for _ in range(EPOCHS):
            for x, y in train:
                key, sub = jax.random.split(key)
                if kind == "es":
                    g = es_gradient(model, loss_fn, sub, batch=(x, y), num_samples=K, sigma=SIGMA)
                    opt.update(model, g)
                else:  # sges
                    sges_step(model, opt, sub, x, y)
    return evaluate(model, test)


# ---------------------------------------------------------------- hypernetwork arms
def _make_hyper(model, seed):
    target = nnx.state(model, nnx.Param)
    return syn.RandomProjection(target, EMB, rngs=nnx.Rngs(seed + 100))


def train_hyper(train, test, seed, kind):
    model = _fresh(seed)
    graphdef, _params, rest = nnx.split(model, nnx.Param, ...)
    hyper = _make_hyper(model, seed)

    def hloss(h, xb, yb):
        m = nnx.merge(graphdef, h(), rest)
        return loss_fn(m, xb, yb)

    opt = nnx.Optimizer(hyper, optax.adam(LR), wrt=nnx.Param)
    key = jax.random.PRNGKey(seed)
    for _ in range(EPOCHS):
        for x, y in train:
            key, sub = jax.random.split(key)
            if kind == "surrogate":
                _loss, g = nnx.value_and_grad(lambda hh: hloss(hh, x, y))(hyper)
                opt.update(hyper, g)
            elif kind == "es":
                g = es_gradient(hyper, hloss, sub, batch=(x, y), num_samples=K, sigma=SIGMA)
                opt.update(hyper, g)
            else:  # sges
                g = sges_gradient(hyper, hloss, hloss, sub, batch=(x, y), num_samples=K, sigma=SIGMA)
                opt.update(hyper, g)

    final = nnx.merge(graphdef, hyper(), rest)
    return evaluate(final, test), syn.param_count(hyper)


def main():
    print(
        f"SHD  C={CHANNELS} T={SAMPLE_T} H={HIDDEN}  train={N_TRAIN_BATCHES}x{BATCH} "
        f"epochs={EPOCHS} K={K} emb={EMB} seeds={SEEDS}",
        flush=True,
    )
    train, test = load_shd()
    full_dim = syn.param_count(_fresh(0))
    print(f"full-space params: {full_dim}", flush=True)

    results = {}
    for space, kind in [
        ("full", "surrogate"),
        ("full", "es"),
        ("full", "sges"),
        ("hyper", "surrogate"),
        ("hyper", "es"),
        ("hyper", "sges"),
    ]:
        t0 = time.perf_counter()
        accs, dims = [], []
        for seed in SEEDS:
            if space == "full":
                acc = train_full(train, test, seed, kind)
                dim = full_dim
            else:
                acc, dim = train_hyper(train, test, seed, kind)
            accs.append(acc)
            dims.append(dim)
        name = f"{space:<5} {kind}"
        results[name] = {
            "test_acc": float(np.mean(accs)),
            "search_dim": int(dims[0]),
        }
        print(
            f"  {name:<16} acc={np.mean(accs) * 100:5.1f}%  search_dim={dims[0]:<5} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    out = {
        "config": {
            "channels": CHANNELS, "sample_T": SAMPLE_T, "hidden": HIDDEN,
            "n_classes": N_CLASSES, "batch": BATCH, "train_batches": len(train),
            "epochs": EPOCHS, "K": K, "emb": EMB, "seeds": SEEDS, "full_dim": full_dim,
        },
        "arms": results,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "shd_hyper_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote shd_hyper_results.json", flush=True)


if __name__ == "__main__":
    main()
