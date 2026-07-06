"""Surrogate-steered SGES vs surrogate-gradient vs ES on real SHD.

Puts the 0th / 1st / 0+1 methods on the Spiking Heidelberg Digits dataset (20
spoken digits) instead of synthetic spikes, under a matched step budget:

* **1st  surrogate** — surrogate-gradient descent (the workhorse).
* **0th  ES**        — pure antithetic ES on the hard-spike loss.
* **0+1  SGES(λ=1)** — surrogate-steered Self-Guided ES (descend on the estimate).
* **0+1  SGES(λ=.5)**— blend: surrogate bulk + variance-reduced SGES correction.

Honest expectation: ES does not scale to ~10 k parameters (variance ∝ dimension),
so pure ES should struggle; the question is how much the surrogate-steered
variance reduction closes the gap to the surrogate. Loads SHD via the torch-free
``SHD_loader.prestage()`` bulk path (dataset must already be in ``./data``). Uses a
reduced budget (subset of train, sample_T, hidden) so the ES arms are tractable on
CPU — this is a *comparison under matched budget*, not a leaderboard number.
Writes ``shd_results.json``.

    uv run python research/new/hybrid_evo_surrogate/methods_shd.py
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import spyx.axn as axn
import spyx.data as data
import spyx.fn as fn
import spyx.nn as snn
from spyx.experimental.hybrid import (
    es_gradient,
    make_sges_hybrid_train_step,
)

CHANNELS = 128
SAMPLE_T = int(os.environ.get("SAMPLE_T", "64"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
N_CLASSES = 20
BATCH = int(os.environ.get("BATCH", "128"))
N_TRAIN_BATCHES = int(os.environ.get("TRAIN_BATCHES", "12"))  # subset for tractability
N_TEST_BATCHES = int(os.environ.get("TEST_BATCHES", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
K = int(os.environ.get("K", "24"))
SIGMA = float(os.environ.get("SIGMA", "0.02"))
LR = float(os.environ.get("LR", "3e-3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def _unpack(obs_bTpC):
    # (B, T_packed, C) uint8 -> (B, T, C) float32, time-major-friendly batch-major.
    dense = jnp.unpackbits(jnp.asarray(obs_bTpC), axis=1)
    return dense[:, :SAMPLE_T, :].astype(jnp.float32)


def load_shd():
    dl = data.SHD_loader(
        batch_size=BATCH, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=0
    )
    tr_obs, tr_lab = dl.prestage("train")
    te_obs, te_lab = dl.prestage("test")
    train = [
        (_unpack(tr_obs[i]), jnp.asarray(tr_lab[i]))
        for i in range(min(N_TRAIN_BATCHES, tr_obs.shape[0]))
    ]
    test = [
        (_unpack(te_obs[i]), jnp.asarray(te_lab[i]))
        for i in range(min(N_TEST_BATCHES, te_obs.shape[0]))
    ]
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
        traces, _ = snn.run(self.net, x_BTC, batch_major=True)  # (B, T, classes)
        return traces


def loss_fn(model, xb, yb):
    return ce(model(xb), yb)


def evaluate(model, test):
    accs = jnp.stack([acc_fn(model(x), y)[0] for x, y in test])
    losses = jnp.stack([loss_fn(model, x, y) for x, y in test])
    return float(jnp.mean(losses)), float(jnp.mean(accs))


def _fresh(seed):
    return SpikingClassifier(rngs=nnx.Rngs(seed))


def train_surrogate(train, test, seed):
    model = _fresh(seed)
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    @nnx.jit
    def step(m, o, xb, yb):
        loss, g = nnx.value_and_grad(lambda mm: loss_fn(mm, xb, yb))(m)
        o.update(m, g)
        return loss

    for _ in range(EPOCHS):
        for x, y in train:
            step(model, opt, x, y)
    return evaluate(model, test)


def _train_step_loop(train, test, seed, step):
    model = _fresh(seed)
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    key = jax.random.PRNGKey(seed)
    for _ in range(EPOCHS):
        for x, y in train:
            key, sub = jax.random.split(key)
            step(model, opt, sub, x, y)
    return evaluate(model, test)


def train_es(train, test, seed):
    def step(model, opt, key, x, y):
        g = es_gradient(model, loss_fn, key, batch=(x, y), num_samples=K, sigma=SIGMA)
        opt.update(model, g)

    return _train_step_loop(train, test, seed, step)


def train_sges(train, test, seed, lam):
    step = make_sges_hybrid_train_step(
        loss_fn, loss_fn, num_samples=K, sigma=SIGMA, lam=lam
    )
    return _train_step_loop(train, test, seed, step)


def main():
    print(
        f"backend={jax.default_backend()} SHD  C={CHANNELS} T={SAMPLE_T} H={HIDDEN} "
        f"train_batches={N_TRAIN_BATCHES}x{BATCH} epochs={EPOCHS} K={K} seeds={SEEDS}",
        flush=True,
    )
    t0 = time.perf_counter()
    train, test = load_shd()
    n_params = sum(
        int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(nnx.state(_fresh(0), nnx.Param))
    )
    print(
        f"loaded SHD: {len(train)} train / {len(test)} test batches, "
        f"~{n_params} params  ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    arms = {
        "1st  surrogate": train_surrogate,
        "0th  ES": train_es,
        "0+1  SGES(λ=1)": lambda tr, te, s: train_sges(tr, te, s, 1.0),
        "0+1  SGES(λ=.5)": lambda tr, te, s: train_sges(tr, te, s, 0.5),
    }
    results = {}
    for name, fn_train in arms.items():
        ta0 = time.perf_counter()
        losses, accs = [], []
        for seed in SEEDS:
            tl, ta = fn_train(train, test, seed)
            losses.append(tl)
            accs.append(ta)
        results[name] = {
            "test_loss": float(np.mean(losses)),
            "test_acc": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
        }
        print(
            f"  {name:<18} test_acc={np.mean(accs) * 100:.1f}%±{np.std(accs) * 100:.1f}"
            f"  test_loss={np.mean(losses):.4f}  ({time.perf_counter() - ta0:.0f}s)",
            flush=True,
        )

    out = {
        "config": {
            "dataset": "SHD", "channels": CHANNELS, "sample_T": SAMPLE_T,
            "hidden": HIDDEN, "n_classes": N_CLASSES, "batch": BATCH,
            "train_batches": len(train), "test_batches": len(test),
            "epochs": EPOCHS, "K": K, "sigma": SIGMA, "lr": LR, "seeds": SEEDS,
            "n_params": n_params,
        },
        "arms": results,
        "wall_s": time.perf_counter() - t0,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "shd_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote shd_results.json  (total {time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
