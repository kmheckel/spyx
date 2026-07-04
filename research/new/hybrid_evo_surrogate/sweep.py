"""Sweep: does the self-normalising orthogonal-ES correction beat pure surrogate?

``run.py`` is a single-config three/four-arm demo. This sweep answers the harder
question the PR #49 study left open: across seeds, ES sample counts ``K`` and
correction fractions ``λ``, does ``hybrid-norm`` (surrogate + self-normalised
orthogonal-ES correction) reach a *lower true loss* than pure surrogate descent,
and is the effect robust to seed noise or just a fluke?

Design (held fixed across arms within a config): same init, optimizer, data, step
budget; arms differ only in how the gradient is built. We compare three arms per
config — ``surrogate``, ``hybrid-raw`` (fixed raw ``λ=0.5``, the PR #49 setting),
and ``hybrid-norm`` (self-normalised ``λ`` = correction fraction). Pure-ES is
omitted; PR #49 already established it trails.

Regime is deliberately harder/noisier than ``run.py``'s SMOKE path (more surrogate
bias for ES to correct) but still CPU-cheap. Runs multiple seeds per cell and
reports the mean Δ(true_loss) = hybrid-norm − surrogate with its spread, so a win
has to survive seed noise. Writes ``sweep_results.json``.

Run::

    uv run python research/new/hybrid_evo_surrogate/sweep.py            # full grid
    SEEDS=0,1 KS=48 LAMS=0.3 uv run python .../sweep.py                 # quick
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
import spyx.fn as fn
import spyx.nn as snn
from spyx.experimental.hybrid import make_hybrid_train_step

# Moderately hard regime: enough surrogate bias to leave room for a correction.
CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 24, 32, 4, 16
N_TRAIN, N_TEST, BATCH = 128, 128, 32
EPOCHS = int(os.environ.get("EPOCHS", "20"))
SIGMA = float(os.environ.get("SIGMA", "0.02"))
LR = float(os.environ.get("LR", "5e-3"))
LAM_RAW = float(os.environ.get("LAM_RAW", "0.5"))

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
KS = [int(k) for k in os.environ.get("KS", "32,96").split(",")]
LAMS = [float(x) for x in os.environ.get("LAMS", "0.15,0.3,0.5").split(",")]

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def synthetic_data(seed):
    """Class-conditional band-rate spikes, time-major (T, B, C) for spyx.nn.run."""
    rng = np.random.default_rng(seed)
    band = max(1, CHANNELS // N_CLASSES)

    def make(n):
        labels = rng.integers(0, N_CLASSES, size=n)
        x = (rng.random((n, SAMPLE_T, CHANNELS)) < 0.05).astype(np.float32)
        for i in range(n):
            lo = labels[i] * band
            x[i, :, lo : lo + band] += (rng.random((SAMPLE_T, band)) < 0.30).astype(
                np.float32
            )
        return np.clip(x, 0.0, 1.0), labels.astype(np.int32)

    def batched(x, y):
        obs, lab = [], []
        for s in range(0, x.shape[0] - BATCH + 1, BATCH):
            obs.append(jnp.transpose(jnp.asarray(x[s : s + BATCH]), (1, 0, 2)))
            lab.append(jnp.asarray(y[s : s + BATCH]))
        return obs, lab

    xtr, ytr = make(N_TRAIN)
    xte, yte = make(N_TEST)
    return batched(xtr, ytr), batched(xte, yte)


class SpikingClassifier(nnx.Module):
    def __init__(self, *, rngs):
        self.net = snn.Sequential(
            nnx.Linear(CHANNELS, HIDDEN, rngs=rngs),
            snn.LIF((HIDDEN,), activation=axn.superspike(), rngs=rngs),
            nnx.Linear(HIDDEN, N_CLASSES, rngs=rngs),
            snn.LI((N_CLASSES,), rngs=rngs),
        )

    def __call__(self, x_TBC):
        traces, _ = snn.run(self.net, x_TBC)
        return jnp.transpose(traces, (1, 0, 2))


def loss_fn(model, xb, yb):
    return ce(model(xb), yb)


def evaluate(model, test):
    xs, ys = test
    losses = jnp.stack([loss_fn(model, x, y) for x, y in zip(xs, ys, strict=True)])
    accs = jnp.stack([acc_fn(model(x), y)[0] for x, y in zip(xs, ys, strict=True)])
    return float(jnp.mean(losses)), float(jnp.mean(accs))


def train_surrogate(train, test, seed):
    model = SpikingClassifier(rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    @nnx.jit
    def step(m, o, xb, yb):
        loss, g = nnx.value_and_grad(lambda mm: loss_fn(mm, xb, yb))(m)
        o.update(m, g)
        return loss

    xs, ys = train
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys, strict=True):
            step(model, opt, x, y)
    return evaluate(model, test)


def train_hybrid(train, test, seed, *, k, lam, normalize):
    model = SpikingClassifier(rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    step = make_hybrid_train_step(
        loss_fn, loss_fn, num_samples=k, sigma=SIGMA, lam=lam, normalize=normalize
    )
    xs, ys = train
    key = jax.random.PRNGKey(seed)
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys, strict=True):
            key, sub = jax.random.split(key)
            step(model, opt, sub, x, y)
    return evaluate(model, test)


def main():
    print(
        f"backend={jax.default_backend()}  regime C={CHANNELS} H={HIDDEN} "
        f"classes={N_CLASSES} T={SAMPLE_T} epochs={EPOCHS}  "
        f"seeds={SEEDS} Ks={KS} lams={LAMS} lam_raw={LAM_RAW}",
        flush=True,
    )
    t0 = time.perf_counter()

    # Surrogate + raw-hybrid depend only on seed (and K for raw); cache per seed.
    sur = {}
    raw = {}  # (seed, k) -> (loss, acc)
    for seed in SEEDS:
        train, test = synthetic_data(seed)
        sur[seed] = train_surrogate(train, test, seed)
        print(f"  seed={seed} surrogate  loss={sur[seed][0]:.4f}", flush=True)

    cells = []  # one per (k, lam): aggregated over seeds
    for k in KS:
        # raw hybrid at fixed LAM_RAW for this K, per seed (the PR#49 failure mode).
        for seed in SEEDS:
            train, test = synthetic_data(seed)
            raw[(seed, k)] = train_hybrid(
                train, test, seed, k=k, lam=LAM_RAW, normalize=False
            )
        for lam in LAMS:
            deltas, norm_losses, norm_accs = [], [], []
            for seed in SEEDS:
                train, test = synthetic_data(seed)
                nl, na = train_hybrid(
                    train, test, seed, k=k, lam=lam, normalize=True
                )
                norm_losses.append(nl)
                norm_accs.append(na)
                deltas.append(nl - sur[seed][0])
            cell = {
                "k": k,
                "lam_norm": lam,
                "mean_surrogate_loss": float(np.mean([sur[s][0] for s in SEEDS])),
                "mean_hybrid_norm_loss": float(np.mean(norm_losses)),
                "mean_hybrid_raw_loss": float(
                    np.mean([raw[(s, k)][0] for s in SEEDS])
                ),
                "mean_delta_norm_minus_sur": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "wins": int(sum(1 for d in deltas if d < 0)),
                "n_seeds": len(SEEDS),
                "mean_hybrid_norm_acc": float(np.mean(norm_accs)),
            }
            cells.append(cell)
            print(
                f"  K={k:>3} lam={lam:<4}  "
                f"sur={cell['mean_surrogate_loss']:.4f}  "
                f"hyb-norm={cell['mean_hybrid_norm_loss']:.4f}  "
                f"hyb-raw={cell['mean_hybrid_raw_loss']:.4f}  "
                f"Δ={cell['mean_delta_norm_minus_sur']:+.4f}"
                f"±{cell['std_delta']:.4f}  wins={cell['wins']}/{cell['n_seeds']}",
                flush=True,
            )

    best = min(cells, key=lambda c: c["mean_delta_norm_minus_sur"])
    dt = time.perf_counter() - t0
    print(
        f"\nbest cell: K={best['k']} lam={best['lam_norm']}  "
        f"Δ={best['mean_delta_norm_minus_sur']:+.4f}±{best['std_delta']:.4f}  "
        f"wins={best['wins']}/{best['n_seeds']}   ({dt:.1f}s total)",
        flush=True,
    )
    verdict = (
        "hybrid-norm beats surrogate"
        if best["mean_delta_norm_minus_sur"] < 0 and best["wins"] > best["n_seeds"] // 2
        else "surrogate still wins / not robust"
    )
    print(f"verdict: {verdict}", flush=True)

    out = {
        "regime": {
            "channels": CHANNELS, "hidden": HIDDEN, "n_classes": N_CLASSES,
            "sample_T": SAMPLE_T, "epochs": EPOCHS, "sigma": SIGMA, "lr": LR,
            "lam_raw": LAM_RAW, "seeds": SEEDS, "Ks": KS, "lams": LAMS,
        },
        "surrogate_per_seed": {str(s): sur[s][0] for s in SEEDS},
        "cells": cells,
        "best_cell": best,
        "verdict": verdict,
        "wall_s": dt,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sweep_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote sweep_results.json", flush=True)


if __name__ == "__main__":
    main()
