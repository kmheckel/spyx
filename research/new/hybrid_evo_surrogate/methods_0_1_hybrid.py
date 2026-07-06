"""0th vs 1st vs 0+1: a clean method comparison for SNN optimisation.

Compares, on one synthetic spiking-classification task under a matched step
budget, the four points of the training-method taxonomy this repo showcases:

* **1st-order** — surrogate gradient (``nnx.value_and_grad`` through spyx.axn).
* **0th-order** — pure antithetic ES on the true (hard-spike) loss
  (``spyx.experimental.hybrid.es_gradient``), no surrogate.
* **0+1 (orthogonal)** — surrogate bulk + ES correction in the orthogonal
  complement, self-normalised (``make_hybrid_train_step(normalize=True)``).
* **0+1 (SGES)** — surrogate-steered Self-Guided ES: the surrogate steers the ES
  sampling for variance reduction (``make_sges_hybrid_train_step``).

Also measures the *variance* of the ES gradient estimate (isotropic vs SGES) at a
trained checkpoint — the quantity SGES is designed to reduce. Writes
``methods_results.json``. Run::

    SPYX_SMOKE=1 uv run python research/new/hybrid_evo_surrogate/methods_0_1_hybrid.py
    uv run python research/new/hybrid_evo_surrogate/methods_0_1_hybrid.py
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
from spyx.experimental.hybrid import (
    _es_flat,
    _sges_flat,
    es_gradient,
    make_hybrid_train_step,
    make_sges_hybrid_train_step,
)

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))
if SMOKE:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 16, 16, 3, 10
    N_TRAIN, BATCH, EPOCHS, K = 32, 16, 8, 12
    SEEDS = [0, 1]
else:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 24, 32, 4, 16
    N_TRAIN, BATCH, EPOCHS, K = 128, 32, 15, 16
    SEEDS = [0, 1, 2]

SIGMA, LR = 0.02, 5e-3
ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def synthetic_data(seed):
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
    xte, yte = make(N_TRAIN)
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

    xs, ys = train
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys, strict=True):
            step(model, opt, x, y)
    return evaluate(model, test)


def _train_with_step(train, test, seed, step):
    model = _fresh(seed)
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    xs, ys = train
    key = jax.random.PRNGKey(seed)
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys, strict=True):
            key, sub = jax.random.split(key)
            step(model, opt, sub, x, y)
    return evaluate(model, test)


def train_es(train, test, seed):
    def step(model, opt, key, x, y):
        g = es_gradient(model, loss_fn, key, batch=(x, y), num_samples=K, sigma=SIGMA)
        opt.update(model, g)
        return loss_fn(model, x, y)

    return _train_with_step(train, test, seed, step)


def train_hybrid_orth(train, test, seed):
    step = make_hybrid_train_step(
        loss_fn, loss_fn, num_samples=K, sigma=SIGMA, lam=0.3, normalize=True
    )
    return _train_with_step(train, test, seed, step)


def train_sges(train, test, seed, lam=1.0):
    step = make_sges_hybrid_train_step(
        loss_fn, loss_fn, num_samples=K, sigma=SIGMA, lam=lam
    )
    return _train_with_step(train, test, seed, step)


def es_estimate_variance(train, seed, n_keys=40):
    """Variance of the ES gradient estimate (isotropic vs SGES) at init — the
    quantity SGES reduces. Returns (iso_var, sges_var, reduction_factor)."""
    model = _fresh(seed)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    theta, unravel = jax.flatten_util.ravel_pytree(params)
    x, y = train[0][0], train[1][0]

    def true_flat(flat):
        return loss_fn(nnx.merge(graphdef, unravel(flat), rest), x, y)

    g_s = jax.grad(true_flat)(theta)  # surrogate==true fwd here; used only as guide
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_keys)
    iso = jax.vmap(lambda k: _es_flat(theta, true_flat, k, num_samples=K, sigma=SIGMA))(
        keys
    )
    sg = jax.vmap(
        lambda k: _sges_flat(theta, true_flat, g_s, k, num_samples=K, sigma=SIGMA, eps=1e-8)[0]
    )(keys)
    iv, sv = float(iso.var(0).sum()), float(sg.var(0).sum())
    return iv, sv, iv / (sv + 1e-12)


def main():
    print(
        f"backend={jax.default_backend()} SMOKE={SMOKE} "
        f"C={CHANNELS} H={HIDDEN} classes={N_CLASSES} T={SAMPLE_T} "
        f"epochs={EPOCHS} K={K} seeds={SEEDS}",
        flush=True,
    )
    arms = {
        "1st  surrogate": train_surrogate,
        "0th  ES": train_es,
        "0+1  hybrid-orth": train_hybrid_orth,
        "0+1  SGES(λ=1)": train_sges,
        "0+1  SGES(λ=.5)": lambda tr, te, s: train_sges(tr, te, s, lam=0.5),
    }
    t0 = time.perf_counter()
    results = {}
    for name, fn_train in arms.items():
        losses, accs = [], []
        for seed in SEEDS:
            train, test = synthetic_data(seed)
            tl, ta = fn_train(train, test, seed)
            losses.append(tl)
            accs.append(ta)
        results[name] = {
            "true_loss": float(np.mean(losses)),
            "true_loss_std": float(np.std(losses)),
            "acc": float(np.mean(accs)),
        }
        print(
            f"  {name:<18} true_loss={np.mean(losses):.4f}±{np.std(losses):.4f}  "
            f"acc={np.mean(accs) * 100:.1f}%",
            flush=True,
        )

    # Variance of the ES estimate: isotropic vs SGES (the SGES payoff).
    train, _ = synthetic_data(SEEDS[0])
    iv, sv, factor = es_estimate_variance(train, SEEDS[0])
    print(
        f"\nES gradient-estimate variance @ init (K={K}): "
        f"isotropic={iv:.4f}  SGES={sv:.4f}  reduction={factor:.1f}x",
        flush=True,
    )

    out = {
        "config": {
            "smoke": SMOKE, "channels": CHANNELS, "hidden": HIDDEN,
            "n_classes": N_CLASSES, "sample_T": SAMPLE_T, "epochs": EPOCHS,
            "K": K, "sigma": SIGMA, "lr": LR, "seeds": SEEDS,
        },
        "arms": results,
        "es_estimate_variance": {
            "isotropic": iv, "sges": sv, "reduction_factor": factor
        },
        "wall_s": time.perf_counter() - t0,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "methods_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote methods_results.json", flush=True)


if __name__ == "__main__":
    main()
