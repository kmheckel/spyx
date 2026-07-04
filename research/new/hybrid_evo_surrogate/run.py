"""Three-arm study: surrogate vs. pure-ES vs. hybrid on a synthetic spiking task.

See README.md for the hypothesis and honest expected outcome. In short: does the
orthogonalised evolutionary correction in ``spyx.experimental.hybrid`` reach a
lower *true* (hard-spike forward) loss than pure surrogate descent or pure ES,
under a matched step budget?

The three arms share init, optimizer, and step count; they differ only in how
the gradient is built:

* **surrogate** — ``∇ loss_surrogate`` via the ``spyx.axn`` superspike backward.
* **es**        — pure antithetic NES on the true loss (no surrogate).
* **hybrid**    — ``g_s + λ·g_orth`` (surrogate bulk + orthogonal ES correction).

``loss_true`` and ``loss_surrogate`` are the *same* cross-entropy: every
``spyx.axn`` activation forwards through the identical Heaviside step, so the
"true" loss is the hard-spike forward loss and the arms differ only in the
backward pass (surrogate differentiates it; ES only evaluates it).

Run::

    SPYX_SMOKE=1 uv run python research/new/hybrid_evo_surrogate/run.py   # seconds
    uv run python research/new/hybrid_evo_surrogate/run.py                # fuller
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
    es_gradient,
    hybrid_diagnostics,
    make_hybrid_train_step,
)

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))

if SMOKE:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 12, 12, 3, 8
    N_TRAIN, N_TEST, BATCH = 24, 12, 12
    EPOCHS = int(os.environ.get("EPOCHS", "8"))
    K = int(os.environ.get("K", "16"))  # antithetic pairs
else:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 40, 48, 5, 24
    N_TRAIN, N_TEST, BATCH = 256, 128, 64
    EPOCHS = int(os.environ.get("EPOCHS", "30"))
    K = int(os.environ.get("K", "64"))

SIGMA = float(os.environ.get("SIGMA", "0.02"))
LAM = float(os.environ.get("LAM", "0.5"))  # raw-mode weight on g_orth
LAM_NORM = float(os.environ.get("LAM_NORM", "0.3"))  # normalize-mode fraction
LR = float(os.environ.get("LR", "5e-3"))
SEED = int(os.environ.get("SEED", "0"))


# --------------------------------------------------------------------------- data
def synthetic_data():
    """Class-conditional band-rate spikes: each class fires an extra channel band.

    Returns time-major ``(T, B, C)`` batches so they feed straight into
    ``spyx.nn.run``. Enough signal that a spiking classifier can separate classes
    on CPU in seconds — this exercises the arms, it is not a benchmark.
    """
    rng = np.random.default_rng(SEED)
    band = max(1, CHANNELS // N_CLASSES)

    def make(n):
        labels = rng.integers(0, N_CLASSES, size=n)
        x = (rng.random((n, SAMPLE_T, CHANNELS)) < 0.05).astype(np.float32)
        for i in range(n):
            lo = labels[i] * band
            x[i, :, lo : lo + band] += (
                rng.random((SAMPLE_T, band)) < 0.35
            ).astype(np.float32)
        return np.clip(x, 0.0, 1.0), labels.astype(np.int32)

    xtr, ytr = make(N_TRAIN)
    xte, yte = make(N_TEST)

    def batched(x, y):
        obs, lab = [], []
        for s in range(0, x.shape[0] - BATCH + 1, BATCH):
            # (B, T, C) -> (T, B, C) time-major for spyx.nn.run
            obs.append(jnp.transpose(jnp.asarray(x[s : s + BATCH]), (1, 0, 2)))
            lab.append(jnp.asarray(y[s : s + BATCH]))
        return obs, lab

    return batched(xtr, ytr), batched(xte, yte)


# ------------------------------------------------------------------------- model
class SpikingClassifier(nnx.Module):
    """Linear -> LIF -> Linear -> LI over time; returns a (B, T, classes) trace."""

    def __init__(self, activation, *, rngs):
        self.net = snn.Sequential(
            nnx.Linear(CHANNELS, HIDDEN, rngs=rngs),
            snn.LIF((HIDDEN,), activation=activation, rngs=rngs),
            nnx.Linear(HIDDEN, N_CLASSES, rngs=rngs),
            snn.LI((N_CLASSES,), rngs=rngs),
        )

    def __call__(self, x_TBC):
        traces, _ = snn.run(self.net, x_TBC)  # (T, B, classes)
        return jnp.transpose(traces, (1, 0, 2))  # (B, T, classes)


ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def loss_fn(model, xb, yb):
    return ce(model(xb), yb)


def evaluate(model, test):
    xs, ys = test
    losses = jnp.stack([loss_fn(model, x, y) for x, y in zip(xs, ys)])
    accs = jnp.stack([acc_fn(model(x), y)[0] for x, y in zip(xs, ys)])
    return float(jnp.mean(losses)), float(jnp.mean(accs))


def fresh_model():
    # Identical init across arms (same seed).
    return SpikingClassifier(axn.superspike(), rngs=nnx.Rngs(SEED))


# ---------------------------------------------------------------------- train arms
def train_surrogate(train, test):
    """Pure surrogate-gradient descent (the cheap biased baseline)."""
    model = fresh_model()
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    @nnx.jit
    def step(m, o, xb, yb):
        loss, g = nnx.value_and_grad(lambda mm: loss_fn(mm, xb, yb))(m)
        o.update(m, g)
        return loss

    xs, ys = train
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys):
            step(model, opt, x, y)
    dt = time.perf_counter() - t0
    tl, ta = evaluate(model, test)
    return {"true_loss": tl, "acc": ta, "train_s": dt}


def train_es(train, test):
    """Pure antithetic-NES on the true (hard-spike forward) loss — no surrogate."""
    model = fresh_model()
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    xs, ys = train
    key = jax.random.PRNGKey(SEED)
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys):
            key, sub = jax.random.split(key)
            grads = es_gradient(
                model, loss_fn, sub, batch=(x, y), num_samples=K, sigma=SIGMA
            )
            opt.update(model, grads)
    dt = time.perf_counter() - t0
    tl, ta = evaluate(model, test)
    return {"true_loss": tl, "acc": ta, "train_s": dt}


def train_hybrid(train, test, *, normalize, lam):
    """Hybrid g_s + lam_eff * g_orth; logs the mean correction diagnostics.

    ``normalize=False`` uses ``lam`` as a raw scale on ``g_orth``;
    ``normalize=True`` treats ``lam`` as a fraction of the surrogate step so the
    applied correction has norm ``lam·‖g_s‖`` (see ``spyx.experimental.hybrid``).
    """
    model = fresh_model()
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    step = make_hybrid_train_step(
        loss_fn, loss_fn, num_samples=K, sigma=SIGMA, lam=lam, normalize=normalize
    )
    xs, ys = train
    key = jax.random.PRNGKey(SEED)
    cosines, orth_norms, s_norms, fracs = [], [], [], []
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        for x, y in zip(xs, ys):
            key, sub = jax.random.split(key)
            step(model, opt, sub, x, y)
            if ep == 0:  # sample the correction geometry on the first epoch
                key, dk = jax.random.split(key)
                d = hybrid_diagnostics(
                    model, loss_fn, loss_fn, dk, batch=(x, y),
                    num_samples=K, sigma=SIGMA, lam=lam, normalize=normalize,
                )
                cosines.append(float(d["cosine"]))
                orth_norms.append(float(d["g_orth_norm"]))
                s_norms.append(float(d["g_s_norm"]))
                fracs.append(float(d["correction_fraction"]))
    dt = time.perf_counter() - t0
    tl, ta = evaluate(model, test)
    return {
        "true_loss": tl,
        "acc": ta,
        "train_s": dt,
        "normalize": normalize,
        "lam": lam,
        "diag": {
            "mean_cosine_es_vs_surrogate": float(np.mean(cosines)) if cosines else None,
            "mean_g_orth_norm": float(np.mean(orth_norms)) if orth_norms else None,
            "mean_g_s_norm": float(np.mean(s_norms)) if s_norms else None,
            "mean_correction_fraction": float(np.mean(fracs)) if fracs else None,
        },
    }


# ---------------------------------------------------------------------------- main
def main():
    print(
        f"backend={jax.default_backend()}  SMOKE={SMOKE}  "
        f"C={CHANNELS} H={HIDDEN} classes={N_CLASSES} T={SAMPLE_T}  "
        f"epochs={EPOCHS} K={K} sigma={SIGMA} lam={LAM} lam_norm={LAM_NORM}",
        flush=True,
    )
    train, test = synthetic_data()
    print(f"train batches={len(train[0])}  test batches={len(test[0])}\n", flush=True)

    print("== 1. surrogate ==", flush=True)
    r_sur = train_surrogate(train, test)
    print(
        f"  true_loss={r_sur['true_loss']:.4f}  acc={r_sur['acc'] * 100:.2f}%  "
        f"({r_sur['train_s']:.2f}s)",
        flush=True,
    )

    print("== 2. pure ES ==", flush=True)
    r_es = train_es(train, test)
    print(
        f"  true_loss={r_es['true_loss']:.4f}  acc={r_es['acc'] * 100:.2f}%  "
        f"({r_es['train_s']:.2f}s)",
        flush=True,
    )

    print("== 3. hybrid (raw lam) ==", flush=True)
    r_hy = train_hybrid(train, test, normalize=False, lam=LAM)
    d = r_hy["diag"]
    print(
        f"  true_loss={r_hy['true_loss']:.4f}  acc={r_hy['acc'] * 100:.2f}%  "
        f"({r_hy['train_s']:.2f}s)  "
        f"mean ||g_orth||/||g_s||="
        f"{d['mean_g_orth_norm'] / d['mean_g_s_norm']:.2f}  "
        f"correction_frac={d['mean_correction_fraction']:.2f}",
        flush=True,
    )

    print("== 4. hybrid (normalized lam) ==", flush=True)
    r_hyn = train_hybrid(train, test, normalize=True, lam=LAM_NORM)
    dn = r_hyn["diag"]
    print(
        f"  true_loss={r_hyn['true_loss']:.4f}  acc={r_hyn['acc'] * 100:.2f}%  "
        f"({r_hyn['train_s']:.2f}s)  "
        f"correction_frac={dn['mean_correction_fraction']:.2f} (== lam_norm)",
        flush=True,
    )

    print("\n== four-arm comparison (final TRUE loss / accuracy) ==", flush=True)
    rows = [
        ("surrogate", r_sur),
        ("es", r_es),
        ("hybrid-raw", r_hy),
        ("hybrid-norm", r_hyn),
    ]
    print(f"  {'arm':<12} {'true_loss':>10} {'accuracy':>10} {'train_s':>9}")
    for name, r in rows:
        print(
            f"  {name:<12} {r['true_loss']:>10.4f} "
            f"{r['acc'] * 100:>9.2f}% {r['train_s']:>9.2f}"
        )

    best = min(rows, key=lambda kv: kv[1]["true_loss"])[0]
    delta = r_hyn["true_loss"] - r_sur["true_loss"]
    print(f"\n  lowest true loss: {best}", flush=True)
    print(
        f"  hybrid-norm − surrogate true_loss = {delta:+.4f} "
        f"({'hybrid-norm wins' if delta < 0 else 'surrogate wins'})",
        flush=True,
    )

    out = {
        "config": {
            "smoke": SMOKE,
            "channels": CHANNELS,
            "hidden": HIDDEN,
            "n_classes": N_CLASSES,
            "sample_T": SAMPLE_T,
            "epochs": EPOCHS,
            "num_samples_K": K,
            "sigma": SIGMA,
            "lam": LAM,
            "lam_norm": LAM_NORM,
            "lr": LR,
            "seed": SEED,
        },
        "results": {
            "surrogate": r_sur,
            "es": r_es,
            "hybrid_raw": r_hy,
            "hybrid_norm": r_hyn,
        },
        "lowest_true_loss_arm": best,
        "hybrid_norm_minus_surrogate": delta,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "study_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote study_results.json", flush=True)


if __name__ == "__main__":
    main()
