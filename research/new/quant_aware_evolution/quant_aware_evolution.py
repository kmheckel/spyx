"""Gradient-free evolution vs. STE-QAT at extreme low precision.

Straight-through estimation (STE) is the workhorse of quantization-aware
training: the forward pass uses hard-quantized weights, but the backward pass
pretends the (zero-a.e.) quantizer gradient is the identity. That substitution is
*biased*, and the bias grows as the quantization grid coarsens - at nvfp4 or
ternary the fake-quant forward and the surrogate backward disagree sharply.

Evolution strategies (here **CR-FM-NES**, Nomura & Ono, CEC 2022) never
differentiate the quantizer: fitness is the *true* quantized-forward loss, so the
search is **unbiased on the exact objective STE only approximates** - at the cost
of high variance and many forward evaluations.

**Claim under test.** Because ES has no STE bias, at precisions where STE bias is
worst it should reach a lower TRUE quantized-forward loss than STE-QAT under a
matched precision, and the ES advantage should *grow* as precision drops
(ternary > nvfp4). We report exactly what the run shows.

Arms (matched init, matched synthetic task, true-quant forward shared by all):

* **fp32** - surrogate SGD, no quantization (reference ceiling).
* **STE-QAT @ prec** - surrogate SGD through straight-through fake-quant weights.
* **ES @ prec** - CR-FM-NES over fp32 weights; each candidate is quantized in the
  TRUE forward before its fitness (loss) is measured. No STE anywhere.

`prec` ranges over nvfp4 (qwix, tile 16) and ternary (BitNet b1.58, {-1,0,1}).
The headline per precision is the STE-bias gap = STE-QAT true loss - ES true loss.

Every arm's *reported* number is the identical true quantized-forward evaluation
(`Q(w) = dequant(quant(w))`, no STE), so the comparison isolates the gradient
path. Writes ``quant_aware_evolution_results.json``. Run::

    SPYX_SMOKE=1 uv run python research/new/quant_aware_evolution/quant_aware_evolution.py
    uv run python research/new/quant_aware_evolution/quant_aware_evolution.py
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from flax import nnx
from jax.flatten_util import ravel_pytree

import spyx.axn as axn
import spyx.fn as fn
import spyx.nn as snn

try:
    from evosax.algorithms import CR_FM_NES
except ImportError:  # pragma: no cover
    from evosax.algorithms.distribution_based import CR_FM_NES

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))
if SMOKE:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 16, 16, 3, 10
    N_TRAIN, BATCH, EPOCHS = 32, 16, 8
    POP, GENERATIONS = 16, 20
    SEEDS = [0, 1]
else:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 32, 64, 4, 16
    N_TRAIN, BATCH, EPOCHS = 128, 32, 15
    POP, GENERATIONS = 64, 150
    SEEDS = [0, 1, 2]

LR, STD_INIT = 5e-3, 0.1
# nvfp4 tiles the contraction axis in blocks of 16, so in-features (CHANNELS,
# HIDDEN) must be multiples of 16. ternary is per-tensor and shape-agnostic.
PRECISIONS = ["nvfp4", "ternary"]
NVFP4_TILE = 16

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


# --------------------------------------------------------------------------- #
# Quantization: one true-forward Q(w), plus an STE wrapper sharing that value.
# --------------------------------------------------------------------------- #
def quant_value(w, scheme):
    """True quantized weight value ``dequant(quant(w))`` (no gradient path).

    * ``nvfp4``  - qwix NVFP4, tile 16 on the contraction axis (verified path).
    * ``ternary`` - BitNet b1.58: per-tensor absmean scale, weights in {-1,0,+1}.
    """
    if scheme == "nvfp4":
        v = qwix.dequantize(qwix.quantize(w, "nvfp4", tiled_axes={0: NVFP4_TILE}))
    elif scheme == "ternary":
        scale = jnp.mean(jnp.abs(w)) + 1e-8
        v = jnp.clip(jnp.round(w / scale), -1.0, 1.0) * scale
    else:  # pragma: no cover - guarded by PRECISIONS
        raise ValueError(f"unknown precision {scheme!r}")
    return jax.lax.stop_gradient(v.astype(w.dtype))


def ste_value(w, scheme):
    """Straight-through estimate: forward equals ``quant_value``, gradient is 1."""
    return w + (quant_value(w, scheme) - jax.lax.stop_gradient(w))


def _apply_to_kernels(state, fn_leaf):
    # Dense kernels are the only rank-2 params; neuron state (beta, thresholds)
    # is rank-1 and stays fp32 - matches spyx.quant's linear-only default.
    return jax.tree.map(
        lambda leaf: fn_leaf(leaf) if getattr(leaf, "ndim", 0) == 2 else leaf,
        state,
    )


# --------------------------------------------------------------------------- #
# Task + model (synthetic spiking classification; no dataset download).
# --------------------------------------------------------------------------- #
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


def build_flat(seed):
    """Return (theta0, graphdef, unravel, rest) so params live in flat fp32 space."""
    model = SpikingClassifier(rngs=nnx.Rngs(seed))
    gd, params, rest = nnx.split(model, nnx.Param, ...)
    theta0, unravel = ravel_pytree(params)
    return theta0, gd, unravel, rest


def _model_from(theta, gd, unravel, rest, *, scheme, ste):
    state = unravel(theta)
    if scheme is not None:
        if ste:
            q = lambda w: ste_value(w, scheme)  # noqa: E731
        else:
            q = lambda w: quant_value(w, scheme)  # noqa: E731
        state = _apply_to_kernels(state, q)
    return nnx.merge(gd, state, rest)


# --------------------------------------------------------------------------- #
# Evaluation: the TRUE quantized forward (no STE), identical for every arm.
# --------------------------------------------------------------------------- #
def evaluate(theta, gd, unravel, rest, test, scheme):
    xs, ys = test
    model = _model_from(theta, gd, unravel, rest, scheme=scheme, ste=False)
    losses = jnp.stack([ce(model(x), y) for x, y in zip(xs, ys, strict=True)])
    accs = jnp.stack([acc_fn(model(x), y)[0] for x, y in zip(xs, ys, strict=True)])
    return float(jnp.mean(losses)), float(jnp.mean(accs))


# --------------------------------------------------------------------------- #
# Arm 1/2: surrogate SGD (fp32 when scheme is None; STE-QAT otherwise).
# --------------------------------------------------------------------------- #
def train_sgd(train, seed, scheme):
    theta0, gd, unravel, rest = build_flat(seed)
    opt = optax.adam(LR)
    opt_state = opt.init(theta0)

    def loss_of(theta, x, y):
        model = _model_from(theta, gd, unravel, rest, scheme=scheme, ste=True)
        return ce(model(x), y)

    @jax.jit
    def step(theta, opt_state, x, y):
        loss, g = jax.value_and_grad(loss_of)(theta, x, y)
        updates, opt_state = opt.update(g, opt_state, theta)
        return optax.apply_updates(theta, updates), opt_state, loss

    xs, ys = train
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys, strict=True):
            theta0, opt_state, _ = step(theta0, opt_state, x, y)
    return theta0, gd, unravel, rest


# --------------------------------------------------------------------------- #
# Arm 3: CR-FM-NES over fp32 weights; fitness = TRUE quantized forward loss.
# --------------------------------------------------------------------------- #
def train_es(train, seed, scheme):
    theta0, gd, unravel, rest = build_flat(seed)
    d = theta0.size
    xs, ys = train
    ex = jnp.stack(xs)
    ey = jnp.stack(ys)

    def fitness(theta):
        model = _model_from(theta, gd, unravel, rest, scheme=scheme, ste=False)
        return jnp.mean(jax.vmap(lambda x, y: ce(model(x), y))(ex, ey))

    strat = CR_FM_NES(population_size=POP, solution=jnp.zeros(d))
    esp = strat.default_params.replace(std_init=STD_INIT)
    state = strat.init(jax.random.PRNGKey(seed), mean=theta0, params=esp)

    @jax.jit
    def evolve(state, key):
        def gen(carry, _):
            state, key = carry
            key, ak, tk = jax.random.split(key, 3)
            cands, state = strat.ask(ak, state, esp)
            fits = jax.vmap(fitness)(cands)
            state, _ = strat.tell(tk, cands, fits, state, esp)
            return (state, key), jnp.min(fits)

        (state, _k), hist = jax.lax.scan(gen, (state, key), None, length=GENERATIONS)
        return state, hist

    state, hist = evolve(state, jax.random.PRNGKey(seed + 1))
    jax.block_until_ready(hist)
    mean = state.mean if hasattr(state, "mean") else state.best_solution
    return mean, gd, unravel, rest, [float(h) for h in np.asarray(hist)]


# --------------------------------------------------------------------------- #
def main():
    print(
        f"backend={jax.default_backend()} SMOKE={SMOKE} "
        f"C={CHANNELS} H={HIDDEN} classes={N_CLASSES} T={SAMPLE_T} "
        f"epochs={EPOCHS} pop={POP} gens={GENERATIONS} seeds={SEEDS} "
        f"precisions={PRECISIONS}",
        flush=True,
    )
    arms: dict[str, dict] = {}

    def record(name, losses, accs, walls):
        arms[name] = {
            "true_loss": float(np.mean(losses)),
            "true_loss_std": float(np.std(losses)),
            "acc": float(np.mean(accs)),
            "wall_s": float(np.mean(walls)),
        }
        print(
            f"  {name:<22} true_loss={np.mean(losses):.4f}±{np.std(losses):.4f}  "
            f"acc={np.mean(accs) * 100:.1f}%  ({np.mean(walls):.1f}s)",
            flush=True,
        )

    # fp32 reference: trained without quant, reported on the fp32 forward.
    fl, fa, fw = [], [], []
    for seed in SEEDS:
        train, test = synthetic_data(seed)
        t0 = time.perf_counter()
        theta, gd, un, rest = train_sgd(train, seed, scheme=None)
        loss, acc = evaluate(theta, gd, un, rest, test, scheme=None)
        fl.append(loss)
        fa.append(acc)
        fw.append(time.perf_counter() - t0)
    record("fp32 (surrogate)", fl, fa, fw)

    # Per precision: STE-QAT vs ES, both scored on the true quantized forward.
    gaps = {}
    for prec in PRECISIONS:
        ste_l, ste_a, ste_w = [], [], []
        es_l, es_a, es_w = [], [], []
        for seed in SEEDS:
            train, test = synthetic_data(seed)
            t0 = time.perf_counter()
            theta, gd, un, rest = train_sgd(train, seed, scheme=prec)
            loss, acc = evaluate(theta, gd, un, rest, test, scheme=prec)
            ste_l.append(loss)
            ste_a.append(acc)
            ste_w.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            theta, gd, un, rest, _hist = train_es(train, seed, scheme=prec)
            loss, acc = evaluate(theta, gd, un, rest, test, scheme=prec)
            es_l.append(loss)
            es_a.append(acc)
            es_w.append(time.perf_counter() - t0)
        record(f"STE-QAT @ {prec}", ste_l, ste_a, ste_w)
        record(f"ES @ {prec}", es_l, es_a, es_w)
        gap = float(np.mean(ste_l) - np.mean(es_l))
        gaps[prec] = gap
        print(
            f"  -> STE-bias gap @ {prec}: (STE {np.mean(ste_l):.4f}) - "
            f"(ES {np.mean(es_l):.4f}) = {gap:+.4f}  "
            f"({'ES lower' if gap > 0 else 'STE lower'})",
            flush=True,
        )

    print(
        "\nSTE-bias gaps (positive = ES reaches lower true loss): "
        + "  ".join(f"{p}={g:+.4f}" for p, g in gaps.items()),
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
            "pop": POP,
            "generations": GENERATIONS,
            "lr": LR,
            "std_init": STD_INIT,
            "seeds": SEEDS,
            "precisions": PRECISIONS,
        },
        "arms": arms,
        "ste_bias_gap": gaps,
        "wall_s": float(sum(a["wall_s"] for a in arms.values())),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "quant_aware_evolution_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote quant_aware_evolution_results.json", flush=True)


if __name__ == "__main__":
    main()
