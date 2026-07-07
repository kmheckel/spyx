"""FP4 block-microscaling for SNNs: weight-QAT + membrane-state quantization.

The first FP4-microscaling x spiking-neural-net datapoints. Two verified
adversarial survey sweeps found **no** prior application of block-scaled FP4
(NVFP4 / MXFP4) to SNNs — the nearest boundary, AQ4SViT (arXiv:2606.15523), is
*integer* search quantization, not micro-scaled floating point. Spyx is the only
SNN library that already carries the FP4 formats (via ``qwix``), so this note
takes the confirmed, unclaimed gap (research/PROGRAM.md flagship **F3**).

**Claim under test.** FP4 microscaling is a viable *weight* format for spiking
classifiers — accuracy competitive with int8 and better than ternary at
matched-ish bits — and **membrane-state quantization** (the SQUAT gap, Chowdhury
et al., arXiv:2404.19668: quantize the neuron membrane potential ``V`` every
step, not just the weights) is feasible at FP4 / int8.

Arms (matched init, matched synthetic task, identical TRUE-quant evaluation):

* **fp32**             - surrogate SGD, no quantization (reference ceiling).
* **int8**             - weight-only STE-QAT, symmetric absmax int8.
* **nvfp4**            - weight-only STE-QAT, qwix NVFP4 (tile 16, contraction axis).
* **mxfp4**            - weight-only STE-QAT, qwix MXFP4 (tile 32, contraction axis).
* **ternary**          - weight-only STE-QAT, BitNet b1.58 {-1, 0, +1}.
* **nvfp4 +Vstate**    - nvfp4 weights AND nvfp4 membrane state V each step (SQUAT).
* **int8 +Vstate**     - int8 weights AND int8 membrane state V each step (SQUAT).
* **QANN (int-b)**     - a matched-architecture rate-coded ANN quantized to
  ``b = ceil(log2(T+1))`` bits, the honest energy-matched baseline of
  arXiv:2409.08290 (a T-step binary accumulator carries ``log2(T+1)`` levels).

Every SNN arm's *reported* number is the identical true quantized forward
(``Q(w) = dequant(quant(w))`` for weights, ``Q(V)`` for state, no STE), so the
comparison isolates the format, not the gradient path. Per arm we report test
accuracy and the weight (and membrane-state) bit-footprint. Writes
``fp4_spiking_qat_results.json``. Run::

    SPYX_SMOKE=1 uv run python research/new/fp4_spiking_qat/fp4_spiking_qat.py
    uv run python research/new/fp4_spiking_qat/fp4_spiking_qat.py   # larger synthetic

The **real-SSC** run (Spiking Speech Commands, 20 classes, 700 channels) is
HUMAN-GATED: it needs a dataset download and a GPU, so it is intentionally NOT
wired here — this script is fully synthetic and self-contained.
"""

from __future__ import annotations

import json
import math
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

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))
if SMOKE:
    # dims are multiples of 32 so both NVFP4 (tile 16) and MXFP4 (tile 32) tile
    # the contraction axis exactly.
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 32, 32, 4, 16
    N_TRAIN, BATCH, EPOCHS = 64, 16, 8
    SEEDS = [0, 1]
else:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 64, 128, 8, 16
    N_TRAIN, BATCH, EPOCHS = 512, 32, 25
    SEEDS = [0, 1, 2]

LR = 5e-3
NVFP4_TILE = 16
MXFP4_TILE = 32
# Honest energy-matched ANN precision: a T-step binary spike accumulator spans
# T+1 integer levels, i.e. ceil(log2(T+1)) bits (arXiv:2409.08290).
ANN_BITS = math.ceil(math.log2(SAMPLE_T + 1))

# Weight formats compared. int8 is near-lossless; the FP4 pair and ternary are
# the extreme-precision datapoints the claim is about.
WEIGHT_SCHEMES = ["int8", "nvfp4", "mxfp4", "ternary"]
# Membrane-state (SQUAT) arms: weights AND V quantized at the same precision.
STATE_SCHEMES = ["nvfp4", "int8"]

# Effective stored bits/element including the shared block-scale overhead:
# NVFP4 -> e4m3 (8-bit) scale per 16 elems (+0.5); MXFP4 -> e8m0 (8-bit) per 32
# (+0.25); ternary -> log2(3); int/fp are their nominal width.
EFFECTIVE_BITS = {
    "fp32": 32.0,
    "int8": 8.0,
    "int4": 4.0,  # symmetric absmax int4 (levels -7..7); matched-bit vs FP4.
    "nvfp4": 4.0 + 8.0 / NVFP4_TILE,
    "mxfp4": 4.0 + 8.0 / MXFP4_TILE,
    "ternary": math.log2(3),
    f"int{ANN_BITS}": float(ANN_BITS),
}

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


# --------------------------------------------------------------------------- #
# Quantization: one true-forward Q(x), plus an STE wrapper sharing that value.
# `tile_axis` is the contraction axis for weights (0) or the feature axis for
# membrane state (-1); both target dims are multiples of 32.
# --------------------------------------------------------------------------- #
def _int_absmax(x, bits):
    """Symmetric absmax integer quantization to ``bits`` bits (dequantized)."""
    max_int = float(2 ** (bits - 1) - 1)
    amax = jnp.max(jnp.abs(x)) + 1e-8
    scale = amax / max_int
    return jnp.clip(jnp.round(x / scale), -max_int, max_int) * scale


def true_quant(x, scheme, tile_axis):
    """True quantized value ``dequant(quant(x))`` for a weight or state tensor.

    * ``int8`` / ``int<b>`` - symmetric absmax integer grid.
    * ``nvfp4``  - qwix NVFP4, tile 16 on ``tile_axis`` (the verified qwix path).
    * ``mxfp4``  - qwix MXFP4, tile 32 on ``tile_axis``.
    * ``ternary`` - BitNet b1.58: per-tensor absmean scale, weights in {-1, 0, +1}.
    """
    axis = tile_axis % x.ndim
    if scheme == "nvfp4":
        v = qwix.dequantize(qwix.quantize(x, "nvfp4", tiled_axes={axis: NVFP4_TILE}))
    elif scheme == "mxfp4":
        v = qwix.dequantize(qwix.quantize(x, "mxfp4", tiled_axes={axis: MXFP4_TILE}))
    elif scheme == "ternary":
        scale = jnp.mean(jnp.abs(x)) + 1e-8
        v = jnp.clip(jnp.round(x / scale), -1.0, 1.0) * scale
    elif scheme.startswith("int"):
        v = _int_absmax(x, int(scheme[3:]))
    else:  # pragma: no cover - guarded by the scheme lists
        raise ValueError(f"unknown scheme {scheme!r}")
    return jax.lax.stop_gradient(v.astype(x.dtype))


def apply_quant(x, scheme, tile_axis, *, ste):
    """Quantize ``x``; ``ste=True`` keeps the identity-gradient STE path."""
    q = true_quant(x, scheme, tile_axis)
    if ste:
        return x + (q - jax.lax.stop_gradient(x))
    return q


def _quant_kernels(state, scheme, ste):
    # Dense kernels are the only rank-2 params; neuron state (beta, thresholds)
    # is rank-1 and stays fp32 - matches spyx.quant's linear-only default. The
    # kernel's contraction axis is 0 (in_features), which is a multiple of 32.
    if scheme is None:
        return state
    return jax.tree.map(
        lambda leaf: apply_quant(leaf, scheme, 0, ste=ste)
        if getattr(leaf, "ndim", 0) == 2
        else leaf,
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


class QStateLIF(nnx.Module):
    """LIF whose membrane ``V`` is (optionally) quantized every timestep (SQUAT).

    Identical dynamics to :class:`spyx.nn.LIF`; when ``state_scheme`` is set the
    post-update membrane is passed through :func:`apply_quant` on its feature
    axis before being carried forward, so the recurrence itself runs at reduced
    precision. ``state_ste`` toggles the straight-through path (True for QAT,
    False for the true-quant evaluation).
    """

    def __init__(
        self, hidden_shape, *, state_scheme=None, state_ste=True, threshold=1.0, rngs
    ):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.state_scheme = state_scheme
        self.state_ste = state_ste
        self.spike = axn.superspike()
        self.beta = nnx.Param(
            nnx.initializers.truncated_normal(stddev=0.25)(rngs.params(), hidden_shape)
            + 0.5
        )

    def __call__(self, x, V):
        beta = jnp.clip(self.beta[...], 0, 1)
        spikes = self.spike(V - self.threshold)
        V = beta * V + x - spikes * self.threshold
        if self.state_scheme is not None:
            V = apply_quant(V, self.state_scheme, -1, ste=self.state_ste)
        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(self.hidden_shape))


class SpikingClassifier(nnx.Module):
    def __init__(self, *, state_scheme=None, state_ste=True, rngs):
        self.net = snn.Sequential(
            nnx.Linear(CHANNELS, HIDDEN, rngs=rngs),
            QStateLIF(
                (HIDDEN,), state_scheme=state_scheme, state_ste=state_ste, rngs=rngs
            ),
            nnx.Linear(HIDDEN, N_CLASSES, rngs=rngs),
            snn.LI((N_CLASSES,), rngs=rngs),
        )

    def __call__(self, x_TBC):
        traces, _ = snn.run(self.net, x_TBC)
        return jnp.transpose(traces, (1, 0, 2))


# Model class that ``build_flat`` instantiates. Overridable so sibling studies
# can swap in a richer neuron (e.g. ALIF) or a deeper stack while keeping the
# weight-QAT path (kernel STE-QAT via ``_quant_kernels``) fully intact. Any
# replacement must accept ``state_scheme``/``state_ste``/``rngs`` and expose the
# same ``__call__(x_TBC) -> [B, T, N_CLASSES]`` contract.
MODEL_CLS = SpikingClassifier


def build_flat(seed, state_scheme):
    """Flat fp32 params + graphdefs for the STE-train and true-eval models.

    The two graphdefs differ only in the static ``state_ste`` flag; the params
    (and ``rest``) are identical, so trained ``theta`` merges into either.
    """
    m_tr = MODEL_CLS(state_scheme=state_scheme, state_ste=True, rngs=nnx.Rngs(seed))
    gd_tr, params, rest = nnx.split(m_tr, nnx.Param, ...)
    m_ev = MODEL_CLS(
        state_scheme=state_scheme, state_ste=False, rngs=nnx.Rngs(seed)
    )
    gd_ev, _, _ = nnx.split(m_ev, nnx.Param, ...)
    theta0, unravel = ravel_pytree(params)
    return theta0, gd_tr, gd_ev, unravel, rest


def _model_from(theta, gd, unravel, rest, *, weight_scheme, weight_ste):
    state = _quant_kernels(unravel(theta), weight_scheme, weight_ste)
    return nnx.merge(gd, state, rest)


# --------------------------------------------------------------------------- #
# Evaluation: the TRUE quantized forward (no STE anywhere), identical per arm.
# --------------------------------------------------------------------------- #
def evaluate(theta, gd_ev, unravel, rest, test, weight_scheme):
    xs, ys = test
    model = _model_from(
        theta, gd_ev, unravel, rest, weight_scheme=weight_scheme, weight_ste=False
    )
    accs = jnp.stack([acc_fn(model(x), y)[0] for x, y in zip(xs, ys, strict=True)])
    return float(jnp.mean(accs))


def train_snn(train, seed, weight_scheme, state_scheme):
    theta, gd_tr, gd_ev, unravel, rest = build_flat(seed, state_scheme)
    opt = optax.adam(LR)
    opt_state = opt.init(theta)

    def loss_of(theta, x, y):
        model = _model_from(
            theta, gd_tr, unravel, rest, weight_scheme=weight_scheme, weight_ste=True
        )
        return ce(model(x), y)

    @jax.jit
    def step(theta, opt_state, x, y):
        loss, g = jax.value_and_grad(loss_of)(theta, x, y)
        updates, opt_state = opt.update(g, opt_state, theta)
        return optax.apply_updates(theta, updates), opt_state, loss

    xs, ys = train
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys, strict=True):
            theta, opt_state, _ = step(theta, opt_state, x, y)
    return theta, gd_ev, unravel, rest


# --------------------------------------------------------------------------- #
# Quantized-ANN baseline: rate-coded, matched arch, int-b STE-QAT.
# --------------------------------------------------------------------------- #
def _ann_init(seed):
    rng = np.random.default_rng(1000 + seed)
    w1 = rng.standard_normal((CHANNELS, HIDDEN)).astype(np.float32) / math.sqrt(CHANNELS)
    b1 = np.zeros(HIDDEN, np.float32)
    w2 = rng.standard_normal((HIDDEN, N_CLASSES)).astype(np.float32) / math.sqrt(HIDDEN)
    b2 = np.zeros(N_CLASSES, np.float32)
    return {
        "w1": jnp.asarray(w1),
        "b1": jnp.asarray(b1),
        "w2": jnp.asarray(w2),
        "b2": jnp.asarray(b2),
    }


def _ann_logits(p, x_TBC, *, ste):
    # Rate-code the spike train: sum over time -> static [B, C] activation.
    rate = jnp.sum(x_TBC, axis=0)
    w1 = apply_quant(p["w1"], f"int{ANN_BITS}", 0, ste=ste)
    w2 = apply_quant(p["w2"], f"int{ANN_BITS}", 0, ste=ste)
    h = jax.nn.relu(rate @ w1 + p["b1"])
    return h @ w2 + p["b2"]


def train_ann(train, test, seed):
    p = _ann_init(seed)
    opt = optax.adam(LR)
    opt_state = opt.init(p)

    def loss_of(p, x, y):
        logits = _ann_logits(p, x, ste=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step(p, opt_state, x, y):
        loss, g = jax.value_and_grad(loss_of)(p, x, y)
        updates, opt_state = opt.update(g, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    xs, ys = train
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys, strict=True):
            p, opt_state, _ = step(p, opt_state, x, y)

    xs, ys = test
    accs = []
    for x, y in zip(xs, ys, strict=True):
        logits = _ann_logits(p, x, ste=False)  # true-quant forward
        accs.append(float(jnp.mean(jnp.argmax(logits, -1) == y)))
    return float(np.mean(accs))


# --------------------------------------------------------------------------- #
def _weight_footprint(bits):
    n = CHANNELS * HIDDEN + HIDDEN * N_CLASSES
    return n, n * bits / 8.0 / 1024.0  # (n_params, KiB)


def main():
    print(
        f"backend={jax.default_backend()} SMOKE={SMOKE} "
        f"C={CHANNELS} H={HIDDEN} classes={N_CLASSES} T={SAMPLE_T} "
        f"epochs={EPOCHS} seeds={SEEDS} ANN_bits=int{ANN_BITS} "
        f"weight_schemes={WEIGHT_SCHEMES} state_schemes={STATE_SCHEMES}",
        flush=True,
    )
    n_params, _ = _weight_footprint(32.0)
    arms: dict[str, dict] = {}

    def record(name, accs, walls, wbits, sbits):
        _, wkib = _weight_footprint(wbits)
        arms[name] = {
            "acc": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "wall_s": float(np.mean(walls)),
            "weight_bits": wbits,
            "weight_KiB": wkib,
            "state_bits": sbits,
        }
        sb = f" Vstate={sbits:.2f}b" if sbits is not None else ""
        print(
            f"  {name:<18} acc={np.mean(accs) * 100:5.1f}%±{np.std(accs) * 100:4.1f}  "
            f"W={wbits:5.2f}b ({wkib:.2f}KiB){sb}  ({np.mean(walls):.1f}s)",
            flush=True,
        )

    def run_snn_arm(name, weight_scheme, state_scheme, wbits, sbits):
        accs, walls = [], []
        for seed in SEEDS:
            train, test = synthetic_data(seed)
            t0 = time.perf_counter()
            theta, gd_ev, un, rest = train_snn(train, seed, weight_scheme, state_scheme)
            accs.append(evaluate(theta, gd_ev, un, rest, test, weight_scheme))
            walls.append(time.perf_counter() - t0)
        record(name, accs, walls, wbits, sbits)

    # fp32 reference (no quantization anywhere).
    run_snn_arm("fp32", None, None, EFFECTIVE_BITS["fp32"], None)

    # Weight-only QAT across precisions.
    for scheme in WEIGHT_SCHEMES:
        run_snn_arm(scheme, scheme, None, EFFECTIVE_BITS[scheme], None)

    # Membrane-state (SQUAT) arms: weights AND V quantized at the same precision.
    for scheme in STATE_SCHEMES:
        run_snn_arm(
            f"{scheme}+Vstate",
            scheme,
            scheme,
            EFFECTIVE_BITS[scheme],
            EFFECTIVE_BITS[scheme],
        )

    # Honest quantized-ANN baseline (rate-coded, int-b, matched arch).
    accs, walls = [], []
    for seed in SEEDS:
        train, test = synthetic_data(seed)
        t0 = time.perf_counter()
        accs.append(train_ann(train, test, seed))
        walls.append(time.perf_counter() - t0)
    record(f"QANN(int{ANN_BITS})", accs, walls, EFFECTIVE_BITS[f"int{ANN_BITS}"], None)

    out = {
        "config": {
            "smoke": SMOKE,
            "channels": CHANNELS,
            "hidden": HIDDEN,
            "n_classes": N_CLASSES,
            "sample_T": SAMPLE_T,
            "epochs": EPOCHS,
            "lr": LR,
            "seeds": SEEDS,
            "ann_bits": ANN_BITS,
            "weight_schemes": WEIGHT_SCHEMES,
            "state_schemes": STATE_SCHEMES,
            "n_weight_params": n_params,
        },
        "arms": arms,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "fp4_spiking_qat_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote fp4_spiking_qat_results.json", flush=True)


if __name__ == "__main__":
    main()
