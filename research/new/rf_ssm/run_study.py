"""S5-RF vs reset-free ResonateFire vs PSU_LIF on a synthetic long-range task.

Compares three parallelizable spiking neurons — the new
``spyx.experimental.rf_ssm.RFSSM`` (S5/HiPPO pole init + PRF decoupled reset),
the reset-free ``spyx.phasor.ResonateFire`` (plain pole init, no reset), and the
real-leak ``spyx.experimental.PSU_LIF`` — on a controlled long-range task, plus
a neuron-primitive ``spyx.bench`` speed comparison.

The claim under test: HiPPO/S5 pole initialisation lets RFSSM integrate over
longer horizons than a plainly-initialised ResonateFire, without giving up the
O(log T) associative-scan parallelism (the decoupled reset stays scan-exact).

Task: **delayed cumulative-sign classification**. Each sequence is white noise of
length T with a single informative "cue" spike planted early; the label is the
sign of a weighted long-range accumulation, so the readout must carry information
across the full sequence — the regime where HiPPO init is supposed to help.

Modes:
* ``SPYX_SMOKE=1`` (used by CI/agents): tiny CPU synthetic run — a few hundred
  samples, short-ish T, a handful of epochs — just enough to prove the whole
  training + bench pipeline runs end-to-end and produces finite numbers. NOT a
  scientific result.
* full run (human-gated): set ``SPYX_SMOKE=0`` (or unset) and bump ``T``,
  ``N_TRAIN``, ``EPOCHS`` via env vars for a real long-range comparison on GPU.
  The full SSC / psMNIST accuracy runs are separate and also human-gated — see
  the README "Findings = PENDING full run".
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

import spyx
import spyx.nn as snn
from spyx import bench
from spyx.experimental import PSU_LIF
from spyx.experimental.rf_ssm import RFSSM
from spyx.phasor import ResonateFire

SMOKE = os.environ.get("SPYX_SMOKE", "1") != "0"

if SMOKE:
    T = int(os.environ.get("T", "64"))
    N_TRAIN = int(os.environ.get("N_TRAIN", "256"))
    N_TEST = int(os.environ.get("N_TEST", "128"))
    HIDDEN = int(os.environ.get("HIDDEN", "32"))
    EPOCHS = int(os.environ.get("EPOCHS", "3"))
    BATCH = 64
else:  # full run — human-gated
    T = int(os.environ.get("T", "784"))
    N_TRAIN = int(os.environ.get("N_TRAIN", "4096"))
    N_TEST = int(os.environ.get("N_TEST", "1024"))
    HIDDEN = int(os.environ.get("HIDDEN", "128"))
    EPOCHS = int(os.environ.get("EPOCHS", "40"))
    BATCH = 128

N_CLASSES = 2


# --------------------------------------------------------------------------- data
def make_task(key, n, T):
    """Delayed cumulative-sign task -> (x [n, T, 1], y [n]).

    A cue of random sign is planted at t=1; the rest is small white noise. The
    label is the sign of the cue, but it must survive a long integration window,
    so a neuron that forgets early input (short effective memory) cannot solve it.
    """
    k_cue, k_noise = jax.random.split(key)
    signs = jax.random.rademacher(k_cue, (n,)).astype(jnp.float32)  # +/-1
    x = 0.1 * jax.random.normal(k_noise, (n, T, 1))
    x = x.at[:, 1, 0].set(signs)  # planted cue
    y = (signs > 0).astype(jnp.int32)
    return x, y


# -------------------------------------------------------------------------- model
def make_neuron(kind, shape, rngs):
    act = spyx.axn.triangular()
    if kind == "RFSSM":
        return RFSSM(
            shape, pole_init="hippo", reset_init=1.0, activation=act, rngs=rngs
        )
    if kind == "ResonateFire":
        return ResonateFire(shape, activation=act, rngs=rngs)
    if kind == "PSU_LIF":
        return PSU_LIF(shape, activation=act, rngs=rngs)
    raise ValueError(kind)


class SeqClassifier(nnx.Module):
    def __init__(self, kind, in_dim, hidden, n_classes, *, rngs):
        self.core = snn.Sequential(
            nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs),
            make_neuron(kind, (hidden,), rngs),
            nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs),
            make_neuron(kind, (hidden,), rngs),
            nnx.Linear(hidden, n_classes, use_bias=False, rngs=rngs),
            snn.LI((n_classes,), rngs=rngs),
        )

    def __call__(self, x_BTC):
        # spyx.nn.run is time-major; keep batch-major I/O for the fn losses.
        out, _ = snn.run(self.core, x_BTC, batch_major=True)
        return out


# ----------------------------------------------------------------------- train/eval
def train_and_eval(kind, data):
    xtr, ytr, xte, yte = data
    model = SeqClassifier(kind, 1, HIDDEN, N_CLASSES, rngs=nnx.Rngs(0))
    loss_fn = spyx.fn.integral_crossentropy()
    acc_fn = spyx.fn.integral_accuracy()
    opt = nnx.Optimizer(model, optax.lion(3e-4), wrt=nnx.Param)

    @nnx.jit
    def step(m, o, ob, lb):
        loss, g = nnx.value_and_grad(lambda mm: loss_fn(mm(ob), lb))(m)
        o.update(m, g)
        return loss

    def batches(x, y):
        for i in range(0, x.shape[0], BATCH):
            yield x[i : i + BATCH], y[i : i + BATCH]

    def test_acc(m):
        accs = [float(acc_fn(m(xb), yb)[0]) for xb, yb in batches(xte, yte)]
        return float(np.mean(accs))

    # warm compile (excluded from the timed loop)
    xb0, yb0 = next(batches(xtr, ytr))
    step(model, opt, xb0, yb0)
    jax.block_until_ready(test_acc(model))

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        for xb, yb in batches(xtr, ytr):
            step(model, opt, xb, yb)
    train_s = time.perf_counter() - t0
    acc = test_acc(model)
    print(f"  [{kind}] acc={acc * 100:.2f}%  train={train_s:.2f}s", flush=True)
    return {"kind": kind, "test_acc": acc, "train_s": train_s, "epochs": EPOCHS}


# ------------------------------------------------------------ neuron-level bench
def neuron_bench():
    """Primitive fwd / fwd+bwd latency per neuron via their .parallel scan."""
    rows = []
    B = BATCH
    for kind in ("RFSSM", "ResonateFire", "PSU_LIF"):
        neuron = make_neuron(kind, (HIDDEN,), nnx.Rngs(0))
        seq = bench.benchmark(
            neuron, (HIDDEN,), seq_len=T, batch=B, run_fn=snn.run, name=f"{kind} (seq)"
        )
        rows.append(seq.as_dict())
        par = bench.benchmark(
            neuron, (HIDDEN,), seq_len=T, batch=B, name=f"{kind} (parallel)"
        )
        rows.append(par.as_dict())
    return rows


def main():
    print(
        f"backend={jax.default_backend()}  device={jax.devices()[0]}  "
        f"SMOKE={SMOKE}  T={T} N_TRAIN={N_TRAIN} HIDDEN={HIDDEN} EPOCHS={EPOCHS}",
        flush=True,
    )
    key = jax.random.PRNGKey(0)
    ktr, kte = jax.random.split(key)
    xtr, ytr = make_task(ktr, N_TRAIN, T)
    xte, yte = make_task(kte, N_TEST, T)
    data = (xtr, ytr, xte, yte)

    print("== accuracy + training wall-clock ==", flush=True)
    acc_rows = [train_and_eval(k, data) for k in ("RFSSM", "ResonateFire", "PSU_LIF")]

    print("\n== neuron-primitive speed (spyx.bench) ==", flush=True)
    bench_rows = neuron_bench()
    print(bench.format_table([bench.BenchResult(**r) for r in bench_rows]))

    out = {
        "config": {
            "T": T,
            "N_TRAIN": N_TRAIN,
            "HIDDEN": HIDDEN,
            "EPOCHS": EPOCHS,
            "smoke": SMOKE,
        },
        "accuracy": acc_rows,
        "neuron_bench": bench_rows,
    }
    with open("study_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote study_results.json")


if __name__ == "__main__":
    main()
