"""Sigma-delta vs rate-coded LIF: fewer transmitted events at matched accuracy.

Claim: on **temporally-redundant** input (a slowly-varying signal), a sigma-delta
graded neuron (``spyx.experimental.SigmaDelta``) reaches the same accuracy as a
rate-coded ``spyx.nn.LIF`` while transmitting **far fewer events** — because it emits
only the quantized *change* of its membrane, which is ~0 when the input is stable.
Event rate is the device-agnostic energy proxy (each transmitted event is one synaptic
op / packet, graded or binary). Also reports the ``.parallel`` associative-scan speedup.

Honest caveats: a graded event carries more bits than a binary spike (so 'fewer events'
is not the whole energy story); and on a dense GPU neither is sparse-exploited — the win
is on event-driven / neuromorphic hardware. Self-contained synthetic task; no dataset.

    SPYX_SMOKE=1 uv run python research/new/sigma_delta_neuron/sigma_delta_bench.py
    uv run python research/new/sigma_delta_neuron/sigma_delta_bench.py
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
import spyx.bench as bench
import spyx.fn as fn
import spyx.nn as snn
from spyx.experimental import SigmaDelta

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))
if SMOKE:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 16, 24, 3, 40
    N_TRAIN, BATCH, EPOCHS, SEEDS = 96, 32, 8, [0]
else:
    CHANNELS, HIDDEN, N_CLASSES, SAMPLE_T = 32, 64, 5, 80
    N_TRAIN, BATCH, EPOCHS, SEEDS = 512, 64, 25, [0, 1]

STEP = 0.5  # sigma-delta grid
ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def redundant_data(seed):
    """Each class = a **step-and-hold**: the class band jumps to a constant level
    early and holds it — genuinely temporally redundant (the regime sigma-delta is
    for). A rate-coded LIF must keep firing through the hold; sigma-delta emits at the
    step then goes silent.
    """
    rng = np.random.default_rng(seed)
    band = max(1, CHANNELS // N_CLASSES)
    t_on = SAMPLE_T // 4

    def make(n):
        y = rng.integers(0, N_CLASSES, size=n)
        x = 0.05 * rng.standard_normal((n, SAMPLE_T, CHANNELS)).astype(np.float32)
        for i in range(n):
            lo = y[i] * band
            x[i, t_on:, lo : lo + band] += 2.0  # step up, then HOLD constant to the end
        return jnp.asarray(x), jnp.asarray(y.astype(np.int32))

    def batched(x, y):
        obs = [jnp.transpose(x[s : s + BATCH], (1, 0, 2)) for s in range(0, x.shape[0] - BATCH + 1, BATCH)]
        lab = [y[s : s + BATCH] for s in range(0, x.shape[0] - BATCH + 1, BATCH)]
        return obs, lab

    return batched(*make(N_TRAIN)), batched(*make(N_TRAIN))


def make_net(kind, rngs):
    hidden = (
        snn.LIF((HIDDEN,), activation=axn.superspike(), rngs=rngs)
        if kind == "lif"
        else SigmaDelta((HIDDEN,), step=STEP, rngs=rngs)
    )
    net = snn.Sequential(
        nnx.Linear(CHANNELS, HIDDEN, rngs=rngs),
        hidden,
        nnx.Linear(HIDDEN, N_CLASSES, rngs=rngs),
        snn.LI((N_CLASSES,), rngs=rngs),
    )
    return net


def hidden_events(net, x_TBC):
    """Mean fraction of NON-ZERO hidden events / step (the SOP proxy).

    Scans the hidden neuron's ``__call__`` over the pre-activation so it works for both
    the binary ``LIF`` (nonzero = spike) and the graded ``SigmaDelta`` (nonzero = event).
    """
    lin = net.layers[0]
    neuron = net.layers[1]
    pre = jnp.einsum("tbc,cd->tbd", x_TBC, lin.kernel.value) + lin.bias.value

    def scan_step(V, xt):
        s, V = neuron(xt, V)
        return V, s

    _, events = jax.lax.scan(scan_step, neuron.initial_state(pre.shape[1]), pre)
    return float(jnp.mean(events != 0))


def train_eval(kind, train, test, seed):
    net = make_net(kind, nnx.Rngs(seed))
    opt = nnx.Optimizer(net, optax.adam(3e-3), wrt=nnx.Param)

    @nnx.jit
    def step(net, opt, x, y):
        def loss(m):
            return ce(snn.run(m, x)[0].transpose(1, 0, 2), y)

        loss_v, grads = nnx.value_and_grad(loss)(net)
        opt.update(net, grads)
        return loss_v

    xs, ys = train
    for _ in range(EPOCHS):
        for x, y in zip(xs, ys, strict=True):
            step(net, opt, x, y)

    xt, yt = test
    accs = [acc_fn(snn.run(net, x)[0].transpose(1, 0, 2), y)[0] for x, y in zip(xt, yt, strict=True)]
    evs = [hidden_events(net, x) for x in xt]
    return float(np.mean(accs)), float(np.mean(evs))


def main():
    print(f"backend={jax.default_backend()} SMOKE={SMOKE} redundant-signal task "
          f"C={CHANNELS} H={HIDDEN} classes={N_CLASSES} T={SAMPLE_T} step={STEP}", flush=True)
    t0 = time.perf_counter()
    rows = {}
    for kind in ["lif", "sigma_delta"]:
        accs, evs = [], []
        for seed in SEEDS:
            tr, te = redundant_data(seed)
            a, e = train_eval(kind, tr, te, seed)
            accs.append(a)
            evs.append(e)
        rows[kind] = {"acc": float(np.mean(accs)), "event_rate": float(np.mean(evs))}
        print(f"  {kind:12s} acc={np.mean(accs) * 100:5.1f}%  event_rate={np.mean(evs) * 100:5.1f}%", flush=True)

    ev_ratio = rows["lif"]["event_rate"] / max(rows["sigma_delta"]["event_rate"], 1e-9)
    print(f"  -> event reduction (LIF / sigma-delta) = {ev_ratio:.2f}x "
          f"at {rows['sigma_delta']['acc'] * 100:.1f}% vs {rows['lif']['acc'] * 100:.1f}%", flush=True)

    # .parallel speedup of the sigma-delta neuron itself
    print("\n== sigma-delta .parallel vs sequential (spyx.bench) ==", flush=True)
    speed = bench.compare(
        {"SigmaDelta": lambda: SigmaDelta((HIDDEN,), step=STEP, rngs=nnx.Rngs(0))},
        (HIDDEN,), seq_lens=[SAMPLE_T], batch=BATCH, backward=False, n_iters=10,
    )
    print(bench.format_table(speed), flush=True)
    dt = time.perf_counter() - t0

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sigma_delta_results.json"), "w") as f:
        json.dump({"config": {"smoke": SMOKE, "device": str(jax.devices()[0]),
                              "channels": CHANNELS, "hidden": HIDDEN, "n_classes": N_CLASSES,
                              "sample_T": SAMPLE_T, "step": STEP, "seeds": SEEDS, "wall_s": dt},
                   "arms": rows, "event_reduction_x": ev_ratio}, f, indent=2)
    print(f"\nwrote sigma_delta_results.json  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
