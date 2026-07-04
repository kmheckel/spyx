"""PSU_LIF & ResonateFire vs LIF: accuracy vs speed on SHD.

Trains one identical architecture (Linear -> neuron -> Linear -> neuron ->
Linear -> LI readout) with three interchangeable hidden neurons — the standard
soft-reset ``spyx.nn.LIF``, the reset-free parallel ``spyx.nn.PSU_LIF``, and the
complex resonate-and-fire ``spyx.phasor.ResonateFire`` — under the same optimiser,
surrogate, seed, and epoch budget, so any accuracy difference is attributable to
the neuron model. Speed is measured two ways: end-to-end training wall-clock (all
three driven by the sequential ``spyx.nn.run`` scan) and, at the neuron-primitive
level, ``spyx.bench`` forward / forward+backward latency — where PSU_LIF and
ResonateFire can additionally use their ``.parallel`` associative-scan path that
LIF cannot.

Data: the SHD prestage is expensive (tonic event->frame). Set SHD_CACHE to a
prebuilt .npz (keys train_obs/train_labels/test_obs/test_labels, obs = uint8
bit-packed along the time axis) to skip it; otherwise this builds one via
``spyx.data.SHD_loader`` and caches to ./shd_cache.npz.
"""

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
from spyx.phasor import ResonateFire

SAMPLE_T = 128
CHANNELS = 128
N_CLASSES = 20
HIDDEN = 64
EPOCHS = int(os.environ.get("EPOCHS", "40"))
CACHE = os.environ.get("SHD_CACHE", "./shd_cache.npz")


# --------------------------------------------------------------------------- data
def _build_cache(path):
    import spyx.data

    dl = spyx.data.SHD_loader(
        batch_size=256, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=16
    )

    def stage(epoch):
        obs, lab = [], []
        for b in epoch:
            obs.append(np.asarray(b.obs))
            lab.append(np.asarray(b.labels))
        return np.stack(obs), np.stack(lab)

    tro, trl = stage(dl.train_epoch())
    teo, tel = stage(dl.test_epoch())
    np.savez(path, train_obs=tro, train_labels=trl, test_obs=teo, test_labels=tel)


def load_dense():
    if not os.path.exists(CACHE):
        print(f"prestaging SHD -> {CACHE} (one-time, slow) ...", flush=True)
        _build_cache(CACHE)
    d = np.load(CACHE)

    def unpack(p):  # (N,B,T_packed,C) uint8 -> (N,B,T,C) f32
        return np.unpackbits(p, axis=2)[:, :, :SAMPLE_T, :].astype(np.float32)

    return (
        jnp.asarray(unpack(d["train_obs"])),
        jnp.asarray(d["train_labels"]),
        jnp.asarray(unpack(d["test_obs"])),
        jnp.asarray(d["test_labels"]),
    )


# -------------------------------------------------------------------------- model
def make_neuron(kind, shape, rngs):
    act = spyx.axn.triangular()
    if kind == "LIF":
        return snn.LIF(shape, activation=act, rngs=rngs)
    if kind == "PSU_LIF":
        return snn.PSU_LIF(shape, activation=act, rngs=rngs)
    if kind == "ResonateFire":
        return ResonateFire(shape, activation=act, rngs=rngs)
    raise ValueError(kind)


class SHDSNN(nnx.Module):
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
        t, _ = snn.run(self.core, jnp.transpose(x_BTC, (1, 0, 2)))
        return jnp.transpose(t, (1, 0, 2))


# ----------------------------------------------------------------------- train/eval
def train_and_eval(kind, data):
    tro, trl, teo, tel = data
    model = SHDSNN(kind, CHANNELS, HIDDEN, N_CLASSES, rngs=nnx.Rngs(0))
    loss_fn = spyx.fn.integral_crossentropy()
    acc_fn = spyx.fn.integral_accuracy()
    opt = nnx.Optimizer(model, optax.lion(3e-4), wrt=nnx.Param)

    @nnx.jit
    def step(m, o, ob, lb):
        def lf(mm):
            return loss_fn(mm(ob), lb)

        loss, g = nnx.value_and_grad(lf)(m)
        o.update(m, g)
        return loss

    def test_acc(m):
        return float(
            jnp.mean(
                jnp.stack([acc_fn(m(teo[i]), tel[i])[0] for i in range(teo.shape[0])])
            )
        )

    # warm compile (excluded from the timed loop)
    step(model, opt, tro[0], trl[0])
    jax.block_until_ready(test_acc(model))

    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        for i in range(tro.shape[0]):
            step(model, opt, tro[i], trl[i])
        if ep % 10 == 0 or ep == 1:
            print(
                f"  [{kind}] epoch {ep:3d} test_acc={test_acc(model) * 100:.2f}%",
                flush=True,
            )
    train_s = time.perf_counter() - t0
    acc = test_acc(model)
    print(f"  [{kind}] FINAL acc={acc * 100:.2f}%  train={train_s:.1f}s", flush=True)
    return {"kind": kind, "test_acc": acc, "train_s": train_s, "epochs": EPOCHS}


# ------------------------------------------------------------ neuron-level bench
def neuron_bench():
    """Primitive fwd / fwd+bwd latency per neuron; PSU/RF also via .parallel."""
    rows = []
    T, B, H = 256, 64, 256
    for kind in ("LIF", "PSU_LIF", "ResonateFire"):
        neuron = make_neuron(kind, (H,), nnx.Rngs(0))
        # sequential path (spyx.nn.run) — the only option for LIF
        seq = bench.benchmark(
            neuron, (H,), seq_len=T, batch=B, run_fn=snn.run, name=f"{kind} (seq)"
        )
        rows.append(seq.as_dict())
        # parallel associative-scan path where available
        if hasattr(neuron, "parallel"):
            par = bench.benchmark(
                neuron, (H,), seq_len=T, batch=B, name=f"{kind} (parallel)"
            )
            rows.append(par.as_dict())
    return rows


def main():
    print(f"backend={jax.default_backend()}  device={jax.devices()[0]}", flush=True)
    data = load_dense()
    print(
        f"SHD: train {data[0].shape}  test {data[2].shape}  epochs={EPOCHS}\n",
        flush=True,
    )

    print("== accuracy + training wall-clock ==", flush=True)
    acc_rows = [train_and_eval(k, data) for k in ("LIF", "PSU_LIF", "ResonateFire")]

    print("\n== neuron-primitive speed (spyx.bench) ==", flush=True)
    bench_rows = neuron_bench()
    print(bench.format_table([bench.BenchResult(**r) for r in bench_rows]))

    out = {"accuracy": acc_rows, "neuron_bench": bench_rows}
    with open("study_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote study_results.json")


if __name__ == "__main__":
    main()
