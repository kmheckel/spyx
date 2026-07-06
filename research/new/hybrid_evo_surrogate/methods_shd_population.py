"""Population-size sweep on SHD (built for the iGPU): does larger K rescue ES/SGES?

ES variance falls as the population size K grows, so the CPU runs (K=24) may simply
have starved the evolutionary arms. This sweeps K over a larger budget, comparing
full-space vs hypernetwork-compressed (synecdoche) search for ES and SGES, with the
surrogate gradient as the (K-independent) reference.

Run on the gfx1151 iGPU::

    SYN=/home/kade/Code/synecdoche/src
    JAX_PLATFORMS=rocm _ROCM_SDK_PRELOADED=0 PYTHONPATH=$SYN \
      ~/.venvs/jax-rocm-0.9.2/bin/python research/new/hybrid_evo_surrogate/methods_shd_population.py

Writes ``shd_population_results.json``.
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
SAMPLE_T = int(os.environ.get("SAMPLE_T", "64"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
N_CLASSES = 20
BATCH = int(os.environ.get("BATCH", "128"))
N_TRAIN_BATCHES = int(os.environ.get("TRAIN_BATCHES", "40"))
N_TEST_BATCHES = int(os.environ.get("TEST_BATCHES", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
SIGMA = float(os.environ.get("SIGMA", "0.02"))
LR = float(os.environ.get("LR", "3e-3"))
EMB = int(os.environ.get("EMB", "128"))
KS = [int(k) for k in os.environ.get("KS", "32,128,512").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def _unpack(o):
    return jnp.unpackbits(jnp.asarray(o), axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)


def load_shd():
    dl = data.SHD_loader(batch_size=BATCH, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=0)
    tr_o, tr_l = dl.prestage("train")
    te_o, te_l = dl.prestage("test")
    ntr = min(N_TRAIN_BATCHES, tr_o.shape[0])
    nte = min(N_TEST_BATCHES, te_o.shape[0])
    train = [(_unpack(tr_o[i]), jnp.asarray(tr_l[i])) for i in range(ntr)]
    test = [(_unpack(te_o[i]), jnp.asarray(te_l[i])) for i in range(nte)]
    return train, test


class SpikingClassifier(nnx.Module):
    def __init__(self, *, rngs):
        self.net = snn.Sequential(
            nnx.Linear(CHANNELS, HIDDEN, use_bias=False, rngs=rngs),
            snn.LIF((HIDDEN,), activation=axn.superspike(), rngs=rngs),
            nnx.Linear(HIDDEN, N_CLASSES, use_bias=False, rngs=rngs),
            snn.LI((N_CLASSES,), rngs=rngs),
        )

    def __call__(self, x):
        return snn.run(self.net, x, batch_major=True)[0]


def loss_fn(m, x, y):
    return ce(m(x), y)


def evaluate(m, test):
    return float(jnp.mean(jnp.stack([acc_fn(m(x), y)[0] for x, y in test])))


def _fresh(seed):
    return SpikingClassifier(rngs=nnx.Rngs(seed))


def _make_hyper(model, seed):
    return syn.RandomProjection(nnx.state(model, nnx.Param), EMB, rngs=nnx.Rngs(seed + 100))


def train(train_data, test, seed, *, space, kind, k):
    """space in {full, hyper}; kind in {surrogate, es, sges}."""
    model = _fresh(seed)
    if space == "hyper":
        graphdef, _p, rest = nnx.split(model, nnx.Param, ...)
        hyper = _make_hyper(model, seed)

        def loss(h, x, y):
            return loss_fn(nnx.merge(graphdef, h(), rest), x, y)

        opt_target, lossf = hyper, loss
    else:
        opt_target, lossf = model, loss_fn

    opt = nnx.Optimizer(opt_target, optax.adam(LR), wrt=nnx.Param)
    key = jax.random.PRNGKey(seed)
    sges_step = make_sges_hybrid_train_step(lossf, lossf, num_samples=k, sigma=SIGMA)

    @nnx.jit  # compile the surrogate step once instead of re-tracing every step
    def sur_step(t, o, xb, yb):
        _v, g = nnx.value_and_grad(lambda tt: lossf(tt, xb, yb))(t)
        o.update(t, g)

    for _ in range(EPOCHS):
        for x, y in train_data:
            key, sub = jax.random.split(key)
            if kind == "surrogate":
                sur_step(opt_target, opt, x, y)
            elif kind == "es":
                g = es_gradient(opt_target, lossf, sub, batch=(x, y), num_samples=k, sigma=SIGMA)
                opt.update(opt_target, g)
            else:
                sges_step(opt_target, opt, sub, x, y)

    final = nnx.merge(graphdef, hyper(), rest) if space == "hyper" else model
    return evaluate(final, test)


def main():
    print(
        f"backend={jax.default_backend()}  SHD C={CHANNELS} T={SAMPLE_T} H={HIDDEN} "
        f"train={N_TRAIN_BATCHES}x{BATCH} epochs={EPOCHS} emb={EMB} KS={KS} seeds={SEEDS}",
        flush=True,
    )
    train_data, test = load_shd()
    full_dim = syn.param_count(_fresh(0))
    hyper_dim = syn.param_count(_make_hyper(_fresh(0), 0))
    print(f"search dims: full={full_dim}  hyper={hyper_dim}\n", flush=True)

    out = {"config": {"epochs": EPOCHS, "emb": EMB, "KS": KS, "seeds": SEEDS,
                      "full_dim": full_dim, "hyper_dim": hyper_dim, "T": SAMPLE_T,
                      "train_batches": N_TRAIN_BATCHES}, "arms": {}}

    def run_arm(space, kind, k):
        t0 = time.perf_counter()
        accs = [train(train_data, test, s, space=space, kind=kind, k=k) for s in SEEDS]
        dt = time.perf_counter() - t0
        return float(np.mean(accs)), dt

    # K-independent surrogate references.
    for space in ("full", "hyper"):
        acc, dt = run_arm(space, "surrogate", KS[0])
        out["arms"][f"{space} surrogate"] = acc
        print(f"  {space:<5} surrogate            acc={acc * 100:5.1f}%  ({dt:.0f}s)", flush=True)

    # Population sweep for the evolutionary / hybrid arms.
    for kind in ("es", "sges"):
        for space in ("full", "hyper"):
            for k in KS:
                acc, dt = run_arm(space, kind, k)
                out["arms"][f"{space} {kind} K={k}"] = acc
                print(
                    f"  {space:<5} {kind:<5} K={k:<4}       acc={acc * 100:5.1f}%  ({dt:.0f}s)",
                    flush=True,
                )

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "shd_population_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote shd_population_results.json", flush=True)


if __name__ == "__main__":
    main()
