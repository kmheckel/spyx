"""Full-JIT SHD comparison: the entire training loop compiled as one XLA program.

The spyx way (Heckel & Nowotny 2024): stage the whole dataset on-device and
``jax.lax.scan`` the training loop over epochs × batches, so a run is a single
compiled kernel with no per-step Python/dispatch overhead. Works for all three
methods by operating on the flat parameter vector: surrogate via ``jax.grad``, ES /
SGES via the flat ``_es_flat`` / ``_sges_flat`` estimators (vmapped over the
population inside the scan). Full-space vs hypernetwork-compressed (synecdoche).

Run on the gfx1151 iGPU::

    SYN=/home/kade/Code/synecdoche/src
    JAX_PLATFORMS=rocm _ROCM_SDK_PRELOADED=0 PYTHONPATH=$SYN \
      ~/.venvs/jax-rocm-0.9.2/bin/python research/new/hybrid_evo_surrogate/methods_shd_jit.py
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
from jax.flatten_util import ravel_pytree

import spyx.axn as axn
import spyx.data as data
import spyx.fn as fn
import spyx.nn as snn
from spyx.experimental.hybrid import _es_flat, _sges_flat

CHANNELS = 128
SAMPLE_T = int(os.environ.get("SAMPLE_T", "128"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
N_CLASSES = 20
BATCH = int(os.environ.get("BATCH", "256"))
N_TRAIN_BATCHES = int(os.environ.get("TRAIN_BATCHES", "100"))
N_TEST_BATCHES = int(os.environ.get("TEST_BATCHES", "20"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
SIGMA = float(os.environ.get("SIGMA", "0.02"))
LR = float(os.environ.get("LR", "3e-3"))
EMB = int(os.environ.get("EMB", "128"))
KS = [int(k) for k in os.environ.get("KS", "512").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]

ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def _unpack(o):
    return jnp.unpackbits(jnp.asarray(o), axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)


def load_shd():
    dl = data.SHD_loader(batch_size=BATCH, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=0)
    tr_o, tr_l = dl.prestage("train")
    te_o, te_l = dl.prestage("test")
    ntr, nte = min(N_TRAIN_BATCHES, tr_o.shape[0]), min(N_TEST_BATCHES, te_o.shape[0])
    bx = jnp.stack([_unpack(tr_o[i]) for i in range(ntr)])  # [nb, B, T, C]
    by = jnp.stack([jnp.asarray(tr_l[i]) for i in range(ntr)])
    tx = jnp.stack([_unpack(te_o[i]) for i in range(nte)])
    ty = jnp.stack([jnp.asarray(te_l[i]) for i in range(nte)])
    return (bx, by), (tx, ty)


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


def _fresh(seed):
    return SpikingClassifier(rngs=nnx.Rngs(seed))


def _make_hyper(model, seed):
    return syn.RandomProjection(nnx.state(model, nnx.Param), EMB, rngs=nnx.Rngs(seed + 100))


def build_flat(space, seed):
    """Return (theta0, loss_flat(theta, x, y), rebuild(theta) -> SNN)."""
    model = _fresh(seed)
    snn_gd, snn_p, snn_rest = nnx.split(model, nnx.Param, ...)
    if space == "full":
        theta0, unravel = ravel_pytree(snn_p)

        def loss_flat(theta, x, y):
            return loss_fn(nnx.merge(snn_gd, unravel(theta), snn_rest), x, y)

        def rebuild(theta):
            return nnx.merge(snn_gd, unravel(theta), snn_rest)
    else:
        hyper = _make_hyper(model, seed)
        hgd, hp, hrest = nnx.split(hyper, nnx.Param, ...)
        theta0, unravel = ravel_pytree(hp)

        def _snn(theta):
            gen = nnx.merge(hgd, unravel(theta), hrest)()
            return nnx.merge(snn_gd, gen, snn_rest)

        def loss_flat(theta, x, y):
            return loss_fn(_snn(theta), x, y)

        rebuild = _snn
    return theta0, loss_flat, rebuild


def make_train(loss_flat, kind, k):
    tx = optax.adam(LR)

    def grad_flat(theta, x, y, key):
        tlf = lambda t: loss_flat(t, x, y)  # noqa: E731
        if kind == "surrogate":
            return jax.grad(tlf)(theta)
        if kind == "es":
            return _es_flat(theta, tlf, key, num_samples=k, sigma=SIGMA)
        g_s = jax.grad(tlf)(theta)
        g_es, *_ = _sges_flat(theta, tlf, g_s, key, num_samples=k, sigma=SIGMA, eps=1e-8)
        return g_es

    def step(carry, batch):
        theta, opt_state, key = carry
        x, y = batch
        key, sub = jax.random.split(key)
        g = grad_flat(theta, x, y, sub)
        updates, opt_state = tx.update(g, opt_state, theta)
        return (optax.apply_updates(theta, updates), opt_state, key), None

    @jax.jit
    def train_all(theta, key, bx, by):
        opt_state = tx.init(theta)

        def epoch(carry, _):
            carry, _ = jax.lax.scan(step, carry, (bx, by))
            return carry, None

        (theta, _os, _k), _ = jax.lax.scan(epoch, (theta, opt_state, key), None, length=EPOCHS)
        return theta

    return train_all


def evaluate(model, test):
    tx, ty = test
    accs = jax.vmap(lambda x, y: acc_fn(model(x), y)[0])(tx, ty)
    return float(jnp.mean(accs))


def main():
    print(
        f"backend={jax.default_backend()}  FULL-JIT  SHD C={CHANNELS} T={SAMPLE_T} H={HIDDEN} "
        f"train={N_TRAIN_BATCHES}x{BATCH} epochs={EPOCHS} emb={EMB} KS={KS} seeds={SEEDS}",
        flush=True,
    )
    train, test = load_shd()
    bx, by = train
    print(f"staged: {bx.shape[0]} train batches of {bx.shape[1]}x{bx.shape[2]}x{bx.shape[3]}", flush=True)

    out = {"config": {"T": SAMPLE_T, "batch": BATCH, "epochs": EPOCHS, "emb": EMB,
                      "KS": KS, "seeds": SEEDS, "train_batches": int(bx.shape[0])}, "arms": {}}

    def run(space, kind, k):
        t0 = time.perf_counter()
        accs = []
        for seed in SEEDS:
            theta0, loss_flat, rebuild = build_flat(space, seed)
            train_all = make_train(loss_flat, kind, k)
            theta = train_all(theta0, jax.random.PRNGKey(seed), bx, by)
            jax.block_until_ready(theta)
            accs.append(evaluate(rebuild(theta), test))
        return float(np.mean(accs)), time.perf_counter() - t0

    for space in ("full", "hyper"):
        acc, dt = run(space, "surrogate", KS[0])
        out["arms"][f"{space} surrogate"] = acc
        print(f"  {space:<5} surrogate         acc={acc * 100:5.1f}%  ({dt:.0f}s)", flush=True)
    for kind in ("es", "sges"):
        for space in ("full", "hyper"):
            for k in KS:
                acc, dt = run(space, kind, k)
                out["arms"][f"{space} {kind} K={k}"] = acc
                print(f"  {space:<5} {kind:<4} K={k:<4}    acc={acc * 100:5.1f}%  ({dt:.0f}s)", flush=True)

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "shd_jit_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote shd_jit_results.json", flush=True)


if __name__ == "__main__":
    main()
