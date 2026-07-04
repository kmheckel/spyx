"""Pretrain-then-finetune curriculum: cheap reset-free PSU_LIF -> hard-reset LIF.

A hard-reset :class:`spyx.nn.LIF` must be trained with a strictly sequential
``O(T)`` BPTT scan (:func:`spyx.nn.run`). The reset-free
:class:`spyx.nn.PSU_LIF` is the *same* neuron minus the ``spikes * threshold``
reset term, so it shares the exact same trainable parameters (the ``nnx.Linear``
kernels and each neuron's ``beta``) but its membrane is a first-order linear
recurrence that can be scored with an ``O(log T)`` associative scan
(:meth:`PSU_LIF.parallel`). That makes PSU_LIF a *cheap pretrainer*: it drives
forward roughly an order of magnitude faster, but — as the sibling study found —
it destabilises if trained too long standalone (peaks ~11% then collapses on
SHD).

This study uses PSU_LIF as a fast warm-start. We train ``K`` epochs reset-free
via the parallel path, TRANSFER the shared parameters into an identical-shape
hard-reset LIF network, then finetune the remaining ``N-K`` epochs sequentially.
The question: can the curriculum match / beat LIF-from-scratch accuracy at lower
TOTAL wall-clock (pretrain + finetune)?

Related work:
  * sibling study ``../parallel_spiking_neurons/`` — the fair neuron-to-neuron
    LIF vs PSU_LIF vs ResonateFire comparison that motivates this curriculum
    (PSU_LIF is fastest but does not train stably alone).
  * Q-S5 (Abreu, Pedersen, Heckel, Pierro; arXiv:2406.09477, 2024) — quantized
    S5: recurrent weights need high precision while other components compress;
    the same "keep the recurrent dynamics well-behaved, cheapen the rest" theme
    motivates spending expensive sequential training only where it is needed.

Data: SHD via a bit-packed ``.npz`` cache behind ``SHD_CACHE`` (identical
mechanism to the sibling study). For a fast self-check set ``SMOKE=1`` (or pass
``--smoke``) to use synthetic random {0,1} spike tensors of SHD-like shape with
tiny epochs / hidden and no dataset.
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import spyx
import spyx.nn as snn

SMOKE = os.environ.get("SMOKE", "0") == "1" or "--smoke" in sys.argv

if SMOKE:
    SAMPLE_T, CHANNELS, N_CLASSES, HIDDEN = 32, 16, 4, 8
    EPOCHS = int(os.environ.get("EPOCHS", "4"))
    K_SWEEP = (1, 2)
else:
    SAMPLE_T, CHANNELS, N_CLASSES, HIDDEN = 128, 128, 20, 64
    EPOCHS = int(os.environ.get("EPOCHS", "40"))
    K_SWEEP = (5, 10)

LR = float(os.environ.get("LR", "3e-4"))
SEED = int(os.environ.get("SEED", "0"))
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


def load_shd():
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


def load_synthetic():
    """SHD-like random {0,1} spike tensors for the smoke self-check."""
    rng = np.random.default_rng(SEED)
    nb, batch = 2, 8

    def spikes(n):
        return (rng.random((n, batch, SAMPLE_T, CHANNELS)) < 0.1).astype(np.float32)

    def labels(n):
        return rng.integers(0, N_CLASSES, size=(n, batch)).astype(np.int32)

    return (
        jnp.asarray(spikes(nb)),
        jnp.asarray(labels(nb)),
        jnp.asarray(spikes(nb)),
        jnp.asarray(labels(nb)),
    )


def load_data():
    return load_synthetic() if SMOKE else load_shd()


# -------------------------------------------------------------------------- model
def make_neuron(kind, shape, rngs):
    act = spyx.axn.triangular()
    if kind == "LIF":
        return snn.LIF(shape, activation=act, rngs=rngs)
    if kind == "PSU_LIF":
        return snn.PSU_LIF(shape, activation=act, rngs=rngs)
    raise ValueError(kind)


def _leaky_scan(beta, x_TB):
    """Parallel first-order leaky integrator V_t = beta*V_{t-1} + x_t (V_-1=0)."""

    def op(ei, ej):
        a_i, b_i = ei
        a_j, b_j = ej
        return a_j * a_i, a_j * b_i + b_j

    a = jnp.broadcast_to(jnp.clip(beta, 0, 1), x_TB.shape)
    _, v = jax.lax.associative_scan(op, (a, x_TB), axis=0)
    return v


class SHDSNN(nnx.Module):
    """Linear -> neuron -> Linear -> neuron -> Linear -> LI readout.

    ``kind`` selects the hidden neuron (``LIF`` or ``PSU_LIF``). Both share the
    identical trainable-parameter structure (Linear kernels + per-neuron beta +
    LI beta), which is what makes the PSU_LIF -> LIF weight transfer exact.
    """

    def __init__(self, kind, in_dim, hidden, n_classes, *, rngs):
        self.kind = kind
        self.core = snn.Sequential(
            nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs),
            make_neuron(kind, (hidden,), rngs),
            nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs),
            make_neuron(kind, (hidden,), rngs),
            nnx.Linear(hidden, n_classes, use_bias=False, rngs=rngs),
            snn.LI((n_classes,), rngs=rngs),
        )

    def __call__(self, x_BTC):
        """Sequential (hard-reset-capable) forward via spyx.nn.run — O(T)."""
        t, _ = snn.run(self.core, jnp.transpose(x_BTC, (1, 0, 2)))
        return jnp.transpose(t, (1, 0, 2))

    def parallel(self, x_BTC):
        """Reset-free forward using each neuron's O(log T) associative scan.

        Only valid when the hidden neurons expose ``.parallel`` (PSU_LIF). Each
        layer is applied over the whole time-major sequence: Linear over the
        last axis, PSU_LIF via its associative scan, the LI readout via a leaky
        parallel scan.
        """
        x = jnp.transpose(x_BTC, (1, 0, 2))  # (T, B, C)
        for layer in self.core.layers:
            if hasattr(layer, "parallel"):
                x = layer.parallel(x)
            elif isinstance(layer, snn.LI):
                x = _leaky_scan(layer.beta[...], x)
            else:  # stateless (Linear) — applies over the last axis
                x = layer(x)
        return jnp.transpose(x, (1, 0, 2))


# ------------------------------------------------------------- parameter transfer
def transfer_params(src, dst):
    """Copy every trainable param from ``src`` (PSU_LIF net) into ``dst`` (LIF).

    Both nets are :class:`SHDSNN` of identical shape, so their ``nnx.Param``
    state trees are structurally identical (Linear ``kernel`` + neuron ``beta``
    + LI ``beta``) and a filtered ``nnx.update`` transfers them exactly. We then
    assert every Linear kernel and neuron beta actually matches post-transfer.
    """
    nnx.update(dst, nnx.state(src, nnx.Param))

    for ls, ld in zip(src.core.layers, dst.core.layers, strict=True):
        if isinstance(ls, nnx.Linear):
            assert jnp.array_equal(ls.kernel[...], ld.kernel[...]), (
                "Linear kernel mismatch after transfer"
            )
        if hasattr(ls, "beta") and hasattr(ld, "beta"):
            assert jnp.array_equal(ls.beta[...], ld.beta[...]), (
                "neuron/LI beta mismatch after transfer"
            )


# ----------------------------------------------------------------------- training
loss_fn = spyx.fn.integral_crossentropy()
acc_fn = spyx.fn.integral_accuracy()


def make_model(kind):
    return SHDSNN(kind, CHANNELS, HIDDEN, N_CLASSES, rngs=nnx.Rngs(SEED))


def make_opt(model):
    return nnx.Optimizer(model, optax.lion(LR), wrt=nnx.Param)


@nnx.jit
def seq_step(m, o, ob, lb):
    """One sequential (spyx.nn.run) training step — used for LIF."""

    def lf(mm):
        return loss_fn(mm(ob), lb)

    loss, g = nnx.value_and_grad(lf)(m)
    o.update(m, g)
    return loss


@nnx.jit
def par_step(m, o, ob, lb):
    """One parallel-scan training step — used to pretrain PSU_LIF."""

    def lf(mm):
        return loss_fn(mm.parallel(ob), lb)

    loss, g = nnx.value_and_grad(lf)(m)
    o.update(m, g)
    return loss


def test_acc(model, teo, tel):
    return float(
        jnp.mean(
            jnp.stack([acc_fn(model(teo[i]), tel[i])[0] for i in range(teo.shape[0])])
        )
    )


def run_epochs(step, model, opt, tro, trl, n_epochs):
    """Time ``n_epochs`` of ``step`` over the batch axis; warm-compile excluded."""
    if n_epochs <= 0:
        return 0.0
    last = step(model, opt, tro[0], trl[0])  # warm compile (untimed)
    jax.block_until_ready(last)
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        for i in range(tro.shape[0]):
            last = step(model, opt, tro[i], trl[i])
    jax.block_until_ready(last)
    return time.perf_counter() - t0


# ------------------------------------------------------------------- experiments
def baseline(data):
    """LIF from scratch, sequential, N epochs."""
    tro, trl, teo, tel = data
    model = make_model("LIF")
    opt = make_opt(model)
    total_s = run_epochs(seq_step, model, opt, tro, trl, EPOCHS)
    acc = test_acc(model, teo, tel)
    print(
        f"  [baseline]        epochs={EPOCHS:2d}  acc={acc * 100:5.2f}%  "
        f"total={total_s:6.2f}s",
        flush=True,
    )
    return {
        "method": "baseline",
        "K": None,
        "epochs": EPOCHS,
        "pretrain_s": 0.0,
        "finetune_s": total_s,
        "total_s": total_s,
        "test_acc": acc,
    }


def curriculum(K, data):
    """PSU_LIF parallel pretrain (K ep) -> transfer -> LIF finetune (N-K ep)."""
    tro, trl, teo, tel = data

    # cheap reset-free pretraining via the parallel associative scan
    psu = make_model("PSU_LIF")
    psu_opt = make_opt(psu)
    pretrain_s = run_epochs(par_step, psu, psu_opt, tro, trl, K)
    psu_acc = test_acc(psu, teo, tel)

    # transfer shared params into an identical-shape hard-reset LIF net
    lif = make_model("LIF")
    transfer_params(psu, lif)
    lif_opt = make_opt(lif)

    # sequential finetuning of the remaining budget
    finetune_s = run_epochs(seq_step, lif, lif_opt, tro, trl, EPOCHS - K)
    acc = test_acc(lif, teo, tel)
    total_s = pretrain_s + finetune_s
    print(
        f"  [curriculum K={K:2d}]  epochs={EPOCHS:2d}  acc={acc * 100:5.2f}%  "
        f"total={total_s:6.2f}s  (pre={pretrain_s:5.2f}s psu_acc={psu_acc * 100:4.1f}% "
        f"+ fine={finetune_s:5.2f}s)",
        flush=True,
    )
    return {
        "method": "curriculum",
        "K": K,
        "epochs": EPOCHS,
        "pretrain_s": pretrain_s,
        "finetune_s": finetune_s,
        "total_s": total_s,
        "test_acc": acc,
        "psu_pretrain_acc": psu_acc,
    }


def format_table(rows):
    base = next(r for r in rows if r["method"] == "baseline")
    header = (
        f"{'method':<16}{'K':>4}{'epochs':>8}{'pre(s)':>9}"
        f"{'fine(s)':>9}{'total(s)':>10}{'acc(%)':>9}{'speedup':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        speedup = base["total_s"] / r["total_s"] if r["total_s"] > 0 else float("nan")
        kstr = "-" if r["K"] is None else str(r["K"])
        lines.append(
            f"{r['method']:<16}{kstr:>4}{r['epochs']:>8}{r['pretrain_s']:>9.2f}"
            f"{r['finetune_s']:>9.2f}{r['total_s']:>10.2f}"
            f"{r['test_acc'] * 100:>9.2f}{speedup:>8.2f}x"
        )
    return "\n".join(lines)


def main():
    print(f"backend={jax.default_backend()}  device={jax.devices()[0]}", flush=True)
    print(f"mode={'SMOKE' if SMOKE else 'SHD'}  epochs={EPOCHS}  K_sweep={K_SWEEP}\n")

    data = load_data()
    print(
        f"data: train {data[0].shape}  test {data[2].shape}  "
        f"hidden={HIDDEN} classes={N_CLASSES}\n",
        flush=True,
    )

    print("== running experiments ==", flush=True)
    rows = [baseline(data)]
    for K in K_SWEEP:
        rows.append(curriculum(K, data))

    print("\n== results: accuracy vs total wall-clock ==", flush=True)
    print(format_table(rows))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out_path, "w") as f:
        json.dump(
            {"config": {"smoke": SMOKE, "epochs": EPOCHS}, "rows": rows}, f, indent=2
        )
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
