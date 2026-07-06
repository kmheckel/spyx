"""Discriminating F3: does FP4 microscaling separate from ternary/int on a HARD task?

The feasibility study (``../fp4_spiking_qat/``) showed NVFP4/MXFP4 weight-QAT is
lossless on an easy synthetic task — but every arm saturated, so it could not RANK
the formats. This runs the same verified weight-QAT arms on **real SHD** (cached) at
**constrained capacity** (a hidden-width sweep), where fp32 is genuinely challenged
and low precision should actually cost accuracy — to see whether matched-footprint
NVFP4 / MXFP4 beat ternary and int8 once precision bites.

Reuses the sibling study's quant machinery (``true_quant`` / STE / ``train_snn`` /
``evaluate``) by importing it and overriding its task config per hidden width.

    SPYX_SMOKE=1 uv run python research/new/fp4_spiking_qat_hard/hard_shd.py   # tiny, CPU
    ~/.venvs/jax-rocm-0.9.2/bin/python research/new/fp4_spiking_qat_hard/hard_shd.py  # full, GPU
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "fp4_spiking_qat")))
import fp4_spiking_qat as fp4  # noqa: E402  (sibling study; verified quant arms)

import spyx.data as data  # noqa: E402

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))
CHANNELS, SAMPLE_T, N_CLASSES, BATCH = 128, 128, 20, 256
if SMOKE:
    HIDDENS, EPOCHS, SEEDS, NTR, NTE = [64, 32], 3, [0], 4, 4
else:
    HIDDENS, EPOCHS, SEEDS, NTR, NTE = [128, 64, 32], 30, [0, 1], None, None

# None => fp32 baseline (no weight quant); rest are STE-QAT weight formats.
SCHEMES = [None, "int8", "nvfp4", "mxfp4", "ternary"]


def _unpack(o):
    return jnp.unpackbits(jnp.asarray(o), axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)


def load_shd():
    dl = data.SHD_loader(batch_size=BATCH, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=0)
    tr_o, tr_l = dl.prestage("train")
    te_o, te_l = dl.prestage("test")
    ntr = tr_o.shape[0] if NTR is None else min(NTR, tr_o.shape[0])
    nte = te_o.shape[0] if NTE is None else min(NTE, te_o.shape[0])
    tr = ([jnp.transpose(_unpack(tr_o[i]), (1, 0, 2)) for i in range(ntr)],
          [jnp.asarray(tr_l[i]) for i in range(ntr)])
    te = ([jnp.transpose(_unpack(te_o[i]), (1, 0, 2)) for i in range(nte)],
          [jnp.asarray(te_l[i]) for i in range(nte)])
    return tr, te


def _set_cfg(hidden):
    fp4.CHANNELS, fp4.HIDDEN, fp4.N_CLASSES, fp4.SAMPLE_T = CHANNELS, hidden, N_CLASSES, SAMPLE_T
    fp4.EPOCHS = EPOCHS


def _bits(scheme):
    return fp4.EFFECTIVE_BITS.get(scheme or "fp32", 32.0)


def main():
    print(
        f"backend={jax.default_backend()} SMOKE={SMOKE} real-SHD C={CHANNELS} T={SAMPLE_T} "
        f"classes={N_CLASSES} hiddens={HIDDENS} epochs={EPOCHS} seeds={SEEDS} schemes={SCHEMES}",
        flush=True,
    )
    t0 = time.perf_counter()
    train, test = load_shd()
    print(f"loaded SHD: {len(train[0])} train / {len(test[0])} test batches of {BATCH}", flush=True)

    rows = []
    for h in HIDDENS:
        _set_cfg(h)
        print(f"\n[H={h}]", flush=True)
        fp32_acc = None
        for scheme in SCHEMES:
            accs = []
            for seed in SEEDS:
                theta, gd_ev, unravel, rest = fp4.train_snn(train, seed, scheme, None)
                accs.append(fp4.evaluate(theta, gd_ev, unravel, rest, test, scheme))
            acc = float(np.mean(accs))
            std = float(np.std(accs))
            name = scheme or "fp32"
            if scheme is None:
                fp32_acc = acc
            drop = None if fp32_acc is None else fp32_acc - acc
            rows.append({"hidden": h, "scheme": name, "acc": acc, "acc_std": std,
                         "eff_bits": _bits(scheme), "drop_vs_fp32": drop})
            dstr = "" if drop is None else f"  drop={drop * 100:+.1f}pt"
            print(f"    {name:8s} acc={acc * 100:5.1f}%±{std * 100:.1f}  "
                  f"{_bits(scheme):.2f}b{dstr}", flush=True)
    dt = time.perf_counter() - t0

    out = {"config": {"smoke": SMOKE, "device": str(jax.devices()[0]), "channels": CHANNELS,
                      "sample_T": SAMPLE_T, "n_classes": N_CLASSES, "hiddens": HIDDENS,
                      "epochs": EPOCHS, "seeds": SEEDS, "schemes": [s or "fp32" for s in SCHEMES],
                      "wall_s": dt}, "rows": rows}
    with open(os.path.join(_HERE, "hard_shd_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote hard_shd_results.json  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
