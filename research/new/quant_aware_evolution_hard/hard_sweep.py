"""Does the STE-bias gap appear once quantization actually *costs* accuracy?

Extension of ``../quant_aware_evolution/``. That study found **no** STE-bias gap on
an easy synthetic task — fp32 hit ~99.5 % and quantization (even ternary) barely
dented it, so there was no bias for gradient-free ES to exploit. The natural next
question, and this study's **claim under test**: the gap (ES reaching a lower *true*
quantized-forward loss than STE-QAT) should emerge **only when the task is hard /
capacity is tight enough that quantization drops accuracy** — and should grow with
that accuracy drop.

Method: sweep a difficulty axis — more classes, tighter hidden width (both multiples
of 16 so nvfp4's contraction-axis tiling is valid) — so quantization has less slack
to hide in. At each point we reuse the **verified 3-arm machinery** of the sibling
study (fp32 / STE-QAT / ES, all scored on the identical true quantized forward) by
importing it and overriding its task config, and report per precision: the fp32→quant
accuracy drop and the STE-bias gap = STE-QAT true loss − ES true loss. Hypothesis
holds if the gap turns positive (ES ahead) where the drop is large.

    SPYX_SMOKE=1 uv run python research/new/quant_aware_evolution_hard/hard_sweep.py
    uv run python research/new/quant_aware_evolution_hard/hard_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "quant_aware_evolution")))
import quant_aware_evolution as qae  # noqa: E402  (sibling study; verified arms)

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))

# Points: (label, channels, hidden, n_classes). Single difficulty knob = hidden
# width (capacity), at a fixed class + channel count, so fp32 stays solvable while
# quantization loses room to hide. Channels & hidden are multiples of 16 (nvfp4
# tiles the contraction axis in blocks of 16).
if SMOKE:
    POINTS = [("H32", 32, 32, 4), ("H16", 32, 16, 4)]
    SEEDS, POP, GENS, EPOCHS, NTRAIN, BATCH, T = [0], 16, 20, 6, 32, 16, 10
else:
    POINTS = [("H64", 64, 64, 8), ("H32", 64, 32, 8), ("H16", 64, 16, 8)]
    SEEDS, POP, GENS, EPOCHS, NTRAIN, BATCH, T = [0, 1], 64, 150, 15, 128, 32, 16

PRECISIONS = ["nvfp4", "ternary"]


def _set_cfg(channels, hidden, n_classes):
    """Override the sibling module's task globals for one difficulty point."""
    qae.CHANNELS, qae.HIDDEN = channels, hidden
    qae.N_CLASSES, qae.SAMPLE_T = n_classes, T
    qae.N_TRAIN, qae.BATCH, qae.EPOCHS = NTRAIN, BATCH, EPOCHS
    qae.POP, qae.GENERATIONS, qae.SEEDS = POP, GENS, SEEDS


def _run_point(channels, hidden, n_classes):
    _set_cfg(channels, hidden, n_classes)

    fp_accs = []
    for seed in SEEDS:
        train, test = qae.synthetic_data(seed)
        theta, gd, un, rest = qae.train_sgd(train, seed, scheme=None)
        _, acc = qae.evaluate(theta, gd, un, rest, test, scheme=None)
        fp_accs.append(acc)
    fp_acc = float(np.mean(fp_accs))

    out = {
        "channels": channels,
        "hidden": hidden,
        "n_classes": n_classes,
        "fp32_acc": fp_acc,
        "precisions": {},
    }
    for prec in PRECISIONS:
        ste_l, ste_a, es_l, es_a = [], [], [], []
        for seed in SEEDS:
            train, test = qae.synthetic_data(seed)
            theta, gd, un, rest = qae.train_sgd(train, seed, scheme=prec)
            loss, acc = qae.evaluate(theta, gd, un, rest, test, scheme=prec)
            ste_l.append(loss)
            ste_a.append(acc)

            theta, gd, un, rest, _ = qae.train_es(train, seed, scheme=prec)
            loss, acc = qae.evaluate(theta, gd, un, rest, test, scheme=prec)
            es_l.append(loss)
            es_a.append(acc)
        ste_loss, ste_acc = float(np.mean(ste_l)), float(np.mean(ste_a))
        es_loss, es_acc = float(np.mean(es_l)), float(np.mean(es_a))
        out["precisions"][prec] = {
            "ste_loss": ste_loss,
            "ste_acc": ste_acc,
            "es_loss": es_loss,
            "es_acc": es_acc,
            "quant_acc_drop": fp_acc - ste_acc,  # cost of quantization under QAT
            "ste_bias_gap": ste_loss - es_loss,  # + = ES reaches lower true loss
        }
    return out


def main():
    print(
        f"backend={jax.default_backend()} SMOKE={SMOKE} points={[p[0] for p in POINTS]} "
        f"seeds={SEEDS} pop={POP} gens={GENS}",
        flush=True,
    )
    t0 = time.perf_counter()
    rows = []
    for label, c, h, k in POINTS:
        r = _run_point(c, h, k)
        r["label"] = label
        rows.append(r)
        print(
            f"\n[{label}] C={c} H={h} K={k}  fp32_acc={r['fp32_acc'] * 100:.1f}%",
            flush=True,
        )
        for prec in PRECISIONS:
            p = r["precisions"][prec]
            print(
                f"    {prec:8s} STE acc={p['ste_acc'] * 100:5.1f}% loss={p['ste_loss']:.4f} | "
                f"ES acc={p['es_acc'] * 100:5.1f}% loss={p['es_loss']:.4f} | "
                f"quant_drop={p['quant_acc_drop'] * 100:+5.1f}pt  "
                f"STE-bias gap={p['ste_bias_gap']:+.4f} "
                f"({'ES lower' if p['ste_bias_gap'] > 0 else 'STE lower'})",
                flush=True,
            )
    dt = time.perf_counter() - t0

    # Hypothesis read-out: does the gap track the quant accuracy drop?
    print("\n=== gap vs quant accuracy drop (per precision) ===", flush=True)
    for prec in PRECISIONS:
        pairs = [
            (
                r["precisions"][prec]["quant_acc_drop"],
                r["precisions"][prec]["ste_bias_gap"],
            )
            for r in rows
        ]
        drops = np.array([d for d, _ in pairs])
        gaps = np.array([g for _, g in pairs])
        corr = (
            float(np.corrcoef(drops, gaps)[0, 1])
            if len(drops) > 1 and drops.std() > 0 and gaps.std() > 0
            else float("nan")
        )
        print(
            f"  {prec:8s} drops={[f'{d * 100:+.1f}pt' for d in drops]}  "
            f"gaps={[f'{g:+.4f}' for g in gaps]}  corr(drop,gap)={corr:+.2f}",
            flush=True,
        )

    out = {
        "config": {
            "smoke": SMOKE,
            "device": str(jax.devices()[0]),
            "points": [
                {"label": p[0], "channels": p[1], "hidden": p[2], "n_classes": p[3]}
                for p in POINTS
            ],
            "seeds": SEEDS,
            "pop": POP,
            "generations": GENS,
            "epochs": EPOCHS,
            "precisions": PRECISIONS,
            "wall_s": dt,
        },
        "rows": rows,
    }
    # SMOKE writes a separate file so a plumbing check never clobbers real results.
    out_name = "hard_sweep_results_smoke.json" if SMOKE else "hard_sweep_results.json"
    with open(os.path.join(_HERE, out_name), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_name}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
