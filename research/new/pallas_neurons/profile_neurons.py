"""Where does SNN wall-clock actually go — the matmul, or the neuron scan?

Motivation for the Pallas-neurons question (issue #24): a fused Pallas kernel for
neuron dynamics only pays off if the **neuron update** is the bottleneck. But the
neuron step is elementwise O(H) while the preceding `Linear` is O(H^2), so which
dominates is *regime-dependent* — small/wide, short/long. This harness measures the
crossover before anyone writes a kernel.

Component ablation at matched (hidden width H, timesteps T, batch B), via
`spyx.bench` (median latency, XLA-cost FLOPs, peak mem):

- **linear**      `Sequential(Linear(H,H))` — matmul per timestep, no neuron.
- **lif_scan**    `LIF((H,))` — the sequential `jax.lax.scan` neuron (bench drives
                   it with `spyx.nn.run`; LIF has no `.parallel`).
- **psu_parallel** `PSU_LIF((H,))` — the *associative-scan* neuron (bench auto-uses
                   its `.parallel` method). The portable parallel alternative to a
                   Pallas kernel, already in the library.
- **linear_lif**  `Sequential(Linear(H,H), LIF((H,)))` — the realistic layer.

Read-out per regime:
- `lif / linear` > 1  → the neuron scan dominates → a fused Pallas/parallel kernel
  is worth it here. < 1 → the matmul dominates → a faster neuron barely moves
  wall-clock (optimise the Linear instead).
- `lif_scan / psu_parallel` → what the *portable* associative-scan already buys,
  for the (linearizable) neurons that can use it.

Device caveat: the crossover shifts with hardware — a GPU's higher matmul
throughput pushes the neuron-dominated regime to larger H than CPU. Run this on the
target device; the JSON records `backend`. Writes `profile_results.json`.
"""

from __future__ import annotations

import json
import os
import time

import jax
from flax import nnx

import spyx.bench as bench
import spyx.nn as snn

HIDDENS = [int(h) for h in os.environ.get("HIDDENS", "64,256,1024").split(",")]
SEQLENS = [int(t) for t in os.environ.get("SEQLENS", "64,256,1024").split(",")]
BATCH = int(os.environ.get("BATCH", "64"))
N_ITERS = int(os.environ.get("N_ITERS", "20"))
N_WARMUP = int(os.environ.get("N_WARMUP", "3"))


def components(h):
    """Fresh-module thunks so each benchmark builds its own params."""
    return {
        "linear": lambda: snn.Sequential(
            nnx.Linear(h, h, use_bias=False, rngs=nnx.Rngs(0))
        ),
        "lif_scan": lambda: snn.LIF((h,), rngs=nnx.Rngs(0)),
        "psu_parallel": lambda: snn.PSU_LIF((h,), rngs=nnx.Rngs(0)),
        "linear_lif": lambda: snn.Sequential(
            nnx.Linear(h, h, use_bias=False, rngs=nnx.Rngs(0)),
            snn.LIF((h,), rngs=nnx.Rngs(0)),
        ),
    }


def _verdict(ratio):
    if ratio >= 1.5:
        return "NEURON-BOUND — fused kernel worth it"
    if ratio >= 0.67:
        return "balanced — profile the specific model"
    return "MATMUL-BOUND — neuron kernel won't move wall-clock"


def main():
    backend = jax.default_backend()
    dev = str(jax.devices()[0])
    print(
        f"backend={backend} device={dev}  B={BATCH} iters={N_ITERS}  "
        f"H={HIDDENS} T={SEQLENS}",
        flush=True,
    )
    print(
        "  (crossover is device-dependent; a GPU pushes NEURON-BOUND to larger H)\n",
        flush=True,
    )

    t0 = time.perf_counter()
    rows = []
    for h in HIDDENS:
        results = bench.compare(
            components(h),
            (h,),
            seq_lens=SEQLENS,
            batch=BATCH,
            n_warmup=N_WARMUP,
            n_iters=N_ITERS,
            backward=True,
        )
        # index by (name, T)
        idx = {(r.name, r.seq_len): r for r in results}
        for t in SEQLENS:
            lin = idx[("linear", t)]
            lif = idx[("lif_scan", t)]
            psu = idx[("psu_parallel", t)]
            comb = idx[("linear_lif", t)]
            neuron_vs_matmul = lif.fwd_latency_ms / max(lin.fwd_latency_ms, 1e-9)
            neuron_share = lif.fwd_latency_ms / max(comb.fwd_latency_ms, 1e-9)
            parallel_speedup = lif.fwd_latency_ms / max(psu.fwd_latency_ms, 1e-9)
            fb = {
                k: idx[(k, t)].fwd_bwd_latency_ms
                for k in ("linear", "lif_scan", "psu_parallel", "linear_lif")
            }
            neuron_vs_matmul_bwd = (
                (fb["lif_scan"] / fb["linear"]) if fb["linear"] else None
            )
            row = {
                "H": h,
                "T": t,
                "batch": BATCH,
                "fwd_ms": {
                    "linear": lin.fwd_latency_ms,
                    "lif_scan": lif.fwd_latency_ms,
                    "psu_parallel": psu.fwd_latency_ms,
                    "linear_lif": comb.fwd_latency_ms,
                },
                "fwd_bwd_ms": fb,
                "peak_mem_mb": {
                    "lif_scan": lif.peak_mem_mb,
                    "psu_parallel": psu.peak_mem_mb,
                    "linear_lif": comb.peak_mem_mb,
                },
                "neuron_vs_matmul_fwd": neuron_vs_matmul,
                "neuron_vs_matmul_fwd_bwd": neuron_vs_matmul_bwd,
                "neuron_share_of_combined_fwd": neuron_share,
                "parallel_speedup_scan_over_assoc": parallel_speedup,
                "verdict_fwd": _verdict(neuron_vs_matmul),
            }
            rows.append(row)
            print(
                f"H={h:<5} T={t:<5} | linear {lin.fwd_latency_ms:7.2f}ms  "
                f"lif_scan {lif.fwd_latency_ms:7.2f}ms  psu {psu.fwd_latency_ms:7.2f}ms "
                f"| neuron/matmul {neuron_vs_matmul:5.2f}x  "
                f"assoc-speedup {parallel_speedup:5.2f}x  | {_verdict(neuron_vs_matmul)}",
                flush=True,
            )
    dt = time.perf_counter() - t0

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "profile_results.json"), "w") as f:
        json.dump(
            {
                "config": {
                    "backend": backend,
                    "device": dev,
                    "hiddens": HIDDENS,
                    "seqlens": SEQLENS,
                    "batch": BATCH,
                    "n_iters": N_ITERS,
                    "wall_s": dt,
                },
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nwrote profile_results.json  ({dt:.0f}s)", flush=True)

    # one-line takeaway: the largest-H, longest-T regime and the smallest one
    neuron_bound = [r for r in rows if r["neuron_vs_matmul_fwd"] >= 1.0]
    if neuron_bound:
        print(
            f"NEURON-BOUND in {len(neuron_bound)}/{len(rows)} regimes "
            f"(e.g. H={neuron_bound[0]['H']} T={neuron_bound[0]['T']}): "
            "a fused Pallas / associative-scan kernel can help there.",
            flush=True,
        )
    else:
        print(
            "MATMUL-BOUND in every tested regime on this device: optimise the "
            "Linear (or shrink it) before writing a neuron kernel.",
            flush=True,
        )


if __name__ == "__main__":
    main()
