"""Packing sparse + quantized activations: exactness, footprint, and the crossover.

Binary spikes are the ``k=1`` corner of activation packing. This study measures the
two-axis generalisation in ``spyx.experimental.compress``:

  * **quantization** — pack activations at ``bits`` bits with ``pack_nbit`` (bit-plane
    packing); ``packed_quant_dense`` is the k-bit BPTT residual (a ``32/bits×`` cut of the
    dominant activation memory), exact for grid-quantized activations.
  * **sparsity** — store a 1-bit occupancy mask plus only the nonzero codes
    (``sparse_quant_pack``). Footprint ``ceil(N/8) + ceil(nnz·bits/8)`` bytes, which beats
    dense k-bit packing below the ``(bits-1)/bits`` density crossover.

Reports, all from real arrays (no hand-waved numbers):
  A. exactness — roundtrip is bit-exact and ``packed_quant_dense`` grads == naive dense.
  B. footprint sweep — empirical packed ``nbytes`` vs the analytic ``packing_footprint``
     model over density × bit-width, and which scheme wins where (the crossover).
  C. BPTT residual saving on a real graded sigma-delta activation.

Honest framing: this is a **memory** win (activation residual / storage / transmission),
not a compute win — on a dense GPU the pack/unpack is extra work and nothing is
sparsity-skipped; the payoff is BPTT memory, event-driven transmission, and neuromorphic
targets. Exactness holds only for grid-quantized inputs.

    SPYX_SMOKE=1 uv run python research/new/activation_packing/activation_packing_bench.py
    uv run python research/new/activation_packing/activation_packing_bench.py
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import spyx.nn as snn
from spyx.experimental import SigmaDelta
from spyx.experimental.compress import (
    pack_nbit,
    packed_quant_dense,
    packing_footprint,
    sparse_quant_pack,
    sparse_quant_unpack,
)

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))
N_ELEM = 4096 if SMOKE else 1 << 20
DENSITIES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9]
BITWIDTHS = [2, 4, 8]
STEP = 0.25


def _grid(key, shape, bits, density):
    """A sparse, grid-quantized tensor: `density` fraction nonzero, values on the
    symmetric `step`-spaced grid representable in `bits` signed levels."""
    hi = 1 << (bits - 1)
    codes = jax.random.randint(key, shape, -hi + 1, hi).astype(jnp.float32)
    keep = jax.random.uniform(jax.random.fold_in(key, 1), shape) < density
    return jnp.where(keep, codes * STEP, 0.0)


def part_a_exactness():
    """Roundtrip is bit-exact; packed_quant_dense grads match the naive dense."""
    ok_roundtrip, max_grad_err = True, 0.0
    for bits in BITWIDTHS:
        x = _grid(jax.random.PRNGKey(bits), (8, 64), bits, 0.3)
        mp, cp, meta = sparse_quant_pack(x, bits, STEP)
        if not bool(jnp.array_equal(sparse_quant_unpack(mp, cp, meta), x)):
            ok_roundtrip = False
        # gradient equivalence of the k-bit BPTT dense on grid-quantized acts
        w = jax.random.normal(jax.random.PRNGKey(bits + 9), (64, 10))
        tgt = jax.random.normal(jax.random.PRNGKey(bits + 99), (8, 10))

        def lp(a, w):
            return jnp.sum((packed_quant_dense(a, w, bits, STEP) - tgt) ** 2)

        def ln(a, w):
            return jnp.sum((a @ w - tgt) ** 2)

        gp = jax.grad(lp, argnums=(0, 1))(x, w)
        gn = jax.grad(ln, argnums=(0, 1))(x, w)
        max_grad_err = max(
            max_grad_err,
            float(jnp.max(jnp.abs(gp[0] - gn[0]))),
            float(jnp.max(jnp.abs(gp[1] - gn[1]))),
        )
    return {"roundtrip_bit_exact": ok_roundtrip, "max_grad_abs_err": max_grad_err}


def part_b_footprint():
    """Empirical packed nbytes vs the analytic model; report the winning scheme."""
    rows = []
    for bits in BITWIDTHS:
        for d in DENSITIES:
            x = _grid(jax.random.PRNGKey(int(d * 1000) + bits), (N_ELEM,), bits, d)
            # empirical bytes actually occupied by each packed representation
            dense_bits = int(
                pack_nbit(jnp.zeros_like(x).astype(jnp.uint32), bits).nbytes
            )
            mp, cp, _ = sparse_quant_pack(x, bits, STEP)
            sparse_bytes = int(mp.nbytes) + int(cp.nbytes)
            model = packing_footprint(N_ELEM, bits, float(jnp.mean(x != 0)))
            emp_best = "sparse" if sparse_bytes < dense_bits else "dense"
            rows.append(
                {
                    "bits": bits,
                    "density": d,
                    "dense_kbit_bytes": dense_bits,
                    "sparse_bytes": sparse_bytes,
                    "model_best": model["best"],
                    "empirical_best": emp_best,
                    "crossover_density": model["crossover_density"],
                }
            )
    return rows


def part_c_bptt_residual():
    """Real graded sigma-delta activation: fp32 residual bytes vs packed k-bit codes."""
    T, B, C, H = (8, 8, 16, 24) if SMOKE else (60, 64, 64, 128)
    net = snn.Sequential(
        nnx.Linear(C, H, rngs=nnx.Rngs(0)),
        SigmaDelta((H,), step=STEP, rngs=nnx.Rngs(1)),
        nnx.Linear(H, 4, rngs=nnx.Rngs(2)),
        snn.LI((4,), rngs=nnx.Rngs(3)),
    )
    x = jax.random.normal(jax.random.PRNGKey(7), (T, B, C))
    lin, neuron = net.layers[0], net.layers[1]
    pre = jnp.einsum("tbc,cd->tbd", x, lin.kernel[...]) + lin.bias[...]

    def scan_step(V, xt):
        s, V = neuron(xt, V)
        return V, s

    _, events = jax.lax.scan(scan_step, neuron.initial_state(B), pre)  # (T,B,H) graded
    density = float(jnp.mean(events != 0))
    # what a naive matmul stashes for the backward residual (fp32) vs the packed forms
    fp32_bytes = int(events.astype(jnp.float32).nbytes)
    # graded events sit on the `step` grid; find the bits needed to index the levels used
    codes = jnp.round(events / STEP).astype(jnp.int32)
    span = int(jnp.max(codes) - jnp.min(codes)) + 1
    bits = max(1, int(np.ceil(np.log2(max(span, 2)))))
    dense_kbit_bytes = int(
        pack_nbit(jnp.zeros_like(events).astype(jnp.uint32), bits).nbytes
    )
    mp, cp, _ = sparse_quant_pack(events, bits, STEP)
    sparse_bytes = int(mp.nbytes) + int(cp.nbytes)
    return {
        "shape_TBH": [T, B, H],
        "event_density": density,
        "grid_bits": bits,
        "fp32_residual_bytes": fp32_bytes,
        "dense_kbit_bytes": dense_kbit_bytes,
        "sparse_bytes": sparse_bytes,
        "dense_vs_fp32_x": fp32_bytes / max(dense_kbit_bytes, 1),
        "sparse_vs_fp32_x": fp32_bytes / max(sparse_bytes, 1),
    }


def main():
    print(
        f"backend={jax.default_backend()} SMOKE={SMOKE} N={N_ELEM} step={STEP}",
        flush=True,
    )

    a = part_a_exactness()
    print(
        f"\n[A] exactness: roundtrip_bit_exact={a['roundtrip_bit_exact']} "
        f"max_grad_abs_err={a['max_grad_abs_err']:.2e}",
        flush=True,
    )

    b = part_b_footprint()
    print("\n[B] footprint sweep (bytes; winner in bold sense):", flush=True)
    print(
        f"    {'bits':>4} {'density':>8} {'dense_kbit':>11} {'sparse':>10} {'winner':>8}",
        flush=True,
    )
    mism = 0
    for r in b:
        # model_best is like 'sparse_mask+4bit' or 'dense_4bit'; empirical_best is
        # 'sparse'/'dense'. They should always agree (the analytic model is exact).
        if r["empirical_best"] not in r["model_best"]:
            mism += 1
        print(
            f"    {r['bits']:>4} {r['density']:>8.2f} {r['dense_kbit_bytes']:>11} "
            f"{r['sparse_bytes']:>10} {r['empirical_best']:>8}",
            flush=True,
        )
    print(
        f"    empirical-vs-model winner mismatches: {mism} (crossover at (bits-1)/bits)",
        flush=True,
    )

    c = part_c_bptt_residual()
    print(
        f"\n[C] BPTT residual on real sigma-delta activation "
        f"(density={c['event_density'] * 100:.1f}%, {c['grid_bits']}-bit grid):",
        flush=True,
    )
    print(f"    fp32 residual   : {c['fp32_residual_bytes']:>10} B", flush=True)
    print(
        f"    dense k-bit pack: {c['dense_kbit_bytes']:>10} B  ({c['dense_vs_fp32_x']:.1f}× vs fp32)",
        flush=True,
    )
    print(
        f"    sparse mask+code: {c['sparse_bytes']:>10} B  ({c['sparse_vs_fp32_x']:.1f}× vs fp32)",
        flush=True,
    )

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "activation_packing_results.json"), "w") as f:
        json.dump(
            {
                "config": {
                    "smoke": SMOKE,
                    "device": str(jax.devices()[0]),
                    "n_elem": N_ELEM,
                    "step": STEP,
                    "densities": DENSITIES,
                    "bitwidths": BITWIDTHS,
                },
                "exactness": a,
                "footprint_sweep": b,
                "bptt_residual": c,
            },
            f,
            indent=2,
        )
    print("\nwrote activation_packing_results.json", flush=True)


if __name__ == "__main__":
    main()
