# Packing sparse + quantized activations: exactness, footprint, crossover

## Title

`spyx.experimental.compress` — bit-plane packing of **quantized** activations
(`pack_nbit` / `packed_quant_dense`) and **sparse + quantized** activations
(`sparse_quant_pack`), generalising the binary-spike `packed_spike_dense`.

## Paper & arXiv/DOI

- **Title:** Spyx utility; the mechanisms are standard. Not a paper claim.
- **Prior art:** bit-plane / bit-slice packing is classic; run-length / mask+value
  (a.k.a. bitmap or "occupancy") sparse formats are standard (CSR/COO cousins);
  microscaling low-bit grids follow the OCP MX spec. The contribution here is a small,
  **exact, JAX-native** activation-packing layer wired into BPTT for spiking / graded
  neurons — the binary case (`packed_spike_dense`) already lived in Spyx; this adds the
  k-bit and sparse axes.
- **Bucket:** new

## Claim under test

Two independent knobs cut activation memory, and the layer is **exact** on grid-quantized
activations:

1. **Quantization axis** — packing at `bits` bits stores `bits/8` B/element, a `32/bits×`
   cut vs fp32; `packed_quant_dense` uses that as the BPTT residual and reproduces the
   naive dense's forward and both gradients (up to fp32 rounding).
2. **Sparsity axis** — a 1-bit occupancy mask + only the nonzero codes costs
   `ceil(N/8) + ceil(nnz·bits/8)` bytes, which **beats dense k-bit packing below a
   density of `(bits-1)/bits`** and loses above it.

## Method

Self-contained; the interesting numbers are byte counts (hardware-independent) and
exactness checks. Three parts:

- **A. Exactness** — `sparse_quant_pack → sparse_quant_unpack` roundtrip is bit-exact,
  and `packed_quant_dense` gradients (w.r.t. activations and weight) match the naive
  `a @ w` on grid-quantized inputs.
- **B. Footprint sweep** — for each `bits ∈ {2,4,8}` × `density ∈ {0.02…0.9}`, the
  *empirical* packed `nbytes` of the dense-k-bit and sparse schemes, checked against the
  analytic `packing_footprint`, and which scheme wins (the crossover).
- **C. BPTT residual** — a real graded `SigmaDelta` hidden activation: fp32 residual
  bytes (what a naive matmul stashes) vs the dense-k-bit and sparse packed forms.

## Spyx modules used

- [`spyx.experimental.compress`](../../../src/spyx/experimental/compress.py) —
  `pack_nbit`/`unpack_nbit`, `packed_quant_dense`, `sparse_quant_pack`/`sparse_quant_unpack`,
  `packing_footprint`
- [`spyx.experimental.SigmaDelta`](../../../src/spyx/experimental/sigma_delta.py), [`spyx.nn`](../../../src/spyx/nn.py)

## How to run

```bash
SPYX_SMOKE=1 uv run python research/new/activation_packing/activation_packing_bench.py
uv run python research/new/activation_packing/activation_packing_bench.py
```

Writes `activation_packing_results.json`.

## Findings

**A. Exact.** Roundtrip is bit-exact for every width; `packed_quant_dense` gradients match
the naive dense to fp32 rounding (`max abs err ≈ 2e-3` on gradients of magnitude ~10²,
i.e. ~1e-5 relative). The correctness tests live in `tests/test_compress.py`.

**B. The crossover is exactly `(bits-1)/bits`, confirmed empirically at N=2²⁰** (bytes):

| bits | density | dense k-bit | sparse mask+code | winner |
| ---: | ---: | ---: | ---: | :--- |
| 4 | 0.10 | 524 288 | 180 272 | **sparse** (2.9× smaller) |
| 4 | 0.25 | 524 288 | 253 356 | **sparse** (2.1×) |
| 4 | 0.75 | 524 288 | 498 484 | **sparse** (just barely) |
| 4 | 0.90 | 524 288 | 571 260 | **dense** |
| 8 | 0.10 | 1 048 576 | 235 272 | **sparse** (4.5×) |
| 8 | 0.90 | 1 048 576 | 1 070 552 | **dense** |

Empirical-vs-analytic winner mismatches: **0/18**. Sparse wins big at low density and at
higher bit-widths (the mask overhead is amortised over fatter codes); dense k-bit wins
once the tensor is nearly full. The `(bits-1)/bits` rule: at 4-bit, mask+value pays off
below 75% density; at 8-bit, below 87.5%.

**C. On a real (dense, random-input) sigma-delta activation** — 91% nonzero, a 6-bit
grid — dense k-bit packing gives **5.3× vs fp32** and edges out sparse (4.9×), exactly as
the crossover predicts for a nearly-full tensor. The sparse scheme's win is realised on
**temporally-redundant** input, where the sigma-delta event density collapses (see
[`../sigma_delta_neuron/`](../sigma_delta_neuron/): 2.9% events) — deep inside the
sparse-wins regime, where `sparse_quant_pack` would give an order-of-magnitude cut.

**Honest caveats.** (1) This is a **memory / transmission** win, not a compute win: on a
dense GPU the pack/unpack is *extra* work and nothing is sparsity-skipped — the payoff is
BPTT activation memory, event-driven transmission, and neuromorphic targets. (2) Exactness
requires the activations to lie on the quantization grid; packing off-grid floats
silently rounds them (fine for graded/spiking events, wrong for arbitrary tensors).
(3) `sparse_quant_pack` is eager (dynamic nonzero count) — for storage/transmission, not a
jit hot loop; `packed_quant_dense`/`pack_nbit` are jit-friendly. (4) First-order VJP only.

## Reproducibility

- **Seeds:** `jax.random.PRNGKey` throughout; `nnx.Rngs(seed)` for the net in part C.
- **JAX / hardware:** part B is byte counts (hardware-independent); parts A/C run on CPU
  or GPU. Device recorded in the JSON.
- **Correctness:** `tests/test_compress.py` (roundtrip exactness, `packed_quant_dense`
  gradient equivalence, footprint crossover).
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** 2026-07-06.
