# Research backlog

The queue the **scheduled agentic runner** pulls from (see the
[research-study skill](../.claude/skills/research-study.md)). Each run picks the
**top unblocked, unclaimed** item, does one study, opens a PR, and stops for review.

**State:** `ready` (pick me) · `claimed: <branch>` (a run is on it) · `in-review: #<pr>`
(PR open, awaiting you) · `done` (merged) · `blocked: <why>`. Ordered by priority;
the runner takes the highest `ready` item. Add items freely — one claim per item.

## Active track: quantization & efficient architectures

1. **`ready` — Hard-task quantization-aware evolution.** The
   [quant_aware_evolution](new/quant_aware_evolution/) null result came from an easy
   task (every precision hit ~100%). Re-run ES-vs-STE-QAT on a **capacity-constrained**
   setup where nvfp4/ternary actually *degrades* accuracy — real SHD (20 classes, 128
   channels, long T) with a tight hidden width. Claim: the STE-bias gap appears (ES
   ahead) only once quantization costs accuracy. *Note: build + smoke only; flag the
   full SHD/GPU run for a human (needs the ROCm venv + dataset).*

2. **`ready` — matfree vs NVFP4 at matched footprint.** Quantify the
   [substrate-doc](../docs/explanation/substrate-and-the-hardware-lottery.md) claim:
   ternary (~1.58-bit) vs NVFP4 (~4-bit) weight quality at matched *memory footprint*
   and at matched *bit-width*, on a small classifier. Claim: NVFP4 wins accuracy at
   equal bits; ternary wins footprint. Uses `spyx.quant` (both formats now supported).

3. **`ready` — NVFP4 on the binary spike→Linear path.** Because spikes are binary,
   quantizing weights adds no activation-side error. Measure NVFP4-weight QAT accuracy
   vs int8 vs fp32 on a spiking classifier via `spyx.quant.spiking_feedforward_rules`
   extended to nvfp4. Claim: the binary-spike quant win holds at 4-bit float.

## Other candidate questions

4. **`ready` — Widen the portable `.parallel` path.** The
   [pallas_neurons](new/pallas_neurons/) finding was "use associative_scan, not a
   kernel." Survey which neuron types (CuBaLIF, ALIF) can be linearized into an
   associative-scan `.parallel` method, and prototype one. Claim: N more neurons get
   the 2–21× GPU speedup with no accuracy loss.

## Conventions for items

- One item = one falsifiable claim. Keep it small enough for a single study.
- Prefer self-contained synthetic-task smoke validation; mark any dataset-download or
  GPU/full-budget step as **human-gated** — the unattended runner must not do it.
- When an item is `done`, leave it here struck through for one cycle, then move its
  outcome to [FINDINGS.md](FINDINGS.md).
