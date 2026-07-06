# Research backlog

The queue the **agentic runner** pulls from (see the
[research-study skill](../.claude/skills/research-study.md)). Each run picks the
**top unblocked, unclaimed** item, does one study, opens a PR, and stops for review.

**State:** `ready` (pick me) · `claimed: <branch>` (a run is on it) · `in-review: #<pr>`
(PR open, awaiting you) · `done` (merged) · `blocked: <why>`. Ordered by priority;
the runner takes the highest `ready` item. Add items freely — one claim per item.

**Bucket** (the taxonomy in [README.md](README.md)): `replication` (reproduce a
paper's claim in Spyx) · `extension` (push a method somewhere new) · `novelty` (a gap
Spyx is suited to fill). [`/research-scout`](../.claude/skills/research-scout.md)
proposes buckets from the literature into **Candidates** below; you triage them to
`ready`.

## Active track: quantization & efficient architectures

1. **`in-review: #66` — Hard-task quantization-aware evolution.** The
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

## Candidates (from scouting — triage before promoting to `ready`)

Proposed from the 2024–2026 field survey behind [PROGRAM.md](PROGRAM.md); each carries
a paper link + bucket. **Not picked up by the runner until you move one to a numbered
`ready` item above.** These are the program's flagship studies (F1–F5).

- **[novelty · Pillar 1] Parallel LIF with EXACT reset in JAX (F1).** Implement an
  associative-scan LIF that preserves the reset nonlinearity via a fixed-point / parallel
  solver, extending `PSU_LIF`; benchmark throughput×accuracy vs sequential LIF and PSN at
  long T. The field's central reset↔parallel-scan tension, with no clean public JAX impl.
  Refs: FPT (arXiv:2506.12087), SPikE-SSM/PMBC (arXiv:2410.17268), Revisiting Reset
  (arXiv:2504.17751), PSN (arXiv:2304.12760). Verify FPT's unpublished wall-clock numbers.
- **[novelty · Pillar 2] Honest spiking-SSM vs dense-SSM on SHD/SSC/LRA (F2).** Implement
  SpikingSSM's surrogate-dynamic-network (arXiv:2408.14909) or SPikE-SSM PMBC in Spyx's
  `ssm`; run the unreported head-to-head vs dense S5 and Event-SSM/S7 (arXiv:2404.18508,
  2410.03464). The decisive comparison the literature is missing.
- **[novelty · Pillar 3] Sub-4-bit spiking: NVFP4/MXFP4 QAT + membrane-state quant (F3).**
  First application of FP4 microscaling to SNNs (unclaimed). Weight- and membrane-state
  quantization (SQUAT gap, arXiv:2404.19668) on SSC, always with a quantized-ANN baseline.
  Spyx already has the formats via `spyx.quant`.
- **[extension · Pillar 4] NeuroBench-style honest energy accounting in `spyx.bench` (F4).**
  Add hardware-agnostic SOP + hardware-aware memory-inclusive + quantized-ANN-equivalent
  energy reporting. Refs: NeuroBench (Nat Commun 2025, s41467-025-56739-4), Reconsidering
  SNN Energy (arXiv:2409.08290). Credibility infrastructure for every other study.
- **[replication · Pillar 2/4] Reproduce a headline in Spyx (F5).** e.g. SpikingSSM LRA
  (arXiv:2408.14909) or QKFormer sparsity (arXiv:2403.16552) — validates the stack and
  seeds the comparisons in F2/F4.

## Conventions for items

- One item = one falsifiable claim. Keep it small enough for a single study.
- Prefer self-contained synthetic-task smoke validation; mark any dataset-download or
  GPU/full-budget step as **human-gated** — the unattended runner must not do it.
- When an item is `done`, leave it here struck through for one cycle, then move its
  outcome to [FINDINGS.md](FINDINGS.md).
