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

The program's flagship studies, **re-scoped after adversarial verification** (see
[PROGRAM.md](PROGRAM.md) → "What verification changed"). Each carries a paper link +
bucket. **Not picked up by the runner until you move one to a numbered `ready` item
above.** Ordered by verified gap-strength.

- **[novelty · CONFIRMED GAP] Sub-4-bit spiking: NVFP4/MXFP4 QAT + membrane-state quant (F3).**
  The one gap that survived two independent adversarial sweeps: FP4 block-microscaling has
  **not** been applied to SNNs, and Spyx uniquely has the formats. Weight- + membrane-state
  quantization (SQUAT gap, arXiv:2404.19668) on SSC, always vs a quantized-ANN baseline.
  Nearest boundary AQ4SViT (arXiv:2606.15523) uses integer/search quant, not FP4. **Headline.**
- **[extension] NeuroBench/NSA honest energy accounting in `spyx.bench` (F4).**
  Hardware-agnostic SOP + hardware-aware memory-inclusive + quantized-ANN-equivalent energy;
  wire NSA's 7 temporal tasks (arXiv:2505.22035). Refs: NeuroBench (Nat Commun 2025,
  s41467-025-56739-4), Reconsidering SNN Energy (arXiv:2409.08290). Credibility infra.
  *(Scaffolded on `research/honest-energy`.)*
- **[extension] CONTROLLED spiking-SSM vs dense-SSM on SHD/SSC (F2′).** NOT "unreported" —
  SiLIF (arXiv:2506.06374) and S5-RF (arXiv:2504.00719) already print numbers; the residual
  gap is an *apples-to-apples* comparison (identical pipeline, comparison as the contribution).
  Implement SpikingSSM's SDN (arXiv:2408.14909) in Spyx's `ssm`; baselines Event-SSM
  (arXiv:2404.18508) / S7 (arXiv:2410.03464); cite SiLIF/S5-RF as prior partial comparisons.
- **[replication/extension] JAX-native associative-scan neuron benchmark (F1′).** Reframed:
  reset-preserving parallel neurons are prior art (FPT arXiv:2506.12087; Parallel Training in
  SNNs arXiv:2602.01133, 25.6×; Bullet Trains arXiv:2603.13283; snnTorch `AssociativeLeaky`).
  Deliver a clean JAX benchmark vs sequential LIF / PSN and reproduce the dynamic-decay neuron
  (2602.01133) vs `PSU_LIF`.
- **[replication] Reproduce a headline in Spyx (F5).** QKFormer (arXiv:2403.16552, 85.65%)
  or SpikingSSM LRA (arXiv:2408.14909); Q-S5 (arXiv:2406.09477, Heckel co-author) as the QS5
  reference. Validates the stack.

## Conventions for items

- One item = one falsifiable claim. Keep it small enough for a single study.
- Prefer self-contained synthetic-task smoke validation; mark any dataset-download or
  GPU/full-budget step as **human-gated** — the unattended runner must not do it.
- When an item is `done`, leave it here struck through for one cycle, then move its
  outcome to [FINDINGS.md](FINDINGS.md).
