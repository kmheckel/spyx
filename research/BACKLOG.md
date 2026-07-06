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

### Neuron models to add (verified prior art — implement + cite)

Scoped via the novelty-verification workflow; all are **extensions** (published methods,
mostly PyTorch-only — Spyx's contribution is the first clean JAX/Flax impl + benchmark +
the associative-scan/SSM/NIR unification). Citations confirmed against arXiv.

- **[extension] Reset-preserving parallel LIF (neuron1).** A K=3 fixed-point iteration
  (FPT) wrapping Spyx's existing `_leaky_associative_op` scan reproduces the sequential
  hard-reset `nn.LIF` spike train to a small tolerance while running in O(log T) — closing
  the reset-free-only parallel gap (`PSU_LIF` drops the reset by design). First JAX/Flax
  reset-PRESERVING parallel neuron: add a `.parallel` fixed-point method to `nn.LIF`,
  benchmark numerical match + speedup vs sequential and `PSU_LIF`. Prior art (all PyTorch;
  JAX parallel neurons are reset-free): FPT (arXiv:2506.12087), Bullet Trains (arXiv:2603.13283,
  ICML 2026), dynamic-decay (arXiv:2602.01133, ICLR 2026), Revisiting Reset (arXiv:2504.17751).
  Spyx: `nn`/experimental, `bench`. Cost: smoke CPU; GPU speedup human-gated. *(= flagship F1′.)*
- **[extension] PMSN — parallel multi-compartment neuron (neuron2).** A JAX PMSN (n
  complex-diagonal compartments + parallel reset via `jax.lax.associative_scan`) reproduces
  the paper's SHD 94.25% / PS-MNIST 97.16% in O(log T) and beats single-compartment
  `phasor.ResonateFire` / `PSU_LIF` on a two-timescale task. First JAX impl (official is
  PyTorch), re-derived in scan form to unify with `spyx.ssm`. Ref: PMSN, Chen et al.
  arXiv:2408.14917 (OpenReview FlH6VB5sJN). Spyx: experimental (`ssm`+`phasor`+`nn`). Cost:
  smoke CPU; SHD/PS-MNIST full runs human-gated.
- **[extension] Sigma-delta / graded-spike neuron (neuron3).** On temporally-redundant
  input, a Spyx sigma-delta neuron that transmits only the quantized *change* cuts events/SOP
  ≥2× vs rate-coded LIF at matched accuracy. First clean JAX/Flax graded neuron (the sigma
  accumulator is a linear recurrence → `.parallel` scan), + a proposed NIR graded-spike
  primitive (NIR has none), + honest sparsity benchmark. Refs: O'Connor & Welling ICLR 2017
  (arXiv:1611.02024); Shrestha et al. ICASSP 2024 (arXiv:2310.03251); NIR (arXiv:2311.14641);
  Intel `lava-dl` `slayer.neuron.sigma_delta` reference. Spyx: `axn`/`nn`/`nir`/`bench`. Cost:
  smoke CPU; video/audio sparsity run human-gated.
- **[extension] Parallel Resonate-and-Fire + R&F-as-SSM (neuron4).** Add PRF's decoupled
  differentiable reset and S5-RF's HiPPO/S5 frequency-decay init to Spyx's reset-free
  `phasor.ResonateFire`; the reset stays `associative_scan`-equivalent to its sequential
  reference AND beats the reset-free baseline on a long-range task at O(log T). Prior art:
  PRF (arXiv:2410.03530, PyTorch/SpikingJelly), S5-RF (arXiv:2504.00719, JAX+Equinox). Spyx:
  `phasor`/`ssm`/`nir`. **Connects flagship F2′ (spiking SSMs).** Cost: smoke + GPU/dataset.
- **[extension] Learnable synaptic delays — DCLS DelayLayer (neuron5).** A JAX DelayLayer
  (per-synapse learnable temporal-kernel positions, Gaussian interp annealed to one discrete
  delay) between feedforward LIF layers reaches within ~1–2% of feedforward SHD ~95% / SSC
  ~80% *without recurrence*. A synaptic mechanism, not a neuron. First JAX impl (method is
  PyTorch-only) + NIR Delay-node export. Refs: Hammouamri et al. ICLR 2024 (arXiv:2306.17670);
  DCLS (arXiv:2112.03740). Spyx: new `experimental` delay module + `data`/`nir`/`bench`. Cost:
  dataset (SHD/SSC) + GPU; smoke = shape/gradient test.

## Conventions for items

- One item = one falsifiable claim. Keep it small enough for a single study.
- Prefer self-contained synthetic-task smoke validation; mark any dataset-download or
  GPU/full-budget step as **human-gated** — the unattended runner must not do it.
- When an item is `done`, leave it here struck through for one cycle, then move its
  outcome to [FINDINGS.md](FINDINGS.md).
