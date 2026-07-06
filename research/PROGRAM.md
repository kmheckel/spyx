# Spyx research program

A standing research agenda for Spyx, built from a 2024–2026 field survey (frameworks,
training methods, efficiency/hardware, applications) **then adversarially verified
against primary sources and recent conferences.** Its job is to find **genuine,
unaddressed gaps** — and a candidate gap only earns a place here if it survives that
verification. Two of our first candidates did not (see below); one did, decisively.

The through-line of the field: **make the spiking neuron a parallelizable linear
recurrence, then reattach the reset nonlinearity as cheaply as possible.**
Surrogate-gradient BPTT is the accuracy-strong but memory-bound baseline everyone is
racing to match at lower cost. That is the math Spyx is built on.

## What verification changed (a gap is not a gap until it survives the literature)

A parallel adversarial pass (6 agents vs primary arXiv/proceedings, plus fake-ID
controls that correctly 404'd) reshaped the agenda:

- **KILLED — "reset-preserving parallel neuron is an open gap."** Already published:
  FPT ([2506.12087](https://arxiv.org/abs/2506.12087), exact reset via fixed-point),
  *Parallel Training in SNNs* (ICLR 2026, [2602.01133](https://arxiv.org/abs/2602.01133),
  25.6×), *Bullet Trains* ([2603.13283](https://arxiv.org/abs/2603.13283)), and
  snnTorch v1.0.0's `AssociativeLeaky`. **Not novel.**
- **KILLED — "spiking-SSM vs dense-SSM on SHD/SSC is unreported."** SiLIF
  ([2506.06374](https://arxiv.org/abs/2506.06374)) and S5-RF
  ([2504.00719](https://arxiv.org/abs/2504.00719)) already print those numbers.
- **CONFIRMED — "FP4/NVFP4/MXFP4 microscaling has not been applied to SNNs."** Two
  independent adversarial sweeps found nothing; the nearest boundary (AQ4SViT,
  [2606.15523](https://arxiv.org/abs/2606.15523)) uses integer/search quantization,
  **not** block-scaled FP4. **This is the real, unclaimed gap — and Spyx is the only
  SNN library that already has the formats.**

The lesson is baked into the process: [`/research-scout`](../.claude/skills/research-scout.md)
now requires a novelty-verification step before a candidate is called a gap.

## Spyx's structural advantages (corrected)

1. **Associative-scan for spiking + SSM in one library.** Spyx expresses spiking
   neurons as O(log T) prefix scans (`PSU_LIF`, `ResonateFire`) *and* unifies them
   with SSMs under the same `jax.lax.associative_scan` math. Caveat (verified): this
   is no longer *unique* — snnTorch v1.0.0 ships `AssociativeLeaky`, and PyTorch 2.8
   added a prototype `torch.associative_scan`. Spyx's defensible edge is being **among
   the earliest JAX-native implementations, with spiking+SSM under one scan API and
   the broadest feature surface** — not sole ownership of the idea.
2. **The only SNN library with FP4 microscaling.** `spyx.quant` (qwix) supports
   **NVFP4/MXFP4/MXFP8**; no other SNN framework has FP4/MX (they cap at int8/low-bit).
   This is the moat that makes the one confirmed gap (below) *ours to take first*.
3. **NIR reference node.** Only full NIR read+write in JAX, authored by a NIR
   co-author (Nat Commun 2024, s41467-024-52259-9). With **Intel Lava archived
   (2026-05-13)** and NengoDL stagnant, the field is consolidating — a timing opening.
4. **A neuroevolution track** — nearly unique among SNN libraries; a natural JAX
   `vmap`/`pmap` fit.

Gaps we design around: no native neuromorphic-hardware backend (we reach silicon via
NIR); smaller community; and a **stale headline release** (v0.1.19 while the code is
at 1.0) that undersells the project.

## Honest field constraints (verified; so we don't over-claim)

- **Fine-grained spike sparsity does not speed up dense accelerators.** Every audited
  sparse-SNN strategy fails to beat dense — *Collapse or Preserve*
  ([2603.13810](https://arxiv.org/abs/2603.13810), on Apple M3 Max SIMD). Wins come
  from coarsening (block/tile) or cutting dense-op count, not zero-skipping.
- **SNN energy advantage is conditional.** Against a memory-inclusive quantized-ANN
  baseline, SNNs win only at small T (≲10) and >90% sparsity — the ~6.4% spike-rate
  threshold of [2409.08290](https://arxiv.org/abs/2409.08290). SNN *training* costs ~1.3× more.
- **Matmul-free relocates the matmul** (attention stays dense; GPU tensor cores lack
  native INT2/3) — an edge/ASIC/FPGA story, not a GPU speedup.
- **LLM-scale spiking is conversion + linear attention.** Even SpikingBrain2.0
  ([2604.22575](https://arxiv.org/abs/2604.22575), 5B, INT8-spiking+FP8) and Sorbet
  ([2409.15298](https://arxiv.org/abs/2409.15298)) reach scale by abandoning pure
  spiking recurrence. Not a headline bet.
- **Benchmarks are in a credibility crisis** — SHD is saturated with test-as-val
  leakage; popular benchmarks provably don't require temporal processing (Ma et al.,
  [2502.09449](https://arxiv.org/abs/2502.09449)). Lead with SSC, the **Neuromorphic
  Sequential Arena** (NSA, [2505.22035](https://arxiv.org/abs/2505.22035), IJCAI 2025),
  and NeuroBench (Nat Commun 2025) — not SHD/N-MNIST.

Consistent with this repo's own honest findings: the Pallas result (matmul-bound; use
`associative_scan`) and the quant-aware-evolution result (ES breaks at ternary because
coarse rounding flattens the fitness landscape).

## Flagship studies (re-scoped after verification)

Priority reflects verified gap-strength, not original ordering.

| # | Study | Gap status | Bucket | Why |
|---|---|---|---|---|
| **F3** | **Sub-4-bit spiking: NVFP4/MXFP4 QAT + membrane-state quant** on SSC | **CONFIRMED gap** | novelty | Unclaimed FP4×SNN; Spyx uniquely has the formats. Include membrane-state quant (SQUAT gap, [2404.19668](https://arxiv.org/abs/2404.19668)); nearest boundary AQ4SViT is integer, not FP4. **The headline bet.** |
| F4 | **NeuroBench/NSA honest energy accounting** in `spyx.bench` | needed infra | extension | SOP + memory-inclusive + quantized-ANN baseline; wire NSA's 7 temporal tasks. Makes every other claim trustworthy. |
| F2′ | **Controlled** spiking-SSM vs dense-SSM head-to-head on SHD/SSC | narrowed gap | extension | The *apples-to-apples* comparison SiLIF/S5-RF leave open; cite them + Event-SSM/S7 as baselines. Extends `ssm`. |
| F1′ | **JAX-native associative-scan neuron** benchmark vs sequential/PSN | reframed | replication/extension | Not novel (FPT, 2602.01133, Bullet Trains, snnTorch `AssociativeLeaky` are prior art) — a clean JAX benchmark + reproduce the dynamic-decay neuron vs `PSU_LIF`. |
| F5 | Reproduce a headline (QKFormer [2403.16552](https://arxiv.org/abs/2403.16552) 85.65%, or SpikingSSM LRA [2408.14909](https://arxiv.org/abs/2408.14909)) | validation | replication | Validates the stack; Q-S5 ([2406.09477](https://arxiv.org/abs/2406.09477), Heckel co-author) is the QS5 reference. |

## Thematic pillars

1. **Sub-4-bit spiking (now the lead).** The FP4×SNN gap. NVFP4/MXFP4 QAT +
   membrane-state quantization, on SSC, always vs a quantized-ANN baseline.
2. **Honest efficiency & temporal benchmarking.** NeuroBench/NSA-aligned dual energy
   accounting; get Spyx into Open Neuromorphic tables.
3. **Spiking state-space models.** The controlled SHD/SSC comparison; close the ~1pt
   LRA gap; extends `ssm`.
4. **Parallel spiking dynamics.** Now a *benchmark/implementation* pillar (the novelty
   moved on): JAX-native associative-scan neurons, reproduce 2602.01133's dynamic-decay
   neuron; extend `.parallel` to CuBaLIF/ALIF.
5. **Gradient-free & hybrid training where it wins.** Control/RL; compare against
   EGGROLL-SNN ([2605.30361](https://arxiv.org/abs/2605.30361)) and SATR
   ([2601.21572](https://arxiv.org/abs/2601.21572)); build on `experimental.hybrid`/SGES.
6. **NIR depth as the deployment path** *(cross-cutting)* — be the most
   round-trip-verified NIR JAX node (Loihi 2, SpiNNaker 2, Xylo, Speck), exploiting the
   Lava vacuum. Update the substrate doc's hardware table with 2025 commercial parts
   (Innatera Pulsar, Akida 2, Xylo-Audio 2).

## Operating model

Runs through the repo's agentic loop:
**[`/research-scout`](../.claude/skills/research-scout.md) (propose + verify novelty)
→ [BACKLOG.md](BACKLOG.md) → [`/research-study`](../.claude/skills/research-study.md)
(web breadth / local-GPU depth) → [FINDINGS.md](FINDINGS.md) ledger →
[`/promote-finding`](../.claude/skills/promote-finding.md) into `spyx.experimental`/core
(human gate).** Honest negatives are first-class.

## Near-term milestones

- **M0 — Ship 1.0.0.** Cheapest, highest-leverage fix (discoverability + credibility).
- **M1 — F3 (sub-4-bit spiking):** attack the one confirmed, unclaimed gap while Spyx
  is the only SNN library with the formats — the first-mover window.
- **M2 — F4 (honest energy):** the credibility infrastructure M1/M3 lean on.
- **M3 — F2′ (controlled spiking-SSM comparison):** on Spyx's own `ssm` module.
- **Ongoing:** Open Neuromorphic benchmarks; deepen NIR round-trip verification.
