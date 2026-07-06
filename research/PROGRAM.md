# Spyx research program

A standing research agenda for Spyx, grounded in a 2024–2026 survey of the SNN
field (frameworks, training methods, efficiency/hardware, applications — see the
sourced briefs behind each claim). It picks a small number of bets where Spyx has a
**structural advantage** and the field has an **open, tractable gap**, and it runs
them through the repo's agentic-research loop with a human promotion gate.

The through-line of the whole field right now: **make the spiking neuron a
parallelizable linear recurrence, then reattach the reset nonlinearity as cheaply as
possible.** Surrogate-gradient BPTT is the accuracy-strong but memory-bound baseline
everyone is trying to match at lower cost. That is *exactly* the math Spyx is built
on — and no other framework is positioned the same way.

## Spyx's structural advantages (why these bets, and not others)

Verified against the landscape:

1. **The associative-scan moat.** Spyx is the only SNN library combining JAX/XLA
   whole-loop JIT + **true parallel-in-time neurons via `jax.lax.associative_scan`**
   (`PSU_LIF`, `ResonateFire`) + first-class **SSM/S5** — in one library. PyTorch has
   no general `associative_scan` primitive (snnTorch's "parallel" neuron is RNN-fusion,
   SpikingJelly's is fused CUDA kernels, Sinabs/Lava-DL parallelize BPTT with custom
   CUDA). The other JAX libraries don't exploit it (SNNAX only unrolls `lax.scan`,
   jaxsnn is event-driven, Slax is online-rule-focused). **Expressing a spiking neuron
   as an O(log T) prefix scan — and unifying it with SSMs under the same math — is
   architecturally unavailable to the mainstream.**
2. **The most modern quantization stack in the entire SNN field.** `spyx.quant` (qwix)
   supports **NVFP4/MXFP4/MXFP8** microscaled formats. No other SNN framework — PyTorch
   or JAX — has FP4/MX; they cap at int8/low-bit QAT.
3. **NIR reference node.** Only full NIR read+write in JAX, authored by a NIR
   co-author. With **Intel Lava archived (May 2026)** and NengoDL stagnant, the field
   is consolidating around NIR + a few active libraries — a timing opening.
4. **A neuroevolution track** (`experimental` ES/CMA-ES/SGES) — nearly unique among
   SNN libraries, and a natural JAX `vmap`/`pmap` fit.

Known gaps we design *around*, not into: no native neuromorphic-hardware backend
(we reach silicon through NIR); smaller community; and a **stale headline release**
(v0.1.19 while the code is at 1.0) that undersells the project.

## Honest constraints (so the program doesn't over-claim)

The field is mid-correction on efficiency, and we adopt the corrected view as a
design constraint:

- **Fine-grained spike sparsity does not speed up GPUs/TPUs.** At 10% firing a 32-wide
  SIMD lane is all-zero only ~3.5% of the time; every audited sparse-SNN GPU strategy
  fails to beat dense. Wins come from *coarsening* (block/tile structure) or *reducing
  dense-op count* (temporal aggregation, fewer timesteps), never zero-skipping.
- **SNN energy advantage is conditional, not intrinsic.** Against a fair, memory-inclusive
  quantized-ANN baseline, SNNs win only at small T (≲10) and very high sparsity (>90%);
  SOP-only "5–87×" headlines omit data movement, and SNN *training* costs ~1.3× more.
- **Matmul-free relocates the matmul** (attention stays dense; GPU tensor cores lack
  native INT2/3) — it's an edge/ASIC/FPGA story, not a GPU speedup.
- **LLM-scale spiking is conversion + linear attention** ("quantization theater"); no
  independently-verified from-scratch billion-param binary SNN exists. Not a headline bet.
- **Benchmarks are in a credibility crisis** — SHD is saturated with test-as-val leakage,
  and popular benchmarks provably don't require temporal processing. Lead with SSC, the
  Neuromorphic Sequential Arena (NSA), and NeuroBench, not SHD/N-MNIST.

These are consistent with what this repo already found the honest way: the Pallas
result (matmul-bound; use `associative_scan`, don't write a kernel), and the
quant-aware-evolution result (ES breaks at ternary because coarse rounding flattens
the fitness landscape).

## Thematic pillars

Each pillar names *why Spyx*, the *open gap*, and concrete studies. Priority in **bold**.

### Pillar 1 — Parallel spiking dynamics: the associative-scan thesis **[flagship]**
*Why Spyx:* the moat. *Gap:* the reset↔parallel-scan tension is **the** central
problem, and **FPT** (fixed-point, O(T)→O(K≈3), reset *preserved*) and **PMBC**
(parallel max-min boundary compression) have **no clean public JAX implementation**.
- Reference JAX "**parallel LIF with exact reset**" (`lax.associative_scan` /
  `lax.custom_root`), extending `PSU_LIF`; verify FPT's unpublished wall-clock numbers.
- Extend the `.parallel` associative-scan path to more neurons (CuBaLIF, ALIF).
- Scaling studies (throughput × accuracy at long T) that no PyTorch library can replicate.

### Pillar 2 — Spiking state-space models **[flagship]**
*Why Spyx:* extends the existing `ssm` module; JAX scan is the natural substrate.
*Gap:* spiking SSMs closed almost the whole LRA gap to dense S4D (84.3% vs 85.5%) at
~90% sparsity, but the decisive **spiking-SSM vs dense-SSM comparison on SHD/SSC is
unreported** — and dense SSMs (S7, Event-SSM) currently hold that crown.
- Implement SpikingSSM's surrogate-dynamic-network or SPikE-SSM's PMBC in Spyx.
- The honest SHD/SSC/LRA head-to-head: spiking-SSM vs dense S5 vs Event-SSM/S7.

### Pillar 3 — Sub-4-bit spiking: own the FP4 × SNN intersection **[flagship]**
*Why Spyx:* the only SNN library with NVFP4/MXFP4. *Gap:* **no published work applies
FP4 microscaling to SNNs** — an unclaimed first-mover slot.
- NVFP4/MXFP4 QAT on spiking classifiers, on SSC (not saturated SHD), with the
  honest quantized-ANN baseline every time.
- **Membrane-state quantization** (the neglected SQUAT gap) — directly in the neuron
  module's wheelhouse.
- Binary-spike × FP4-weight accumulate-only path (where the two compound).

### Pillar 4 — Honest efficiency & temporal benchmarking **[credibility infrastructure]**
*Why Spyx:* `bench`/`quant` already lean this way. *Gap:* most SNN papers still quote
SOP-only energy; a library that reports *fair* numbers is a differentiator and makes
every other pillar's claims trustworthy.
- NeuroBench-style **dual energy accounting** in `spyx.bench`: hardware-agnostic SOP
  *and* hardware-aware (memory-inclusive) *and* the quantized-ANN-equivalent baseline.
- Support the temporal suites (NSA, NeuroBench tasks); get Spyx into the Open
  Neuromorphic benchmark tables (converts the associative-scan advantage into a citation).

### Pillar 5 — Gradient-free & hybrid training where it genuinely wins
*Why Spyx:* the neuroevolution track + JAX `vmap`. *Gap:* narrow but real — control/RL,
non-differentiable objectives. Calibrate expectations (honest negatives are common).
- Compare against 2026 signals: **EGGROLL** (low-rank ES) and **SATR** (trust-region ES
  for recurrent-SNN control). Build on `spyx.experimental.hybrid`/SGES.
- Carry forward the honest ES-at-ternary finding (coarse quant flattens the ES landscape).

### Pillar 6 — NIR depth as the deployment path *(cross-cutting)*
Turn "no hardware backend" into a strength: be the **most rigorously round-trip-verified
NIR JAX node** (Loihi 2, SpiNNaker 2, Xylo, Speck), exploiting the Lava-archival vacuum.
This is how findings from Pillars 1–3 reach silicon without us building a backend.

*(Deferred: EventProp / online-learning-rule menu in JAX — contests jaxsnn/GeNN/Slax,
but Slax already stakes the online-rule claim; a module, not a pillar, for now.)*

## Flagship studies (seed the backlog; classified replication/extension/novelty)

| # | Study | Bucket | Pillar | Why it matters |
|---|---|---|---|---|
| F1 | Parallel LIF with **exact reset** in JAX (FPT/PMBC) vs sequential & PSN | novelty | 1 | The field's central problem; no JAX impl; Spyx's moat |
| F2 | Honest **spiking-SSM vs dense-SSM** on SHD/SSC/LRA | novelty | 2 | The unreported decisive comparison; extends `ssm` |
| F3 | **Sub-4-bit spiking**: NVFP4/MXFP4 QAT + membrane-state quant on SSC | novelty | 3 | Unclaimed FP4×SNN slot; Spyx already has the formats |
| F4 | **NeuroBench-style honest energy accounting** in `spyx.bench` | extension | 4 | Credibility infra; corrective to field over-claiming |
| F5 | Reproduce a headline (SpikingSSM LRA / QKFormer sparsity) in Spyx | replication | 2/4 | Validates the stack; seeds the comparisons above |

## Operating model

Everything runs through the repo's agentic-research loop:
**[`/research-scout`](../.claude/skills/research-scout.md) → [BACKLOG.md](BACKLOG.md)
→ [`/research-study`](../.claude/skills/research-study.md) (scheduled web for breadth,
local `/loop` on the AMD GPU for depth) → [FINDINGS.md](FINDINGS.md) ledger →
[`/promote-finding`](../.claude/skills/promote-finding.md) into `spyx.experimental`/core
(human gate, see [PROMOTION.md](PROMOTION.md)).** Honest negatives and nulls are
first-class and stay recorded.

## Near-term milestones

- **M0 — Ship 1.0.0.** The stale headline release is the cheapest, highest-leverage fix
  (discoverability + credibility). Blocks nothing technical; gated only on the PyPI step.
- **M1 — Pillar 1 flagship (F1):** the parallel exact-reset neuron — establishes the moat
  with a result no PyTorch library can produce.
- **M2 — Pillar 4 infra (F4):** honest energy accounting — makes M1/M3 claims trustworthy.
- **M3 — Pillar 2 flagship (F2):** the spiking-SSM vs dense-SSM comparison — a genuinely
  novel, publishable contribution on Spyx's own `ssm` module.
- **Ongoing:** get Spyx into Open Neuromorphic benchmarks; deepen NIR round-trip
  verification (Pillar 6).
