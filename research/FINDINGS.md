# Findings ledger

The single review surface for research done with Spyx. Every study lands a row
here; the **Status** column is the promotion gate — see [PROMOTION.md](PROMOTION.md)
for the rungs and criteria. Agents append rows and keep verdicts honest; **only a
human flips Status** (research → experimental → core).

**Verdict:** ✅ positive · ➖ null / parity · ❌ negative · ⏳ in progress.
**Status:** `new` (in `research/`) · `experimental` (landed in `spyx.experimental`) ·
`core` (stable API) · `archived`.

> **Honest negatives are first-class.** A ❌ or ➖ is a *result*, not a failure. It
> stays in `research/` permanently so the question is not re-litigated. It is never
> deleted, and never reshaped into a positive to justify promotion.

Grouped by theme. The **active track is quantization & low precision** (the confirmed
FP4×SNN gap and its neighbours).

### Quantization & low precision

| Study | Claim (short) | Verdict | Status | Landed in / note |
| --- | --- | --- | --- | --- |
| [fp4_spiking_qat_hard](new/fp4_spiking_qat_hard/) | Which sub-8-bit weight format is best for SNNs on a hard task | ✅ | new | **NVFP4 most reliable**: nvfp4 ≥ int4 (holds better when tight) ≥ mxfp4/ternary; int8 near-lossless; ALIF didn't lift the ceiling |
| [fp4_spiking_qat](new/fp4_spiking_qat/) | FP4 (NVFP4/MXFP4) microscaling is a viable SNN weight format | ✅ | new | feasibility: first FP4×SNN datapoints, lossless on an easy task (no ranking there) |
| [ternary_llm](new/ternary_llm/) | BitNet-ternary QAT generalizes from spiking nets to transformers | ✅ | experimental | `spyx.experimental.matfree` |
| [honest_energy_accounting](new/honest_energy_accounting/) | Honest (memory-inclusive, QANN-baselined) SNN energy accounting | ✅ | new | crossover 5.99% ≈ literature 6.4%; SOP proxy under-reports ~700× |
| [quant_aware_evolution](new/quant_aware_evolution/) | Gradient-free ES beats STE-QAT at extreme precision (no STE bias) | ➖ | new | null on easy task |
| [quant_aware_evolution_hard](new/quant_aware_evolution_hard/) | The STE-bias gap appears once quantization costs accuracy | ❌ | new | reversed: ES *breaks* at ternary (coarse rounding flattens the ES landscape) |
| [activation_packing](new/activation_packing/) | Sparse+quantized activations pack exactly; mask+value beats dense k-bit below a density crossover | ✅ | experimental | `spyx.experimental.compress`: k-bit `pack_nbit`/`packed_quant_dense` + `sparse_quant_pack`; crossover `(bits-1)/bits` confirmed 18/18; 5.3× vs fp32 BPTT residual |

### Parallel spiking neurons

| Study | Claim (short) | Verdict | Status | Landed in / note |
| --- | --- | --- | --- | --- |
| [parallel_spiking_neurons](new/parallel_spiking_neurons/) | Reset-free / R&F neurons parallelize the time loop, no accuracy loss | ✅ | core | `spyx.nn.PSU_LIF`, `spyx.phasor.ResonateFire` |
| [reset_preserving_parallel_lif](new/reset_preserving_parallel_lif/) | Parallelize a hard-reset LIF while keeping the exact reset (FPT) | ✅ | experimental | `ParallelResetLIF`: exact + ~1.7–2.4× over sequential on GPU |
| [rf_ssm](new/rf_ssm/) | S5-RF (HiPPO init + decoupled reset) beats plain R&F on long-range | ➖ | experimental | `RFSSM`: scan-exact + ~4×, but **no accuracy win** over the simpler neurons here |
| [sigma_delta_neuron](new/sigma_delta_neuron/) | Graded sigma-delta neuron transmits fewer events at matched accuracy on redundant input | ✅ | experimental | `spyx.experimental.SigmaDelta`: **3.84× fewer events** vs LIF at matched acc (redundancy-dependent; event-driven-HW win) |
| [pallas_neurons](new/pallas_neurons/) | A fused Pallas neuron kernel is worth building (#24) | ❌ | new | matmul-bound; `associative_scan` already gives 2–21× portably |

### Training methods, SSMs & memory

| Study | Claim (short) | Verdict | Status | Landed in / note |
| --- | --- | --- | --- | --- |
| [hybrid_evo_surrogate](new/hybrid_evo_surrogate/) | Orthogonal-ES / SGES correction beats surrogate on the hard-spike loss | ➖ | experimental | `spyx.experimental.hybrid` (safe, not a win — needs large-bias regime) |
| [raven_sparse_memory_recall](new/raven_sparse_memory_recall/) | Routing-slot memory beats a diagonal SSM on recall | ✅ (modest) | experimental | `spyx.experimental.raven` (regime-dependent) |
| [ssm_to_spiking_transfer](new/ssm_to_spiking_transfer/) | R&F *is* a thresholded S5Diag; transferring SSM dynamics helps spiking | ➖ | new | equivalence exact; S5Diag still beats the spiking variant |
| [pretrain_finetune_curriculum](new/pretrain_finetune_curriculum/) | A pretrain→finetune curriculum lifts spiking accuracy | ❌ | new | honest negative; boundary recorded |

## How to read / update this

- **Adding a study (agent):** append a row with verdict ⏳ while running, then the
  real verdict once the study's README Findings are filled from a real run. Link the
  study folder. Do not invent numbers.
- **Promoting (human):** when you decide a `new` finding earns a home in the library,
  run [`/promote-finding`](../.claude/skills/promote-finding.md) — it walks the
  [PROMOTION.md](PROMOTION.md) checklist and, on your go-ahead, extracts the API +
  tests + docs and flips the Status here.
- **Starting a study (agent):** [`/research-study`](../.claude/skills/research-study.md)
  scaffolds and adversarially verifies a new study, then stops at this ledger for your
  review — it never promotes on its own.
