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

## Active track: quantization & efficient architectures

The current focus. Studies probing low-precision (NVFP4/MXFP4, ternary),
matmul-free layers, SSMs, and the training methods that fit them.

| Study | Claim (short) | Verdict | Status | Landed in / next |
| --- | --- | --- | --- | --- |
| [ternary_llm](new/ternary_llm/) | BitNet-ternary QAT path generalizes from spiking nets to transformers | ✅ | experimental | `spyx.experimental.matfree` |
| [quant_aware_evolution](new/quant_aware_evolution/) | Gradient-free ES beats STE-QAT at extreme precision (no STE bias) | ➖ | new | null on easy task; needs a hard, capacity-constrained task |
| [hybrid_evo_surrogate](new/hybrid_evo_surrogate/) | Orthogonal-ES correction / SGES beats surrogate on the hard-spike loss | ➖ | experimental | `spyx.experimental.hybrid` (safe, not a win — needs large-bias regime) |
| [pallas_neurons](new/pallas_neurons/) | A fused Pallas neuron kernel is worth building (#24) | ❌ | new | matmul-bound; `associative_scan` already gives 2–21× portably |

## Other studies

| Study | Claim (short) | Verdict | Status | Landed in / next |
| --- | --- | --- | --- | --- |
| [parallel_spiking_neurons](new/parallel_spiking_neurons/) | Reset-free / R&F neurons parallelize the time loop with no accuracy loss | ✅ | core | `spyx.nn.PSU_LIF`, `spyx.phasor.ResonateFire` |
| [raven_sparse_memory_recall](new/raven_sparse_memory_recall/) | Routing-slot memory beats a diagonal SSM on recall | ✅ (modest) | experimental | `spyx.experimental.raven` (regime-dependent) |
| [ssm_to_spiking_transfer](new/ssm_to_spiking_transfer/) | R&F *is* a thresholded S5Diag; transferring SSM dynamics helps spiking | ➖ | new | equivalence exact; S5Diag still beats the spiking variant |
| [pretrain_finetune_curriculum](new/pretrain_finetune_curriculum/) | A pretrain→finetune curriculum lifts spiking accuracy | ❌ | new | honest negative; boundary condition recorded |

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
