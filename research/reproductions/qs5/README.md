# Q-S5: Towards Quantized State Space Models

> Reference / index entry. This folder is a pointer to the maintainer's own
> paper and its reference implementation, not a self-contained Spyx run. It
> records the paper's claims and its relevance to the Spyx quantization and SSM
> subsystems. Follow the template headers; sections that do not apply to a
> pointer entry are marked `N/A`.

## Title

Q-S5: Towards Quantized State Space Models — a study of full and heterogeneous
quantization of the S5 state space model.

## Paper & arXiv/DOI

- **Title:** Q-S5: Towards Quantized State Space Models
- **Authors / venue / year:** Steven Abreu, Jens E. Pedersen, Kade M. Heckel,
  Alessandro Pierro. arXiv preprint, June 2024.
- **Link:** https://arxiv.org/abs/2406.09477 (arXiv:2406.09477)
- **Code:** https://github.com/kmheckel/Q-S5 and
  https://github.com/stevenabreu7/Q-S5 (AQT + JAX).
- **Bucket:** reproductions

## Claim under test

Full quantization of the S5 state space model is possible with under 1% test
accuracy degradation on sequential MNIST (sMNIST) and most of the Long Range
Arena (LRA), but the *recurrent* weights degrade sharply below 8 bits while
*other* components tolerate much lower precision — so a heterogeneous,
per-component precision assignment reaches the best accuracy/memory trade-off.

## Method

The study quantizes the S5 architecture (a diagonal complex-valued linear state
space model) with the AQT quantization library in JAX. It sweeps bit-widths
independently over the model's components — recurrent state-transition weights
versus feedforward, input, and output projections — under both post-training
quantization (PTQ) and quantization-aware training (QAT), and measures test
accuracy on sMNIST and the LRA benchmark suite against the full-precision
baseline.

Deviation from a standard reproduction: this entry does not re-run the
experiments inside Spyx. It indexes the published result and maps its findings
onto Spyx's own quantization and SSM code. A Spyx-native reproduction would
port the heterogeneous-precision assignment onto `spyx.ssm.S5Diag` and drive it
through `spyx.quant`.

## Key findings (from the paper)

- Fully quantized S5 loses less than 1% test accuracy on sMNIST and on most LRA
  tasks.
- Recurrent (state-transition) weights are the precision bottleneck: they need
  at least 8 bits, and accuracy collapses below that.
- The remaining components (input/output/feedforward projections) compress much
  further without hurting accuracy.
- Exploiting this heterogeneity with a per-component precision strategy cuts
  memory footprint by roughly 6x.
- PTQ alone is sufficient only on the language-based LRA tasks; the other tasks
  require QAT to recover accuracy.

## Spyx modules used

Modules this entry connects to (a Spyx-native reproduction would build on these):

- [`spyx.ssm.S5Diag`](../../src/spyx/ssm.py) — diagonal complex SSM; the direct
  analogue of the S5 recurrence the paper quantizes.
- [`spyx.quant`](../../src/spyx/quant.py) — qwix-based op-level quantization
  rules.
- [`spyx.phasor.ResonateFire`](../../src/spyx/phasor.py) — shares S5Diag's
  diagonal-eigenvalue pole form; a spiking relative of the quantized recurrence.

## How to run

`N/A` for this pointer entry — no Spyx experiment is checked in here. To run the
original study, clone one of the reference repos above (AQT + JAX) and follow
its instructions.

## Results

`N/A` — see the paper for the full sMNIST and LRA numbers. No Spyx-native
measurements are recorded in this folder.

## Findings

Recorded above under "Key findings". These are the paper's reported results, not
a Spyx re-measurement.

## Relevance to spyx

Q-S5 is the empirical justification for how quantization is scoped in Spyx.

- **`spyx.quant` heterogeneous scoping.** Spyx's rule builders
  (`linear_only_rules`, `weights_only_rules`, `bitnet_ternary_rules`) match by
  *operation* — `dot_general` and `conv_general_dilated` — and deliberately
  exclude `einsum`, which keeps the SSM's `B`/`C` input-output projections in
  fp32 while compressing the feedforward projections. (The diagonal state
  recurrence itself is an elementwise complex multiply in `associative_scan`,
  fp32 regardless of the quant rules.) That op-based split is a direct,
  mechanical realization of Q-S5's central finding: keep the precision-sensitive
  recurrent-path weights high-precision (>=8-bit) and push the other components
  lower.
- **`spyx.ssm.S5Diag`.** The model Q-S5 quantizes is exactly this diagonal
  complex SSM. It is the natural target for a Spyx-native reproduction: apply
  `spyx.quant` rules to its projections while leaving the diagonal recurrence in
  fp32, then sweep bit-widths to re-check the ~6x memory / <1% accuracy result.
- **Binary-activation-aware quantization.** Spyx pairs quantized weights with
  spiking (binary {0,1}) activations. Q-S5 quantizes the weight/state path of a
  non-spiking SSM; combining its heterogeneous weight-precision strategy with
  Spyx's binary activations is the open direction that the quant subsystem is
  built to explore.
- **`ssm_to_spiking_transfer` study.** A quantized S5 backbone (recurrence in
  fp32, projections compressed per Q-S5) is a candidate feature extractor for a
  spiking readout — using the quantized SSM as the front end and Spyx spiking
  neurons as the classifier head.

## Reproducibility

- **Seeds / JAX / hardware / Spyx commit / date run:** `N/A` — this entry checks
  in no Spyx run. Provenance for the original results lives in the linked paper
  and reference repositories.
