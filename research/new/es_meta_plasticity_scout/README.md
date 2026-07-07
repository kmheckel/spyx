# Scout verdict: ES-meta-learned spiking plasticity — **gap is CLAIMED**

## Question

Is "using an **evolutionary/ES outer loop** to meta-learn a **local synaptic
plasticity / three-factor** rule for **spiking** networks (optionally with the plastic
update framed as a parallelizable associative-scan)" an *unclaimed* research gap worth
a novelty claim?

## Method

Multi-agent scout: 5 parallel literature-search angles (evolved plasticity, spiking-evo
online learning, TTT/fast-weights, e-prop/OTTT, FA/DFA-for-SNN) → 43 direct/adjacent
prior-art candidates → **3 independent adversarial skeptics** each hunting the closest
prior art, defaulting to *refuted* when in doubt (the same discipline that killed the
F1/F2 false gaps and confirmed F3 in [PR #67](../../FINDINGS.md)). Majority rules.

## Verdict — **CLAIMED, 3/3 skeptics, high confidence**

The three core ingredients — (1) evolutionary/gradient-free outer loop, (2) local
plasticity / three-factor inner rule, (3) spiking neurons — are already **jointly**
claimed:

| Prior art | What it does |
| --- | --- |
| **Confavreux, Agnes, Zenke, Lillicrap, Vogels et al.**, *Balancing complexity, performance and plausibility to meta-learn plasticity rules in recurrent spiking networks*, **PLOS Comput. Biol. 2025** ([article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012910), bioRxiv 2024.06.17.599260) | **CMA-ES** (a literal evolution strategy) as the outer loop meta-learns *local* co-active plasticity rules (polynomial → MLP parameterizations) inside *recurrent spiking* E/I networks. |
| **Jordan, Schmidt, Senn, Petrovici**, *Evolving to learn: discovering interpretable plasticity rules for spiking networks*, **eLife 2021** ([article](https://elifesciences.org/articles/66273), arXiv:2005.14149); RL variant arXiv:2202.12322 | **Cartesian genetic programming** (evolutionary outer loop) discovers interpretable *local three-factor* spiking plasticity rules across reward-, error-, and correlation-driven regimes. |

Per the ruling criterion, ES × local-plasticity × spiking being co-present — even absent
parallelization — makes the gap claimed. **We do not claim novelty here.**

## The one thin residual (not worth a novelty claim on its own)

The *only* differentiator the target framing adds is expressing the evolved local
delta/three-factor update as a **parallelizable associative-scan (affine-recurrence)**
inner loop. That machinery exists but only *separately and non-spiking*: Yang et al.,
*Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, NeurIPS
2024 ([arXiv:2406.06484](https://arxiv.org/abs/2406.06484)), which is gradient-trained.
This is an **implementation/efficiency** delta, not a new meta-learning capability — an
engineering nicety, not a paper.

## What we built anyway (library building blocks, cited to their sources)

The scout was paired with a build: three **implementations of published methods** now
live in `spyx.experimental` — useful regardless of the (negative) novelty finding, and
they make the "associative-scan-parallel evolved plasticity" residual cheap to try:

- [`ttt.TTTFastWeight`](../../../src/spyx/experimental/ttt.py) — fast-weight / TTT layer
  (Sun 2024 arXiv:2407.04620; Schlag 2021 arXiv:2102.11174). Its **hebb** rule is a
  matrix-valued `PSU_LIF` (scalar transition → `associative_scan`); its **delta** rule is
  sequential-only here (matrix transition; points to DeltaNet).
- [`local_learning.ThreeFactorLIF`](../../../src/spyx/experimental/local_learning.py) —
  online e-prop/OTTT three-factor plasticity (Bellec 2020; Xiao 2022). Coefficients are
  the only `nnx.Param`s, so `spyx.experimental.hybrid`/`evolve` (ES) meta-learn the rule —
  i.e. this *is* the substrate of the (already-claimed) idea, provided as a clean tool.
- [`feedback_alignment`](../../../src/spyx/experimental/feedback_alignment.py) — FA
  (Lillicrap 2016) and DFA (Nøkland 2016) backprop-free training via random feedback.

## Reproducibility

- **Scout:** 11-agent workflow (5 scout + 3 verify + 3 design), web-search-grounded,
  honesty-constrained; all citations verified with real titles/URLs.
- **Correctness of the builds:** `tests/test_ttt.py`, `tests/test_local_learning.py`,
  `tests/test_feedback_alignment.py` (parallel==sequential, FA↔BP alignment, DFA >0.6 acc,
  trace decay, meta-grad/ES reachability).
- **Date run:** 2026-07-06.
