# Choosing an approach

You have a task, a dataset, and a deadline. This page turns the
[training-methods spine](training-methods.md) into a decision: given **what you
are building** (the application) and **how you want to represent it** (the
architecture), which **training method** should you reach for, and where is the
Spyx entry point?

Read it as two lookup tables plus a short flow at the end. The methods are the
five from [Training methods](training-methods.md); skim that page first if a row
name is unfamiliar.

## Matrix A — method × application

Rows are training methods; columns are the three application shapes Spyx targets.
Each cell says how well the pairing **fits** and *why*.

| Method | Control / RL | Classification | Language modelling |
|---|---|---|---|
| **Evolutionary** (0th) | **Strong.** No gradient through the environment needed; short episodes, small nets, population `vmap`s well. The canonical fit — see [`cartpole_evo`](../examples/neuroevolution/cartpole_evo.ipynb). | **Niche.** Works on tiny models or when optimising the exact hard-spike objective, but sample cost makes it uncompetitive at scale. | **Poor.** Parameter counts and sequence lengths blow up ES variance; not a serious option. |
| **Surrogate gradient** (1st) | **OK.** Works when the environment is differentiable or you train a policy by BPTT, but for black-box control ES is usually simpler. | **Strong — the default.** Labelled data + differentiable surrogate + Optax scaling. This is the workhorse path ([SurrogateGradientTutorial](../examples/surrogate_gradient/SurrogateGradientTutorial.ipynb)). | **Strong.** BPTT through long sequences is exactly what surrogate + SSM/phasor backbones are for; the scalable choice. |
| **Conversion & QAT** (transfer) | **Rare.** Little to convert *from* in RL; skip unless you already have a trained policy to quantize. | **Strong.** Convert a trained ANN classifier to a rate-coded SNN, or QAT-fine-tune an fp32 SNN for int8/ternary deployment. | **Emerging.** Quantized sequence models (e.g. Q-S5) are viable; conversion is harder because temporal structure matters. |
| **Local / bio-inspired** | **Roadmap.** Online RL is a natural fit for e-prop, but not yet a Spyx API ([#28](https://github.com/kmheckel/spyx/issues/28)). | **Roadmap.** Feedback alignment / synthetic grads apply, but unshipped ([#27](https://github.com/kmheckel/spyx/issues/27)). | **Roadmap.** Long-context memory-bounded training is the motivation; aspirational today. |
| **0+1 hybrid** | **Good.** When you optimise a hard-spike control loss and want a cheap descent bulk plus an unbiased hard-objective correction. | **Good when 1st-order plateaus.** Use it to close the surrogate-bias gap on the hard-spike classifier, not as a first move. | **Situational.** Extra hard-forward passes cost more at long `T`; reserve for when surrogate bias visibly hurts. |

**How to read a cell.** "Strong" = the method's information order matches the
task's structure, so it is the efficient choice. "Roadmap" = the fit is real but
the Spyx API does not exist yet — track the linked issue. When two cells are both
"Strong", let the architecture (Matrix B) and your compute budget break the tie.

## Matrix B — method × architecture

Columns are the four Spyx backbones you might build with. Each cell says which
methods **pair well** with that architecture and why.

| Architecture | Pairs well with | Notes |
|---|---|---|
| **LIF / RSNN** — [`spyx.nn`](../reference/nn.md) | Surrogate gradient (default); Evolutionary (small nets / control); Conversion & QAT | The reference case: hard spikes make surrogate gradients *necessary* and make weight-only quantization *lossless* ([`spiking_feedforward_rules`](../reference/quant.md)). ES trains the same `nnx.Module` gradient-free. |
| **SSM / S5** — [`spyx.ssm`](../reference/ssm.md) | Surrogate gradient (long sequences); QAT (Q-S5); Transfer *into* spiking | Linear associative-scan recurrence is fully differentiable and BPTT-friendly, so 1st-order shines. Diagonal state quantizes well; [`ssm_to_spiking_transfer`](https://github.com/kmheckel/spyx/tree/main/research/new/ssm_to_spiking_transfer) studies moving SSM dynamics into spiking neurons. |
| **Phasor** — [`spyx.phasor`](../reference/phasor.md) | Surrogate gradient (complex/oscillatory dynamics) | Complex poles are stored as **real** params, so a stock `optax` + `jax.grad` loop over a real loss trains them without Wirtinger surprises. ES also applies; QAT on complex weights is unexplored. |
| **Slot-memory / Raven** — [`spyx.experimental.raven`](../reference/experimental.md) | Surrogate gradient (recall tasks); (roadmap) local rules for online memory | Routing-slot memory (`RavenRSM` + spiking sibling) is trained by BPTT for high-recall sequence tasks — see [`raven_sparse_memory_recall`](https://github.com/kmheckel/spyx/tree/main/research/new/raven_sparse_memory_recall). Being experimental, its API can move. |

**The through-line.** Surrogate gradient ([`spyx.axn`](../reference/axn.md) +
[`spyx.optimize`](../reference/optimize.md)) pairs with *every* architecture — it
is the common substrate. The other methods are specialisations: ES when there is
no usable gradient, conversion/QAT when you inherit or must shrink a model, local
rules when locality is a hard constraint (not yet shipped), and the hybrid when
the surrogate's bias on the hard-spike loss is the thing costing you accuracy.

## Start here

A four-step flow from a cold start:

1. **Name the task → pick the application.** Is it *control/RL*, *classification*,
   or *language modelling*? That is the column in Matrix A.
2. **Pick the architecture** for how you want to represent it — a
   [LIF/RSNN](../reference/nn.md) for event-driven and neuromorphic-bound work, an
   [SSM/S5](../reference/ssm.md) for long differentiable sequences, a
   [phasor](../reference/phasor.md) net for oscillatory/frequency-selective
   dynamics, or [Raven slot-memory](../reference/experimental.md) for
   high-recall memory tasks. That is the row in Matrix B.
3. **Pick the method** where both matrices agree:
    - Differentiable forward + labelled data → **surrogate gradient** (start with
      [`spyx.optimize.fit`](../reference/optimize.md) and a
      [`spyx.axn`](../reference/axn.md) surrogate). This is the right first move
      for the large majority of tasks.
    - Non-differentiable forward, or you must optimise the exact hard-spike / reward
      objective, or it's small-net control → **evolutionary** (`evosax` +
      `spyx.experimental.zoo`; [cartpole notebook](../examples/neuroevolution/cartpole_evo.ipynb)).
    - You already have a trained model, or you're hitting an efficiency/energy
      budget → **conversion & QAT** ([`spyx.quant`](../reference/quant.md);
      [QAT notebook](../examples/quantization/qat_intro.ipynb)).
    - Surrogate-gradient training *works but plateaus below the hard-spike model
      you deploy* → add the **0+1 hybrid** correction (`spyx.experimental.hybrid`).
    - You need *online / on-chip* local learning → it's **roadmap**
      ([#27](https://github.com/kmheckel/spyx/issues/27),
      [#28](https://github.com/kmheckel/spyx/issues/28)); use surrogate-gradient
      BPTT plus [`spyx.experimental.compress`](../reference/experimental.md) for
      the memory pressure in the meantime.
4. **Measure, then iterate.** Confirm the choice with real numbers —
   [`spyx.bench`](../reference/bench.md) for latency / throughput / spike-rate
   energy proxy (see the [benchmarking how-to](../how-to/benchmarking.md)) — and
   only reach for a heavier method (ES samples, a hybrid correction) once the
   default has actually under-delivered.

When in doubt: **start with a surrogate-gradient LIF network and
[`spyx.optimize.fit`](../reference/optimize.md)**, measure, and specialise from
there. Every other method on this page is a considered deviation from that
baseline, not a replacement for trying it first.
