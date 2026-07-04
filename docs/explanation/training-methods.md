# Training methods

There is no single way to train a spiking network. The methods differ in **what
information about the loss they exploit** — and that choice, more than any
architectural detail, decides which problems a method can solve, how fast it
converges, and whether the thing you train is faithful to the hard-spike model
you will eventually deploy.

This page organises Spyx's training methods by the **order of information** they
use, from zeroth-order (only loss *values*) up through first-order (loss
*gradients*), the transfer methods that reuse an already-trained model, the
local/bio-inspired family (mostly roadmap in Spyx today), and a hybrid that
combines a first-order bulk direction with a zeroth-order error correction. For
each method you get the same three things: **when to use it**, **the trade-off**,
and **the Spyx entry point**.

If you already know your task and just want to be pointed at a method, jump to
[Choosing an approach](choosing-an-approach.md) and read this page for the *why*.

## The ladder of information

A training method can only use the information it asks for:

| Order | What it sees | Differentiable forward needed? | Family |
|---|---|---|---|
| **0th** | loss *values* at sampled parameters | no | evolutionary / ES |
| **1st** | loss *gradient* via a surrogate | yes (surrogate) | surrogate-gradient BPTT |
| **transfer** | a pretrained ANN / an fp32 SNN | reuses another method | conversion & QAT |
| **local** | per-layer signals, no global backprop | partial | feedback alignment, e-prop, synthetic grads |
| **0+1 hybrid** | surrogate gradient **+** sampled hard-spike loss | yes (surrogate) | evolutionary error-correction |

The higher the order, the cheaper the descent — a gradient points you straight
downhill, whereas value-only search has to *discover* the direction from
samples. But higher order also demands more structure: a first-order method needs
the forward pass to be (surrogate-)differentiable, which a hard spike is not.
Every method below is a different answer to that tension.

## 0th-order — evolutionary / neuroevolution

**Idea.** Never differentiate anything. Keep a search distribution over parameter
vectors, sample a *population*, evaluate the true loss of each sample, and move
the distribution toward the samples that scored well. Evolution strategies (ES)
estimate a search gradient from those fitness values alone.

**When to use it.**

- The forward pass is **non-differentiable** end to end — a discrete environment,
  a black-box simulator, a non-differentiable reward — so no surrogate exists.
- You want to optimise the **hardware-faithful hard-spike loss directly**, with no
  surrogate approximation in the loop, accepting that you pay for it in samples.
- **Reinforcement-learning / control** with short episodes and small networks,
  where a gradient through the environment is unavailable or high-variance and the
  population evaluates embarrassingly in parallel under `jax.vmap`.

**The trade-off.** Sample complexity. ES needs many forward evaluations per update
and its variance grows with the parameter count, so it does not scale to large
classifiers or language models the way gradient descent does. In exchange you get
a method that is indifferent to non-differentiability and optimises exactly the
objective you deploy.

**Spyx entry point.** Spyx leans on
[`evosax`](https://github.com/RobertTLange/evosax) (installed via the `[evo]`
extra) for the algorithms; the network is a plain `nnx.Module` whose parameters
you flatten into the ES search vector.

- Algorithms to reach for: `evosax.algorithms.Open_ES` (the OpenAI-ES baseline),
  `SNES` (separable natural ES — a strong, cheap default), and `GuidedES` (feeds a
  surrogate gradient *into* the search — see the hybrid section, which is its
  complement).
- The [Cartpole Evolution notebook](../examples/neuroevolution/cartpole_evo.ipynb)
  is the worked control example.
- A packaged control recipe is being staged at **`spyx.experimental.zoo`** — import
  it as `from spyx.experimental.zoo import ...` — so the population loop, fitness
  shaping, and parameter (un)flattening don't have to be re-derived per study.
- The neuroevolution notebooks under
  [`research/misc/`](https://github.com/kmheckel/spyx/tree/main/research/misc)
  (`nmnist_evo_*`, `shd_evo_*`) are additional starting points.

## 1st-order — surrogate gradient (the workhorse)

**Idea.** The spike \(S = H(V-\theta)\) has a derivative that is zero almost
everywhere and undefined at threshold, so backprop-through-time (BPTT) would see
no gradient. A **surrogate gradient** keeps the hard Heaviside on the *forward*
pass but substitutes a smooth, bounded pseudo-derivative on the *backward* pass.
The network is then trained by ordinary BPTT with Optax.

**When to use it.** This is the **default** for almost everything: classification,
sequence modelling, and any task where you have labelled data and a
(surrogate-)differentiable forward pass. It is by far the most sample-efficient
option here and scales with the usual deep-learning machinery (`jit`, `vmap`,
Optax, quantization).

**The trade-off.** The gradient is *biased* — it is not the gradient of the
hard-spike loss, because that gradient does not exist. The surrogate is a modelling
choice (its shape and steepness `k` are hyperparameters), and there is a subspace
of directions in which the surrogate is systematically blind to how the true
hard-spike loss changes. Usually the bias is benign; when it is not, the hybrid
method below exists to correct it.

**Spyx entry point.**

- [`spyx.axn`](../reference/axn.md) provides the surrogate factories —
  `superspike`, `arctan`, `tanh`, `triangular`, `boxcar` — each a
  `jax.custom_gradient` you attach to a neuron — plus `heaviside`, the plain
  step whose gradient is straight-through.
- [`spyx.optimize.fit`](../reference/optimize.md) (and `make_train_step` /
  `make_eval_step`) wrap the canonical `nnx.Optimizer(model, tx, wrt=nnx.Param)`
  + `nnx.value_and_grad` loop so you don't re-write the epoch boilerplate.
- The [Surrogate Gradient Tutorial](../examples/surrogate_gradient/SurrogateGradientTutorial.ipynb)
  is the end-to-end walkthrough, with the
  [surrogate](../examples/surrogate_gradient/shd_sg_surrogate_comparison.ipynb) and
  [neuron-model](../examples/surrogate_gradient/shd_sg_neuron_model_comparison.ipynb)
  comparisons as follow-ups.

See the [SNN primer](snn-primer.md) for the mechanics of the surrogate and why the
hard spike has no usable gradient.

## Transfer — conversion & quantization-aware training

**Idea.** Don't train the SNN from scratch — **inherit** from an already-trained
model, then adapt. Two flavours:

- **ANN→SNN conversion.** Map a trained real-valued network onto a spiking one by
  reinterpreting activations as firing rates (rate coding), so a ReLU MLP/CNN
  becomes a rate-coded SNN with little or no retraining.
- **Quantization-aware training (QAT).** Start from an fp32 SNN and *fine-tune it
  to be robust to low-precision weights/activations*, so it stays accurate after
  int8 / int4 / ternary quantization for deployment.

**When to use it.** You already have a strong fp32 model (yours or off the shelf)
and want a spiking or low-precision version without paying full training cost, or
you are targeting an efficiency/energy budget (memory-bound deployment,
neuromorphic silicon) and need the accuracy/precision frontier quantified.

**The trade-off.** Conversion inherits the source model's inductive biases and
typically needs longer time windows `T` to let rates converge, trading latency for
accuracy; it rarely beats a natively-trained temporal SNN on genuinely temporal
tasks. QAT adds a fake-quantization step to every training iteration and its own
hyperparameters (which layers, which qtype).

**Spyx entry point.**

- [`spyx.quant`](../reference/quant.md) supplies the QAT machinery: `quantize(model,
  *example_inputs, mode="qat")` plus rule builders — `linear_only_rules`,
  `weights_only_rules`, `bitnet_ternary_rules`, and `spiking_feedforward_rules`
  (weight-only, **lossless on binary spike activations** because a spike is already
  exactly `{0,1}` on the integer grid). `binary_activation_error` proves that
  losslessness argument / catches graded surrogate activations.
- The [Quantization-Aware Training notebook](../examples/quantization/qat_intro.ipynb)
  is the walkthrough; the [quantize how-to](../how-to/quantize.md) is the recipe.
- Conversion and transfer studies live under
  [`research/new/`](https://github.com/kmheckel/spyx/tree/main/research/new) —
  e.g. [`ssm_to_spiking_transfer/`](https://github.com/kmheckel/spyx/tree/main/research/new/ssm_to_spiking_transfer)
  (transferring SSM dynamics into a spiking model) and
  [`pretrain_finetune_curriculum/`](https://github.com/kmheckel/spyx/tree/main/research/new/pretrain_finetune_curriculum)
  — and the [`reproductions/qs5/`](https://github.com/kmheckel/spyx/tree/main/research/reproductions/qs5)
  quantized-S5 reproduction.

!!! note "A binary-spike quantization win"
    Because spike activations are already binary, quantizing **only the weights** on
    the spike→Linear path adds *zero* activation-side error — the recipe
    `spyx.quant.spiking_feedforward_rules` builds exactly this, and it is a genuine
    free efficiency gain rather than an accuracy trade.

## Local / bio-inspired — mostly roadmap

**Idea.** Global BPTT is biologically implausible and memory-hungry (it stores the
whole `[T, B, …]` activation history). The *local learning* family replaces the
exact backward pass with signals available locally in space and/or time:

- **Feedback alignment** — replace the transposed forward weights in the backward
  pass with fixed random feedback, removing weight transport.
- **Synthetic / decoupled gradients** — a small auxiliary network *predicts* the
  gradient for a layer, so layers update without waiting for a full backward pass.
- **e-prop (eligibility propagation)** — an online, forward-in-time approximation
  to BPTT for recurrent SNNs using per-synapse eligibility traces, so you never
  unroll the whole sequence.

**Honest status in Spyx.** These are **research directions, not shipped APIs**.
Spyx today gives you the substrate they need — `jax.custom_gradient` surrogates in
[`spyx.axn`](../reference/axn.md), plain `nnx.Param` state you can attach traces to,
and the `(x, state) -> (out, state)` neuron contract that an online rule slots into
— but there is **no** `spyx.axn.feedback_alignment` or `spyx.optimize.eprop` to
call. Treat this family as **aspirational**:

- **Synthetic / decoupled gradients** — tracked in
  [issue #27](https://github.com/kmheckel/spyx/issues/27).
- **e-prop** — tracked in [issue #28](https://github.com/kmheckel/spyx/issues/28).

**When it will matter.** Online, memory-bounded training (long sequences where
storing the BPTT history is the bottleneck) and neuromorphic on-chip learning,
where locality is a hardware constraint rather than a preference. Until the issues
above land, use surrogate-gradient BPTT and, for the memory pressure specifically,
[`spyx.experimental.compress`](../reference/experimental.md) (bit-packed activation
storage) as the pragmatic stopgap.

## 0+1 hybrid — evolutionary error-correction of the surrogate gradient

**Idea.** Combine the two cheapest sources of information so each covers the
other's blind spot. The surrogate gradient (1st-order) gives a cheap, low-variance
**bulk descent direction** — but it is *biased*, blind to a subspace of how the
**true hard-spike loss** actually moves. So estimate a small correction in exactly
that blind subspace with a zeroth-order method:

1. Take the surrogate gradient \(g_s\) as the bulk step (cheap, most of the signal).
2. Form a small **antithetic ES** estimate \(g_h\) of the gradient of the *true
   hard-spike loss* — a handful of paired \(\pm\varepsilon\) perturbations scored on
   the real, non-differentiable forward pass.
3. **Project \(g_h\) orthogonal to \(g_s\)** and add it back as an error correction:

$$
g \;=\; g_s \;+\; \big(g_h - \tfrac{\langle g_h, g_s\rangle}{\langle g_s, g_s\rangle}\, g_s\big).
$$

The projection is the whole point: the surrogate already owns the direction it
points, so you only spend ES samples on the **complementary** subspace it cannot
see. This is precisely the **complement of Guided-ES**, which does the opposite —
it searches *within* the surrogate-gradient subspace to accelerate along a
direction it already trusts. Here we spend the samples *outside* that subspace,
where the surrogate is wrong.

**When to use it.** The surrogate-gradient model trains but plateaus or drifts
from the hard-spike model you deploy — i.e. the surrogate bias is costing you real
accuracy — and you can afford a modest population of extra hard-forward
evaluations per step. It is a targeted fix, not a default; reach for it when plain
1st-order under-delivers on the *hard* objective.

**The trade-off.** Each step now costs the surrogate backward pass **plus** a small
antithetic ES batch of hard-spike forward passes — more compute per update than
pure 1st-order, far less than pure 0th-order — in exchange for a descent direction
that is unbiased in the subspace the surrogate misses.

**Spyx entry point.** The concrete implementation is being staged at
**`spyx.experimental.hybrid`** (import as `from spyx.experimental.hybrid import
...`). Until it lands you can assemble the pieces by hand:
[`spyx.axn`](../reference/axn.md) for \(g_s\), `evosax`'s antithetic sampling for
\(g_h\), and a projection you drop into your `nnx.value_and_grad` step.

## Summary

| Method | Uses | Best for | Cost | Spyx entry point |
|---|---|---|---|---|
| **Evolutionary** | loss values | control/RL, non-differentiable or hard-spike objectives | many forward evals | `evosax` + `spyx.experimental.zoo` |
| **Surrogate gradient** | surrogate loss gradient | classification, sequence modelling — the default | 1 fwd+bwd / step | [`spyx.axn`](../reference/axn.md) + [`spyx.optimize.fit`](../reference/optimize.md) |
| **Conversion & QAT** | a pretrained model | deploying an existing model spiking / low-precision | fine-tune, longer `T` | [`spyx.quant`](../reference/quant.md) |
| **Local / bio-inspired** | local signals | online / on-chip learning | — (roadmap) | issues [#27](https://github.com/kmheckel/spyx/issues/27), [#28](https://github.com/kmheckel/spyx/issues/28) |
| **0+1 hybrid** | surrogate grad + hard-spike ES | fixing surrogate bias on the hard objective | fwd+bwd **+** small ES batch | `spyx.experimental.hybrid` |

Now pick one for *your* task and architecture: [Choosing an approach](choosing-an-approach.md).
