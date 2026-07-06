# Substrates & the hardware lottery

Should a spiking model use **hard-reset** neurons, or is the real intersection of
"neuromorphic ideas" and today's accelerators a **quantized, matmul-free,
linear-recurrence** model (a low-precision SSM)? The answer is not one ranking —
it splits by *which hardware you are actually targeting*. This page is the
substrate-level companion to [Choosing an approach](choosing-an-approach.md) (which
picks a *training method*); here we pick the *silicon*.

!!! note "What is measured vs. what is a position"
    The **spyx numbers** below (profiling ratios, SSM-vs-spiking, the binary-spike
    quant win) are measured and reproducible in this repo. The **substrate
    argument** and the **frontier section** (NVFP4, structured sparsity, dynamic
    block-skipping) are a reasoned position and a roadmap — flagged as such, not
    settled fact.

## Two lotteries, not one

Sara Hooker's *hardware lottery*: ideas win when they fit the hardware that
happens to exist, not (only) on merit. There are two lotteries in play here, and
they reward opposite things:

| | **Event-driven neuromorphic** | **Dense-parallel (the GPU/LLM lottery)** |
|---|---|---|
| Examples | Loihi 2, Akida, SpiNNaker 2, memristor / analog crossbars | NVIDIA/AMD GPUs, TPUs |
| Rewards | sparse, asynchronous, stateful, low-precision *events* | parallelizable, matmul-dense, high arithmetic intensity, block-regular |
| Energy win from | doing **nothing** between spikes | doing **everything** at once, cheaply per FLOP |

A design that is optimal on one is usually a tax on the other. "Which model is
best" is under-specified until you name the substrate.

## Hard reset: correct on one substrate, a tax on the other

The hard reset (`V ← V − Sₜ·θ`, see [Parallel spiking neurons](parallel-spiking-neurons.md))
is the mechanism that turns integration into **discrete, sparse events**. That
mechanism only pays off where the hardware is built to exploit sparsity:

- **On event-driven silicon it is native and free.** The neuron circuit *is* a
  hard-reset integrator; the async fabric does work only when a spike occurs, so a
  sparse temporal workload (event cameras, always-on audio) runs at µW where a
  clocked accelerator cannot. Here hard reset is simply correct and SSMs are beside
  the point.
- **On a GPU it is the worst of both worlds.** The reset breaks the linear
  recurrence, forcing a **sequential scan** (no parallel prefix); the discontinuity
  needs **surrogate gradients** (training bias — the reason
  [`spyx.experimental.hybrid`](../reference/experimental.md) exists); and a "spike"
  is just a 0/1 float going through the *same dense matmul*, so you get **none** of
  the event-driven energy payoff. You pay spiking's costs and keep the matmul cost.

spyx's own profiling makes the GPU penalty concrete (`research/new/pallas_neurons/`,
Radeon 8060S): the sequential hard-reset scan is **2–21× slower** than the
associative-scan (reset-free) neuron, while the neuron itself is only **21–45 %** of
the `Linear` it feeds — the matmul, not the neuron, is the bottleneck.

**Rule of thumb:** reach for hard reset when the deployment target is event-driven;
almost never otherwise.

## What actually transfers to the GPU lottery

Two neuromorphic-flavored properties *do* transfer, and one does not. Keep the two,
drop the reset:

1. **Linear, parallelizable recurrence (SSM).** Rides the parallel-scan +
   tensor-core ticket and trains without surrogate bias. spyx's own SSM-vs-spiking
   studies land here honestly: the diagonal S5 (`spyx.ssm`) generally *beat* the
   spiking models on the tasks tried, with routing-memory advantages
   (`experimental.raven`) real but modest and regime-dependent.
2. **Low precision / matmul-free weights.** Ternary (BitNet) and shift-add
   (DeepShift) collapse the multiply into accumulation / bit-shift —
   [`spyx.experimental.matfree`](../reference/experimental.md). This is the *real*
   portable energy lever because it attacks the measured bottleneck (the matmul),
   and it rides the same wave the LLM frontier is on (BitNet b1.58, 1-bit LLMs). It
   is "accumulation not multiplication" — neuromorphic in spirit — in the form the
   GPU lottery rewards.

There is even a place the two worlds *compound* rather than compete: because spikes
are already binary, quantizing **only the weights** on the spike→`Linear` path adds
zero activation-side error — the free win behind `spyx.quant.spiking_feedforward_rules`.

## Frontier: reclaiming *temporal* sparsity on dense hardware

The open prize is the one event-driven silicon gets for free and dense hardware
throws away: most `(neuron, timestep)` entries in a spike tensor are zero, yet a
GPU computes the whole dense block anyway. Three levers, in increasing difficulty —
and increasing payoff:

**1. Quantize the matmul to the floor (NVFP4 / MXFP4).** Blackwell's block-scaled
4-bit float (NVFP4: E2M1 with an FP8 per-16 block scale; MXFP4: the OCP variant, E8M0
per-32) gives near-int4 matmul throughput at better accuracy. This is orthogonal to
sparsity — it makes the dense matmul *cheaper*, not smaller — but it pairs unusually
well with spikes: with **binary activations**, the spike→`Linear` matmul is already
"select and accumulate weight columns," so **binary spikes × NVFP4 weights** is a
natural, high-arithmetic-intensity fit and the obvious next step for `matfree` /
`spyx.quant`. *This lever is available today; it is the safe bet.*

**2. Structured weight sparsity (2:4).** Ampere+ Sparse Tensor Cores give 2× on a
2-of-4 zero pattern. The catch is a **structural mismatch**: 2:4 is *static,
structured, weight-side*, whereas spike sparsity is *dynamic, unstructured,
activation-side, and temporal*. So 2:4 accelerates the weights, not the "skip the
quiet neurons" you actually want. The bridge — speculative — is to *impose*
structure on firing (k-winners-take-all, block-sparse spike codes) so that active
events line up with a sparse-matmul tile. Whether an imposed firing structure keeps
accuracy is an open research question, not a free lunch.

**3. Dynamic / block-sparse computation — the real path.** To exploit *temporal*
sparsity you skip work at the granularity the hardware can schedule:

- **Block-skip quiet regions.** If no neuron in a layer fires at a timestep (or in a
  tile), skip that matmul entirely via control flow / block-sparse kernels. Gains
  scale with how *clustered* the silence is.
- **Gather–scatter active events.** Compact the firing rows, run a *dense* matmul on
  the compacted form, scatter back — the software analog of event-driven routing.
- **Ragged / block-sparse event batching** (Pallas or `jax.experimental.sparse`
  block kernels) over jagged event streams.

The honest caveat, and where intuition about TPUs needs care: **fine-grained,
unstructured sparsity loses on dense accelerators.** A systolic array / tensor core
wants dense, statically-shaped tiles; scattering individual random spikes collapses
arithmetic intensity and gather/scatter overhead eats the theoretical saving. The
win is real only when sparsity is **high *and* block-structured** — which is exactly
why MoE routing and block-sparse attention are dense-within-block. So the user
intuition ("skip empty sections of spiking data") is the right *goal*; the
engineering reality is it must be made **coarse and structured** to beat dense.
spyx already holds the storage half of this — bit-packed spikes in
[`spyx.experimental.compress`](../reference/experimental.md) — the missing half is a
block-sparse consumer kernel.

## Synthesis

- **Targeting event-driven silicon / sub-mW sensing?** Hard-reset spiking on the
  chip is the point; a ternary SSM on a GPU does not serve that goal.
- **Targeting the datacenter / edge-GPU efficiency frontier?** Quantized,
  matmul-free, linear-recurrence models are the stronger bet; hard reset buys
  nothing there.
- **Want optionality across both?** The graceful-degradation play is a **reset-free
  spiking SSM with ternary weights**: binary *events* (sparsity, a path to a
  neuromorphic backend) on *linearizable* dynamics (GPU-trainable via associative
  scan, per [`spyx.experimental.PSU_LIF`](../reference/experimental.md)), matmul
  driven toward NVFP4/ternary. That is the intersection spyx's measured results keep
  pointing at — neuromorphic in spirit, without betting on the hardware that lost
  the lottery.

This is a position on a live question, not a verdict. The ternary-at-scale accuracy
gap is narrowing but real, and neuromorphic-vs-dense energy is workload-dependent —
neuromorphic wins specifically on *sparse temporal* data. Match the model to the
substrate you are actually shipping to.
