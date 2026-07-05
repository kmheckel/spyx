# Welcome to Spyx!

Spyx (pronounced "spikes") is a compact spiking neural network (SNN) library built on JAX and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). It delivers the flexibility and extensibility of PyTorch-based SNN libraries while training at speeds comparable to — or faster than — frameworks that hand-write custom CUDA kernels, thanks to JAX's JIT compilation and auto-differentiation. Trained models can be exported to neuromorphic hardware via the [Neuromorphic Intermediate Representation](https://nnir.readthedocs.io/) (NIR).

Be sure to give it a star on GitHub: [kmheckel/spyx](https://github.com/kmheckel/spyx)

## The module map

Spyx is a **stable core** plus a clearly-fenced **experimental** staging area.
Build on the core; reach into `spyx.experimental` deliberately, knowing its API
can move without a deprecation cycle.

**Stable core** — public API, kept backwards-compatible:

| Module | One-liner |
|---|---|
| [`spyx.nn`](reference/nn.md) | Spiking neurons (LIF, ALIF, CuBaLIF, LI, IF + recurrent variants), `Sequential`, `Flatten`, and the time-major `run` helper. |
| [`spyx.axn`](reference/axn.md) | Surrogate-gradient factories (SuperSpike, Arctan, Tanh, Boxcar, Triangular, straight-through). |
| [`spyx.fn`](reference/fn.md) | Losses & metrics — `integral_crossentropy`, `integral_accuracy`, `silence_reg`, `sparsity_reg`, `mse_spikerate`. |
| [`spyx.ssm`](reference/ssm.md) | State-space layers — LRU, S5Diag, Mamba, ChunkedSSM. |
| [`spyx.phasor`](reference/phasor.md) | Phasor networks — `PhasorLinear/Activation/Readout/MLP`, `SpikingPhasor`. |
| [`spyx.data`](reference/data.md) | Grain event-data loaders with rate / angle / latency encoding (`SHD_loader`, `NMNIST_loader`). |
| [`spyx.optimize`](reference/optimize.md) | Quick-training helpers — `fit`, `make_train_step`, `make_eval_step`. |
| [`spyx.nir`](reference/nir.md) | NIR import/export for neuromorphic-hardware interop. |
| [`spyx.quant`](reference/quant.md) | `qwix`-backed int8 / int4 / ternary quantization-aware training. |
| [`spyx.bench`](reference/bench.md) | Measurement — latency, throughput, peak memory, FLOP/MFU, spike-rate energy proxy. |

**Experimental** — unstable API under [`spyx.experimental`](reference/experimental.md), the staging area for in-progress research:

| Module | One-liner |
|---|---|
| `spyx.experimental.PSU_LIF` :material-flask: | Reset-free parallel LIF (associative-scan). |
| `spyx.experimental.ResonateFire` :material-flask: | Complex resonate-and-fire oscillatory neuron. |
| [`spyx.experimental.raven`](reference/experimental.md) :material-flask: | Routing-slot memory (`RavenRSM`) + spiking sibling for high-recall sequence modeling. |
| [`spyx.experimental.compress`](reference/experimental.md) :material-flask: | Bit-packed activation storage for memory-efficient BPTT. |
| [`spyx.experimental.stochastic`](reference/experimental.md) :material-flask: | Stochastic (Bernoulli-spiking) & parallelizable prototypes (`SPSN`, ...). |

Import experimental pieces from `spyx.experimental` so the dependency is explicit.
See [Research with Spyx](explanation/research.md) for how work graduates from
here into the core.

## How the documentation is organised

These docs follow the [Diátaxis](https://diataxis.fr) framework: four sections, each answering a different kind of question.

### :material-school: Tutorials — *learning-oriented*

**New to Spyx?** Start here. Guided lessons that take you from an empty environment to a working, trained SNN, one step at a time.

- [Your first SNN](tutorials/first-snn.md) — install Spyx, build a spiking network, and train it on the Spiking Heidelberg Digits dataset.
- Notebook tutorials on [surrogate gradients](examples/surrogate_gradient/SurrogateGradientTutorial.ipynb), [neuroevolution](examples/neuroevolution/cartpole_evo.ipynb), [NIR conversion](examples/nir/conversion.ipynb), [quantization](examples/quantization/qat_intro.ipynb), [state-space models](examples/ssm/ssm_intro.ipynb), and [phasor networks](examples/phasor/phasor_intro.ipynb).

### :material-wrench: How-to guides — *task-oriented*

**Need to get something done?** Short, focused recipes for specific tasks, assuming you already know the basics.

- [Install Spyx](how-to/install.md) · [Train a model](how-to/train.md) · [Quantize a model](how-to/quantize.md)
- [Export / import via NIR](how-to/nir.md) · [Use SSM and phasor layers](how-to/recurrent-layers.md) · [Load event data](how-to/load-data.md)

### :material-book-open-variant: Reference — *information-oriented*

**Looking up a signature?** Exhaustive, auto-generated API documentation for every public module: [`spyx.nn`](reference/nn.md), [`spyx.axn`](reference/axn.md), [`spyx.fn`](reference/fn.md), [`spyx.data`](reference/data.md), [`spyx.optimize`](reference/optimize.md), [`spyx.nir`](reference/nir.md), [`spyx.quant`](reference/quant.md), [`spyx.ssm`](reference/ssm.md), [`spyx.phasor`](reference/phasor.md), [`spyx.bench`](reference/bench.md), [`spyx.experimental`](reference/experimental.md).

### :material-lightbulb-on: Explanation — *understanding-oriented*

**Want to understand why?** Background and design discussion, read away from the keyboard.

!!! tip "Not sure how to train your model? Start here."
    [**Choosing an approach**](explanation/choosing-an-approach.md) turns the
    [training-methods spine](explanation/training-methods.md) into a decision:
    pick a training method by the *kind of information* it uses (evolutionary,
    surrogate-gradient, conversion/QAT, local, or the 0+1 hybrid), then see which
    applications and architectures it fits, and where the Spyx entry points are.

- [Training methods](explanation/training-methods.md) — the method spine: evolutionary, surrogate-gradient, conversion/QAT, local/bio-inspired, and the 0+1 hybrid, each with when-to-use, the trade-off, and the Spyx entry point.
- [Choosing an approach](explanation/choosing-an-approach.md) — decision matrices (method × application, method × architecture) plus a task → application → architecture → method flow.
- [Design and architecture](explanation/design.md) — why Spyx is built the way it is, the module map, and how it compares to PyTorch SNN libraries.
- [A primer on spiking neural networks](explanation/snn-primer.md) — spikes, LIF dynamics, surrogate gradients, and rate vs. latency coding.
- [Parallel spiking neurons](explanation/parallel-spiking-neurons.md) — reset-free, associative-scan neurons that train in `O(log T)` depth.
- [Research with Spyx](explanation/research.md) — how the `research/` corpus is organised and how to add a study.

## For developers

If you're contributing to Spyx or working with AI coding agents, check out [AGENTS.md](https://github.com/kmheckel/spyx/blob/main/AGENTS.md) for a comprehensive overview of the project structure, development workflow, and coding standards.
