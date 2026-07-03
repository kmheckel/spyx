# Welcome to Spyx!

Spyx (pronounced "spikes") is a compact spiking neural network (SNN) library built on JAX and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). It delivers the flexibility and extensibility of PyTorch-based SNN libraries while training at speeds comparable to — or faster than — frameworks that hand-write custom CUDA kernels, thanks to JAX's JIT compilation and auto-differentiation. Trained models can be exported to neuromorphic hardware via the [Neuromorphic Intermediate Representation](https://nnir.readthedocs.io/) (NIR).

Be sure to give it a star on GitHub: [kmheckel/spyx](https://github.com/kmheckel/spyx)

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

**Looking up a signature?** Exhaustive, auto-generated API documentation for every public module: [`spyx.nn`](reference/nn.md), [`spyx.axn`](reference/axn.md), [`spyx.fn`](reference/fn.md), [`spyx.data`](reference/data.md), [`spyx.optimize`](reference/optimize.md), [`spyx.nir`](reference/nir.md), [`spyx.quant`](reference/quant.md), [`spyx.ssm`](reference/ssm.md), [`spyx.phasor`](reference/phasor.md), [`spyx.experimental`](reference/experimental.md).

### :material-lightbulb-on: Explanation — *understanding-oriented*

**Want to understand why?** Background and design discussion, read away from the keyboard.

- [Design and architecture](explanation/design.md) — why Spyx is built the way it is, the module map, and how it compares to PyTorch SNN libraries.
- [A primer on spiking neural networks](explanation/snn-primer.md) — spikes, LIF dynamics, surrogate gradients, and rate vs. latency coding.

## For developers

If you're contributing to Spyx or working with AI coding agents, check out [AGENTS.md](https://github.com/kmheckel/spyx/blob/main/AGENTS.md) for a comprehensive overview of the project structure, development workflow, and coding standards.
