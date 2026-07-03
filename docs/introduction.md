# Introduction

Spyx is a compact spiking neural network (SNN) library built on JAX and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). It is designed to deliver PyTorch-class ergonomics on top of JAX's JIT and auto-differentiation so researchers can iterate quickly on GPUs and TPUs, then export trained models to neuromorphic silicon via the [Neuromorphic Intermediate Representation](https://nnir.readthedocs.io/) (NIR).

## Design principles

1. **JAX-first.** Every layer composes cleanly with `jax.jit`, `jax.vmap`, `jax.lax.scan`, and `jax.lax.associative_scan`. Nothing in Spyx blocks JIT compilation or forces state out to the host.
2. **NNX modules, not transforms.** Every stateful piece — neurons, SSMs, phasor layers — inherits from `flax.nnx.Module`. Parameters are plain `nnx.Param` nodes, so standard Optax workflows (`nnx.Optimizer` + `nnx.value_and_grad`) work out of the box.
3. **Surrogate gradients are `jax.custom_gradient`.** No hand-rolled backward passes, no PyTree gymnastics. You can use any autograd tool that works on plain JAX.
4. **Functional core.** Losses, regularisers, and data transforms in `spyx.fn` and `spyx.data` are pure functions wrapped in `jax.jit`, so they compose with anything.
5. **Optional extras, lean core.** The default install is minimal. Data loaders (`tonic`) and quantization (`qwix`) live behind `[loaders]` and `[quant]` extras.

## Module map

| Module | What it contains |
|---|---|
| [`spyx.nn`](api.md#neuron-models) | Spiking neuron models (IF, LIF, ALIF, CuBaLIF, recurrent variants), `Sequential`, and the time-major `run` helper. |
| [`spyx.axn`](api.md#surrogate-gradients) | Surrogate-gradient factories (SuperSpike, Arctan, Tanh, Boxcar, Triangular, straight-through). |
| [`spyx.fn`](api.md#losses-and-metrics) | Losses and metrics — `integral_crossentropy`, `integral_accuracy`, `silence_reg`, `sparsity_reg`, `mse_spikerate`. |
| [`spyx.data`](api.md#data-loading) | Grain-based data pipeline with rate / angle / latency encoding and `SHD_loader` / `NMNIST_loader`. |
| [`spyx.optimize`](api.md#training-helpers) | Quick-training helpers — `fit`, `make_train_step`, `make_eval_step`. |
| [`spyx.nir`](api.md#nir-interoperability) | NIR import/export for interop with neuromorphic toolchains. |
| [`spyx.quant`](api.md#quantization-optional) | `qwix`-backed int8 / int4 / BitNet-ternary quantization-aware training. |
| [`spyx.ssm`](api.md#state-space-models) | State-space layers — LRU, S5Diag, Mamba, MambaBlock, ChunkedSSM (H-Net skeleton). *Feature branch.* |
| [`spyx.phasor`](api.md#phasor-networks) | Complex-valued phasor networks with spike-time conversion. *Feature branch.* |
| [`spyx.experimental`](api.md#experimental) | Research-grade neurons (SPSN, stochastic-associative LIF/CuBaLIF, PSU_LIF). |

## Where Spyx fits

| | PyTorch SNNs (snnTorch, Norse, SpikingJelly) | Spyx |
|---|---|---|
| Framework | PyTorch eager | JAX JIT + vmap |
| Module system | `torch.nn.Module` | `flax.nnx.Module` |
| Accelerator | CUDA | CUDA / ROCm / TPU |
| Speed | Depends on custom kernels | Fully JIT-compiled, competitive with custom kernels |
| Quantization | external (TorchAO, BitsAndBytes) | `spyx.quant` (qwix) |
| Neuromorphic export | NIR | NIR |

Spyx is the right choice if you want JAX's compilation model and benefit from `vmap` across populations (e.g. neuroevolution, per-seed sweeps) without paying the price of hand-writing CUDA kernels for SNN dynamics.

## Workshop talk

For a guided walkthrough, see the [Spyx Workshop](https://www.youtube.com/live/gKNegntASLI?si=Jz6In4CfPYL3fuUd). Note that the video predates the Flax NNX migration; the concepts carry over but the API surface has since moved to match what's documented here.
