# AGENTS.md

This document provides a concise overview of the Spyx project structure for AI coding agents and developers.

## Project Overview

Spyx is a spiking neural network (SNN) library built on JAX and Flax NNX. It provides a compact, high-performance framework for training SNNs via surrogate gradient descent and neuroevolution, and — increasingly — a research vehicle for efficient sequence modeling (SNNs, state-space models, phasor/complex networks, and quantization under one roof).

## Stable core vs. `spyx.experimental`

Spyx has two tiers, and agents should respect the boundary when advising users:

- **Stable core** — the supported, API-stable surface: `nn`, `ssm`, `phasor`,
  `nir`, `bench`, `quant`, `data`, `optimize`, `fn`, `axn`. Rely on these for
  anything a user will depend on.
- **`spyx.experimental`** — research-stage building blocks whose **API may change
  without a deprecation cycle**: `PSU_LIF`, `ResonateFire`, `raven` (RavenRSM +
  SpikingSlotMemory), `compress` (packed-bit activations), `stochastic` (SPSN,
  stochastic-associative neurons). Always import these from `spyx.experimental`
  (e.g. `from spyx.experimental import PSU_LIF, RavenRSM`), never from a top-level
  module, so usage signals the stability contract. `PSU_LIF`/`ResonateFire` are
  physically defined in `nn`/`phasor` for code locality but are *surfaced* here.

When something in `experimental` matures (stable API, tests, docs, a real use
case), it graduates into the core namespace in a minor release.

## Technology Stack

- **Core Framework**: JAX (JIT compilation, automatic differentiation)
- **Neural Network Library**: Flax NNX (module system)
- **Optimization**: Optax (gradient-based optimization)
- **Data Loading**: Google Grain (data pipelines)
- **Package Management**: uv
- **Testing**: pytest
- **Linting**: Ruff
- **Documentation**: MkDocs with Material theme

## Directory Structure

```
spyx/
├── src/spyx/           # Main library code
│   ├── __init__.py     # Package initialization, public API exports
│   ├── axn.py          # Surrogate gradient functions (activation/axon)
│   ├── bench.py        # Benchmark harness (latency, throughput, MFU, spike-rate)
│   ├── data.py         # Data loading utilities and Grain transforms
│   ├── fn.py           # Functional utilities (losses, metrics, regularizers)
│   ├── nir.py          # NIR (Neuromorphic Intermediate Representation) support
│   ├── nn.py           # Neuron models (LIF, ALIF, CuBaLIF, etc.)
│   ├── optimize.py     # High-level training loop (fit, make_train/eval_step)
│   ├── phasor.py       # Complex-valued phasor & spiking-phasor networks
│   ├── quant.py        # int8/int4/BitNet quantization (qwix wrapper, optional)
│   ├── ssm.py          # State-space layers (LRU, S5Diag, Mamba, ChunkedSSM)
│   ├── experimental/   # Research-stage, UNSTABLE API (see below)
│   │   ├── __init__.py #   PSU_LIF, ResonateFire (re-exported), + the modules:
│   │   ├── raven.py    #   RavenRSM routing-slot memory + SpikingSlotMemory
│   │   ├── compress.py #   bit-packed activation storage for BPTT memory
│   │   └── stochastic.py #  SPSN, StochasticAssociative*, sigmoid_bernoulli
│   └── _version.py     # Version information
├── tests/              # Test suite (conftest.py pins JAX to CPU + seeds fixtures)
├── docs/               # MkDocs docs, organized by Diátaxis
│   ├── tutorials/      # Learning-oriented lessons
│   ├── how-to/         # Goal-oriented guides
│   ├── reference/      # API reference (mkdocstrings)
│   ├── explanation/    # Background & design
│   └── examples/       # Tutorial notebooks
├── research/           # Research experiments and benchmarks
├── scripts/            # Smoke tests, demos, release automation
├── .claude/            # Agent skills, SessionStart hook, settings
├── .github/workflows/  # CI (ci.yml) + PyPI publish (python-publish.yml)
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # User-facing documentation
```

## Core Modules

### `axn.py` - Surrogate Gradients
Defines surrogate gradient functions for backpropagation through spiking neurons. Each is a factory that returns a JIT-compiled `jax.custom_gradient` function:
- `custom(bwd, fwd)`: Build an activation with arbitrary forward / surrogate-gradient functions
- `heaviside()`: The forward spiking nonlinearity
- `superspike(k=25)`: SuperSpike surrogate gradient (Zenke & Ganguli, 2018)
- `arctan(k=2)`: Arctangent surrogate gradient
- `triangular(k=2)`: Triangular surrogate gradient
- `boxcar(width=2, height=0.5)`: Boxcar surrogate gradient
- `tanh(k=1)`: Hyperbolic-tangent surrogate gradient

### `nn.py` - Neuron Models
Spiking neuron implementations using Flax NNX:
- **IF**: Integrate-and-Fire
- **LIF**: Leaky Integrate-and-Fire
- **LI**: Leaky Integrator (non-spiking output layer)
- **ALIF**: Adaptive LIF (with threshold adaptation)
- **CuBaLIF**: Current-based LIF (separate current and voltage dynamics)
- **Recurrent variants**: RIF, RLIF, RCuBaLIF
- **Sequential**: Container for stateful layer composition
- **SumPool**: Spatial pooling for spike trains
- **ActivityRegularization**: Mutable spike-count tracker for per-layer regularization
- **`run(model, x, state=None)`**: Time-major scan helper (`[T, B, ...]` input → `[T, B, ...]` output, final state)

All neuron models follow a consistent interface:
- `__init__`: Initialize parameters with optional random initialization
- `__call__(x, state)`: Forward pass returning (output, new_state)
- `initial_state(batch_size)`: Create initial state tensors

### `data.py` - Data Loading
Grain-based data loading with SNN-specific transforms:
- **Functional transforms**: `rate_code()`, `angle_code()`, `latency_code()`, `shift_augment()`
- **Grain transforms**: `RateCode`, `AngleCode`, `LatencyCode`, `ShiftAugment`
- **Dataset loaders**: `NMNIST_loader`, `SHD_loader` (requires tonic)
- Integration with Tonic for neuromorphic datasets

### `fn.py` - Training Utilities
Functional training and evaluation tools (factory functions returning JIT-compiled callables):
- `integral_accuracy(time_axis=1)`: Argmax-of-summed-traces accuracy + predictions
- `integral_crossentropy(smoothing=0.3, time_axis=1)`: Softmax cross-entropy on summed traces with optional label smoothing
- `mse_spikerate(sparsity=0.25, smoothing=0.0, time_axis=1)`: MSE between mean spike rate and a target sparsity
- `silence_reg(min_spikes)`: Penalize neurons that spike below a target rate
- `sparsity_reg(max_spikes, norm=optax.huber_loss)`: Penalize layers whose mean spiking exceeds a target

For training, drive the SNN with `nn.run(model, x)` and combine with an `nnx.Optimizer` + `nnx.value_and_grad` loop; there is no built-in `update_step` helper.

### `optimize.py` - High-level Training Loop
Convenience helpers that wrap the canonical `nnx.Optimizer` + `nnx.value_and_grad` pattern:
- `fit(model, tx, loss_fn, train_iter, *, epochs, eval_iter=None, eval_fn=None, on_epoch_end=None)`: end-to-end epoch loop.
- `make_train_step(loss_fn)` / `make_eval_step(metric_fn)`: JIT-compiled step primitives for custom loops.

### `nir.py` - Neuromorphic Intermediate Representation
Import/export to NIR format for interoperability:
- `from_nir()`: Convert NIR graph to Spyx model
- `to_nir()`: Export Spyx model to NIR graph
- Support for standard NIR nodes (LIF, Conv2d, Linear, etc.)
- RNN subgraph handling for recurrent models

### `bench.py` - Benchmark harness
Measure any spyx module/neuron with correct methodology (JIT once, warmup, median,
`block_until_ready` before timing):
- `benchmark(module, input_shape, *, seq_len, batch, run_fn=None, ...)` → `BenchResult`
  (fwd / fwd+bwd latency, throughput, best-effort peak memory, XLA-cost-model
  FLOPs/MFU, and **spike-rate as an SNN energy proxy**).
- `compare({name: module}, ...)` sweeps models × configs; `format_table(results)`
  pretty-prints. Pass `run_fn=` to drive a module a specific way (e.g. `spyx.nn.run`
  vs a neuron's own `.parallel` path).

### `ssm.py` / `phasor.py` - Sequence & complex layers
- `ssm.py`: diagonal complex state-space layers — `LRU`, `S5Diag` (HiPPO-init),
  `Mamba`/`MambaBlock` (selective), `ChunkedSSM` — all parallelized with
  `jax.lax.associative_scan`.
- `phasor.py`: complex-valued `PhasorLinear`/`PhasorActivation`/`PhasorReadout`/
  `PhasorMLP`, `SpikingPhasor`, and phase↔spike helpers.

### `experimental/` - Research-stage (UNSTABLE API)
Import from `spyx.experimental` (see the "Stable core vs. `spyx.experimental`"
section above for the contract):
- `PSU_LIF` — reset-free parallel LIF: a `(x, V) -> (spikes, V)` step **and** a
  `.parallel(x)` associative-scan path (O(log T) depth). Physically in `nn`.
- `ResonateFire` — complex resonate-and-fire oscillator; sequential + `.parallel`.
  Physically in `phasor`.
- `raven` — `RavenRSM` (Routing Slot Memories: sparse-routed slot memory with a
  `SlotRouter`), `SpikingSlotMemory` (spiking sibling), `make_recall_batch` (MQAR
  task). After Raven (Afzal, Bick, Xing, Cevher, Gu 2026).
- `compress` — `packed_spike_dense` (a `custom_vjp` matmul that stores its backward
  residual bit-packed for memory-efficient BPTT) + `pack_spikes`/`unpack_spikes`.
- `stochastic` — `SPSN`, `StochasticAssociativeLIF`/`CuBaLIF`, `sigmoid_bernoulli`.

### `quant.py` - Quantization (optional)
Thin SNN-aware wrapper around Google's `qwix` library:
- `quantize(model, *example_inputs, rules=None, mode="qat" | "ptq")`: convert an SNN to a quantized version; defaults to int8 weights+activations on Linear/Conv only.
- `linear_only_rules(weight_qtype, act_qtype)`: shorthand qwix rules that match only dense layers (spiking dynamics stay fp32).
- `weights_only_rules(weight_qtype)`: weight-only quantization for memory-bound deployment.
- `bitnet_ternary_rules(act_qtype="int8")`: BitNet b1.58-style ternary weights (via `int2`) + int8 activations for dense / conv layers.
- `spiking_feedforward_rules(weight_qtype="int8")`: weight-only recipe for the spike→Linear path that is **lossless on binary activations** — because a spike is exactly `{0,1}` it already lies on the integer grid, so quantizing only the weights adds zero activation-side error (recurrent/einsum stays fp32, per Q-S5).
- `binary_activation_error(spikes, *, weight_qtype="int8")`: qwix-free check that returns `0.0` iff the activations are truly binary (proves the losslessness argument / catches graded surrogate activations).
- `available()`: returns True iff qwix is importable (the dependency lives behind the `[quant]` extra).

## Development Workflow

### Setup
```bash
git clone https://github.com/kmheckel/spyx
cd spyx
uv sync
```

### Code Quality
```bash
uv run ruff check          # Lint code
uv run ruff check --fix    # Auto-fix issues
uv run ruff format         # Format code
```

### Testing
```bash
uv run pytest -m "not network"   # Full suite minus dataset-downloading tests (what CI runs)
uv run pytest                     # Everything, including network-gated loader tests
uv run pytest -v                  # Verbose output
uv run pytest tests/test_data_grain.py  # Specific test
uv run pytest --cov=spyx          # With coverage (pytest-cov + [tool.coverage] config)
```

Tests that download a dataset carry `@pytest.mark.network` and are excluded
from CI. Keep any new network-dependent test behind that marker. `conftest.py`
forces JAX onto CPU and provides seeded `rngs` / `key` fixtures.

### Continuous integration
`.github/workflows/ci.yml` runs four jobs on every PR: `lint` (ruff check +
format), `test` (pytest matrix over Python 3.11/3.12 with the quant extra),
`docs` (`mkdocs build --strict`), and `smoke` (notebook-API drift). Match all
four locally before pushing. `python-publish.yml` builds and publishes to PyPI
when a GitHub release is published.

### Notebook smoke tests
Before tagging a release (or whenever dependencies shift), verify that every
published tutorial's code path still matches the `src/spyx/` API without
requiring dataset downloads:

```bash
uv run python scripts/smoke_notebook_apis.py
```

This exercises the model-construction + one `nnx.Optimizer` step for each of
the five surrogate-gradient/cartpole notebooks plus the QAT tutorial on
synthetic data. Runs in ~10 seconds; catches the kind of API drift that
silently breaks notebooks between `flax` / `evosax` / `qwix` releases.

### End-to-end install check
After `uv sync` on a new machine (especially a GPU laptop), run:

```bash
uv run python scripts/check_install.py
```

Seven PASS/FAIL/SKIP checks in ~30 seconds:
1. JAX version + visible accelerator devices (flags CPU-only install).
2. Core Spyx imports.
3. Forward pass through a `Sequential(Linear, LIF, Linear, LI)` SNN.
4. One full `spyx.optimize.fit` training epoch on synthetic data.
5. NIR export + re-import roundtrip.
6. `smoke_notebook_apis.py` passthrough (all 6 tutorials).
7. Optional-extra availability (`tonic` for loaders, `qwix` for quant).

Exit code is 0 if everything required passed, nonzero on any hard failure.
Missing optional extras are reported but don't fail the run.

### Documentation
```bash
uv run mkdocs serve        # Preview docs locally
uv run mkdocs build        # Build static site
```

## Key Design Principles

1. **JAX-first**: Leverage JAX's JIT compilation, vmap, and automatic differentiation
2. **Functional core**: Most utilities are pure functions for easy composition
3. **NNX modules**: Stateful neuron models use Flax NNX for clean parameter management
4. **Grain pipelines**: Modern data loading with Google Grain for performance
5. **Type hints**: Extensive use of type annotations for clarity
6. **Surrogate gradients**: All spiking operations use differentiable surrogates

## Configuration

### Ruff (`pyproject.toml`)
- Target: Python 3.11–3.12 (`requires-python = ">=3.11, <3.13"`)
- Line length: 88 (Black-compatible)
- Enabled rules: Pyflakes (F), pycodestyle errors (E), flake8-bugbear (B), import sorting (I)
- Excluded: research/, docs/, scripts/, data/

### Dependencies
- **Core**: optax, jax_tqdm, nir, flax>=0.11.0, grain
- **Optional** (`[loaders]`): tonic, numba>=0.59.0
- **Optional** (`[quant]`): qwix (installed from GitHub via `tool.uv.sources`)
- **Dev**: pytest, ruff, mkdocs, mkdocs-material, mkdocstrings

## Common Tasks for Agents

### Adding a new neuron model
1. Create class in `nn.py` inheriting from `nnx.Module`
2. Implement `__init__`, `__call__`, and `initial_state`
3. Add tests in `tests/`
4. Document in `docs/reference/nn.md` (auto-rendered from docstrings)

### Adding a new surrogate gradient
1. Create class in `axn.py` inheriting from `Axon`
2. Implement `fwd()` and `grad()` methods
3. Add factory function for convenience
4. Benchmark against existing surrogates

### Fixing linting errors
```bash
uv run ruff check --fix    # Auto-fix most issues
# Manually fix remaining errors (see ruff output)
uv run ruff check          # Verify all issues resolved
```

### Using Spyx for research (helping a user spin up a study)
Spyx is designed to be a research vehicle, and the `research/` tree is organized
for it (excluded from linting so exploration is friction-free):

- **`research/README.md`** — the taxonomy: `reproductions/` (faithful paper
  repros), `extensions/` (extend a paper), `new/` (novel work), plus a shared
  `_template/` study template. Point users here first.
- **To add a study**: copy `research/_template/`, pick the bucket, and write a
  `run.py` + `README.md`. Model it on an existing study such as
  `research/new/parallel_spiking_neurons/run_study.py` (SHD via a cached `.npz`,
  `spyx.optimize`/manual loop, `spyx.fn` losses/metrics). A `SMOKE=1` synthetic
  mode that runs on CPU in seconds makes studies self-checking.
- **What makes spyx good for this**: the *same* `associative_scan` machinery
  parallelizes SNNs (`experimental.PSU_LIF`, `experimental.ResonateFire`), SSMs
  (`ssm.S5Diag`/`Mamba`), and phasors — so you can swap dynamics behind one
  interface. Use **`spyx.bench`** to measure (latency / MFU / spike-rate), stage
  new ideas under **`spyx.experimental`**, and reach for **`spyx.quant`** +
  `experimental.compress` for efficiency. Report honest results — several existing
  studies are negative/boundary results, and that is the point.
- **Promotion path**: when an experimental module earns a stable API + tests +
  docs + a real use case, graduate it into the core namespace.

## Claude Code workflows

This repository ships skill files under `.claude/skills/` that Claude Code can invoke by name. Each skill is a focused recipe for a common workflow; they compose — e.g. `setup-gpu` runs before `run-tutorial` on a fresh machine.

| Skill | Use when |
|---|---|
| `snn-primer` | You're new to SNNs and want a JAX-ready mental model (spikes, LIF dynamics, surrogate gradients, time-major tensors). |
| `setup-gpu` | JAX is falling back to CPU and you have an NVIDIA GPU that isn't being seen. |
| `smoke-check` | You want one command to verify install + tests + notebook APIs are healthy. |
| `run-tutorial` | You want to open and execute one of the bundled tutorial notebooks. |
| `new-experiment` | You want to scaffold a new Spyx training script against your own dataset. |
| `debug-training` | Your SNN isn't learning: flat loss, NaN, silent neurons, or exploding gradients. |
| `add-neuron-model` | You're implementing a new spiking neuron in `spyx.nn` (the `(x, state) -> (out, state)` contract, tests, NIR export). |
| `quantize-model` | You want int8 / int4 / BitNet-ternary quantization of a model via `spyx.quant`. |
| `sequence-models` | You're adding an SSM (LRU/S5/Mamba/ChunkedSSM) or phasor layer, or mixing them with spiking layers. |
| `nir-export` | You're exporting to / importing from NIR for neuromorphic-hardware interop. |
| `cut-release` | You're bumping the version, tagging, and publishing to PyPI. |

Invoke via `/snn-primer`, `/quantize-model`, etc. from Claude Code. Each skill is self-contained; see the file in `.claude/skills/` for the full instructions.

### Environment automation
`.claude/hooks/session-start.sh` runs on Claude Code on the web sessions and
`uv sync --extra quant`s the environment so tests/linters/docs work on the
first turn. `.claude/settings.json` registers that hook and allowlists common
read-only commands. A `.github/PULL_REQUEST_TEMPLATE.md` structures new PRs.

## Public API

The package exports the following in `src/spyx/__init__.py`:
- `jax`, `jnp` - JAX and JAX NumPy
- `axn` - Surrogate gradients
- `bench` - Benchmark harness (latency, throughput, MFU, spike-rate)
- `data` - Data loading
- `experimental` - Research-stage, **unstable-API** namespace (PSU_LIF,
  ResonateFire, raven, compress, stochastic)
- `fn` - Functional utilities
- `nir` - NIR conversion
- `nn` - Neuron models
- `optimize` - High-level training loop (`fit`, `make_train_step`, `make_eval_step`)
- `phasor` - Complex-valued phasor & spiking-phasor networks
- `quant` - Quantization helpers (qwix wrapper, optional)
- `ssm` - State-space layers (LRU, S5Diag, Mamba, MambaBlock, ChunkedSSM)
- `__version__` - Version string

All exports are defined in `__all__` for explicit API declaration.
