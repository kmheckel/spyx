# AGENTS.md

This document provides a concise overview of the Spyx project structure for AI coding agents and developers.

## Project Overview

Spyx is a spiking neural network (SNN) library built on JAX and Flax NNX. It provides a compact, high-performance framework for training SNNs via surrogate gradient descent and neuroevolution.

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
│   ├── data.py         # Data loading utilities and Grain transforms
│   ├── experimental.py # Experimental features (phasor networks, etc.)
│   ├── fn.py           # Functional utilities (training loops, metrics)
│   ├── nir.py          # NIR (Neuromorphic Intermediate Representation) support
│   ├── nn.py           # Neuron models (LIF, ALIF, CuBaLIF, etc.)
│   └── _version.py     # Version information
├── tests/              # Test suite
├── docs/               # MkDocs documentation
│   ├── examples/       # Tutorial notebooks
│   └── *.md           # Documentation pages
├── research/           # Research experiments and benchmarks
├── scripts/            # Utility scripts (release automation)
├── data/               # Dataset storage (gitignored)
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # User-facing documentation
```

## Core Modules

### `axn.py` - Surrogate Gradients
Defines surrogate gradient functions for backpropagation through spiking neurons:
- `Axon`: Base class for activation functions with custom gradients
- `superspike()`: SuperSpike surrogate gradient
- `arctan()`: Arctangent surrogate gradient
- `triangular()`: Triangular surrogate gradient

### `nn.py` - Neuron Models
Spiking neuron implementations using Flax NNX:
- **IF**: Integrate-and-Fire
- **LIF**: Leaky Integrate-and-Fire
- **ALIF**: Adaptive LIF (with threshold adaptation)
- **CuBaLIF**: Current-based LIF (separate current and voltage dynamics)
- **Recurrent variants**: RIF, RLIF, RCuBaLIF
- **Sequential**: Container for stateful layer composition
- **SumPool**: Spatial pooling for spike trains

All neuron models follow a consistent interface:
- `__init__`: Initialize parameters with optional random initialization
- `__call__(x, state)`: Forward pass returning (output, new_state)
- `initial_state(batch_size)`: Create initial state tensors

### `data.py` - Data Loading
Grain-based data loading with SNN-specific transforms:
- **Functional transforms**: `rate_code()`, `angle_code()`, `shift_augment()`
- **Grain transforms**: `RateCode`, `AngleCode`, `ShiftAugment`
- **Dataset loaders**: `NMNIST_loader`, `SHD_loader` (requires tonic)
- Integration with Tonic for neuromorphic datasets

### `fn.py` - Training Utilities
Functional training and evaluation tools:
- `integral_accuracy()`: Compute accuracy from spike counts
- `integral_crossentropy()`: Cross-entropy loss for spike trains
- `scan_snn()`: Execute SNN over time using jax.lax.scan
- `update_step()`: Single gradient descent step
- `evaluate()`: Batch evaluation

### `nir.py` - Neuromorphic Intermediate Representation
Import/export to NIR format for interoperability:
- `from_nir()`: Convert NIR graph to Spyx model
- `to_nir()`: Export Spyx model to NIR graph
- Support for standard NIR nodes (LIF, Conv2d, Linear, etc.)
- RNN subgraph handling for recurrent models

### `experimental.py`
Experimental features under development:
- Phasor network implementations
- Novel neuron dynamics
- Training algorithms (e.g., DECOLLE)

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
uv run pytest              # Run all tests
uv run pytest -v           # Verbose output
uv run pytest tests/test_data_grain.py  # Specific test
```

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
- Target: Python 3.10+
- Line length: 88 (Black-compatible)
- Enabled rules: Pyflakes (F), pycodestyle errors (E), flake8-bugbear (B), import sorting (I)
- Excluded: research/, docs/, scripts/, data/

### Dependencies
- **Core**: optax, jax_tqdm, nir, flax>=0.10.7, grain
- **Optional** (`[loaders]`): tonic, numba>=0.59.0
- **Dev**: pytest, ruff, mkdocs, mkdocs-material, mkdocstrings

## Common Tasks for Agents

### Adding a new neuron model
1. Create class in `nn.py` inheriting from `nnx.Module`
2. Implement `__init__`, `__call__`, and `initial_state`
3. Add tests in `tests/`
4. Document in `docs/api.md`

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

### Running experiments
Research notebooks are in `research/` but excluded from linting. Use these for exploratory work without strict code quality requirements.

## Public API

The package exports the following in `src/spyx/__init__.py`:
- `jax`, `jnp` - JAX and JAX NumPy
- `axn` - Surrogate gradients
- `data` - Data loading
- `experimental` - Experimental features
- `fn` - Functional utilities
- `nir` - NIR conversion
- `nn` - Neuron models
- `__version__` - Version string

All exports are defined in `__all__` for explicit API declaration.
