⚡🧠💻 Welcome to Spyx! 💻🧠⚡
============================
[![arXiv](https://img.shields.io/badge/arXiv-2402.18994-b31b1b.svg)](https://arxiv.org/abs/2402.18994) [![DOI](https://zenodo.org/badge/656877506.svg)](https://zenodo.org/badge/latestdoi/656877506) [![PyPI version](https://badge.fury.io/py/spyx.svg)](https://badge.fury.io/py/spyx) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmheckel/spyx/blob/main/docs/examples/surrogate_gradient/SurrogateGradientTutorial.ipynb)

[![](https://dcbadge.vercel.app/api/server/TCYQFWsBwj)](https://discord.gg/TCYQFWsBwj)


![README Art](spyx.png "Spyx")

Why use Spyx?
=============

Spyx (pronounced "spikes") is a compact spiking neural network library built on JAX and Flax NNX. It offers the flexibility and extensibility of a PyTorch-style framework while reaching the throughput of SNN libraries that hand-write custom CUDA kernels — because the whole network, including its temporal dynamics, is JIT-compiled by XLA.

What ships today:

- **Two ways to train** — surrogate-gradient descent (backprop through time) and gradient-free neuroevolution (`spyx[evo]`, via evosax).
- **A neuron zoo** — LIF, LI, ALIF, CuBaLIF, IF and their recurrent variants, all as plain `flax.nnx.Module`s. Defining your own and dropping it into `spyx.nn.Sequential` is a few lines.
- **Sequence & complex layers** — diagonal state-space models (`spyx.ssm`: LRU, S5Diag, Mamba, ChunkedSSM) and complex-valued phasor / spiking-phasor networks (`spyx.phasor`), all parallelized with the same associative-scan machinery as the spiking neurons.
- **Efficiency tooling** — int8 / int4 / BitNet-ternary quantization (`spyx.quant`, QAT + PTQ) and a benchmark harness that reports latency, throughput, MFU, and spike-rate as an energy proxy (`spyx.bench`).
- **Interoperability** — import/export to the [Neuromorphic Intermediate Representation](https://neuroir.org) (`spyx.nir`) for neuromorphic hardware, plus ONNX export (`spyx.experimental.onnx`).
- **A research vehicle** — `spyx.experimental` stages unstable building blocks (parallel spiking neurons, resonate-and-fire, routing-slot memory, a hybrid surrogate+evolution trainer, runnable recipe zoo) before they graduate into the stable core.

New to spiking networks? Start with the [Quickstart](#quickstart) below (zero downloads), then [Your first SNN](docs/tutorials/first-snn.md) trains a real model, and the [glossary](docs/explanation/glossary.md) defines the vocabulary.

Installation:
=============

Spyx is on PyPI and installs with either [uv](https://github.com/astral-sh/uv) or pip:

```bash
uv add spyx          # into a uv-managed project
pip install spyx     # or with plain pip
```

The default install is CPU-only and lean; a laptop CPU is plenty for the quickstart and the tutorials. For the event-dataset loaders, add the extra:

```bash
uv add "spyx[loaders]"      # or: pip install "spyx[loaders]"
```

See [How to install Spyx](docs/how-to/install.md) for the full extras table (`loaders`, `quant`, `evo`, `docs`) and for GPU/TPU wheels.

Note: as with other JAX libraries, install the accelerator build of JAX to train on a GPU/TPU. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

Quickstart
==========

This trains a tiny SNN on synthetic spike trains — **no dataset download** — and prints a falling loss and rising accuracy. Copy-paste it into a file and run `python quickstart.py`:

```python
import jax, jax.numpy as jnp, optax
from flax import nnx
import spyx, spyx.nn as snn, spyx.optimize as opt

rngs = nnx.Rngs(0)
model = snn.Sequential(
    nnx.Linear(8, 32, use_bias=False, rngs=rngs),
    snn.LIF((32,), activation=spyx.axn.triangular(), rngs=rngs),
    nnx.Linear(32, 3, use_bias=False, rngs=rngs),
    snn.LI((3,), rngs=rngs),  # non-spiking leaky readout -> class logits
)

T, B, C, n_cls = 16, 32, 8, 3  # time, batch, channels, classes

def make_batch(k):  # class c => channel c fires often (learnable structure)
    ky, ks = jax.random.split(k)
    y = jax.random.randint(ky, (B,), 0, n_cls)
    prob = jnp.full((B, C), 0.05).at[jnp.arange(B), y].set(0.5)
    x = (jax.random.uniform(ks, (T, B, C)) < prob).astype(jnp.float32)
    return x, y  # x is time-major (T, B, C)

Loss = spyx.fn.integral_crossentropy(time_axis=0)
Acc = spyx.fn.integral_accuracy(time_axis=0)

def loss_fn(m, x, y):
    return Loss(snn.run(m, x)[0], y)

def eval_fn(m, x, y):
    traces = snn.run(m, x)[0]
    return Acc(traces, y)[0], Loss(traces, y)

key = jax.random.PRNGKey(0)
train_iter = lambda: (make_batch(jax.random.fold_in(key, i)) for i in range(8))
eval_iter = lambda: iter([make_batch(jax.random.PRNGKey(999))])

opt.fit(
    model, optax.adam(2e-3), loss_fn, train_iter,
    epochs=15, eval_iter=eval_iter, eval_fn=eval_fn,
    on_epoch_end=lambda e, m: print(
        f"epoch {e:2d}  train_loss={m['train_loss']:.3f}  eval_acc={m['eval_acc']:.2%}"),
)
```

You should see the loss fall and accuracy climb well above the 33% chance level:

```text
epoch  0  train_loss=3.536  eval_acc=28.12%
epoch  3  train_loss=1.192  eval_acc=68.75%
epoch  7  train_loss=0.862  eval_acc=81.25%
epoch 14  train_loss=0.790  eval_acc=90.62%
```

That's a complete surrogate-gradient training loop, JIT-compiled end to end. For real data, continue to [Your first SNN](docs/tutorials/first-snn.md); to choose between surrogate gradients, evolution, quantization, and conversion, see [Choosing an approach](docs/explanation/choosing-an-approach.md).

Hardware Requirements:
======================

Spyx runs anywhere JAX does — a laptop CPU is enough for the quickstart, the tutorials, and small-to-medium models. For larger workloads it leans on an accelerator: Spyx reaches its headline throughput by keeping the entire dataset resident in GPU vRAM, so big SNNs and neuroevolution sweeps benefit from a higher-memory card. Networks of a few hundred thousand parameters train comfortably on a laptop GPU with 6 GB of vRAM.

Cloud TPUs: Spyx tracks the current JAX release, so it does not run on Google Colab's older-JAX TPU runtime. Use a **GPU** (or CPU) Colab runtime — the [Colab tutorial](https://colab.research.google.com/github/kmheckel/spyx/blob/main/docs/examples/surrogate_gradient/SurrogateGradientTutorial.ipynb) runs there directly.

Development:
============

To set up a development environment:

```bash
git clone https://github.com/kmheckel/spyx
cd spyx
uv sync
```

This will install all dependencies including development tools (pytest, ruff, mkdocs).

### End-to-end install check

Once installed, run the bundled install-check script to confirm JAX, Spyx, and optional extras are all wired up correctly:

```bash
uv run python scripts/check_install.py
```

Seven checks in ~30 seconds — JAX version + visible devices, Spyx imports, SNN forward pass, one training epoch, NIR roundtrip, notebook-API smoke tests, and optional-extra detection (`tonic`, `qwix`). Useful right after `uv sync` on a new machine, especially if you expect GPU / TPU devices to show up.

### Code Quality

Spyx uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting. Before committing changes, run:

```bash
# Check for linting errors
uv run ruff check

# Auto-fix linting errors
uv run ruff check --fix

# Format code
uv run ruff format
```

Ruff is configured in `pyproject.toml` to enforce code quality standards including:
- Import sorting (isort-compatible)
- PEP 8 style guidelines
- Common bug patterns (flake8-bugbear)
- Exclusion of research, docs, and scripts directories

### Testing

Run the test suite using pytest:

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_data_grain.py

# Run tests with coverage
uv run pytest --cov=spyx --cov-report=html
```

Tests are located in the `tests/` directory and cover core functionality including data loading, neuron models, and training utilities.

### Releasing new versions

A utility script is provided to automate the release process:

```bash
# Dry run to see what would happen
python scripts/release.py --dry-run --github --pypi

# Perform a full release
python scripts/release.py --github --pypi
```

This script will:
1. Build the package using `uv build`.
2. Create a git tag and GitHub release using `gh`.
3. Publish to PyPI using `uv publish`.

### Building Documentation

The documentation is built using MkDocs:

```bash
# Preview documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```


Research and Projects Using Spyx:
=================================

Experiments/Benchmarks used in the Spyx Paper: [Benchmark Notebooks](https://github.com/kmheckel/spyx/tree/main/research/paper)

Master's Thesis: Neuroevolution of Spiking Neural Networks [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10620442.svg)](https://doi.org/10.5281/zenodo.10620442)

*** Your projects and research could be here! ***

Note: notebooks under `research/` predate the Flax NNX migration and still use the legacy Haiku API. They are kept for reproducibility of the Spyx paper experiments and will be ported in a follow-up effort. New work should use the tutorials under `docs/examples/`.


Contributing:
=============

If you'd like to contribute, head on over to the issues page to find proposed enhancements and leave a comment! Also head over to the Open Neuromorphic Discord server to ask questions!

Citation:
=========

If you find Spyx useful in your work please cite it using the following Bibtex entries:

```
@misc{heckel2024spyx,
    title={Spyx: A Library for Just-In-Time Compiled Optimization of Spiking Neural Networks},
    author={Kade M. Heckel and Thomas Nowotny},
    year={2024},
    eprint={2402.18994},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

```
@software{kade_heckel_2024_10635178,
  author       = {Kade Heckel and
                  Steven Abreu and
                  Gregor Lenz and
                  Thomas Nowotny},
  title        = {kmheckel/spyx: v0.1.17},
  month        = feb,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {camera-ready},
  doi          = {10.5281/zenodo.10635178},
  url          = {https://doi.org/10.5281/zenodo.10635178}
}
```
