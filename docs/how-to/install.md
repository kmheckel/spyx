# How to install Spyx

Spyx supports **Python 3.11 and 3.12** (`>=3.11, <3.13`). The default install is
CPU-only and deliberately lean; accelerators and heavier features live behind
extras (see below).

## Quickest start: clone and sync

The fastest way to get a working environment — with the tutorials, tests, and
docs tooling — is to clone the repo and let [uv](https://github.com/astral-sh/uv)
build the environment for you:

```bash
git clone https://github.com/kmheckel/spyx
cd spyx
uv sync                     # core + dev tooling (pytest, ruff, mkdocs)
uv run python scripts/check_install.py
```

`uv sync` creates a `.venv/` and installs the locked dependency set. Prefix
commands with `uv run` to execute them inside that environment (`uv run pytest`,
`uv run python my_script.py`, `uv run mkdocs serve`) — no manual `activate` step.

To pull an optional extra into the synced environment, add `--extra`:

```bash
uv sync --extra quant       # adds qwix for quantization-aware training
uv sync --extra loaders     # adds tonic + numba for event-dataset loaders
uv sync --all-extras         # everything
```

## Add Spyx to your own project

If you just want Spyx as a dependency of another project:

```bash
uv add spyx                          # into a uv-managed project
uv add "spyx[loaders]"               # with an extra
```

Plain pip works too:

```bash
pip install spyx
pip install "spyx[loaders]"
```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `spyx[loaders]` | `tonic`, `numba` | `spyx.data.SHD_loader`, `spyx.data.NMNIST_loader` |
| `spyx[quant]` | `qwix` (from GitHub) | `spyx.quant` (int8 / int4 / ternary QAT) |
| `spyx[docs]` | `mkdocs`, `mkdocs-material`, `mkdocstrings`, `matplotlib` | building these docs locally |

!!! note "qwix has no PyPI release"
    The `spyx[quant]` extra needs [qwix](https://github.com/google/qwix), which
    isn't published on PyPI. Spyx pins it via `tool.uv.sources`, but uv sources
    aren't transitive — so `spyx[quant]` resolves qwix **only inside the Spyx
    repo** (`uv sync --extra quant`), not in your own project. Everywhere else,
    install qwix from GitHub directly. This works with **both uv and pip**:

    ```bash
    uv add  spyx "qwix @ git+https://github.com/google/qwix"
    pip install spyx "qwix @ git+https://github.com/google/qwix"
    ```

## GPU / TPU support

The default `jax` dependency is the **CPU wheel**. To train on an accelerator,
install the JAX build matching your hardware **alongside** Spyx:

```bash
uv add "jax[cuda12]"     # NVIDIA GPU, CUDA 12
uv add "jax[tpu]"        # Cloud TPU
```

For **AMD GPUs (ROCm)** — the reference machine for Spyx's parallel-neuron work
is a Radeon 8060S (gfx1151) — JAX ships ROCm wheels/containers; follow the ROCm
section of JAX's install guide rather than hard-pinning a version here, since the
right wheel depends on your ROCm runtime.

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for the full hardware matrix (ROCm, older CUDA, Metal, etc.). Install the JAX
build **after** `uv sync` / `uv add spyx` so it wins the resolution over the
default CPU wheel.

## Verify the installation

Inside the Spyx repo, run the bundled install check:

```bash
uv run python scripts/check_install.py
```

It verifies JAX devices, Spyx imports, an SNN forward pass, a training epoch, a
NIR roundtrip, and optional-extra detection in about 30 seconds.

## Hello, SNN

A minimal smoke test — build a tiny spiking MLP and run it over a time axis:

```python
import jax
import jax.numpy as jnp
from flax import nnx
import spyx

# A 2-layer spiking MLP: Linear -> LIF -> Linear -> LI (leaky readout).
net = spyx.nn.Sequential(
    nnx.Linear(32, 64, rngs=nnx.Rngs(0)),
    spyx.nn.LIF((64,), rngs=nnx.Rngs(0)),
    nnx.Linear(64, 10, rngs=nnx.Rngs(0)),
    spyx.nn.LI((10,), rngs=nnx.Rngs(0)),
)

# Time-major input: (T=20 steps, B=4 batch, C=32 features).
x = jnp.where(jax.random.uniform(jax.random.PRNGKey(1), (20, 4, 32)) > 0.9, 1.0, 0.0)

# run() scans the model over the time axis -> (outputs, final_state).
outputs, _ = spyx.nn.run(net, x)
print(outputs.shape)   # (20, 4, 10) — leaky-readout trace per timestep
```

From here, head to [Your first SNN](../tutorials/first-snn.md) to train a real
model, or [Train a model](train.md) for the training-loop recipe.
