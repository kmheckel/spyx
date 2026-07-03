# How to install Spyx

To add Spyx to a project managed with [uv](https://github.com/astral-sh/uv):

```bash
uv add spyx
```

With plain pip:

```bash
pip install spyx
```

## Install optional extras

The core install is deliberately lean. Pull in extras only for the features you need:

| Extra | Installs | Needed for |
|---|---|---|
| `spyx[loaders]` | `tonic`, `numba` | `spyx.data.SHD_loader`, `spyx.data.NMNIST_loader` |
| `spyx[quant]` | `qwix` (from GitHub) | `spyx.quant` (int8 / int4 / BitNet QAT) |
| `spyx[docs]` | `mkdocs`, `mkdocs-material`, `mkdocstrings` | building these docs locally |

```bash
uv add "spyx[loaders]"
```

!!! note "qwix has no PyPI release"
    The `spyx[quant]` extra needs [qwix](https://github.com/google/qwix), which
    isn't published on PyPI. Spyx pins it via `tool.uv.sources`, but uv sources
    aren't transitive — so `spyx[quant]` alone resolves qwix **only inside the
    Spyx repo** (`uv sync --extra quant`), not in your own project. Everywhere
    else, install qwix from GitHub directly. This one-liner works with **both uv
    and pip**:

    ```bash
    uv add  spyx "qwix @ git+https://github.com/google/qwix"
    pip install spyx "qwix @ git+https://github.com/google/qwix"
    ```

## Get GPU / TPU support

The default `jax` dependency is the CPU wheel. To train on an accelerator, install the JAX build matching your hardware **in addition to** Spyx — for example:

```bash
uv add "jax[cuda12]"     # NVIDIA GPU, CUDA 12
uv add "jax[tpu]"        # Cloud TPU
```

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the full matrix (ROCm, older CUDA versions, etc.).

## Install from source (development)

```bash
git clone https://github.com/kmheckel/spyx
cd spyx
uv sync
```

This installs all dependencies plus development tooling (pytest, ruff, mkdocs).

## Verify the installation

Inside the Spyx repo, run the bundled install check:

```bash
uv run python scripts/check_install.py
```

It verifies JAX devices, Spyx imports, an SNN forward pass, a training epoch, a NIR roundtrip, and optional-extra detection in about 30 seconds. In your own project, a quick smoke test:

```python
import jax, spyx
print(spyx.__version__, jax.devices())
```
