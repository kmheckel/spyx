---
name: setup-gpu
description: Diagnose and fix the common JAX-on-GPU install problem where a CUDA-capable card is present but jax falls back to CPU. Use when the user runs scripts/check_install.py or scripts/smoke_notebook_apis.py and sees "platforms: ['cpu']" or "CUDA-enabled jaxlib is not installed", or when they report training on CPU despite having an NVIDIA GPU.
---

# Get JAX talking to the local GPU

The most common "my SNN is running slow" issue on Spyx is that the CPU-only `jax` wheel is installed, not the CUDA-enabled one. This skill walks through the fix.

## Step 1 — confirm the diagnosis

Run:

```bash
uv run python -c "import jax; print(jax.__version__, jax.devices())"
nvidia-smi
```

- If `jax.devices()` returns `[CudaDevice(id=0)]` or similar: JAX is already using the GPU. Stop here — the slow training is something else. Invoke `debug-training` instead.
- If it returns `[CpuDevice(id=0)]` or `[TFRT_CPU_0]` but `nvidia-smi` shows the card: the GPU is visible to the OS but JAX hasn't been built against CUDA. Continue.

## Step 2 — install the CUDA-enabled wheel

JAX ships its own CUDA + cuDNN libraries; the user does NOT need a system CUDA toolkit. Install the `cuda12` extras:

```bash
uv pip install -U "jax[cuda12]"
```

This replaces the CPU `jaxlib` with `jaxlib==<same version>+cuda12` and pulls in the bundled CUDA libs. Works on any driver supporting CUDA 12 or 13 (the wheel is forward-compatible).

If the resolver complains about conflicting versions, pin explicitly:

```bash
uv run python -c "import jax; print(jax.__version__)"   # note the version
uv pip install -U "jax[cuda12]==<that version>"
```

## Step 3 — verify

```bash
uv run python -c "import jax; print(jax.devices())"
# expect: [CudaDevice(id=0)] ...
uv run python scripts/check_install.py
# expect: environment line reports platforms: ['cuda']
```

If `check_install.py` prints `platforms: ['cuda']` and all seven checks pass, you're done.

## Common failure modes

- **`CUDA_ERROR_NO_DEVICE` despite `nvidia-smi` working** → another process is holding the GPU; run `nvidia-smi` to check memory usage and free anything pinned.
- **Resolver says `jax[cuda12]` isn't compatible with `pyproject.toml`** → the project's `jax` pin is too strict. Edit `pyproject.toml` to replace `"jax"` with `"jax[cuda12]"` in the `dependencies` list, then `uv sync --extra loaders --extra quant`.
- **Card is AMD or Apple Silicon** → use `jax[rocm]` or the Mac-Metal wheel instead. `jax[cuda12]` is NVIDIA-only.

## After fixing

Point the user at `snn-primer` if they're new, or `run-tutorial` to try a GPU-backed training run. The SHD surrogate-gradient tutorial should go from tens of minutes on CPU to tens of seconds on an RTX 3060 or better.
