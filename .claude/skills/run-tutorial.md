---
name: run-tutorial
description: Open and execute one of the bundled Spyx tutorial notebooks end-to-end. Use when the user asks to run / see / try a specific example (surrogate-gradient, cartpole, QAT, NIR, etc.) or wants a guided tour of what Spyx can do on their machine.
---

# Run a Spyx tutorial notebook

## Step 1 — pick the right notebook

| User intent | Notebook |
|---|---|
| "Train an SNN end-to-end on SHD" | `docs/examples/surrogate_gradient/SurrogateGradientTutorial.ipynb` |
| "Compare neuron types (LIF vs ALIF vs RLIF)" | `docs/examples/surrogate_gradient/shd_sg_neuron_model_comparison.ipynb` |
| "Compare surrogate gradients" | `docs/examples/surrogate_gradient/shd_sg_surrogate_comparison.ipynb` |
| "Template I can copy for my own SHD experiment" | `docs/examples/surrogate_gradient/shd_sg_template.ipynb` |
| "Neuroevolution of a controller" | `docs/examples/neuroevolution/cartpole_evo.ipynb` |
| "Quantization-aware training" | `docs/examples/quantization/qat_intro.ipynb` |
| "Convert a model to NIR" | `docs/examples/nir/conversion.ipynb` |

Ask the user if unclear; default to `SurrogateGradientTutorial.ipynb`.

## Step 2 — ensure the right extras are installed

| Notebook | Required extra |
|---|---|
| Any SHD/NMNIST surrogate-gradient notebook | `[loaders]` (pulls in `tonic`) |
| `cartpole_evo.ipynb` | `gymnax`, `evosax` (not part of any extra; `uv pip install gymnax evosax`) |
| `qat_intro.ipynb`, QAT bonus sections | `[quant]` (pulls in `qwix` from GitHub) |

```bash
uv sync --extra loaders --extra quant
# cartpole also needs:
uv pip install gymnax evosax
```

## Step 3 — confirm GPU if training is involved

Before executing, run one line:

```bash
uv run python -c "import jax; print(jax.devices())"
```

- `CudaDevice` → good, the 30-epoch SHD notebook will finish in a minute or two on an RTX 3060.
- `CpuDevice` → warn the user that training-heavy notebooks may take 10–30 minutes per epoch and offer to invoke `setup-gpu` first.

## Step 4 — execute

For interactive exploration:

```bash
uv run jupyter lab <notebook-path>
```

For a non-interactive end-to-end run (useful before committing outputs):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace <notebook-path>
```

The SHD notebooks are set to 30 epochs by default — bump to 80–100 if the user wants to match the published accuracy numbers.

## Step 5 — interpret output

The surrogate-gradient tutorials print per-epoch `train_loss / val_acc / val_loss`. Reasonable ranges after 30 epochs on SHD:

- `val_acc` ≥ 0.80 → training worked.
- `val_acc` ≈ 0.05 (random-chance for 20 classes) → invoke `debug-training`.

## First-run gotchas

- **SHD dataset download** is ~300MB; tonic caches under `./data` by default. First run of any SHD notebook takes a couple of minutes extra.
- **Cartpole notebook** imports `evosax.algorithms.distribution_based.cma_es.CMA_ES`, which only exists in evosax 0.2+. If the user has an older evosax, suggest `uv pip install -U evosax`.
- **Mixed precision** was dropped in the NNX migration. If the user wants the throughput, suggest `jax.config.update("jax_default_matmul_precision", "bfloat16")` at the top of the notebook.
