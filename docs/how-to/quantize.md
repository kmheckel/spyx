# How to quantize a model

To run quantization-aware training (QAT) on a Spyx SNN, use [`spyx.quant`](../reference/quant.md) — a thin SNN-aware wrapper around Google's [qwix](https://github.com/google/qwix) library.

## Prerequisite: install qwix

`spyx.quant` is built on [qwix](https://github.com/google/qwix), which has no
PyPI release. Because uv sources aren't transitive, the `spyx[quant]` extra only
auto-resolves qwix inside the Spyx repo; in your own project install qwix from
GitHub directly. This works with **both uv and pip**:

```bash
uv add  spyx "qwix @ git+https://github.com/google/qwix"
pip install spyx "qwix @ git+https://github.com/google/qwix"
```

Gate any quantization code on availability — `import spyx.quant` is always safe, and the helpers raise `ImportError` with these install instructions if you call them without qwix:

```python
import spyx

if not spyx.quant.available():
    raise SystemExit(
        "quantization needs qwix: "
        'pip install "qwix @ git+https://github.com/google/qwix"'
    )
```

## Quantize with the int8 defaults

To quantize a model for QAT, call `spyx.quant.quantize` with example inputs matching your model's `__call__` signature (qwix traces the module graph to find the layers):

```python
import jax.numpy as jnp
from flax import nnx
import spyx
import spyx.nn as snn

rngs = nnx.Rngs(0)
model = snn.Sequential(
    nnx.Linear(128, 64, use_bias=False, rngs=rngs),
    snn.LIF((64,), rngs=rngs),
    nnx.Linear(64, 20, use_bias=False, rngs=rngs),
    snn.LI((20,), rngs=rngs),
)

B = 32
sample_x = jnp.zeros((B, 128))               # one timestep of input
sample_state = model.initial_state(B)
qmodel = spyx.quant.quantize(model, sample_x, sample_state)
```

By default this applies **int8 weights + activations to `nnx.Linear` / `nnx.Conv` layers only**. The spiking dynamics (`LIF`, `CuBaLIF`, `ALIF`, `IF`) and the `LI` readout stay in fp32 — their state recurrences (`V = beta * V + x - reset`) involve cancellations that integer rounding tends to collapse into silence.

The returned `qmodel` is an ordinary NNX module: train it with `spyx.optimize.fit` or a hand-rolled `nnx.Optimizer` loop exactly as in [How to train a model](train.md).

## Choose a different precision with rules

To override the defaults, pass a list of qwix `QuantizationRule`s via `rules=`. Spyx ships three shorthand factories:

```python
# int4 weights + int8 activations on Linear / Conv:
rules = spyx.quant.linear_only_rules(weight_qtype="int4", act_qtype="int8")

# weights-only int8 (activations stay fp32) — for memory-bound deployment:
rules = spyx.quant.weights_only_rules("int8")

# BitNet b1.58-style ternary weights + int8 activations:
rules = spyx.quant.bitnet_ternary_rules()

qmodel = spyx.quant.quantize(model, sample_x, sample_state, rules=rules)
```

!!! note "About the BitNet 'ternary' rules"
    Qwix doesn't expose a true ternary qtype today, so `bitnet_ternary_rules` falls back to `"int2"` (values in `{-2, -1, 0, 1}`). That gives the same memory profile and storage class as ternary; for strict {-1, 0, +1} semantics you'd need a custom `qwix.QuantizationRule` with a hand-rolled calibration. Pass `act_qtype=None` for pure weights-only ternary.

For anything the shorthands don't cover, build `qwix.QuantizationRule` instances directly. Two fields select what gets quantized, and it is easy to get the first one wrong:

- **`op_names`** — the underlying JAX op to quantize. This is the one you almost always want: `"dot_general"` (dense `nnx.Linear`), `"conv_general_dilated"` (`nnx.Conv`), and `"einsum"` (recurrent / SSM state transitions — usually left in fp32). Spiking neuron updates are elementwise and use none of these, so they stay fp32 automatically.
- **`module_path`** — a regex matched with `re.fullmatch` against the `/`-joined **NNX attribute path** (e.g. `core/layers/0`), *not* the class name. Use `".*"` to match everywhere, or a real path like `r".*layers/4"` to target one layer.

!!! warning "`module_path=r\".*Linear.*\"` silently quantizes nothing"
    A tempting-looking `module_path=r".*Linear.*"` matches **nothing** and turns `quantize()` into a silent no-op: qwix `re.fullmatch`es the *attribute path* (`core/layers/0`), which never contains the class name `Linear`. Select dense/conv work by **op** instead — this is exactly what `linear_only_rules` does and what the [QAT notebook](../examples/quantization/qat_intro.ipynb) uses:

    ```python
    import qwix
    rule = qwix.QuantizationRule(
        module_path=".*",                               # match everywhere ...
        op_names=("dot_general", "conv_general_dilated"),  # ... quantize dense/conv only
        weight_qtype="int8",
        act_qtype="int8",
    )
    qmodel = spyx.quant.quantize(model, sample_x, sample_state, rules=[rule])
    ```

    To target a *specific* layer by position, use a genuine attribute-path regex such as `module_path=r".*layers/4"`. See the `spyx.quant.linear_only_rules` docstring for the full explanation.

## Post-training quantization

To quantize an already-trained model without further training, pass `mode="ptq"`:

```python
qmodel = spyx.quant.quantize(model, sample_x, sample_state, mode="ptq")
```

## The recommended workflow

1. Train the fp32 model to convergence.
2. Wrap it with `quantize(...)` (QAT mode) and fine-tune for a few epochs.
3. Compare `integral_accuracy` between the fp32 and quantized models before deployment.

For a full worked example, see the [Quantization-Aware Training notebook](../examples/quantization/qat_intro.ipynb), and `scripts/ssm_demo.py` for quantizing the linear layers around an SSM.

## Exporting a quantized model

Quantization and export interact, and the two export paths behave differently:

- **ONNX** (`spyx.experimental.onnx`) is the int8 path. Export the quantized model, then run it under onnxruntime, which has real int8 kernels. See [How to export to ONNX](onnx.md).
- **NIR** (`spyx.nir`) **cannot carry quantization** — a NIR graph stores fp32 weights and continuous time constants, with no int8/int4 or scale/zero-point fields. `spyx.nir.to_nir` therefore **raises a `ValueError`** on a qwix-quantized model rather than silently dropping the quantization. Export to NIR *before* quantizing (or dequantize first).

The [deployment guide](deploy.md) walks the whole measure → shrink → export → verify story end to end.
