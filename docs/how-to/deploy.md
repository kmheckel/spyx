# How to deploy a trained SNN

You have a trained spiking network. This guide takes it the last mile:
**measure** it, **shrink** it, **export** it to a runtime, and **verify** the
round-trip — and is honest about where the two export paths (ONNX and NIR)
diverge. It ties together the tool-specific how-tos:
[benchmarking](benchmarking.md), [quantization](quantize.md), and
[ONNX export](onnx.md).

The one decision that drives everything below:

| Target | Export path | Numeric format | Why |
|---|---|---|---|
| CPU / GPU / mobile / browser via a general runtime | **ONNX** (`spyx.experimental.onnx`) | fp32 **or int8** | onnxruntime has real int8 kernels |
| Neuromorphic hardware (Loihi, SpiNNaker, …) via the NIR ecosystem | **NIR** (`spyx.nir`) | **fp32 only** | NIR stores fp32 weights + continuous time constants; it has no field to encode a quantized tensor |

The seam to remember: **NIR cannot carry quantization.** Quantize for the ONNX
path; export to NIR *before* quantizing.

## 0. The model

We'll use a small feed-forward SNN throughout.

```python
import jax, jax.numpy as jnp
from flax import nnx
import spyx
import spyx.nn as snn

rngs = nnx.Rngs(0)
model = snn.Sequential(
    nnx.Linear(32, 64, use_bias=False, rngs=rngs),
    snn.LIF((64,), beta=0.9, rngs=rngs),
    nnx.Linear(64, 10, use_bias=False, rngs=rngs),
    snn.LI((10,), rngs=rngs),
)
# ... assume this has been trained (see How to train a model).
```

## 1. Measure — know your baseline

Before changing anything, get numbers with [`spyx.bench`](benchmarking.md) so you
can tell whether a later optimization actually helped. It reports latency,
throughput, an XLA-cost-model FLOP/MFU estimate, and the SNN-specific
**spike-rate energy proxy** (the mean fraction of non-zero activations — lower
means sparser spiking, which on neuromorphic hardware means less energy).

```python
result = spyx.bench.benchmark(model, (32,), seq_len=100, batch=32)
print(spyx.bench.format_table([result]))
print("spike rate:", result.spike_rate)   # event-driven energy proxy
```

!!! warning "MFU is an fp32 ceiling"
    `spyx.bench` computes MFU against the device's **dense fp32** peak. Once you
    quantize (next step), the int8 model's MFU is measured against that fp32
    ceiling, so it is **not comparable** to the fp32 MFU — a quantized model can
    be faster while showing a *lower* MFU. Compare quantized models on latency
    and throughput, not MFU. (Details in [benchmarking](benchmarking.md).)

## 2. Shrink — quantize for the ONNX/int8 path

[`spyx.quant`](quantize.md) wraps Google's [qwix](https://github.com/google/qwix)
to quantize the **dense `Linear`/`Conv` layers** while leaving the spiking
dynamics (`LIF`/`CuBaLIF`/`LI`) in fp32 — their state recurrences are sensitive
to integer rounding. On an SNN the feedforward activation is a spike train
(values in `{0, 1}`), so it already lies exactly on the integer grid: a
**weight-only** scheme is lossless on the activation side.

```python
if not spyx.quant.available():          # import is always safe; call needs qwix
    raise SystemExit('quant needs qwix: pip install "qwix @ git+https://github.com/google/qwix"')

B = 32
sample_x = jnp.zeros((B, 32))
sample_state = model.initial_state(B)

# Post-training int8 (use mode="qat" + a few fine-tune epochs for better accuracy):
qmodel = spyx.quant.quantize(model, sample_x, sample_state, mode="ptq")
```

Always compare accuracy before and after — quantization is a lossy transform.
See [How to quantize a model](quantize.md) for QAT, custom precisions, and the
`module_path` regex footgun.

## 3. Export

### Path A — ONNX (int8 or fp32, general runtimes)

This is where a quantized model goes: [`spyx.experimental.onnx`](onnx.md) lowers
the model (the per-timestep step, or the whole `spyx.nn.run` loop as a native
ONNX `Loop`) to an `onnx.ModelProto`, and onnxruntime runs it with real int8
kernels.

```python
from spyx.experimental import onnx

# Per-timestep step; the app threads state. (Pass sequence_length=T for the
# whole temporal loop in one graph.)
onnx_bytes = onnx.to_onnx(qmodel, (32,), batch=1)
with open("model.onnx", "wb") as f:
    f.write(onnx_bytes)
```

`jax2onnx`/`onnx` are optional, lazily-imported deps
(`pip install jax2onnx onnx onnxruntime`). Full walkthrough and the flat state
signature: [How to export to ONNX](onnx.md).

### Path B — NIR (fp32 only, neuromorphic hardware)

For neuromorphic targets, export to [NIR](../reference/nir.md) — but export the
**fp32** model, *not* the quantized one:

```python
from spyx import nir as spyx_nir

# NIR export covers the spiking feed-forward stack (Linear + LIF/CuBaLIF/IF,
# conv, pooling, recurrent subgraphs). Export the un-quantized model:
spiking = snn.Sequential(
    nnx.Linear(32, 64, use_bias=False, rngs=rngs),
    snn.LIF((64,), beta=0.9, rngs=rngs),
)
graph = spyx_nir.to_nir(spiking, {"input": (32,)}, {"output": (64,)})
```

If you hand `to_nir` a **quantized** model it does **not** silently produce a
half-converted graph — it raises a clear `ValueError`:

```python
spyx_nir.to_nir(qmodel, {"input": (32,)}, {"output": (10,)})
# ValueError: Layer 0 (Linear) is qwix-quantized, but NIR export cannot carry
# quantization ... Export the model to NIR *before* quantizing it ... For an
# int8 deployment target, export to ONNX instead ...
```

That is deliberate: NIR stores fp32 weights and continuous time constants, with
no int8/int4 or scale/zero-point fields, so a quantized tensor has nowhere to go.
Export to NIR before quantizing, or dequantize first; put int8 on the ONNX path.

## 4. Verify — round-trip the export

Never trust an export you haven't checked against the source model.

**NIR** re-imports and runs in one call (`from_nir`); compare to `spyx.nn.run`:

```python
x = jax.random.normal(jax.random.PRNGKey(1), (100, 8, 32))   # (T, B, in)
imported, nir_out = spyx_nir.from_nir(graph, x, dt=1, rngs=nnx.Rngs(2))
ref_out, _ = snn.run(spiking, x)
assert jnp.allclose(ref_out, nir_out, atol=1e-4)
```

**ONNX** parity: drive the exported graph under onnxruntime, thread the state,
and compare to `spyx.nn.run` — see the runnable example in
[How to export to ONNX](onnx.md#verify-parity-with-onnxruntime).

## The whole story in one line

**Measure** with `spyx.bench` → **shrink** with `spyx.quant` (for the ONNX/int8
target only) → **export** to ONNX (int8, general runtimes) or NIR (fp32,
neuromorphic) → **verify** the round-trip. The single seam to keep straight: NIR
is fp32-only and refuses a quantized model, so quantization belongs on the ONNX
side.
