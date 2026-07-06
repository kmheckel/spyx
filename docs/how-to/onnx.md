# How to export to ONNX

Use [`spyx.experimental.onnx`](../reference/onnx.md) to export a spiking model to
[ONNX](https://onnx.ai/) so it can run under a general runtime — onnxruntime on a
server, ONNX Runtime Mobile on a phone, or a browser/embedded target — including
**int8** deployment, which onnxruntime has real integer kernels for.

!!! warning "Experimental — unstable API"
    `spyx.experimental.onnx` is research-stage; its API may change without a
    deprecation cycle. Import it from `spyx.experimental`.

## Install the conversion stack

Conversion is a direct jaxpr → ONNX lowering via
[jax2onnx](https://pypi.org/project/jax2onnx/) — no TensorFlow, no jax2tf, no
tf2onnx. `jax2onnx` and `onnx` are imported **lazily** inside the export
functions, so `import spyx.experimental.onnx` works without them; you install
them yourself:

```bash
pip install jax2onnx onnx onnxruntime
```

Inference only needs `onnxruntime` (or ONNX Runtime Mobile on-device), not the
conversion stack. Only the forward Heaviside spike is exported — the surrogate
gradient is training-only and irrelevant to inference.

## Two export modes

A spyx neuron (or a `spyx.nn.Sequential` of them) implements *one* timestep of
the temporal loop, `(x_t, state) -> (out, new_state)`, and `spyx.nn.run` scans it
over time. There are two ways to hand that to a runtime:

| Mode | `sequence_length` | ONNX signature | Who runs the temporal loop |
|---|---|---|---|
| **Per-timestep** (default) | `None` | `step(x_t, *state) -> (out, *new_state)` | your application, calling the graph once per timestep |
| **Full-sequence** | `T` (an int) | `run(x_seq, *state) -> (out_seq, *final_state)` | the ONNX graph itself, as a native `Loop` |

ONNX speaks *flat tensor* I/O, not pytrees, so the neuron state (membrane
potentials, adaptive thresholds, …) is flattened into one tensor per state leaf,
in a fixed order. Use `step_signature` to learn that layout.

### Full-sequence: the whole loop in one graph

With `sequence_length=T`, jax2onnx's scan plugin lowers the `jax.lax.scan`
driving `spyx.nn.run` straight to a native ONNX `Loop`, so no host-side temporal
loop is needed at all — a real advantage over runtimes that lack a clean scan
primitive. `x_seq` is shaped `(T, batch, *input_shape)` and `out_seq` is
`(T, batch, *out)`.

## Inspect the flat signature first

`step_signature` needs **only JAX** (never the conversion stack), so you can plan
state seeding/threading before installing anything:

```python
import jax.numpy as jnp
from flax import nnx
from spyx import nn
from spyx.experimental import onnx

rngs = nnx.Rngs(0)
model = nn.Sequential(
    nnx.Linear(8, 16, rngs=rngs),
    nn.LIF((16,), rngs=rngs),
    nnx.Linear(16, 4, rngs=rngs),
    nn.LI((4,), rngs=rngs),
)

sig = onnx.step_signature(model, (8,), batch=1)
sig.input_names    # ['x', 'state_0', 'state_1']
sig.output_names   # ['out', 'new_state_0', 'new_state_1']
sig.state_shapes   # [(1, 16), (1, 4)]  -> the LIF membrane and the LI trace
sig.output_shape   # (1, 4)
sig.seed_state()   # [zeros((1,16)), zeros((1,4))] : how to initialize state
```

## Export to ONNX bytes

```python
# Per-timestep step: (x_t, state) -> (out, new_state)
step_bytes = onnx.to_onnx(model, (8,), batch=1)
with open("step.onnx", "wb") as f:
    f.write(step_bytes)

# Or the whole temporal loop over T=100 timesteps as one native ONNX Loop:
seq_bytes = onnx.to_onnx(model, (8,), batch=1, sequence_length=100)
with open("run.onnx", "wb") as f:
    f.write(seq_bytes)
```

## Verify parity with onnxruntime

Round-trip check: run the exported **per-timestep** graph under onnxruntime,
thread the state yourself, and confirm it matches `spyx.nn.run` in JAX.

```python
import numpy as np
import onnxruntime as ort
import jax.numpy as jnp
from spyx import nn

T, B = 20, 1
x_seq = np.random.randn(T, B, 8).astype(np.float32)

# 1) Reference: run the model in JAX over the whole sequence.
ref_out, _ = nn.run(model, jnp.asarray(x_seq))   # (T, B, 4)

# 2) ONNX: drive the exported step once per timestep, threading the state.
sess = ort.InferenceSession(step_bytes)
state = sig.seed_state()                          # [zeros((1,16)), zeros((1,4))]
onnx_out = []
for t in range(T):
    feeds = {"x": x_seq[t], "state_0": state[0], "state_1": state[1]}
    out, *state = sess.run(sig.output_names, feeds)  # out, new_state_0, new_state_1
    onnx_out.append(out)
onnx_out = np.stack(onnx_out)                     # (T, B, 4)

np.testing.assert_allclose(np.asarray(ref_out), onnx_out, atol=1e-4)
```

The full-sequence graph gives the same result in a single call — feed the whole
`x_seq` plus the seed state and read back `out_seq` directly:

```python
seq_sess = ort.InferenceSession(seq_bytes)
seq_sig = onnx.step_signature(model, (8,), batch=1, sequence_length=T)
feeds = {"x": x_seq, "state_0": sig.seed_state()[0], "state_1": sig.seed_state()[1]}
out_seq, *final_state = seq_sess.run(seq_sig.output_names, feeds)  # (T, B, 4)
np.testing.assert_allclose(np.asarray(ref_out), out_seq, atol=1e-4)
```

## int8 deployment

ONNX is the **int8 path** for spyx: unlike NIR (which stores fp32 weights and
cannot carry quantization — see [How to quantize a model](quantize.md)),
onnxruntime has real integer kernels. Quantize the model with `spyx.quant`,
export to ONNX, and run it under onnxruntime. The
[deployment guide](deploy.md) walks the full measure → shrink → export → verify
story, including where the ONNX and NIR paths diverge.

## Requirements on the model

`to_onnx` exports a *stateful* `(x_t, state) -> (out, new_state)` step (or a
`spyx.nn.run` over one), so the model must expose `initial_state` — wrap any
stateless layers in `spyx.nn.Sequential` with at least one stateful neuron. The
default `opset` is 21, recent enough for the native `Loop` used by the
full-sequence export.
