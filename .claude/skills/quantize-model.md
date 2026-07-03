---
name: quantize-model
description: Quantize a trained or in-training Spyx model with spyx.quant (int8 / int4 / BitNet-ternary via qwix). Use when the user asks to "quantize my SNN", "do QAT", "shrink the model for hardware", "run int8 / ternary", or wants to deploy to a low-precision target.
---

# Quantize a Spyx model

`spyx.quant` is a thin, SNN-aware wrapper around Google's [qwix](https://github.com/google/qwix).
It quantizes the `nnx.Linear` / `nnx.Conv` layers around the spiking dynamics
while leaving the neuron state (membrane voltages, time constants) in fp32 —
quantizing the recurrence directly destroys the dynamics.

## 0. Check the extra is installed

qwix has no PyPI release yet, so it lives behind the `[quant]` extra
(installed from GitHub via `tool.uv.sources`):

```bash
uv sync --extra quant
```

In code, gate on availability so the path degrades cleanly:

```python
import spyx.quant as q
if not q.available():
    raise RuntimeError("install the quant extra: uv sync --extra quant")
```

## 1. Pick a rule set

`quantize()` defaults to `linear_only_rules()` (int8 QAT). The choices:

| Rule factory | What it does | When |
|---|---|---|
| `linear_only_rules(weight_qtype="int8", act_qtype="int8")` | int8 weights **and** activations on Linear/Conv | default; best hardware match |
| `weights_only_rules(weight_qtype="int8")` | int8 weights, fp activations | memory-bound, accuracy-sensitive |
| `bitnet_ternary_rules(act_qtype="int8")` | ternary {-1,0,+1} weights, int8 acts | aggressive; BitNet-style |

Both `weight_qtype` and `act_qtype` accept `"int8"`, `"int4"`, etc. Note the
caveat in the source: some backends fall back to int2 representation for
ternary — read `bitnet_ternary_rules`' docstring before trusting the bit count.

## 2. Quantize

`quantize()` needs **example inputs** matching `model.__call__` so qwix can
trace the module tree. For a `spyx.nn.Sequential`, that's a sample batch plus
its initial state:

```python
from flax import nnx
import jax.numpy as jnp
import spyx.quant as q

sample_x = jnp.zeros((batch, in_features))
sample_state = model.initial_state(batch)

# QAT (default): train the returned model as usual.
qmodel = q.quantize(model, sample_x, sample_state)

# Post-training instead: mode="ptq" on an already-trained model.
qmodel = q.quantize(model, sample_x, sample_state, mode="ptq")

# Custom precision:
qmodel = q.quantize(model, sample_x, sample_state,
                    rules=q.bitnet_ternary_rules())
```

`mode="qat"` (default) inserts fake-quant ops so gradients flow — keep training
`qmodel` with the normal `nnx.Optimizer(model, tx, wrt=nnx.Param)` +
`nnx.value_and_grad` loop (or `spyx.optimize.fit`). `mode="ptq"` quantizes a
frozen model with no further training.

## 3. Verify

- Confirm accuracy on a held-out set before vs. after — QAT should recover
  most of the fp32 accuracy; a large drop means the rule set is too aggressive
  (fall back to `weights_only_rules`).
- `tests/test_quant.py` is the reference for what a correct call looks like and
  what's expected to skip when qwix is absent.

## Don'ts

- Don't quantize the SSM transition matrices or neuron time constants — the
  default rules already exclude them; only the surrounding `nnx.Linear` layers
  should be targeted (`tests/test_ssm.py::test_ssm_can_be_quantized...` shows
  this for SSM models).
- Don't forget the example inputs; qwix can't trace without them.
- Don't compare int4/ternary accuracy without QAT first — PTQ at very low
  precision usually needs calibration data to be usable.
