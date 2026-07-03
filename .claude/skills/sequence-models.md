---
name: sequence-models
description: Use Spyx's non-spiking sequence layers — state-space models (spyx.ssm — LRU, S5Diag, Mamba, MambaBlock, ChunkedSSM) and phasor networks (spyx.phasor). Use when the user asks to "add an SSM / Mamba / S5 layer", "build a phasor network", "mix SNN and SSM blocks", or wants a long-range recurrent model in Spyx.
---

# Sequence models: SSM and phasor layers

Spyx ships two families of non-LIF sequence layers. Both are Flax-NNX modules,
but their calling contract differs from the spiking neurons — read this first.

## The whole-sequence contract (SSMs)

Unlike `spyx.nn` neurons (which take `(x_t, state) -> (out, state)` one step at
a time via `spyx.nn.run`), every `spyx.ssm` layer consumes the **entire
time-major sequence at once** and returns the same shape:

```
__call__(u: [T, B, d_model]) -> [T, B, d_model]
```

They run their recurrence through `jax.lax.associative_scan`, so depth is
O(log T) — that's the whole point, don't wrap them in a per-step scan.

| Layer | Constructor | Use for |
|---|---|---|
| `LRU(d_model, d_state=64, *, rngs)` | Linear Recurrent Unit (Orvieto 2023) | simple, stable long-range; the safe default |
| `S5Diag(d_model, d_state=64, *, rngs)` | diagonal S4D/S5, HiPPO-LegS init | when you want principled HiPPO memory |
| `Mamba(d_inner, d_state=16, *, rngs)` | selective-SSM core | input-dependent gating; usually via MambaBlock |
| `MambaBlock(d_model, d_state=16, d_conv=4, expand=2, *, rngs)` | full block (in-proj → conv → SSM → gate → out-proj) | drop-in residual block |
| `ChunkedSSM(inner, outer, *, chunk_size, pool)` | hierarchical H-Net skeleton | multi-resolution sequences |

`|λ| < 1` stability is enforced by construction (radial/angular
parameterisation), not clipping — don't add manual clamps.

### Hybrid SNN → SSM stacks

An SSM composes with spiking layers inside `spyx.nn.Sequential` as long as the
shapes line up. `tests/test_ssm.py` runs a `Linear → LIF → LRU → readout`
stack through `spyx.nn.run`; use it as the reference. The `MambaBlock`
residual connection is left to the caller (usually alongside an `RMSNorm`).

### ChunkedSSM (H-Net skeleton)

`inner` and `outer` can be *any* `(T,B,D) -> (T,B,D)` module — including an
`LRU` or a `MambaBlock`. Input length must be a multiple of `chunk_size`.
This is a structural skeleton, not the full Hwang et al. 2024 recipe (no
dynamic chunking or byte tokenisation) — say so if the user expects full H-Net.

## Phasor networks (spyx.phasor)

Complex-valued networks encoding information in **phase**. Two modes:

**Training (fast, continuous):** use `PhasorMLP` or raw `PhasorLinear` on the
complex domain directly.

```python
from flax import nnx
import spyx.phasor as ph
model = ph.PhasorMLP(in_features, hidden, out_features, depth=2, rngs=nnx.Rngs(0))
logits = model(x_real)   # real_to_phasor encoding is applied internally
```

**The convergence gotcha:** weights are stored as *paired* `kernel_re` /
`kernel_im` float32 params, not a single `complex64` param. This is
deliberate — JAX returns the conjugate Wirtinger derivative for a real loss
w.r.t. a complex param, and stock `optax.adam` doesn't unwind the conjugation,
so a single complex param drifts and never converges. Keep the real/imag
split; don't "simplify" it back to one complex param.

**Spiking inference (deployment):** wrap a trained `PhasorLinear` in
`SpikingPhasor(layer, period_T)` to get a `[T,B,in] -> [T,B,out]` spike-domain
module that drops into `spyx.nn.Sequential`. Convert **after** training, not
before — the complex-domain forward pass is much faster to train.

Codec helpers: `phase_to_spikes(theta, T)` (one spike per cycle) and
`spikes_to_phase(spikes, T)` (centroid decode) are inverses within `2π/T`
quantization error.

## Quantization

Both families quantize via `spyx.quant` — but only the `nnx.Linear` layers
around the recurrence get low precision; the SSM transition matrices stay
fp32. See the `quantize-model` skill.

## Verify

Run `uv run python scripts/ssm_demo.py` and `uv run python scripts/phasor_demo.py`
for end-to-end examples (copy task, HiPPO stats, hybrid stack, spike round-trip).
The how-to guide `docs/how-to/recurrent-layers.md` is the user-facing version.
