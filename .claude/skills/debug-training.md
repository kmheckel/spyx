---
name: debug-training
description: Diagnose the common training failure modes of spiking networks in Spyx — flat loss, NaN loss, all-silent neurons, exploding gradients, or validation stuck at chance. Use when the user reports "my SNN isn't learning" or training output shows any of the above symptoms.
---

# Debug a Spyx training run

Work through the following in order. Each block has a symptom, the first thing to check, and the most likely fix.

## Symptom 1 — Loss is flat or decreases only in the first few steps

**Check**: print `jnp.mean(m.layers[i].beta[...])` for each LIF layer. If they're all stuck at 0 or 1, the decay parameterisation is pinned.

**Likely fix**: the `beta=` keyword argument on `LIF` was set too aggressively. Default (draw from `truncated_normal(stddev=0.5) + 0.25`, clipped to `[0, 1]`) gives a useful spread. Passing a scalar like `beta=0.99` fixes it pre-training and often prevents learning.

**Second check**: shape mismatch. If `spyx.fn.integral_crossentropy` was accepting `[B, T, C]` for traces but getting `[T, B, C]` (or vice versa), the shape-check (#25) now raises; if it doesn't fire, verify `time_axis=` matches your data layout.

## Symptom 2 — NaN loss after a few steps

**Check**: which parameter went NaN first. Run one step with `jax.tree_util.tree_map(lambda p: jnp.any(jnp.isnan(p)).item(), model)` and find the first True.

**Common culprits**:

1. **Learning rate too high** for the surrogate. `spyx.axn.superspike(k=25)` with `optax.adam(1e-2)` can blow up because SuperSpike's backward peaks at `|x|=0`. Drop to 3e-4 or use `triangular()` which is less peaky.
2. **`optax.clip_by_global_norm(1.0)` missing** on a recurrent network. Recurrent LIFs (`RLIF`, `RCuBaLIF`) can diverge; wrap the optimizer: `optax.chain(optax.clip_by_global_norm(1.0), optax.lion(3e-4))`.
3. **`jnp.log(0)` somewhere** — spike counts dropped to zero and silence_reg got fed an empty spike tree. Check `spyx.fn.silence_reg`'s `min_spikes` parameter is sensible (default 4.0 assumes a certain T).

## Symptom 3 — Every neuron is silent; val_acc at chance

**Check**: `float(spike_train.mean())` on a hidden-layer output. Should be roughly the target firing rate (commonly 5–15% over the T dimension). If 0, no neurons are crossing threshold.

**Likely fixes**:

1. **Input scale too small**. If inputs are binary spikes passed through a `Linear` with default init, the pre-activation may never reach the threshold (which is 1.0 by default). Multiply input by 10×, or lower the LIF threshold: `LIF(hidden, threshold=0.1, ...)`.
2. **Surrogate too narrow**. `triangular(k=2)` has gradient support only in `|x| < 0.5`. If pre-activations concentrate outside that, no gradient flows. Try `arctan()` (wider) or reduce `k`.
3. **`silence_reg` too weak**. If you're training with activity regularization, make sure it actually pushes silent units to fire. `spyx.fn.silence_reg(min_spikes=4.0)` is a floor.

## Symptom 4 — Training accuracy climbs but validation stays flat

Overfitting, not a Spyx-specific issue. Standard fixes apply:

1. **Add `spyx.data.shift_augment(max_shift=16, axes=(2,))`** — cheap per-batch data augmentation along the channel axis.
2. **Reduce `HIDDEN`** or depth.
3. **Add dropout** via a Flax NNX dropout between layers; Spyx doesn't wrap this natively but `nnx.Dropout` composes into `spyx.nn.Sequential` cleanly.
4. **Lower learning rate** and train longer.

## Symptom 5 — Gradients explode / weights go to infinity

Recurrent SNN-specific. Likely causes:

1. **BPTT through too many timesteps without clipping**. Long SHD sequences (T=128) need `optax.clip_by_global_norm(1.0)`.
2. **Recurrent weights unbounded**. The `recurrent_w` in `RIF`/`RLIF`/`RCuBaLIF` is not clipped by construction. Add an L2 penalty: multiply a small `(recurrent_w**2).sum()` into the loss.

## Symptom 6 — QAT model way worse than fp32

Expected if using `bitnet_ternary_rules()` from scratch; BitNet usually wants fine-tuning from a trained fp32 checkpoint, not from-scratch training.

1. Train the fp32 model first.
2. Call `qmodel = spyx.quant.quantize(fp32_model, ...)` — this wraps the *trained* weights.
3. Fine-tune the quantized model for a small number of epochs with a lower learning rate (`1e-4`).

If int8 (`linear_only_rules()`) is diverging, try `linear_only_rules("int8", None)` — weights-only quantization, leaving activations in fp32. Often recovers most of the accuracy with the same memory profile.

## Before escalating

Always run:

```bash
uv run python scripts/check_install.py
```

If that fails, the problem is install-level, not training-level — invoke `setup-gpu` or `smoke-check`.
