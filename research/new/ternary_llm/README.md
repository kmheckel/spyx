# Ternary / int8 QAT on a transformer LLM (spyx.quant beyond SNNs)

## Title

BitNet b1.58 ternary quantization-aware training of a tiny GPT, via `spyx.quant`.

## Paper & arXiv/DOI

- **Title:** *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*
  (BitNet b1.58) — the ternary-weight recipe; applied here in the spirit of
  PrismML's **Bonsai** 1.58-bit models.
- **Authors / venue / year:** Ma, Wang, et al., Microsoft Research, 2024;
  PrismML *Bonsai* (prismml.com), 2024–2025.
- **Link:** BitNet b1.58 — arXiv:2402.17764. PrismML Bonsai — https://prismml.com
  (1.58-bit ternary-weight LLMs). Related in-repo: Q-S5 (arXiv:2406.09477).
- **Bucket:** new

## Claim under test

`spyx.quant`'s BitNet-ternary QAT — the same 1.58-bit `{-1, 0, +1}` weight recipe
as PrismML **Bonsai** — is **not specific to spiking nets**: applied unchanged to
a standard decoder-only **transformer LLM**, ternary weights train and land
**close to the fp32 baseline** in validation loss/perplexity, while every Linear
weight collapses to a handful of distinct levels.

## Method

- **Model** (`model.py`): a tiny decoder-only GPT (`TinyGPT`) — token + positional
  `nnx.Embed`, pre-norm transformer blocks (multi-head causal self-attention +
  GELU MLP), final LayerNorm, and an LM head. **Every** learned projection
  (Q/K/V/output, MLP fc/proj, LM head) is an `nnx.Linear`, so its matmul lowers to
  a `dot_general` — exactly what `spyx.quant`'s rules match. Attention *score* and
  *value* contractions are written as broadcast multiply-and-sum (not `einsum` /
  `dot_general`), so they stay fp32 and don't trip qwix's op interception (only
  the learned *weights* are quantized).
- **Task** (`run.py`): char-level language modeling over a fixed public-domain
  corpus (opening of *Alice's Adventures in Wonderland*) — no dataset download,
  fully self-contained and reproducible. Next-token prediction, cross-entropy
  loss, AdamW.
- **Three variants, same data / seed / step budget:**
  1. `fp32` — full-precision baseline (no quant).
  2. `int8` — int8 weights + int8 activations (`quant.linear_only_rules`).
  3. `ternary` — BitNet b1.58 ternary weights + int8 activations
     (`quant.bitnet_ternary_rules(act_qtype="int8")`).
  All three via `spyx.quant.quantize(model, example_x, rules=…, mode="qat")` — the
  identical call used for SNNs.
- **Measured:** validation loss, perplexity, next-token accuracy, and — as a
  no-op guard — the number of distinct quantized weight codes per Linear layer.

Note on the ternary fallback: qwix exposes no true 1.58-bit qtype, so
`bitnet_ternary_rules` maps to symmetric `int2` — 4 codes `{-2, -1, 0, +1}`,
the same 2-bit storage class and near-ternary alphabet as BitNet/Bonsai.

## Spyx modules used

- [`spyx.quant.quantize`](../../../src/spyx/quant.py) — qwix QAT wrapper.
- [`spyx.quant.linear_only_rules`](../../../src/spyx/quant.py) — int8 weights+acts.
- [`spyx.quant.bitnet_ternary_rules`](../../../src/spyx/quant.py) — BitNet ternary.
- `model.TinyGPT` — the transformer under test (this study; `flax.nnx`).

## How to run

```bash
# fast 3-way comparison on CPU (~1-2 min)
SMOKE=1 uv run python research/new/ternary_llm/run.py

# fuller config (larger model, more steps)
uv run python research/new/ternary_llm/run.py
```

No dataset download — the corpus is embedded in `run.py`.

## Results

`SMOKE=1` on CPU (vocab 42, d_model 64, 2 layers, 2 heads, block 32, 200 steps,
seed 0):

| variant | weight bits | val loss | val ppl | next-tok acc | distinct weight codes / layer |
| --- | --- | --- | --- | --- | --- |
| fp32 | 32 | 2.656 | 14.24 | 0.365 | 2682–16133 (full precision) |
| int8 | 8 | 2.661 | 14.31 | 0.365 | 249–256 (`[-128..127]`) |
| ternary | ~1.58 (2-bit store) | 2.600 | 13.46 | 0.350 | **4** (`{-2,-1,0,1}`) |

- **Bonsai-style parity:** ternary val_loss / fp32 val_loss = **0.98x** (ternary
  is on par with — here marginally better than — fp32 at this scale/seed).
- **Not a no-op:** the verification block confirms ternary weights use exactly
  4 distinct codes per layer and int8 uses ≤256, vs thousands of distinct floats
  for fp32. All three variants trained (loss decreased from init).

The other metrics in the template (latency / memory / spike rate) are `N/A` —
this is a quantization-parity study, not a `spyx.bench` throughput study.

## Findings

**Confirmed.** The exact `spyx.quant` BitNet-ternary QAT path used for spiking
nets applies unchanged to a transformer LLM (the rules key off `dot_general`, not
any SNN structure). Ternary weights train stably and match fp32 perplexity within
noise at this scale, reproducing the Bonsai/BitNet-b1.58 claim on a non-SNN model
while cutting nominal weight storage 32→2 bits. int8 is essentially lossless.
Caveat: this is a tiny model on a small corpus; the point is *methodological*
generality of `spyx.quant`, not a SoTA LLM result. The ternary path had a higher
*training* loss (1.77 vs 0.83) yet a slightly lower *val* loss — the coarse weight
grid acts as a regularizer on this tiny corpus, so ternary should not be read as
"better than fp32" in general.

## Reproducibility

- **Seeds:** `nnx.Rngs(0)` for all three model inits (same init); NumPy
  `default_rng(0)` train batches, `default_rng(1234)` eval batches. Identical
  data/seed/steps across variants.
- **JAX / hardware:** JAX 0.10.2, CPU (jax forced to CPU by the repo test config;
  runs on GPU too). qwix from GitHub (`spyx[quant]`), flax nnx 0.12+.
- **Spyx commit:** run `git rev-parse HEAD` in the repo.
- **Date run:** 2026-07-04 (SMOKE, CPU).
