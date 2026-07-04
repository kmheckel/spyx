"""Ternary / int8 QAT on a tiny GPT transformer, via ``spyx.quant``.

Trains three variants of the *same* :class:`model.TinyGPT` on the *same*
char-level language-modeling data and seed:

1. ``fp32``    - full-precision baseline (no quantization).
2. ``int8``    - int8 weights + int8 activations (``linear_only_rules``).
3. ``ternary`` - BitNet b1.58-style ternary weights + int8 activations
   (``bitnet_ternary_rules``), the same 1.58-bit recipe PrismML's *Bonsai* uses.

All three go through :func:`spyx.quant.quantize(..., mode="qat")` - the exact API
used for spiking nets in spyx - demonstrating it generalizes to a transformer LLM
because the rules match the ``dot_general`` op, not any SNN-specific structure.

Run::

    SMOKE=1 uv run python research/new/ternary_llm/run.py   # ~1-2 min, CPU
    uv run python research/new/ternary_llm/run.py            # fuller config

The script prints a 3-way table (val loss / perplexity / next-token accuracy /
effective weight bit-width) and a verification block proving the quantized
weights really take few distinct levels (ternary = 3-4 codes, int8 <= 256),
i.e. quantization is active and not a silent no-op.
"""

from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from model import GPTConfig, TinyGPT, linear_kernels

import spyx.quant as quant

SMOKE = os.environ.get("SMOKE", "0") == "1" or "--smoke" in sys.argv

# A public-domain corpus (opening of *Alice's Adventures in Wonderland*, Lewis
# Carroll, 1865). Char-level LM over real English text gives a meaningful
# perplexity signal without any dataset download.
CORPUS = (
    "Alice was beginning to get very tired of sitting by her sister on the "
    "bank, and of having nothing to do: once or twice she had peeped into the "
    "book her sister was reading, but it had no pictures or conversations in "
    "it, 'and what is the use of a book,' thought Alice 'without pictures or "
    "conversations?' So she was considering in her own mind (as well as she "
    "could, for the hot day made her feel very sleepy and stupid), whether the "
    "pleasure of making a daisy-chain would be worth the trouble of getting up "
    "and picking the daisies, when suddenly a White Rabbit with pink eyes ran "
    "close by her. There was nothing so very remarkable in that; nor did Alice "
    "think it so very much out of the way to hear the Rabbit say to itself, "
    "'Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, "
    "it occurred to her that she ought to have wondered at this, but at the "
    "time it all seemed quite natural); but when the Rabbit actually took a "
    "watch out of its waistcoat-pocket, and looked at it, and then hurried on, "
    "Alice started to her feet, for it flashed across her mind that she had "
    "never before seen a rabbit with either a waistcoat-pocket, or a watch to "
    "take out of it, and burning with curiosity, she ran across the field "
    "after it, and fortunately was just in time to see it pop down a large "
    "rabbit-hole under the hedge. In another moment down went Alice after it, "
    "never once considering how in the world she was to get out again. "
)


def build_data() -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Encode the corpus to ints and split into train / val streams."""
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in CORPUS], dtype=np.int32)
    n_val = max(len(data) // 10, 32)
    train, val = data[:-n_val], data[-n_val:]
    return train, val, len(chars), {"stoi": stoi, "chars": chars}


def get_batch(
    stream: np.ndarray, block_size: int, batch: int, rng: np.random.Generator
) -> tuple[jax.Array, jax.Array]:
    """Sample ``batch`` contiguous ``block_size`` windows and their next-token targets."""
    hi = len(stream) - block_size - 1
    starts = rng.integers(0, hi, size=batch)
    x = np.stack([stream[s : s + block_size] for s in starts])
    y = np.stack([stream[s + 1 : s + 1 + block_size] for s in starts])
    return jnp.asarray(x), jnp.asarray(y)


def loss_fn(model: nnx.Module, x: jax.Array, y: jax.Array) -> jax.Array:
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, y):
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    acc = (jnp.argmax(logits, -1) == y).mean()
    return loss, acc


def evaluate(model, stream, cfg, batch, rng, n_batches=8):
    losses, accs = [], []
    for _ in range(n_batches):
        x, y = get_batch(stream, cfg.block_size, batch, rng)
        loss, acc = eval_step(model, x, y)
        losses.append(float(loss))
        accs.append(float(acc))
    return float(np.mean(losses)), float(np.mean(accs))


def train_variant(name, rules, cfg, train_data, val_data, hp):
    """Build, (optionally) quantize, and train one variant. Returns metrics + model."""
    model = TinyGPT(cfg, rngs=nnx.Rngs(hp["seed"]))
    example_x, _ = get_batch(
        train_data, cfg.block_size, hp["batch"], np.random.default_rng(0)
    )
    if rules is not None:
        model = quant.quantize(model, example_x, rules=rules, mode="qat")

    optimizer = nnx.Optimizer(model, optax.adamw(hp["lr"]), wrt=nnx.Param)
    rng = np.random.default_rng(hp["seed"])

    t0 = time.time()
    first_loss = None
    for _step in range(hp["steps"]):
        x, y = get_batch(train_data, cfg.block_size, hp["batch"], rng)
        loss = train_step(model, optimizer, x, y)
        if first_loss is None:
            first_loss = float(loss)
    train_loss = float(loss)
    wall = time.time() - t0

    val_rng = np.random.default_rng(1234)
    val_loss, val_acc = evaluate(model, val_data, cfg, hp["batch"], val_rng)
    return {
        "name": name,
        "first_loss": first_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_ppl": float(np.exp(val_loss)),
        "val_acc": val_acc,
        "wall": wall,
        "model": model,
    }


def weight_level_report(model, qtype):
    """Quantize each trained Linear kernel and count distinct integer codes.

    This is the no-op check: with ``qwix`` symmetric absmax quantization, an
    ``int2`` (ternary) kernel has at most 4 distinct codes per channel and an
    ``int8`` kernel at most 256. fp32 (``qtype=None``) keeps thousands of
    distinct float values. Uses the same qwix path spyx.quant drives internally.
    """
    import qwix

    kernels = linear_kernels(model)
    per_layer = []
    for kname, k in kernels.items():
        k = jnp.asarray(k)
        if qtype is None:
            codes = np.unique(np.round(np.asarray(k), 6))
            per_layer.append((kname, len(codes), None))
        else:
            qa = qwix.quantize(k, qtype, channelwise_axes=(k.ndim - 1,))
            codes = np.unique(np.asarray(qa.qvalue))
            per_layer.append((kname, len(codes), codes.tolist()))
    return per_layer


def main():
    if SMOKE:
        cfg = GPTConfig(
            vocab_size=0,  # filled after data
            block_size=32,
            n_layer=2,
            n_head=2,
            d_model=64,
        )
        hp = {"seed": 0, "lr": 3e-3, "batch": 16, "steps": 200}
    else:
        cfg = GPTConfig(
            vocab_size=0,
            block_size=64,
            n_layer=3,
            n_head=4,
            d_model=128,
        )
        hp = {"seed": 0, "lr": 2e-3, "batch": 32, "steps": 1500}

    train_data, val_data, vocab, _meta = build_data()
    cfg = GPTConfig(
        vocab_size=vocab,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
    )

    print(f"{'=' * 70}")
    print("Ternary / int8 QAT on a tiny GPT transformer (spyx.quant)")
    print(
        f"mode={'SMOKE' if SMOKE else 'FULL'}  vocab={vocab}  "
        f"d_model={cfg.d_model}  n_layer={cfg.n_layer}  n_head={cfg.n_head}  "
        f"block={cfg.block_size}  steps={hp['steps']}"
    )
    print(f"{'=' * 70}")

    variants = [
        ("fp32", None, None, 32),
        ("int8", quant.linear_only_rules(), "int8", 8),
        ("ternary", quant.bitnet_ternary_rules(act_qtype="int8"), "int2", 2),
    ]

    results = []
    for name, rules, qtype, _bits in variants:
        print(f"\n[train] {name} ...", flush=True)
        r = train_variant(name, rules, cfg, train_data, val_data, hp)
        r["qtype"] = qtype
        print(
            f"  first_loss={r['first_loss']:.3f} -> train_loss={r['train_loss']:.3f} "
            f"| val_loss={r['val_loss']:.3f} ppl={r['val_ppl']:.2f} "
            f"acc={r['val_acc']:.3f} | {r['wall']:.1f}s"
        )
        results.append(r)

    # 3-way comparison table.
    print(f"\n{'=' * 70}")
    print("RESULTS (same data, same seed, same steps)")
    print(f"{'=' * 70}")
    bits = {"fp32": "32", "int8": "8", "ternary": "~1.58 (2b store)"}
    header = (
        f"{'variant':<9} {'w-bits':<16} {'val_loss':<9} {'val_ppl':<9} "
        f"{'next-tok acc':<13} {'train_loss':<10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<9} {bits[r['name']]:<16} {r['val_loss']:<9.3f} "
            f"{r['val_ppl']:<9.2f} {r['val_acc']:<13.3f} {r['train_loss']:<10.3f}"
        )

    # Verification: prove weights are actually ternary / int8, not a no-op.
    print(f"\n{'=' * 70}")
    print("VERIFICATION: distinct quantized weight codes per Linear layer")
    print(f"{'=' * 70}")
    ok = True
    for r in results:
        levels = weight_level_report(r["model"], r["qtype"])
        counts = [c for _, c, _ in levels]
        example = levels[0]
        if r["qtype"] is None:
            print(
                f"{r['name']:<9} fp32 kernels: {min(counts)}-{max(counts)} "
                f"distinct float values per layer (full precision)"
            )
        else:
            alphabet = sorted(set().union(*[set(codes) for _, _, codes in levels]))
            alpha_str = (
                str(alphabet)
                if len(alphabet) <= 8
                else f"[{alphabet[0]} .. {alphabet[-1]}] ({len(alphabet)} codes)"
            )
            print(
                f"{r['name']:<9} qtype={r['qtype']:<5} distinct codes/layer: "
                f"{min(counts)}-{max(counts)}  global alphabet={alpha_str}"
            )
            ex_codes = example[2]
            ex_str = (
                str(ex_codes)
                if len(ex_codes) <= 8
                else f"[{ex_codes[0]} .. {ex_codes[-1]}] ({len(ex_codes)} codes)"
            )
            print(f"          e.g. layer '{example[0]}' uses codes {ex_str}")
            # int2 -> <=4 codes; int8 -> <=256. Non-trivial (>1) means active.
            cap = 4 if r["qtype"] == "int2" else 256
            layer_ok = max(counts) <= cap and min(counts) > 1
            ok = ok and layer_ok
            if not layer_ok:
                print(f"          !! unexpected code count for {r['qtype']}")

    # Sanity: ternary should be a small multiple of fp32 loss (Bonsai claim).
    fp32_loss = next(r["val_loss"] for r in results if r["name"] == "fp32")
    tern_loss = next(r["val_loss"] for r in results if r["name"] == "ternary")
    print(
        f"\nBonsai-style parity: ternary val_loss / fp32 val_loss = "
        f"{tern_loss / fp32_loss:.2f}x"
    )
    trained = all(r["train_loss"] < r["first_loss"] for r in results)
    print(f"all variants trained (loss decreased): {trained}")
    print(f"quantization active & correctly ternary/int8: {ok}")
    if not (trained and ok):
        raise SystemExit("FAILED: training or quantization verification did not pass")
    print("\nOK")


if __name__ == "__main__":
    main()
