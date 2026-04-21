"""Standalone demo of spyx.ssm.

Builds an LRU and an S5Diag layer, trains them on a tiny copy task, and then
shows the qwix-quantized version of an SSM stack. Useful as a sanity check
without needing the scaling-experiments notebook runtime.

Run with:

    uv run python scripts/ssm_demo.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import spyx
import spyx.nn as snn
from spyx import ssm


def train_lru_on_copy_task(T: int = 32, B: int = 16, d_model: int = 4) -> None:
    print("\n== LRU trained on identity (copy) task ==")
    rngs = nnx.Rngs(0)
    layer = ssm.LRU(d_model=d_model, d_state=16, rngs=rngs)
    optimizer = nnx.Optimizer(layer, optax.adam(5e-3), wrt=nnx.Param)

    u = jax.random.normal(jax.random.PRNGKey(0), (T, B, d_model))

    @nnx.jit
    def step(model, optimizer, u):
        def loss_fn(m):
            return jnp.mean((m(u) - u) ** 2)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    for epoch in range(150):
        loss = float(step(layer, optimizer, u))
        if epoch % 25 == 0 or epoch == 149:
            print(f"  step {epoch:3d}: loss={loss:.4f}")


def s5_diag_forward_demo(T: int = 64, B: int = 2, d_model: int = 8) -> None:
    print("\n== S5Diag forward pass (HiPPO-LegS init) ==")
    rngs = nnx.Rngs(1)
    layer = ssm.S5Diag(d_model=d_model, d_state=32, rngs=rngs)
    u = jax.random.normal(jax.random.PRNGKey(2), (T, B, d_model))
    y = layer(u)
    lam, _, _ = layer._complex_matrices()
    print(f"  output shape: {y.shape}")
    print(f"  |λ| range:    [{float(jnp.abs(lam).min()):.4f}, {float(jnp.abs(lam).max()):.4f}]")


def hybrid_snn_ssm_demo() -> None:
    print("\n== hybrid SNN → SSM → readout ==")
    rngs = nnx.Rngs(0)
    snn_front = snn.Sequential(
        nnx.Linear(4, 8, use_bias=False, rngs=rngs),
        snn.LIF((8,), activation=spyx.axn.triangular(), rngs=rngs),
    )
    ssm_layer = ssm.LRU(d_model=8, d_state=16, rngs=rngs)
    readout = nnx.Linear(8, 3, use_bias=False, rngs=rngs)

    T, B = 12, 4
    u = jax.random.normal(jax.random.PRNGKey(0), (T, B, 4))
    spikes, _ = snn.run(snn_front, u)
    h = ssm_layer(spikes)
    logits = readout(h.sum(axis=0))
    print(f"  input:  (T={T}, B={B}, C=4)")
    print(f"  spikes: {spikes.shape}  sparsity: {float(spikes.mean()):.3f}")
    print(f"  logits: {logits.shape}")


def quantized_ssm_demo() -> None:
    print("\n== quantize the Linear layers around an SSM (int8 + BitNet ternary) ==")
    if not spyx.quant.available():
        print("  qwix not installed; skipping")
        return

    rngs = nnx.Rngs(0)

    class SSMBlock(nnx.Module):
        def __init__(self, in_dim, hidden, out_dim, *, rngs):
            self.pre = nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs)
            self.ssm = ssm.LRU(d_model=hidden, d_state=8, rngs=rngs)
            self.post = nnx.Linear(hidden, out_dim, use_bias=False, rngs=rngs)

        def __call__(self, u):
            return self.post(self.ssm(self.pre(u)))

    T, B = 8, 2
    sample = jax.random.normal(jax.random.PRNGKey(0), (T, B, 4))

    for label, rules in (
        ("int8  W+A", spyx.quant.linear_only_rules("int8", "int8")),
        ("ternary BitNet", spyx.quant.bitnet_ternary_rules()),
    ):
        model = SSMBlock(4, 16, 3, rngs=nnx.Rngs(0))
        qmodel = spyx.quant.quantize(model, sample, rules=rules)
        out_fp = model(sample)
        out_q = qmodel(sample)
        max_diff = float(jnp.max(jnp.abs(out_fp - out_q)))
        print(f"  {label:16s} -> max |fp - q|: {max_diff:.4f}")


if __name__ == "__main__":
    train_lru_on_copy_task()
    s5_diag_forward_demo()
    hybrid_snn_ssm_demo()
    quantized_ssm_demo()
