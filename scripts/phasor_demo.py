"""Standalone demo of spyx.phasor.

Builds a small phasor MLP, runs it both in the continuous (complex) domain and
through the spiking-inference wrapper, and prints round-trip statistics so a
reader can sanity-check the implementation without needing a full notebook
runtime.

Run with:

    uv run python scripts/phasor_demo.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from spyx import phasor


def main() -> None:
    rngs = nnx.Rngs(0)
    in_features, hidden, out = 8, 16, 4

    print("== continuous (complex) forward pass ==")
    model = phasor.PhasorMLP(
        in_features=in_features, hidden_features=hidden, out_features=out, depth=2,
        rngs=rngs,
    )
    x = jax.random.uniform(jax.random.PRNGKey(1), (5, in_features))
    logits = model(x)
    print(f"  input shape:  {x.shape}")
    print(f"  logit shape:  {logits.shape}")
    print(f"  logit dtype:  {logits.dtype}")
    print(f"  logit range:  [{float(logits.min()):.3f}, {float(logits.max()):.3f}]")

    print("\n== phase <-> spike round-trip ==")
    theta = jnp.linspace(-jnp.pi + 1e-3, jnp.pi - 1e-3, 16)
    for T in (8, 32, 128):
        spikes = phasor.phase_to_spikes(theta, T)
        recovered = phasor.spikes_to_phase(spikes, T)
        max_err = float(jnp.max(jnp.abs(recovered - theta)))
        bin_size = 2.0 * jnp.pi / T
        print(
            f"  T={T:3d}  bin={bin_size:.4f}  max round-trip err={max_err:.4f}"
            f"  ({'OK' if max_err <= bin_size else 'FAIL'})"
        )

    print("\n== spiking inference through a single PhasorLinear ==")
    linear = phasor.PhasorLinear(in_features=hidden, out_features=hidden, rngs=rngs)
    sp = phasor.SpikingPhasor(linear, period_T=32)

    # Build a batched spike train from random phases.
    theta_in = jax.random.uniform(jax.random.PRNGKey(42), (3, hidden), minval=-jnp.pi, maxval=jnp.pi)
    spikes_in = phasor.phase_to_spikes(theta_in, T=32)
    spikes_out = sp(spikes_in)
    spikes_per_unit_in = float(jnp.mean(jnp.sum(spikes_in, axis=0)))
    spikes_per_unit_out = float(jnp.mean(jnp.sum(spikes_out, axis=0)))
    print(f"  input  spikes/unit/cycle: {spikes_per_unit_in:.2f}")
    print(f"  output spikes/unit/cycle: {spikes_per_unit_out:.2f}")
    print(f"  output shape: {spikes_out.shape}")

    print("\n== gradients flow through complex layers ==")
    optimizer_target = jnp.array([0, 1, 2, 0, 3])

    def loss_fn(m):
        out = m(x)
        return jnp.mean(jax.nn.log_softmax(out)[jnp.arange(5), optimizer_target] * -1)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    flat = jax.tree_util.tree_leaves(grads)
    print(f"  loss: {float(loss):.4f}")
    print(f"  number of complex grad leaves: {sum(g.dtype == jnp.complex64 for g in flat)}")
    print(
        f"  max |grad|: "
        f"{float(max(jnp.max(jnp.abs(g)) for g in flat)):.4e}"
    )


if __name__ == "__main__":
    main()
