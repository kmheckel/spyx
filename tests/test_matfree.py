"""Tests for the matmul-free primitives + ternary block."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from spyx.experimental import matfree


def test_ternary_weights_are_in_the_ternary_set():
    w = jax.random.normal(jax.random.PRNGKey(0), (16, 16))
    wt, scale = matfree.ternary_weights(w)
    assert set(float(v) for v in jnp.unique(wt)) <= {-1.0, 0.0, 1.0}
    assert float(scale) > 0


def test_power_of_two_weights_are_exact_powers_of_two():
    w = jax.random.normal(jax.random.PRNGKey(1), (16, 16))
    wq = matfree.power_of_two_weights(w)
    log2 = jnp.log2(jnp.abs(wq))
    assert bool(jnp.all(jnp.abs(log2 - jnp.round(log2)) < 1e-5))


@pytest.mark.parametrize("Layer", [matfree.TernaryLinear, matfree.ShiftAddLinear])
def test_matfree_linear_forward_and_ste_training(Layer):
    """Forward is finite and STE lets the shadow weights learn a toy regression."""
    rngs = nnx.Rngs(0)

    class Net(nnx.Module):
        def __init__(self, rngs):
            self.a = Layer(8, 16, rngs=rngs)
            self.b = Layer(16, 4, rngs=rngs)

        def __call__(self, x):
            return self.b(jax.nn.relu(self.a(x)))

    net = Net(rngs)
    x = jax.random.normal(jax.random.PRNGKey(1), (32, 8))
    y = jnp.tanh(x @ jax.random.normal(jax.random.PRNGKey(2), (8, 4)))
    assert bool(jnp.all(jnp.isfinite(net(x))))

    opt = nnx.Optimizer(net, optax.adam(5e-3), wrt=nnx.Param)

    @nnx.jit
    def step(m, o):
        loss, g = nnx.value_and_grad(lambda mm: jnp.mean((mm(x) - y) ** 2))(m)
        o.update(m, g)
        return loss

    losses = [float(step(net, opt)) for _ in range(200)]
    assert losses[-1] < losses[0]


def test_matmulfree_block_is_causal_and_trains():
    """The MLGRU token mixer is strictly causal; a tiny LM over it learns."""
    mix = matfree.MLGRU(24, 24, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 12, 24))  # [B, T, D]
    y0 = mix(x)
    x2 = x.at[:, 6, :].add(10.0)  # perturb only t=6
    y1 = mix(x2)
    assert bool(jnp.allclose(y0[:, :6], y1[:, :6], atol=1e-5))  # past untouched
    assert bool(jnp.any(jnp.abs(y0[:, 6:] - y1[:, 6:]) > 1e-4))  # future changed

    # a tiny two-block LM learns a copy task
    V, T, B = 8, 16, 16
    seq = jax.random.randint(jax.random.PRNGKey(3), (B, T), 0, V)

    class LM(nnx.Module):
        def __init__(self, r):
            self.emb = nnx.Embed(V, 24, rngs=r)
            self.b1 = matfree.MatMulFreeBlock(24, rngs=r)
            self.head = nnx.Linear(24, V, rngs=r)

        def __call__(self, ids):
            return self.head(self.b1(self.emb(ids)))

    net = LM(nnx.Rngs(0))
    opt = nnx.Optimizer(net, optax.adam(3e-3), wrt=nnx.Param)

    @nnx.jit
    def step(m, o):
        loss, g = nnx.value_and_grad(
            lambda mm: optax.softmax_cross_entropy_with_integer_labels(
                mm(seq), seq
            ).mean()
        )(m)
        o.update(m, g)
        return loss

    losses = [float(step(net, opt)) for _ in range(120)]
    assert losses[-1] < losses[0] * 0.5
