"""TTT fast-weight layer: hebb parallel==sequential, delta sequential-only, gradients."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import spyx.nn as snn
from spyx.experimental import TTTFastWeight


def _seq_scan(layer, x):
    """Reference: scan ``__call__`` over time and stack the outputs."""
    W = layer.initial_state(x.shape[1])
    outs = []
    for t in range(x.shape[0]):
        out, W = layer(x[t], W)
        outs.append(out)
    return jnp.stack(outs)


@pytest.mark.parametrize("decay", [0.9, 1.0])
@pytest.mark.parametrize("eta", [0.3, 1.0])
def test_hebb_parallel_equals_sequential(decay, eta):
    """The hebb ``.parallel`` associative scan == scanning ``__call__`` over time."""
    T, B, C, OUT, KD, VD = 32, 6, 5, 7, 4, 9  # non-square, distinct dims
    layer = TTTFastWeight(
        C,
        OUT,
        key_dim=KD,
        val_dim=VD,
        eta=eta,
        decay=decay,
        rule="hebb",
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, C))
    s_par = layer.parallel(x)
    s_seq = _seq_scan(layer, x)
    assert s_par.shape == (T, B, OUT)
    # Equal in exact arithmetic; at decay=eta=1.0 the fast weights grow (the
    # documented unbounded-memory regime), so the tree-reduction scan and the
    # left-fold sequential differ only by float32 reassociation -> relative tol.
    assert jnp.allclose(s_par, s_seq, rtol=1e-4, atol=1e-5)


def test_delta_parallel_raises():
    """The delta rule has a matrix transition — ``.parallel`` refuses, points to run."""
    layer = TTTFastWeight(5, 5, rule="delta", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 3, 5))
    with pytest.raises(NotImplementedError, match="spyx.nn.run"):
        layer.parallel(x)


def test_invalid_rule_raises():
    with pytest.raises(ValueError, match="hebb"):
        TTTFastWeight(4, 4, rule="bogus", rngs=nnx.Rngs(0))


def test_initial_state_shape():
    layer = TTTFastWeight(6, 3, key_dim=4, val_dim=5, rngs=nnx.Rngs(0))
    W = layer.initial_state(8)
    assert W.shape == (8, 5, 4)
    assert jnp.all(W == 0.0)


@pytest.mark.parametrize("rule", ["hebb", "delta"])
def test_drops_into_run(rule):
    """Both rules honour the (x, state) -> (out, new_state) contract under run."""
    T, B, C, H = 12, 4, 6, 8
    net = snn.Sequential(
        nnx.Linear(C, H, rngs=nnx.Rngs(0)),
        TTTFastWeight(H, H, rule=rule, rngs=nnx.Rngs(1)),
        nnx.Linear(H, 3, rngs=nnx.Rngs(2)),
        snn.LI((3,), rngs=nnx.Rngs(3)),
    )
    x = jax.random.normal(jax.random.PRNGKey(4), (T, B, C))
    out, _ = snn.run(net, x)
    assert out.shape == (T, B, 3)
    assert jnp.all(jnp.isfinite(out))


def test_gradient_flows_to_slow_params():
    """A downstream loss reaches every slow param through the fast-weight recurrence."""
    T, B, C, OUT = 16, 4, 5, 3
    layer = TTTFastWeight(C, OUT, eta=0.5, decay=0.9, rule="hebb", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, C))

    def loss(m):
        return jnp.mean(m.parallel(x) ** 2)

    grads = nnx.grad(loss)(layer)
    leaves = [jnp.asarray(g) for g in jax.tree.leaves(grads)]
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(bool(jnp.any(g != 0.0)) for g in leaves)


def test_delta_gradient_flows_through_sequential():
    """The delta rule's online recurrence still passes gradient to eta / projections."""
    T, B, C, OUT = 14, 4, 5, 3
    layer = TTTFastWeight(C, OUT, eta=0.3, rule="delta", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, C))

    def loss(m):
        return jnp.mean(_seq_scan(m, x) ** 2)

    grads = nnx.grad(loss)(layer)
    assert bool(jnp.any(jnp.asarray(grads["eta"][...]) != 0.0))
    assert bool(jnp.any(grads["W_k"]["kernel"][...] != 0.0))


def test_delta_corrects_repeated_association():
    """On a repeated (implicit) key the delta rule's prediction error shrinks.

    Uses a small ``eta`` so ``eta*||k||^2 < 2`` and the delta transition
    ``(1 - eta k k^T)`` is contractive — the stable regime the docstring warns must
    be respected (large ``eta`` makes the fast weights expansive and diverge).
    """
    T, B, C, OUT = 40, 1, 4, 4
    x = jnp.broadcast_to(
        jax.random.normal(jax.random.PRNGKey(0), (C,)), (T, B, C)
    )  # a held (redundant) token -> a repeated key/value
    delta = TTTFastWeight(
        C,
        OUT,
        key_dim=C,
        val_dim=OUT,
        eta=0.1,
        decay=1.0,
        rule="delta",
        rngs=nnx.Rngs(1),
    )

    # track the internal prediction error |v - W_{t-1} k| over the sequence
    W = delta.initial_state(B)
    errs = []
    for t in range(T):
        k = delta.W_k(x[t])
        v = delta.W_v(x[t])
        errs.append(float(jnp.mean(jnp.abs(v - jnp.einsum("bvk,bk->bv", W, k)))))
        _, W = delta(x[t], W)
    # the delta rule drives its own prediction error down on a repeated key.
    assert errs[-1] < errs[0]
