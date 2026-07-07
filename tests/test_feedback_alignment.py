"""Feedback Alignment / Direct FA: exact forward, sign/angle alignment, learning."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import spyx.nn as snn
from spyx.experimental.feedback_alignment import (
    FALinear,
    dfa_gradient,
    fa_dense,
)


def _cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12))


def test_fa_forward_matches_linear():
    """FALinear's forward equals nnx.Linear with the same weight and bias."""
    in_f, out_f = 12, 7
    lin = nnx.Linear(in_f, out_f, rngs=nnx.Rngs(0))
    fa = FALinear(in_f, out_f, rngs=nnx.Rngs(1))
    fa.weight = nnx.Param(lin.kernel[...])
    fa.bias = nnx.Param(lin.bias[...])
    x = jax.random.normal(jax.random.PRNGKey(2), (5, in_f))
    assert jnp.allclose(fa(x), lin(x), atol=1e-6)


def test_fa_dense_backward_uses_feedback_not_transpose():
    """fa_dense input-grad is g @ feedback, weight-grad stays exact (x^T g)."""
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 6))
    w = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
    b = jax.random.normal(jax.random.PRNGKey(2), (3, 6))  # feedback (out, in)
    g = jax.random.normal(jax.random.PRNGKey(3), (4, 3))
    _, vjp = jax.vjp(lambda x_, w_: fa_dense(x_, w_, b), x, w)
    dx, dw = vjp(g)
    assert jnp.allclose(dx, g @ b, atol=1e-6)  # random feedback, not g @ w.T
    assert jnp.allclose(dw, x.T @ g, atol=1e-6)  # weight grad exact


def _fa_mlp(seed):
    """Feedforward spiking MLP with FALinear layers (PSU_LIF fires on drive)."""
    r = nnx.Rngs(seed)
    return snn.Sequential(
        FALinear(8, 16, rngs=r),
        snn.PSU_LIF((16,), beta=0.0, threshold=0.0, rngs=r),
        FALinear(16, 3, rngs=r),
    )


def _bp_mirror(fa_model):
    """A plain-Linear twin of an FA MLP sharing its forward weights (BP ref)."""
    r = nnx.Rngs(99)
    lin0 = nnx.Linear(8, 16, rngs=r)
    lin0.kernel = nnx.Param(fa_model.layers[0].weight[...])
    lin0.bias = nnx.Param(fa_model.layers[0].bias[...])
    lin1 = nnx.Linear(16, 3, rngs=r)
    lin1.kernel = nnx.Param(fa_model.layers[2].weight[...])
    lin1.bias = nnx.Param(fa_model.layers[2].bias[...])
    return snn.Sequential(
        lin0, snn.PSU_LIF((16,), beta=0.0, threshold=0.0, rngs=r), lin1
    )


def _feedforward(model, x):
    h = x
    for layer in model.layers:
        if hasattr(layer, "initial_state"):
            h, _ = layer(h, layer.initial_state(x.shape[0]))
        else:
            h = layer(h)
    return h


def _ce(logits, y):
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


def test_fa_gradient_sign_alignment_and_learning():
    """FA aligns with BP in sign/angle and the loss decreases while training."""
    key = jax.random.PRNGKey(0)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (64, 8))
    w_true = jax.random.normal(ky, (8, 3))
    y = jnp.argmax(x @ w_true, axis=-1)

    fa = _fa_mlp(1)

    def fa_loss(m):
        return _ce(_feedforward(m, x), y)

    def bp_loss_of(m):
        return _ce(_feedforward(m, x), y)

    tx = optax.sgd(0.2)
    opt = nnx.Optimizer(fa, tx, wrt=nnx.Param)
    loss0 = float(fa_loss(fa))

    cos_hidden = []
    for _ in range(40):
        grads = nnx.grad(fa_loss)(fa)
        # Compare the FA hidden-weight grad against the true-BP hidden-weight grad.
        bp = _bp_mirror(fa)
        g_bp = nnx.grad(bp_loss_of)(bp)
        g_fa_h = grads["layers"][0]["weight"][...]
        g_bp_h = g_bp["layers"][0]["kernel"][...]
        cos_hidden.append(_cos(g_fa_h, g_bp_h))
        opt.update(fa, grads)

    loss1 = float(fa_loss(fa))
    assert loss1 < loss0  # FA actually trains
    # Alignment: the FA hidden pseudo-gradient acquires a positive projection on BP.
    assert np.mean(cos_hidden[-10:]) > 0.0
    # Sign agreement of the two hidden gradients is better than chance.
    g_fa_h = nnx.grad(fa_loss)(fa)["layers"][0]["weight"][...]
    bp = _bp_mirror(fa)
    g_bp_h = nnx.grad(bp_loss_of)(bp)["layers"][0]["kernel"][...]
    sign_agree = float(jnp.mean(jnp.sign(g_fa_h) == jnp.sign(g_bp_h)))
    assert sign_agree > 0.5


def test_dfa_reaches_nontrivial_accuracy():
    """DFA trains a feedforward spiking MLP to well above chance on a 3-way task."""
    key = jax.random.PRNGKey(0)
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (96, 8))
    w_true = jax.random.normal(kw, (8, 3))
    y = jnp.argmax(x @ w_true, axis=-1)

    r = nnx.Rngs(3)
    model = snn.Sequential(
        nnx.Linear(8, 32, rngs=r),
        snn.PSU_LIF((32,), beta=0.0, threshold=0.0, rngs=r),
        nnx.Linear(32, 32, rngs=r),
        snn.PSU_LIF((32,), beta=0.0, threshold=0.0, rngs=r),
        nnx.Linear(32, 3, rngs=r),
    )
    # One fixed random feedback per hidden neuron, shape (n_out=3, hidden).
    fk = jax.random.split(jax.random.PRNGKey(7), 2)
    feedbacks = [
        jax.random.normal(fk[0], (3, 32)) * 0.1,
        jax.random.normal(fk[1], (3, 32)) * 0.1,
    ]

    tx = optax.adam(3e-3)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    def accuracy():
        return float(jnp.mean(jnp.argmax(_feedforward(model, x), -1) == y))

    acc0 = accuracy()
    for _ in range(200):
        grads = dfa_gradient(model, x, y, feedbacks, loss_fn=_ce)
        opt.update(model, grads)
    acc1 = accuracy()

    assert acc1 > acc0
    assert acc1 > 0.6  # non-trivial: well above the 1/3 chance floor


def test_falinear_composes_with_run_over_time():
    """FALinear drops into spyx.nn.run; BPTT differentiates through fa_dense."""
    T, B, C, H = 10, 4, 6, 8
    r = nnx.Rngs(0)
    net = snn.Sequential(
        FALinear(C, H, rngs=r),
        snn.LIF((H,), beta=0.8, rngs=r),
        FALinear(H, 3, rngs=r),
        snn.LI((3,), beta=0.8, rngs=r),
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, C))
    y = jnp.zeros((B,), dtype=jnp.int32)

    def loss(m):
        out, _ = snn.run(m, x)
        return _ce(jnp.sum(out, axis=0), y)

    grads = nnx.grad(loss)(net)
    # A finite FA gradient flows to the first FALinear's weight through the scan.
    g0 = grads["layers"][0]["weight"][...]
    assert jnp.all(jnp.isfinite(g0))
    assert float(jnp.linalg.norm(g0)) > 0.0
