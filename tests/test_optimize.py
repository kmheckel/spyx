"""Tests for spyx.optimize (#26)."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import spyx
import spyx.nn as snn
import spyx.optimize as opt


def _make_model_and_data():
    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(4, 8, use_bias=False, rngs=rngs),
        snn.LIF((8,), activation=spyx.axn.triangular(), rngs=rngs),
        nnx.Linear(8, 3, use_bias=False, rngs=rngs),
        snn.LI((3,), rngs=rngs),
    )

    T, B = 6, 4
    x = jax.random.normal(jax.random.PRNGKey(0), (T, B, 4))
    targets = jnp.array([0, 1, 2, 0])

    def forward(m):
        traces, _ = snn.run(m, x)  # (T, B, n_classes)
        return jnp.transpose(traces, (1, 0, 2))  # (B, T, n_classes)

    return model, x, targets, forward


def test_fit_reduces_loss_and_returns_history():
    model, x, targets, forward = _make_model_and_data()
    Loss = spyx.fn.integral_crossentropy()
    Acc = spyx.fn.integral_accuracy()

    def loss_fn(m, x, targets):
        return Loss(forward(m), targets)

    def eval_fn(m, x, targets):
        traces = forward(m)
        acc, _preds = Acc(traces, targets)
        return acc, Loss(traces, targets)

    # Static single-batch iterator for the test.
    def train_iter():
        return iter([(x, targets)])

    def eval_iter():
        return iter([(x, targets)])

    history = opt.fit(
        model,
        optax.adam(5e-3),
        loss_fn,
        train_iter,
        epochs=20,
        eval_iter=eval_iter,
        eval_fn=eval_fn,
    )
    assert len(history) == 20
    assert "train_loss" in history[0]
    assert "eval_acc" in history[0]
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_fit_without_eval_returns_train_loss_only():
    model, x, targets, forward = _make_model_and_data()
    Loss = spyx.fn.integral_crossentropy()

    def loss_fn(m, x, targets):
        return Loss(forward(m), targets)

    def train_iter():
        return iter([(x, targets)])

    history = opt.fit(model, optax.adam(1e-3), loss_fn, train_iter, epochs=3)
    assert len(history) == 3
    assert set(history[0].keys()) == {"train_loss"}


def test_fit_invokes_on_epoch_end_callback():
    model, x, targets, forward = _make_model_and_data()
    Loss = spyx.fn.integral_crossentropy()

    def loss_fn(m, x, targets):
        return Loss(forward(m), targets)

    def train_iter():
        return iter([(x, targets)])

    seen = []
    opt.fit(
        model,
        optax.adam(1e-3),
        loss_fn,
        train_iter,
        epochs=4,
        on_epoch_end=lambda epoch, metrics: seen.append((epoch, metrics["train_loss"])),
    )
    assert [s[0] for s in seen] == [0, 1, 2, 3]
    assert all(jnp.isfinite(jnp.array([s[1] for s in seen])))
