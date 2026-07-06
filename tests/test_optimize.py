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


def test_fit_raises_on_empty_train_iter():
    """fit() should raise a clear error when an epoch yields no batches."""
    import pytest

    model, _x, _targets, forward = _make_model_and_data()
    Loss = spyx.fn.integral_crossentropy()

    def loss_fn(m, x, targets):
        return Loss(forward(m), targets)

    def empty_iter():
        return iter(())

    with pytest.raises(RuntimeError, match="yielded no batches"):
        opt.fit(model, optax.adam(1e-3), loss_fn, empty_iter, epochs=1)


def test_compile_fit_whole_loop_jit():
    """compile_fit compiles the whole epochs x batches loop and trains."""
    import numpy as np

    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(6, 12, use_bias=False, rngs=rngs),
        snn.LIF((12,), activation=spyx.axn.superspike(), rngs=rngs),
        nnx.Linear(12, 3, use_bias=False, rngs=rngs),
        snn.LI((3,), rngs=rngs),
    )

    class Wrap(nnx.Module):
        def __init__(self, net):
            self.net = net

        def __call__(self, x):
            return snn.run(self.net, x, batch_major=True)[0]

    m = Wrap(model)
    rng = np.random.default_rng(0)

    def stage(nb, B=16, T=8, C=6, K=3):
        xs, ys = [], []
        for _ in range(nb):
            lab = rng.integers(0, K, B)
            x = (rng.random((B, T, C)) < 0.05).astype(np.float32)
            for i in range(B):
                lo = lab[i] * (C // K)
                x[i, :, lo : lo + C // K] += rng.random((T, C // K)) < 0.4
            xs.append(jnp.asarray(np.clip(x, 0, 1)))
            ys.append(jnp.asarray(lab))
        return jnp.stack(xs), jnp.stack(ys)

    X, Y = stage(4)
    tX, tY = stage(2)
    ce = spyx.fn.integral_crossentropy(smoothing=0.2)
    acc = spyx.fn.integral_accuracy()

    trained, hist = opt.compile_fit(
        m,
        optax.adam(5e-3),
        lambda mm, x, y: ce(mm(x), y),
        (X, Y),
        epochs=30,
        eval_data=(tX, tY),
        metric_fn=lambda mm, x, y: acc(mm(x), y)[0],
    )
    # per-epoch history arrays and a trained module back
    assert hist["train_loss"].shape == (30,)
    assert hist["eval_metric"].shape == (30,)
    assert float(hist["train_loss"][-1]) < float(hist["train_loss"][0])
    assert isinstance(trained, nnx.Module)


def test_compile_fit_es_solver():
    """The ES solver plugs gradient-free training into the same compiled loop."""
    from spyx.experimental import evolve

    rngs = nnx.Rngs(0)

    class Wrap(nnx.Module):
        def __init__(self, rngs):
            self.net = snn.Sequential(
                nnx.Linear(6, 10, use_bias=False, rngs=rngs),
                snn.LIF((10,), activation=spyx.axn.superspike(), rngs=rngs),
                nnx.Linear(10, 3, use_bias=False, rngs=rngs),
                snn.LI((3,), rngs=rngs),
            )

        def __call__(self, x):
            return snn.run(self.net, x, batch_major=True)[0]

    import numpy as np

    rng = np.random.default_rng(1)
    lab = rng.integers(0, 3, 16)
    x = (rng.random((16, 8, 6)) < 0.05).astype(np.float32)
    for i in range(16):
        x[i, :, lab[i] * 2 : lab[i] * 2 + 2] += rng.random((8, 2)) < 0.4
    X = jnp.stack([jnp.asarray(np.clip(x, 0, 1))] * 3)
    Y = jnp.stack([jnp.asarray(lab)] * 3)
    ce = spyx.fn.integral_crossentropy(smoothing=0.2)

    _, hist = opt.compile_fit(
        Wrap(rngs),
        evolve.es(optax.adam(5e-3), num_samples=12, sigma=0.02),
        lambda mm, x, y: ce(mm(x), y),
        (X, Y),
        epochs=25,
        key=jax.random.PRNGKey(0),
    )
    assert bool(hist["train_loss"][-1] <= hist["train_loss"][0])
