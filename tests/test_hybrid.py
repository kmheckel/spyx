"""Tests for the hybrid surrogate/ES training method.

Covers, per the algorithm in ``spyx.experimental.hybrid``:

* the projection identity ``g_orth ⟂ g_s`` and ``g == g_s`` when ``λ = 0``;
* returned grads match the model's ``Param`` pytree and are finite;
* the antithetic-ES estimate correlates with a *known* analytic gradient on a
  smooth quadratic (cosine > 0.5 with enough samples and small ``σ``);
* one end-to-end hybrid step on a tiny spiking classifier keeps the TRUE loss
  finite and does not blow it up.

None of these download data, so no ``@pytest.mark.network``.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import spyx.axn as axn
import spyx.fn as fn
import spyx.nn as snn
from spyx.experimental.hybrid import (
    _es_flat,
    _sges_flat,
    es_gradient,
    hybrid_diagnostics,
    hybrid_gradient,
    make_hybrid_train_step,
    make_sges_hybrid_train_step,
    sges_gradient,
)


class VecModel(nnx.Module):
    """Trivial model: a single flat ``nnx.Param`` vector ``w``.

    Lets tests express the loss directly as a function of the flat parameter, so
    the analytic gradient is known and the projection algebra is easy to check.
    """

    def __init__(self, w0):
        self.w = nnx.Param(jnp.asarray(w0, dtype=jnp.float32))


class TinyClassifier(nnx.Module):
    """Minimal spiking classifier: Linear -> LIF -> Linear -> LI, time-major.

    ``__call__`` takes ``(T, B, C)`` spikes and returns a ``(B, T, n_classes)``
    logit trace, matching the ``spyx.fn.integral_*`` (sum-over-time) convention.
    """

    def __init__(self, in_dim, hidden, n_classes, activation, *, rngs):
        self.enc = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.lif = snn.LIF((hidden,), activation=activation, rngs=rngs)
        self.dec = nnx.Linear(hidden, n_classes, rngs=rngs)
        self.li = snn.LI((n_classes,), rngs=rngs)
        self.net = snn.Sequential(self.enc, self.lif, self.dec, self.li)

    def __call__(self, x_TBC):
        traces, _ = snn.run(self.net, x_TBC)  # (T, B, n_classes)
        return jnp.transpose(traces, (1, 0, 2))  # (B, T, n_classes)


def test_orthogonality_and_lambda_zero():
    """g_orth ⟂ g_s, and λ=0 recovers pure surrogate descent."""
    key = jax.random.PRNGKey(0)
    model = VecModel(jnp.linspace(-1.0, 1.0, 6))

    # Two genuinely different scalar objectives so g_s and g_es are not parallel.
    def loss_surrogate(m, target):
        return jnp.sum((m.w[...] - target) ** 2)

    def loss_true(m, target):
        # A different landscape (shifted + quartic) so ES points elsewhere.
        return jnp.sum((m.w[...] + 0.5) ** 2 + 0.1 * m.w[...] ** 4)

    target = jnp.zeros((6,))

    _, diag = hybrid_gradient(
        model,
        loss_surrogate,
        loss_true,
        key,
        batch=(target,),
        num_samples=256,
        sigma=0.02,
        lam=1.0,
        return_diagnostics=True,
    )

    # Global orthogonality: <g_orth, g_s> ~ 0 relative to the vector scales.
    dot = float(jnp.dot(diag["g_orth"], diag["g_s"]))
    scale = float(jnp.linalg.norm(diag["g_orth"]) * jnp.linalg.norm(diag["g_s"]))
    assert abs(dot) <= 1e-5 * (scale + 1.0)

    # λ = 0 must return exactly g_s.
    grads0 = hybrid_gradient(
        model,
        loss_surrogate,
        loss_true,
        key,
        batch=(target,),
        num_samples=256,
        sigma=0.02,
        lam=0.0,
    )
    g0_flat = jax.flatten_util.ravel_pytree(grads0)[0]
    assert jnp.allclose(g0_flat, diag["g_s"], atol=1e-6)

    # λ = 1 must return exactly g_s + g_orth.
    grads1 = hybrid_gradient(
        model,
        loss_surrogate,
        loss_true,
        key,
        batch=(target,),
        num_samples=256,
        sigma=0.02,
        lam=1.0,
    )
    g1_flat = jax.flatten_util.ravel_pytree(grads1)[0]
    assert jnp.allclose(g1_flat, diag["g_s"] + diag["g_orth"], atol=1e-6)


def test_normalize_bounds_correction_to_fraction_of_surrogate():
    """normalize=True makes ‖applied correction‖ == λ·‖g_s‖ (and still ⟂ g_s)."""
    key = jax.random.PRNGKey(7)
    model = VecModel(jnp.linspace(-1.0, 1.0, 6))

    def loss_surrogate(m, target):
        return jnp.sum((m.w[...] - target) ** 2)

    def loss_true(m, target):
        return jnp.sum((m.w[...] + 0.5) ** 2 + 0.1 * m.w[...] ** 4)

    target = jnp.zeros((6,))
    lam = 0.3
    kw = dict(batch=(target,), num_samples=256, sigma=0.02)

    grads, diag = hybrid_gradient(
        model,
        loss_surrogate,
        loss_true,
        key,
        lam=lam,
        normalize=True,
        return_diagnostics=True,
        **kw,
    )
    g = jax.flatten_util.ravel_pytree(grads)[0]
    g_s = diag["g_s"]
    correction = g - g_s

    # The applied correction has norm exactly λ·‖g_s‖ ...
    assert jnp.allclose(
        jnp.linalg.norm(correction), lam * jnp.linalg.norm(g_s), rtol=1e-4
    )
    # ... points along g_orth ...
    assert jnp.allclose(correction, diag["lam_eff"] * diag["g_orth"], atol=1e-5)
    # ... stays orthogonal to the surrogate ...
    dot = float(jnp.dot(correction, g_s))
    scale = float(jnp.linalg.norm(correction) * jnp.linalg.norm(g_s))
    assert abs(dot) <= 1e-4 * (scale + 1.0)
    # ... and the reported fraction equals λ.
    assert jnp.allclose(diag["correction_fraction"], lam, rtol=1e-4)

    # normalize=False leaves λ as a raw scale (correction == λ·g_orth).
    _, diag_raw = hybrid_gradient(
        model,
        loss_surrogate,
        loss_true,
        key,
        lam=lam,
        normalize=False,
        return_diagnostics=True,
        **kw,
    )
    assert jnp.allclose(diag_raw["lam_eff"], lam, rtol=1e-6)

    # λ = 0 still recovers pure surrogate even in normalize mode.
    g0 = jax.flatten_util.ravel_pytree(
        hybrid_gradient(
            model,
            loss_surrogate,
            loss_true,
            key,
            lam=0.0,
            normalize=True,
            **kw,
        )
    )[0]
    assert jnp.allclose(g0, g_s, atol=1e-6)


def test_grads_match_param_pytree_and_finite(rngs):
    """Returned grads share the model's Param structure and are finite."""
    model = TinyClassifier(4, 8, 3, axn.superspike(), rngs=rngs)
    T, B, C = 5, 2, 4
    x = (jax.random.uniform(jax.random.PRNGKey(1), (T, B, C)) < 0.3).astype(jnp.float32)
    y = jnp.array([0, 2])

    loss = fn.integral_crossentropy()

    def loss_fn(m, xb, yb):
        return loss(m(xb), yb)

    grads = hybrid_gradient(
        model,
        loss_fn,
        loss_fn,
        jax.random.PRNGKey(2),
        batch=(x, y),
        num_samples=4,
        sigma=0.01,
        lam=1.0,
    )

    params = nnx.state(model, nnx.Param)
    g_leaves = jax.tree_util.tree_leaves(grads)
    p_leaves = jax.tree_util.tree_leaves(params)
    # Same pytree structure as the params.
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(params)
    assert len(g_leaves) == len(p_leaves) and len(g_leaves) > 0
    for g, p in zip(g_leaves, p_leaves, strict=True):
        assert g.shape == p.shape
        assert bool(jnp.all(jnp.isfinite(g)))


def test_antithetic_es_matches_analytic_gradient():
    """On a smooth quadratic the ES estimate aligns with the true gradient."""
    key = jax.random.PRNGKey(3)
    w0 = jnp.array([1.0, -2.0, 0.5, 3.0, -1.5], dtype=jnp.float32)
    model = VecModel(w0)
    center = jnp.array([0.2, 0.1, -0.3, 0.0, 0.7], dtype=jnp.float32)

    def loss_true(m):
        return 0.5 * jnp.sum((m.w[...] - center) ** 2)

    analytic = w0 - center  # ∇ = w - center

    grads = es_gradient(model, loss_true, key, num_samples=400, sigma=0.01)
    g_es = jax.flatten_util.ravel_pytree(grads)[0]

    cos = float(
        jnp.dot(g_es, analytic)
        / (jnp.linalg.norm(g_es) * jnp.linalg.norm(analytic) + 1e-12)
    )
    assert cos > 0.5, f"ES estimate poorly aligned with analytic grad: cos={cos}"


def test_diagnostics_keys():
    """Diagnostics helper exposes the correction magnitudes."""
    model = VecModel(jnp.arange(5.0))

    def loss_s(m):
        return jnp.sum(m.w[...] ** 2)

    def loss_t(m):
        return jnp.sum((m.w[...] - 1.0) ** 2)

    diag = hybrid_diagnostics(
        model, loss_s, loss_t, jax.random.PRNGKey(4), num_samples=32, sigma=0.02
    )
    for k in ("cosine", "g_orth_norm", "g_s_norm", "g_es_norm", "proj"):
        assert k in diag
        assert bool(jnp.isfinite(diag[k]))
    # cosine is a valid cosine similarity.
    assert -1.0 - 1e-4 <= float(diag["cosine"]) <= 1.0 + 1e-4


def test_end_to_end_hybrid_step_keeps_true_loss_finite(rngs):
    """A few hybrid steps on a tiny spiking classifier keep the true loss sane."""
    model = TinyClassifier(6, 12, 3, axn.superspike(), rngs=rngs)
    T, B, C = 8, 4, 6
    dk = jax.random.PRNGKey(5)
    x = (jax.random.uniform(dk, (T, B, C)) < 0.3).astype(jnp.float32)
    y = jnp.array([0, 1, 2, 1])

    # loss_true uses the same hard-Heaviside forward the model already computes;
    # the surrogate only differs in the backward pass, so they share this fn but
    # are used differently (grad vs. forward-only ES).
    ce = fn.integral_crossentropy(smoothing=0.2)

    def loss_fn(m, xb, yb):
        return ce(m(xb), yb)

    optimizer = nnx.Optimizer(model, optax.adam(5e-3), wrt=nnx.Param)
    step = make_hybrid_train_step(loss_fn, loss_fn, num_samples=8, sigma=0.01, lam=0.5)

    key = jax.random.PRNGKey(6)
    losses = []
    for _ in range(4):
        key, sub = jax.random.split(key)
        losses.append(float(step(model, optimizer, sub, x, y)))

    for value in losses:
        assert jnp.isfinite(value)
    # Should not blow up: final within a small tolerance of (or below) the start.
    assert losses[-1] <= losses[0] + 0.5


def test_sges_unbiased_and_lower_variance_than_isotropic_es():
    """Surrogate-steered SGES: unbiased for the true grad, lower variance than ES.

    On a quadratic (grad known), the SGES estimate steered by a biased guide must
    (a) be unbiased (its mean aligns with the analytic gradient), and (b) have
    strictly lower total variance than isotropic ES at the same sample budget —
    the whole point of the Self-Guided-ES stratification.
    """
    key = jax.random.PRNGKey(0)
    kc, kt, kg = jax.random.split(key, 3)
    d = 40
    center = jax.random.normal(kc, (d,))
    theta = jax.random.normal(kt, (d,)) * 0.5

    def f(x):
        return 0.5 * jnp.sum((x - center) ** 2)

    grad_true = theta - center
    guide = grad_true + 0.6 * jax.random.normal(kg, (d,))  # biased surrogate-like

    K, sigma = 12, 1e-3
    seeds = jax.random.split(jax.random.PRNGKey(99), 150)
    es = jax.vmap(lambda k: _es_flat(theta, f, k, num_samples=K, sigma=sigma))(seeds)
    sg = jax.vmap(
        lambda k: _sges_flat(theta, f, guide, k, num_samples=K, sigma=sigma, eps=1e-8)[
            0
        ]
    )(seeds)

    def cos(a, b):
        return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))

    # (a) unbiased: mean estimate aligns with the analytic gradient.
    assert cos(sg.mean(0), grad_true) > 0.95
    # (b) strictly lower total variance than isotropic ES at the same budget.
    assert float(sg.var(0).sum()) < float(es.var(0).sum())
    # and by a clear margin (>2x here) since the guide direction is exact.
    assert float(es.var(0).sum()) > 2.0 * float(sg.var(0).sum())


def test_sges_lambda_zero_recovers_surrogate_and_finite(rngs):
    """sges_gradient at lam=0 is exactly the surrogate gradient; grads are finite."""
    model = TinyClassifier(4, 8, 3, axn.superspike(), rngs=rngs)
    T, B, C = 5, 2, 4
    x = (jax.random.uniform(jax.random.PRNGKey(1), (T, B, C)) < 0.3).astype(jnp.float32)
    y = jnp.array([0, 2])
    loss = fn.integral_crossentropy()

    def loss_fn(m, xb, yb):
        return loss(m(xb), yb)

    g0 = sges_gradient(
        model, loss_fn, loss_fn, jax.random.PRNGKey(2), batch=(x, y), lam=0.0
    )
    g_s = nnx.grad(lambda m: loss_fn(m, x, y))(model)
    for a, b in zip(
        jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g_s), strict=True
    ):
        assert jnp.allclose(a, b, atol=1e-5)

    grads, diag = sges_gradient(
        model,
        loss_fn,
        loss_fn,
        jax.random.PRNGKey(3),
        batch=(x, y),
        num_samples=6,
        lam=1.0,
        return_diagnostics=True,
    )
    params = nnx.state(model, nnx.Param)
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(params)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    for k in ("along_guide_deriv", "g_orth_norm", "g_s_norm", "cosine"):
        assert bool(jnp.isfinite(diag[k]))


def test_sges_hybrid_step_keeps_true_loss_finite(rngs):
    """A few surrogate-steered-SGES steps keep the true loss finite/sane."""
    model = TinyClassifier(6, 12, 3, axn.superspike(), rngs=rngs)
    T, B, C = 8, 4, 6
    x = (jax.random.uniform(jax.random.PRNGKey(5), (T, B, C)) < 0.3).astype(jnp.float32)
    y = jnp.array([0, 1, 2, 1])
    ce = fn.integral_crossentropy(smoothing=0.2)

    def loss_fn(m, xb, yb):
        return ce(m(xb), yb)

    optimizer = nnx.Optimizer(model, optax.adam(5e-3), wrt=nnx.Param)
    step = make_sges_hybrid_train_step(loss_fn, loss_fn, num_samples=8, sigma=0.01)
    key = jax.random.PRNGKey(6)
    losses = []
    for _ in range(4):
        key, sub = jax.random.split(key)
        losses.append(float(step(model, optimizer, sub, x, y)))
    for value in losses:
        assert jnp.isfinite(value)
    assert losses[-1] <= losses[0] + 0.5
