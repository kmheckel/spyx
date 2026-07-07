"""Local three-factor (e-prop/OTTT-style) plasticity: trace decay, online
association learning, and meta-learnable/evolvable coefficients.

None of these download data, so no ``@pytest.mark.network``.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from spyx.experimental.hybrid import es_gradient
from spyx.experimental.local_learning import ThreeFactorLIF, surrogate_deriv


def test_surrogate_deriv_matches_superspike():
    """The forward pseudo-derivative equals SuperSpike's backward slope."""
    v = jnp.linspace(-2.0, 2.0, 11)
    assert jnp.allclose(surrogate_deriv(v, 25.0), 1.0 / (1.0 + 25.0 * jnp.abs(v)) ** 2)
    # peaks at the threshold (v == 0) and decays away from it.
    assert float(surrogate_deriv(jnp.array(0.0))) == 1.0
    assert float(surrogate_deriv(jnp.array(1.0))) < float(
        surrogate_deriv(jnp.array(0.1))
    )


def test_eligibility_trace_decays_by_gamma():
    """With no pre-filter memory, a pre pulse then silence gives geometric decay.

    Setting ``alpha = 0`` makes ``pbar_t = pre_t``, so after the input stops the
    coincidence term is zero and ``e_t = gamma * e_{t-1}`` exactly. ``eta = 0``
    freezes ``W`` so the drive (and thus ``psi``) never changes the test.
    """
    B, IN, OUT = 1, 3, 2
    gamma = 0.7
    layer = ThreeFactorLIF(
        IN, OUT, eta=0.0, gamma=gamma, alpha=0.0, beta=0.5, threshold=0.0
    )
    T = 6
    pre = jnp.zeros((T, B, IN)).at[0].set(1.0)  # single pulse at t=0
    mod = jnp.ones((T, B))

    # Manually scan to inspect the eligibility trace at every step.
    state = layer.initial_state(B)
    traces = []
    for t in range(T):
        _, state = layer((pre[t], mod[t]), state)
        traces.append(state[2])  # e is the third carry element
    e = jnp.stack(traces)

    peak = float(jnp.max(jnp.abs(e[0])))
    assert peak > 0.0  # the pulse created eligibility
    for t in range(1, T - 1):
        ratio = jnp.abs(e[t + 1]) / (jnp.abs(e[t]) + 1e-12)
        assert jnp.allclose(ratio[jnp.abs(e[t]) > 1e-9], gamma, atol=1e-5)


def test_reward_modulated_rule_learns_association():
    """A hand-set positive-reward rule strengthens the driven synapse online.

    Input unit 0 is active every step; with a positive modulator the three-factor
    update should grow ``W[0, :]`` (the eligible synapses) while the silent input
    unit 1's weights stay put — a trivial reward-modulated association.
    """
    B, IN, OUT = 4, 2, 3
    layer = ThreeFactorLIF(
        IN, OUT, eta=0.05, gamma=0.9, alpha=0.5, beta=0.0, threshold=0.0
    )
    T = 30
    pre = jnp.zeros((T, B, IN)).at[:, :, 0].set(1.0)  # only input 0 fires
    mod = jnp.ones((T, B))  # constant positive reward

    # Warm-start W slightly positive so the neuron is driven from the first step.
    V, pbar, e, W = layer.initial_state(B)
    W = W + 0.1
    _, (_, _, _, W_final) = layer.apply_sequence(pre, mod, (V, pbar, e, W))

    # The active input's synapses grew; the silent input's did not.
    assert jnp.all(W_final[0] > W[0] + 1e-3)
    assert jnp.allclose(W_final[1], W[1])
    # A negative modulator drives the same synapses the other way.
    _, (_, _, _, W_neg) = layer.apply_sequence(pre, -mod, (V, pbar, e, W))
    assert jnp.all(W_neg[0] < W[0] - 1e-3)


def _meta_loss(layer, pre, mod, target):
    """Meta-objective: drive the online-learned W toward a target matrix."""
    _, (_, _, _, W_final) = layer.apply_sequence(pre, mod)
    return jnp.mean((W_final - target) ** 2)


def test_coefficients_are_differentiable():
    """A meta-loss on the learned W has finite, nonzero grads w.r.t. the coeffs."""
    B, IN, OUT, T = 3, 2, 2, 12
    layer = ThreeFactorLIF(IN, OUT, threshold=0.0, beta=0.0)
    key = jax.random.PRNGKey(0)
    pre = jax.random.bernoulli(key, 0.5, (T, B, IN)).astype(jnp.float32)
    mod = jnp.ones((T, B))
    target = jnp.ones((IN, OUT)) * 0.2

    grads = nnx.grad(_meta_loss)(layer, pre, mod, target)
    leaves = [jnp.asarray(g) for g in jax.tree.leaves(grads)]
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(bool(jnp.any(g != 0.0)) for g in leaves)


def test_coefficients_are_evolvable():
    """The coeffs are the only nnx.Params, so ES (forward-only) reaches them."""
    B, IN, OUT, T = 3, 2, 2, 12
    layer = ThreeFactorLIF(IN, OUT, threshold=0.0, beta=0.0)
    key = jax.random.PRNGKey(1)
    pre = jax.random.bernoulli(key, 0.5, (T, B, IN)).astype(jnp.float32)
    mod = jnp.ones((T, B))
    target = jnp.ones((IN, OUT)) * 0.2

    # Only eta/gamma/alpha/beta are Params; W and the traces are carried state.
    _, params, _ = nnx.split(layer, nnx.Param, ...)
    assert len(jax.tree.leaves(params)) == 4

    grads = es_gradient(
        layer,
        lambda m: _meta_loss(m, pre, mod, target),
        key,
        num_samples=4,
        sigma=0.02,
    )
    leaves = [jnp.asarray(g) for g in jax.tree.leaves(grads)]
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
