import jax
import jax.numpy as jnp

from spyx import axn


def _check_forward_is_heaviside(act):
    x = jnp.array([-1.0, -0.01, 0.0, 0.01, 1.0])
    y = act(x)
    # Heaviside with our convention: x > 0 -> 1, else 0.
    assert jnp.array_equal(y, jnp.array([0.0, 0.0, 0.0, 1.0, 1.0]))


def _check_grad_is_finite_and_positive_at_zero(act):
    g = jax.grad(lambda u: jnp.sum(act(u)))(jnp.array([-0.5, 0.0, 0.5]))
    assert jnp.all(jnp.isfinite(g))
    assert g[1] > 0  # surrogate gradient at threshold should pass signal


def test_heaviside_module_level():
    x = jnp.array([-2.0, 0.0, 2.0])
    assert jnp.array_equal(axn.heaviside(x), jnp.array([0.0, 0.0, 1.0]))


def test_custom_default_is_straight_through():
    act = axn.custom()
    _check_forward_is_heaviside(act)
    g = jax.grad(lambda u: jnp.sum(act(u)))(jnp.array([0.0, 1.0]))
    # Default bwd is identity, so grad equals input.
    assert jnp.allclose(g, jnp.array([0.0, 1.0]))


def test_superspike():
    act = axn.superspike()
    _check_forward_is_heaviside(act)
    _check_grad_is_finite_and_positive_at_zero(act)


def test_arctan():
    act = axn.arctan()
    _check_forward_is_heaviside(act)
    _check_grad_is_finite_and_positive_at_zero(act)


def test_triangular():
    act = axn.triangular()
    _check_forward_is_heaviside(act)
    _check_grad_is_finite_and_positive_at_zero(act)


def test_boxcar():
    act = axn.boxcar()
    _check_forward_is_heaviside(act)
    _check_grad_is_finite_and_positive_at_zero(act)


def test_tanh():
    act = axn.tanh()
    _check_forward_is_heaviside(act)
    _check_grad_is_finite_and_positive_at_zero(act)


def test_boxcar_zero_outside_window():
    act = axn.boxcar(width=2, height=0.5)
    g = jax.grad(lambda u: jnp.sum(act(u)))(jnp.array([-2.0, 0.0, 2.0]))
    # Outside |x| > width/2 the surrogate gradient should be zero.
    assert g[0] == 0.0
    assert g[2] == 0.0
    assert g[1] == 0.5
