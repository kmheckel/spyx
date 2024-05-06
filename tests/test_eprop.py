from spyx.axn import abs_linear
import jax.numpy as jnp
import jax
import numpy as np


def test_axn_abs_linear():
    """
    Test the abs_linear surrogate gradient function.
    """

    # Test the abs_linear surrogate gradient function.
    x = jnp.linspace(-10, 10, 100, dtype=jnp.float32)
    f = abs_linear(dampening_factor=0.3)

    y = f(x)
    y_grad = jax.jacrev(f)(x).diagonal()

    y_true = jnp.greater(x, 0.).astype(jnp.float32)
    y_true_grad = jnp.maximum(0.3*(1 - jnp.abs(x)), 0).astype(jnp.float32)
 
    assert np.allclose(y, y_true, atol=1e-5)
    assert np.allclose(y_grad, y_true_grad, atol=1e-5)

def test_eprop():
    pass