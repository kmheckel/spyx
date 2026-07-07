"""Sigma-delta / graded-spike neuron: equivalence, sparsity, telescoping, gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import spyx.nn as snn
from spyx.experimental import SigmaDelta, graded_quant


def _seq_scan(neuron, x):
    """Reference: scan ``__call__`` over time -> stacked graded events."""
    V = neuron.initial_state(x.shape[1])
    outs = []
    for t in range(x.shape[0]):
        s, V = neuron(x[t], V)
        outs.append(s)
    return jnp.stack(outs)


@pytest.mark.parametrize("beta", [0.5, 0.9])
@pytest.mark.parametrize("step", [0.25, 1.0])
def test_parallel_equals_sequential(beta, step):
    """The `.parallel` associative-scan path == scanning `__call__` over time."""
    T, B, C = 40, 8, 16
    neuron = SigmaDelta((C,), beta=beta, step=step, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, C))
    s_par = neuron.parallel(x)
    s_seq = _seq_scan(neuron, x)
    assert float(jnp.max(jnp.abs(s_par - s_seq))) < 1e-5


def test_sparse_on_redundant_input():
    """A constant (temporally redundant) input -> membrane settles -> events vanish."""
    T, B, C = 60, 4, 32
    neuron = SigmaDelta((C,), beta=0.8, step=0.5, rngs=nnx.Rngs(0))
    x_const = jnp.broadcast_to(jnp.ones((C,)), (T, B, C))
    s = neuron.parallel(x_const)
    # after the transient, the membrane is stable so nearly all events are zero.
    late_sparsity = float(jnp.mean(s[T // 2 :] == 0))
    assert late_sparsity > 0.9

    # a noisy (non-redundant) input transmits strictly more events than the constant one.
    x_noisy = jax.random.normal(jax.random.PRNGKey(2), (T, B, C)) * 2.0
    s_noisy = neuron.parallel(x_noisy)
    assert float(jnp.mean(s_noisy != 0)) > float(jnp.mean(s != 0))


def test_telescoping_reconstruction():
    """Graded events integrate back to the membrane: cumsum(s)_t tracks V_t."""
    T, B, C = 50, 4, 16
    beta, step = 0.9, 0.05  # fine grid -> tight tracking
    neuron = SigmaDelta((C,), beta=beta, step=step, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(3), (T, B, C))
    # membrane trace via the same scan the neuron uses
    A = jnp.broadcast_to(beta, x.shape)
    _, V = jax.lax.associative_scan(snn._leaky_associative_op, (A, x), axis=0)
    recon = jnp.cumsum(neuron.parallel(x), axis=0)
    # feedforward delta form: per-step error <= step/2, drift ~ sqrt(T)*step.
    assert float(jnp.max(jnp.abs(recon - V))) < 3.0 * step * np.sqrt(T)


def test_graded_quant_is_sparse_and_ste():
    """graded_quant rounds to the grid (zero in the dead-zone) with an STE gradient."""
    d = jnp.array([0.0, 0.1, 0.4, 0.6, -0.6, 2.3])
    q = graded_quant(d, 1.0)
    # |d| < 0.5 rounds to 0 (the sparsity dead-zone); else to the nearest grid point.
    assert jnp.allclose(q, jnp.array([0.0, 0.0, 0.0, 1.0, -1.0, 2.0]))
    # straight-through: gradient of the (piecewise-constant) round is passed through as 1.
    g = jax.grad(lambda z: jnp.sum(graded_quant(z, 1.0)))(d)
    assert jnp.allclose(g, jnp.ones_like(d))


def test_drops_into_sequential_run():
    """SigmaDelta honours the (x, state) -> (out, new_state) contract via spyx.nn.run."""
    T, B, C, H = 20, 4, 8, 12
    net = snn.Sequential(
        nnx.Linear(C, H, rngs=nnx.Rngs(0)),
        SigmaDelta((H,), beta=0.9, step=0.5, rngs=nnx.Rngs(1)),
        nnx.Linear(H, 3, rngs=nnx.Rngs(2)),
        snn.LI((3,), rngs=nnx.Rngs(3)),
    )
    x = jax.random.normal(jax.random.PRNGKey(4), (T, B, C))
    out, _ = snn.run(net, x)
    assert out.shape == (T, B, 3)
    assert jnp.all(jnp.isfinite(out))


def test_beta_gradient_flows():
    """STE lets a gradient reach the learnable leak through the quantizer."""
    T, B, C = 16, 4, 8
    neuron = SigmaDelta((C,), step=0.5, rngs=nnx.Rngs(0))  # learnable per-unit beta
    x = jax.random.normal(jax.random.PRNGKey(5), (T, B, C))

    def loss(m):
        return jnp.mean(m.parallel(x) ** 2)

    grads = nnx.grad(loss)(neuron)
    leaves = [jnp.asarray(g) for g in jax.tree.leaves(grads)]
    assert leaves and any(bool(jnp.any(g != 0.0)) for g in leaves)
