"""Numeric-parity tests for :class:`spyx.experimental.AssociativeLIF`.

Two independent layers of assurance:

A. **Analytic** (always runs, no external dependency): the associative-scan
   membrane and a scan of ``__call__`` both match the closed-form geometric
   recurrence ``mem_t = sum_{s<=t} beta**(t-s) * x_s``, and spikes follow the
   Heaviside threshold.
B. **Cross-check vs snnTorch** (skipped if torch/snntorch are missing): the
   spyx membrane and spikes match ``snntorch.StateLeaky`` once ``beta`` is
   reparameterised via :meth:`AssociativeLIF.beta_from_snntorch`.  A regression
   assertion documents that the *naive* (unconverted) mapping does **not**
   match, guarding against anyone "fixing" the reparameterisation wrongly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from spyx import nn
from spyx.experimental import AssociativeLIF


def _reference_membrane(x, beta):
    """Closed-form reset-free membrane via an explicit python recurrence."""
    T = x.shape[0]
    V = jnp.zeros(x.shape[1:])
    out = []
    for t in range(T):
        V = beta * V + x[t]
        out.append(V)
    return jnp.stack(out)


def test_is_psu_lif_subclass_and_helpers_roundtrip():
    """AssociativeLIF is a PSU_LIF subclass; beta helpers invert each other."""
    from spyx.nn import PSU_LIF

    assert issubclass(AssociativeLIF, PSU_LIF)

    for b in (0.5, 0.8, 0.9, 0.95):
        beta_spyx = AssociativeLIF.beta_from_snntorch(b)
        back = AssociativeLIF.snntorch_beta_from_beta(beta_spyx)
        assert jnp.allclose(back, b, atol=1e-6)
        # StateLeaky's effective per-step decay is exp(-(1-beta_snn)).
        assert jnp.allclose(beta_spyx, jnp.exp(-(1.0 - b)), atol=1e-7)


@pytest.mark.parametrize("beta", [0.5, 0.8, 0.9, 0.95])
@pytest.mark.parametrize("T", [8, 64])
def test_analytic_membrane_and_spikes(beta, T):
    """parallel + scan(__call__) match the geometric recurrence and spikes."""
    B, C = 3, 6
    model = AssociativeLIF((C,), beta=beta, threshold=1.0, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(T), (T, B, C))

    ref_mem = _reference_membrane(x, jnp.clip(model.beta[...], 0, 1))

    # Membrane recovered from the associative scan itself.
    A = jnp.broadcast_to(jnp.clip(model.beta[...], 0, 1), x.shape)
    _, par_mem = jax.lax.associative_scan(nn._leaky_associative_op, (A, x), axis=0)
    assert jnp.allclose(par_mem, ref_mem, atol=1e-5)

    # Spikes are a strict Heaviside on the membrane.
    par_spk = model.parallel(x)
    assert jnp.allclose(par_spk, (ref_mem > model.threshold).astype(x.dtype))

    # Sequential scan of __call__ matches parallel exactly.
    seq_spk, _ = nn.run(model, x)
    assert jnp.allclose(seq_spk, par_spk, atol=1e-6)


def test_parallel_equals_sequential_per_unit_beta():
    """Learnable per-unit beta: parallel == sequential."""
    model = AssociativeLIF((12,), rngs=nnx.Rngs(1))
    x = jax.random.normal(jax.random.key(0), (20, 4, 12))
    par = model.parallel(x)
    seq, _ = nn.run(model, x)
    assert jnp.allclose(par, seq, atol=1e-6)


@pytest.mark.parametrize("beta_snn", [0.5, 0.8, 0.9, 0.95])
@pytest.mark.parametrize("T", [8, 64])
@pytest.mark.parametrize("scalar_beta", [True, False])
def test_snntorch_stateleaky_parity(beta_snn, T, scalar_beta):
    """spyx AssociativeLIF == snnTorch StateLeaky for a matched, converted beta."""
    torch = pytest.importorskip("torch")
    snntorch = pytest.importorskip("snntorch")

    B, C = 3, 5
    rng = np.random.default_rng(T)
    x = rng.standard_normal((T, B, C)).astype(np.float32)

    sl = snntorch.StateLeaky(
        beta=beta_snn,
        channels=C,
        output=True,
        surrogate_disable=True,
        kernel_truncation_steps=None,
    )
    sl_spk, sl_mem = sl(torch.tensor(x))
    sl_mem = sl_mem.detach().numpy()
    sl_spk = sl_spk.detach().numpy()

    beta_spyx = float(AssociativeLIF.beta_from_snntorch(beta_snn))
    model = AssociativeLIF((C,), beta=beta_spyx, threshold=1.0, rngs=nnx.Rngs(0))
    if not scalar_beta:
        # Exercise the per-unit (broadcast) beta path with the same value on
        # every channel; PSU_LIF's __init__ only takes a scalar fixed beta.
        model.beta = nnx.Param(jnp.full((C,), beta_spyx, dtype=jnp.float32))

    xa = jnp.asarray(x)
    A = jnp.broadcast_to(jnp.clip(model.beta[...], 0, 1), xa.shape)
    _, mem = jax.lax.associative_scan(nn._leaky_associative_op, (A, xa), axis=0)
    spk = model.parallel(xa)

    mem_diff = float(np.max(np.abs(np.asarray(mem) - sl_mem)))
    assert mem_diff < 1e-5, f"membrane max-abs-diff {mem_diff}"
    assert np.array_equal(np.asarray(spk), sl_spk)


def test_snntorch_naive_mapping_regression():
    """Regression: using beta == beta_snn (no reparam) does NOT match StateLeaky.

    Guards against anyone "simplifying" the reparameterisation away.
    """
    torch = pytest.importorskip("torch")
    snntorch = pytest.importorskip("snntorch")

    B, C, T, beta_snn = 3, 5, 32, 0.9
    rng = np.random.default_rng(1)
    x = rng.standard_normal((T, B, C)).astype(np.float32)

    sl = snntorch.StateLeaky(
        beta=beta_snn,
        channels=C,
        output=True,
        surrogate_disable=True,
        kernel_truncation_steps=None,
    )
    _, sl_mem = sl(torch.tensor(x))
    sl_mem = sl_mem.detach().numpy()

    # Naive (wrong) mapping: pass beta_snn straight through.
    naive = AssociativeLIF((C,), beta=beta_snn, threshold=1.0, rngs=nnx.Rngs(0))
    xa = jnp.asarray(x)
    A = jnp.broadcast_to(jnp.clip(naive.beta[...], 0, 1), xa.shape)
    _, mem = jax.lax.associative_scan(nn._leaky_associative_op, (A, xa), axis=0)
    naive_diff = float(np.max(np.abs(np.asarray(mem) - sl_mem)))
    assert naive_diff > 1e-3, (
        f"naive mapping unexpectedly matched (diff {naive_diff}); "
        "the beta reparameterisation must be preserved"
    )


if __name__ == "__main__":
    test_is_psu_lif_subclass_and_helpers_roundtrip()
    for b in (0.5, 0.9):
        for t in (8, 64):
            test_analytic_membrane_and_spikes(b, t)
    test_parallel_equals_sequential_per_unit_beta()
    print("Analytic tests passed!")
