"""Tests for spyx.compress: bit-packed activation storage for BPTT."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spyx.compress import pack_spikes, packed_spike_dense, unpack_spikes


def _naive_dense(spikes, weight):
    """Reference: plain matmul over the trailing feature axis."""
    in_features = spikes.shape[-1]
    flat = spikes.reshape(-1, in_features)
    out = flat @ weight
    return out.reshape(*spikes.shape[:-1], weight.shape[-1])


@pytest.mark.parametrize("length", [8, 16, 5, 7, 13, 33])
@pytest.mark.parametrize("axis", [-1, 0])
def test_pack_unpack_roundtrip(length, axis):
    """pack -> unpack is lossless on random {0,1} tensors, any length."""
    key = jax.random.PRNGKey(length + (axis + 1) * 100)
    shape = [3, length] if axis == -1 else [length, 4]
    x = (jax.random.uniform(key, tuple(shape)) < 0.5).astype(jnp.float32)

    packed = pack_spikes(x, axis=axis)
    assert packed.dtype == jnp.uint8

    restored = unpack_spikes(packed, length, axis=axis)
    assert restored.shape == x.shape
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(x.astype(jnp.uint8)))


def test_packed_dense_forward_matches_naive():
    """Forward output equals the naive spikes @ weight."""
    kspk, kw = jax.random.split(jax.random.PRNGKey(0))
    spikes = (jax.random.uniform(kspk, (12, 7, 20)) < 0.4).astype(jnp.float32)
    weight = jax.random.normal(kw, (20, 5))

    got = packed_spike_dense(spikes, weight)
    ref = _naive_dense(spikes, weight)
    assert got.shape == ref.shape
    assert jnp.allclose(got, ref, atol=1e-6)


def test_packed_dense_gradients_match_naive():
    """grad w.r.t. BOTH weight and spikes equals the naive matmul's grads."""
    kspk, kw, kt = jax.random.split(jax.random.PRNGKey(1), 3)
    spikes = (jax.random.uniform(kspk, (10, 16)) < 0.5).astype(jnp.float32)
    weight = jax.random.normal(kw, (16, 6))
    target = jax.random.normal(kt, (10, 6))

    def loss_packed(s, w):
        return jnp.sum((packed_spike_dense(s, w) - target) ** 2)

    def loss_naive(s, w):
        return jnp.sum((_naive_dense(s, w) - target) ** 2)

    g_packed = jax.grad(loss_packed, argnums=(0, 1))(spikes, weight)
    g_naive = jax.grad(loss_naive, argnums=(0, 1))(spikes, weight)

    # dspikes
    assert jnp.allclose(g_packed[0], g_naive[0], atol=1e-5)
    # dweight
    assert jnp.allclose(g_packed[1], g_naive[1], atol=1e-5)


def test_saved_residual_is_uint8_and_compact():
    """The bit-packed residual is uint8 and ~1/8 the element count."""
    key = jax.random.PRNGKey(2)
    n = 20
    spikes = (jax.random.uniform(key, (12, 7, n)) < 0.5).astype(jnp.float32)

    packed = pack_spikes(spikes, axis=-1)
    assert packed.dtype == jnp.uint8

    expected_bytes = spikes.shape[0] * spikes.shape[1] * math.ceil(n / 8)
    assert packed.size == expected_bytes
    # For n=20 -> 3 bytes vs 20 floats: strictly smaller element count.
    assert packed.size < spikes.size
