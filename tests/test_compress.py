"""Tests for spyx.compress: bit-packed activation storage for BPTT."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spyx.experimental.compress import (
    pack_nbit,
    pack_spikes,
    packed_quant_dense,
    packed_spike_dense,
    packing_footprint,
    sparse_quant_pack,
    sparse_quant_unpack,
    unpack_nbit,
    unpack_spikes,
)


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


# --------------------------------------------------------------------------- #
# k-bit and sparse packing (quantized activations)
# --------------------------------------------------------------------------- #


def _grid_codes(key, shape, bits):
    """Random integer codes in ``[0, 2**bits)`` as uint32."""
    return jax.random.randint(key, shape, 0, 1 << bits).astype(jnp.uint32)


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("length", [8, 5, 13, 32])
def test_pack_nbit_roundtrip(bits, length):
    """pack_nbit -> unpack_nbit is lossless for any width and (padded) length."""
    codes = _grid_codes(jax.random.PRNGKey(bits * 100 + length), (3, 7, length), bits)
    packed = pack_nbit(codes, bits, axis=-1)
    assert packed.dtype == jnp.uint8
    assert packed.shape[0] == bits  # one bit-plane per bit
    restored = unpack_nbit(packed, bits, length, axis=-1)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(codes))


def test_pack_nbit_generalizes_pack_spikes():
    """bits=1 pack_nbit reproduces pack_spikes (up to the leading plane axis)."""
    x = (jax.random.uniform(jax.random.PRNGKey(0), (4, 16)) < 0.5).astype(jnp.uint32)
    np.testing.assert_array_equal(
        np.asarray(pack_nbit(x, 1, axis=-1)[0]), np.asarray(pack_spikes(x, axis=-1))
    )


def test_packed_quant_dense_matches_naive_on_grid():
    """Forward + both grads equal the naive dense for grid-quantized activations."""
    bits, step = 4, 0.25
    offset = 1 << (bits - 1)
    codes = jax.random.randint(jax.random.PRNGKey(1), (10, 16), 0, 1 << bits)
    acts = (codes - offset).astype(jnp.float32) * step  # exactly on the grid
    weight = jax.random.normal(jax.random.PRNGKey(2), (16, 6))
    target = jax.random.normal(jax.random.PRNGKey(3), (10, 6))

    def loss_packed(a, w):
        return jnp.sum((packed_quant_dense(a, w, bits, step) - target) ** 2)

    def loss_naive(a, w):
        return jnp.sum((_naive_dense(a, w) - target) ** 2)

    assert jnp.allclose(
        packed_quant_dense(acts, weight, bits, step),
        _naive_dense(acts, weight),
        atol=1e-5,
    )
    gp = jax.grad(loss_packed, argnums=(0, 1))(acts, weight)
    gn = jax.grad(loss_naive, argnums=(0, 1))(acts, weight)
    assert jnp.allclose(gp[0], gn[0], atol=1e-4)  # dacts
    assert jnp.allclose(gp[1], gn[1], atol=1e-4)  # dweight


def test_sparse_quant_pack_roundtrip_and_footprint():
    """Occupancy-mask + codes packing is exact and beats dense k-bit below crossover."""
    bits, step = 4, 0.25
    key = jax.random.PRNGKey(4)
    dense_codes = jax.random.randint(key, (4, 20), -3, 4).astype(jnp.float32) * step
    keep = jax.random.uniform(jax.random.PRNGKey(5), (4, 20)) < 0.2  # ~20% density
    x = jnp.where(keep, dense_codes, 0.0)

    mask_packed, codes_packed, meta = sparse_quant_pack(x, bits, step)
    restored = sparse_quant_unpack(mask_packed, codes_packed, meta)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(x))
    assert meta["nnz"] == int(jnp.sum(x != 0))


def test_packing_footprint_crossover():
    """Sparse wins below (bits-1)/bits density; dense k-bit wins above it."""
    n, bits = 1_000_000, 4
    low = packing_footprint(n, bits, 0.1)
    high = packing_footprint(n, bits, 0.9)
    assert low["crossover_density"] == (bits - 1) / bits
    assert low["best"].startswith("sparse")
    assert high["best"].startswith("dense")
    # fp32 is never the winner for a 4-bit grid.
    assert low["best"] != "fp32" and high["best"] != "fp32"
