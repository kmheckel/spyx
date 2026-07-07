"""Bit-packed activation storage for memory-efficient BPTT.

Training spiking networks with backpropagation-through-time is dominated,
memory-wise, by the *activations saved for the backward pass*. In an SNN the
activations feeding each linear layer are the spikes, which are exactly
``{0, 1}`` valued. A dense op ``spikes @ weight`` normally stashes the full
floating-point ``spikes`` tensor as its backward residual so it can later form
``dW = spikes^T @ g``. Storing one bit per spike as a float wastes 8x-32x the
memory it needs.

This module bit-packs that residual with :func:`jax.numpy.packbits` (8 spikes
per ``uint8``) and unpacks it lazily inside the backward pass. The forward
output and *both* gradients (w.r.t. ``weight`` and ``spikes``) are numerically
identical to the naive ``spikes @ weight`` -- we only trade a cheap
unpack-recompute for a large cut in the dominant activation residual.

Correctness relies on the input being exactly binary (values in ``{0, 1}``);
:func:`packed_spike_dense` is only valid for spike tensors, not arbitrary
floats.

The lower half of this module generalises the same idea to **quantized** and
**sparse** activations (graded sigma-delta events, ternary, int-N): pack at
``bits`` bits with :func:`pack_nbit` (bit-plane packing), use
:func:`packed_quant_dense` for the k-bit BPTT residual, or — when the tensor is
also sparse — store a 1-bit occupancy mask plus only the nonzero codes with
:func:`sparse_quant_pack`. :func:`packing_footprint` gives the byte counts and
the density crossover between the dense-k-bit and sparse schemes.
"""

import jax
import jax.numpy as jnp

__all__ = [
    "pack_spikes",
    "unpack_spikes",
    "packed_spike_dense",
    "pack_nbit",
    "unpack_nbit",
    "packed_quant_dense",
    "sparse_quant_pack",
    "sparse_quant_unpack",
    "packing_footprint",
]


def pack_spikes(x, axis=-1):
    """Bit-pack a binary spike tensor along ``axis``.

    Mirrors the ``np.packbits(..., axis=...)`` convention used by
    :mod:`spyx.data` (which packs along the time axis): every group of 8
    consecutive ``{0, 1}`` values along ``axis`` is packed into a single
    ``uint8``, big-endian bit order. If the axis length is not a multiple of
    8 the final byte is zero-padded on the low bits, so the original length
    must be supplied to :func:`unpack_spikes` to recover the exact tensor.

    :param x: binary tensor (values in ``{0, 1}``); cast to ``uint8``.
    :param axis: axis along which to pack (default last).
    :return: ``uint8`` tensor with ``ceil(len/8)`` entries along ``axis``.
    """
    return jnp.packbits(x.astype(jnp.uint8), axis=axis)


def unpack_spikes(packed, length, axis=-1):
    """Invert :func:`pack_spikes`, recovering ``length`` values along ``axis``.

    :param packed: ``uint8`` tensor produced by :func:`pack_spikes`.
    :param length: original (pre-pack) size of ``axis``; trims the zero
        padding introduced when ``length`` is not a multiple of 8.
    :param axis: axis along which the tensor was packed (default last).
    :return: ``uint8`` tensor of ``{0, 1}`` values, ``length`` long on ``axis``.
    """
    return jnp.unpackbits(packed, axis=axis, count=length)


@jax.custom_vjp
def packed_spike_dense(spikes, weight):
    """``spikes @ weight`` with a bit-packed backward residual.

    Forward numerics are a plain matmul over the trailing feature axis of
    ``spikes`` (shape ``(..., in)``) against ``weight`` (shape ``(in, out)``),
    yielding ``(..., out)``. The custom VJP saves ``packbits(spikes)`` -- a
    ``uint8`` tensor 8x smaller than ``spikes`` would be as bf16/fp -- instead
    of the dense activations, unpacking it in the backward pass to form
    ``dW = spikes^T @ g`` and ``dspikes = g @ weight^T``.

    Both first-order gradients equal those of the naive ``spikes @ weight``.

    Limitations: valid only when ``spikes`` is exactly binary (values in
    ``{0, 1}``) -- packing a general float tensor silently binarizes the saved
    residual, so the forward stays exact but ``dW`` becomes wrong. Only the
    first-order VJP is correct; second-order derivatives (grad-of-grad) are not,
    since the packed residual is not itself differentiated. Both are fine for
    ordinary first-order BPTT, the intended use.
    """
    return _dense(spikes, weight)


def _dense(spikes, weight):
    """Naive dense over the trailing feature axis, preserving batch dims."""
    in_features = spikes.shape[-1]
    flat = spikes.reshape(-1, in_features)
    out = flat @ weight
    return out.reshape(*spikes.shape[:-1], weight.shape[-1])


def _packed_spike_dense_fwd(spikes, weight):
    out = _dense(spikes, weight)
    # Pack along the feature axis; that is the axis contracted with weight and
    # the one we must restore for the dW = spikes^T @ g product.
    packed = pack_spikes(spikes, axis=-1)
    residual = (packed, spikes.shape, weight)
    return out, residual


def _packed_spike_dense_bwd(residual, g):
    packed, spikes_shape, weight = residual
    in_features = spikes_shape[-1]
    spikes = unpack_spikes(packed, in_features, axis=-1).astype(g.dtype)

    flat_spikes = spikes.reshape(-1, in_features)
    flat_g = g.reshape(-1, weight.shape[-1])

    dweight = flat_spikes.T @ flat_g
    dspikes = (flat_g @ weight.T).reshape(spikes_shape)
    return dspikes, dweight


packed_spike_dense.defvjp(_packed_spike_dense_fwd, _packed_spike_dense_bwd)


# --------------------------------------------------------------------------- #
# Generalization: packing *quantized* (k-bit) and *sparse* activations.
#
# Binary spikes are the k=1, no-sparsity-exploited special case. Two axes:
#   * quantization -> pack each value at `bits` bits (not 1). `pack_nbit` stores
#     `bits/8` bytes/element vs 4 for fp32 -> a 32/bits x cut. Jit-friendly, and
#     `packed_quant_dense` is the k-bit `packed_spike_dense` for BPTT residuals.
#   * sparsity     -> store only the nonzeros: a 1-bit occupancy mask + the
#     nonzero values' k-bit codes. `sparse_quant_pack` costs
#     ceil(N/8) + ceil(nnz*bits/8) bytes, which beats dense k-bit packing when the
#     density nnz/N < (bits-1)/bits (e.g. < 3/4 at 4-bit). Graded sigma-delta / high-
#     sparsity spiking sit far inside that regime.
# --------------------------------------------------------------------------- #


def pack_nbit(codes, bits, axis=-1):
    """Bit-pack an integer-code tensor (values in ``[0, 2**bits)``) along ``axis``.

    Generalises :func:`pack_spikes` (``bits=1``) to any width by packing each of the
    ``bits`` bit-planes with :func:`jax.numpy.packbits` and stacking them on a new
    leading axis. Storage is ``bits/8`` bytes/element (a ``32/bits`` x cut vs fp32).
    """
    codes = codes.astype(jnp.uint32)
    planes = [
        jnp.packbits(((codes >> b) & 1).astype(jnp.uint8), axis=axis)
        for b in range(bits)
    ]
    return jnp.stack(planes, axis=0)


def unpack_nbit(packed, bits, length, axis=-1):
    """Invert :func:`pack_nbit`, recovering integer codes ``length`` long on ``axis``."""
    out = None
    for b in range(bits):
        plane = jnp.unpackbits(packed[b], axis=axis, count=length).astype(jnp.uint32)
        shifted = plane << jnp.uint32(b)
        out = shifted if out is None else (out | shifted)
    return out


@jax.custom_vjp
def packed_quant_dense(acts, weight, bits, step):
    """``acts @ weight`` with a **k-bit-packed** backward residual.

    The k-bit generalisation of :func:`packed_spike_dense`: for activations that are
    grid-quantised (symmetric uniform grid of spacing ``step`` representable in ``bits``
    signed levels -- e.g. graded sigma-delta events, ternary, int-N), the backward saves
    the ``bits``-bit codes instead of the fp residual (a ``32/bits`` x cut), unpacking
    them to reform ``dW = acts^T @ g`` exactly. First-order VJP only; exact iff ``acts``
    lie on the grid ``{(c - 2**(bits-1)) * step}``.
    """
    return _dense(acts, weight)


def _pqd_fwd(acts, weight, bits, step):
    out = _dense(acts, weight)
    offset = 1 << (bits - 1)
    codes = jnp.clip(
        jnp.round(acts / step).astype(jnp.int32) + offset, 0, (1 << bits) - 1
    )
    packed = pack_nbit(codes.astype(jnp.uint32), bits, axis=-1)
    return out, (packed, acts.shape, weight, bits, step)


def _pqd_bwd(residual, g):
    packed, acts_shape, weight, bits, step = residual
    in_features = acts_shape[-1]
    offset = 1 << (bits - 1)
    codes = unpack_nbit(packed, bits, in_features, axis=-1)
    acts = (codes.astype(g.dtype) - offset) * step
    flat_acts = acts.reshape(-1, in_features)
    flat_g = g.reshape(-1, weight.shape[-1])
    dweight = flat_acts.T @ flat_g
    dacts = (flat_g @ weight.T).reshape(acts_shape)
    # gradients w.r.t. the non-differentiable bits / step are zero.
    return dacts, dweight, None, None


packed_quant_dense.defvjp(_pqd_fwd, _pqd_bwd)


def sparse_quant_pack(x, bits, step):
    """Pack a **sparse + quantised** tensor as ``(mask_packed, codes_packed, meta)``.

    A 1-bit occupancy mask (``packbits`` of ``x != 0``) plus the nonzero values' ``bits``-bit
    codes (:func:`pack_nbit`). Footprint ``ceil(N/8) + ceil(nnz*bits/8)`` bytes, which beats
    dense k-bit packing when density ``nnz/N < (bits-1)/bits``. Exact for grid-quantised ``x``.
    Eager (uses the dynamic nonzero count) -- for storage / event transmission, not a jit loop.
    """
    flat = x.reshape(-1)
    mask = flat != 0
    mask_packed = jnp.packbits(mask.astype(jnp.uint8))
    nz = flat[mask]
    offset = 1 << (bits - 1)
    codes = jnp.clip(
        jnp.round(nz / step).astype(jnp.int32) + offset, 0, (1 << bits) - 1
    )
    codes_packed = pack_nbit(codes.astype(jnp.uint32), bits, axis=-1)
    meta = {
        "shape": tuple(x.shape),
        "bits": bits,
        "step": float(step),
        "nnz": int(nz.size),
    }
    return mask_packed, codes_packed, meta


def sparse_quant_unpack(mask_packed, codes_packed, meta):
    """Invert :func:`sparse_quant_pack` to the dense grid-quantised tensor."""
    shape, bits, step, nnz = meta["shape"], meta["bits"], meta["step"], meta["nnz"]
    n = 1
    for d in shape:
        n *= d
    mask = jnp.unpackbits(mask_packed, count=n).astype(bool)
    codes = unpack_nbit(codes_packed, bits, nnz, axis=-1)
    offset = 1 << (bits - 1)
    vals = (codes.astype(jnp.float32) - offset) * step
    idx = jnp.nonzero(mask, size=nnz)[0]
    return jnp.zeros(n, jnp.float32).at[idx].set(vals).reshape(shape)


def packing_footprint(n_elements, bits, density):
    """Bytes to store ``n_elements`` grid-quantised activations at ``bits`` bits and the
    given nonzero ``density``, under three schemes, plus which one wins.

    Schemes: ``fp32`` (4 B/elem), ``dense_kbit`` (``N*bits/8``), and
    ``sparse`` (mask ``N/8`` + nonzero codes ``nnz*bits/8``). The sparse scheme wins below
    the ``(bits-1)/bits`` density crossover.
    """
    import math

    nnz = round(density * n_elements)
    schemes = {
        "fp32": 4 * n_elements,
        "dense_%dbit" % bits: math.ceil(n_elements * bits / 8),
        "sparse_mask+%dbit" % bits: math.ceil(n_elements / 8)
        + math.ceil(nnz * bits / 8),
    }
    best = min(schemes, key=lambda name: schemes[name])
    return {**schemes, "best": best, "crossover_density": (bits - 1) / bits}
