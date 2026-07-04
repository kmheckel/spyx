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
"""

import jax
import jax.numpy as jnp

__all__ = [
    "pack_spikes",
    "unpack_spikes",
    "packed_spike_dense",
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
