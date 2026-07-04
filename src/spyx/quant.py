"""Quantization helpers for Spyx SNNs, built on top of Google's `qwix`_.

Qwix is a JAX-native quantization library that integrates with Flax NNX modules
via :func:`qwix.quantize_model`. ``spyx.quant`` provides:

* :func:`quantize` - an SNN-aware wrapper around ``qwix.quantize_model`` that
  picks sensible defaults for spiking networks: it quantizes the dense
  ``nnx.Linear`` and ``nnx.Conv`` layers but leaves the spiking dynamics
  (``LIF``/``CuBaLIF``/``ALIF``/``IF``) and the leaky readout (``LI``) at full
  precision, since their state recurrences are sensitive to integer rounding.
* :func:`linear_only_rules` and :func:`weights_only_rules` - shorthand
  :class:`qwix.QuantizationRule` lists for the common cases.
* :func:`spiking_feedforward_rules` - a weight-only recipe for the
  spike->Linear feedforward path that is **lossless on binary activations**.
* :func:`binary_activation_error` - a qwix-free check that proves spikes lie
  exactly on the integer grid.
* :func:`available` - cheap test for whether qwix is installed.

The functions deliberately raise :class:`ImportError` with an actionable hint
when qwix is missing, so that ``import spyx.quant`` is always safe.

Binary-activation losslessness
------------------------------

An SNN's activations are *spikes*: each is exactly ``0`` or ``1`` (see
:mod:`spyx.data`, which bit-packs them). A binary vector is already a 1-bit
signal, so it lands **exactly** on the symmetric-integer grid used by
quantization: with absmax calibration the scale is ``1/max_int`` and
``round(x / scale) * scale`` maps ``0->0`` and ``1->1`` with zero error, for
int8, int4 or even int2 (:func:`binary_activation_error` returns ``0.0``).

The practical consequence is that a **weight-only** scheme on the feedforward
``spike -> Linear`` path introduces *no activation-side error at all*. The
matmul ``dequant(quant(W)) @ spikes`` is simply a masked sum of weight columns;
because the activation carries no fractional part there is nothing for an
activation quantizer to round away. Weight-only int8 is therefore not a
compromise for spiking nets - the activation is already maximally compressed at
1 bit - and :func:`spiking_feedforward_rules` builds exactly this recipe while
leaving ``einsum`` (the recurrent / SSM state transition) in fp32.

This heterogeneous precision - low-bit feedforward weights, high-precision
recurrent path - matches the empirical finding of **Q-S5** (Abreu, Pedersen,
Heckel & Pierro, *Q-S5: Towards Quantized State Space Models*,
arXiv:2406.09477, 2024): a fully quantized S5 loses <1% on sMNIST and most of
LRA, but the *recurrent* weights need >=8 bits while other components compress
much further, giving ~6x memory savings at heterogeneous precision. Keeping the
einsum-based recurrence in fp32 (as every rule builder here does) is the
conservative end of that trade-off.

.. _qwix: https://github.com/google/qwix
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import jax.numpy as jnp

try:
    import qwix as _qwix

    _HAS_QWIX = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _qwix = None  # ty: ignore[invalid-assignment]  # module-or-None sentinel
    _HAS_QWIX = False


# qwix has no PyPI release. Spyx pins it via tool.uv.sources, but uv sources
# aren't transitive, so downstream `spyx[quant]` doesn't resolve it in either uv
# or pip. The portable fix (works in both) is installing qwix from GitHub
# directly; inside the Spyx repo `uv sync --extra quant` handles it.
_INSTALL_HINT = (
    "spyx.quant needs the optional `qwix` dependency, which has no PyPI "
    "release. Install it from GitHub (works with uv and pip):\n"
    '    uv add "qwix @ git+https://github.com/google/qwix"\n'
    '    # or: pip install "qwix @ git+https://github.com/google/qwix"\n'
    "Inside the Spyx repo, `uv sync --extra quant` resolves it automatically."
)


def available() -> bool:
    """Return True if ``qwix`` is importable in the current environment."""
    return _HAS_QWIX


def _require_qwix() -> Any:
    if not _HAS_QWIX:
        raise ImportError(_INSTALL_HINT)
    return _qwix


def linear_only_rules(
    weight_qtype: str | None = "int8",
    act_qtype: str | None = "int8",
    extra_op_names: Sequence[str] = (),
) -> list[Any]:
    """Quantize only the dense / conv layers of an SNN, leaving spiking dynamics alone.

    Spiking neuron state updates (``V = beta * V + x``) involve subtle
    cancellations that integer quantization rounds away, often collapsing the
    whole network to silence. The recommended default is to quantize the linear
    transforms and let the neurons run in fp32.

    :param weight_qtype: dtype string accepted by :class:`qwix.QuantizationRule`
        (e.g. ``"int8"``, ``"int4"``, ``"fp8"``). ``None`` disables weight quant.
    :param act_qtype: same options for activations. ``None`` disables.
    :param extra_op_names: additional op names to quantize alongside the default
        matmul / conv ops (e.g. a custom primitive).
    :return: a single-element list with a qwix :class:`QuantizationRule`.

    .. note::
        qwix matches a rule's ``module_path`` regex with :func:`re.fullmatch`
        against the ``/``-joined **NNX attribute path** (e.g. ``core/layers/0``),
        which never contains the module's class name. Targeting layers with
        ``module_path=r".*Linear.*"`` therefore matches *nothing* and silently
        quantizes the model to a no-op. Instead we match every module and select
        the dense/conv work by the underlying **op**: ``dot_general``
        (``nnx.Linear``) and ``conv_general_dilated`` (``nnx.Conv``). Spiking
        neuron updates are elementwise and use neither, so they stay in fp32
        without needing a path filter. ``einsum`` is intentionally excluded so
        that SSM state transitions (``spyx.ssm``) keep full precision; pass it
        via ``extra_op_names`` to quantize einsum-based layers too.
    """
    qwix = _require_qwix()
    op_names = ("dot_general", "conv_general_dilated", *extra_op_names)
    return [
        qwix.QuantizationRule(
            module_path=".*",
            op_names=op_names,
            weight_qtype=weight_qtype,
            act_qtype=act_qtype,
        )
    ]


def weights_only_rules(
    weight_qtype: str = "int8",
    module_path: str = ".*",
    op_names: Sequence[str] = ("dot_general", "conv_general_dilated"),
) -> list[Any]:
    """Quantize only the weights, leaving activations in fp32.

    Useful for memory-bound deployment scenarios where the matmul itself runs
    on dequantized weights but the storage is compressed.

    Selects work by op name rather than module class — see
    :func:`linear_only_rules` for why (qwix ``module_path`` matches the NNX
    attribute path, not the class name).
    """
    qwix = _require_qwix()
    return [
        qwix.QuantizationRule(
            module_path=module_path,
            op_names=tuple(op_names),
            weight_qtype=weight_qtype,
            act_qtype=None,
        )
    ]


def spiking_feedforward_rules(
    weight_qtype: str = "int8",
    *,
    module_path: str = ".*",
    extra_op_names: Sequence[str] = (),
) -> list[Any]:
    """Weight-only quantization of the ``spike -> Linear`` feedforward path.

    This is the recommended recipe for spiking networks. It quantizes the
    *weights* of the dense / conv feedforward transforms while leaving both the
    activations and the recurrent ``einsum`` state transitions in full
    precision. On an SNN the feedforward activation is a spike train - values in
    ``{0, 1}`` - so a weight-only scheme is **lossless on the activation side**:
    the spike already lies exactly on the integer grid, so there is nothing for
    an activation quantizer to round away (see the module docstring and
    :func:`binary_activation_error`). The quantized forward pass therefore
    equals ``dequant(quant(W)) @ spikes`` *exactly* - the only error is the
    (unavoidable) weight rounding, never the activation.

    Because the activation carries no fractional information, the feedforward
    weights can be pushed below int8 without incurring activation error; pass
    ``weight_qtype="int4"`` (or ``"int2"``) to compress the feedforward path
    further, mirroring the Q-S5 finding that non-recurrent components tolerate
    aggressive quantization (arXiv:2406.09477).

    ``einsum`` is deliberately **excluded** from ``op_names`` so that recurrent
    / SSM state transitions (:mod:`spyx.ssm`, :class:`spyx.nn.PSU_LIF`,
    :class:`spyx.phasor.ResonateFire`) keep fp32 precision, which Q-S5 shows the
    recurrent path requires.

    :param weight_qtype: dtype string for the feedforward weights (``"int8"``
        default; ``"int4"`` / ``"int2"`` for more compression). Activations are
        never quantized here.
    :param module_path: qwix ``module_path`` regex (matched against the NNX
        attribute path, *not* the class name - see :func:`linear_only_rules`).
    :param extra_op_names: additional feedforward op names to quantize. Do
        **not** add ``"einsum"`` unless you intend to quantize the recurrent
        path (which Q-S5 advises against).
    :return: a single-element list with a weight-only qwix
        :class:`QuantizationRule`.
    """
    qwix = _require_qwix()
    op_names = ("dot_general", "conv_general_dilated", *extra_op_names)
    return [
        qwix.QuantizationRule(
            module_path=module_path,
            op_names=op_names,
            weight_qtype=weight_qtype,
            act_qtype=None,
        )
    ]


def _qtype_bits(qtype: str) -> int:
    """Return the bit-width of an integer qtype string like ``"int8"``.

    :raises ValueError: if ``qtype`` is not an ``"int<N>"`` string. The binary
        losslessness argument is specific to symmetric *integer* grids.
    """
    if not qtype.startswith("int"):
        raise ValueError(
            f"binary_activation_error is defined for integer qtypes "
            f"(e.g. 'int8', 'int4'); got {qtype!r}."
        )
    try:
        bits = int(qtype[3:])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Malformed integer qtype {qtype!r}.") from exc
    if bits < 2:
        raise ValueError(
            f"Need at least 2 bits for signed quantization; got {qtype!r}."
        )
    return bits


def binary_activation_error(spikes: Any, *, weight_qtype: str = "int8") -> float:
    """Max absolute error of round-tripping ``spikes`` through symmetric int quant.

    This is the qwix-free proof behind :func:`spiking_feedforward_rules`: it
    applies textbook symmetric absmax integer quantization
    (``scale = amax / max_int``; ``round(x / scale) * scale``) to ``spikes`` and
    returns ``max|x - dequant(quant(x))|``. For a genuine spike train - values
    in ``{0, 1}`` - this is **exactly** ``0.0`` at int8, int4 and int2, because
    a binary signal already lies on the integer grid. A non-zero result means
    the input is not truly binary (e.g. a graded / surrogate activation), which
    is the only way the feedforward activation could lose precision.

    :param spikes: array-like of activations, expected in ``{0, 1}``.
    :param weight_qtype: integer qtype whose grid to test against (default
        ``"int8"``).
    :return: the maximum absolute round-trip error as a Python ``float``.
    """
    bits = _qtype_bits(weight_qtype)
    x = jnp.asarray(spikes, dtype=jnp.float32)
    max_int = float(2 ** (bits - 1) - 1)
    amax = jnp.max(jnp.abs(x))
    scale = jnp.where(amax > 0, amax / max_int, 1.0)
    dequant = jnp.clip(jnp.round(x / scale), -max_int, max_int) * scale
    return float(jnp.max(jnp.abs(x - dequant)))


def bitnet_ternary_rules(act_qtype: str = "int8") -> list[Any]:
    """BitNet b1.58-style "ternary" weight quantization for dense / conv layers.

    The published BitNet recipe stores each weight as one of ``{-1, 0, +1}``,
    a 1.58-bit code. Qwix doesn't expose a true ternary qtype today, so this
    helper falls back to ``"int2"`` (4-level symmetric quantization with values
    in ``{-2, -1, 0, 1}``). The two-bit fallback gets you the same memory
    profile and storage class as ternary; the extra ``-2`` level slightly
    inflates representational range but keeps the symmetric absmax calibration
    well-behaved.

    For exact ternary semantics, follow up with a custom
    :class:`qwix.QuantizationRule` using a hand-rolled calibration method;
    spiking SSMs in particular may benefit from the strict 1.58-bit recipe.

    :param act_qtype: activation quantization dtype. ``"int8"`` matches the
        official BitNet recipe ("BitNet b1.58 + 8-bit activations"). Pass
        ``None`` for a pure weights-only ternary mode.
    :return: list of qwix :class:`QuantizationRule` instances ready to feed to
        :func:`quantize`.
    """
    qwix = _require_qwix()
    return [
        qwix.QuantizationRule(
            module_path=".*",
            op_names=("dot_general", "conv_general_dilated"),
            weight_qtype="int2",
            act_qtype=act_qtype,
        )
    ]


def quantize(
    model: Any,
    *example_inputs: Any,
    rules: Iterable[Any] | None = None,
    mode: str = "qat",
    methods: Sequence[str] = ("__call__",),
    **example_kwargs: Any,
) -> Any:
    """Apply qwix quantization to a Spyx / Flax NNX model.

    By default, uses :func:`linear_only_rules` for int8 quantization-aware
    training (QAT). Pass ``mode="ptq"`` for post-training quantization.

    Example::

        model = spyx.nn.Sequential(
            nnx.Linear(128, 64, use_bias=False, rngs=rngs),
            spyx.nn.LIF((64,), rngs=rngs),
            nnx.Linear(64, 20, use_bias=False, rngs=rngs),
            spyx.nn.LI((20,), rngs=rngs),
        )
        sample_x = jnp.zeros((batch, 128))
        sample_state = model.initial_state(batch)
        qmodel = spyx.quant.quantize(model, sample_x, sample_state)
        # then train qmodel with the usual nnx.Optimizer + nnx.value_and_grad loop

    :param model: an ``nnx.Module``; usually a ``spyx.nn.Sequential`` or a
        custom module wrapping one.
    :param example_inputs: positional args matching the signature of
        ``model.__call__`` (qwix needs them to trace and discover the modules).
    :param rules: list of :class:`qwix.QuantizationRule`. Defaults to
        :func:`linear_only_rules` (int8 weights + activations on Linear / Conv).
    :param mode: ``"qat"`` for quantization-aware training (default) or
        ``"ptq"`` for post-training quantization.
    :param methods: model methods to quantize. Defaults to ``("__call__",)``.
    :param example_kwargs: keyword args matching ``model.__call__``.
    :return: a new quantized model; can be fed straight into ``nnx.Optimizer``.
    """
    qwix = _require_qwix()
    if rules is None:
        rules = linear_only_rules()
    rules = list(rules)

    if mode == "qat":
        provider = qwix.QtProvider(rules=rules)
    elif mode == "ptq":
        provider = qwix.PtqProvider(rules=rules)
    else:
        raise ValueError(
            f"Unknown quantization mode {mode!r}; expected 'qat' or 'ptq'."
        )

    return qwix.quantize_model(
        model,
        provider,
        *example_inputs,
        methods=tuple(methods),
        **example_kwargs,
    )


__all__ = [
    "available",
    "binary_activation_error",
    "bitnet_ternary_rules",
    "linear_only_rules",
    "quantize",
    "spiking_feedforward_rules",
    "weights_only_rules",
]
