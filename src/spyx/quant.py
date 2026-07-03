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
* :func:`available` - cheap test for whether qwix is installed.

The functions deliberately raise :class:`ImportError` with an actionable hint
when qwix is missing, so that ``import spyx.quant`` is always safe.

.. _qwix: https://github.com/google/qwix
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

try:
    import qwix as _qwix

    _HAS_QWIX = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _qwix = None
    _HAS_QWIX = False


_INSTALL_HINT = (
    "spyx.quant requires the optional `qwix` dependency. Install it with "
    '`uv pip install "git+https://github.com/google/qwix"` (qwix has no PyPI '
    "release yet)."
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
    :param extra_op_names: additional op names to include alongside ``nnx.Linear``
        / ``nnx.Conv`` (e.g. ``("dot_general",)``).
    :return: list of qwix :class:`QuantizationRule` instances.
    """
    qwix = _require_qwix()
    rules: list[Any] = [
        qwix.QuantizationRule(
            module_path=r".*Linear.*",
            weight_qtype=weight_qtype,
            act_qtype=act_qtype,
        ),
        qwix.QuantizationRule(
            module_path=r".*Conv.*",
            weight_qtype=weight_qtype,
            act_qtype=act_qtype,
        ),
    ]
    for op in extra_op_names:
        rules.append(
            qwix.QuantizationRule(
                module_path=".*",
                op_names=(op,),
                weight_qtype=weight_qtype,
                act_qtype=act_qtype,
            )
        )
    return rules


def weights_only_rules(
    weight_qtype: str = "int8",
    module_path: str = r".*Linear.*",
) -> list[Any]:
    """Quantize only the weights, leaving activations in fp32.

    Useful for memory-bound deployment scenarios where the matmul itself runs
    on dequantized weights but the storage is compressed.
    """
    qwix = _require_qwix()
    return [
        qwix.QuantizationRule(
            module_path=module_path,
            weight_qtype=weight_qtype,
            act_qtype=None,
        )
    ]


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
            module_path=r".*Linear.*",
            weight_qtype="int2",
            act_qtype=act_qtype,
        ),
        qwix.QuantizationRule(
            module_path=r".*Conv.*",
            weight_qtype="int2",
            act_qtype=act_qtype,
        ),
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
    "bitnet_ternary_rules",
    "linear_only_rules",
    "weights_only_rules",
    "quantize",
]
