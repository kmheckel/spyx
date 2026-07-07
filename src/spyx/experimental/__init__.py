"""Experimental spyx modules — **unstable API, may change between releases**.

This namespace collects research-stage building blocks that are not part of the
stable spyx surface. They are tested and usable, but their APIs (and in some cases
their numerical behaviour) may change without a deprecation cycle as the research
matures. Anything you rely on for production should come from the stable top-level
modules (``spyx.nn``, ``spyx.ssm``, ``spyx.phasor``, ``spyx.nir``, ``spyx.bench``,
``spyx.quant``, ``spyx.data``, ``spyx.optimize``).

Contents:

- :class:`~spyx.experimental.PSU_LIF` — reset-free parallel LIF (parallel
  associative-scan spiking neuron). *Physically defined in* ``spyx.nn`` *and
  surfaced here as its supported, experimental entry point.*
- :class:`~spyx.experimental.AssociativeLIF` — thin alias of ``PSU_LIF`` at
  exact numeric parity with snnTorch v1.0.0 ``snntorch.StateLeaky`` (after a
  ``beta`` reparameterisation). It does **not** replicate snnTorch's
  ``AssociativeLeaky``, which is a different matrix-state associative-memory SSM.
- :class:`~spyx.experimental.ResonateFire` — complex resonate-and-fire oscillatory
  neuron. *Physically defined in* ``spyx.phasor``.
- :class:`~spyx.experimental.ParallelResetLIF` — reset-*preserving* parallel LIF
  (FPT fixed-point scan: keeps the exact hard reset while parallelising the time loop).
- :class:`~spyx.experimental.RFSSM` — resonate-and-fire as a scaled spiking SSM
  (S5/HiPPO pole init + PRF decoupled reset; alias ``ResonateFireSSM``).
- :class:`~spyx.experimental.SigmaDelta` — graded sigma-delta neuron that transmits
  only the quantized *change* in its membrane, so temporally-redundant input is nearly
  free (``graded_quant`` is the straight-through grid quantiser).
- :mod:`spyx.experimental.raven` — Routing Slot Memories (``RavenRSM``) and the
  spiking sibling ``SpikingSlotMemory``, after Raven (Afzal, Bick, Xing, Cevher,
  Gu 2026). Plus ``SlotRouter`` and the ``make_recall_batch`` MQAR generator.
- :mod:`spyx.experimental.compress` — bit-packed activation storage for
  memory-efficient BPTT. Binary spikes (``packed_spike_dense``, ``pack_spikes``,
  ``unpack_spikes``) plus the *quantized/sparse* generalisation: k-bit bit-plane
  packing (``pack_nbit``/``unpack_nbit``), a k-bit BPTT dense
  (``packed_quant_dense``), sparse occupancy-mask + code packing
  (``sparse_quant_pack``/``sparse_quant_unpack``), and a ``packing_footprint``
  byte-count / crossover helper.
- :mod:`spyx.experimental.onnx` — export a spiking model to ONNX: the
  per-timestep step ``(x_t, state) -> (out, new_state)`` (flat tensor I/O), or,
  when a sequence length is given, the whole ``spyx.nn.run`` temporal loop as a
  native ONNX ``Scan``/``Loop``. Conversion deps (tensorflow/tf2onnx/onnx) are
  imported lazily.
- :mod:`spyx.experimental.stochastic` — stochastic (Bernoulli-spiking) and
  parallelizable prototypes: ``SPSN``, ``StochasticAssociativeLIF``,
  ``StochasticAssociativeCuBaLIF`` and the ``sigmoid_bernoulli`` activations.
- :mod:`spyx.experimental.hybrid` — the 0+1 hybrid trainer: a surrogate
  gradient corrected by an antithetic-NES estimate of the true (hard-spike)
  loss, projected orthogonal to the surrogate (``hybrid_gradient``,
  ``make_hybrid_train_step``, ``es_gradient``, ``hybrid_diagnostics``).
- :mod:`spyx.experimental.zoo` — runnable reference recipes keyed by
  application (control / classification / language) and tagged by training
  method × architecture (``REGISTRY``, ``list_recipes``, ``get``).

Related research studies live under ``research/new/`` in the repository.
"""

# Re-exported from the stable modules where they are physically defined; grouped
# here so the experimental surface is discoverable in one place.
from ..nn import PSU_LIF, AssociativeLIF
from ..phasor import ResonateFire
from . import (
    compress,
    evolve,
    hybrid,
    matfree,
    onnx,
    parallel_reset,
    raven,
    rf_ssm,
    sigma_delta,
    stochastic,
    zoo,
)
from .compress import (
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
from .hybrid import (
    es_gradient,
    hybrid_diagnostics,
    hybrid_gradient,
    make_hybrid_train_step,
    make_sges_hybrid_train_step,
    sges_gradient,
)
from .parallel_reset import ParallelResetLIF
from .raven import RavenRSM, SlotRouter, SpikingSlotMemory, make_recall_batch
from .rf_ssm import RFSSM, ResonateFireSSM
from .sigma_delta import SigmaDelta, graded_quant
from .stochastic import (
    SPSN,
    StochasticAssociativeCuBaLIF,
    StochasticAssociativeLIF,
    refractory_sigmoid_bernoulli,
    sigmoid_bernoulli,
)

__all__ = [
    "compress",
    "evolve",
    "hybrid",
    "matfree",
    "onnx",
    "parallel_reset",
    "raven",
    "rf_ssm",
    "sigma_delta",
    "stochastic",
    "zoo",
    "ParallelResetLIF",
    "RFSSM",
    "ResonateFireSSM",
    "SigmaDelta",
    "graded_quant",
    "PSU_LIF",
    "AssociativeLIF",
    "ResonateFire",
    "RavenRSM",
    "SpikingSlotMemory",
    "SlotRouter",
    "make_recall_batch",
    "packed_spike_dense",
    "pack_spikes",
    "unpack_spikes",
    "pack_nbit",
    "unpack_nbit",
    "packed_quant_dense",
    "sparse_quant_pack",
    "sparse_quant_unpack",
    "packing_footprint",
    "SPSN",
    "StochasticAssociativeLIF",
    "StochasticAssociativeCuBaLIF",
    "sigmoid_bernoulli",
    "refractory_sigmoid_bernoulli",
    "hybrid_gradient",
    "make_hybrid_train_step",
    "es_gradient",
    "sges_gradient",
    "make_sges_hybrid_train_step",
    "hybrid_diagnostics",
]
