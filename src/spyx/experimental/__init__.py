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
- :class:`~spyx.experimental.TTTFastWeight` — test-time-training / fast-weight
  sequence layer whose hidden *state is a weight matrix* updated online per token
  by a hebb (``rule="hebb"``, scalar transition → ``.parallel`` associative scan,
  a matrix-valued ``PSU_LIF``) or error-correcting delta rule (``rule="delta"``,
  matrix transition → sequential ``spyx.nn.run``; chunked-DeltaNet for parallel).
- :mod:`spyx.experimental.local_learning` — local online three-factor
  (e-prop / OTTT-style) plasticity: :class:`ThreeFactorLIF`, a plastic-synapse
  LIF that maintains a per-synapse eligibility trace and applies a
  modulator-gated ``ΔW = eta * <mod * trace>`` *during* the forward-through-time
  pass (no BPTT), with meta-learnable coefficients an outer SGD/ES loop can tune.
  It is the neuromorphic, spiking sibling of the ``TTTFastWeight`` delta rule.
- :mod:`spyx.experimental.feedback_alignment` — backprop-free training via fixed
  *random* feedback (weight-transport-free): :class:`FALinear` / ``fa_dense``
  (layer-local Feedback Alignment, Lillicrap 2016) and ``dfa_inject`` /
  ``dfa_gradient`` (Direct Feedback Alignment, Nøkland 2016), composing with
  surrogate spiking neurons and ``spyx.nn.run`` via ``jax.custom_vjp``.
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
    feedback_alignment,
    hybrid,
    local_learning,
    matfree,
    onnx,
    parallel_reset,
    raven,
    rf_ssm,
    sigma_delta,
    stochastic,
    ttt,
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
from .feedback_alignment import (
    FALinear,
    Feedback,
    dfa_gradient,
    dfa_inject,
    fa_dense,
)
from .hybrid import (
    es_gradient,
    hybrid_diagnostics,
    hybrid_gradient,
    make_hybrid_train_step,
    make_sges_hybrid_train_step,
    sges_gradient,
)
from .local_learning import ThreeFactorLIF, surrogate_deriv
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
from .ttt import TTTFastWeight

__all__ = [
    "compress",
    "evolve",
    "feedback_alignment",
    "hybrid",
    "local_learning",
    "matfree",
    "onnx",
    "parallel_reset",
    "raven",
    "rf_ssm",
    "sigma_delta",
    "stochastic",
    "ttt",
    "zoo",
    "ParallelResetLIF",
    "RFSSM",
    "ResonateFireSSM",
    "SigmaDelta",
    "graded_quant",
    "ThreeFactorLIF",
    "surrogate_deriv",
    "TTTFastWeight",
    "Feedback",
    "FALinear",
    "fa_dense",
    "dfa_inject",
    "dfa_gradient",
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
