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
- :class:`~spyx.experimental.ResonateFire` — complex resonate-and-fire oscillatory
  neuron. *Physically defined in* ``spyx.phasor``.
- :mod:`spyx.experimental.raven` — Routing Slot Memories (``RavenRSM``) and the
  spiking sibling ``SpikingSlotMemory``, after Raven (Afzal, Bick, Xing, Cevher,
  Gu 2026). Plus ``SlotRouter`` and the ``make_recall_batch`` MQAR generator.
- :mod:`spyx.experimental.compress` — bit-packed activation storage for
  memory-efficient BPTT (``packed_spike_dense``, ``pack_spikes``,
  ``unpack_spikes``).
- :mod:`spyx.experimental.stochastic` — stochastic (Bernoulli-spiking) and
  parallelizable prototypes: ``SPSN``, ``StochasticAssociativeLIF``,
  ``StochasticAssociativeCuBaLIF`` and the ``sigmoid_bernoulli`` activations.
- :mod:`spyx.experimental.litert` — export a spiking model's single-timestep
  step ``(x_t, state) -> (out, new_state)`` to a LiteRT/TFLite flatbuffer
  (``to_litert``, ``step_signature``) so a phone app can run the temporal loop
  on-device. Conversion needs TensorFlow (``spyx[litert]``), imported lazily.

Related research studies live under ``research/new/`` in the repository.
"""

# Re-exported from the stable modules where they are physically defined; grouped
# here so the experimental surface is discoverable in one place.
from ..nn import PSU_LIF
from ..phasor import ResonateFire
from . import compress, litert, raven, stochastic
from .compress import pack_spikes, packed_spike_dense, unpack_spikes
from .litert import LiteRTStepSignature, step_signature, to_litert
from .raven import RavenRSM, SlotRouter, SpikingSlotMemory, make_recall_batch
from .stochastic import (
    SPSN,
    StochasticAssociativeCuBaLIF,
    StochasticAssociativeLIF,
    refractory_sigmoid_bernoulli,
    sigmoid_bernoulli,
)

__all__ = [
    "compress",
    "litert",
    "raven",
    "stochastic",
    "to_litert",
    "step_signature",
    "LiteRTStepSignature",
    "PSU_LIF",
    "ResonateFire",
    "RavenRSM",
    "SpikingSlotMemory",
    "SlotRouter",
    "make_recall_batch",
    "packed_spike_dense",
    "pack_spikes",
    "unpack_spikes",
    "SPSN",
    "StochasticAssociativeLIF",
    "StochasticAssociativeCuBaLIF",
    "sigmoid_bernoulli",
    "refractory_sigmoid_bernoulli",
]
