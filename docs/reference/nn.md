# spyx.nn

Spiking-neuron layers (IF, LIF, ALIF, CuBaLIF and recurrent variants), the stateful `Sequential` container, and the time-major `run` scan helper.

`PSU_LIF` (documented below) is a reset-free parallel spiking neuron: a pure linear leaky integrator `V_t = clip(beta)·V_{t-1} + x_t` that exposes both the standard stepwise `__call__` and a `parallel(x)` associative-scan path with `O(log T)` depth. See the [parallel spiking neurons](../explanation/parallel-spiking-neurons.md) explanation for the sequential-vs-parallel trade-off and the [benchmarking how-to](../how-to/benchmarking.md) to measure it.

::: spyx.nn
