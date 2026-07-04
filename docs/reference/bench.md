# spyx.bench

Benchmarking harness for Spyx neurons and models. `benchmark` measures median forward and forward+backward latency, throughput, peak memory, XLA-cost-model FLOPs/MFU, and the `spike_rate` energy proxy for any module following the `spyx.nn` stateful contract; `compare` sweeps a set of modules across sequence lengths and `format_table` renders the results. See the [benchmarking how-to](../how-to/benchmarking.md) for the timing methodology and runnable examples.

::: spyx.bench
