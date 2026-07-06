# spyx.optimize

High-level training loops for Spyx models. Three entry points, in increasing order of "how much is compiled":

- **`fit(...)`** — a Python epoch loop driving a per-step `@nnx.jit`. The common case; returns a `History` of per-epoch metrics. Use `make_train_step` / `make_eval_step` to roll your own loop with the same JIT'd pieces.
- **`compile_fit(...)`** — stages the whole dataset on-device and `jax.lax.scan`s the loop over epochs × batches under a *single* `jax.jit`, so a full run compiles to one XLA kernel with no per-step Python or re-tracing (the throughput pattern the Spyx paper relies on).
- **`Solver` protocol** — `compile_fit` is optimiser-agnostic: pass an Optax `GradientTransformation` (auto-wrapped by `backprop(...)` for the surrogate-gradient path) *or* a custom `Solver` implementing `init` / `step`, which is how the evolutionary trainers (ES / CMA-ES, see [`spyx.experimental.hybrid`](experimental.md)) drop into the same whole-loop-JIT machinery.

::: spyx.optimize
