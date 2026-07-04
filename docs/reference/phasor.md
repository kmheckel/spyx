# spyx.phasor

Complex-valued phasor layers with spike-time conversion helpers. Weights are stored as paired `kernel_re` / `kernel_im` float32 parameters so a stock `optax.adam` loop converges (avoiding JAX's Wirtinger-conjugate gradient surprise on complex parameters).

`ResonateFire` (documented below) is the complex/oscillatory sibling of [`spyx.nn.PSU_LIF`](nn.md): a reset-free resonate-and-fire neuron whose complex membrane `z_t = a·z_{t-1} + x_t` is a linear recurrence, so it exposes both a stepwise `__call__` and a `parallel(x)` associative-scan path with `O(log T)` depth. See the [parallel spiking neurons](../explanation/parallel-spiking-neurons.md) explanation.

::: spyx.phasor
