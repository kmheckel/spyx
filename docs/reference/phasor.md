# spyx.phasor

Complex-valued phasor layers with spike-time conversion helpers. Weights are stored as paired `kernel_re` / `kernel_im` float32 parameters so a stock `optax.adam` loop converges (avoiding JAX's Wirtinger-conjugate gradient surprise on complex parameters).

::: spyx.phasor
