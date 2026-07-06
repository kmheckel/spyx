# spyx.experimental.onnx

!!! warning "Experimental — unstable API"
    `spyx.experimental.onnx` is research-stage; its API may change without a
    deprecation cycle. Import it from `spyx.experimental`.

Export a spiking model to [ONNX](https://onnx.ai/) — either the single
feed-forward timestep `(x_t, state) -> (out, new_state)`, or the whole
`spyx.nn.run` temporal loop as a native ONNX `Loop` (pass `sequence_length=T`).
Conversion goes through a direct jaxpr → ONNX lowering
([jax2onnx](https://pypi.org/project/jax2onnx/)); `jax2onnx` and `onnx` are
imported **lazily**, so `import spyx.experimental.onnx` works without them, and
you install them yourself (`pip install jax2onnx onnx onnxruntime`). See
[How to export to ONNX](../how-to/onnx.md) for a runnable walkthrough.

::: spyx.experimental.onnx
