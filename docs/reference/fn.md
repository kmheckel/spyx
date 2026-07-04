# spyx.fn

Losses, metrics, and activity regularisers. All factories return JIT-compiled callables of signature `(traces, targets) -> ...`; shape checks raise `ValueError` at trace time if `traces` and `targets` disagree.

::: spyx.fn
