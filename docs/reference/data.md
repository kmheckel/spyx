# spyx.data

Grain-based data pipeline. The functional encoders (`rate_code`, `angle_code`, `latency_code`, `shift_augment`) return JIT-compiled callables; the `RateCode` / `AngleCode` / `LatencyCode` / `ShiftAugment` classes are their `grain.MapTransform` counterparts for use inside dataset pipelines. The `SHD_loader` / `NMNIST_loader` classes require the `[loaders]` extra.

::: spyx.data
