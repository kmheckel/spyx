# How to load event data

To feed neuromorphic datasets into a Spyx model, use the [Grain](https://github.com/google/grain)-based pipeline in [`spyx.data`](../reference/data.md). The ready-made loaders need the `[loaders]` extra (they wrap [tonic](https://tonic.readthedocs.io/) datasets):

```bash
uv add "spyx[loaders]"
```

Without the extra, constructing a loader raises an `ImportError`; the functional encoders and Grain transforms below work with the core install.

## Load SHD or N-MNIST

```python
import spyx

dl = spyx.data.SHD_loader(batch_size=256, sample_T=128, channels=128)
# or:
dl = spyx.data.NMNIST_loader(batch_size=32, sample_T=40)
```

The first call downloads the dataset to `./data` (override with `download_dir=`). Iterate one epoch at a time; each batch is a `State` namedtuple with `.obs` and `.labels`:

```python
for batch in dl.train_epoch():
    ...
for batch in dl.test_epoch():
    ...
```

### Unpack the spikes

Observations arrive **bit-packed along the time axis** (`uint8`, shape `(B, ceil(T/8), C)`) to save memory. Recover the dense `(B, T, C)` spike tensor with `jnp.unpackbits`:

```python
import jax.numpy as jnp

SAMPLE_T = 128

def unpack(batch_obs):
    obs = jnp.asarray(batch_obs)
    return jnp.unpackbits(obs, axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)
```

### Tune throughput

- `worker_count=` controls Grain multiprocessing. The default (half your cores, capped at 4) cuts first-batch latency from ~30 s to a few seconds at `batch_size=256`; pass `0` to disable multiprocessing when debugging.
- For maximum throughput, `SHD_loader.prestage(split)` bulk-loads an entire split into a single on-device array (`(n_batches, batch_size, T_packed, C)` packed `uint8` plus labels), matching the "whole dataset in vRAM" pattern from the Spyx paper:

```python
obs_NBTC, labels_NB = dl.prestage("train")
```

!!! note
    `spyx.data` monkey-patches tonic's SHD reader at import time to drop non-finite spike timestamps that otherwise corrupt a handful of samples (pending an upstream tonic fix).

## Encode non-spiking data

To turn continuous-valued data into spike trains, use the encoder factories. Each returns a JIT-compiled function:

```python
import jax

# Rate coding: spike probability proportional to input value (in [0, 1]).
encode = spyx.data.rate_code(num_steps=64, max_r=0.75)
spikes = encode(x, jax.random.PRNGKey(0))

# Latency (time-to-first-spike) coding: larger values fire earlier;
# one spike per neuron, inputs below `threshold` stay silent.
encode = spyx.data.latency_code(num_steps=64, threshold=0.01)
spikes = encode(x)                       # (num_steps, ..., C) uint8

# Angle coding: one-hot discretisation of a continuous value into N channels.
encode = spyx.data.angle_code(neuron_count=32, min_val=0.0, max_val=1.0)
onehot = encode(x)
```

See the [SNN primer](../explanation/snn-primer.md#rate-coding-vs-latency-coding) for when to prefer rate vs. latency coding.

## Augment spike trains

To randomly shift spikes along an axis (a cheap, effective augmentation for SHD-style data):

```python
augment = spyx.data.shift_augment(max_shift=10, axes=(-1,))   # shift channels
batch = augment(batch, jax.random.PRNGKey(step))
```

## Build a custom Grain pipeline

Each functional encoder has a `grain.MapTransform` counterpart — `RateCode`, `LatencyCode`, `AngleCode`, `ShiftAugment` — for use inside dataset pipelines. They read and write dict records (default key `"obs"`) and bit-pack their output like the built-in loaders:

```python
import grain.python as grain

transforms = [
    spyx.data.ShiftAugment(max_shift=10, axes=(-1,)),
    spyx.data.LatencyCode(sample_T=64),
]
```

To wrap your own dataset, mirror what `SHD_loader` does internally: expose it as a `len + getitem` source of `{"obs": ..., "labels": ...}` dicts and hand it to `spyx.data.GrainLoader(dataset, batch_size, shuffle, seed=0, worker_count=None)`, which yields the same `State(obs, labels)` batches as the built-in loaders.
