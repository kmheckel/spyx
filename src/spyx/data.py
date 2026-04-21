from collections import namedtuple

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

try:
    from tonic import datasets, transforms
    tonic_installed = True
except ImportError:
    tonic_installed = False


def _patch_tonic_hsd() -> None:
    """Monkey-patch tonic's HSD loader to cope with non-finite spike times.

    SHD and SSC h5 files occasionally contain NaN / inf entries in
    ``spikes/times`` (or very large int-overflowing values). Tonic's
    stock ``HSD.__getitem__`` multiplies the raw (float32) timestamps
    by ``1e6`` and feeds them into a structured array with ``t`` typed
    as ``int``, which triggers two cascading NumPy warnings:

    * ``overflow encountered in cast`` (at the float32 * 1e6 step).
    * ``invalid value encountered in cast`` (NaN -> int coerces to 0
      or INT_MIN, garbling the event stream).

    The resulting garbage timestamps propagate through ``Downsample``
    and ``ToFrame``; some samples end up allocating giant sparse frames
    or producing all-zero spike tensors, and at BATCH=256 the cumulative
    effect is a data loader that appears to hang for ~minutes before
    yielding the first batch.

    This patch replaces ``HSD.__getitem__`` with a version that:

    1. Reads ``spikes/times`` explicitly as ``float64`` before scaling,
       so the ``* 1e6`` never overflows.
    2. Drops non-finite timestamps (and the matching units) before
       they reach the ``int`` cast.

    The fix is the one we plan to send upstream as a tonic PR; this
    monkey-patch exists so Spyx users get a working SHD loader today.
    The patch is a no-op on tonic versions that ship a compatible
    ``__getitem__`` (we detect the upstream fix by looking for
    ``_spyx_patched`` on the method).
    """
    import os

    import h5py
    from tonic.datasets.hsd import HSD
    from tonic.io import make_structured_array

    # Avoid double-patching / patching over an upstream fix we don't
    # recognise.
    if getattr(HSD.__getitem__, "_spyx_patched", False):
        return

    def __getitem__(self, index):
        file = h5py.File(
            os.path.join(self.location_on_system, self.data_filename), "r"
        )
        # Explicit float64 cast: tonic stores spikes/times as float32,
        # and (float32 * 1e6) overflows for a handful of malformed
        # samples even though the numerical value fits float64.
        times_seconds = np.asarray(
            file["spikes/times"][index], dtype=np.float64
        )
        units = np.asarray(file["spikes/units"][index])

        # A handful of SHD samples carry NaN / inf entries that would
        # otherwise cast to 0 / INT_MIN and blow up ToFrame.
        finite = np.isfinite(times_seconds)
        if not np.all(finite):
            times_seconds = times_seconds[finite]
            units = units[finite]

        events = make_structured_array(
            times_seconds * 1e6,
            units,
            1,
            dtype=self.dtype,
        )
        target = file["labels"][index].astype(int)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    __getitem__._spyx_patched = True
    HSD.__getitem__ = __getitem__


if tonic_installed:
    _patch_tonic_hsd()


State = namedtuple("State", "obs labels")

# --- Functional Transforms ---

def shift_augment(max_shift=10, axes=(-1,)):
    """Shift data augmentation tool. Rolls data along specified axes randomly up to a certain amount.

        
    :max_shift: maximum to which values can be shifted.
    :axes: the data axis or axes along which the input will be randomly shifted.
    """

    def _shift(data, rng):
        shift = jax.random.randint(rng, (len(axes),), -max_shift, max_shift)
        return jnp.roll(data, shift, axes)
    
    return jax.jit(_shift)


def shuffler(dataset, batch_size):
    """
    Higher-order-function which builds a shuffle function for a dataset.

    :dataset: jnp.array [# samples, time, channels...]
    :batch_size: desired batch size.
    """
    x, y = dataset
    cutoff = (y.shape[0] // batch_size) * batch_size
    data_shape = (-1, batch_size) + x.shape[1:]

    def _shuffle(dataset, shuffle_rng):
        """
        Given a dataset as a single tensor, shuffle its batches.

        :dataset: tuple of jnp.arrays with shape [# batches, batch size, time, ...] and [# batches, batchsize]
        :shuffle_rng: JAX.random.PRNGKey
        """
        x, y = dataset

        indices = jax.random.permutation(shuffle_rng, y.shape[0])[:cutoff]
        obs, labels = x[indices], y[indices]

        obs = jnp.reshape(obs, data_shape)
        labels = jnp.reshape(labels, (-1, batch_size)) # should make batch size a global

        return (obs, labels)

    return jax.jit(_shuffle)




def rate_code(num_steps, max_r=0.75):
    """
    Unrolls input data along axis 1 and converts to rate encoded spikes; the probability of spiking is based on the input value multiplied by a max rate, with each time step being a sample drawn from a Bernoulli distribution.
    Currently Assumes input values have been rescaled to between 0 and 1.
    """

    def _call(data, key):
        data = jnp.array(data, dtype=jnp.float16)
        unrolled_data = jnp.repeat(data, num_steps, axis=1)
        return jax.random.bernoulli(key, unrolled_data*max_r).astype(jnp.uint8)
    
    return jax.jit(_call)



def angle_code(neuron_count, min_val, max_val):
    """
    Higher-order-function which returns an angle encoding function; given a continuous value, an angle converter generates a one-hot vector corresponding to where the value falls between a specified minimum and maximum.
    To achieve non-linear descritization, apply a function to the continuous value before feeding it into the encoder.

    :neuron_count: The number of output channels for the angle encoder
    :min_val: A lower bound on the continuous input channel
    :max_val: An upper bound on the continuous input channel.
    """
    neurons = jnp.linspace(min_val, max_val, neuron_count)

    def _call(obs):
        digital = jnp.digitize(obs, neurons) - 1
        digital = jnp.clip(digital, 0, neuron_count - 1)
        return jax.nn.one_hot(digital, neuron_count)

    return jax.jit(_call)


def latency_code(num_steps, threshold=0.01):
    """Time-to-first-spike (latency) encoding.

    Large input values fire earlier in the cycle, small values fire later.
    Concretely, an input in ``[0, 1]`` is mapped to a spike time
    ``t = round((1 - x) * (num_steps - 1))`` and a one-hot spike train is
    emitted along a new leading time axis. Inputs below ``threshold`` never
    fire, producing all-zero rows.

    The encoding preserves total information in a single spike per neuron,
    which is both far sparser than rate coding and matches the time-to-
    first-spike training scheme used in the neuromorphic hardware literature.

    :param num_steps: length of the emitted spike train (time axis).
    :param threshold: values ``<= threshold`` are considered silent.
    :return: JIT-compiled function mapping ``data: [..., C]`` (values in
        ``[0, 1]``) to ``spikes: [num_steps, ..., C]`` of dtype ``uint8``.
    """

    def _call(data):
        x = jnp.asarray(data, dtype=jnp.float32)
        x = jnp.clip(x, 0.0, 1.0)
        spike_idx = jnp.round((1.0 - x) * (num_steps - 1)).astype(jnp.int32)
        # One-hot along a new time axis at the end, then move it to the front.
        one_hot = jax.nn.one_hot(spike_idx, num_steps, dtype=jnp.uint8)
        moved = jnp.moveaxis(one_hot, -1, 0)  # [T, ..., C]
        # Zero out silent units.
        silent_mask = (x <= threshold).astype(jnp.uint8)
        return moved * (1 - silent_mask)

    return jax.jit(_call)

# --- Grain-compatible MapTransforms ---

class RateCode(grain.MapTransform):
    """
    Grain MapTransform for rate encoding.
    """
    def __init__(self, sample_T, max_r=0.75, input_key="obs", output_key="obs"):
        self.sample_T = sample_T
        self.max_r = max_r
        self.input_key = input_key
        self.output_key = output_key

    def map(self, record):
        data = record[self.input_key]
        # Assumes data is scaled 0-1
        # We use numpy for transformation in grain pipelines
        spikes = np.random.rand(self.sample_T, *data.shape) < (data * self.max_r)
        record[self.output_key] = np.packbits(spikes.astype(np.uint8), axis=0)
        return record

class ShiftAugment(grain.MapTransform):
    """
    Grain MapTransform for random shift augmentation.
    """
    def __init__(self, max_shift=10, axes=(-1,), input_key="obs"):
        self.max_shift = max_shift
        self.axes = axes
        self.input_key = input_key

    def map(self, record):
        data = record[self.input_key]
        shift = np.random.randint(-self.max_shift, self.max_shift, size=len(self.axes))
        record[self.input_key] = np.roll(data, shift, axis=self.axes)
        return record

class LatencyCode(grain.MapTransform):
    """Grain MapTransform for time-to-first-spike (latency) encoding.

    Counterpart to :func:`latency_code`, wrapped in the Grain op interface so
    it can slot into an existing ``SHD_loader``-style pipeline.
    """

    def __init__(self, sample_T, threshold=0.01, input_key="obs", output_key="obs"):
        self.sample_T = sample_T
        self.threshold = threshold
        self.input_key = input_key
        self.output_key = output_key

    def map(self, record):
        data = np.asarray(record[self.input_key], dtype=np.float32)
        data = np.clip(data, 0.0, 1.0)
        spike_idx = np.round((1.0 - data) * (self.sample_T - 1)).astype(np.int32)
        # Build a one-hot mask along axis 0.
        spikes = np.zeros((self.sample_T,) + data.shape, dtype=np.uint8)
        idx_grid = np.indices(data.shape)
        silent = data <= self.threshold
        # Fire only at the computed time bin, and only for non-silent units.
        spikes[(spike_idx, *idx_grid)] = np.where(silent, 0, 1).astype(np.uint8)
        record[self.output_key] = np.packbits(spikes, axis=0)
        return record


class AngleCode(grain.MapTransform):
    """
    Grain MapTransform for angle encoding.
    """
    def __init__(self, neuron_count, min_val, max_val, input_key="obs", output_key="obs"):
        self.neuron_count = neuron_count
        self.min_val = min_val
        self.max_val = max_val
        self.input_key = input_key
        self.output_key = output_key
        self.neurons = np.linspace(min_val, max_val, neuron_count)

    def map(self, record):
        obs = record[self.input_key]
        digital = np.digitize(obs, self.neurons) - 1
        idx = np.clip(digital, 0, self.neuron_count - 1)
        record[self.output_key] = np.eye(self.neuron_count, dtype=np.uint8)[idx]
        return record

# --- Grain Loaders (Merged from loaders.py) ---

class TonicSource:
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        data, label = self.ds[i]
        # remove any dimensions of size 1 (e.g. for SHD)
        data = np.squeeze(data)
        # Return as dict for Grain MapDataset
        return {"obs": np.packbits(data, axis=0), "labels": label}

class SpyxMapDataset(grain.MapDataset):
    def __init__(self, source):
        self._source = source
    def __len__(self):
        return len(self._source)
    def __getitem__(self, i):
        return self._source[i]

def _default_worker_count() -> int:
    """Choose a sane default worker count for Grain.

    Tonic's h5 reads + ``ToFrame`` pipeline are single-threaded Python;
    running them on a single prefetch thread (``worker_count=0``) makes
    the first SHD / NMNIST batch take tens of seconds to produce on a
    laptop with ``batch_size=256``. Half the CPU cores, capped at 4, is
    the sweet spot between throughput and memory.
    """
    import os

    n = (os.cpu_count() or 2) // 2
    return max(1, min(4, n))


class GrainLoader:
    """A wrapper around Grain's DataLoader with a Spyx-compatible interface.

    :dataset: grain ``MapDataset`` (or any ``len + getitem`` source).
    :batch_size: items per emitted ``State``.
    :shuffle: whether to shuffle sample order.
    :seed: RNG seed for the sampler.
    :worker_count: number of Grain worker processes. ``None`` picks
        ``_default_worker_count()``; ``0`` disables multiprocessing
        (useful for debugging but slow for tonic-backed sources).
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        seed=0,
        worker_count=None,
    ):
        if worker_count is None:
            worker_count = _default_worker_count()

        sampler = grain.IndexSampler(
            num_records=len(dataset),
            shuffle=shuffle,
            seed=seed,
            shard_options=grain.NoSharding()
        )

        self.data_loader = grain.DataLoader(
            data_source=dataset,
            sampler=sampler,
            worker_count=worker_count,
            operations=[grain.Batch(batch_size, drop_remainder=True)]
        )

    def __iter__(self):
        for batch in self.data_loader:
            yield State(obs=batch["obs"], labels=batch["labels"])

class NMNIST_loader:
    """Dataloader for the Neuromorphic MNIST dataset using Google Grain and Tonic.

    :worker_count: number of Grain worker processes. ``None`` picks a sensible
        default (half your CPU cores, capped at 4); ``0`` disables multi-
        processing. Passing a positive integer usually cuts first-batch
        latency significantly when tonic is decoding samples.
    """

    def __init__(
        self, batch_size=32, sample_T=40, key=0, download_dir='./data',
        worker_count=None,
    ):
        if not tonic_installed:
            raise ImportError("Please install the optional dependencies by running 'pip install spyx[loaders]' to use this feature.")

        self.batch_size = batch_size
        self.sample_T = sample_T
        self.obs_shape = (2, 34, 34)
        self.act_shape = (10,)

        transform = transforms.Compose([
            transforms.ToFrame(sensor_size=(34, 34, 2), n_time_bins=sample_T),
        ])

        train_ds = datasets.NMNIST(download_dir, train=True, transform=transform)
        test_ds = datasets.NMNIST(download_dir, train=False, transform=transform)

        train_mds = SpyxMapDataset(TonicSource(train_ds))
        test_mds = SpyxMapDataset(TonicSource(test_ds))

        self._train_dl = GrainLoader(train_mds, batch_size, shuffle=True, seed=key, worker_count=worker_count)
        self._test_dl = GrainLoader(test_mds, batch_size, shuffle=False, seed=key, worker_count=worker_count)

    def train_epoch(self):
        return iter(self._train_dl)

    def test_epoch(self):
        return iter(self._test_dl)

class SHD_loader:
    """Dataloader for the Spiking Heidelberg Dataset using Google Grain and Tonic.

    Notes:

    * Tonic's stock ``HSD.__getitem__`` reads ``spikes/times`` as
      float32 and doesn't filter non-finite values, which triggers
      noisy NumPy warnings and, for a handful of samples, produces
      garbage event streams that make ``ToFrame`` allocate enormous
      frames. ``spyx.data`` patches ``HSD`` at import time (see
      :func:`_patch_tonic_hsd`) to cast to float64 and drop non-finite
      timestamps. The patch is a monkey-patch pending an upstream fix
      in tonic.

    :worker_count: number of Grain worker processes. ``None`` picks a
        sensible default (half your CPU cores, capped at 4); ``0``
        disables multiprocessing. For the default ``batch_size=256``
        on a laptop, setting ``worker_count=4`` cuts first-batch
        latency from ~30s to a few seconds.
    """

    def __init__(
        self, batch_size=256, sample_T=128, channels=128, key=0,
        download_dir='./data', worker_count=None,
    ):
        if not tonic_installed:
            raise ImportError("Please install the optional dependencies by running 'pip install spyx[loaders]' to use this feature.")

        net_channels = channels
        self.obs_shape = (channels,)
        self.act_shape = (20,)
        self.batch_size = batch_size
        self.sample_T = sample_T

        # Note: we intentionally leave timestamps in microseconds and let
        # ToFrame(n_time_bins=sample_T) do the temporal binning. Applying
        # Downsample(time_factor=...) first (which we used to do) compresses
        # the timestamps into [0, sample_T] integer range; tonic's
        # SliceByTimeBins then computes the per-bin window as
        # ``(times[-1] - times[0]) // n_time_bins``, which floor-divides to
        # 0 or 1 and silently yields empty frames for every SHD sample.
        # See https://github.com/neuromorphs/tonic/issues/313 for the
        # upstream tracking of the underlying slicing behaviour.
        transform = transforms.Compose([
            transforms.Downsample(spatial_factor=net_channels / 700),
            transforms.ToFrame(
                sensor_size=(net_channels, 1, 1),
                n_time_bins=sample_T,
            ),
        ])

        train_ds = datasets.SHD(download_dir, train=True, transform=transform)
        test_ds = datasets.SHD(download_dir, train=False, transform=transform)

        train_mds = SpyxMapDataset(TonicSource(train_ds))
        test_mds = SpyxMapDataset(TonicSource(test_ds))

        self._train_dl = GrainLoader(train_mds, batch_size, shuffle=True, seed=key, worker_count=worker_count)
        self._test_dl = GrainLoader(test_mds, batch_size, shuffle=False, seed=key, worker_count=worker_count)

    def train_epoch(self):
        return iter(self._train_dl)

    def test_epoch(self):
        return iter(self._test_dl)