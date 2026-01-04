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

class GrainLoader:
    """
    A wrapper around Grain's DataLoader to provide a Spyx-compatible interface.
    """
    def __init__(self, dataset, batch_size, shuffle, seed=0):
        sampler = grain.IndexSampler(
            num_records=len(dataset),
            shuffle=shuffle,
            seed=seed,
            shard_options=grain.NoSharding()
        )
        
        self.data_loader = grain.DataLoader(
            data_source=dataset,
            sampler=sampler,
            worker_count=0,
            operations=[grain.Batch(batch_size, drop_remainder=True)]
        )
        
    def __iter__(self):
        for batch in self.data_loader:
            yield State(obs=batch["obs"], labels=batch["labels"])

class NMNIST_loader:
    """
    Dataloader for the Neuromorphic MNIST dataset using Google Grain and Tonic.
    """
    def __init__(self, batch_size=32, sample_T=40, key=0, download_dir='./data'):
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
        
        self._train_dl = GrainLoader(train_mds, batch_size, shuffle=True, seed=key)
        self._test_dl = GrainLoader(test_mds, batch_size, shuffle=False, seed=key)

    def train_epoch(self):
        return iter(self._train_dl)

    def test_epoch(self):
        return iter(self._test_dl)

class SHD_loader:
    """
    Dataloader for the Spiking Heidelberg Dataset using Google Grain and Tonic.
    """
    def __init__(self, batch_size=256, sample_T=128, channels=128, key=0, download_dir='./data'):
        if not tonic_installed:
            raise ImportError("Please install the optional dependencies by running 'pip install spyx[loaders]' to use this feature.")

        shd_timestep = 1e-6
        net_channels = channels
        net_dt = 1/sample_T
        self.obs_shape = (channels,)
        self.act_shape = (20,)
        self.batch_size = batch_size
        self.sample_T = sample_T
        
        transform = transforms.Compose([
            transforms.Downsample(
                time_factor=shd_timestep / net_dt,
                spatial_factor=net_channels / 700
            ),
            transforms.ToFrame(sensor_size=(net_channels, 1, 1), n_time_bins=sample_T)
        ])
        
        train_ds = datasets.SHD(download_dir, train=True, transform=transform)
        test_ds = datasets.SHD(download_dir, train=False, transform=transform)

        train_mds = SpyxMapDataset(TonicSource(train_ds))
        test_mds = SpyxMapDataset(TonicSource(test_ds))
        
        self._train_dl = GrainLoader(train_mds, batch_size, shuffle=True, seed=key)
        self._test_dl = GrainLoader(test_mds, batch_size, shuffle=False, seed=key)

    def train_epoch(self):
        return iter(self._train_dl)

    def test_epoch(self):
        return iter(self._test_dl)