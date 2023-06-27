import tonic
from tonic import datasets, transforms
import torchvision as tv
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from collections import namedtuple
from itertools import cycle

import numpy as np
import jax
import jax.numpy as jnp


State = namedtuple("State", "obs labels")


class shift_augment:
    """
        Shift data ugmentation tool. Rolls data along specified axes randomly up to a certain amount.
    """

    def __init__(self, max_shift=10, axes=(-1,), key=0):
        self.rng = jax.random.PRNGKey(key)
        self.max_shift = max_shift
        self.axes = axes

        @jax.jit
        def _shift(data, rng):
            rng, shift_rng = jax.random.split(rng)
            shift = jax.random.randint(shift_rng, (len(self.axes),), -self.max_shift, self.max_shift)
            return jnp.roll(data, shift, self.axes), rng

        self._shift = _shift


    def __call__(self, data):
        shifted, self.rng = self._shift(data, self.rng)
        return shifted






# Here we scale the max probability to .8 so that we don't have inputs that are continually spiking.
# might need to find a home for this in the lib.
def rate_code(data, steps, key, max_r=0.8):
    """Unrolls input data along axis 1 and converts to rate encoded spikes."""

    data = jnp.array(data, dtype=jnp.float16)
    unrolled_data = jnp.repeat(data, steps, axis=1)
    return jax.random.bernoulli(key, unrolled_data*max_r).astype(jnp.int8)


class MNIST_loader():
    """
    Dataloader for the MNIST dataset, right now it is rate encoded.
    
    """

    # Change this to allow a config dictionary of 
    def __init__(self, time_steps=64, max_rate = 0.5, batch_size=32, val_size=0.3, key=0, download_dir='./MNIST'):
           
        self.key = jax.random.PRNGKey(key)
        self.sample_T = time_steps
        self.max_rate = max_rate
        self.val_size = val_size
        self.batch_size = batch_size
        self.obs_shape = (28,28)
        self.act_shape = tuple([10,])
        
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.obs_shape),
            tv.transforms.Grayscale(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0,), (1,))])
        
        # fix this
        train_val_dataset = tv.datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_dataset = tv.datasets.MNIST("./data", train=False, download=True, transform=transform)
        # create train/validation split here...
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, val_indices = train_test_split(
        range(len(train_val_dataset)),
        test_size=self.val_size,
        random_state=0,
        shuffle=True
        )
    
    
        train_split = Subset(train_val_dataset, train_indices)
        self.train_len = len(train_indices)
        
        val_split = Subset(train_val_dataset, val_indices)
        self.val_len = len(val_indices)
                        
        self.test_len = len(test_dataset)
        
        # change this to just dl and add if statement based on test=T/F
        self._train_dl = DataLoader(train_split, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False)
        self.train_dl = iter(self._train_dl)
        
        self._val_dl = DataLoader(val_split, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False)
        self.val_dl = iter(self._val_dl) 
        
        self._test_dl = DataLoader(test_dataset, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False)
        self.test_dl = iter(self._test_dl)
        
    def train_reset(self):
        self.train_dl = iter(self._train_dl)
        
    def train_step(self):
        batch_data, batch_labels = next(self.train_dl)
        key, self.key = jax.random.split(self.key)
        spike_data = jnp.expand_dims(rate_code(batch_data, self.sample_T, key, self.max_rate), -1)
        return State(obs=jnp.array(spike_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))
    
    def val_reset(self):
        self.val_dl = iter(self._val_dl)
        
    def val_step(self):
        batch_data, batch_labels = next(self.val_dl)
        key, self.key = jax.random.split(self.key)
        spike_data = jnp.expand_dims(rate_code(batch_data, self.sample_T, key, self.max_rate), -1)
        return State(obs=jnp.array(spike_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))
        
    def test_reset(self):
        self.test_dl = iter(self._test_dl)
        
    def test_step(self):
        batch_data, batch_labels = next(self.test_dl)
        key, self.key = jax.random.split(self.key)
        spike_data = jnp.expand_dims(rate_code(batch_data, self.sample_T, key, self.max_rate), -1)
        return State(obs=jnp.array(spike_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))


###############################################

class NMNIST_loader():
    """
    Dataloading wrapper for the Neuromorphic MNIST dataset.
    """

    # Change this to allow a config dictionary of 
    def __init__(self, batch_size=32, val_size=0.3, download_dir='./NMNIST'):
        sample_T = 64
           
        self.val_size = val_size
        self.batch_size = batch_size
        self.obs_shape = datasets.NMNIST.sensor_size
        self.act_shape = tuple([10,])
        
        transform = transforms.ToFrame(sensor_size=self.obs_shape, 
                                       n_time_bins=sample_T)
        

        train_val_dataset = datasets.NMNIST("./data", first_saccade_only=True, train=True, transform=transform)
        test_dataset = datasets.NMNIST("./data", first_saccade_only=True, train=False, transform=transform)
        train_indices, val_indices = train_test_split(
        range(len(train_val_dataset)),
        test_size=self.val_size,
        random_state=0,
        shuffle=True
        )
    
    
        train_split = Subset(train_val_dataset, train_indices)
        self.train_len = len(train_indices)
        
        val_split = Subset(train_val_dataset, val_indices)
        self.val_len = len(val_indices)
                        
        self.test_len = len(test_dataset)
        
        # change this to just dl and add if statement based on test=T/F
        self._train_dl = DataLoader(train_split, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False)
        self.train_dl = iter(self._train_dl)
        
        self._val_dl = DataLoader(val_split, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False)
        self.val_dl = iter(self._val_dl) 
        
        self._test_dl = DataLoader(test_dataset, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False)
        self.test_dl = iter(self._test_dl)
    
    def train_reset(self):
        self.train_dl = iter(self._train_dl)
        
    def train_step(self):
        batch_data, batch_labels = next(self.train_dl)
        return State(obs=jnp.array(batch_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))
    
    def val_reset(self):
        self.val_dl = iter(self._val_dl)
        
    def val_step(self):
        batch_data, batch_labels = next(self.val_dl)
        return State(obs=jnp.array(batch_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))
    
    def test_reset(self):
        self.test_dl = iter(self._test_dl)
        
    def test_step(self):
        batch_data, batch_labels = next(self.test_dl)
        return State(obs=jnp.array(batch_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))

###########################################################

# should push all of the transforms into here and make a single bin/frame func
class SHD2Raster():
    """ Tool for rastering SHD samples into frames."""

    def __init__(self, encoding_dim, sample_T = 100, binarize=True):
        self.encoding_dim = encoding_dim
        self.sample_T = sample_T
        self.binarize = binarize
        
    def __call__(self, events):
        # tensor has dimensions (time_steps, encoding_dim)
        tensor = np.zeros((events["t"].max()+1, self.encoding_dim), dtype=int)
        np.add.at(tensor, (events["t"], events["x"]), 1)
        #return tensor[:self.sample_T,:]
        tensor = tensor[:self.sample_T,:]
        if self.binarize:
            tensor = np.minimum(tensor, 1)
        return tensor
    

class SHD_loader():
    """
    Dataloading wrapper for the Spiking Heidelberg Dataset.

    https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
    """


    # Change this to allow a config dictionary of 
    def __init__(self, batch_size=128, sample_T = 100, channels=128, holdout=1, binarize=True):        
        shd_timestep = 1e-6
        shd_channels = 700
        net_channels = channels
        net_dt = 10e-3
           
        self.batch_size = batch_size
        self.obs_shape = tuple([net_channels,])
        self.act_shape = tuple([20,])
        
        transform = transforms.Compose([
        transforms.Downsample(
            time_factor=shd_timestep / net_dt,
            spatial_factor=net_channels / shd_channels
            ),
            SHD2Raster(net_channels, sample_T = sample_T, binarize=binarize)
        ])
        
        self.train_val_dataset = datasets.SHD("./data", train=True, transform=transform)
        test_dataset = datasets.SHD("./data", train=False, transform=transform)
        
        cross_validator = GroupKFold(n_splits=10-holdout)
        self.cross_validator = cycle(cross_validator.split([*range(len(self.train_val_dataset))], groups=self.train_val_dataset.speaker))
        train_indices, val_indices = next(self.cross_validator)
        
        
        train_split = Subset(self.train_val_dataset, train_indices)
        self.train_len = len(train_indices)
        
        val_split = Subset(self.train_val_dataset, val_indices)
        self.val_len = len(val_indices)
                        
        self.test_len = len(test_dataset)
        
        # change this to just dl and add if statement based on test=T/F
        self._train_dl = DataLoader(train_split, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=True)
        self.train_dl = iter(self._train_dl)
        
        self._val_dl = DataLoader(val_split, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=True)
        self.val_dl = iter(self._val_dl) 
        
        self._test_dl = DataLoader(test_dataset, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False)
        self.test_dl = iter(self._test_dl)
        
        
    # This class implements Leave One Group Out, so that each epoch is performed with one
    # speaker being retained for the validation set.
    def train_reset(self):
        train_indices, val_indices = next(self.cross_validator)
        self.train_len = len(train_indices)
        self.val_len = len(val_indices)
        train_split = Subset(self.train_val_dataset, train_indices)
        val_split = Subset(self.train_val_dataset, val_indices)
        
        self._train_dl = DataLoader(train_split, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=True)
        self._val_dl = DataLoader(val_split, batch_size=self.batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=True)
        
        self.val_dl = iter(self._val_dl)
        self.train_dl = iter(self._train_dl)
        
    def train_step(self):
        batch_data, batch_labels = next(self.train_dl)
        return State(obs=jnp.array(batch_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))
    
    def val_reset(self):
        self.val_dl = iter(self._val_dl)
        
    def val_step(self):
        batch_data, batch_labels = next(self.val_dl)
        return State(obs=jnp.array(batch_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))
    
    def test_reset(self):
        self.test_dl = iter(self._test_dl)
        
    def test_step(self):
        batch_data, batch_labels = next(self.test_dl)
        return State(obs=jnp.array(batch_data, dtype=jnp.int8), # perform cast to jnp.int8 here
                     labels=jnp.array(batch_labels))