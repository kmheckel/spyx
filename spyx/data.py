import tonic
from tonic import datasets, transforms
import torchvision as tv
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from collections import namedtuple
from itertools import cycle

import numpy as np
import jax
import jax.numpy as jnp


State = namedtuple("State", "obs labels")


class shift_augment:
    """
        Shift data augmentation tool. Rolls data along specified axes randomly up to a certain amount.
    """

    def __init__(self, max_shift=10, axes=(-1,)):
        self.max_shift = max_shift
        self.axes = axes

        @jax.jit
        def _shift(data, rng):
            shift = jax.random.randint(rng, (len(self.axes),), -self.max_shift, self.max_shift)
            return jnp.roll(data, shift, self.axes)

        self._shift = _shift


    def __call__(self, data, rng):
        return self._shift(data, rng)





# Here we scale the max probability to .75 so that we don't have inputs that are continually spiking.
def rate_code(data, steps, key, max_r=0.75):
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
            tv.transforms.Normalize((0,), (1,)),
            lambda x: np.expand_dims(x, axis=-1)
            ])
        
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

        train_dl = iter(DataLoader(train_split, batch_size=self.train_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
        x_train, y_train = next(train_dl)
        self.x_train = jnp.array(x_train, dtype=jnp.uint8)
        self.y_train = jnp.array(y_train, dtype=jnp.uint8)
        ############################
        
        val_split = Subset(train_val_dataset, val_indices)
        self.val_len = len(val_indices)

        val_dl = iter(DataLoader(val_split, batch_size=self.val_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
        x_val, y_val = next(val_dl)
        self.x_val = jnp.array(x_val, dtype=jnp.uint8)
        self.y_val = jnp.array(y_val, dtype=jnp.uint8)
        ##########################
        # Test set setup
        ##########################
        self.test_len = len(test_dataset)
        test_dl = iter(DataLoader(test_dataset, batch_size=self.test_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
                
        x_test, y_test = next(test_dl)
        self.x_test = jnp.array(x_test, dtype=jnp.uint8)
        self.y_test = jnp.array(y_test, dtype=jnp.uint8)


        
        @jax.jit
        def _train_epoch(shuffle_key):
            cutoff = self.train_len % self.batch_size
            
            obs = jax.random.permutation(shuffle_key, self.x_train, axis=0)[:-cutoff] # self.x_train[:-cutoff]
            labels = jax.random.permutation(shuffle_key, self.y_train, axis=0)[:-cutoff] # self.y_train[:-cutoff]
            
            obs = jnp.reshape(obs, (-1, self.batch_size) + obs.shape[1:])
            labels = jnp.reshape(labels, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
            
        self.train_epoch = _train_epoch
            
        @jax.jit
        def _val_epoch():
            cutoff = self.val_len % self.batch_size
            
            x_val = self.x_val[:-cutoff]
            y_val = self.y_val[:-cutoff]
            
            obs = jnp.reshape(x_val, (-1, self.batch_size) + x_val.shape[1:])
            labels = jnp.reshape(y_val, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
        
        self.val_epoch = _val_epoch
        
        
        @jax.jit
        def _test_epoch():
            cutoff = self.test_len % self.batch_size
            
            x_test = self.x_test[:-cutoff]
            y_test = self.y_test[:-cutoff]
            
            obs = jnp.reshape(x_test, (-1, self.batch_size) + x_test.shape[1:])
            labels = jnp.reshape(y_test, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
        
        self.test_epoch = _test_epoch


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
        
        transform =  transforms.Compose([
            transforms.ToFrame(sensor_size=self.obs_shape, 
                                       n_time_bins=sample_T),
            lambda x: np.packbits(x, axis=1)

        ])
        

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

        train_dl = iter(DataLoader(train_split, batch_size=self.train_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
        x_train, y_train = next(train_dl)
        self.x_train = jnp.array(x_train, dtype=jnp.uint8)
        self.y_train = jnp.array(y_train, dtype=jnp.uint8)
        ############################
        
        val_split = Subset(train_val_dataset, val_indices)
        self.val_len = len(val_indices)

        val_dl = iter(DataLoader(val_split, batch_size=self.val_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
        x_val, y_val = next(val_dl)
        self.x_val = jnp.array(x_val, dtype=jnp.uint8)
        self.y_val = jnp.array(y_val, dtype=jnp.uint8)
        ##########################
        # Test set setup
        ##########################
        self.test_len = len(test_dataset)
        test_dl = iter(DataLoader(test_dataset, batch_size=self.test_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
                
        x_test, y_test = next(test_dl)
        self.x_test = jnp.array(x_test, dtype=jnp.uint8)
        self.y_test = jnp.array(y_test, dtype=jnp.uint8)


        
        @jax.jit
        def _train_epoch(shuffle_key):
            cutoff = self.train_len % self.batch_size
            
            obs = jax.random.permutation(shuffle_key, self.x_train, axis=0)[:-cutoff] # self.x_train[:-cutoff]
            labels = jax.random.permutation(shuffle_key, self.y_train, axis=0)[:-cutoff] # self.y_train[:-cutoff]
            
            obs = jnp.reshape(obs, (-1, self.batch_size) + obs.shape[1:])
            labels = jnp.reshape(labels, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
            
        self.train_epoch = _train_epoch
            
        @jax.jit
        def _val_epoch():
            cutoff = self.val_len % self.batch_size
            
            x_val = self.x_val[:-cutoff]
            y_val = self.y_val[:-cutoff]
            
            obs = jnp.reshape(x_val, (-1, self.batch_size) + x_val.shape[1:])
            labels = jnp.reshape(y_val, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
        
        self.val_epoch = _val_epoch
        
        
        @jax.jit
        def _test_epoch():
            cutoff = self.test_len % self.batch_size
            
            x_test = self.x_test[:-cutoff]
            y_test = self.y_test[:-cutoff]
            
            obs = jnp.reshape(x_test, (-1, self.batch_size) + x_test.shape[1:])
            labels = jnp.reshape(y_test, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
        
        self.test_epoch = _test_epoch

###########################################################

# Builds 2D tensors from data, with the time axis being packed to save memory. 
class SHD2Raster():
    """ Tool for rastering SHD samples into frames. Packs bits along the temporal axis for memory efficiency. This means
        that the used will have to apply jnp.unpackbits(events, axis=<time axis>) prior to feeding the data to the network.
    """

    def __init__(self, encoding_dim, sample_T = 100):
        self.encoding_dim = encoding_dim
        self.sample_T = sample_T
        
    def __call__(self, events):
        # tensor has dimensions (time_steps, encoding_dim)
        tensor = np.zeros((events["t"].max()+1, self.encoding_dim), dtype=int)
        np.add.at(tensor, (events["t"], events["x"]), 1)
        #return tensor[:self.sample_T,:]
        tensor = tensor[:self.sample_T,:]
        tensor = np.minimum(tensor, 1)
        tensor = np.packbits(tensor, axis=0)
        return tensor
    

class SHD_loader():
    """
    Dataloading wrapper for the Spiking Heidelberg Dataset. The entire dataset is loaded to vRAM in a temporally compressed format. The user must
    apply jnp.unpackbits(events, axis=<time axis>) prior to feeding to an SNN. Right now the timescaling is suited for windows of about 100 timesteps.

    https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
    """


    # Change this to allow a config dictionary of 
    def __init__(self, batch_size=256, sample_T = 100, channels=128, val_size=0.2):
        #####################################
        # Load datasets and process them using tonic.
        #####################################
        shd_timestep = 1e-6
        shd_channels = 700
        net_channels = channels
        net_dt = 1/sample_T
           
        self.batch_size = batch_size
        self.val_size = val_size
        self.obs_shape = tuple([net_channels,])
        self.act_shape = tuple([20,])
        
        transform = transforms.Compose([
        transforms.Downsample(
            time_factor=shd_timestep / net_dt,
            spatial_factor=net_channels / shd_channels
            ),
            SHD2Raster(net_channels, sample_T = sample_T)
        ])
        
        train_val_dataset = datasets.SHD("./data", train=True, transform=transform)
        test_dataset = datasets.SHD("./data", train=False, transform=transform)
        
        
        #########################################################################
        # load entire dataset to GPU as JNP Array, create methods for splits
        #########################################################################

    
        # create train/validation split here...
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, val_indices = train_test_split(
            range(len(train_val_dataset)),
            test_size=self.val_size,
            random_state=0,
            shuffle=True # This really should be set externally!!!!!
        )


        train_split = Subset(train_val_dataset, train_indices)
        self.train_len = len(train_indices)

        train_dl = iter(DataLoader(train_split, batch_size=self.train_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
        x_train, y_train = next(train_dl)
        self.x_train = jnp.array(x_train, dtype=jnp.uint8)
        self.y_train = jnp.array(y_train, dtype=jnp.uint8)
        ############################
        
        val_split = Subset(train_val_dataset, val_indices)
        self.val_len = len(val_indices)

        val_dl = iter(DataLoader(val_split, batch_size=self.val_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
        x_val, y_val = next(val_dl)
        self.x_val = jnp.array(x_val, dtype=jnp.uint8)
        self.y_val = jnp.array(y_val, dtype=jnp.uint8)
        ##########################
        # Test set setup
        ##########################
        self.test_len = len(test_dataset)
        test_dl = iter(DataLoader(test_dataset, batch_size=self.test_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
                
        x_test, y_test = next(test_dl)
        self.x_test = jnp.array(x_test, dtype=jnp.uint8)
        self.y_test = jnp.array(y_test, dtype=jnp.uint8)


        
        @jax.jit
        def _train_epoch(shuffle_key):
            cutoff = self.train_len % self.batch_size
            
            obs = jax.random.permutation(shuffle_key, self.x_train, axis=0)[:-cutoff] # self.x_train[:-cutoff]
            labels = jax.random.permutation(shuffle_key, self.y_train, axis=0)[:-cutoff] # self.y_train[:-cutoff]
            
            obs = jnp.reshape(obs, (-1, self.batch_size) + obs.shape[1:])
            labels = jnp.reshape(labels, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
            
        self.train_epoch = _train_epoch
            
        @jax.jit
        def _val_epoch():
            cutoff = self.val_len % self.batch_size
            
            x_val = self.x_val[:-cutoff]
            y_val = self.y_val[:-cutoff]
            
            obs = jnp.reshape(x_val, (-1, self.batch_size) + x_val.shape[1:])
            labels = jnp.reshape(y_val, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
        
        self.val_epoch = _val_epoch
        
        
        @jax.jit
        def _test_epoch():
            cutoff = self.test_len % self.batch_size
            
            x_test = self.x_test[:-cutoff]
            y_test = self.y_test[:-cutoff]
            
            obs = jnp.reshape(x_test, (-1, self.batch_size) + x_test.shape[1:])
            labels = jnp.reshape(y_test, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
        
        self.test_epoch = _test_epoch
    
    