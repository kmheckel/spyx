import jax
import jax.numpy as jnp
import haiku as hk

import numpy as np
import tonic
from tonic import datasets, transforms


from collections import namedtuple


State = namedtuple("State", "obs labels")


@jax.jit
def train_test_split_indices(data, split_ratio, seed=0):
    """
    Split indices into train and test sets based on a given ratio.
    
    Args:
        data (jax.numpy array): Input data array, used to determine the number of indices.
        split_ratio (float): Fraction of data to be used for training (0 < split_ratio < 1).
        seed (int): Seed for random number generator.
        
    Returns:
        tuple: Two arrays of indices, one for training and one for testing.
    """
    key = jax.random.PRNGKey(seed)
    num_samples = data.shape[0]
    indices = jax.random.permutation(key, jnp.arange(num_samples))
    
    train_size = int(num_samples * split_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    return train_indices, test_indices

def tonic_dataset2jnp(tonic_dataset):
    X = jnp.stack([i[0] for i in tonic_dataset])
    y = jnp.stack([i[1] for i in tonic_dataset])
    return X, y

class NMNIST_loader():
    """
    Dataloading wrapper for the Neuromorphic MNIST dataset.

    :batch_size: Samples per batch.
    :sample_T: Timesteps per sample/length of time axis.
    :data_subsample: Use a fraction of the training/validation sets to reduce computational demand.
    :val_size: Proportion of dataset to set aside for validation.
    :key: Integer specifying the random seed for the train/val split.
    :download_dir: The local directory to save the data to. 
    """

    # Change this to allow a config dictionary of 
    def __init__(self, batch_size=32, sample_T = 40, val_size=0.3, seed=0, download_dir='./NMNIST'):

        self.val_size = val_size
        self.batch_size = batch_size
        self.obs_shape = datasets.NMNIST.sensor_size
        self.act_shape = tuple([10,])
        
        transform =  transforms.Compose([
            transforms.ToFrame(sensor_size=self.obs_shape, 
                                       n_time_bins=sample_T),
            lambda x: np.packbits(x, axis=0)

        ])
        

        train_val_dataset = datasets.NMNIST("./data", first_saccade_only=True, train=True, transform=transform)
        test_dataset = datasets.NMNIST("./data", first_saccade_only=True, train=False, transform=transform)

        train_val_dataset = tonic_dataset2jnp(train_val_dataset)
        test_dataset = tonic_dataset2jnp(test_dataset) 
        
        train_indices, val_indices = train_test_split_indices(
        train_val_dataset,
        test_size=self.val_size,
        seed=seed
        )
    
    
        self.x_train = train_val_X[train_indices]
        self.y_train = train_val_y[train_indices]
        self.train_len = len(train_indices)

        ############################
                
        self.x_val = train_val_X[val_indices]
        self.y_val = train_val_y[val_indices]
        self.val_len = len(val_indices)

        ##########################
        # Test set setup
        ##########################
                
        self.x_test, self.y_test = test_dataset
        self.test_len = len(self.x_test)


        
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

def pad_axis_0(array, target_size):
    """
    Zero-pads the first axis of a NumPy array to reach the specified target size.
    
    Args:
        array (numpy.ndarray): The input array to pad.
        target_size (int): The desired size along the first axis after padding.
        
    Returns:
        numpy.ndarray: The zero-padded array.
    """
    current_size = array.shape[0]
    if current_size >= target_size:
        return array  # No padding needed if already at or above target size
    
    padding = ((0, target_size - current_size),) + ((0, 0),) * (array.ndim - 1)
    return np.pad(array, padding, mode='constant', constant_values=0)

# Builds 2D tensors from data, with the time axis being packed to save memory. 
class _SHD2Raster():
    """ 
    Tool for rastering SHD samples into frames. Packs bits along the temporal axis for memory efficiency. This means
        that the used will have to apply jnp.unpackbits(events, axis=<time axis>) prior to feeding the data to the network.
    """

    def __init__(self, encoding_dim, sample_T = 100):
        if not optional_dependencies_installed:
            raise ImportError("Please install the optional dependencies by running 'pip install spyx[loaders]' to use this feature.")
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
        return pad_axis_0(tensor, self.sample_T//8)
    

class SHD_loader():
    """
    Dataloading wrapper for the Spiking Heidelberg Dataset. The entire dataset is loaded to vRAM in a temporally compressed format. The user must
    apply jnp.unpackbits(events, axis=<time axis>) prior to feeding to an SNN. 

    https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/

    
    :batch_size: Number of samples per batch.
    :sample_T: Number of time steps per sample.
    :channels: Number of frequency channels used.
    :val_size: Fraction of the training dataset to set aside for validation.
    """


    # Change this to allow a config dictionary of 
    def __init__(self, batch_size=256, sample_T = 128, channels=128, val_size=0.2, seed=0):
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
            _SHD2Raster(net_channels, sample_T = sample_T)
        ])
        
        train_val_dataset = datasets.SHD("./data", train=True, transform=transform)
        test_dataset = datasets.SHD("./data", train=False, transform=transform)
        
        train_val_X, train_val_y = tonic_dataset2jnp(train_val_dataset)
        test_dataset = tonic_dataset2jnp(test_dataset)
        
        #########################################################################
        # load entire dataset to GPU as JNP Array, create methods for splits
        #########################################################################

    
        # create train/validation split here...
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, val_indices = train_test_split_indices(
            train_val_dataset[0],
            test_size=self.val_size,
            seed=seed,
        )

        self.x_train = train_val_X[train_indices]
        self.y_train = train_val_y[train_indices]
        self.train_len = len(train_indices)

        ############################
                
        self.x_val = train_val_X[val_indices]
        self.y_val = train_val_y[val_indices]
        self.val_len = len(val_indices)

        ##########################
        # Test set setup
        ##########################
                
        self.x_test, self.y_test = test_dataset
        self.test_len = len(self.x_test)


        
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
    
    