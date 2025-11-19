"""
Grain-based dataloaders for spyx using pure JAX ecosystem.
No torch/torchvision dependencies - uses TFDS + grain + dm-pix instead.
"""
import jax
import jax.numpy as jnp
import grain as grain
import numpy as np
from typing import Any, Callable, Optional, Tuple
from functools import partial

try:
    import tonic
    from tonic import datasets
    from sklearn.model_selection import train_test_split
    tonic_available = True
except ImportError:
    tonic_available = False


class TonicDataSource(grain.sources.RandomAccessDataSource):
    """Wrapper for Tonic datasets (event-based) to be used with Grain."""
    
    def __init__(self, tonic_dataset):
        """
        Args:
            tonic_dataset: A Tonic dataset (e.g., datasets.NMNIST)
        """
        self._dataset = tonic_dataset
        
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        return self._dataset[idx]


def rate_encode_batch(batch, sample_T: int, max_rate: float = 0.75):
    """
    Apply rate encoding to a batch of images.
    
    Args:
        batch: Dict with 'image' and 'label' keys
        sample_T: Number of time steps
        max_rate: Maximum firing rate
        
    Returns:
        Tuple of (encoded_images, labels)
    """
    images = batch['image']
    labels = batch['label']
    
    # Normalize to [0, 1] if needed
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    
    # Repeat for time dimension: (B, H, W, C) -> (B, T, H, W, C)
    images = np.repeat(images[:, None, ...], sample_T, axis=1)
    
    # Rate coding (Bernoulli sampling)
    rand = np.random.rand(*images.shape)
    spikes = (rand < (images * max_rate)).astype(np.uint8)
    
    # Pack bits along time dimension for memory efficiency
    # (B, T, H, W, C) -> (B, T//8, H, W, C)
    spikes = np.packbits(spikes, axis=1)
    
    return spikes, labels


def create_loader(
    source: grain.sources.RandomAccessDataSource,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    drop_remainder: bool = True,
    worker_count: int = 0,
    transform_fn: Optional[Callable] = None,
):
    """
    Creates a Grain DataLoader for a given data source.
    
    Args:
        source: A Grain RandomAccessDataSource
        batch_size: The batch size
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling
        drop_remainder: Whether to drop the last batch if it's smaller than batch_size
        worker_count: Number of child processes for data loading
        transform_fn: Optional function to apply to each batch
    
    Returns:
        A Grain DataLoader
    """
    
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, 
            shard_count=1, 
            drop_remainder=drop_remainder
        )
    )
    
    operations = [
        grain.transforms.Batch(batch_size=batch_size, drop_remainder=drop_remainder)
    ]

    if transform_fn:
        class _Map(grain.transforms.Map):
            def map(self, batch):
                return transform_fn(batch)
        
        operations.append(_Map())
        
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
    )
    
    return loader


class NMNIST_loader:
    """
    Grain-based Neuromorphic MNIST dataloader using Tonic (without torch dependencies).
    
    Args:
        batch_size: Number of samples per batch
        sample_T: Number of time bins
        val_size: Fraction of training set for validation
        data_subsample: Fraction of data to use
        seed: Random seed
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        sample_T: int = 40,
        val_size: float = 0.3,
        data_subsample: float = 1.0,
        seed: int = 0,
        download_dir: str = './data/NMNIST'
    ):
        if not tonic_available:
            raise ImportError("Tonic is required for NMNIST loader. Install with: pip install tonic")
        
        self.batch_size = batch_size
        self.sample_T = sample_T
        self.obs_shape = datasets.NMNIST.sensor_size
        self.act_shape = (10,)
        
        # Tonic transform to convert events to frames
        transform = tonic.transforms.Compose([
            tonic.transforms.ToFrame(
                sensor_size=self.obs_shape,
                n_time_bins=sample_T
            ),
            # Pack bits for memory efficiency
            lambda x: np.packbits(x, axis=0)
        ])
        
        # Load datasets
        train_dataset = datasets.NMNIST(
            download_dir, 
            train=True, 
            first_saccade_only=True,
            transform=transform
        )
        test_dataset = datasets.NMNIST(
            download_dir,
            train=False,
            first_saccade_only=True,
            transform=transform
        )
        
        # Create train/val split
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=val_size,
            random_state=seed,
            shuffle=True
        )
        
        # Subsample if requested
        if data_subsample < 1.0:
            train_indices = train_indices[:int(len(train_indices) * data_subsample)]
            val_indices = val_indices[:int(len(val_indices) * data_subsample)]
        
        # Create index-based subsets using simple wrapper
        class IndexSubset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        train_subset = IndexSubset(train_dataset, train_indices)
        val_subset = IndexSubset(train_dataset, val_indices)
        
        # Create data sources
        self.train_source = TonicDataSource(train_subset)
        self.val_source = TonicDataSource(val_subset)
        self.test_source = TonicDataSource(test_dataset)
        
        self.train_len = len(self.train_source)
        self.val_len = len(self.val_source)
        self.test_len = len(self.test_source)
        
        def transform_batch(index, batch):
            """Collate batch of Tonic samples."""
            # batch is a list of (events, label) tuples
            events = np.stack([item[0] for item in batch])
            labels = np.array([item[1] for item in batch])
            return events.astype(np.uint8), labels.astype(np.int32)
        
        self.transform_batch = transform_batch

    def train_loader(self, worker_count: int = 0):
        return create_loader(
            self.train_source,
            self.batch_size,
            shuffle=True,
            worker_count=worker_count,
            transform_fn=self.transform_batch
        )

    def val_loader(self, worker_count: int = 0):
        return create_loader(
            self.val_source,
            self.batch_size,
            shuffle=False,
            worker_count=worker_count,
            transform_fn=self.transform_batch
        )

    def test_loader(self, worker_count: int = 0):
        return create_loader(
            self.test_source,
            self.batch_size,
            shuffle=False,
            worker_count=worker_count,
            transform_fn=self.transform_batch
        )


class SHD_loader:
    """
    Grain-based Spiking Heidelberg Dataset dataloader using Tonic.
    
    Args:
        batch_size: Number of samples per batch
        sample_T: Number of time steps per sample
        channels: Number of frequency channels
        val_size: Fraction of training set for validation
        seed: Random seed
    """
    
    def __init__(
        self,
        batch_size: int = 256,
        sample_T: int = 128,
        channels: int = 128,
        val_size: float = 0.2,
        seed: int = 0,
        download_dir: str = './data/SHD'
    ):
        if not tonic_available:
            raise ImportError("Tonic is required for SHD loader.")
        
        self.batch_size = batch_size
        self.sample_T = sample_T
        self.obs_shape = (channels,)
        self.act_shape = (20,)
        
        # SHD-specific parameters
        shd_timestep = 1e-6
        shd_channels = 700
        net_dt = 1 / sample_T
        
        # Custom rasterizer for SHD
        class SHD2Raster:
            def __init__(self, encoding_dim, sample_T):
                self.encoding_dim = encoding_dim
                self.sample_T = sample_T
            
            def __call__(self, events):
                tensor = np.zeros((events["t"].max() + 1, self.encoding_dim), dtype=int)
                np.add.at(tensor, (events["t"], events["x"]), 1)
                tensor = tensor[:self.sample_T, :]
                tensor = np.minimum(tensor, 1)
                tensor = np.packbits(tensor, axis=0)
                return tensor
        
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(
                time_factor=shd_timestep / net_dt,
                spatial_factor=channels / shd_channels
            ),
            SHD2Raster(channels, sample_T=sample_T)
        ])
        
        # Load datasets
        train_dataset = datasets.SHD(download_dir, train=True, transform=transform)
        test_dataset = datasets.SHD(download_dir, train=False, transform=transform)
        
        # Create train/val split
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=val_size,
            random_state=seed,
            shuffle=True
        )
        
        # Create index-based subsets
        class IndexSubset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        train_subset = IndexSubset(train_dataset, train_indices)
        val_subset = IndexSubset(train_dataset, val_indices)
        
        # Create data sources
        self.train_source = TonicDataSource(train_subset)
        self.val_source = TonicDataSource(val_subset)
        self.test_source = TonicDataSource(test_dataset)
        
        self.train_len = len(self.train_source)
        self.val_len = len(self.val_source)
        self.test_len = len(self.test_source)
        
        def transform_batch(index, batch):
            """Collate batch of SHD samples."""
            events = np.stack([item[0] for item in batch])
            labels = np.array([item[1] for item in batch])
            return events.astype(np.uint8), labels.astype(np.int32)
        
        self.transform_batch = transform_batch

    def train_loader(self, worker_count: int = 0):
        return create_loader(
            self.train_source,
            self.batch_size,
            shuffle=True,
            worker_count=worker_count,
            transform_fn=self.transform_batch
        )

    def val_loader(self, worker_count: int = 0):
        return create_loader(
            self.val_source,
            self.batch_size,
            shuffle=False,
            worker_count=worker_count,
            transform_fn=self.transform_batch
        )

    def test_loader(self, worker_count: int = 0):
        return create_loader(
            self.test_source,
            self.batch_size,
            shuffle=False,
            worker_count=worker_count,
            transform_fn=self.transform_batch
        )
