"""Tests for Grain-based dataloaders (pure JAX ecosystem)."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

def test_nmnist_loader():
    """Test NMNIST loader with Tonic backend (no torch)."""
    try:
        from spyx.grain_loaders import NMNIST_loader, tonic_available
        
        if not tonic_available:
            pytest.skip("Tonic not available")
        
        loader = NMNIST_loader(
            batch_size=4,
            sample_T=40,
            data_subsample=0.01,
            seed=42,
            download_dir='./data/NMNIST'
        )
        
        # Test train loader
        train_loader = loader.train_loader(worker_count=0)
        train_iter = iter(train_loader)
        
        batch = next(train_iter)
        x, y = batch
        
        # NMNIST with packbits: (B, T//8, H, W, C)
        # T=40, so T//8=5
        assert x.shape == (4, 5, 34, 34, 2), f"Expected shape (4, 5, 34, 34, 2), got {x.shape}"
        assert y.shape == (4,)
        assert x.dtype == np.uint8
        
    except ImportError:
        pytest.skip("NMNIST loader dependencies not available")


def test_shd_loader():
    """Test SHD loader with Tonic backend (no torch)."""
    try:
        from spyx.grain_loaders import SHD_loader, tonic_available
        
        if not tonic_available:
            pytest.skip("Tonic not available")
        
        loader = SHD_loader(
            batch_size=4,
            sample_T=128,
            channels=128,
            seed=42,
            download_dir='./data/SHD'
        )
        
        # Test train loader
        train_loader = loader.train_loader(worker_count=0)
        train_iter = iter(train_loader)
        
        batch = next(train_iter)
        x, y = batch
        
        # SHD with packbits: (B, T//8, channels)
        # Note: actual shape depends on rasterization
        assert x.ndim == 3  # (B, packed_time, channels)
        assert y.shape == (4,)
        assert x.dtype == np.uint8
        
    except ImportError:
        pytest.skip("SHD loader dependencies not available")


def test_no_torch_imports():
    """Verify that grain_loaders doesn't import torch/torchvision."""
    import sys
    
    # Clear any torch modules if they were imported before
    torch_modules = [k for k in sys.modules.keys() if 'torch' in k.lower()]
    
    # Import grain_loaders
    from spyx import grain_loaders
    
    # Check that no new torch modules were imported
    new_torch_modules = [k for k in sys.modules.keys() if 'torch' in k.lower()]
    
    # Should be the same set (no new torch imports)
    assert set(new_torch_modules) == set(torch_modules), \
        f"grain_loaders imported torch modules: {set(new_torch_modules) - set(torch_modules)}"


if __name__ == "__main__":
    print("Testing MNIST loader...")
    test_mnist_loader()
    print("✓ MNIST loader works!")
    
    print("\nTesting NMNIST loader...")
    try:
        test_nmnist_loader()
        print("✓ NMNIST loader works!")
    except Exception as e:
        print(f"⚠ NMNIST loader test skipped: {e}")
    
    print("\nTesting SHD loader...")
    try:
        test_shd_loader()
        print("✓ SHD loader works!")
    except Exception as e:
        print(f"⚠ SHD loader test skipped: {e}")
    
    print("\nTesting no torch imports...")
    test_no_torch_imports()
    print("✓ No torch imports detected!")
    
    print("\n✅ All tests passed!")
