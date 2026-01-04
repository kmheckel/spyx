from spyx import data


def test_nmnist_loader():
    batch_size = 4
    sample_T = 16
    loader = data.NMNIST_loader(batch_size=batch_size, sample_T=sample_T)
    
    assert loader.obs_shape == (2, 34, 34)
    
    train_dl = loader.train_epoch()
    batch = next(train_dl)
    print("NMNIST batch shape:", batch.obs.shape)
    
    # NMNIST: (batch_size, packed_T, 2, 34, 34)
    # packed_T = (sample_T + 7) // 8
    assert batch.obs.shape == (batch_size, (sample_T + 7) // 8, 2, 34, 34)
    assert batch.labels.shape == (batch_size,)

def test_shd_loader():
    batch_size = 2
    sample_T = 32
    channels = 128
    loader = data.SHD_loader(batch_size=batch_size, sample_T=sample_T, channels=channels)
    
    assert loader.obs_shape == (channels,)
    
    train_dl = loader.train_epoch()
    batch = next(train_dl)
    
    # SHD: (batch_size, packed_T, channels)
    assert batch.obs.shape == (batch_size, (sample_T + 7) // 8, channels)
    assert batch.labels.shape == (batch_size,)

if __name__ == "__main__":
    try:
        test_nmnist_loader()
        print("NMNIST loader test passed!")
    except Exception as e:
        print(f"NMNIST loader test failed: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        test_shd_loader()
        print("SHD loader test passed!")
    except Exception as e:
        print(f"SHD loader test failed: {e}")
        import traceback
        traceback.print_exc()
