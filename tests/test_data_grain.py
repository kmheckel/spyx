import jax.numpy as jnp
import numpy as np

from spyx import data


def test_rate_code_transform():
    sample_T = 10
    max_r = 1.0
    transform = data.RateCode(sample_T=sample_T, max_r=max_r)
    
    # Input image (2, 2)
    obs = np.array([[0.5, 0.1], [0.9, 0.0]])
    record = {"obs": obs}
    
    new_record = transform.map(record)
    
    # Output should be packed bits
    # packed shape: (ceil(sample_T/8), 2, 2)
    expected_packed_T = (sample_T + 7) // 8
    assert new_record["obs"].shape == (expected_packed_T, 2, 2)
    assert new_record["obs"].dtype == np.uint8

def test_shift_augment_transform():
    transform = data.ShiftAugment(max_shift=2, axes=(0, 1))
    
    # (H, W) = (4, 4)
    obs = np.random.rand(4, 4)
    record = {"obs": obs.copy()}
    
    new_record = transform.map(record)
    
    # Shape should be same
    assert new_record["obs"].shape == (4, 4)
    # Values should be different (most likely, but random)
    # We just check it ran without error for now as verification of shift logic 
    # is usually visual or statistical.

def test_angle_code_transform():
    neuron_count = 8
    transform = data.AngleCode(neuron_count=neuron_count, min_val=0, max_val=1)
    
    obs = np.array([0.0, 0.5, 1.0]) # 3 elements
    record = {"obs": obs}
    
    new_record = transform.map(record)
    
    # Output should be (3, 8) one-hot
    assert new_record["obs"].shape == (3, 8)
    # Based on refined Spyx behavior:
    # 0.0 maps to index 0
    # 0.5 maps to index 3
    # 1.0 maps to index 7
    assert new_record["obs"][0, 0] == 1
    assert new_record["obs"][1, 3] == 1
    assert new_record["obs"][2, 7] == 1
    # Check that everything has exactly one spike
    assert np.all(np.sum(new_record["obs"], axis=-1) == 1)

# ---------------------------------------------------------------------------
# Latency / time-to-first-spike encoding (issue #21)
# ---------------------------------------------------------------------------


def test_latency_code_functional_bright_fires_early():
    T = 8
    encoder = data.latency_code(num_steps=T, threshold=-1.0)  # disable silencing
    x = jnp.array([1.0, 0.5, 0.1])  # bright / mid / dim
    spikes = encoder(x)  # [T, C]
    assert spikes.shape == (T, 3)
    assert spikes.dtype == jnp.uint8
    # Every unit fires exactly once (no silencing with threshold < 0).
    assert jnp.all(jnp.sum(spikes, axis=0) == 1)
    # Brighter input fires earlier.
    times = jnp.argmax(spikes, axis=0)
    assert int(times[0]) == 0              # 1.0 -> t=0
    assert int(times[0]) < int(times[1])   # 1.0 before 0.5
    assert int(times[1]) < int(times[2])   # 0.5 before 0.1


def test_latency_code_threshold_silences_subthreshold_units():
    encoder = data.latency_code(num_steps=8, threshold=0.2)
    x = jnp.array([0.1, 0.3])
    spikes = encoder(x)
    # Below-threshold unit never fires, above-threshold unit fires once.
    assert int(jnp.sum(spikes[:, 0])) == 0
    assert int(jnp.sum(spikes[:, 1])) == 1


def test_latency_code_grain_transform():
    transform = data.LatencyCode(sample_T=8, threshold=-1.0)  # disable silencing
    obs = np.array([[1.0, 0.5], [0.1, 0.0]])
    new = transform.map({"obs": obs})
    # Packed bits along axis 0: ceil(8 / 8) = 1.
    assert new["obs"].shape == (1, 2, 2)
    assert new["obs"].dtype == np.uint8
    unpacked = np.unpackbits(new["obs"], axis=0)[:8]
    # One spike per unit.
    assert np.all(np.sum(unpacked, axis=0) == 1)
    # Brightest pixel (1.0) fires at t=0, dimmest (0.0) at t=7.
    assert unpacked[0, 0, 0] == 1
    assert unpacked[7, 1, 1] == 1


if __name__ == "__main__":
    try:
        test_rate_code_transform()
        print("RateCode transform test passed!")
    except Exception as e:
        print(f"RateCode transform test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_shift_augment_transform()
        print("ShiftAugment transform test passed!")
    except Exception as e:
        print(f"ShiftAugment transform test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_angle_code_transform()
        print("AngleCode transform test passed!")
    except Exception as e:
        print(f"AngleCode transform test failed: {e}")
        import traceback
        traceback.print_exc()
