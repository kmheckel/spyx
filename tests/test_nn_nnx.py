from flax import nnx
import jax
import jax.numpy as jnp
from spyx import nn
import pytest

def test_lif():
    rngs = nnx.Rngs(0)
    hidden_shape = (10,)
    model = nn.LIF(hidden_shape, rngs=rngs)
    
    x = jnp.ones((5, 10)) # batch of 5
    V = model.initial_state(5)
    
    spikes, V_next = model(x, V)
    
    assert spikes.shape == (5, 10)
    assert V_next.shape == (5, 10)
    assert jnp.any(spikes == 0) # Initially no spikes if threshold is 1 and dynamic V starts at 0

def test_rlif():
    rngs = nnx.Rngs(0)
    hidden_shape = (10,)
    model = nn.RLIF(hidden_shape, rngs=rngs)
    
    x = jnp.ones((5, 10))
    V = model.initial_state(5)
    
    spikes, V_next = model(x, V)
    
    assert spikes.shape == (5, 10)
    assert V_next.shape == (5, 10)

def test_activity_reg():
    model = nn.ActivityRegularization()
    spikes = jnp.array([[0, 1], [1, 0]])
    
    out = model(spikes)
    assert jnp.array_equal(out, spikes)
    assert jnp.array_equal(model.spike_count[...], spikes)
    
    out = model(spikes)
    assert jnp.array_equal(model.spike_count[...], spikes * 2)

def test_sequential_run():
    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nnx.Linear(10, 20, rngs=rngs),
        nn.LIF((20,), rngs=rngs),
        nnx.Linear(20, 10, rngs=rngs),
        nn.LIF((10,), rngs=rngs)
    )
    
    x = jnp.ones((32, 5, 10)) # [T, B, C]
    
    outputs, final_state = nn.run(model, x)
    
    assert outputs.shape == (32, 5, 10)
    assert len(final_state) == 4
    assert final_state[1].shape == (5, 20)
    assert final_state[3].shape == (5, 10)

if __name__ == "__main__":
    test_lif()
    test_rlif()
    test_activity_reg()
    test_sequential_run()
    print("Tests passed!")
