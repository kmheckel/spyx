import jax
import jax.numpy as jnp
import haiku as hk
import spyx
import pytest

neurons = [
    spyx.nn.ALIF,
    spyx.nn.LI,
    spyx.nn.IF,
    spyx.nn.LIF,
    spyx.nn.CuBaLIF,
    spyx.nn.RIF,
    spyx.nn.RLIF,
    spyx.nn.RCuBaLIF
]

@pytest.mark.parametrize("Neuron", neurons)
def test_neuron_forward(Neuron):
    hidden_shape = (32,)
    batch_size = 4
    
    def forward(x):
        neuron = Neuron(hidden_shape)
        state = neuron.initial_state(batch_size)
        return neuron(x, state)

    x = jnp.ones((batch_size,) + hidden_shape)
    
    # Transform
    f = hk.transform(forward)
    rng = jax.random.PRNGKey(0)
    params = f.init(rng, x)
    
    out, state = f.apply(params, rng, x)
    
    assert out.shape == (batch_size,) + hidden_shape











