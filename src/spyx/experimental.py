import jax
import jax.numpy as jnp
from flax import nnx

from .axn import arctan


def sigmoid_bernoulli(k=10, threshold=1., max_prob=0.8):

    @jax.custom_gradient
    def activation(x, key):
        U = x - threshold
        p_n = jax.nn.sigmoid(k*U) * max_prob
        return jax.random.bernoulli(key, p_n).astype(U.dtype), lambda g: (g * p_n, None)

    return activation

def refractory_sigmoid_bernoulli(k=50, threshold=1):

    freq = 2 * jnp.pi * threshold

    @jax.custom_gradient
    def activation(x, key):
        U = x - threshold
        r = jnp.cos(freq * U)
        s = jax.nn.sigmoid(k*U)
        p_n = jnp.maximum(r * s, 0)
        return jax.random.bernoulli(key, p_n).astype(U.dtype), lambda g: (g * p_n, None)

    return activation

# from S5 paper, simplified structured state space models
def _binary_operator(element_i, element_j):

    A_i, Bu_i = element_i
    A_j, Bu_j = element_j

    return A_j * A_i, A_j * Bu_i + Bu_j

def _pscan(tau, x):
    tau =  jnp.repeat(tau[None, ...], x.shape[0], axis=0)
    return jax.lax.associative_scan(_binary_operator, (tau, x))

class PSU_LIF(nnx.Module):

    def __init__(self, hidden_shape, threshold=1, k=2, spike=True, *, rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        if spike:
            self.spike = arctan(k)
        else:
            self.spike = lambda x: x
            
        self.beta = nnx.Param(
            nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
        )

    # x.shape = B, T, C
    def __call__(self, x):
        beta = jnp.clip(self.beta[:], 0, 1)

        _, V = jax.vmap(_pscan, in_axes=(None, 0))(beta, x)
        _, R = jax.vmap(_pscan, in_axes=(None, 0))(beta, jax.nn.sigmoid(V))

        return self.spike(V - R - self.threshold), V

class StochasticAssociativeLIF(nnx.Module):

    def __init__(self, hidden_shape, threshold=1, k=100, spike=True, *, rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        if spike:
            self.spike = sigmoid_bernoulli(k, threshold)
        else:
            self.spike = lambda x, k: x
            
        self.beta = nnx.Param(
            nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
        )

    # x.shape = B, T, C
    def __call__(self, key, x):
        beta = jnp.clip(self.beta[:], 0, 1)

        _, V = jax.vmap(_pscan, in_axes=(None, 0))(beta, x)

        return self.spike(V, key), V

# prototype / proof of concept
class StochasticAssociativeCuBaLIF(nnx.Module):

    def __init__(self, hidden_shape, threshold=1, k=100, *, rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.spike = refractory_sigmoid_bernoulli(k, threshold)
        
        self.alpha = nnx.Param(
            nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
        )
        self.beta = nnx.Param(
            nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
        )

    def __call__(self, key, u):
        alpha = jnp.clip(self.alpha[:], 0, 1)
        beta = jnp.clip(self.beta[:], 0, 1)

        # this can probably be condensed.
        _, x = jax.vmap(_pscan, in_axes=(None, 0))(alpha, u)
        _, V = jax.vmap(_pscan, in_axes=(None, 0))(beta, x)
 
        return self.spike(V, key)

class SPSN(nnx.Module):
    """
    Prototype implementation of Stochastic Parallelizable Spiking Neuron:

    https://doi.org/10.48550/arXiv.2306.12666
    """

    def __init__(self, hidden_shape: tuple, threshold=1, k=10, *, rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = sigmoid_bernoulli(k, threshold)
        
        self.beta = nnx.Param(
            nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
        )
    
    def __call__(self, key, x):
        beta = jnp.clip(self.beta[:], 0, 1)

        B = jnp.power(beta, jnp.arange(x.shape[1])) * (1 - beta)

        fft_B = jnp.expand_dims(jnp.fft.rfft(B, n=2 * x.shape[1]), 1)
        fft_X = jnp.fft.rfft(x, n=2 * x.shape[1], axis=1)

        V = jnp.fft.irfft(fft_X * fft_B, n=2 * x.shape[1], axis=1)[:, :x.shape[1] :,]

        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(key, V)
        
        return spikes, V
