import jax
import jax.numpy as jnp
import haiku as hk

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

class PSU_LIF(hk.Module):

    def __init__(self, hidden_shape, threshold=1, k=2, spike=True, name="PSULIF"):
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        if spike:
            self.spike = arctan(k)
        else:
            self.spike = lambda x: x

    # x.shape = B, T, C
    def __call__(self, x):
        # Beta is our learnable neuron time constant / the diagonal operator.
        beta = hk.get_parameter("beta", self.hidden_shape,
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
        beta = jnp.clip(beta, 0, 1)

        _, V = jax.vmap(_pscan, in_axes=(None,0))(beta, x)

        _, R = jax.vmap(_pscan, in_axes=(None,0))(beta, jax.nn.sigmoid(V))



        return self.spike(V - R - self.threshold), V

class StochasticAssociativeLIF(hk.Module):

    def __init__(self, hidden_shape, threshold=1, k=100, spike=True, name="SALIF"):
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        if spike:
            self.spike = sigmoid_bernoulli(k, threshold)
        else:
            self.spike = lambda x, k: x

    # x.shape = B, T, C
    def __call__(self, key, x):
        # Beta is our learnable neuron time constant / the diagonal operator.
        beta = hk.get_parameter("beta", self.hidden_shape,
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
        beta = jnp.clip(beta, 0, 1)

        _, V = jax.vmap(_pscan, in_axes=(None,0))(beta, x)

        return self.spike(V, key), V

# prototype / proof of concept
class StochasticAssociativeCuBaLIF(hk.Module):

    def __init__(self, hidden_shape, threshold=1, k=100, name="SACuBaLIF"):
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.spike = refractory_sigmoid_bernoulli(k, threshold)

    def __call__(self, key, u):
        # Beta is our learnable neuron time constant / the diagonal operator.
        alpha = hk.get_parameter("alpha", self.hidden_shape,
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
        alpha = jnp.clip(alpha, 0, 1)
        
        beta = hk.get_parameter("beta", self.hidden_shape,
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
        beta = jnp.clip(beta, 0, 1)

        # this can probably be condensed.
        _, x = jax.vmap(_pscan, in_axes=(None,0))(alpha, u)
        _, V = jax.vmap(_pscan, in_axes=(None,0))(beta, x)
 
        return self.spike(V, key)

class SPSN(hk.Module):
    """
    Prototype implementation of Stochastic Parallelizable Spiking Neuron:

    https://doi.org/10.48550/arXiv.2306.12666

    Currently only has Sigmoid-Bernoulli as a firing activation option. Needs further debugging.

    Currently the output layer can only have a single beta value, and it tends to optimize to 0 which is odd.
    
    """

    def __init__(self, hidden_shape: tuple, threshold=1, k=10,
                 name="SPSN"):

        """
        
        :hidden_shape: Size of hidden layer for the number of Voltages to track.
        :threshold: Value for which probability of firing exceeds 50%
        :k: The slope of the sigmoid function, the higher the value the closer membrane voltage must to the threshold to have a chance of firing but also a higher chance of continuous firing

        """
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = sigmoid_bernoulli(k, threshold)
    
    def __call__(self, key, x):
        """
        :x: input tensor coming from previous layer. [Batch, Time, channels ...]
        :key: JAX PRNGKey for computing stochastic spikes based on voltage potentials.

        """

        beta = hk.get_parameter("beta", self.hidden_shape,
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
        beta = jnp.clip(beta, 0, 1)

        B = jnp.power(beta, jnp.arange(x.shape[1])) * (1-beta)

        fft_B = jnp.expand_dims(jnp.fft.rfft(B, n=2*x.shape[1]), 1)

        fft_X = jnp.fft.rfft(x, n=2*x.shape[1], axis=1)

        V = jnp.fft.irfft(fft_X*fft_B, n=2*x.shape[1], axis=1)[:,:x.shape[1]:,]

        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(key, V)
        
        return spikes, V
