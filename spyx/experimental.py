import jax
import jax.numpy as jnp
import haiku as hk

def _SigmoidBernoulli():
    """
    Experimental! This builds a Sigmoid Bernoulli activation function with STE gradient function.

    Will eventually be removed in favor of a more general stochastic Axon construct. (most probably.)

    The construction is a tad awkward because the custom VJP requires returning Nones for the rng gradient 
    (would be nice if it could be ignored but sadly traced values can't be set as nondiff according to the docs.)
    """
        
    @jax.custom_vjp
    def f(U, key): # primal function
        return jax.random.bernoulli(key, U) * jnp.ones_like(U), None
        
    # returns value, grad context
    def f_fwd(U, key):
        return f(U, key), U
            
    # accepts context, primal val
    def f_bwd(U, grad):
        return (grad[0] * U , None )
            
    f.defvjp(f_fwd, f_bwd)
    
    return jax.jit(f)



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
        self.spike_fn = _SigmoidBernoulli()
        self.k = k
    
    def __call__(self, key, x):
        """
        :x: input tensor coming from previous layer. [Batch, Time, channels ...]
        :key: JAX PRNGKey for computing stochastic spikes based on voltage potentials.

        """

        beta = hk.get_parameter("beta", self.hidden_shape,
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
        beta = jnp.minimum(jax.nn.relu(beta),1)

        B = jnp.power(beta, jnp.arange(x.shape[1])) * (1-beta)

        fft_B = jnp.expand_dims(jnp.fft.rfft(B, n=2*x.shape[1]), 1)

        fft_X = jnp.fft.rfft(x, n=2*x.shape[1], axis=1)

        V = jnp.fft.irfft(fft_X*fft_B, n=2*x.shape[1], axis=1)[:,:x.shape[1]:,]

        # calculate whether spike is generated, and update membrane potential
        spikes, _ = self.spike_fn(jax.nn.sigmoid(self.k*(V-self.threshold)), key)
        
        return spikes, V
