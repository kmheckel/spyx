import jax
import jax.numpy as jnp
import haiku as hk

# This entire file should be restructured
# There should be a single synapse class that has a default surrogate gradient
# of straight through estimation and then the constructor can take an argument
# to override that with either a premade surrogate such as tanh, sigmoid, etc
# or a user defined function. This would allow easier experimentation as 
# You could just define a lambda function and pass it in rather than rewriting
# a whole class to redefine the gradient function. each of the surrogate classes 
# should be replaced by second order functions that take the parameters of 
# their respective function and return the associated JIT'ed func.
# 
# This would also open the door up to easier parameterization of surrogates,
# facilitating evolution/meta learning. 

class ActivityRegularization(hk.Module):
    """
    Add state to the SNN to track the average number of spikes emitted per neuron per batch.

    Adding this to a network requires using the Haiku transform_with_state transform, which will also return an initial regularization state vector.
    This blank initial vector can be reused and is provided as the second arg to the SNN's apply function. 
    """

    def __init__(self, name="ActReg"):
        super().__init__(name=name)
        
    def __call__(self, spikes):
        spike_count = hk.get_state("spike_count", spikes.shape, init=jnp.zeros, dtype=spikes.dtype)
        hk.set_state("spike_count", spike_count + spikes) 
        return spikes



def tanh(k=1):
    """
        Hyperbolic Tangent activation. Very simple.

    """
    def g(x):
        kx = k * x
        return 4 / (jnp.exp(-kx) + jnp.exp(kx))**2
    return jax.jit(g)





def ptanh(a=2, b=25):
    """
        Parameterized Hyperbolic Tangent activation.

        \frac{e^{x/a} - e^{-x/b}}{e^{ax} + e^{-bx}}

    """
    def g(x):
        exa = jnp.exp(x/a)
        exb = jnp.exp(-x/b)
        eax = jnp.exp(a*x)
        ebx = jnp.exp(-b*x)

        term1 = ( (exa/a) + (exb/b) ) / (eax + ebx)
        term2 = ( (exa-exb) * ((a*eax) - (b*ebx)) ) / (eax+ebx)**2
        return term1 - term2

    return jax.jit(g)




def boxcar(width=1, height=0.5):
    """
        Boxcar activation. Very simple.

    """
    k = width / 2
    h = height
    def g(x):
        return h * jnp.heaviside(-(jnp.abs(x) - k), 0)
    return jax.jit(g)




def triangular(k=0.5):
    """
        Triangular activation inspired by Esser et. al. Very simple. https://arxiv.org/abs/1603.08270

    """
    def g(x):
        return jnp.maximum(0, 1-jnp.abs(k*x))
    return jax.jit(g)





def arctan(k=2):
    """
    This class implements the Arctangent surrogate gradient activation function for a spiking neuron.
    
    The Arctangent function is a smooth function that approximates the step function. 
    It is used as a surrogate gradient for the step function in the context of spiking neurons. 
    The surrogate gradient is used during the backpropagation process to update the weights of the neuron.
    
    The Arctangent function returns a value between -pi/2 and pi/2 for inputs in the range of -Infinity to Infinity.
    It is often used in the context of spiking neurons because it provides a smooth approximation to the step function 
    that is differentiable everywhere, which is a requirement for gradient-based optimization methods.
    
    Attributes:
        scale_factor: A scaling factor that can be used to adjust the steepness of the 
                      Arctangent function. Default is 2.
    """
    def g(U):
        x = jnp.pi * U * k
        return (1 / (1+x**2)) / jnp.pi

    return jax.jit(g)




def sigmoid(k=4):
    """
    This class implements the Sigmoid surrogate gradient activation function for a spiking neuron.
    
    The Sigmoid function is a smooth function that approximates the step function. 
    It is used as a surrogate gradient for the step function in the context of spiking neurons. 
    The surrogate gradient is used during the backpropagation process to update the weights of the neuron.
    
    The Sigmoid function returns a value between 0 and 1 for inputs in the range of -Infinity to Infinity.
    It is often used in the context of spiking neurons because it provides a smooth approximation to the step function 
    that is differentiable everywhere, which is a requirement for gradient-based optimization methods.
    
    Attributes:
        scale_factor: A scaling factor that can be used to adjust the steepness of the 
                      Sigmoid function. Default is 4.
    """
    def g(x):
        kx = -k * x
        num = k * jnp.exp(kx)
        den = (jnp.exp(kx)+1)**2
        return num / den
    return jax.jit(g)


def superspike(k=25):
    """
    This class implements the SuperSpike surrogate gradient activation function for a spiking neuron.
    
    The SuperSpike function is a smooth function that approximates the step function. 
    It is used as a surrogate gradient for the step function in the context of spiking neurons. 
    The surrogate gradient is used during the backpropagation process to update the weights of the neuron.
    
    The SuperSpike function is defined as 1/(1+k|U|)^2, where U is the input to the function and k is a scaling factor.
    It returns a value between 0 and 1 for inputs in the range of -Infinity to Infinity.
    
    It is often used in the context of spiking neurons because it provides a smooth approximation to the step function 
    that is differentiable everywhere, which is a requirement for gradient-based optimization methods.

    It is a fast approximation of the Sigmoid function adapted from:

    F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.
    
    Attributes:
        scale_factor: A scaling factor that can be used to adjust the steepness of the 
                      SuperSpike function. Default is 25.
    """

    def g(x):
        return 1 / (1 + k*jnp.abs(x))**2
    return jax.jit(g)


class Axon:

    def __init__(self, bwd=jax.jit(lambda x: x), fwd=jnp.heaviside):
        self._grad = bwd
        
        @jax.custom_vjp
        def f(U): # primal function
            return fwd(U,0)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), U
            
        # accepts context, primal val
        def f_bwd(U, grad):
            return (grad * self._grad(U) , )
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U)