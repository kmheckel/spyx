import jax
import jax.numpy as jnp
import haiku as hk



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
    This function implements the SuperSpike surrogate gradient activation function for a spiking neuron.
    
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


# This could also just be changed to be a function which yields the proper VJP func...
def Axon(bwd=jax.jit(lambda x: x), 
         fwd=jax.jit(lambda x: jnp.heaviside(x,0))):
    """
    This function serves as the activation function for the SNNs, allowing for custom definitions of both surrogate gradients for backwards
    passes as well as substitution of the Heaviside function for relaxations such as sigmoids. 

    In short, this function takes functions that define the backward and forward passes for the spiking activation function and fuses them
    into a single function using VJP, allowing for surrogate gradient methods as well as for using alternative forward activations when performing
    neuroevolution or ANN to SNN conversion. The user can also define and JIT their own function to use as a surrogate, or
    even use more dynamic functions that change over the course of training.

    The default behavior is a Heaviside forward activation with a stragiht through estimator surrogate gradient.

    Attributes:
        bwd: Function that calculates the gradient to be used in the backwards pass.
        fwd: Function that returns a value between 0 and 1. Default is Heaviside.
     
    """
        
    @jax.custom_vjp
    def f(U): # primal function
        return fwd(U)
        
    # returns value, grad context
    def f_fwd(U):
        return f(U), U
            
    # accepts context, primal val
    def f_bwd(U, grad):
        return (grad * bwd(U) , )
            
    f.defvjp(f_fwd, f_bwd)
    
    return jax.jit(f)