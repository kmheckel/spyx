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
    """Hyperbolic Tangent activation.

    .. math:: 4 / (e^{-kx} + e^{kx})^2

    :k: Value for scaling the slope of the surrogate gradient.
    :return: JIT compiled tanh surrogate gradient function.
    """
    def g(x):
        kx = k * x
        return 4 / (jnp.exp(-kx) + jnp.exp(kx))**2
    return jax.jit(g)




def boxcar(width=2, height=0.5):
    """Boxcar activation.

    :width: Total width of non-zero gradient flow, centered on 0.
    :height: Value for gradient within the specified window.
    :return: JIT compiled boxcar surrogate gradient function.
    """
    k = width / 2
    h = height
    def g(x):
        return h * jnp.heaviside(-(jnp.abs(x) - k), 0)
    return jax.jit(g)




def triangular(k=0.5):
    """
    Triangular activation inspired by Esser et. al. https://arxiv.org/abs/1603.08270

    .. math::
        max(0, 1-|kx|)

        
    :k: scale factor
    :return: JIT compiled triangular surrogate gradient function.
    """
    def g(x):
        return jnp.maximum(0, 1-jnp.abs(k*x))
    return jax.jit(g)





def arctan(k=2):
    """
    This class implements the Arctangent surrogate gradient activation function for a spiking neuron.
    
    The Arctangent function returns a value between -pi/2 and pi/2 for inputs in the range of -Infinity to Infinity.
    It is often used in the context of spiking neurons because it provides a smooth approximation to the step function 
    that is differentiable everywhere, which is a requirement for gradient-based optimization methods.
    
    :k: A scaling factor that can be used to adjust the steepness of the 
                      Arctangent function. Default is 2.
    :return: JIT compiled arctangent-derived surrogate gradient function.
    """
    def g(U):
        x = jnp.pi * U * k
        return (1 / (1+x**2)) / jnp.pi

    return jax.jit(g)




def sigmoid(k=4):
    """
    This class implements the Sigmoid surrogate gradient activation function for a spiking neuron.
    
    The Sigmoid function returns a value between 0 and 1 for inputs in the range of -Infinity to Infinity.
    It is often used in the context of spiking neurons because it provides a smooth approximation to the step function 
    that is differentiable everywhere, which is a requirement for gradient-based optimization methods. 
    As a surrogate gradient, the derivative of sigmoid family functions are used as substitutes for the non-differentiable Heaviside.
    
    
    :k: A scaling factor that can be used to adjust the steepness of the 
                      Sigmoid function. Default is 4.
    :return: JIT compiled sigmoid-derived surrogate gradient function. 
    """
    sig = jax.grad(jax.nn.sigmoid)
    def g(x):
        v = sig # development found a really weird numerical instability with computing the derivative outright
        for axis in x.shape: # where the model wouldn't learn, but using jax.grad of the sigmoid function
            v = jax.vmap(v) # would learn despite the differences being < 1e-7 without a bias +/1.
        return v(4*x) # to make the grad transformed func work on any shape input, we have to use this ugly
    return jax.jit(g) # hack which vmaps to accomodate the shape. Luckily, it appears to not impact the compiled func.



def superspike(k=25):
    """
    This function implements the SuperSpike surrogate gradient activation function for a spiking neuron.
    
    The SuperSpike function is defined as 1/(1+k|U|)^2, where U is the input to the function and k is a scaling factor.
    It returns a value between 0 and 1 for inputs in the range of -Infinity to Infinity.
    
    It is often used in the context of spiking neurons because it provides a smooth approximation to the step function 
    that is differentiable everywhere, which is a requirement for gradient-based optimization methods.

    It is a fast approximation of the Sigmoid function adapted from:

    F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.
    
    
    :k: A scaling factor that can be used to adjust the steepness of the 
                      SuperSpike function. Default is 25.
    :return: JIT compiled SuperSpike surrogate gradient function.
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
    
    :bwd: Function that calculates the gradient to be used in the backwards pass.
    :fwd: Function that returns a value between 0 and 1. Default is Heaviside.
    :return: A JIT compiled activation function comprised of the specified forward and backward functions.
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