import jax
import jax.numpy as jnp
import haiku as hk

# eventually, it would be good to make an abstract class that all activations below extend...

# should this be added to nn?
class ActivityRegularization(hk.Module):
    """
    Add state to the SNN to track the average number of spikes emitted per neuron per batch.

    Adding this to a network requires using the Haiku transform_with_state transform, which will also return an initial regularization state vector.
    This blank initial vector can be reused and is provided as the second arg to the SNN's apply function. 
    """

    def __init__(self, name="ActReg"):
        super().__init__(name=name)
        
    def __call__(self, spikes):
        spike_count = hk.get_state("spike_count", [spikes.shape[-1]], init=jnp.zeros)
        hk.set_state("spike_count", spike_count + jnp.mean(spikes, axis=0))
        return spikes


class AdaSpike:
    """
    Simplified version of SuperSpike, dropping the power from the denominator.
    Features an increasing scale factor with linear schedule, increasing the
    sharpness of the surrogate gradient over time.

    $$\frac{\delta S}{\delta U} = \frac{1}{1+k|U|}$$

    Attributes:
        growth_rate: The amount the scale factor k is incremented by after each batch.
    """

    def __init__(self, growth_rate=0.5):
        self.k = 1
        self.gr = growth_rate
        
        @jax.custom_vjp
        def f(U, k): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U, k):
            return f(U), (U, k)
            
        # accepts context, primal val
        # not sure if k actually changes or it gets jit'ed and stays static..
        def f_bwd(context, grad):
            U, k = context
            return (grad / (1+k*jnp.abs(U)) , )
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        self.k += self.growth_rate
        return self.f(U, self.k)

# Surrogate functions
class Arctan:
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
    

    def __init__(self, scale_factor=2):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), ()
            
        # Straight Through Estimator
        def f_bwd(U, grad):
            return ( (1 / (jnp.pi * (1+(jnp.pi*U*self.k/2)**2))) * grad ) 
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U) 

class Boxcar:
    """
    Boxcar surrogate gradient activation function. Under construction.
    """

    def __init__(self, scale_factor=1):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), U
            
        # Needs fixed.
        def f_bwd(U, grad):
            if jnp.abs(U) <= 0.5:
                return ( 0.5 * grad )
            return ( 0 ) 
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U)

class Heaviside:
    """
    This class implements the Heaviside activation function for a spiking neuron.
    
    The Heaviside function is a step function that takes a single argument. 
    It returns 0 if the input is less than 0, and 1 if the input is greater than 0.
    
    The Heaviside function is often used in the context of spiking neurons because 
    it can be used to model the firing of a neuron. When the input to the neuron 
    (the sum of the weighted inputs) exceeds a certain threshold, the neuron fires, 
    producing a spike of activity. This can be modeled as the Heaviside function, 
    where the output is 0 when the input is below the threshold, and 1 when the input 
    is above the threshold.
    
    Attributes:
        scale_factor: A scaling factor that can be used to adjust the steepness of the 
                      step function. Default is 25.
    """
    
    
    
    def __init__(self, scale_factor=25):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), ()
            
        # Straight Through Estimator
        def f_bwd(U, grad):
            return ( grad*U ) 
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U)


class Sigmoid:
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
                      Sigmoid function. Default is 25.
    """
    


    def __init__(self, scale_factor=25):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), U
            
        # accepts context, primal val
        def f_bwd(U, grad): # is spk needed at all???
            return ((self.k*jnp.exp(-self.k*U) / (jnp.exp(-self.k*U)+1)**2) * grad)
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, V, T):
        return self.f(V, T)

class SuperSpike:
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



    def __init__(self, scale_factor=25):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), U
            
        # accepts context, primal val
        def f_bwd(U, grad):
            return (grad / (1+self.k*jnp.abs(U))**2 , )
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U)