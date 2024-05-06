import jax
import jax.numpy as jnp
import haiku as hk

def heaviside(x):
    return jnp.where(x > 0, 1, 0).astype(x.dtype)

def custom(bwd=lambda x: x, 
           fwd=lambda x: heaviside(x)):
    """
    This function serves as the activation function for the SNNs, allowing for custom definitions of both surrogate gradients for backwards
    passes as well as substitution of the Heaviside function for relaxations such as sigmoids. 

    It is assumed that the input to this layer has already had it's threshold subtracted within the neuron model dynamics.

    The default behavior is a Heaviside forward activation with a straight through estimator surrogate gradient.
    
    :bwd: Function that calculates the gradient to be used in the backwards pass.
    :fwd: Forward activation/spiking function. Default is the heaviside function centered at 0.
    :return: A JIT compiled activation function comprised of the specified forward and backward functions.
    """

    @jax.custom_gradient
    def f(x):
        return fwd(x), lambda g: g * bwd(x)

    return jax.jit(f)

def tanh(k=1):
    """Hyperbolic Tangent activation.

    .. math:: 4 / (e^{-kx} + e^{kx})^2

    :k: Value for scaling the slope of the surrogate gradient.
    :return: JIT compiled tanh surrogate gradient function.
    """
    def grad_tanh(x):
        kx = k * x
        return 4 / (jnp.exp(-kx) + jnp.exp(kx))**2

    return custom(grad_tanh, heaviside)


def boxcar(width=2, height=0.5):
    """Boxcar activation.

    :width: Total width of non-zero gradient flow, centered on 0.
    :height: Value for gradient within the specified window.
    :return: JIT compiled boxcar surrogate gradient function.
    """
    k = width / 2
    h = height

    def grad_boxcar(x):
        return h * heaviside(-(jnp.abs(x) - k))

    return custom(grad_boxcar, heaviside)


def triangular(k=2):
    """
    Triangular activation inspired by Esser et. al. https://arxiv.org/abs/1603.08270

    .. math::
        max(0, 1-|kx|)

        
    :k: scale factor
    :return: JIT compiled triangular surrogate gradient function.
    """

    def grad_triangle(x):
        return jnp.maximum(0, 1-jnp.abs(k*x))
    
    return custom(grad_triangle, heaviside)


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
    k_pi = k*jnp.pi
        
    def grad_arctan(x):
        k_pi_x = k_pi * x
        return 1 / ((1+k_pi_x**2) * jnp.pi)

    return custom(grad_arctan, heaviside)


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
    def grad_superspike(x):
        return 1 / (1 + k*jnp.abs(x))**2
    
    return custom(grad_superspike, heaviside)

def abs_linear(dampening_factor=0.3):
    """
    This function implements the SpikeFunction surrogate gradient activation function for a spiking neuron.

    It was introduced in Bellec, Guillaume, et al. Long short-term memory and learning-to-learn in networks of spiking neurons. 
    arXiv:1803.09574, arXiv, 25 dec 2018. arXiv.org, 
    https://doi.org/10.48550/arXiv.1803.09574.

    :v_scaled: The normalized membrane potential of the neuron scaled by the threshold.
    :dampening_factor: The dampening factor for the surrogate gradient, 
        which can improve the stability of the training process
        for deep networks. Default is 0.3.
    """
    def fwd(v_scaled):
        z_ = jnp.greater(v_scaled, 0.)
        z_ = z_.astype(jnp.float32)
        return z_

    def grad(v_scaled):
        dz_dv_scaled = jnp.maximum(1 - jnp.abs(v_scaled), 0).astype(v_scaled.dtype)
        dz_dv_scaled *= dampening_factor

        return dz_dv_scaled

    return custom(grad, fwd)
