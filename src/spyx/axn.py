"""Surrogate-gradient activations for spiking neurons.

Each public factory in this module returns a JIT-compiled
``jax.custom_gradient`` function of signature ``(x: jax.Array) -> jax.Array``
suitable for passing to the ``activation=`` argument of any neuron in
``spyx.nn``. The forward pass is always the Heaviside step (spike / no
spike); the factories differ only in the surrogate they expose to the
backward pass.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

Activation = Callable[[jax.Array], jax.Array]
"""Type alias for a surrogate-gradient activation function.

A mapping from a pre-activation tensor to a binary spike tensor of the
same shape. Produced by :func:`custom`, :func:`superspike`,
:func:`arctan`, and friends.
"""


def heaviside(x: jax.Array) -> jax.Array:
    """Heaviside step: ``1`` where ``x > 0``, else ``0``, cast to ``x.dtype``."""
    return jnp.where(x > 0, 1, 0).astype(x.dtype)


def custom(
    bwd: Callable[[jax.Array], jax.Array] = lambda x: x,
    fwd: Callable[[jax.Array], jax.Array] = lambda x: heaviside(x),
) -> Activation:
    """Activation with a user-supplied surrogate gradient.

    Used as the spiking nonlinearity inside every Spyx neuron. The default
    ``fwd`` is the Heaviside step and the default ``bwd`` is the identity,
    which together give the straight-through estimator (STE):

    .. math::
        y = \\mathrm{Heaviside}(x), \\qquad \\frac{\\partial y}{\\partial x} \\approx 1.

    It is assumed that the input has already had its threshold subtracted by
    the calling neuron model.

    :bwd: Function that computes the surrogate gradient :math:`g(x)` used during
        the backward pass. Should return an array of the same shape as ``x``.
    :fwd: Forward activation / spiking function. Defaults to
        :func:`heaviside` centred at zero.
    :return: A JIT-compiled activation function comprised of the specified
        forward and backward functions.
    """

    @jax.custom_gradient
    def f(x):
        return fwd(x), lambda g: g * bwd(x)

    return jax.jit(f)

def tanh(k: float = 1) -> Activation:
    """Hyperbolic-tangent surrogate gradient.

    The forward pass is the Heaviside step; the backward pass uses the
    derivative of :math:`\\tanh(kx)`:

    .. math::
        g(x) = \\frac{4}{(e^{-kx} + e^{kx})^2}.

    :k: Slope factor. Larger values make the gradient more peaked around
        the threshold and closer to a true Heaviside derivative.
    :return: JIT-compiled tanh surrogate gradient function.
    """
    def grad_tanh(x):
        kx = k * x
        return 4 / (jnp.exp(-kx) + jnp.exp(kx))**2

    return custom(grad_tanh, heaviside)


def boxcar(width: float = 2, height: float = 0.5) -> Activation:
    """Boxcar surrogate gradient.

    The forward pass is the Heaviside step; the backward pass uses a
    rectangular window of half-width ``width/2`` centred at zero:

    .. math::
        g(x) = \\begin{cases}
            h & |x| \\le w/2 \\\\
            0 & \\text{otherwise}
        \\end{cases}

    The boxcar is the simplest symmetric surrogate and has been shown to
    train competitively on SHD despite its discontinuity.

    :width: Total width :math:`w` of the non-zero gradient window, centred
        on zero.
    :height: Value :math:`h` of the gradient inside the window.
    :return: JIT-compiled boxcar surrogate gradient function.
    """
    k = width / 2
    h = height

    def grad_boxcar(x):
        return h * heaviside(-(jnp.abs(x) - k))

    return custom(grad_boxcar, heaviside)


def triangular(k: float = 2) -> Activation:
    """
    Triangular activation inspired by Esser et. al. https://arxiv.org/abs/1603.08270

    .. math::
        max(0, 1-|kx|)

        
    :k: scale factor
    :return: JIT compiled triangular surrogate gradient function.
    """

    def grad_traingle(x):
        return jnp.maximum(0, 1-jnp.abs(k*x))
    
    return custom(grad_traingle, heaviside)


def arctan(k: float = 2) -> Activation:
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


def superspike(k: float = 25) -> Activation:
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
