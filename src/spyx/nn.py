import jax
import jax.numpy as jnp
from flax import nnx
from .axn import superspike

from collections.abc import Sequence
from typing import Optional, Union, Any
import warnings

class ALIF(nnx.Module): 
    """
    Adaptive LIF Neuron based on the model used in LSNNs:

    Bellec, G., Salaj, D., Subramoney, A., Legenstein, R. & Maass, Maass, W. 
    Long short- term memory and learning-to-learn in networks of spiking neurons. 
    32nd Conference on Neural Information Processing Systems (2018).
    
    """

    def __init__(self, hidden_shape, beta=None, gamma=None,
                 threshold = 1,
                 activation = superspike(),
                 *,
                 rngs: nnx.Rngs):
        """
        :hidden_shape: Hidden layer shape.
        :beta: Membrane decay/inverse time constant.
        :gamma: Threshold adaptation constant.
        :threshold: Neuron firing threshold.
        :activation: spyx.axn.Axon object determining forward function and surrogate gradient function.
        """
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation
        
        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))
            
        if gamma is None:
            self.gamma = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
            )
        else:
            self.gamma = nnx.Param(jnp.full((), gamma))

    def __call__(self, x, VT):
        """
        :x: Tensor from previous layer.
        :VT: Neuron state vector.
        """
        V, T = jnp.split(VT, 2, -1)
        
        beta = jnp.clip(self.beta[...], 0, 1)
        gamma = jnp.clip(self.gamma[...], 0, 1)

        # calculate whether spike is generated, and update membrane potential
        thresh = self.threshold + T
        spikes = self.spike(V - thresh) # T is the dynamic threshold adaptation
        V = beta * V + x - spikes * thresh
        T = gamma * T + (1 - gamma) * spikes
        
        VT = jnp.concatenate([V, T], axis=-1)
        return spikes, VT
    
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2 * s for s in self.hidden_shape))
         
class LI(nnx.Module):
    """
    Leaky-Integrate (Non-spiking) neuron model.
    """

    def __init__(self, layer_shape, beta=None, *, rngs: nnx.Rngs):
        """
        :layer_shape: Shape of the layer.
        :beta: Decay rate on membrane potential (voltage).
        """
        self.layer_shape = layer_shape
        if beta is None:
            self.beta = nnx.Param(jnp.full(layer_shape, 0.8))
        else:
            self.beta = nnx.Param(jnp.full((), beta))
    
    def __call__(self, x, Vin):
        """
        :x: Input tensor from previous layer.
        :Vin: Neuron state tensor. 
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        Vout = beta * Vin + x
        return Vout, Vout
    
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.layer_shape)

class IF(nnx.Module): 
    """
    Integrate and Fire neuron model.
    """

    def __init__(self, hidden_shape,
                 threshold = 1,
                 activation = superspike()):
        """
        :hidden_shape: Shape of the layer.
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.activation function.
        """
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation
    
    def __call__(self, x, V):
        """
        :x: Vector coming from previous layer.
        :V: Neuron state tensor.
        """
        spikes = self.spike(V - self.threshold)
        V = V + x - spikes * self.threshold
        return spikes, V

    def initial_state(self, batch_size): 
        return jnp.zeros((batch_size,) + self.hidden_shape)


class LIF(nnx.Module):
    """
    Leaky Integrate and Fire neuron model.
    """

    def __init__(self, 
                 hidden_shape: tuple, 
                 beta=None,
                 threshold = 1.,
                 activation = superspike(),
                 *,
                 rngs: nnx.Rngs):
        """
        :hidden_shape: Shape of the layer.
        :beta: decay rate.
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.axn.Axon object.
        """
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation
        
        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))
    
    def __call__(self, x, V):
        """
        :x: input vector coming from previous layer.
        :V: neuron state tensor.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        spikes = self.spike(V - self.threshold)
        V = beta * V + x - spikes * self.threshold
        return spikes, V

    def initial_state(self, batch_size): 
        return jnp.zeros((batch_size,) + self.hidden_shape)

class CuBaLIF(nnx.Module): 
    def __init__(self, 
                 hidden_shape, 
                 alpha=None, beta=None,
                 threshold = 1,
                 activation = superspike(),
                 *,
                 rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation

        if alpha is None:
            self.alpha = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
            )
        else:
            self.alpha = nnx.Param(jnp.full((), alpha))

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))
    
    def __call__(self, x, VI):
        V, I = jnp.split(VI, 2, -1)
        
        alpha = jnp.clip(self.alpha[...], 0, 1)
        beta = jnp.clip(self.beta[...], 0, 1)

        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V - self.threshold)
        reset = spikes * self.threshold
        V = V - reset
        I = alpha * I + x
        V = beta * V + I - reset
        
        VI = jnp.concatenate([V, I], axis=-1)
        return spikes, VI
    
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2 * v for v in self.hidden_shape))
    
class RIF(nnx.Module): 
    """
    Recurrent Integrate and Fire neuron model.
    """

    def __init__(self, hidden_shape, 
                 threshold = 1,
                 activation = superspike(),
                 *,
                 rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation
        
        # recurrent weight matrix
        self.recurrent_w = nnx.Param(
            nnx.initializers.truncated_normal()(rngs.params(), self.hidden_shape + self.hidden_shape)
        )
    
    def __call__(self, x, V):
        """
        :x: Vector coming from previous layer.
        :V: Neuron state tensor.
        """
        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V - self.threshold)
        feedback = spikes @ self.recurrent_w[...]
        V = V + x + feedback - spikes * self.threshold
        
        return spikes, V

    def initial_state(self, batch_size): 
        return jnp.zeros((batch_size,) + self.hidden_shape)

class RLIF(nnx.Module): 
    """
    Recurrent LIF Neuron.
    """

    def __init__(self, hidden_shape, beta=None,
                 threshold = 1,
                 activation = superspike(),
                 *,
                 rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation

        # recurrent weight matrix
        self.recurrent_w = nnx.Param(
            nnx.initializers.truncated_normal()(rngs.params(), self.hidden_shape + self.hidden_shape)
        )

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))
    
    def __call__(self, x, V):
        """
        :x: The input data/latent vector from another layer.
        :V: The state tensor.
        """
        beta = jnp.clip(self.beta[...], 0, 1)
        
        spikes = self.spike(V - self.threshold)
        feedback = spikes @ self.recurrent_w[...]
        V = beta * V + x + feedback - spikes * self.threshold
        
        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)

class RCuBaLIF(nnx.Module): 
    def __init__(self, hidden_shape, alpha=None, beta=None,  
                 threshold = 1, activation = superspike(),
                 *,
                 rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation

        # recurrent weight matrix
        self.recurrent_w = nnx.Param(
            nnx.initializers.truncated_normal()(rngs.params(), self.hidden_shape + self.hidden_shape)
        )

        if alpha is None:
            self.alpha = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
            )
        else:
            self.alpha = nnx.Param(jnp.full((), alpha))

        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), self.hidden_shape) + 0.25
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))
    
    def __call__(self, x, VI):
        V, I = jnp.split(VI, 2, -1)
        
        alpha = jnp.clip(self.alpha[...], 0, 1)
        beta = jnp.clip(self.beta[...], 0, 1)

        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V - self.threshold)
        V = V - spikes * self.threshold
        feedback = spikes @ self.recurrent_w[...]
        I = alpha * I + x + feedback
        V = beta * V + I
        
        VI = jnp.concatenate([V, I], axis=-1)
        return spikes, VI
    
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2 * v for v in self.hidden_shape))

class ActivityRegularization(nnx.Module):
    """
    Add state to the SNN to track the average number of spikes emitted per neuron per batch.
    """

    def __init__(self):
        # In NNX, state is just a Variable.
        self.spike_count = nnx.Variable(None) # we'll initialize on first call with proper shape
        
    def __call__(self, spikes):
        if self.spike_count.get_value() is None:
            self.spike_count.set_value(jnp.zeros(spikes.shape, dtype=spikes.dtype))
            
        self.spike_count[...] += spikes
        return spikes

def PopulationCode(num_classes):
    def _pop_code(x):
        return jnp.sum(jnp.reshape(x, (-1, num_classes)), axis=-1)
    return jax.jit(_pop_code)

def _infer_shape(
    x: jax.Array,
    size: Union[int, Sequence[int]],
    channel_axis: Optional[int] = -1,
) -> tuple[int, ...]:
  """Infer shape for pooling window or strides."""
  if isinstance(size, int):
    if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
      raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
    if channel_axis and channel_axis < 0:
      channel_axis = x.ndim + channel_axis
    return (1,) + tuple(size if d != channel_axis else 1
                        for d in range(1, x.ndim))
  elif len(size) < x.ndim:
    # Assume additional dimensions are batch dimensions.
    return (1,) * (x.ndim - len(size)) + tuple(size)
  else:
    assert x.ndim == len(size)
    return tuple(size)

def sum_pool(
    value: jax.Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    channel_axis: Optional[int] = -1,
) -> jax.Array:
  """Sum pool."""
  if padding not in ("SAME", "VALID"):
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  return jax.lax.reduce_window(value, 0., jax.lax.add, window_shape, strides,
                           padding)

class SumPool(nnx.Module):
  """Sum pool."""

  def __init__(
      self,
      window_shape: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]],
      padding: str,
      channel_axis: Optional[int] = -1,
  ):
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.channel_axis = channel_axis

  def __call__(self, value: jax.Array) -> jax.Array:
    return sum_pool(value, self.window_shape, self.strides,
                    self.padding, self.channel_axis)

class Sequential(nnx.Sequential):
    """
    A Sequential container that supports passing state through its layers.
    """
    def initial_state(self, batch_size):
        return [layer.initial_state(batch_size) if hasattr(layer, "initial_state") else None for layer in self.layers]
    
    def __call__(self, x, state):
        new_state = []
        for layer, s in zip(self.layers, state):
            if s is not None:
                x, s_new = layer(x, s)
                new_state.append(s_new)
            else:
                x = layer(x)
                new_state.append(None)
        return x, new_state

def run(model, x, state=None):
    """
    Execute a model over a sequence of inputs using jax.lax.scan.
    
    :model: A Flax NNX Module (e.g. nnx.Sequential).
    :x: Input data with shape [Time, Batch, ...].
    :state: Initial state for the model. If None, model.initial_state is used if available.
    :return: (outputs, final_state)
    """
    
    if state is None:
        # We need batch size from x
        batch_size = x.shape[1]
        if hasattr(model, "initial_state"):
            state = model.initial_state(batch_size)
    
    def scan_fn(carry, x_t):
        out, next_state = model(x_t, carry)
        return next_state, out
        
    final_state, outputs = jax.lax.scan(scan_fn, state, x)
    return outputs, final_state

