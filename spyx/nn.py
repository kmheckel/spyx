import jax
import jax.numpy as jnp
import haiku as hk
from .axn import superspike, abs_linear

from collections.abc import Sequence
from typing import Optional, Union
import warnings

from collections import namedtuple

#needs fixed.
class ALIF(hk.RNNCore): 
    """
    Adaptive LIF Neuron based on the model used in LSNNs

    Bellec, G., Salaj, D., Subramoney, A., Legenstein, R. & Maass, W. 
    Long short- term memory and learning-to-learn in networks of spiking neurons. 
    32nd Conference on Neural Information Processing Systems (2018).
    
    """

    def __init__(self, hidden_shape, beta=None, gamma=None,
                 threshold = 1,
                 activation = superspike(),
                 name="ALIF"):

        """
        :hidden_shape: Hidden layer shape.
        :beta: Membrane decay/inverse time constant.
        :gamma: Threshold adaptation constant.
        :threshold: Neuron firing threshold.
        :activation: spyx.axn.Axon object determining forward function and surrogate gradient function.
        """

        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.spike = activation
    
    def __call__(self, x, VT):
        """
        :x: Tensor from previous layer.
        :VT: Neuron state vector.
        """

        V, T = jnp.split(VT, 2, -1)
        
        gamma = self.gamma
        beta = self.beta
        # threshold adaptation
        if not gamma:
            gamma = hk.get_parameter("gamma", self.hidden_shape, 
                                 init=hk.initializers.TruncatedNormal(0.25, 0.5))
            gamma = jnp.clip(gamma, 0, 1)
        else:
            gamma = hk.get_parameter("gamma", [],
                                init=hk.initializers.Constant(gamma))
            gamma = jnp.clip(gamma, 0, 1)

        if not beta:
            beta = hk.get_parameter("beta", self.hidden_shape, 
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.clip(beta, 0, 1)
        else:
            beta = hk.get_parameter("beta", [],
                                init=hk.initializers.Constant(beta))
            beta = jnp.clip(beta, 0, 1)

        # calculate whether spike is generated, and update membrane potential
        thresh = self.threshold + T
        spikes = self.spike(V - thresh) # T is the dynamic threshold adaptation
        V = beta*V + x - spikes*thresh
        T = gamma*T + (1-gamma)*spikes
        
        VT = jnp.concatenate([V,T], axis=-1)
        return spikes, VT
    
    # not sure if this is borked.
    def initial_state(self, batch_size): # this might need fixed to match CuBaLIF...
        return jnp.zeros((batch_size,) + tuple(2*s for s in self.hidden_shape))


CustomALIFStateTuple = namedtuple('CustomALIFStateTuple', ('s', 'z', 'r', 'z_local'))


class RecurrentLIFLight(hk.RNNCore):
    """
    Recurrent Adaptive Leaky Integrate and Fire neuron model with threshold adaptation.
    It can be used for LIF only by setting beta to 0.

    Original code from https://github.com/IGITUGraz/eligibility_propagation for RecurrentLIFLight
    Copyright 2019-2020, the e-prop team:
    Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, Wolfgang Maass
    from the Institute for theoretical computer science, TU Graz, Austria.

    Params
    ------
    n_rec: int
        Number of recurrent neurons.
    tau: float or array
        Membrane time constant (ms)
    thr: float or array
        Firing threshold.
    dt: float
        Time step (ms)
    dtype:
        Data type.
    dampening_factor: float
        Dampening factor for the surrogate gradient (see abs_linear).
    tau_adaptation: float or array
        Time constant for threshold adaptation (ALIF model)
    beta: float or array
        Decay rate for threshold adaptation (ALIF model)
    tag: str
        parameter tag.
    stop_gradients: bool
        Whether to stop gradients.
        If True, e-prop will be applied
        If False, exact BPTT will be applied
    w_rec_init: array
        Initial value for the recurrent weights.
    n_refractory: float
        Refractory period (ms)
    rec: bool
        Whether to include recurrent connections.   
    name: str
        Name of the Haiku module.
    """

    def __init__(self, 
                 n_rec, tau=20., thr=.615, dt=1., dtype=jnp.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=.16, tag='',
                 stop_gradients=False, w_rec_init=None, n_refractory=1, rec=True,
                 name="RecurrentLIFLight"):
        super().__init__(name=name)

        self.n_refractory = n_refractory
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = jnp.exp(-dt / tau_adaptation)

        if jnp.isscalar(tau): tau = jnp.ones(n_rec, dtype=dtype) * jnp.mean(tau)
        if jnp.isscalar(thr): thr = jnp.ones(n_rec, dtype=dtype) * jnp.mean(thr)        

        tau = jnp.array(tau, dtype=dtype)
        dt = jnp.array(dt, dtype=dtype)
        self.rec = rec

        self.dampening_factor = dampening_factor
        self.stop_gradients = stop_gradients
        self.dt = dt
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = jnp.exp(-dt / tau)
        self.thr = thr

        if rec:
            init_w_rec_var = w_rec_init if w_rec_init is not None else hk.initializers.TruncatedNormal(1./jnp.sqrt(n_rec))
            self.w_rec_var = hk.get_parameter("w_rec" + tag, (n_rec, n_rec), dtype, init_w_rec_var)

            self.recurrent_disconnect_mask = jnp.diag(jnp.ones(n_rec, dtype=bool))

            self.w_rec_val = jnp.where(self.recurrent_disconnect_mask, jnp.zeros_like(self.w_rec_var), self.w_rec_var)

        self.built = True

    def initial_state(self, batch_size, dtype=jnp.float32):
        """
        Initialize the state of the neuron model.
        
        :batch_size: tuple
            Batch size.
        :dtype:
            Data type.
        """
        n_rec = self.n_rec

        s0 = jnp.zeros(shape=(batch_size, n_rec, 2), dtype=dtype)
        z0 = jnp.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z_local0 = jnp.zeros(shape=(batch_size, n_rec), dtype=dtype)
        r0 = jnp.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return CustomALIFStateTuple(s=s0, z=z0, r=r0, z_local=z_local0)
    
    def compute_z(self, v, b):
        """
        Compute the surrogate gradient.
        """
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = abs_linear(self.dampening_factor)(v_scaled)
        z = z * 1 / self.dt

        return z
        
    def __call__(self, inputs, state):
        decay = self._decay

        z = state.z
        z_local = state.z_local
        s = state.s

        if self.stop_gradients:
            z = jax.lax.stop_gradient(z)
            
        i_in = inputs.reshape(-1, self.n_rec)

        if self.rec:
            if len(self.w_rec_val.shape) == 3:
                i_rec = jnp.einsum('bi,bij->bj', z, self.w_rec_val)
            else:
                i_rec = jnp.matmul(z, self.w_rec_val)

            i_t = i_in + i_rec
        else:
            i_t = i_in

        v, b = s[..., 0], s[..., 1]
        new_b = self.decay_b * b + z_local

        I_reset = z * self.thr * self.dt
        new_v = decay * v + i_t  - I_reset

        is_refractory = state.r > 0
        zeros_like_spikes = jnp.zeros_like(z)
        z_computed = self.compute_z(new_v, new_b)
        new_z = jnp.where(is_refractory, zeros_like_spikes, z_computed)
        new_z_local = jnp.where(is_refractory, zeros_like_spikes, z_computed)
        new_r = state.r + self.n_refractory * new_z - 1
        new_r = jnp.clip(new_r, 0., float(self.n_refractory))

        if self.stop_gradients:
            new_r = jax.lax.stop_gradient(new_r)
        new_s = jnp.stack((new_v, new_b), axis=-1)

        new_state = CustomALIFStateTuple(s=new_s, z=new_z, r=new_r, z_local=new_z_local)
        return new_z, new_state


class LeakyLinear(hk.RNNCore):
    """
    Leaky real-valued output neuron from the code of the paper https://github.com/IGITUGraz/eligibility_propagation

    To be replace with Linear + LI in the future.
 
    """
    def __init__(self, n_in, n_out, kappa, dtype=jnp.float32, name="LeakyLinear"):
        super().__init__(name=name)
        self.n_in = n_in
        self.n_out = n_out
        self.kappa = kappa

        self.dtype = dtype

        self.weights = hk.get_parameter("weights", shape=[n_in, n_out], dtype=dtype,
                                        init=hk.initializers.TruncatedNormal(1./jnp.sqrt(n_in)))
        
        # self.weights = hk.get_parameter("weights", shape=[n_in, n_out], dtype=dtype,
        #                                 init=hk.initializers.Constant(
        #                                     jnp.eye(n_in, n_out)
        #                                 ))

        self._num_units = self.n_out
        self.built = True


    def initial_state(self, batch_size, dtype=jnp.float32):
        s0 = jnp.zeros(shape=(batch_size, self.n_out), dtype=dtype)
        return s0

    def __call__(self, inputs, state, scope=None, dtype=jnp.float32):
        if len(self.weights.shape) == 3:
            outputs = jnp.einsum('bi,bij->bj', inputs, self.weights)
        else:
            outputs = jnp.matmul(inputs, self.weights)
        new_s = self.kappa * state  + (1 - self.kappa) * outputs
        return new_s, new_s
         
class LI(hk.RNNCore):
    """
    Leaky-Integrate (Non-spiking) neuron model.

 
    """

    def __init__(self, layer_shape, beta=None, name="LI"):
        """
        
        :layer_size: Number of output neurons from the previous linear layer.
        :beta: Decay rate on membrane potential (voltage). Set uniformly across the layer.
        """
        super().__init__(name=name)
        self.layer_shape = layer_shape
        self.beta = beta
    
    def __call__(self, x, Vin):
        """
        :x: Input tensor from previous layer.
        :Vin: Neuron state tensor. 
        """
        beta = self.beta
        if not beta:
            beta = hk.get_parameter("beta", self.layer_shape,
                                init=hk.initializers.Constant(0.8))
            beta = jnp.clip(beta, 0, 1)
        else:
            beta = hk.get_parameter("beta", [],
                                init=hk.initializers.Constant(beta))
            beta = jnp.clip(beta, 0, 1)

        # calculate whether spike is generated, and update membrane potential
        Vout = beta*Vin + x
        return Vout, Vout
    
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.layer_shape)

class IF(hk.RNNCore): 
    """
    Integrate and Fire neuron model. While not being as powerful/rich as other neuron models, they are very easy to implement in hardware.
    
    """

    def __init__(self, hidden_shape,
                 threshold = 1,
                 activation = superspike(),
                 name="IF"):
        """
        :hidden_size: Size of preceding layer's outputs
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.activation function, default is Heaviside with Straight-Through-Estimation.
        """
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation
    
    def __call__(self, x, V):
        """
        :x: Vector coming from previous layer.
        :V: Neuron state tensor.
        """
        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V-self.threshold)
        V = V + x - spikes*self.threshold
        
        return spikes, V

    def initial_state(self, batch_size): 
        return jnp.zeros((batch_size,) + self.hidden_shape)


class LIF(hk.RNNCore):
    """
    Leaky Integrate and Fire neuron model inspired by the implementation in
    snnTorch:

    https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html
    
    """

    def __init__(self, 
                 hidden_shape: tuple, 
                 beta=None,
                 threshold = 1.,
                 activation = superspike(),
                 name="LIF"):

        """
        
        :hidden_size: Size of preceding layer's outputs
        :beta: decay rate. Set to float in range (0,1] for uniform decay across layer, otherwise it will be a normal
                distribution centered on 0.5 with stddev of 0.25
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.axn.Axon object, default is Heaviside with Straight-Through-Estimation.
        """
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.beta = beta
        self.threshold = threshold
        self.spike = activation
    
    def __call__(self, x, V):
        """
        :x: input vector coming from previous layer.
        :V: neuron state tensor.

        """
        beta = self.beta # this line can probably be deleted, and the check changed to self.beta
        if not beta:
            beta = hk.get_parameter("beta", self.hidden_shape,
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.clip(beta, 0, 1)
        else:
            beta = hk.get_parameter("beta", [],
                                init=hk.initializers.Constant(beta))
            beta = jnp.clip(beta, 0, 1)
            
        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V-self.threshold)
        V = beta*V + x - spikes * self.threshold
        
        return spikes, V

    def initial_state(self, batch_size): 
        return jnp.zeros((batch_size,) + self.hidden_shape)


class CuBaLIF(hk.RNNCore): 
    def __init__(self, 
                 hidden_shape, 
                 alpha=None, beta=None,
                 threshold = 1,
                 activation = superspike(),
                 name="CuBaLIF"):
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.spike = activation
    
    def __call__(self, x, VI):
        V, I = jnp.split(VI, 2, -1)
        
        alpha = self.alpha
        beta = self.beta

        if not alpha:
            alpha = hk.get_parameter("alpha", self.hidden_shape, 
                                 init=hk.initializers.TruncatedNormal(0.25, 0.5))
            alpha = jnp.clip(alpha, 0, 1)
        else:
            alpha = hk.get_parameter("alpha", [], 
                                 init=hk.initializers.Constant(alpha))
            alpha = jnp.clip(alpha, 0, 1)
        if not beta:
            beta = hk.get_parameter("beta", self.hidden_shape, 
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.clip(beta, 0, 1)
        else:
            beta = hk.get_parameter("beta", [], 
                                init=hk.initializers.Constant(beta))
            beta = jnp.clip(beta, 0, 1)
        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V-self.threshold)
        reset = spikes*self.threshold
        V = V - reset
        I = alpha*I + x
        V = beta*V + I - reset # cast may not be needed?
        
        VI = jnp.concatenate([V,I], axis=-1)
        return spikes, VI
    
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2*v for v in self.hidden_shape))
    
class RIF(hk.RNNCore): 
    """
    Recurrent Integrate and Fire neuron model.
    
    """

    def __init__(self, hidden_shape, 
                 threshold = 1,
                 activation = superspike(),
                 name="RIF"):
        """
        :hidden_size: Size of preceding layer's outputs
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.activation function, default is Heaviside with Straight-Through-Estimation.
        """
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation
    
    def __call__(self, x, V):
        """
        :x: Vector coming from previous layer.
        :V: Neuron state tensor.
        """

        recurrent = hk.get_parameter("w", self.hidden_shape*2, init=hk.initializers.TruncatedNormal())

        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V-self.threshold)
        feedback = spikes@recurrent
        V = V + x + feedback - spikes*self.threshold
        
        return spikes, V

    def initial_state(self, batch_size): 
        return jnp.zeros((batch_size,) + self.hidden_shape)

class RLIF(hk.RNNCore): 
    """
    Recurrent LIF Neuron adapted from snnTorch. 

    https://snntorch.readthedocs.io/en/latest/snn.neurons_rleaky.html
    """

    def __init__(self, hidden_shape, beta=None,
                 threshold = 1,
                 activation = superspike(),
                 name="RLIF"):

        """
        Initialization function.

        :hidden_shape: The tuple describing the layer's shape. Can accomodate varying shapes to directly stack on convolution layers without flattening.
        :beta: Decay constant. Unless explicitly set to a float of range [0,1], it is treated as a learnable parameter.
        :threshold: Firing threshold for the layer. Does not currently support learning/trainable thresholds.
        :activation: A spyx.axn.Axon object specifying the forward and reverse activation function. By default it is Heaviside with Straight Through Estimation.
        """

        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.beta = beta
        self.threshold = threshold
        self.spike = activation
    
    def __call__(self, x, V):
        """
        :x: The input data/latent vector from another layer.
        :V: The state tensor.
        """

        recurrent = hk.get_parameter("w", self.hidden_shape*2, init=hk.initializers.TruncatedNormal())

        beta = self.beta
        if not beta:
            beta = hk.get_parameter("beta", self.hidden_shape, 
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.clip(beta, 0, 1)
        else:
            beta = hk.get_parameter("beta", [], 
                                init=hk.initializers.Constant(beta))
            beta = jnp.clip(beta, 0, 1)
        
        spikes = self.spike(V-self.threshold)
        feedback = spikes@recurrent # investigate and fix this...
        V = beta*V + x + feedback - spikes*self.threshold
        
        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)

class RCuBaLIF(hk.RNNCore): 
    def __init__(self, hidden_shape, alpha=None, beta=None,  
                 activation = superspike(),
                 name="RCuBaLIF"):
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.alpha = alpha
        self.beta = beta
        self.spike = activation
    
    def __call__(self, x, VI):
        V, I = jnp.split(VI, 2, -1)
        
        alpha = self.alpha
        beta = self.beta

        recurrent = hk.get_parameter("w", self.hidden_shape*2, init=hk.initializers.TruncatedNormal())

        if not alpha:
            alpha = hk.get_parameter("alpha", self.hidden_shape, 
                                 init=hk.initializers.TruncatedNormal(0.25, 0.5))
            alpha = jnp.clip(alpha, 0, 1)
        else:
            alpha = hk.get_parameter("alpha", [], 
                                 init=hk.initializers.Constant(alpha))
            alpha = jnp.clip(alpha, 0, 1)
        if not beta:
            beta = hk.get_parameter("beta", self.hidden_shape, 
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.clip(beta, 0, 1)
        else:
            beta = hk.get_parameter("beta", [], 
                                init=hk.initializers.Constant(beta))
            beta = jnp.clip(beta, 0, 1)

        # calculate whether spike is generated, and update membrane potential
        spikes = self.spike(V-self.threshold)
        V = V - spikes * self.threshold
        feedback = spikes@recurrent
        I = alpha*I + x + feedback
        V = beta*V + I
        
        VI = jnp.concatenate([V,I], axis=-1)
        return spikes, VI
    
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2*v for v in self.hidden_shape))

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

def PopulationCode(num_classes):
    """
    Add population coding to the preceding neuron layer. Preceding layer's output shape must be a multiple of
    the number of classes. Use this for rate coded SNNs where the time steps are too few to get a good spike count.
    """
    def _pop_code(x):
        return jnp.sum(jnp.reshape(x, (-1,num_classes)), axis=-1)
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

_VMAP_SHAPE_INFERENCE_WARNING = (
    "When running under vmap, passing an `int` (except for `1`) for "
    "`window_shape` or `strides` will result in the wrong shape being inferred "
    "because the batch dimension is not visible to Haiku. Please update your "
    "code to specify a full unbatched size.\n"
    "For example if you had `pool(x, window_shape=3, strides=1)` before, you "
    "should now pass `pool(x, window_shape=(3, 3, 1), strides=1)`. \n"
    "Haiku will assume that any additional dimensions in your input are "
    "batch dimensions, and will pad `window_shape` and `strides` accordingly "
    "making your module support both batched and per-example inputs."
)


def _warn_if_unsafe(window_shape, strides):
  unsafe = lambda size: isinstance(size, int) and size != 1
  if unsafe(window_shape) or unsafe(strides):
    warnings.warn(_VMAP_SHAPE_INFERENCE_WARNING, DeprecationWarning)


def sum_pool(
    value: jax.Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    channel_axis: Optional[int] = -1,
) -> jax.Array:
  """Sum pool.

  Args:
    value: Value to pool.
    window_shape: Shape of the pooling window, same rank as value.
    strides: Strides of the pooling window, same rank as value.
    padding: Padding algorithm. Either ``VALID`` or ``SAME``.
    channel_axis: Axis of the spatial channels for which pooling is skipped.

  Returns:
    Pooled result. Same rank as value.
  """
  if padding not in ("SAME", "VALID"):
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  _warn_if_unsafe(window_shape, strides)
  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  return jax.lax.reduce_window(value, 0., jax.lax.add, window_shape, strides,
                           padding)

class SumPool(hk.Module):
  """Sum pool.

  Returns the total number of spikes emitted in a receptive field.
  """

  def __init__(
      self,
      window_shape: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]],
      padding: str,
      channel_axis: Optional[int] = -1,
      name: Optional[str] = None,
  ):
    """Sum pool.

    Args:
      window_shape: Shape of the pooling window, same rank as value.
      strides: Strides of the pooling window, same rank as value.
      padding: Padding algorithm. Either ``VALID`` or ``SAME``.
      channel_axis: Axis of the spatial channels for which pooling is skipped.
      name: String name for the module.
    """
    super().__init__(name=name)
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.channel_axis = channel_axis

  def __call__(self, value: jax.Array) -> jax.Array:
    return sum_pool(value, self.window_shape, self.strides,
                    self.padding, self.channel_axis)