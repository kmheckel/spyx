import jax
import jax.numpy as jnp
import haiku as hk
from .axn import Axon

# need to add shape checking/warning
def PopulationCode(num_classes):
    """
    Add population coding to the preceding neuron layer. Preceding layer's output shape must be a multiple of
    the number of classes. Use this for rate coded SNNs where the time steps are too few to get a good spike count.
    """
    def _pop_code(x):
        return jnp.sum(jnp.reshape(x, (-1,num_classes)), axis=-1)
    return jax.jit(_pop_code)

class ALIF(hk.RNNCore): 
    """
    Adaptive LIF Neuron based on the model used in LSNNs:

    Bellec, G., Salaj, D., Subramoney, A., Legenstein, R. & Maass, W. 
    Long short- term memory and learning-to-learn in networks of spiking neurons. 
    32nd Conference on Neural Information Processing Systems (2018).
    
    """


    def __init__(self, hidden_shape, beta=None, gamma=None, threshold=1,
                 activation = Axon(),
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
        self.init_threshold = threshold
        self.act = activation
    
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
            gamma = jnp.minimum(jax.nn.relu(gamma),1)
        if not beta:
            beta = hk.get_parameter("beta", self.hidden_shape, 
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.minimum(jax.nn.relu(beta),1)
        # calculate whether spike is generated, and update membrane potential
        thresh = self.init_threshold + T
        spikes = self.act(V - thresh)
        V = beta*V + x - spikes*thresh
        T = gamma*T + (1-gamma)*spikes
        
        VT = jnp.concatenate([V,T], axis=-1)
        return spikes, VT
    
    # not sure if this is borked.
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + tuple(2*s for s in self.hidden_shape))
         
class LI(hk.RNNCore):
    """
    Leaky-Integrate (Non-spiking) neuron model.

 
    """

    def __init__(self, layer_shape, beta=0.8, name="LI"):
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
        # calculate whether spike is generated, and update membrane potential
        Vout = self.beta*Vin + x
        return Vout, Vout
    
    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.layer_shape)

class IF(hk.RNNCore): 
    """
    Integrate and Fire neuron model. While not being as powerful/rich as other neuron models, they are very easy to implement in hardware.
    
    """

    def __init__(self, hidden_shape, threshold=1, 
                 activation = Axon(),
                 name="IF"):
        """
        :hidden_size: Size of preceding layer's outputs
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.activation function, default is Heaviside with Straight-Through-Estimation.
        """
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.act = activation
    
    def __call__(self, x, V):
        """
        :x: Vector coming from previous layer.
        :V: Neuron state tensor.
        """
        # calculate whether spike is generated, and update membrane potential
        spikes = self.act(V - self.threshold)
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

    def __init__(self, hidden_shape: tuple, beta=None, threshold=1, 
                 activation = Axon(),
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
        self.act = activation
    
    def __call__(self, x, V):
        """
        :x: input vector coming from previous layer.
        :V: neuron state tensor.

        """
        beta = self.beta
        if not beta:
            beta = hk.get_parameter("beta", self.hidden_shape,
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.minimum(jax.nn.relu(beta),1)
            
        # calculate whether spike is generated, and update membrane potential
        spikes = self.act(V - self.threshold)
        V = beta*V + x - spikes*self.threshold
        
        return spikes, V

    def initial_state(self, batch_size): 
        return jnp.zeros((batch_size,) + self.hidden_shape)


class CuBaLIF(hk.RNNCore): 
    def __init__(self, hidden_size, alpha=None, beta=None, threshold=1, 
                 activation = Axon(),
                 name="CuBaLIF"):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.act = activation
    
    def __call__(self, x, VI):
        V, I = jnp.split(VI, 2, -1)
        
        alpha = self.alpha
        beta = self.beta

        if not alpha:
            alpha = hk.get_parameter("alpha", [self.hidden_size], 
                                 init=hk.initializers.TruncatedNormal(0.25, 0.5))
            alpha = jnp.minimum(jax.nn.relu(alpha),1)
        if not beta:
            beta = hk.get_parameter("beta", [self.hidden_size], 
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.minimum(jax.nn.relu(beta),1)
        # calculate whether spike is generated, and update membrane potential
        spikes = self.act(V - self.threshold)
        I = alpha*I + x
        V = beta*V + I - spikes*self.threshold # cast may not be needed?
        
        VI = jnp.concatenate([V,I], axis=-1)
        return spikes, VI
    
    def initial_state(self, batch_size):
        return jnp.zeros([batch_size, self.hidden_size*2])
    
class RIF(hk.RNNCore): 
    """
    Integrate and Fire neuron model. While not being as powerful/rich as other neuron models, they are very easy to implement in hardware.
    
    """

    def __init__(self, hidden_shape, threshold=1, 
                 activation = Axon(),
                 name="RIF"):
        """
        :hidden_size: Size of preceding layer's outputs
        :threshold: threshold for reset. Defaults to 1.
        :activation: spyx.activation function, default is Heaviside with Straight-Through-Estimation.
        """
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.act = activation
    
    def __call__(self, x, V):
        """
        :x: Vector coming from previous layer.
        :V: Neuron state tensor.
        """

        recurrent = hk.get_parameter("w", self.hidden_shape, init=hk.initializers.TruncatedNormal())

        # calculate whether spike is generated, and update membrane potential
        spikes = self.act(V - self.threshold)
        V = V + x + recurrent*spikes - spikes*self.threshold
        
        return spikes, V

    def initial_state(self, batch_size): 
        return jnp.zeros((batch_size,) + self.hidden_shape)

class RLIF(hk.RNNCore): 
    """
    Recurrent LIF Neuron adapted from snnTorch. 

    https://snntorch.readthedocs.io/en/latest/snn.neurons_rleaky.html
    """

    def __init__(self, hidden_shape, beta=None, threshold=1,
                 activation = Axon(),
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
        self.act = activation
    
    def __call__(self, x, V):
        """
        :x: The input data/latent vector from another layer.
        :V: The state tensor.
        """

        recurrent = hk.get_parameter("w", self.hidden_shape, init=hk.initializers.TruncatedNormal())
        
        beta = self.beta
        if not beta:
            beta = hk.get_parameter("beta", self.hidden_shape, 
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.minimum(jax.nn.relu(beta),1)
        
        spikes = self.act(V - self.threshold)
        V = beta*V + x + recurrent*spikes - spikes*self.threshold
        
        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)

class RCuBaLIF(hk.RNNCore): 
    def __init__(self, hidden_size, alpha=None, beta=None, threshold=1, 
                 activation = Axon(),
                 name="RCuBaLIF"):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.act = activation
    
    def __call__(self, x, VI):
        V, I = jnp.split(VI, 2, -1)
        
        alpha = self.alpha
        beta = self.beta

        recurrent = hk.get_parameter("w", self.hidden_shape, init=hk.initializers.TruncatedNormal())

        if not alpha:
            alpha = hk.get_parameter("alpha", [self.hidden_size], 
                                 init=hk.initializers.TruncatedNormal(0.25, 0.5))
            alpha = jnp.minimum(jax.nn.relu(alpha),1)
        if not beta:
            beta = hk.get_parameter("beta", [self.hidden_size], 
                                init=hk.initializers.TruncatedNormal(0.25, 0.5))
            beta = jnp.minimum(jax.nn.relu(beta),1)
        # calculate whether spike is generated, and update membrane potential
        spikes = self.act(V - self.threshold)
        I = alpha*I + x
        V = beta*V + I + recurrent*spikes - spikes*self.threshold # cast may not be needed?
        
        VI = jnp.concatenate([V,I], axis=-1)
        return spikes, VI
    
    def initial_state(self, batch_size):
        return jnp.zeros([batch_size, self.hidden_size*2])