import jax
import jax.numpy as jnp
import optax

from jax import tree_util as tree


class silence_reg:
    """
    L2-Norm per-neuron activation normalization for spiking less than a target number of times.

    Attributes:
        min_spikes: neurons which spike below this value on average
        over the batch incur quadratic penalty.
    """
    def __init__(self, min_spikes):
        def _loss(x):
            return (jnp.maximum(0, min_spikes-jnp.mean(x, axis=0)))**2
        
        def _flatten(x):
            return jnp.reshape(x, (x.shape[0], -1))
        
        def _call(spikes):
            flat_spikes = tree.tree_map(_flatten, spikes)
            loss_vectors = tree.tree_map(_loss, flat_spikes)
            return jnp.sum(jnp.concatenate(tree.tree_flatten(loss_vectors)[0]))
        
        self.call = jax.jit(_call)
        
    def __call__(self, spikes):
        return self.call(spikes)
        
class sparsity_reg:
    """
    Layer activation normalization that seeks to discourage all neurons having a high firing rate.

    Attributes:
        max_spikes: Threshold for which penalty is incurred if the average number of
        spikes in the layer exceeds it.

        norm: an Optax loss function. Default is Huber loss.
    """
    def __init__(self, max_spikes, norm=optax.huber_loss):
        def _loss(x):
            return norm(jnp.maximum(0, jnp.mean(x, axis=-1) - max_spikes))
        
        def _flatten(x):
            return jnp.reshape(x, (x.shape[0], -1))
        
        def _call(spikes):
            flat_spikes = tree.tree_map(_flatten, spikes)
            loss_vectors = tree.tree_map(_loss, flat_spikes)
            return jnp.sum(jnp.concatenate(tree.tree_flatten(loss_vectors)[0]))
        
        self.call = jax.jit(_call)
        
    def __call__(self, spikes):
        return self.call(spikes) 


@jax.jit
def integral_accuracy(traces, targets):
    """
    Calculate the accuracy of a network's predictions based on the voltage traces.
    Used in combination with a Leaky-Integrate neuron model as the final layer.

    """

    preds = jnp.argmax(jnp.sum(traces, axis=-2), axis=-1)
    return jnp.sum(preds == targets) / traces.shape[0], preds

# should expose the smoothing rate and allow for users to partial it away or possibly schedule it...
@jax.jit
def integral_crossentropy(traces, targets, smoothing=0.2):
    """
    Calculate the crossentropy between the integral of membrane potentials.
    Allows for label smoothing to discourage silencing 
    the other neurons in the readout layer.

    Attributes:
        traces: the output of the final layer of the SNN
        targets: the integer labels for each class
        smoothing: [optional] rate at which to smooth labels.
    """

    logits = jnp.sum(traces, axis=-2) # time axis.
    labels = optax.smooth_labels(jax.nn.one_hot(targets, logits.shape[-1]), smoothing)
    return optax.softmax_cross_entropy(logits, labels).mean() 

