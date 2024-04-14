import jax
import jax.numpy as jnp
import optax

from jax import tree_util as tree


def silence_reg(min_spikes):
    """L2-Norm per-neuron activation normalization for spiking less than a target number of times.

    :param min_spikes: neurons which spike below this value on average over the batch incur quadratic penalty.
    :return: JIT compiled regularization function.
    """
    def _loss(x):
        return (jnp.maximum(0, min_spikes-jnp.mean(x, axis=0)))**2
        
    def _flatten(x):
        return jnp.reshape(x, (x.shape[0], -1))
        
    def _call(spikes):
        flat_spikes = tree.tree_map(_flatten, spikes)
        loss_vectors = tree.tree_map(_loss, flat_spikes)
        return jnp.sum(jnp.concatenate(tree.tree_flatten(loss_vectors)[0]))
        
    return jax.jit(_call)
        
        
def sparsity_reg(max_spikes, norm=optax.huber_loss):
    """Layer activation normalization that seeks to discourage all neurons having a high firing rate.

    :param max_spikes: Threshold for which penalty is incurred if the average number of spikes in the layer exceeds it.
    :param norm: an Optax loss function. Default is Huber loss.
    :return: JIT compiled regularization function. 
    """
    def _loss(x):
        return norm(jnp.maximum(0, jnp.mean(x, axis=-1) - max_spikes)) # this may not work for convolution layers....
        
    def _flatten(x):
        return jnp.reshape(x, (x.shape[0], -1))
        
    def _call(spikes):
        flat_spikes = tree.tree_map(_flatten, spikes)
        loss_vectors = tree.tree_map(_loss, flat_spikes)
        return jnp.sum(jnp.concatenate(tree.tree_flatten(loss_vectors)[0]))
        
    return jax.jit(_call)
        


def integral_accuracy(time_axis=1):
    """Calculate the accuracy of a network's predictions based on the voltage traces. Used in combination with a Leaky-Integrate neuron model as the final layer.

    :param traces: the output of the final layer of the SNN
    :param targets: the integer labels for each class
    :return: function which computes Accuracy score and predictions that takes SNN output traces and integer index labels.
    """
    def _integral_accuracy(traces, targets):
        preds = jnp.argmax(jnp.sum(traces, axis=time_axis), axis=-1)
        return jnp.mean(preds == targets), preds
        
    return jax.jit(_integral_accuracy)

# smoothing can be critical to the performance of your model...
# change this to be a higher-order function yielding a func with a set smoothing rate.

def integral_crossentropy(smoothing=0.3, time_axis=1):
    """Calculate the crossentropy between the integral of membrane potentials. Allows for label smoothing to discourage silencing the other neurons in the readout layer.

    :param smoothing: rate at which to smooth labels.
    :param time_axis: temporal axis of data
    :return: crossentropy loss function that takes SNN output traces and integer index labels.
    """

    def _integral_crossentropy(traces, targets):
        logits = jnp.sum(traces, axis=time_axis) # time axis.
        one_hot = jax.nn.one_hot(targets, logits.shape[-1])
        labels = optax.smooth_labels(one_hot, smoothing)
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))

    return _integral_crossentropy

# convert to function that returns compiled function
def mse_spikerate(sparsity=0.25, smoothing=0.0, time_axis=1):
    """Calculate the mean squared error of the mean spike rate. Allows for label smoothing to discourage silencing the other neurons in the readout layer.

    :param sparsity: the percentage of the time you want the neurons to spike
    :param smoothing: [optional] rate at which to smooth labels.
    :return: Mean-Squared-Error loss function on the spike rate that takes SNN output traces and integer index labels.
    """
    def _mse_spikerate(traces, targets):

        t = traces.shape[time_axis]
        logits = jnp.mean(traces, axis=time_axis) # time axis.
        labels = optax.smooth_labels(jax.nn.one_hot(targets, logits.shape[-1]), smoothing)
        return jnp.mean(optax.squared_error(logits, labels * sparsity * t))

    return jax.jit(_mse_spikerate)
