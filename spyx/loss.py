import jax
import jax.numpy as jnp
import optax

from jax import tree_util as tree
# need to make these consistent...


class l1_reg:
    def __init__(self, target_rate, tolerance, time_steps, num_classes):
        self.l1_loss = lambda x: jnp.abs(jnp.sum(x,axis=1)/time_steps - (x.shape[1]/num_classes)*target_rate)
        self.clip = lambda x: jnp.maximum(0, x - tolerance)
        
    def __call__(self, spikes):
        loss_vectors = tree.tree_map(self.l1_loss, spikes)
        clipped_error = tree.tree_map(self.clip, loss_vectors)
        loss_vectors = tree.tree_map(jnp.ravel, clipped_error)
        return jnp.mean(jnp.concatenate(tree.tree_flatten(loss_vectors)[0]))

class l2_reg:

    def __init__(self, target_rate, tolerance, time_steps, num_classes):
        #                          spikes  per  expected number of samples
        self.rate_map = lambda x: (jnp.sum(x, axis=0) / num_classes) / time_steps
        self.sq_err_map = lambda x: optax.squared_error(x, jnp.array([target_rate]*x.size))
        self.clip = lambda x: jnp.maximum(0, (x/tolerance) - tolerance)
    
    def __call__(self, spikes):
        avg_neuron_activity = tree.tree_map(self.rate_map, spikes)
        activity_error = tree.tree_map(self.sq_err_map, avg_neuron_activity)
        clipped_error = tree.tree_map(self.clip, activity_error)
        loss_vectors = tree.tree_map(jnp.ravel, clipped_error)
        return jnp.mean(jnp.concatenate(tree.tree_flatten(loss_vectors)[0]))

        
class lasso_reg:
    def __init__(self, target_rate, tolerance, time_steps, num_classes):
        self.l1 = l1_reg(target_rate, tolerance, time_steps, num_classes)
        self.l2 = l2_reg(target_rate, tolerance, time_steps, num_classes)
        
    def __call__(self, spikes):
        return self.l1(spikes) + self.l2(spikes)


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
def integral_crossentropy(traces, targets, smoothing=0.3):
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
    return optax.softmax_cross_entropy(logits, labels).mean() #change to mean

