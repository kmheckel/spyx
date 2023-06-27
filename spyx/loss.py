import jax
import jax.numpy as jnp
import optax

# need to make these consistent...

@jax.jit
def l2_spike_reg(avg_spike_counts, target_count):
    """Spike rate regularization based on mean squared error from target rate."""

    flat = jnp.concatenate(jax.tree_util.tree_flatten(avg_spike_counts)[0])
    return jnp.sum(optax.l2_loss(flat, jnp.array([target_count]*flat.shape[0])))

@jax.jit
def clipped_sq_err_spike_reg(avg_spike_counts, target_count, radius):
    """
    Spike rate regularization based on clipped squared error from target spike count.
    
    Attributes:
        avg_spike_counts: array of average spikes per neuron over a batch.
        target_count: the target number of spikes to be emitted by a neuron
        radius: the tolerance +/- r from the target count where loss is clipped to zero
    
    """

    flat = jnp.concatenate(jax.tree_util.tree_flatten(avg_spike_counts)[0])
    return jnp.sum(jnp.maximum(0,optax.squared_error(flat, jnp.array([target_count]*flat.shape[0]))/radius - radius))

@jax.jit
def inverse_spike_count_reg(avg_spike_counts, time_len):
    """
    Spike rate regularization based on an inverse function that strongly
    discourages neurons from silencing and gradually penalizes higher spike rates.
    
    """

    flat = jnp.concatenate(jax.tree_util.tree_flatten(avg_spike_counts)[0])
    k = jnp.sqrt(time_len)
    loss = jnp.mean((flat/k) + (k/(flat+1)) )
    return loss

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
def integral_crossentropy(traces, targets, encode_labels=True, smoothing=0.3):
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
    if encode_labels:
        targets = jax.nn.one_hot(targets, logits.shape[-1])
    labels = optax.smooth_labels(targets, smoothing)
    return optax.softmax_cross_entropy(logits, labels).mean() #change to mean



# needs fixing.
@jax.jit
def exp_integral_crossentropy(spikes, targets):
    """
    Integral crossentropy with exponential weighting to promote pushing voltage
    deflections and therefore spikes to earlier in the network rollout.

    Still under construction.

    Adapted from:
    
    Nowotny, T., Turner, J. P., & Knight, J. C. (2022). Loss shaping enhances exact gradient learning with EventProp in Spiking Neural Networks. arXiv preprint arXiv:2212.01232.
    """

    exp = jnp.exp(-jnp.arange(0,spikes.shape[-2])/spikes.shape[-2])
    exp = jnp.repeat(jnp.expand_dims(exp, 0).T, repeats=spikes.shape[-1], axis=1)
    reweighted = exp * spikes
    preds = jnp.sum(spikes, axis=-2)
    return optax.softmax_cross_entropy(preds, jax.nn.one_hot(targets, preds.shape[-1])).mean()